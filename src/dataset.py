"""
dataset.py — Dataset loader, FAISS index, BM25 index, hybrid retrieval,
             HyDE retrieval, and optional cross-encoder reranking.

Retrieval pipeline (new architecture)
--------------------------------------
Standard RAG uses a single dense embedding search. This module now supports
three progressively better retrieval strategies:

1. dense_search()   — original FAISS cosine-similarity search
2. hybrid_search()  — BM25 keyword + FAISS dense, fused via RRF
3. hyde_retrieve()  — HyDE: generate a hypothetical reply, embed that,
                       then hybrid_search in reply-space

Why hybrid + HyDE?
  Dense search misses exact keyword matches (product names, error codes like
  "error 403", plan names like "Enterprise Plus"). BM25 catches these but
  misses semantic meaning. RRF fusion gets both.

  HyDE (Hypothetical Document Embeddings, Gao et al. 2022) addresses the
  semantic gap between a short incoming email and the past *replies* in the
  index. Instead of embedding the question, we generate a hypothetical ideal
  answer, embed that, and search — aligning the query representation with
  the reply corpus it's searching over.

Cross-encoder reranking (optional)
  After any retrieval, a cross-encoder model jointly encodes the query + each
  candidate and re-scores them. More accurate than bi-encoder similarity but
  slower — applied only to the top-N candidates, not the full index.
  Requires sentence-transformers (already installed).

Environment variables
---------------------
  EMBEDDING_MODEL           text-embedding-3-small (default)
  DATA_PATH                 data/emails.json
  RAG_TOP_K                 3
  RETRIEVAL_MODE            dense | hybrid | hyde  (default: hybrid)
  RERANK                    true | false (default: false, requires sentence-transformers)
  RERANK_MODEL              cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
_DATA_PATH = os.getenv("DATA_PATH", "data/emails.json")
_EMBED_DIM = 1536  # text-embedding-3-small output dimension
_RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
_RERANK = os.getenv("RERANK", "false").lower() in ("1", "true", "yes")
_RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")

# HyDE prompt — generate a hypothetical ideal reply for the incoming email
_HYDE_SYSTEM = """\
You are a customer-support agent for a B2B SaaS company.
Given a customer email, write a brief, realistic example of what a good reply
might look like. This is for retrieval purposes only — keep it to 2-3 sentences.
Do not sign off. Output ONLY the hypothetical reply text.
"""


def load_emails(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the email dataset from JSON."""
    p = Path(path or _DATA_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at {p}. "
            "Run `uv run python scripts/generate_dataset.py` first."
        )
    with p.open() as f:
        data = json.load(f)
    return data


def embed_texts(texts: list[str], client: OpenAI) -> np.ndarray:
    """
    Embed a list of texts using the configured OpenAI embedding model.
    Returns a float32 numpy array of shape (len(texts), dim).
    """
    batch_size = 512
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=_EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return np.array(all_embeddings, dtype=np.float32)


def _reciprocal_rank_fusion(
    lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF) — merges multiple ranked lists into one.

    Each list is [(doc_id, score), ...] sorted best→worst.
    RRF score = sum(1 / (k + rank)) across all lists.
    k=60 is the standard value from Cormack et al. 2009.
    """
    scores: dict[int, float] = {}
    for ranked_list in lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # Sort by descending RRF score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class EmailIndex:
    """
    Unified retrieval index combining FAISS (dense) and BM25 (sparse).

    Supports three retrieval modes:
      dense    — FAISS cosine similarity only (original behaviour)
      hybrid   — BM25 + FAISS fused with RRF (recommended)
      hyde     — HyDE hypothetical reply → hybrid search

    Optionally applies cross-encoder reranking after any retrieval.
    """

    def __init__(self, emails: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        self.emails = emails

        # --- FAISS dense index ---
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalised = embeddings / norms
        self.faiss_index = faiss.IndexFlatIP(normalised.shape[1])
        self.faiss_index.add(normalised)

        # --- BM25 sparse index ---
        self._bm25 = None
        self._build_bm25()

        # --- Cross-encoder (lazy-loaded on first use) ---
        self._cross_encoder = None

    def _build_bm25(self) -> None:
        """Build BM25 index from email subjects + bodies."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
            corpus = []
            for e in self.emails:
                text = f"{e.get('subject', '')} {e.get('body', '')}".lower()
                tokens = text.split()
                corpus.append(tokens)
            self._bm25 = BM25Okapi(corpus)
        except ImportError:
            # rank_bm25 not installed — hybrid falls back to dense only
            self._bm25 = None

    def _get_cross_encoder(self):
        """Lazy-load the cross-encoder model (avoids slow import at startup)."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
                self._cross_encoder = CrossEncoder(_RERANK_MODEL)
            except (ImportError, Exception):
                self._cross_encoder = False  # Mark as unavailable
        return self._cross_encoder if self._cross_encoder is not False else None

    def _dense_top_n(
        self, query_embedding: np.ndarray, n: int
    ) -> list[tuple[int, float]]:
        """Return (idx, similarity) pairs from FAISS."""
        vec = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        distances, indices = self.faiss_index.search(vec, n)
        return [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx >= 0
        ]

    def _bm25_top_n(self, query_text: str, n: int) -> list[tuple[int, float]]:
        """Return (idx, bm25_score) pairs."""
        if self._bm25 is None:
            return []
        tokens = query_text.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _rerank(
        self, query_text: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Apply cross-encoder reranking to a candidate list.
        Cross-encoder processes query + document jointly for more accurate scoring.
        Falls back silently if sentence-transformers is unavailable.
        """
        ce = self._get_cross_encoder()
        if ce is None or not candidates:
            return candidates
        pairs = [
            (query_text, f"{c.get('subject', '')} {c.get('body', '')}")
            for c in candidates
        ]
        try:
            ce_scores = ce.predict(pairs)  # type: ignore
            scored = sorted(
                zip(ce_scores, candidates), key=lambda x: x[0], reverse=True
            )
            reranked = []
            for score, cand in scored:
                c = dict(cand)
                c["_rerank_score"] = float(score)
                reranked.append(c)
            return reranked
        except Exception:
            return candidates

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        query_text: str = "",
        mode: Literal["dense", "hybrid"] = "hybrid",
        rerank: bool = _RERANK,
    ) -> list[dict[str, Any]]:
        """
        Return top-k most relevant emails.

        Parameters
        ----------
        query_embedding : np.ndarray
            Embedding of the query (shape: (dim,)).
        k : int
            Number of results to return.
        query_text : str
            Raw query text (required for BM25 and reranking).
        mode : "dense" | "hybrid"
            Retrieval strategy. hybrid = BM25 + FAISS + RRF fusion.
        rerank : bool
            If True, apply cross-encoder reranking to results.
        """
        fetch_n = max(k * 4, 20)  # over-fetch for fusion/reranking

        if mode == "dense" or self._bm25 is None:
            # Pure dense retrieval
            top = self._dense_top_n(query_embedding, fetch_n)
            indices = [i for i, _ in top[:k]]
            sims = {i: s for i, s in top}
            results = []
            for idx in indices:
                r = dict(self.emails[idx])
                r["_similarity"] = sims.get(idx, 0.0)
                results.append(r)
        else:
            # Hybrid: BM25 + dense → RRF
            dense_list = self._dense_top_n(query_embedding, fetch_n)
            bm25_list = self._bm25_top_n(query_text, fetch_n)
            fused = _reciprocal_rank_fusion([dense_list, bm25_list])
            dense_sim = {i: s for i, s in dense_list}
            results = []
            for idx, rrf_score in fused[:k if not rerank else fetch_n]:
                if 0 <= idx < len(self.emails):
                    r = dict(self.emails[idx])
                    r["_similarity"] = dense_sim.get(idx, rrf_score)
                    r["_rrf_score"] = rrf_score
                    results.append(r)

        if rerank and query_text:
            results = self._rerank(query_text, results)
            results = results[:k]
        elif not rerank:
            results = results[:k]

        return results


def hyde_retrieve(
    subject: str,
    body: str,
    index: "EmailIndex",
    client: OpenAI,
    k: int = 3,
    rerank: bool = _RERANK,
    generation_model: str = _GENERATION_MODEL,
) -> tuple[list[dict[str, Any]], str]:
    """
    HyDE (Hypothetical Document Embeddings) retrieval.

    Instead of embedding the incoming email to find similar past emails,
    we:
      1. Ask the LLM to generate a *hypothetical ideal reply* (2-3 sentences)
      2. Embed that hypothetical reply
      3. Search the index — which contains past emails indexed by their body

    This aligns the query representation with the reply corpus, since we're
    searching for past emails whose *context* is similar to our hypothetical
    reply rather than whose *question* is similar to the incoming email.

    Particularly effective for:
      - Vague or unusual incoming emails
      - Cases where customer language differs from agent language
      - Short, ambiguous subject lines

    Returns: (retrieved_examples, hypothetical_reply_used)
    """
    user_msg = f"Subject: {subject}\n\n{body}"
    try:
        resp = client.chat.completions.create(
            model=generation_model,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        hypothetical = (resp.choices[0].message.content or "").strip()
    except Exception:
        # Fall back to original query if HyDE generation fails
        hypothetical = f"Subject: {subject}\n\n{body}"

    # Embed the hypothetical reply and search with it
    hyp_embedding = embed_texts([hypothetical], client)[0]
    results = index.search(
        query_embedding=hyp_embedding,
        k=k,
        query_text=hypothetical,
        mode="hybrid",
        rerank=rerank,
    )
    return results, hypothetical


def build_index(
    emails: list[dict[str, Any]] | None = None,
    client: OpenAI | None = None,
    data_path: str | Path | None = None,
) -> tuple["EmailIndex", OpenAI]:
    """
    Convenience function: load dataset, embed, build and return the index.
    Now builds both FAISS (dense) and BM25 (sparse) indices.

    Returns (index, openai_client) so callers can reuse the client.
    """
    if client is None:
        client = OpenAI()
    if emails is None:
        emails = load_emails(data_path)

    texts = [f"Subject: {e['subject']}\n\n{e['body']}" for e in emails]
    print(f"Embedding {len(texts)} emails with {_EMBEDDING_MODEL}…")
    embeddings = embed_texts(texts, client)
    index = EmailIndex(emails, embeddings)
    has_bm25 = index._bm25 is not None
    print(
        f"Index built: {index.faiss_index.ntotal} vectors (dense)"
        + (", BM25 ready (hybrid)" if has_bm25 else ", BM25 unavailable (dense only)")
    )
    return index, client
