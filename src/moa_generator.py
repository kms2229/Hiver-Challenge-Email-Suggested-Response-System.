"""
moa_generator.py — Mixture-of-Agents (MoA) email reply generator.

Architecture
------------
Concept: "Mixture-of-Agents" from Wang et al. (2024) — multiple LLM instances
each independently generate a candidate reply, then a synthesizer model reads
all candidates and merges the best elements into a final reply.

Why it works:
  - High-temperature candidates explore different phrasings, orderings, and
    emphasis — one may nail the tone, another the factual coverage.
  - The synthesizer operates at low temperature and is explicitly instructed to
    cherry-pick the strongest elements of each.
  - MoA consistently outperforms single-shot generation on tasks where quality
    has multiple orthogonal dimensions (tone + coverage + professionalism here).

Trade-offs vs. standard generation
  + Higher quality: synthesizer can combine the best of N independent attempts
  + Robust to any single bad draft (worst candidate is ignored by synthesizer)
  - N+1 API calls per email (N candidate calls + 1 synthesizer call)
  - Higher latency (candidates can be generated in parallel in principle,
    but we use the synchronous OpenAI client here for simplicity)

Usage
-----
  from src.moa_generator import moa_reply
  result = moa_reply(subject, body, index, client, n_candidates=3)
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from src.dataset import EmailIndex, embed_texts
from src.generator import _build_few_shot_block

load_dotenv()

_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
_MOA_CANDIDATES = int(os.getenv("MOA_CANDIDATES", "3"))
_RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

_CANDIDATE_SYSTEM = """\
You are a professional customer-support agent for a B2B SaaS company called Hiver.
Write a helpful, empathetic, and concise reply to the customer email below.
Sign off as "The Hiver Support Team".
Output ONLY the reply text — no preamble.
"""

_SYNTHESIZER_SYSTEM = """\
You are an expert editor for Hiver customer support.
You will receive multiple candidate replies to the same customer email.
Each candidate was written independently and may have different strengths.

Your task:
1. Identify the strongest elements across all candidates:
   - Best opening / empathy statement
   - Most complete coverage of the customer's issues
   - Clearest action items or next steps
   - Most appropriate tone and sign-off
2. Synthesize a single FINAL reply that combines the best of all candidates.
   Do not just pick one — genuinely merge their strengths.
3. The result should be tighter and better than any individual candidate.

Output ONLY the final reply text. Sign off as "The Hiver Support Team".
"""


def _generate_candidate(
    incoming_email: str,
    few_shot_block: str,
    candidate_num: int,
    client: OpenAI,
    model: str,
) -> str:
    """Generate a single candidate reply with high temperature for diversity."""
    user_msg = (
        f"{few_shot_block}"
        f"Write a reply for the following customer email:\n\n"
        f"{incoming_email}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CANDIDATE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.85,   # High temp → diverse candidates
        max_tokens=500,
    )
    return (response.choices[0].message.content or "").strip()


def _synthesize(
    incoming_email: str,
    candidates: list[str],
    client: OpenAI,
    model: str,
) -> str:
    """Synthesizer reads all candidates and produces the final merged reply."""
    candidates_block = "\n\n".join(
        f"--- Candidate {i+1} ---\n{c}"
        for i, c in enumerate(candidates)
    )
    user_msg = (
        f"Customer email:\n{incoming_email}\n\n"
        f"=== Candidate Replies ===\n\n"
        f"{candidates_block}\n\n"
        f"=== Your task ===\n"
        f"Synthesize the best final reply by combining the strongest elements of all candidates."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYNTHESIZER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,   # Low temp → focused synthesis
        max_tokens=500,
    )
    return (response.choices[0].message.content or "").strip()


def moa_reply(
    subject: str,
    body: str,
    index: EmailIndex,
    client: OpenAI,
    n_candidates: int = _MOA_CANDIDATES,
    top_k: int = _RAG_TOP_K,
    model: str = _MODEL,
) -> dict[str, Any]:
    """
    Generate a reply using Mixture-of-Agents.

    Parameters
    ----------
    subject, body : str
        The incoming email.
    index : EmailIndex
        FAISS index for RAG retrieval.
    client : OpenAI
        Authenticated client.
    n_candidates : int
        Number of independent candidate replies to generate (default 3).
    top_k : int
        Number of RAG examples to retrieve.
    model : str
        Model for both candidates and synthesizer.

    Returns
    -------
    dict with keys:
      - generated_reply   : final synthesized reply
      - candidates        : list of all N independent drafts (for inspection)
      - retrieved_examples: RAG examples used
      - n_candidates      : number of candidates generated
      - mode              : "moa"
    """
    # 1. RAG retrieval (shared across all candidates for fair comparison)
    incoming_email = f"Subject: {subject}\n\n{body}"
    query_embedding = embed_texts([incoming_email], client)[0]
    retrieved = index.search(query_embedding, k=top_k)
    few_shot_block = _build_few_shot_block(retrieved)

    # 2. Generate N independent candidates
    candidates: list[str] = []
    for i in range(n_candidates):
        draft = _generate_candidate(
            incoming_email=incoming_email,
            few_shot_block=few_shot_block,
            candidate_num=i + 1,
            client=client,
            model=model,
        )
        candidates.append(draft)

    # 3. Synthesize final reply
    final_reply = _synthesize(
        incoming_email=incoming_email,
        candidates=candidates,
        client=client,
        model=model,
    )

    return {
        "generated_reply": final_reply,
        "candidates": candidates,
        "retrieved_examples": [
            {
                "id": ex.get("id"),
                "subject": ex.get("subject"),
                "category": ex.get("category"),
                "similarity": ex.get("_similarity"),
            }
            for ex in retrieved
        ],
        "n_candidates": n_candidates,
        "mode": "moa",
        "model": model,
    }
