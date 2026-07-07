"""
generator.py — RAG-based email reply generator with optional advanced modes.

Modes
-----
standard   Original single-pass RAG → LLM generation.
refine     Self-Refine: generate → self-critique → revise.

New in this version
-------------------
1. Structured output (Pydantic-style JSON schema)
   The LLM is asked to return a JSON object with fields:
     greeting, body, action_items, sign_off
   This prevents meta-commentary, forgotten sign-offs, or multiple drafts.
   The final reply is assembled from these fields.

2. Sentiment/urgency context injection
   If a `classifier.EmailClassification` is passed, the system prompt is
   augmented with tone guidance derived from the emotional register of the
   incoming email. A frustrated + high-urgency email gets a very different
   prompt nudge than a neutral + low-urgency one.

3. Agentic RAG (carried forward from previous version)
   Re-query if initial retrieval similarity is below AGENTIC_RAG_THRESHOLD.

4. Retrieval mode selection
   Callers can specify "dense", "hybrid", or "hyde" retrieval.
   hyde uses HyDE (Hypothetical Document Embeddings) for better retrieval
   on unusual or vague emails.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from openai import OpenAI
from dotenv import load_dotenv

from src.dataset import EmailIndex, embed_texts, hyde_retrieve

load_dotenv()

_GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
_RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
_AGENTIC_RAG_THRESHOLD = float(os.getenv("AGENTIC_RAG_THRESHOLD", "0.60"))
_RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

_STRUCTURED_SYSTEM = """\
You are a professional customer-support agent for a B2B SaaS company called Hiver.

{tone_context}

BEFORE YOU DRAFT THE REPLY — work through this in your JSON response:

1. Consider 2-3 different ways to interpret what the customer is asking.
   What is the most likely real issue behind their email?
2. Flag any hidden assumptions in the email that might be wrong or incomplete.
3. Pick the interpretation that best fits the customer's situation.

Then write your reply based on that best interpretation.

You MUST respond with a JSON object with exactly these fields:
{{
  "interpretation": "<one sentence: the real issue you're addressing and why>",
  "hidden_assumptions": ["<assumption 1>", "<assumption 2>"],
  "greeting": "<opening line, e.g. 'Dear Sarah,' or 'Hi there,'>",
  "body": "<the main response — address the real issue, not just the surface question>",
  "action_items": ["<specific next step — at least one concrete commitment>"],
  "sign_off": "The Hiver Support Team"
}}

Rules:
- interpretation: one sentence explaining the real issue you identified
- hidden_assumptions: list of 0-2 assumptions the customer might be making wrongly
- greeting: address the customer by name if available, else use "Hi there,"
- body: empathetic, professional, concise — address the REAL issue, not just the words
- action_items: ALWAYS include at least one concrete, time-bound commitment or next step
- sign_off: always "The Hiver Support Team"
- Output ONLY the JSON object. No other text.
"""

_CRITIQUE_SYSTEM = """\
You are a quality reviewer for Hiver customer support emails.
Read the customer email and the drafted reply, then identify up to 3 specific weaknesses.
Focus on: missing information, tone mismatch, vague commitments, or unprofessional phrasing.
Be concise — 1 sentence per issue. If the reply is already good, say "No major issues."
"""

_REFINE_SYSTEM = """\
You are a professional customer-support agent for a B2B SaaS company called Hiver.
You have received feedback on a draft reply. Revise it to fix the specific issues noted.
Output ONLY the revised reply text. Sign off as "The Hiver Support Team".
"""

_QUERY_REFORMULATE_SYSTEM = """\
You are a search query specialist. Given a customer support email that returned
weak search results, reformulate the search query to find more relevant past examples.
Focus on the core problem type, not the customer's specific details.
Output ONLY the reformulated query text (1–2 sentences).
"""


def _build_few_shot_block(examples: list[dict[str, Any]]) -> str:
    """Format retrieved examples as few-shot demonstrations."""
    if not examples:
        return ""
    lines = ["Here are some examples of past emails and the replies we sent:\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"--- Example {i} ---")
        lines.append(f"Incoming email (Subject: {ex['subject']}):")
        lines.append(ex["body"].strip())
        lines.append(f"\nOur reply:")
        lines.append(ex["reply"].strip())
        lines.append("")
    lines.append("--- End of examples ---\n")
    return "\n".join(lines)


def _assemble_reply(structured: dict[str, Any]) -> str:
    """Assemble the final reply string from the structured JSON output."""
    parts = []
    greeting = structured.get("greeting", "").strip()
    if greeting:
        parts.append(greeting)
    body = structured.get("body", "").strip()
    if body:
        parts.append(body)
    action_items = structured.get("action_items", [])
    if action_items and isinstance(action_items, list):
        for item in action_items:
            item_str = str(item).strip()
            if item_str:
                parts.append(f"• {item_str}")
    sign_off = structured.get("sign_off", "The Hiver Support Team").strip()
    parts.append(f"\n{sign_off}")
    return "\n\n".join(parts)


def _extract_interpretation_note(structured: dict[str, Any]) -> str:
    """
    Extract the 'Before You Answer' reasoning as a human-readable note.
    Shown in CLI output and stored in the result for transparency.
    """
    parts = []
    interpretation = structured.get("interpretation", "").strip()
    if interpretation:
        parts.append(f"Interpretation: {interpretation}")
    assumptions = structured.get("hidden_assumptions", [])
    if assumptions and isinstance(assumptions, list):
        for a in assumptions:
            a_str = str(a).strip()
            if a_str:
                parts.append(f"⚠ Assumption flagged: {a_str}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agentic RAG — re-query if retrieval quality is low
# ---------------------------------------------------------------------------

def _agentic_retrieve(
    subject: str,
    body: str,
    index: EmailIndex,
    client: OpenAI,
    top_k: int,
    threshold: float,
    retrieval_mode: str = "hybrid",
) -> tuple[list[dict[str, Any]], bool]:
    """
    Attempt retrieval; if max similarity < threshold, reformulate the query
    and retrieve again (Agentic RAG pattern).

    Returns: (retrieved_examples, was_re_queried: bool)
    """
    query_text = f"Subject: {subject}\n\n{body}"
    query_embedding = embed_texts([query_text], client)[0]
    retrieved = index.search(
        query_embedding=query_embedding,
        k=top_k,
        query_text=query_text,
        mode=retrieval_mode,
    )

    # Check retrieval quality
    max_sim = max((ex.get("_similarity", 0) for ex in retrieved), default=0)
    if max_sim >= threshold or not retrieved:
        return retrieved, False

    # Low similarity → reformulate query
    reformulate_prompt = (
        f"Customer email:\nSubject: {subject}\n\n{body}\n\n"
        f"The search returned weak results (best similarity: {max_sim:.2f}). "
        f"Reformulate the search query to find more relevant past support examples."
    )
    response = client.chat.completions.create(
        model=_GENERATION_MODEL,
        messages=[
            {"role": "system", "content": _QUERY_REFORMULATE_SYSTEM},
            {"role": "user", "content": reformulate_prompt},
        ],
        temperature=0.3,
        max_tokens=100,
    )
    reformulated = (response.choices[0].message.content or query_text).strip()

    # Re-query with reformulated text
    new_embedding = embed_texts([reformulated], client)[0]
    new_retrieved = index.search(
        query_embedding=new_embedding,
        k=top_k,
        query_text=reformulated,
        mode=retrieval_mode,
    )

    # Use whichever retrieval was better
    new_max_sim = max((ex.get("_similarity", 0) for ex in new_retrieved), default=0)
    best = new_retrieved if new_max_sim > max_sim else retrieved
    return best, True


# ---------------------------------------------------------------------------
# Structured generation (single pass)
# ---------------------------------------------------------------------------

def _single_pass_structured(
    subject: str,
    body: str,
    retrieved: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    tone_context: str = "",
) -> tuple[str, dict[str, Any]]:
    """
    Core RAG + LLM generation with structured JSON output.

    Returns: (assembled_reply_text, raw_structured_dict)
    """
    few_shot_block = _build_few_shot_block(retrieved)
    system_prompt = _STRUCTURED_SYSTEM.format(
        tone_context=tone_context or "Write a professional, empathetic reply."
    )
    user_message = (
        f"{few_shot_block}"
        f"Now write a structured JSON reply for:\n\n"
        f"Subject: {subject}\n\n"
        f"{body.strip()}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.4,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    raw_content = response.choices[0].message.content or "{}"

    try:
        structured = json.loads(raw_content)
    except json.JSONDecodeError:
        # Fallback: treat entire content as body
        structured = {
            "greeting": "Hi there,",
            "body": raw_content,
            "action_items": [],
            "sign_off": "The Hiver Support Team",
        }

    assembled = _assemble_reply(structured)
    return assembled, structured


# ---------------------------------------------------------------------------
# Self-Refine loop
# ---------------------------------------------------------------------------

def _self_refine(
    subject: str,
    body: str,
    draft: str,
    client: OpenAI,
    model: str,
) -> tuple[str, str]:
    """
    Self-Refine: critique the draft, then revise it.
    Returns: (revised_reply, critique_text)
    """
    incoming_email = f"Subject: {subject}\n\n{body}"

    # Step 1 — Self-critique
    critique_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CRITIQUE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Customer email:\n{incoming_email}\n\n"
                    f"Draft reply:\n{draft}\n\n"
                    f"Identify up to 3 specific weaknesses."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=200,
    )
    critique = (critique_response.choices[0].message.content or "No major issues.").strip()

    # If no issues found, return original draft
    if "no major" in critique.lower():
        return draft, critique

    # Step 2 — Revise based on critique
    revise_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _REFINE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Customer email:\n{incoming_email}\n\n"
                    f"Original draft:\n{draft}\n\n"
                    f"Issues to fix:\n{critique}\n\n"
                    f"Write the improved reply."
                ),
            },
        ],
        temperature=0.3,
        max_tokens=600,
    )
    revised = (revise_response.choices[0].message.content or draft).strip()
    return revised, critique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reply(
    subject: str,
    body: str,
    index: EmailIndex,
    client: OpenAI,
    top_k: int = _RAG_TOP_K,
    model: str = _GENERATION_MODEL,
    mode: Literal["standard", "refine"] = "standard",
    agentic_rag: bool = True,
    retrieval_mode: Literal["dense", "hybrid", "hyde"] = _RETRIEVAL_MODE,
    classification: Any | None = None,
    role_stakes_context: str = "",
) -> dict[str, Any]:
    """
    Generate a suggested reply for an incoming email.

    Parameters
    ----------
    subject, body : str
        The incoming email.
    index : EmailIndex
        FAISS + BM25 index of past emails.
    client : OpenAI
        Authenticated client.
    top_k : int
        Number of RAG examples to retrieve.
    model : str
        LLM to use.
    mode : "standard" | "refine"
        - standard: single-pass structured RAG generation
        - refine: generate → self-critique → revise
    agentic_rag : bool
        Re-query if initial retrieval quality is below threshold.
    retrieval_mode : "dense" | "hybrid" | "hyde"
        - dense: FAISS cosine similarity only
        - hybrid: BM25 + FAISS fused with RRF (default)
        - hyde: HyDE hypothetical reply → hybrid search
    classification : EmailClassification | None
        If provided, injects tone guidance into the system prompt.

    Returns
    -------
    dict with keys:
      - generated_reply      : str (assembled from structured fields)
      - structured_output    : dict (raw JSON fields: greeting, body, action_items, sign_off)
      - retrieved_examples   : list
      - model, mode          : str
      - was_re_queried       : bool
      - retrieval_mode       : str
      - critique             : str (refine mode only)
      - classification       : dict (if classifier was run)
    """
    # 1. Build tone context from classification + role/stakes context
    tone_context = ""
    classification_dict: dict[str, Any] = {}
    if classification is not None:
        try:
            from src.classifier import build_tone_context
            tone_context = build_tone_context(classification)
            classification_dict = {
                "sentiment": classification.sentiment,
                "urgency": classification.urgency,
                "escalation_risk": classification.escalation_risk,
                "primary_issue": classification.primary_issue,
            }
        except Exception:
            pass
    # Append Role+Stakes context if provided (Technique 6)
    if role_stakes_context:
        tone_context = (tone_context + "\n\n" + role_stakes_context).strip()

    # 2. Retrieve (with HyDE, agentic re-query, or hybrid)
    hyde_hypothetical: str | None = None

    if retrieval_mode == "hyde":
        retrieved, hyde_hypothetical = hyde_retrieve(
            subject=subject,
            body=body,
            index=index,
            client=client,
            k=top_k,
        )
        was_re_queried = False
    elif agentic_rag:
        retrieved, was_re_queried = _agentic_retrieve(
            subject=subject,
            body=body,
            index=index,
            client=client,
            top_k=top_k,
            threshold=_AGENTIC_RAG_THRESHOLD,
            retrieval_mode=retrieval_mode,
        )
    else:
        query_text = f"Subject: {subject}\n\n{body}"
        query_embedding = embed_texts([query_text], client)[0]
        retrieved = index.search(
            query_embedding=query_embedding,
            k=top_k,
            query_text=query_text,
            mode=retrieval_mode,
        )
        was_re_queried = False

    # 3. Generate initial draft (structured output)
    draft, structured = _single_pass_structured(
        subject, body, retrieved, client, model, tone_context
    )

    # Extract Before-You-Answer interpretation note
    interpretation_note = _extract_interpretation_note(structured)

    # 4. Self-refine if requested
    critique = None
    if mode == "refine":
        draft, critique = _self_refine(subject, body, draft, client, model)

    result: dict[str, Any] = {
        "generated_reply": draft,
        "structured_output": structured,
        "interpretation_note": interpretation_note,
        "rag_context_text": _build_few_shot_block(retrieved),
        "retrieved_examples": [
            {
                "id": ex.get("id"),
                "subject": ex.get("subject"),
                "category": ex.get("category"),
                "similarity": ex.get("_similarity"),
                "rrf_score": ex.get("_rrf_score"),
            }
            for ex in retrieved
        ],
        "model": model,
        "mode": mode,
        "retrieval_mode": retrieval_mode,
        "was_re_queried": was_re_queried,
    }
    if critique is not None:
        result["critique"] = critique
    if classification_dict:
        result["classification"] = classification_dict
    if hyde_hypothetical is not None:
        result["hyde_hypothetical"] = hyde_hypothetical

    return result
