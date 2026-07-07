"""
debate_generator.py — Agent-Agent Debate email reply generator.

Architecture
------------
Concept: "agent-agent debate" from multi-agent systems research.
Two specialist LLM agents debate over the best reply across N rounds:

  Composer  — drafts and revises the reply
  Critic    — challenges the reply, identifies tone/content/coverage flaws
  Judge     — reads the full debate transcript and declares the final reply

This adversarial loop surfaces problems that single-shot generation misses:
  - Tone mismatches (Critic specialises in tone calibration)
  - Missing key facts (Critic checks coverage against the incoming email)
  - Vague commitments (Critic flags non-actionable language)

The final reply comes from the Judge, not from the last Composer draft,
which implements the "verifier/generator separation" pattern — the model
that generates is not the same as the one that selects.

Trade-offs vs. standard RAG generation
  + Higher quality replies (adversarial loop catches what a single pass misses)
  + Interpretable: the full debate transcript is returned for inspection
  - 2–4x more API calls per email (configurable via DEBATE_ROUNDS)
  - Higher latency (rounds are sequential, not parallel)

Usage
-----
  from src.debate_generator import debate_reply
  result = debate_reply(subject, body, index, client, rounds=2)
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
_DEBATE_ROUNDS = int(os.getenv("DEBATE_ROUNDS", "2"))
_RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# ---------------------------------------------------------------------------
# System prompts — each agent has a fixed persona
# ---------------------------------------------------------------------------

_COMPOSER_SYSTEM = """\
You are the COMPOSER agent for Hiver customer support.
Your job: write or revise the email reply to the customer.

Guidelines:
- Be empathetic, professional, and concise.
- Address every issue the customer raised.
- If you're revising after Critic feedback, directly fix the identified issues.
- Sign off as "The Hiver Support Team".
- Output ONLY the reply text — no meta-commentary, no "Here is my reply:".
"""

_CRITIC_SYSTEM = """\
You are the CRITIC agent for Hiver customer support.
Your job: review a drafted reply and identify specific, actionable weaknesses.

You must evaluate:
1. TONE — Is the tone appropriate for this customer's emotional state?
   (e.g., too cold for an upset customer; too casual for a formal enterprise inquiry)
2. COVERAGE — Does the reply address ALL the specific issues the customer raised?
   List any that are missed or only partially addressed.
3. CLARITY — Are there any vague commitments ("we'll look into it") that should
   be more specific?
4. PROFESSIONALISM — Any awkward phrasing, grammatical errors, or inappropriate
   sign-off?

If the reply is genuinely good, say "ACCEPT" and briefly state why.
If it needs work, say "REVISE" and give 1–3 specific, numbered improvement instructions.
Do NOT rewrite the reply yourself — only diagnose.
"""

_JUDGE_SYSTEM = """\
You are the JUDGE agent for Hiver customer support.
You have observed a debate between a Composer and a Critic over the best reply
to a customer email.

Your job: select or write the best possible final reply based on the full debate.

Rules:
- If the Composer's final draft addressed all of the Critic's concerns, use it directly.
- If issues remain, write a corrected final reply yourself.
- Output ONLY the final reply text — no commentary, no "Final reply:".
- Sign off as "The Hiver Support Team".
"""


# ---------------------------------------------------------------------------
# Agent call helpers
# ---------------------------------------------------------------------------

def _composer_draft(
    incoming_email: str,
    few_shot_block: str,
    prior_critique: str | None,
    prior_draft: str | None,
    client: OpenAI,
    model: str,
) -> str:
    """Composer writes or revises a reply."""
    if prior_critique is None:
        # First draft
        user_msg = (
            f"{few_shot_block}"
            f"Write a reply for the following customer email:\n\n"
            f"{incoming_email}"
        )
    else:
        # Revision
        user_msg = (
            f"Customer email:\n{incoming_email}\n\n"
            f"Your previous draft:\n{prior_draft}\n\n"
            f"Critic's feedback:\n{prior_critique}\n\n"
            f"Please revise the reply to address all the Critic's concerns."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _COMPOSER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=500,
    )
    return (response.choices[0].message.content or "").strip()


def _critic_review(
    incoming_email: str,
    draft: str,
    round_num: int,
    client: OpenAI,
    model: str,
) -> tuple[str, bool]:
    """
    Critic reviews a draft.
    Returns: (critique_text, accepted: bool)
    """
    user_msg = (
        f"Customer email:\n{incoming_email}\n\n"
        f"Composer's draft (round {round_num}):\n{draft}\n\n"
        f"Review this draft. Start with ACCEPT or REVISE."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CRITIC_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    critique = (response.choices[0].message.content or "").strip()
    accepted = critique.upper().startswith("ACCEPT")
    return critique, accepted


def _judge_final(
    incoming_email: str,
    debate_transcript: list[dict[str, str]],
    final_draft: str,
    client: OpenAI,
    model: str,
) -> str:
    """Judge reads the full debate and delivers the final reply."""
    transcript_str = "\n\n".join(
        f"[{turn['role'].upper()} — Round {turn['round']}]\n{turn['content']}"
        for turn in debate_transcript
    )

    user_msg = (
        f"Customer email:\n{incoming_email}\n\n"
        f"=== Debate Transcript ===\n{transcript_str}\n\n"
        f"=== Composer's Final Draft ===\n{final_draft}\n\n"
        f"Based on this debate, deliver the best possible final reply."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Main debate function
# ---------------------------------------------------------------------------

def debate_reply(
    subject: str,
    body: str,
    index: EmailIndex,
    client: OpenAI,
    rounds: int = _DEBATE_ROUNDS,
    top_k: int = _RAG_TOP_K,
    model: str = _MODEL,
    role_stakes_context: str = "",
) -> dict[str, Any]:
    """
    Generate a reply using agent-agent debate.

    Parameters
    ----------
    subject, body : str
        The incoming email.
    index : EmailIndex
        FAISS index of past emails for RAG retrieval.
    client : OpenAI
        Authenticated client.
    rounds : int
        Number of Composer→Critic debate rounds (default 2).
        More rounds = higher quality but more API calls.
    top_k : int
        Number of RAG examples to retrieve.
    model : str
        The model used for all agents (Composer, Critic, Judge).
    role_stakes_context : str
        Optional Role+Stakes context (who reads this, what's riding on it).
        Injected into the Composer's initial prompt.

    Returns
    -------
    dict with keys:
      - generated_reply      : final reply selected by the Judge
      - debate_transcript    : list of all turns (for interpretability)
      - rounds_completed     : how many rounds ran before acceptance
      - accepted_early       : True if Critic accepted before max rounds
      - retrieved_examples   : RAG examples used
      - mode                 : "debate"
    """
    # 1. RAG retrieval
    incoming_email = f"Subject: {subject}\n\n{body}"
    if role_stakes_context:
        incoming_email += f"\n\n[Context for this reply: {role_stakes_context}]"
    query_embedding = embed_texts([incoming_email], client)[0]
    retrieved = index.search(query_embedding, k=top_k)
    few_shot_block = _build_few_shot_block(retrieved)

    debate_transcript: list[dict[str, str]] = []
    current_draft: str = ""
    last_critique: str | None = None
    accepted_early = False

    for round_num in range(1, rounds + 1):
        # --- Composer turn ---
        current_draft = _composer_draft(
            incoming_email=incoming_email,
            few_shot_block=few_shot_block if round_num == 1 else "",
            prior_critique=last_critique,
            prior_draft=current_draft if round_num > 1 else None,
            client=client,
            model=model,
        )
        debate_transcript.append({
            "role": "composer",
            "round": round_num,
            "content": current_draft,
        })

        # --- Critic turn ---
        critique, accepted = _critic_review(
            incoming_email=incoming_email,
            draft=current_draft,
            round_num=round_num,
            client=client,
            model=model,
        )
        debate_transcript.append({
            "role": "critic",
            "round": round_num,
            "content": critique,
        })
        last_critique = critique

        if accepted:
            accepted_early = True
            break

    # 2. Judge selects final reply
    final_reply = _judge_final(
        incoming_email=incoming_email,
        debate_transcript=debate_transcript,
        final_draft=current_draft,
        client=client,
        model=model,
    )

    return {
        "generated_reply": final_reply,
        "debate_transcript": debate_transcript,
        "rounds_completed": len([t for t in debate_transcript if t["role"] == "composer"]),
        "accepted_early": accepted_early,
        "rag_context_text": few_shot_block,
        "retrieved_examples": [
            {
                "id": ex.get("id"),
                "subject": ex.get("subject"),
                "category": ex.get("category"),
                "similarity": ex.get("_similarity"),
            }
            for ex in retrieved
        ],
        "mode": "debate",
        "model": model,
    }
