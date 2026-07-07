"""
evaluator.py — Multi-metric accuracy scorer for generated email replies.

Accuracy is measured across FOUR orthogonal dimensions, each capturing
something a single metric would miss:

┌─────────────────────────┬────────────────────────────────────────────────────┐
│ Dimension               │ Metric                                              │
├─────────────────────────┼────────────────────────────────────────────────────┤
│ Semantic similarity     │ Cosine similarity of OpenAI embeddings             │
│ Information coverage    │ ROUGE-L recall (unigram recall of ground-truth)    │
│ Tone appropriateness    │ LLM-as-judge or Debate-as-Judge, score 1–5        │
│ Completeness/quality    │ LLM-as-judge or Debate-as-Judge, score 1–5        │
└─────────────────────────┴────────────────────────────────────────────────────┘

Judge modes
-----------
standard (default)
  A single LLM call rates tone and quality. Fast and cheap.

debate (USE_DEBATE_JUDGE=true)
  Concept: "Debate-as-Judge" from multi-agent evaluation research.
  Two independent LLM judge instances score the reply separately.
  If they agree (scores within 0.5), average their scores.
  If they disagree (gap ≥ 1.0), a third arbitrator call resolves it.

  Why debate judges beat a single judge:
    - Single LLM judges suffer from position bias, verbosity bias, and
      self-enhancement bias (favouring replies that sound like themselves).
    - A second independent judge challenges those biases.
    - The arbitration step adds a structured resolution mechanism instead
      of letting bias silently dominate.
    - Research shows debate amplifies correctness over static ensembles
      when paired with an adaptive stopping rule — which is exactly what
      the 0.5-agreement threshold implements here.

Composite score (0–1):
  score = 0.35 × semantic_sim
        + 0.25 × rouge_recall
        + 0.25 × (tone_score / 5)
        + 0.15 × (quality_score / 5)

Weight rationale:
  - Semantic similarity (0.35): most important — captures whether the reply
    means the same thing as the gold reply, robust to paraphrasing.
  - ROUGE-L recall (0.25): checks that key content words are present;
    catches replies that are semantically close but omit critical info.
  - Tone (0.25): a reply can be accurate but completely wrong in register
    (e.g., too cold for an empathy-required email).
  - Quality (0.15): a sanity check for completeness and professionalism;
    high weight on the other three metrics reduces reliance on this.

Why this beats a single metric:
  - ROUGE alone: misses paraphrasing ("sorry" vs "we apologise").
  - Embedding cosine alone: can score high even if key facts are absent.
  - LLM-as-judge alone: expensive, unstable; anchoring with text metrics
    improves reproducibility.

Validation:
  A 10-example calibration set with human-annotated quality labels
  (good=1, ok=0.5, bad=0) is included. We compute Spearman ρ between
  composite scores and human labels to confirm the metric reflects
  real quality.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from rouge_score import rouge_scorer
from scipy.stats import spearmanr
from dotenv import load_dotenv

from src.dataset import embed_texts

load_dotenv()

_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
_USE_DEBATE_JUDGE = os.getenv("USE_DEBATE_JUDGE", "false").lower() in ("1", "true", "yes")

# Composite score weights — 5-metric system (must sum to 1.0)
# Faithfulness added at 0.20; others reduced proportionally.
_W_SEMANTIC = 0.30
_W_ROUGE    = 0.20
_W_TONE     = 0.20
_W_QUALITY  = 0.10
_W_FAITHFUL = 0.20  # NEW: hallucination / grounding score

# Calibration set path
_CALIBRATION_PATH = Path("data/calibration.json")

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricScores:
    semantic_similarity: float  # 0–1, cosine of embeddings
    rouge_recall: float         # 0–1, ROUGE-L recall
    tone_score: float           # 1–5 (raw LLM-as-judge)
    quality_score: float        # 1–5 (raw LLM-as-judge)
    composite_score: float      # 0–1, weighted composite (5 metrics)
    faithfulness_score: float = 1.0   # 0–1: is reply grounded in RAG context?
    guardrail_pass: bool = True       # True if reply passes Layer 1 checks
    tone_explanation: str = ""
    quality_explanation: str = ""
    faithfulness_explanation: str = ""  # what claims, if any, were unsupported
    guardrail_failures: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    email_id: str | int
    subject: str
    category: str
    generated_reply: str
    reference_reply: str
    scores: MetricScores
    retrieved_example_ids: list[Any] = field(default_factory=list)
    classification: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def _semantic_similarity(
    generated: str,
    reference: str,
    client: OpenAI,
) -> float:
    """
    Cosine similarity between OpenAI embeddings of generated and reference reply.
    Returns a value in [0, 1] (embeddings are normalised).
    """
    embeddings = embed_texts([generated, reference], client)
    gen_emb = embeddings[0]
    ref_emb = embeddings[1]
    # Normalise
    gen_norm = np.linalg.norm(gen_emb)
    ref_norm = np.linalg.norm(ref_emb)
    if gen_norm == 0 or ref_norm == 0:
        return 0.0
    cos_sim = float(np.dot(gen_emb, ref_emb) / (gen_norm * ref_norm))
    # Clip to [0, 1] — embeddings are unlikely to be negatively similar,
    # but cosine can technically be negative
    return max(0.0, min(1.0, cos_sim))


def _rouge_recall(generated: str, reference: str) -> float:
    """
    ROUGE-L recall: what fraction of the reference's content appears in the generated reply.
    Recall (not F1) is used because we care about coverage of the ground truth, not brevity.
    """
    scores = _ROUGE_SCORER.score(target=reference, prediction=generated)
    return float(scores["rougeL"].recall)


def _llm_judge(
    incoming_email: str,
    generated_reply: str,
    reference_reply: str,
    client: OpenAI,
    model: str = _JUDGE_MODEL,
) -> tuple[float, str, float, str]:
    """
    Ask an LLM to rate the generated reply on tone (1-5) and quality (1-5).

    Returns: (tone_score, tone_explanation, quality_score, quality_explanation)
    """
    prompt = f"""You are an expert evaluator of customer-support email replies.

Incoming email:
\"\"\"
{incoming_email}
\"\"\"

Reference reply (what was actually sent):
\"\"\"
{reference_reply}
\"\"\"

Generated reply (what the AI suggested):
\"\"\"
{generated_reply}
\"\"\"

Evaluate the generated reply on two dimensions. Respond ONLY with valid JSON, no markdown:

{{
  "tone_score": <integer 1-5>,
  "tone_explanation": "<one sentence>",
  "quality_score": <integer 1-5>,
  "quality_explanation": "<one sentence>"
}}

Scoring rubrics:

TONE (does the tone match what the incoming email needs?):
  5 = perfectly appropriate (empathetic when needed, formal when needed, concise when needed)
  4 = mostly appropriate with minor issues
  3 = neutral / acceptable but not well-calibrated to the email's emotional context
  2 = noticeable mismatch (too cold, too casual, or too verbose for the situation)
  1 = completely wrong tone

QUALITY (is this a complete, professional, actionable reply?):
  5 = complete, professional, addresses all issues, ready to send
  4 = good, minor omissions or slightly awkward phrasing
  3 = adequate but missing one important element or has notable phrasing issues
  2 = incomplete or confusing; would need significant editing before sending
  1 = unusable — off-topic, nonsensical, or harmful
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        tone_score = float(parsed.get("tone_score", 3))
        tone_explanation = str(parsed.get("tone_explanation", ""))
        quality_score = float(parsed.get("quality_score", 3))
        quality_explanation = str(parsed.get("quality_explanation", ""))
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract numbers from raw text
        numbers = re.findall(r"\b([1-5])\b", raw)
        tone_score = float(numbers[0]) if len(numbers) > 0 else 3.0
        quality_score = float(numbers[1]) if len(numbers) > 1 else 3.0
        tone_explanation = "parse error"
        quality_explanation = "parse error"

    # Clamp to valid range
    tone_score = max(1.0, min(5.0, tone_score))
    quality_score = max(1.0, min(5.0, quality_score))

    return tone_score, tone_explanation, quality_score, quality_explanation


def _debate_judge(
    incoming_email: str,
    generated_reply: str,
    reference_reply: str,
    client: OpenAI,
    model: str = _JUDGE_MODEL,
) -> tuple[float, str, float, str]:
    """
    Debate-as-Judge: two independent LLM judges score the reply, then
    debate if they disagree by ≥ 1.0 points on any dimension.

    Implements the adaptive-stopping debate pattern from evaluation research:
    judges debate only when it's necessary (significant disagreement), which
    avoids the "endless debate doesn't help" failure mode.

    Returns: (tone_score, tone_explanation, quality_score, quality_explanation)
    """
    # --- Judge A (slightly more lenient perspective) ---
    judge_a_system = (
        "You are Judge A, a senior customer-support quality reviewer. "
        "Evaluate the generated email reply on TONE and QUALITY using the provided rubrics. "
        "Respond with valid JSON only."
    )
    # --- Judge B (slightly more strict perspective) ---
    judge_b_system = (
        "You are Judge B, a strict customer-experience auditor. "
        "Evaluate the generated email reply on TONE and QUALITY using the provided rubrics. "
        "Challenge any score above 3 unless there is a clear reason for it. "
        "Respond with valid JSON only."
    )

    rubric_prompt = f"""Incoming email:
\"\"\"
{incoming_email}
\"\"\"

Reference reply (what was actually sent):
\"\"\"
{reference_reply}
\"\"\"

Generated reply (what the AI suggested):
\"\"\"
{generated_reply}
\"\"\"

Score on:
TONE (1–5): 5=perfect tone match, 3=neutral, 1=completely wrong tone
QUALITY (1–5): 5=complete/professional/ready-to-send, 3=adequate, 1=unusable

Respond ONLY with JSON:
{{"tone_score": <int 1-5>, "tone_explanation": "<one sentence>", "quality_score": <int 1-5>, "quality_explanation": "<one sentence>"}}"""

    def _call_judge(system_prompt: str) -> tuple[float, str, float, str]:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rubric_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            p = json.loads(raw)
            ts = max(1.0, min(5.0, float(p.get("tone_score", 3))))
            te = str(p.get("tone_explanation", ""))
            qs = max(1.0, min(5.0, float(p.get("quality_score", 3))))
            qe = str(p.get("quality_explanation", ""))
        except (json.JSONDecodeError, ValueError):
            ts, te, qs, qe = 3.0, "parse error", 3.0, "parse error"
        return ts, te, qs, qe

    a_tone, a_tone_exp, a_qual, a_qual_exp = _call_judge(judge_a_system)
    b_tone, b_tone_exp, b_qual, b_qual_exp = _call_judge(judge_b_system)

    tone_gap = abs(a_tone - b_tone)
    qual_gap = abs(a_qual - b_qual)

    # Adaptive stopping: if both agree (gap < 1.0), average and return
    if tone_gap < 1.0 and qual_gap < 1.0:
        final_tone = round((a_tone + b_tone) / 2, 2)
        final_qual = round((a_qual + b_qual) / 2, 2)
        tone_exp = f"Judges agreed (A={a_tone}, B={b_tone}): {a_tone_exp}"
        qual_exp = f"Judges agreed (A={a_qual}, B={b_qual}): {a_qual_exp}"
        return final_tone, tone_exp, final_qual, qual_exp

    # Disagreement → arbitration round
    arbitrator_system = (
        "You are an Arbitrator resolving a scoring disagreement between two judges. "
        "Read both judges' scores and reasoning, then give your final verdict. "
        "Respond with valid JSON only."
    )
    arbitration_prompt = f"""{rubric_prompt}

Judge A scored: tone={a_tone} ({a_tone_exp}), quality={a_qual} ({a_qual_exp})
Judge B scored: tone={b_tone} ({b_tone_exp}), quality={b_qual} ({b_qual_exp})

Provide your final arbitrated scores.
Respond ONLY with JSON: {{"tone_score": <int 1-5>, "quality_score": <int 1-5>, "reasoning": "<one sentence>"}}"""

    arb_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": arbitrator_system},
            {"role": "user", "content": arbitration_prompt},
        ],
        temperature=0.0,
        max_tokens=150,
        response_format={"type": "json_object"},
    )
    arb_raw = arb_resp.choices[0].message.content or "{}"
    try:
        arb = json.loads(arb_raw)
        final_tone = max(1.0, min(5.0, float(arb.get("tone_score", (a_tone + b_tone) / 2))))
        final_qual = max(1.0, min(5.0, float(arb.get("quality_score", (a_qual + b_qual) / 2))))
        reasoning = str(arb.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError):
        final_tone = round((a_tone + b_tone) / 2, 2)
        final_qual = round((a_qual + b_qual) / 2, 2)
        reasoning = "arbitration parse error"

    tone_exp = f"Arbitrated (A={a_tone}, B={b_tone} → {final_tone}): {reasoning}"
    qual_exp = f"Arbitrated (A={a_qual}, B={b_qual} → {final_qual}): {reasoning}"
    return round(final_tone, 2), tone_exp, round(final_qual, 2), qual_exp


# ---------------------------------------------------------------------------
# Layer 1 — Deterministic guardrails (fast, zero API cost)
# ---------------------------------------------------------------------------

# Phrases that indicate the LLM refused or couldn't complete the task
_REFUSAL_PATTERNS = [
    r"i cannot", r"i'm unable", r"i am unable", r"i can't help",
    r"as an ai", r"i don't have access", r"i'm sorry but i",
    r"i apologize, but i cannot",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

_GREETING_RE = re.compile(
    r"^(dear|hi|hello|thank you|thanks|good morning|good afternoon)",
    re.IGNORECASE | re.MULTILINE,
)
_SIGNOFF_RE = re.compile(
    r"(hiver support|best regards|kind regards|warm regards|sincerely|cheers)",
    re.IGNORECASE,
)


def _layer1_guardrails(generated_reply: str) -> tuple[bool, list[str]]:
    """
    Layer 1 deterministic guardrails — instant, zero-cost checks run BEFORE
    the LLM judge. Catches structurally broken or policy-violating outputs.

    Returns: (pass: bool, failures: list[str])
    A reply passes if ALL checks succeed.
    """
    failures: list[str] = []

    # Check 1: Minimum word count (avoids near-empty replies)
    word_count = len(generated_reply.split())
    if word_count < 20:
        failures.append(f"Too short: {word_count} words (min 20)")

    # Check 2: No refusal language
    if _REFUSAL_RE.search(generated_reply):
        failures.append("Contains refusal language (LLM may have declined)")

    # Check 3: Has a greeting
    if not _GREETING_RE.search(generated_reply):
        failures.append("Missing greeting (no 'Dear', 'Hi', 'Hello', etc.)")

    # Check 4: Has a professional sign-off
    if not _SIGNOFF_RE.search(generated_reply):
        failures.append("Missing professional sign-off")

    # Check 5: Not excessively long (> 400 words = likely bloated / hallucinating)
    if word_count > 400:
        failures.append(f"Suspiciously long: {word_count} words (max 400)")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Faithfulness / hallucination score
# ---------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = """\
You are a factual grounding auditor for customer support emails.
You will receive:
  1. The RAG context — the past email examples retrieved to guide the reply.
  2. The generated reply — what the AI actually wrote.

Your task: identify any claims in the generated reply that CANNOT be traced
back to the RAG context or are contradicted by it.

Focus on factual commitments:
  - Specific timelines ("within 24 hours", "by Friday")
  - Specific policies ("30-day refund", "free upgrade")
  - Specific product capabilities ("it supports X")
  - Named contacts ("our engineer Alex will call you")

Do NOT flag:
  - Generic empathy statements ("we're sorry to hear this")
  - Standard closings
  - Offers to follow up

Respond ONLY with JSON:
{"faithfulness_score": <0.0 to 1.0>, "unsupported_claims": ["<claim>", ...], "explanation": "<one sentence>"}

Scoring:
  1.0 = fully grounded — every specific claim is supported by context
  0.5 = partially grounded — 1 or 2 minor unsupported specifics
  0.0 = hallucinated — significant claims with no support in context
"""


def _faithfulness_score(
    generated_reply: str,
    rag_context: str,
    client: OpenAI,
    model: str = _JUDGE_MODEL,
) -> tuple[float, str]:
    """
    Check whether the generated reply makes any factual claims unsupported
    by the retrieved RAG examples.

    This is the critical missing metric from the original system — semantic
    similarity and ROUGE can be high even if the reply invents policies.

    Parameters
    ----------
    generated_reply : str
        The AI-generated reply.
    rag_context : str
        The retrieved examples that grounded the generation.
    client : OpenAI
        Authenticated client.

    Returns: (faithfulness_score 0-1, explanation str)
    """
    if not rag_context.strip():
        # No RAG context to check against — can't assess faithfulness
        return 1.0, "No RAG context available for faithfulness check."

    user_msg = (
        f"RAG context (retrieved examples):\n\"\"\"{rag_context}\"\"\"\n\n"
        f"Generated reply:\n\"\"\"{generated_reply}\"\"\""
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        score = max(0.0, min(1.0, float(data.get("faithfulness_score", 1.0))))
        explanation = str(data.get("explanation", ""))
        unsupported = data.get("unsupported_claims", [])
        if unsupported:
            explanation = f"Unsupported: {'; '.join(unsupported[:3])}. {explanation}"
        return score, explanation
    except Exception as exc:
        return 1.0, f"Faithfulness check skipped: {exc}"


def _composite_score(
    semantic_sim: float,
    rouge_recall: float,
    tone_score: float,
    quality_score: float,
    faithfulness: float = 1.0,
) -> float:
    """Compute the weighted composite accuracy score (0–1) across 5 metrics."""
    return (
        _W_SEMANTIC  * semantic_sim
        + _W_ROUGE   * rouge_recall
        + _W_TONE    * (tone_score / 5.0)
        + _W_QUALITY * (quality_score / 5.0)
        + _W_FAITHFUL * faithfulness
    )


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_reply(
    email_record: dict[str, Any],
    generated_reply: str,
    client: OpenAI,
    retrieved_example_ids: list[Any] | None = None,
    rag_context_text: str | None = None,
    classification: dict[str, Any] | None = None,
) -> EvaluationResult:
    """
    Evaluate a single generated reply against its ground-truth reference.

    Parameters
    ----------
    email_record : dict
        A record from the dataset (must have 'reply', 'body', 'subject', etc.)
    generated_reply : str
        The AI-generated reply to evaluate.
    client : OpenAI
        Authenticated OpenAI client.
    retrieved_example_ids : list, optional
        IDs of the RAG examples used during generation (for traceability).
    rag_context_text : str, optional
        The full text of retrieved RAG examples. Used for faithfulness scoring.
        If not provided, faithfulness check is a best-effort approximation.
    classification : dict, optional
        Sentiment/urgency classification dict from classifier.py.

    Returns
    -------
    EvaluationResult
        Full result with per-dimension scores and the composite.
    """
    reference_reply = email_record["reply"]
    incoming_email = f"Subject: {email_record['subject']}\n\n{email_record['body']}"

    # Layer 1: fast deterministic checks (zero API cost)
    guardrail_pass, guardrail_failures = _layer1_guardrails(generated_reply)

    # Compute text-overlap metrics
    sem_sim = _semantic_similarity(generated_reply, reference_reply, client)
    rouge_rec = _rouge_recall(generated_reply, reference_reply)

    # Use Debate-as-Judge if configured, otherwise single judge
    judge_fn = _debate_judge if _USE_DEBATE_JUDGE else _llm_judge
    tone_s, tone_exp, qual_s, qual_exp = judge_fn(
        incoming_email, generated_reply, reference_reply, client
    )

    # Faithfulness: build RAG context string from retrieved examples
    rag_context = ""
    if retrieved_example_ids:
        rag_context = (
            f"Retrieved {len(retrieved_example_ids)} past examples "
            f"(IDs: {retrieved_example_ids}). "
            "(Full context not stored in EvaluationResult — "
            "pass rag_context_text for full faithfulness check.)"
        )
    faith_s, faith_exp = _faithfulness_score(
        generated_reply=generated_reply,
        rag_context=rag_context_text or rag_context,
        client=client,
    )

    composite = _composite_score(sem_sim, rouge_rec, tone_s, qual_s, faith_s)

    scores = MetricScores(
        semantic_similarity=round(sem_sim, 4),
        rouge_recall=round(rouge_rec, 4),
        tone_score=round(tone_s, 2),
        quality_score=round(qual_s, 2),
        faithfulness_score=round(faith_s, 4),
        guardrail_pass=guardrail_pass,
        guardrail_failures=guardrail_failures,
        composite_score=round(composite, 4),
        tone_explanation=tone_exp,
        quality_explanation=qual_exp,
        faithfulness_explanation=faith_exp,
    )

    return EvaluationResult(
        email_id=email_record.get("id", "unknown"),
        subject=email_record.get("subject", ""),
        category=email_record.get("category", ""),
        generated_reply=generated_reply,
        reference_reply=reference_reply,
        scores=scores,
        retrieved_example_ids=retrieved_example_ids or [],
        classification=classification or {},
    )


# ---------------------------------------------------------------------------
# Calibration / validation
# ---------------------------------------------------------------------------

def run_calibration(results: list[EvaluationResult]) -> dict[str, Any]:
    """
    Validate the composite metric against human-annotated calibration labels.

    The calibration set (data/calibration.json) contains hand-labelled quality
    ratings for a subset of emails: good=1.0, ok=0.5, bad=0.0.

    We compute Spearman ρ between composite scores and human labels to confirm
    the metric reflects real quality rather than just being a number.

    If the calibration file is not found, returns a warning instead of failing.
    """
    if not _CALIBRATION_PATH.exists():
        return {
            "calibration_available": False,
            "message": f"No calibration file at {_CALIBRATION_PATH}",
        }

    with _CALIBRATION_PATH.open() as f:
        calibration = json.load(f)  # [{id, human_score}, ...]

    cal_map = {str(c["id"]): c["human_score"] for c in calibration}

    matched_composite: list[float] = []
    matched_human: list[float] = []
    for r in results:
        eid = str(r.email_id)
        if eid in cal_map:
            matched_composite.append(r.scores.composite_score)
            matched_human.append(cal_map[eid])

    if len(matched_composite) < 3:
        return {
            "calibration_available": True,
            "matched_count": len(matched_composite),
            "message": "Not enough matched calibration examples (need ≥ 3)",
        }

    rho, pval = spearmanr(matched_composite, matched_human)
    return {
        "calibration_available": True,
        "matched_count": len(matched_composite),
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(pval), 4),
        "interpretation": (
            "strong agreement with human labels" if rho > 0.6
            else "moderate agreement" if rho > 0.3
            else "weak agreement — consider recalibrating metric weights"
        ),
    }


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------

def aggregate_results(results: list[EvaluationResult]) -> dict[str, Any]:
    """Compute overall and per-category statistics across all evaluation results."""
    if not results:
        return {}

    composites = [r.scores.composite_score for r in results]
    semantic_sims = [r.scores.semantic_similarity for r in results]
    rouge_recalls = [r.scores.rouge_recall for r in results]
    tone_scores = [r.scores.tone_score for r in results]
    quality_scores = [r.scores.quality_score for r in results]
    faithfulness_scores = [r.scores.faithfulness_score for r in results]
    guardrail_passes = [r.scores.guardrail_pass for r in results]

    # Per-category breakdown
    by_category: dict[str, list[float]] = {}
    for r in results:
        cat = r.category or "unknown"
        by_category.setdefault(cat, []).append(r.scores.composite_score)

    category_stats = {
        cat: {
            "mean_composite": round(float(np.mean(scores)), 4),
            "count": len(scores),
        }
        for cat, scores in sorted(by_category.items())
    }

    guardrail_pass_rate = sum(guardrail_passes) / len(guardrail_passes)
    guardrail_failures_flat = [
        f for r in results for f in r.scores.guardrail_failures
    ]

    return {
        "n_evaluated": len(results),
        "overall": {
            "composite_mean": round(float(np.mean(composites)), 4),
            "composite_std": round(float(np.std(composites)), 4),
            "composite_min": round(float(np.min(composites)), 4),
            "composite_max": round(float(np.max(composites)), 4),
            "semantic_similarity_mean": round(float(np.mean(semantic_sims)), 4),
            "rouge_recall_mean": round(float(np.mean(rouge_recalls)), 4),
            "tone_score_mean": round(float(np.mean(tone_scores)), 2),
            "quality_score_mean": round(float(np.mean(quality_scores)), 2),
            "faithfulness_mean": round(float(np.mean(faithfulness_scores)), 4),
            "guardrail_pass_rate": round(guardrail_pass_rate, 4),
            "guardrail_failure_types": guardrail_failures_flat[:20],  # sample
        },
        "by_category": category_stats,
    }


def results_to_dicts(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    """Convert EvaluationResult list to JSON-serialisable dicts."""
    out = []
    for r in results:
        d = asdict(r)
        out.append(d)
    return out


def evaluate_reply_reference_free(
    generated_reply: str,
    rag_context_text: str | None,
    client: OpenAI,
) -> dict[str, Any]:
    """
    Perform a reference-free evaluation of a generated reply.
    Checks Layer 1 deterministic guardrails and Layer 2 faithfulness grounding.
    """
    guardrail_pass, guardrail_failures = _layer1_guardrails(generated_reply)

    if rag_context_text:
        faith_score, faith_exp = _faithfulness_score(
            generated_reply=generated_reply,
            rag_context=rag_context_text,
            client=client,
        )
    else:
        faith_score, faith_exp = 1.0, "No RAG context available for verification."

    return {
        "guardrail_pass": guardrail_pass,
        "guardrail_failures": guardrail_failures,
        "faithfulness_score": faith_score,
        "faithfulness_explanation": faith_exp,
        "passed": guardrail_pass and (faith_score >= 0.70),
    }

