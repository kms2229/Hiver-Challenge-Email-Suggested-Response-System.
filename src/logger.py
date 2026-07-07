"""
logger.py — Preference pair logger for future DPO/RLHF fine-tuning.

Every evaluated email reply is logged to a JSONL file in DPO-compatible
format. Over time, this accumulates the training data needed to fine-tune
a model directly on this system's quality signal — without any human
annotation cost today.

Format
------
Each JSONL line is a JSON object with:
  - prompt        : the incoming email (system + user)
  - chosen        : the reply with the highest composite score seen so far
  - rejected      : the reply with the lowest composite score seen so far
  - metadata      : scores, mode, timestamp, email_id

DPO-readiness
-------------
When you have 500+ preference pairs (chosen vs. rejected replies for the
same email), you can run DPO training with a library like trl:
  trainer = DPOTrainer(model, ref_model, train_dataset=log_data, ...)

The current module just logs. Training is a future step.

Design decision: we log ALL evaluation results, not just "good" ones.
This means the file accumulates naturally as you run evaluations.
The `to_dpo_pairs()` function post-processes the log into training pairs
when you're ready.

Usage
-----
  from src.logger import log_evaluation, to_dpo_pairs
  log_evaluation(eval_result, generation_result)
  pairs = to_dpo_pairs("logs/preference_log.jsonl")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_PATH = Path(os.getenv("PREFERENCE_LOG_PATH", "logs/preference_log.jsonl"))

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional customer-support agent for a B2B SaaS company called Hiver. "
    "Write a helpful, empathetic, and concise email reply."
)


def log_evaluation(
    eval_result: Any,          # EvaluationResult from evaluator.py
    generation_result: dict,   # dict from generator
    log_path: Path | str | None = None,
) -> None:
    """
    Append a single evaluation result to the JSONL preference log.

    Parameters
    ----------
    eval_result : EvaluationResult
        The full evaluation result with scores.
    generation_result : dict
        The raw dict returned by the generator (includes mode, retrieved_examples, etc.)
    log_path : Path | str, optional
        Where to write the log. Defaults to PREFERENCE_LOG_PATH env var or
        logs/preference_log.jsonl.
    """
    path = Path(log_path or _LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "email_id": str(eval_result.email_id),
        "subject": eval_result.subject,
        "category": eval_result.category,
        # DPO prompt format (OpenAI messages style)
        "prompt": [
            {"role": "system", "content": _SYSTEM_PROMPT_TEMPLATE},
            {
                "role": "user",
                "content": (
                    f"Subject: {eval_result.subject}\n\n"
                    f"{eval_result.reference_reply.split('The Hiver Support Team')[0].strip()}"
                    # Use the actual email body from reference (not stored directly in EvaluationResult)
                    # This is a best-effort placeholder; in production you'd store the body too
                ),
            },
        ],
        # The generated reply is the "chosen" candidate if score ≥ 0.7, else "rejected"
        "generated_reply": eval_result.generated_reply,
        "reference_reply": eval_result.reference_reply,
        "scores": {
            "composite": eval_result.scores.composite_score,
            "semantic_similarity": eval_result.scores.semantic_similarity,
            "rouge_recall": eval_result.scores.rouge_recall,
            "tone_score": eval_result.scores.tone_score,
            "quality_score": eval_result.scores.quality_score,
        },
        "generation_mode": generation_result.get("mode", "unknown"),
        "model": generation_result.get("model", "unknown"),
        "was_re_queried": generation_result.get("was_re_queried", False),
        "retrieved_example_ids": [
            ex.get("id") for ex in generation_result.get("retrieved_examples", [])
        ],
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_dpo_pairs(
    log_path: Path | str | None = None,
    score_threshold_chosen: float = 0.70,
    score_threshold_rejected: float = 0.50,
) -> list[dict[str, Any]]:
    """
    Post-process the preference log into DPO training pairs.

    A DPO pair consists of:
      - chosen  : the best reply for a given email (composite ≥ threshold_chosen)
      - rejected: the worst reply for the same email (composite ≤ threshold_rejected)

    Groups log records by email_id. For each email that has at least one
    chosen and one rejected candidate, yields a training pair.

    Parameters
    ----------
    log_path : Path | str, optional
        Path to the JSONL log file.
    score_threshold_chosen : float
        Minimum composite score to qualify as "chosen". Default 0.70.
    score_threshold_rejected : float
        Maximum composite score to qualify as "rejected". Default 0.50.

    Returns
    -------
    list of dicts, each with keys: prompt, chosen, rejected, metadata
    """
    path = Path(log_path or _LOG_PATH)
    if not path.exists():
        return []

    # Group by email_id
    by_email: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                eid = rec.get("email_id", "unknown")
                by_email.setdefault(eid, []).append(rec)
            except json.JSONDecodeError:
                continue

    pairs = []
    for email_id, records in by_email.items():
        # Find best and worst reply for this email
        sorted_recs = sorted(
            records,
            key=lambda r: r.get("scores", {}).get("composite", 0),
            reverse=True,
        )
        chosen_candidates = [
            r for r in sorted_recs
            if r.get("scores", {}).get("composite", 0) >= score_threshold_chosen
        ]
        rejected_candidates = [
            r for r in sorted_recs
            if r.get("scores", {}).get("composite", 1) <= score_threshold_rejected
        ]

        if not chosen_candidates or not rejected_candidates:
            continue

        best = chosen_candidates[0]
        worst = rejected_candidates[-1]

        pairs.append({
            "prompt": best.get("prompt", []),
            "chosen": best["generated_reply"],
            "rejected": worst["generated_reply"],
            "metadata": {
                "email_id": email_id,
                "subject": best.get("subject"),
                "category": best.get("category"),
                "chosen_score": best["scores"]["composite"],
                "rejected_score": worst["scores"]["composite"],
                "chosen_mode": best.get("generation_mode"),
                "rejected_mode": worst.get("generation_mode"),
            },
        })

    return pairs


def log_stats(log_path: Path | str | None = None) -> dict[str, Any]:
    """Return summary statistics about the preference log."""
    path = Path(log_path or _LOG_PATH)
    if not path.exists():
        return {"exists": False, "records": 0}

    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return {"exists": True, "records": 0}

    composites = [r.get("scores", {}).get("composite", 0) for r in records]
    modes = {}
    for r in records:
        m = r.get("generation_mode", "unknown")
        modes[m] = modes.get(m, 0) + 1

    dpo_pairs = to_dpo_pairs(log_path)

    return {
        "exists": True,
        "records": len(records),
        "unique_emails": len({r.get("email_id") for r in records}),
        "avg_composite": round(sum(composites) / len(composites), 4),
        "by_mode": modes,
        "dpo_pairs_available": len(dpo_pairs),
        "log_path": str(path.resolve()),
    }
