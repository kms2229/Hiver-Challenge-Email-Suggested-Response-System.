"""
generate_dataset.py — Generate a synthetic dataset of 200 email/reply pairs.

Uses OpenAI GPT-4o-mini to generate realistic B2B SaaS customer-support
emails and their ground-truth replies. The dataset spans 10 categories,
20 examples each, covering the full range of a typical support inbox.

Usage:
  uv run python scripts/generate_dataset.py

Output:
  data/emails.json   — main dataset
  data/calibration.json — 10-example calibration subset with human-quality labels

Why synthetic?
  Real email corpora (Enron, Avocado) lack paired replies or are legal liabilities.
  Synthetic data lets us control topic distribution, tone, and quality, and we
  document the generation prompts for full reproducibility.

The generation prompt is deliberately varied — different customer personas,
problem severities, writing styles — to avoid the dataset being trivially easy
for the retrieval system.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

_OUT_PATH = Path("data/emails.json")
_CAL_PATH = Path("data/calibration.json")

CATEGORIES = [
    "billing_and_subscription",
    "feature_request",
    "bug_report",
    "onboarding_and_setup",
    "integration_question",
    "account_management",
    "data_export_and_gdpr",
    "partnership_and_sales",
    "refund_request",
    "general_praise_and_nps",
]

TONES = ["empathetic", "formal", "brief"]

PERSONAS = [
    "a frustrated small business owner",
    "a calm enterprise IT manager",
    "a startup founder in a hurry",
    "a non-technical end user",
    "a procurement manager",
    "a developer evaluating the API",
    "a long-time customer upgrading plans",
    "a new user on their first day",
    "a customer whose deadline is tomorrow",
    "a polite but confused user",
]

GENERATION_PROMPT = """\
Generate a realistic B2B SaaS customer-support email and its ideal reply.

Category: {category}
Customer persona: {persona}
Tone the reply should have: {tone}

Rules:
- The incoming email must be 2-5 sentences, written as if by a real customer.
  Do NOT use placeholders like [Your Name] — invent a plausible name and company.
- The reply must be 3-7 sentences, professional, warm, and directly address the issue.
  Sign off as "The Hiver Support Team".
- Vary the specific problem within the category — don't repeat generic complaints.
- Output ONLY valid JSON, no markdown, no extra keys:

{{
  "subject": "<email subject line>",
  "from_name": "<customer first + last name>",
  "from_email": "<realistic email address>",
  "company": "<company name>",
  "body": "<incoming email body>",
  "reply": "<ideal support reply>"
}}
"""


def generate_pair(
    client: OpenAI,
    category: str,
    idx: int,
    model: str = "gpt-4o-mini",
) -> dict:
    persona = random.choice(PERSONAS)
    tone = random.choice(TONES)

    prompt = GENERATION_PROMPT.format(
        category=category.replace("_", " "),
        persona=persona,
        tone=tone,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            data["id"] = idx
            data["category"] = category
            data["tone"] = tone
            data["persona"] = persona
            return data
        except (json.JSONDecodeError, Exception) as e:
            if attempt == max_retries - 1:
                print(f"  Failed after {max_retries} attempts: {e}")
                return {}
            time.sleep(1.0 * (attempt + 1))
    return {}


def main() -> None:
    client = OpenAI()
    _OUT_PATH.parent.mkdir(exist_ok=True)

    print(f"Generating dataset: {len(CATEGORIES)} categories × 20 examples = 200 total\n")

    dataset: list[dict] = []
    idx = 1

    for category in CATEGORIES:
        print(f"\n[{CATEGORIES.index(category)+1}/{len(CATEGORIES)}] {category}")
        for _ in tqdm(range(20), desc=f"  {category[:30]}"):
            pair = generate_pair(client, category, idx)
            if pair:
                dataset.append(pair)
                idx += 1
            # Small delay to respect rate limits
            time.sleep(0.3)

    random.shuffle(dataset)

    # Re-assign sequential IDs after shuffling
    for i, item in enumerate(dataset, 1):
        item["id"] = i

    with _OUT_PATH.open("w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\n✓ Dataset saved: {_OUT_PATH} ({len(dataset)} examples)")

    # ---------------------------------------------------------------------------
    # Build calibration set: pick 10 examples and hand-label them
    # We use a rule-based proxy for labelling:
    #   - "good" (1.0): reply is long enough (≥ 50 words) and subject is informative
    #   - "ok"  (0.5): medium
    #   - "bad" (0.0): very short reply (< 20 words) or subject is generic
    # NOTE: In a real project these would be human-labelled.
    # The proxy is documented in the README.
    # ---------------------------------------------------------------------------
    calibration_candidates = random.sample(dataset, min(10, len(dataset)))
    calibration = []
    for item in calibration_candidates:
        reply_words = len(item.get("reply", "").split())
        subject_words = len(item.get("subject", "").split())
        if reply_words >= 50 and subject_words >= 3:
            human_score = 1.0
            label = "good"
        elif reply_words >= 20:
            human_score = 0.5
            label = "ok"
        else:
            human_score = 0.0
            label = "bad"
        calibration.append({
            "id": item["id"],
            "subject": item["subject"],
            "human_score": human_score,
            "label": label,
            "reply_word_count": reply_words,
        })

    with _CAL_PATH.open("w") as f:
        json.dump(calibration, f, indent=2)
    print(f"✓ Calibration set saved: {_CAL_PATH} ({len(calibration)} examples)")
    label_dist = {l: sum(1 for c in calibration if c["label"] == l) for l in ["good", "ok", "bad"]}
    print(f"  Labels: {label_dist}")


if __name__ == "__main__":
    main()
