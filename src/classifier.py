"""
classifier.py — Email sentiment + urgency pre-classifier.

Before generating a reply, this module analyses the incoming email on two
independent dimensions:

  Sentiment   frustrated | neutral | satisfied
  Urgency     low | medium | high | critical

These are injected into the generator as additional context that shapes tone
instructions at generation time:
  - frustrated + critical  → strong empathy opening, immediate concrete action
  - neutral + medium       → standard professional reply
  - satisfied + low        → brief, warm, no-fluff response

Why this matters
----------------
The current system gives every email the same system prompt regardless of
the customer's emotional state. A neutral, procedural reply to a furious
customer will score reasonably on semantic similarity — but is a real failure
in production. This classifier surfaces that signal before generation.

It also computes `escalation_risk` — a boolean flag for emails showing signs
of churn, threats of public complaint, or multi-issue frustration — so the
caller can apply stricter quality controls or route to a senior agent.

Usage
-----
  from src.classifier import classify_email
  cls = classify_email(subject, body, client)
  print(cls.sentiment, cls.urgency, cls.tone_instruction)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gpt-4o-mini")

_SYSTEM_PROMPT = """\
You are an expert customer-support email analyst.
Analyse the incoming customer email and return a JSON object with these exact fields:

{
  "sentiment": "<frustrated|neutral|satisfied>",
  "urgency": "<low|medium|high|critical>",
  "escalation_risk": <true|false>,
  "primary_issue": "<one sentence describing the customer's main problem>",
  "tone_instruction": "<specific instruction for the reply writer about how to calibrate tone>"
}

Definitions:
  sentiment:
    frustrated  — customer expresses anger, disappointment, repeated failures, or strong negative emotion
    neutral     — matter-of-fact inquiry, neither positive nor negative
    satisfied   — customer is happy, grateful, or giving positive feedback

  urgency:
    critical    — customer mentions legal action, public posting, service is completely down, safety risk
    high        — customer is blocked, losing money, or has a hard deadline mentioned
    medium      — issue affects workflow but workarounds exist; customer wants resolution but not desperate
    low         — general question, enhancement request, positive feedback

  escalation_risk: true if any of:
    - mentions lawyer, legal, complaint, social media, negative review, cancellation, churn
    - expresses extreme frustration across multiple paragraphs
    - reports data loss or security issue

  tone_instruction:
    A specific, actionable instruction for the reply writer.
    Examples:
      "Open with a strong, specific apology. Acknowledge the downtime directly. Offer a concrete timeline."
      "Keep reply brief and warm. Customer is happy — don't over-explain."
      "Use formal language. Customer is asking a precise technical question."

Respond ONLY with the JSON object. No commentary.
"""

_FALLBACK_INSTRUCTION = (
    "Write a professional, empathetic reply that addresses the customer's question clearly."
)


@dataclass
class EmailClassification:
    sentiment: Literal["frustrated", "neutral", "satisfied"]
    urgency: Literal["low", "medium", "high", "critical"]
    escalation_risk: bool
    primary_issue: str
    tone_instruction: str


def classify_email(
    subject: str,
    body: str,
    client: OpenAI,
    model: str = _CLASSIFIER_MODEL,
) -> EmailClassification:
    """
    Classify the sentiment, urgency, and escalation risk of an incoming email.

    Parameters
    ----------
    subject, body : str
        Incoming email content.
    client : OpenAI
        Authenticated client.
    model : str
        Model to use (default: gpt-4o-mini for cost efficiency).

    Returns
    -------
    EmailClassification
        Structured classification result with tone instruction for the generator.
    """
    user_msg = f"Subject: {subject}\n\n{body}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        sentiment = data.get("sentiment", "neutral")
        if sentiment not in ("frustrated", "neutral", "satisfied"):
            sentiment = "neutral"

        urgency = data.get("urgency", "medium")
        if urgency not in ("low", "medium", "high", "critical"):
            urgency = "medium"

        return EmailClassification(
            sentiment=sentiment,
            urgency=urgency,
            escalation_risk=bool(data.get("escalation_risk", False)),
            primary_issue=str(data.get("primary_issue", ""))[:300],
            tone_instruction=str(data.get("tone_instruction", _FALLBACK_INSTRUCTION))[:400],
        )

    except Exception:
        # Graceful fallback — never crash generation because classification failed
        return EmailClassification(
            sentiment="neutral",
            urgency="medium",
            escalation_risk=False,
            primary_issue="",
            tone_instruction=_FALLBACK_INSTRUCTION,
        )


def build_tone_context(classification: EmailClassification) -> str:
    """
    Build the tone-context string injected into the generator system prompt.
    """
    urgency_label = {
        "low": "low urgency",
        "medium": "moderate urgency",
        "high": "HIGH URGENCY",
        "critical": "CRITICAL — treat as top priority",
    }.get(classification.urgency, "moderate urgency")

    sentiment_label = {
        "frustrated": "FRUSTRATED customer",
        "neutral": "neutral customer",
        "satisfied": "satisfied customer",
    }.get(classification.sentiment, "neutral customer")

    parts = [
        f"Customer context: {sentiment_label}, {urgency_label}.",
    ]
    if classification.primary_issue:
        parts.append(f"Primary issue: {classification.primary_issue}")
    if classification.escalation_risk:
        parts.append(
            "⚠ ESCALATION RISK: This customer shows signs of churn or public complaint. "
            "Be extra careful with tone and commitments."
        )
    parts.append(f"Tone guidance: {classification.tone_instruction}")

    return "\n".join(parts)
