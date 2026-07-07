"""
scrubber.py — PII scrubbing utilities for customer support emails and replies.
"""

from __future__ import annotations

import re

# Compiled regex patterns for PII detection
# 1. Emails
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    re.IGNORECASE,
)

# 2. Phone numbers (matches +1-555-555-5555, (555) 555-5555, 5555555555, etc.)
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# 3. Credit cards (matches 13 to 16 digit sequences with optional spaces/dashes)
_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ -]*?){13,16}\b"
)

# 4. API keys & Auth tokens (e.g., sk-..., bearer tokens, or similar key structures)
_API_KEY_RE = re.compile(
    r"\b(?:sk-[a-zA-Z0-9]{20,80}|bearer\s+[a-zA-Z0-9\-._~+/]+=*)\b",
    re.IGNORECASE,
)


def scrub_pii(text: str | None) -> str:
    """
    Scan text and replace emails, phone numbers, credit cards, and API keys
    with anonymized placeholders.
    """
    if not text:
        return ""

    # Scrub emails
    text = _EMAIL_RE.sub("[EMAIL]", text)

    # Scrub credit cards first (since they are long digit sequences and could overlap with phone numbers)
    text = _CREDIT_CARD_RE.sub("[CREDIT_CARD]", text)

    # Scrub phone numbers
    text = _PHONE_RE.sub("[PHONE]", text)

    # Scrub API keys
    text = _API_KEY_RE.sub("[API_KEY]", text)

    return text
