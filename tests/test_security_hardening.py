"""
tests/test_security_hardening.py — Unit tests for PII scrubbing and
reference-free evaluation.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.scrubber import scrub_pii
from src.evaluator import evaluate_reply_reference_free


# ===========================================================================
# PII Scrubber Tests
# ===========================================================================

def test_scrub_pii_empty():
    assert scrub_pii("") == ""
    assert scrub_pii(None) == ""


def test_scrub_pii_emails():
    text = "Please reach out to support@hiver.com or jane.doe+test@sub.domain.org for info."
    scrubbed = scrub_pii(text)
    assert "[EMAIL]" in scrubbed
    assert "support@hiver.com" not in scrubbed
    assert "jane.doe+test" not in scrubbed
    assert scrubbed == "Please reach out to [EMAIL] or [EMAIL] for info."


def test_scrub_pii_phone_numbers():
    text = "Call us at +1-555-867-5309 or (123) 456-7890."
    scrubbed = scrub_pii(text)
    assert "[PHONE]" in scrubbed
    assert "555-867-5309" not in scrubbed
    assert "123" not in scrubbed
    assert scrubbed == "Call us at [PHONE] or [PHONE]."


def test_scrub_pii_credit_cards():
    text = "My visa number is 4111-2222-3333-4444 and my mastercard is 5123 4567 8901 2345."
    scrubbed = scrub_pii(text)
    assert "[CREDIT_CARD]" in scrubbed
    assert "4111" not in scrubbed
    assert "5123" not in scrubbed
    assert scrubbed == "My visa number is [CREDIT_CARD] and my mastercard is [CREDIT_CARD]."


def test_scrub_pii_api_keys():
    text = "Set OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD and bearer fake-token-1234."
    scrubbed = scrub_pii(text)
    assert "[API_KEY]" in scrubbed
    assert "sk-abc" not in scrubbed
    assert "bearer" not in scrubbed


# ===========================================================================
# Reference-Free Evaluator Tests
# ===========================================================================

def test_evaluate_reply_reference_free_pass():
    """A well-formed reply with RAG context should pass."""
    generated = (
        "Hi Jane,\n\n"
        "Thank you for contacting us. We will look into your billing issue "
        "and resolve it as soon as possible.\n\n"
        "The Hiver Support Team"
    )
    rag_context = "Billing issues are resolved within 24 hours."
    mock_client = MagicMock()

    # Mock faithfulness score return
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "faithfulness_score": 1.0,
                        "unsupported_claims": [],
                        "explanation": "Fully grounded.",
                    }
                )
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    res = evaluate_reply_reference_free(generated, rag_context, mock_client)

    assert res["guardrail_pass"] is True
    assert res["faithfulness_score"] == 1.0
    assert res["passed"] is True
    assert res["guardrail_failures"] == []


def test_evaluate_reply_reference_free_fail_guardrails():
    """A reply that violates guardrails (too short) should fail audit."""
    generated = "Hi, sorry. We will fix it."
    rag_context = "RAG context."
    mock_client = MagicMock()

    res = evaluate_reply_reference_free(generated, rag_context, mock_client)

    assert res["guardrail_pass"] is False
    assert res["passed"] is False
    assert any("too short" in f.lower() for f in res["guardrail_failures"])


def test_evaluate_reply_reference_free_fail_faithfulness():
    """A reply that is ungrounded / hallucinated should fail audit."""
    generated = (
        "Hi Jane,\n\n"
        "Thank you for contacting us. We will refund you $10,000 immediately "
        "and delete your account.\n\n"
        "The Hiver Support Team"
    )
    rag_context = "We do not offer cash refunds for subscription plans."
    mock_client = MagicMock()

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "faithfulness_score": 0.0,
                        "unsupported_claims": ["Refund of $10,000"],
                        "explanation": "Hallucinated refund amount.",
                    }
                )
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    res = evaluate_reply_reference_free(generated, rag_context, mock_client)

    assert res["guardrail_pass"] is True  # standard guardrails pass
    assert res["faithfulness_score"] == 0.0
    assert res["passed"] is False  # fails because faithfulness < 0.70
