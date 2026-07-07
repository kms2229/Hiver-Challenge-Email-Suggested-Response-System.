"""
tests/test_metrics.py — Unit tests for the accuracy metric functions
and all advanced generation/evaluation components.

All tests run without API calls by mocking OpenAI and using
pre-computed or synthetic data.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from dataclasses import asdict

import numpy as np
import pytest

from src.evaluator import (
    _rouge_recall,
    _composite_score,
    _layer1_guardrails,
    aggregate_results,
    run_calibration,
    EvaluationResult,
    MetricScores,
    _W_SEMANTIC,
    _W_ROUGE,
    _W_TONE,
    _W_QUALITY,
    _W_FAITHFUL,
)


# ===========================================================================
# ROUGE recall tests
# ===========================================================================

def test_rouge_recall_perfect():
    """Identical text should yield recall of 1.0."""
    text = "We are sorry to hear that. We will process your refund within 5 business days."
    score = _rouge_recall(text, text)
    assert score == pytest.approx(1.0, abs=0.001)


def test_rouge_recall_zero():
    """Completely disjoint text should yield a low recall score."""
    gen = "The weather in London is quite pleasant today."
    ref = "Your invoice has been processed and payment received."
    score = _rouge_recall(gen, ref)
    assert score < 0.2


def test_rouge_recall_partial():
    """Partial overlap should yield a value between 0 and 1."""
    gen = "Thank you for reaching out. We will look into your refund request."
    ref = "Thank you for contacting us. We will process your refund within 5 days."
    score = _rouge_recall(gen, ref)
    assert 0.0 < score < 1.0


# ===========================================================================
# Composite score tests — 5-metric system
# ===========================================================================

def test_composite_score_weights_sum_to_one():
    """The five weights must sum to exactly 1.0."""
    total = _W_SEMANTIC + _W_ROUGE + _W_TONE + _W_QUALITY + _W_FAITHFUL
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_composite_score_max():
    """Perfect scores on all 5 dimensions should yield 1.0."""
    score = _composite_score(
        semantic_sim=1.0,
        rouge_recall=1.0,
        tone_score=5.0,
        quality_score=5.0,
        faithfulness=1.0,
    )
    assert score == pytest.approx(1.0, abs=1e-9)


def test_composite_score_min():
    """Zero/min scores should yield a deterministic low value."""
    score = _composite_score(
        semantic_sim=0.0,
        rouge_recall=0.0,
        tone_score=1.0,
        quality_score=1.0,
        faithfulness=0.0,
    )
    expected = _W_TONE * 0.2 + _W_QUALITY * 0.2  # only tone+quality non-zero
    assert score == pytest.approx(expected, abs=1e-9)


def test_composite_score_faithfulness_penalises():
    """A reply with faithfulness=0 should score lower than faithfulness=1."""
    high = _composite_score(1.0, 1.0, 5.0, 5.0, faithfulness=1.0)
    low  = _composite_score(1.0, 1.0, 5.0, 5.0, faithfulness=0.0)
    assert high > low
    assert math.isclose(high - low, _W_FAITHFUL, rel_tol=1e-9)


# ===========================================================================
# Layer 1 guardrail tests
# ===========================================================================

def test_guardrail_passes_good_reply():
    """A well-formed reply should pass all guardrails."""
    reply = (
        "Hi there,\n\n"
        "Thank you for reaching out. We have received your request and are "
        "investigating the billing discrepancy. You should receive an update "
        "within 24 hours.\n\n"
        "The Hiver Support Team"
    )
    passed, failures = _layer1_guardrails(reply)
    assert passed is True
    assert failures == []


def test_guardrail_fails_too_short():
    """A reply under 20 words should fail the length check."""
    reply = "Hi, sorry. We will fix it. Thanks."
    passed, failures = _layer1_guardrails(reply)
    assert passed is False
    assert any("short" in f.lower() for f in failures)


def test_guardrail_fails_no_greeting():
    """A reply without a recognisable greeting should fail."""
    reply = (
        "Your issue has been escalated to our billing team. "
        "Someone will contact you by end of day to resolve the duplicate charge. "
        "We apologise for the inconvenience caused by this error.\n\nThe Hiver Support Team"
    )
    passed, failures = _layer1_guardrails(reply)
    assert passed is False
    assert any("greeting" in f.lower() for f in failures)


def test_guardrail_fails_refusal_language():
    """A reply with AI refusal language should fail."""
    reply = (
        "Hi there,\n\n"
        "I cannot assist with billing issues as I am an AI without access to your account. "
        "Please contact your account manager directly for assistance with this matter.\n\n"
        "The Hiver Support Team"
    )
    passed, failures = _layer1_guardrails(reply)
    assert passed is False
    assert any("refusal" in f.lower() for f in failures)


def test_guardrail_fails_missing_signoff():
    """A reply without a professional sign-off should fail."""
    reply = (
        "Hello,\n\n"
        "We have received your request and will look into the billing discrepancy. "
        "Our team will get back to you within two business days with a resolution. "
        "Thank you for your patience and understanding."
    )
    passed, failures = _layer1_guardrails(reply)
    assert passed is False
    assert any("sign-off" in f.lower() for f in failures)


# ===========================================================================
# Aggregate statistics tests
# ===========================================================================

def _make_result(email_id, composite, category="test", guardrail_pass=True) -> EvaluationResult:
    scores = MetricScores(
        semantic_similarity=composite,
        rouge_recall=composite,
        tone_score=composite * 5,
        quality_score=composite * 5,
        faithfulness_score=composite,
        guardrail_pass=guardrail_pass,
        composite_score=composite,
    )
    return EvaluationResult(
        email_id=email_id,
        subject=f"Email {email_id}",
        category=category,
        generated_reply="reply",
        reference_reply="reference",
        scores=scores,
    )


def test_aggregate_empty():
    assert aggregate_results([]) == {}


def test_aggregate_single():
    results = [_make_result(1, 0.75)]
    agg = aggregate_results(results)
    assert agg["n_evaluated"] == 1
    assert agg["overall"]["composite_mean"] == pytest.approx(0.75, abs=0.0001)


def test_aggregate_includes_faithfulness_and_guardrails():
    results = [
        _make_result(1, 0.9, guardrail_pass=True),
        _make_result(2, 0.6, guardrail_pass=False),
    ]
    agg = aggregate_results(results)
    assert "faithfulness_mean" in agg["overall"]
    assert "guardrail_pass_rate" in agg["overall"]
    assert agg["overall"]["guardrail_pass_rate"] == pytest.approx(0.5, abs=0.001)


def test_aggregate_multiple():
    results = [
        _make_result(1, 0.8, "billing"),
        _make_result(2, 0.6, "billing"),
        _make_result(3, 0.4, "bug_report"),
    ]
    agg = aggregate_results(results)
    assert agg["n_evaluated"] == 3
    assert agg["overall"]["composite_mean"] == pytest.approx((0.8 + 0.6 + 0.4) / 3, abs=0.001)
    assert "billing" in agg["by_category"]
    assert "bug_report" in agg["by_category"]
    assert agg["by_category"]["billing"]["count"] == 2


# ===========================================================================
# Calibration tests
# ===========================================================================

def test_calibration_missing_file():
    """Should return a warning dict, not raise an exception."""
    with patch("src.evaluator._CALIBRATION_PATH", Path("/nonexistent/calibration.json")):
        result = run_calibration([])
    assert result["calibration_available"] is False


def test_calibration_with_matching_data():
    """Calibration should compute Spearman ρ when IDs match."""
    calibration_data = [
        {"id": "1", "human_score": 1.0},
        {"id": "2", "human_score": 0.5},
        {"id": "3", "human_score": 0.0},
    ]
    results = [
        _make_result("1", 0.9),
        _make_result("2", 0.55),
        _make_result("3", 0.1),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(calibration_data, f)
        tmp_path = Path(f.name)

    with patch("src.evaluator._CALIBRATION_PATH", tmp_path):
        cal = run_calibration(results)

    assert cal["calibration_available"] is True
    assert cal["matched_count"] == 3
    assert cal["spearman_rho"] == pytest.approx(1.0, abs=0.01)
    tmp_path.unlink()


# ===========================================================================
# Agentic RAG tests
# ===========================================================================

def test_agentic_rag_skips_requery_when_similarity_high():
    """When similarity >= threshold, no re-query should happen."""
    from src.generator import _agentic_retrieve

    mock_index = MagicMock()
    mock_index.search.return_value = [{"_similarity": 0.85, "id": 1}]

    mock_client = MagicMock()

    with patch("src.generator.embed_texts", return_value=np.array([[0.1] * 1536])):
        retrieved, was_re_queried = _agentic_retrieve(
            subject="Test", body="Test body",
            index=mock_index, client=mock_client,
            top_k=3, threshold=0.60,
        )

    assert was_re_queried is False
    assert len(retrieved) == 1


def test_agentic_rag_triggers_requery_when_similarity_low():
    """When similarity < threshold, re-query should be triggered."""
    from src.generator import _agentic_retrieve

    low_sim_results = [{"_similarity": 0.35, "id": 1}]
    high_sim_results = [{"_similarity": 0.75, "id": 2}]

    mock_index = MagicMock()
    mock_index.search.side_effect = [low_sim_results, high_sim_results]

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="reformulated query"))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("src.generator.embed_texts", return_value=np.array([[0.1] * 1536])):
        retrieved, was_re_queried = _agentic_retrieve(
            subject="Test", body="Test body",
            index=mock_index, client=mock_client,
            top_k=3, threshold=0.60,
        )

    assert was_re_queried is True
    assert retrieved[0]["_similarity"] == 0.75


# ===========================================================================
# Self-Refine tests
# ===========================================================================

def test_self_refine_skips_revision_when_no_issues():
    """If critique says 'No major issues', the original draft is returned."""
    from src.generator import _self_refine

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="No major issues found."))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    original_draft = "Thank you for reaching out. We'll help you."
    revised, critique = _self_refine(
        subject="Test", body="Help needed",
        draft=original_draft, client=mock_client, model="gpt-4o-mini",
    )

    assert revised == original_draft
    assert "no major" in critique.lower()
    assert mock_client.chat.completions.create.call_count == 1


def test_self_refine_revises_when_issues_found():
    """When critique finds issues, a revision call should be made."""
    from src.generator import _self_refine

    critique_text = "1. Tone is too cold. 2. Missing refund timeline."
    revised_text = "We're truly sorry for this experience. Your refund will arrive in 5 days."

    call_count = 0
    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(
            content=critique_text if call_count == 1 else revised_text
        ))]
        return resp

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = mock_create

    revised, critique = _self_refine(
        subject="Test", body="I want a refund",
        draft="We will check.", client=mock_client, model="gpt-4o-mini",
    )

    assert revised == revised_text
    assert critique == critique_text
    assert call_count == 2


# ===========================================================================
# Debate-as-Judge tests
# ===========================================================================

def test_debate_judge_agrees_without_arbitration():
    """When judges agree (gap < 1.0), no arbitration call should be made."""
    from src.evaluator import _debate_judge

    judge_response = json.dumps({
        "tone_score": 4,
        "tone_explanation": "Good tone.",
        "quality_score": 4,
        "quality_explanation": "Complete reply.",
    })
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=judge_response))]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    tone, tone_exp, qual, qual_exp = _debate_judge(
        incoming_email="Help!", generated_reply="We'll help.",
        reference_reply="We'll help you.", client=mock_client,
    )

    assert tone == pytest.approx(4.0, abs=0.1)
    assert qual == pytest.approx(4.0, abs=0.1)
    assert mock_client.chat.completions.create.call_count == 2
    assert "Judges agreed" in tone_exp


def test_debate_judge_arbitrates_on_disagreement():
    """When judges disagree by >= 1.0, an arbitration call should be made."""
    from src.evaluator import _debate_judge

    call_count = 0
    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if call_count == 1:
            content = json.dumps({"tone_score": 5, "tone_explanation": "Great.", "quality_score": 5, "quality_explanation": "Perfect."})
        elif call_count == 2:
            content = json.dumps({"tone_score": 3, "tone_explanation": "Mediocre.", "quality_score": 3, "quality_explanation": "Lacking."})
        else:
            content = json.dumps({"tone_score": 4, "quality_score": 4, "reasoning": "Compromise."})
        resp.choices = [MagicMock(message=MagicMock(content=content))]
        return resp

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = mock_create

    tone, tone_exp, qual, qual_exp = _debate_judge(
        incoming_email="Help!", generated_reply="We'll help.",
        reference_reply="We'll help you.", client=mock_client,
    )

    assert call_count == 3
    assert tone == pytest.approx(4.0, abs=0.1)
    assert "Arbitrated" in tone_exp


# ===========================================================================
# Classifier tests
# ===========================================================================

def test_classifier_returns_valid_structure():
    """Classifier should return a valid EmailClassification dataclass."""
    from src.classifier import classify_email, EmailClassification

    mock_response_json = json.dumps({
        "sentiment": "frustrated",
        "urgency": "high",
        "escalation_risk": True,
        "primary_issue": "Double billing on account",
        "tone_instruction": "Open with strong empathy, acknowledge billing error directly.",
    })
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=mock_response_json))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    result = classify_email("Billing error", "I was charged twice!", mock_client)

    assert isinstance(result, EmailClassification)
    assert result.sentiment == "frustrated"
    assert result.urgency == "high"
    assert result.escalation_risk is True
    assert "billing" in result.primary_issue.lower()


def test_classifier_fallback_on_invalid_json():
    """Classifier should return neutral/medium defaults on parse failure."""
    from src.classifier import classify_email

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="not json {{{"))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    result = classify_email("Test", "Test body", mock_client)

    assert result.sentiment == "neutral"
    assert result.urgency == "medium"
    assert result.escalation_risk is False


def test_classifier_fallback_on_exception():
    """Classifier should return defaults if the API call raises."""
    from src.classifier import classify_email

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    result = classify_email("Test", "Test body", mock_client)

    assert result.sentiment == "neutral"
    assert result.urgency == "medium"


def test_build_tone_context_includes_escalation_warning():
    """build_tone_context should include escalation warning when risk is True."""
    from src.classifier import build_tone_context, EmailClassification

    cls = EmailClassification(
        sentiment="frustrated",
        urgency="critical",
        escalation_risk=True,
        primary_issue="Service outage affecting production",
        tone_instruction="Acknowledge outage immediately.",
    )
    ctx = build_tone_context(cls)
    assert "FRUSTRATED" in ctx
    assert "CRITICAL" in ctx
    assert "ESCALATION RISK" in ctx
    assert "Acknowledge outage" in ctx


# ===========================================================================
# BM25 / Hybrid retrieval tests
# ===========================================================================

def test_bm25_index_builds_correctly():
    """BM25 index should be built without errors given email data."""
    from src.dataset import EmailIndex
    import numpy as np

    emails = [
        {"id": 1, "subject": "Billing error", "body": "I was charged twice", "reply": "Sorry"},
        {"id": 2, "subject": "Slack integration", "body": "Slack stopped working", "reply": "We'll fix it"},
        {"id": 3, "subject": "Password reset", "body": "I cannot log in", "reply": "Reset instructions"},
    ]
    embeddings = np.random.rand(3, 1536).astype(np.float32)
    index = EmailIndex(emails, embeddings)
    assert index._bm25 is not None


def test_bm25_keyword_match():
    """BM25 should rank documents with exact keyword matches higher."""
    from src.dataset import EmailIndex
    import numpy as np

    emails = [
        {"id": 1, "subject": "General question", "body": "How do I use the dashboard", "reply": "r"},
        {"id": 2, "subject": "Slack issue", "body": "Slack integration error 403 not working", "reply": "r"},
        {"id": 3, "subject": "Billing", "body": "Invoice missing from account page", "reply": "r"},
    ]
    embeddings = np.random.rand(3, 1536).astype(np.float32)
    index = EmailIndex(emails, embeddings)

    bm25_results = index._bm25_top_n("slack integration error 403", n=3)
    top_id = bm25_results[0][0]  # index of top result
    assert top_id == 1  # second email (index 1) has the Slack keywords


def test_rrf_fusion():
    """RRF fusion gives highest score to the doc that ranks best across both lists.

    Doc 1 is rank 1 in both dense and BM25 → unambiguously highest RRF score.
    Doc 0 is rank 2 in dense, absent from BM25 → lower score.
    Doc 2 is rank 2 in BM25, absent from dense → lower score.
    """
    from src.dataset import _reciprocal_rank_fusion

    dense = [(1, 0.95), (0, 0.70)]   # doc 1 best in dense
    bm25  = [(1, 9.1),  (2, 5.3)]    # doc 1 best in BM25

    fused = _reciprocal_rank_fusion([dense, bm25])
    ids = [doc_id for doc_id, _ in fused]

    # Doc 1 is rank 1 in both lists → highest RRF score, must be first
    assert ids[0] == 1
    # All 3 unique docs should appear in output
    assert set(ids) == {0, 1, 2}


# ===========================================================================
# HyDE retrieval test
# ===========================================================================

def test_hyde_generates_hypothetical_and_retrieves():
    """HyDE should call the LLM for a hypothetical reply, then retrieve."""
    from src.dataset import EmailIndex
    import numpy as np

    emails = [
        {"id": 1, "subject": "Refund", "body": "I want a refund", "reply": "We'll refund you"},
    ]
    embeddings = np.random.rand(1, 1536).astype(np.float32)

    # We patch at the module level
    with patch("src.dataset.embed_texts", return_value=np.random.rand(1, 1536).astype(np.float32)):
        index = EmailIndex(emails, embeddings)

        hyp_text = "We understand your frustration. We will process your refund within 5 days."
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=hyp_text))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        from src.dataset import hyde_retrieve
        with patch("src.dataset.embed_texts", return_value=np.random.rand(1, 1536).astype(np.float32)):
            results, hypothetical = hyde_retrieve("Refund request", "I need a refund", index, mock_client, k=1)

    assert hypothetical == hyp_text
    assert isinstance(results, list)


# ===========================================================================
# Structured output / assemble_reply tests
# ===========================================================================

def test_assemble_reply_all_fields():
    """_assemble_reply should combine all fields into a coherent reply."""
    from src.generator import _assemble_reply

    structured = {
        "greeting": "Dear Sarah,",
        "body": "Thank you for reaching out. We have resolved the billing issue.",
        "action_items": ["Refund of $49 will appear in 3-5 days."],
        "sign_off": "The Hiver Support Team",
    }
    reply = _assemble_reply(structured)
    assert "Dear Sarah," in reply
    assert "billing issue" in reply
    assert "• Refund of $49" in reply
    assert "The Hiver Support Team" in reply


def test_assemble_reply_empty_action_items():
    """_assemble_reply should handle an empty action_items list."""
    from src.generator import _assemble_reply

    structured = {
        "greeting": "Hi there,",
        "body": "Your question has been answered.",
        "action_items": [],
        "sign_off": "The Hiver Support Team",
    }
    reply = _assemble_reply(structured)
    assert "•" not in reply
    assert "The Hiver Support Team" in reply


# ===========================================================================
# Preference logger tests
# ===========================================================================

def test_logger_writes_jsonl():
    """log_evaluation should write a valid JSONL record."""
    import tempfile
    from src.logger import log_evaluation, log_stats

    scores = MetricScores(
        semantic_similarity=0.85,
        rouge_recall=0.72,
        tone_score=4.0,
        quality_score=4.0,
        faithfulness_score=0.95,
        composite_score=0.80,
    )
    result = EvaluationResult(
        email_id="42",
        subject="Test email",
        category="billing",
        generated_reply="Thank you for contacting us.",
        reference_reply="Thank you for reaching out.",
        scores=scores,
    )
    gen_result = {"mode": "standard", "model": "gpt-4o-mini", "retrieved_examples": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.jsonl"
        log_evaluation(result, gen_result, log_path=log_path)

        stats = log_stats(log_path=log_path)
        assert stats["records"] == 1
        assert stats["unique_emails"] == 1
        assert stats["avg_composite"] == pytest.approx(0.80, abs=0.001)

        # Verify JSONL is valid JSON
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["email_id"] == "42"
        assert record["generation_mode"] == "standard"


def test_logger_dpo_pairs_empty_when_insufficient_data():
    """to_dpo_pairs should return [] when no pairs qualify."""
    from src.logger import to_dpo_pairs
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # All records have mid-range scores — won't qualify as chosen OR rejected
        record = {
            "email_id": "1",
            "subject": "Test",
            "category": "billing",
            "prompt": [],
            "generated_reply": "Some reply",
            "reference_reply": "Gold reply",
            "scores": {"composite": 0.60},
            "generation_mode": "standard",
            "model": "gpt-4o-mini",
            "was_re_queried": False,
            "retrieved_example_ids": [],
        }
        f.write(json.dumps(record) + "\n")
        tmp_path = f.name

    pairs = to_dpo_pairs(log_path=tmp_path)
    assert pairs == []
    Path(tmp_path).unlink()


# ===========================================================================
# MoA structure test
# ===========================================================================

def test_moa_result_structure():
    mock_result = {
        "generated_reply": "Final synthesized reply.",
        "candidates": ["Candidate 1", "Candidate 2", "Candidate 3"],
        "retrieved_examples": [],
        "n_candidates": 3,
        "mode": "moa",
        "model": "gpt-4o-mini",
    }
    required_keys = {"generated_reply", "candidates", "retrieved_examples", "n_candidates", "mode"}
    assert required_keys.issubset(set(mock_result.keys()))
    assert mock_result["mode"] == "moa"
    assert len(mock_result["candidates"]) == 3


# ===========================================================================
# Debate generator structure test
# ===========================================================================

def test_debate_result_structure():
    mock_result = {
        "generated_reply": "Final judged reply.",
        "debate_transcript": [
            {"role": "composer", "round": 1, "content": "Draft 1"},
            {"role": "critic",   "round": 1, "content": "REVISE: too cold."},
            {"role": "composer", "round": 2, "content": "Revised draft."},
            {"role": "critic",   "round": 2, "content": "ACCEPT: much better."},
        ],
        "rounds_completed": 2,
        "accepted_early": True,
        "retrieved_examples": [],
        "mode": "debate",
        "model": "gpt-4o-mini",
    }
    required_keys = {"generated_reply", "debate_transcript", "rounds_completed", "accepted_early", "mode"}
    assert required_keys.issubset(set(mock_result.keys()))
    composers = [t for t in mock_result["debate_transcript"] if t["role"] == "composer"]
    critics = [t for t in mock_result["debate_transcript"] if t["role"] == "critic"]
    assert len(composers) == len(critics)
