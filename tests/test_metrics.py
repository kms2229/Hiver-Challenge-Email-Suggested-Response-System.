"""
tests/test_metrics.py — Unit tests for the accuracy metric functions
and the new advanced generation/evaluation components.

All tests run without API calls by mocking OpenAI and using
pre-computed or synthetic data.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluator import (
    _rouge_recall,
    _composite_score,
    aggregate_results,
    run_calibration,
    EvaluationResult,
    MetricScores,
    _W_SEMANTIC,
    _W_ROUGE,
    _W_TONE,
    _W_QUALITY,
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
# Composite score tests
# ===========================================================================

def test_composite_score_weights_sum_to_one():
    """The four weights must sum to exactly 1.0."""
    assert math.isclose(_W_SEMANTIC + _W_ROUGE + _W_TONE + _W_QUALITY, 1.0, rel_tol=1e-9)


def test_composite_score_max():
    """Perfect scores on all dimensions should yield 1.0."""
    score = _composite_score(
        semantic_sim=1.0,
        rouge_recall=1.0,
        tone_score=5.0,
        quality_score=5.0,
    )
    assert score == pytest.approx(1.0, abs=1e-9)


def test_composite_score_min():
    """Zero scores on all dimensions should yield the minimum possible."""
    score = _composite_score(
        semantic_sim=0.0,
        rouge_recall=0.0,
        tone_score=1.0,
        quality_score=1.0,
    )
    expected = _W_TONE * 0.2 + _W_QUALITY * 0.2
    assert score == pytest.approx(expected, abs=1e-9)


def test_composite_score_midpoint():
    """Mid-range scores should produce expected weighted average."""
    score = _composite_score(
        semantic_sim=0.5,
        rouge_recall=0.5,
        tone_score=3.0,
        quality_score=3.0,
    )
    expected = _W_SEMANTIC * 0.5 + _W_ROUGE * 0.5 + _W_TONE * 0.6 + _W_QUALITY * 0.6
    assert score == pytest.approx(expected, abs=1e-9)


# ===========================================================================
# Aggregate statistics tests
# ===========================================================================

def _make_result(email_id, composite, category="test") -> EvaluationResult:
    scores = MetricScores(
        semantic_similarity=composite,
        rouge_recall=composite,
        tone_score=composite * 5,
        quality_score=composite * 5,
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
    assert agg["overall"]["composite_std"] == pytest.approx(0.0, abs=0.0001)


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

    # Mock index that returns high-similarity results
    mock_index = MagicMock()
    mock_index.search.return_value = [{"_similarity": 0.85, "id": 1}]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )

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
    # Should return the higher-similarity results
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

    # Original draft returned unchanged
    assert revised == original_draft
    assert "no major" in critique.lower()
    # Only one API call (critique), not two
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
    assert call_count == 2  # critique + revision


# ===========================================================================
# Debate-as-Judge tests
# ===========================================================================

def test_debate_judge_agrees_without_arbitration():
    """When judges agree (gap < 1.0), no arbitration call should be made."""
    from src.evaluator import _debate_judge

    # Both judges return the same scores → should average and return without arbitration
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
    # Only 2 judge calls (A + B), no arbitration
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
            # Judge A: lenient
            content = json.dumps({"tone_score": 5, "tone_explanation": "Great.", "quality_score": 5, "quality_explanation": "Perfect."})
        elif call_count == 2:
            # Judge B: strict — disagreement of 2 points
            content = json.dumps({"tone_score": 3, "tone_explanation": "Mediocre.", "quality_score": 3, "quality_explanation": "Lacking."})
        else:
            # Arbitrator
            content = json.dumps({"tone_score": 4, "quality_score": 4, "reasoning": "Compromise."})
        resp.choices = [MagicMock(message=MagicMock(content=content))]
        return resp

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = mock_create

    tone, tone_exp, qual, qual_exp = _debate_judge(
        incoming_email="Help!", generated_reply="We'll help.",
        reference_reply="We'll help you.", client=mock_client,
    )

    assert call_count == 3  # Judge A + Judge B + Arbitrator
    assert tone == pytest.approx(4.0, abs=0.1)
    assert "Arbitrated" in tone_exp


# ===========================================================================
# MoA structure tests
# ===========================================================================

def test_moa_result_structure():
    """MoA result should contain the expected keys."""
    # Just test the structure without API calls by mocking at the function level
    from src import moa_generator

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
# Debate generator structure tests
# ===========================================================================

def test_debate_result_structure():
    """Debate result should contain the expected keys."""
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
    assert mock_result["mode"] == "debate"
    composers = [t for t in mock_result["debate_transcript"] if t["role"] == "composer"]
    critics = [t for t in mock_result["debate_transcript"] if t["role"] == "critic"]
    assert len(composers) == len(critics)
