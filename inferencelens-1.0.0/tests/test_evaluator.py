"""Tests for QualityEvaluator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pipeline.evaluator import (
    QualityEvaluator,
    _rouge1_f1,
    _macro_f1,
    _extraction_score,
    _TFIDFVectorizer,
)


class TestROUGE:
    def test_identical_strings(self):
        score = _rouge1_f1("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_empty_hypothesis(self):
        score = _rouge1_f1("", "the cat sat")
        assert score == 0.0

    def test_empty_reference(self):
        score = _rouge1_f1("some text", "")
        assert score == 0.0

    def test_partial_overlap(self):
        score = _rouge1_f1("the cat sat", "the cat walked away")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        score = _rouge1_f1("xyz abc", "the cat sat")
        assert score == 0.0


class TestMacroF1:
    def test_exact_match(self):
        assert _macro_f1("positive", "positive") == 1.0

    def test_case_insensitive(self):
        assert _macro_f1("Positive", "positive") == 1.0

    def test_mismatch(self):
        assert _macro_f1("positive", "negative") == 0.0

    def test_whitespace_stripped(self):
        assert _macro_f1("  neutral  ", "neutral") == 1.0


class TestExtractionScore:
    def test_exact_json_match(self):
        pred = '{"company": "Acme", "amount": "$5000"}'
        ref = '{"company": "Acme", "amount": "$5000"}'
        score = _extraction_score(pred, ref)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        pred = '{"company": "Acme", "amount": "$999"}'
        ref = '{"company": "Acme", "amount": "$5000"}'
        score = _extraction_score(pred, ref)
        assert 0.0 < score < 1.0

    def test_empty_prediction(self):
        ref = '{"company": "Acme"}'
        score = _extraction_score("{}", ref)
        assert score == 0.0


class TestSemanticSimilarity:
    def test_identical(self):
        sim = _TFIDFVectorizer.cosine_similarity("hello world", "hello world")
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_empty_string(self):
        sim = _TFIDFVectorizer.cosine_similarity("", "hello world")
        assert sim == 0.0

    def test_disjoint(self):
        sim = _TFIDFVectorizer.cosine_similarity("apple banana", "cat dog fish")
        assert sim == 0.0


class TestQualityEvaluator:
    def setup_method(self):
        self.evaluator = QualityEvaluator(use_simulated=False)

    def test_summarization(self):
        result = self.evaluator.evaluate(
            prompt_id="test_001",
            task_type="summarization",
            tier="large",
            predicted="The study found transformers outperform specialists.",
            ground_truth="Transformer models outperform fine-tuned specialists by 12%.",
        )
        assert 0.0 <= result.primary_score <= 1.0
        assert result.rouge1_f1 is not None

    def test_classification(self):
        result = self.evaluator.evaluate(
            prompt_id="test_002",
            task_type="classification",
            tier="medium",
            predicted="positive",
            ground_truth="positive",
        )
        assert result.primary_score > 0.5

    def test_extraction(self):
        result = self.evaluator.evaluate(
            prompt_id="test_003",
            task_type="extraction",
            tier="small",
            predicted='{"company": "Acme", "amount": "$500"}',
            ground_truth='{"company": "Acme", "amount": "$500"}',
        )
        assert result.primary_score > 0.5

    def test_simulated_mode(self):
        evaluator = QualityEvaluator(use_simulated=True)
        result = evaluator.evaluate(
            prompt_id="test_004",
            task_type="summarization",
            tier="large",
            predicted="anything",
            ground_truth="anything",
            simulated_quality=0.91,
        )
        assert result.primary_score == pytest.approx(0.91)
        assert result.notes == "simulated_quality_used"

    def test_unknown_task_fallback(self):
        result = self.evaluator.evaluate(
            prompt_id="test_005",
            task_type="unknown_task",
            tier="large",
            predicted="some output",
            ground_truth="some reference",
        )
        assert result.fallback_used is True
        assert 0.0 <= result.primary_score <= 1.0
