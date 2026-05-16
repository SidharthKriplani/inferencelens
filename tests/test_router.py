"""Tests for RoutingRecommender."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.generator import SyntheticDataGenerator
from pipeline.profiler import InferenceProfiler
from pipeline.evaluator import QualityEvaluator
from pipeline.cost_tracker import CostTracker
from pipeline.pareto import ParetoAnalyzer
from pipeline.router import RoutingRecommender, ComplexityBand


@pytest.fixture(scope="module")
def router_fixture():
    gen = SyntheticDataGenerator(seed=42)
    records = gen.generate()
    cost_tracker = CostTracker()
    evaluator = QualityEvaluator(use_simulated=True)
    profiler = InferenceProfiler(cost_tracker=cost_tracker, evaluator=evaluator)
    runs = profiler.profile_all(records)
    analyzer = ParetoAnalyzer()
    pareto_results = analyzer.analyze_all(runs)
    recommender = RoutingRecommender()
    routing_recs = recommender.generate_all(pareto_results)
    return recommender, routing_recs


def test_all_task_types_have_rules(router_fixture):
    _, routing_recs = router_fixture
    for task_type in ["summarization", "classification", "extraction"]:
        assert task_type in routing_recs
        assert len(routing_recs[task_type].rules) > 0


def test_three_complexity_bands_covered(router_fixture):
    _, routing_recs = router_fixture
    for task_type, rec in routing_recs.items():
        bands = {rule.condition.complexity_band for rule in rec.rules}
        assert ComplexityBand.LOW in bands
        assert ComplexityBand.MEDIUM in bands
        assert ComplexityBand.HIGH in bands


def test_rule_human_readable_format(router_fixture):
    _, routing_recs = router_fixture
    for task_type, rec in routing_recs.items():
        for rule in rec.rules:
            assert "IF task_type=" in rule.human_readable
            assert "THEN use=" in rule.human_readable
            assert "saves" in rule.human_readable


def test_routing_verdict_values(router_fixture):
    _, routing_recs = router_fixture
    valid_verdicts = {"PASS", "WARN", "FAIL"}
    for task_type, rec in routing_recs.items():
        for rule in rec.rules:
            assert rule.verdict in valid_verdicts


def test_complexity_classification(router_fixture):
    recommender, _ = router_fixture
    assert recommender.classify_complexity(0.1) == ComplexityBand.LOW
    assert recommender.classify_complexity(0.5) == ComplexityBand.MEDIUM
    assert recommender.classify_complexity(0.9) == ComplexityBand.HIGH


def test_routing_returns_valid_tier(router_fixture):
    recommender, routing_recs = router_fixture
    from config import MODEL_CONFIGS
    for task_type, rec in routing_recs.items():
        tier, rule = recommender.route(task_type, 0.2, rec)
        assert tier in MODEL_CONFIGS


def test_low_complexity_routes_cheaper(router_fixture):
    """Low complexity should route to cheaper model than high complexity."""
    _, routing_recs = router_fixture
    from config import MODEL_CONFIGS
    for task_type, rec in routing_recs.items():
        low_rule = next(r for r in rec.rules if r.condition.complexity_band == ComplexityBand.LOW)
        high_rule = next(r for r in rec.rules if r.condition.complexity_band == ComplexityBand.HIGH)
        low_cost = MODEL_CONFIGS[low_rule.recommended_tier].input_cost_per_1k
        high_cost = MODEL_CONFIGS[high_rule.recommended_tier].input_cost_per_1k
        # Low complexity should use equal or cheaper model
        assert low_cost <= high_cost + 1e-9


def test_savings_pct_non_negative(router_fixture):
    _, routing_recs = router_fixture
    for task_type, rec in routing_recs.items():
        for rule in rec.rules:
            # All models cheaper than large should have non-negative savings
            if rule.recommended_tier != "large":
                assert rule.cost_savings_pct >= 0.0
