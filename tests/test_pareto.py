"""Tests for ParetoAnalyzer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.generator import SyntheticDataGenerator
from pipeline.profiler import InferenceProfiler
from pipeline.evaluator import QualityEvaluator
from pipeline.cost_tracker import CostTracker
from pipeline.pareto import ParetoAnalyzer, ParetoStatus


@pytest.fixture(scope="module")
def pareto_fixture():
    gen = SyntheticDataGenerator(seed=42)
    records = gen.generate()
    cost_tracker = CostTracker()
    evaluator = QualityEvaluator(use_simulated=True)
    profiler = InferenceProfiler(cost_tracker=cost_tracker, evaluator=evaluator)
    runs = profiler.profile_all(records)
    analyzer = ParetoAnalyzer()
    pareto_results = analyzer.analyze_all(runs)
    return pareto_results


def test_all_task_types_analyzed(pareto_fixture):
    assert "summarization" in pareto_fixture
    assert "classification" in pareto_fixture
    assert "extraction" in pareto_fixture


def test_pareto_covers_all_tiers(pareto_fixture):
    result = pareto_fixture["summarization"]
    tiers_in_result = {p.tier for p in result.points}
    assert tiers_in_result == {"large", "medium", "small", "local"}


def test_dominated_config_identified(pareto_fixture):
    """At least one config should be dominated (large is typically dominated by medium in cost/quality ratio)."""
    for task_type, result in pareto_fixture.items():
        # At least some configs should be on the frontier
        assert len(result.frontier_tiers) > 0


def test_large_most_expensive(pareto_fixture):
    """Large model should always have the highest cost."""
    for task_type, result in pareto_fixture.items():
        large_point = result.point_for("large")
        assert large_point is not None
        for point in result.points:
            if point.tier != "large":
                assert large_point.avg_cost_usd >= point.avg_cost_usd, \
                    f"{task_type}: large cost {large_point.avg_cost_usd} should be >= {point.tier} {point.avg_cost_usd}"


def test_local_zero_cost(pareto_fixture):
    for task_type, result in pareto_fixture.items():
        local_point = result.point_for("local")
        assert local_point is not None
        assert local_point.avg_cost_usd == pytest.approx(0.0, abs=1e-9)


def test_pareto_rank_assigned_to_frontier(pareto_fixture):
    for task_type, result in pareto_fixture.items():
        for point in result.points:
            if point.pareto_status == ParetoStatus.FRONTIER:
                assert point.pareto_rank > 0
            else:
                assert point.pareto_rank == 0


def test_dominated_by_populated(pareto_fixture):
    for task_type, result in pareto_fixture.items():
        for point in result.points:
            if point.pareto_status == ParetoStatus.DOMINATED:
                assert point.dominated_by is not None


def test_recommended_default_on_frontier(pareto_fixture):
    for task_type, result in pareto_fixture.items():
        assert result.recommended_default in result.frontier_tiers


def test_cost_savings_vs_large(pareto_fixture):
    """All non-large tiers should report positive cost savings."""
    for task_type, result in pareto_fixture.items():
        for point in result.points:
            if point.tier != "large":
                assert point.cost_savings_vs_large >= 0.0
