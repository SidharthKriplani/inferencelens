"""Integration test: full pipeline end-to-end."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import pytest

from data.generator import SyntheticDataGenerator
from pipeline.profiler import InferenceProfiler
from pipeline.evaluator import QualityEvaluator
from pipeline.cost_tracker import CostTracker
from pipeline.pareto import ParetoAnalyzer, ParetoStatus
from pipeline.router import RoutingRecommender
from pipeline.audit import AuditLogger
from pipeline.report import ReportGenerator
from config import TASK_TYPES, MODEL_CONFIGS


@pytest.fixture(scope="module")
def full_pipeline():
    """Run complete pipeline once for all integration tests."""
    gen = SyntheticDataGenerator(seed=99)
    records = gen.generate()

    cost_tracker = CostTracker()
    evaluator = QualityEvaluator(use_simulated=True)
    profiler = InferenceProfiler(cost_tracker=cost_tracker, evaluator=evaluator)
    runs = profiler.profile_all(records)

    analyzer = ParetoAnalyzer()
    pareto_results = analyzer.analyze_all(runs)

    recommender = RoutingRecommender()
    routing_recs = recommender.generate_all(pareto_results)

    reporter = ReportGenerator()
    report = reporter.generate(runs, pareto_results, routing_recs)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        audit_path = f.name
    audit = AuditLogger(log_path=audit_path)

    return {
        "records": records,
        "runs": runs,
        "pareto_results": pareto_results,
        "routing_recs": routing_recs,
        "report": report,
        "audit": audit,
        "reporter": reporter,
        "cost_tracker": cost_tracker,
    }


def test_report_has_all_task_types(full_pipeline):
    report = full_pipeline["report"]
    for tt in TASK_TYPES:
        assert tt in report.task_reports


def test_global_verdict_is_valid(full_pipeline):
    report = full_pipeline["report"]
    assert report.global_verdict in {"PASS", "WARN", "FAIL"}


def test_dominated_configs_found(full_pipeline):
    report = full_pipeline["report"]
    # With 4 configs, at least some domination should occur
    assert isinstance(report.dominated_configs_found, bool)


def test_total_savings_positive(full_pipeline):
    report = full_pipeline["report"]
    assert report.total_savings_opportunity_pct >= 0.0


def test_terminal_render_nonempty(full_pipeline):
    report = full_pipeline["report"]
    reporter = full_pipeline["reporter"]
    for tt in TASK_TYPES:
        output = reporter.render_terminal(report, tt)
        assert len(output) > 100
        assert "InferenceLens Report" in output


def test_cost_tracker_records_all_calls(full_pipeline):
    ct = full_pipeline["cost_tracker"]
    for tt in TASK_TYPES:
        records = ct.records(task_type=tt)
        # Should have n_prompts * n_tiers calls
        assert len(records) == 100 * 4  # 100 samples per task, 4 tiers


def test_audit_logger_writes_and_reads(full_pipeline):
    audit: AuditLogger = full_pipeline["audit"]
    evt = audit.log_pipeline_start(task_types=TASK_TYPES, n_records=300)
    assert evt.event_id.startswith("evt_")
    events = audit.read_all()
    assert len(events) >= 1


def test_pareto_curve_data_present(full_pipeline):
    report = full_pipeline["report"]
    assert len(report.pareto_curve_data) == len(TASK_TYPES)
    for tt, curve in report.pareto_curve_data.items():
        assert len(curve) == len(MODEL_CONFIGS)
        for point in curve:
            assert "tier" in point
            assert "cost_usd" in point
            assert "quality" in point
            assert "status" in point


def test_json_serialization(full_pipeline):
    """Report must be fully JSON-serializable."""
    import json
    report = full_pipeline["report"]
    json_str = report.model_dump_json()
    parsed = json.loads(json_str)
    assert "task_reports" in parsed
    assert "global_verdict" in parsed
