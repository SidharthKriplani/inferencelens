"""
InferenceLens — FastAPI Application
Endpoints:
  GET  /health
  GET  /configs
  POST /profile
  GET  /report/{task_type}
  GET  /audit-log
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import MODEL_CONFIGS, TASK_TYPES
from data.generator import SyntheticDataGenerator
from pipeline.profiler import InferenceProfiler
from pipeline.evaluator import QualityEvaluator
from pipeline.cost_tracker import CostTracker
from pipeline.pareto import ParetoAnalyzer
from pipeline.router import RoutingRecommender
from pipeline.audit import AuditLogger, AuditEventType
from pipeline.report import ReportGenerator, InferenceLensReport


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="InferenceLens",
    description="Inference Cost/Quality Tradeoff Auditor API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Singleton state (loaded once on startup)
_state: Dict[str, Any] = {}


def _get_or_build_report() -> InferenceLensReport:
    if "report" not in _state:
        gen = SyntheticDataGenerator()
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
        _state["report"] = report
        _state["runs"] = runs
        _state["pareto_results"] = pareto_results
        _state["routing_recs"] = routing_recs
        _state["reporter"] = reporter
    return _state["report"]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ProfileRequest(BaseModel):
    task_types: Optional[List[str]] = None     # default: all
    n_samples: Optional[int] = 10              # samples per task type
    tiers: Optional[List[str]] = None          # default: all 4


class ProfileResponse(BaseModel):
    run_ids: Dict[str, str]
    total_calls: int
    cost_summary: Dict[str, Any]
    pareto_summary: Dict[str, Any]
    routing_rules: Dict[str, List[str]]
    global_verdict: str
    duration_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health_check():
    return {
        "status": "ok",
        "service": "InferenceLens",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@app.get("/configs", tags=["Meta"])
def list_configs():
    """Return available model configurations and cost table."""
    return {
        "model_configs": {
            tier: {
                "model_id": cfg.model_id,
                "display_name": cfg.display_name,
                "input_cost_per_1k_usd": cfg.input_cost_per_1k,
                "output_cost_per_1k_usd": cfg.output_cost_per_1k,
                "avg_latency_ms": cfg.avg_latency_ms,
                "tier": cfg.tier,
            }
            for tier, cfg in MODEL_CONFIGS.items()
        },
        "task_types": TASK_TYPES,
    }


@app.post("/profile", response_model=ProfileResponse, tags=["Profiling"])
def profile(request: ProfileRequest):
    """
    Run inference profiling across model configurations.
    Returns Pareto analysis, routing rules, and cost summary.
    """
    t0 = time.time()
    task_types = request.task_types or TASK_TYPES
    n_samples = min(request.n_samples or 10, 100)
    tiers = request.tiers or list(MODEL_CONFIGS.keys())

    # Validate
    for tt in task_types:
        if tt not in TASK_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid task_type: {tt!r}")
    for tier in tiers:
        if tier not in MODEL_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {tier!r}")

    # Generate subset
    from data.generator import SyntheticDataGenerator
    gen = SyntheticDataGenerator()
    all_records = gen.generate()
    records = [r for r in all_records if r.task_type in task_types]
    # Sample evenly
    sampled = []
    for tt in task_types:
        tt_records = [r for r in records if r.task_type == tt][:n_samples]
        sampled.extend(tt_records)

    cost_tracker = CostTracker()
    evaluator = QualityEvaluator(use_simulated=True)
    profiler = InferenceProfiler(cost_tracker=cost_tracker, evaluator=evaluator, tiers=tiers)
    runs = {tt: profiler.profile_task(sampled, tt) for tt in task_types}

    analyzer = ParetoAnalyzer()
    pareto_results = analyzer.analyze_all(runs)

    recommender = RoutingRecommender()
    routing_recs = recommender.generate_all(pareto_results)

    reporter = ReportGenerator()
    report = reporter.generate(runs, pareto_results, routing_recs)

    # Flatten routing rules for response
    routing_rules: Dict[str, List[str]] = {}
    for tt, rec in routing_recs.items():
        routing_rules[tt] = [r.human_readable for r in rec.rules]

    # Cost summary
    cost_summary: Dict[str, Any] = {}
    for tt in task_types:
        summaries = cost_tracker.summarize(task_type=tt)
        cost_summary[tt] = [
            {
                "tier": s.tier,
                "avg_cost_usd": s.avg_cost_usd,
                "avg_latency_ms": s.avg_latency_ms,
                "n_calls": s.n_calls,
            }
            for s in summaries
        ]

    # Pareto summary
    pareto_summary: Dict[str, Any] = {}
    for tt, pr in pareto_results.items():
        pareto_summary[tt] = {
            "frontier": pr.frontier_tiers,
            "dominated": pr.dominated_tiers,
            "recommended_default": pr.recommended_default,
            "points": [
                {
                    "tier": p.tier,
                    "avg_cost_usd": p.avg_cost_usd,
                    "avg_quality": p.avg_quality,
                    "pareto_status": p.pareto_status.value,
                    "cost_savings_pct": round(p.cost_savings_vs_large * 100, 1),
                    "quality_delta": p.quality_delta_vs_large,
                }
                for p in pr.points
            ],
        }

    duration_ms = (time.time() - t0) * 1000
    return ProfileResponse(
        run_ids={tt: runs[tt].run_id for tt in task_types},
        total_calls=sum(len(runs[tt].calls) for tt in task_types),
        cost_summary=cost_summary,
        pareto_summary=pareto_summary,
        routing_rules=routing_rules,
        global_verdict=report.global_verdict,
        duration_ms=round(duration_ms, 2),
    )


@app.get("/report/{task_type}", tags=["Reports"])
def get_report(task_type: str):
    """
    Get full report for a specific task type.
    Runs the full pipeline if not already cached.
    """
    if task_type not in TASK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_type: {task_type!r}. Valid: {TASK_TYPES}",
        )
    report = _get_or_build_report()
    tr = report.task_reports.get(task_type)
    if tr is None:
        raise HTTPException(status_code=404, detail=f"No report for task_type={task_type!r}")

    # Also render terminal table
    reporter: ReportGenerator = _state["reporter"]
    terminal_output = reporter.render_terminal(report, task_type)

    return {
        "task_type": task_type,
        "overall_verdict": tr.overall_verdict,
        "n_dominated_configs": tr.n_dominated_configs,
        "savings_opportunity_usd": tr.savings_opportunity_usd,
        "summary": tr.summary_lines,
        "pareto_points": [
            {
                "tier": p.tier,
                "model_id": p.model_id,
                "avg_cost_usd": p.avg_cost_usd,
                "avg_quality": p.avg_quality,
                "avg_latency_ms": p.avg_latency_ms,
                "pareto_status": p.pareto_status.value,
                "cost_savings_vs_large_pct": round(p.cost_savings_vs_large * 100, 1),
                "quality_delta_vs_large": p.quality_delta_vs_large,
                "routing_efficiency": p.routing_efficiency,
                "pareto_rank": p.pareto_rank,
                "dominated_by": p.dominated_by,
            }
            for p in tr.pareto_result.points
        ],
        "routing_rules": [
            {
                "rule_id": r.rule_id,
                "complexity_band": r.condition.complexity_band.value,
                "recommended_tier": r.recommended_tier,
                "cost_savings_pct": r.cost_savings_pct,
                "quality_delta": r.quality_delta,
                "verdict": r.verdict,
                "human_readable": r.human_readable,
            }
            for r in tr.routing_recommendation.rules
        ],
        "terminal_table": terminal_output,
        "global_report_id": report.report_id,
        "generated_at": report.generated_at,
    }


@app.get("/audit-log", tags=["Audit"])
def get_audit_log(
    limit: int = Query(default=50, le=500),
    event_type: Optional[str] = Query(default=None),
    task_type: Optional[str] = Query(default=None),
    tier: Optional[str] = Query(default=None),
    verdict: Optional[str] = Query(default=None),
):
    """
    Query the JSONL audit log with optional filters.
    """
    audit = AuditLogger()
    et = None
    if event_type:
        try:
            et = AuditEventType(event_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event_type: {event_type!r}. Valid: {[e.value for e in AuditEventType]}",
            )

    events = audit.filter(
        event_type=et,
        task_type=task_type,
        tier=tier,
        verdict=verdict,
        limit=limit,
    )
    stats = audit.stats()

    return {
        "total_in_log": stats["total_events"],
        "returned": len(events),
        "filters": {
            "event_type": event_type,
            "task_type": task_type,
            "tier": tier,
            "verdict": verdict,
        },
        "stats": stats,
        "events": [e.model_dump() for e in events],
    }


@app.get("/", tags=["Meta"])
def root():
    return {
        "service": "InferenceLens",
        "tagline": "An auditor for AI inference routing decisions — makes the cost/quality tradeoff explicit, measurable, and defensible.",
        "endpoints": ["/health", "/configs", "/profile", "/report/{task_type}", "/audit-log", "/docs"],
    }
