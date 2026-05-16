"""
InferenceLens — Demo
End-to-end demonstration of the full pipeline:
  1. Generate synthetic dataset (300 prompts × 4 configs)
  2. Profile all task types
  3. Run Pareto analysis
  4. Generate routing recommendations
  5. Log audit trail
  6. Render terminal report
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on path when running as script
sys.path.insert(0, str(Path(__file__).parent))

from config import TASK_TYPES, MODEL_CONFIGS
from data.generator import SyntheticDataGenerator, load_dataset
from pipeline.profiler import InferenceProfiler
from pipeline.evaluator import QualityEvaluator
from pipeline.cost_tracker import CostTracker
from pipeline.pareto import ParetoAnalyzer
from pipeline.router import RoutingRecommender
from pipeline.audit import AuditLogger, AuditEventType
from pipeline.report import ReportGenerator


def run_demo(verbose: bool = True) -> None:
    start = time.time()
    print("\n" + "=" * 65)
    print("  InferenceLens — Inference Cost/Quality Tradeoff Auditor")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Synthetic data
    # ------------------------------------------------------------------
    print("\n[1/6] Generating synthetic dataset...")
    gen = SyntheticDataGenerator()
    records = gen.generate()
    print(f"      {len(records)} prompts across {len(TASK_TYPES)} task types × {len(MODEL_CONFIGS)} configs")

    # ------------------------------------------------------------------
    # 2. Profiling
    # ------------------------------------------------------------------
    print("\n[2/6] Profiling inference across all configurations...")
    cost_tracker = CostTracker()
    evaluator = QualityEvaluator(use_simulated=True)
    profiler = InferenceProfiler(cost_tracker=cost_tracker, evaluator=evaluator)
    runs = profiler.profile_all(records)
    total_calls = sum(len(run.calls) for run in runs.values())
    print(f"      {total_calls} inference calls logged")

    # ------------------------------------------------------------------
    # 3. Pareto analysis
    # ------------------------------------------------------------------
    print("\n[3/6] Running Pareto frontier analysis...")
    analyzer = ParetoAnalyzer()
    pareto_results = analyzer.analyze_all(runs)
    for task_type, result in pareto_results.items():
        dominated = ", ".join(result.dominated_tiers) or "none"
        frontier = ", ".join(result.frontier_tiers)
        print(f"      {task_type:>16}: frontier=[{frontier}]  dominated=[{dominated}]")

    # ------------------------------------------------------------------
    # 4. Routing recommendations
    # ------------------------------------------------------------------
    print("\n[4/6] Generating routing rules...")
    recommender = RoutingRecommender()
    routing_recs = recommender.generate_all(pareto_results)
    for task_type, rec in routing_recs.items():
        print(f"\n      Task: {task_type}")
        for rule in rec.rules:
            print(f"        [{rule.verdict}] {rule.human_readable}")

    # ------------------------------------------------------------------
    # 5. Audit logging
    # ------------------------------------------------------------------
    print("\n[5/6] Writing audit trail...")
    audit = AuditLogger()
    audit.clear()

    audit.log_pipeline_start(task_types=TASK_TYPES, n_records=len(records))

    for task_type, run in runs.items():
        pareto = pareto_results[task_type]
        routing = routing_recs[task_type]
        rec = routing_recs[task_type]

        for point in pareto.points:
            audit.log_pareto_analysis(
                task_type=task_type,
                tier=point.tier,
                pareto_status=point.pareto_status.value,
                pareto_rank=point.pareto_rank,
                dominated_by=point.dominated_by,
                routing_efficiency=point.routing_efficiency,
            )

        # Sample 5 calls per tier for audit (not all 400)
        for tier in run.tiers_profiled:
            tier_calls = run.calls_for_tier(tier)[:5]
            p_point = pareto.point_for(tier)
            for call in tier_calls:
                audit.log_inference(
                    task_type=task_type,
                    prompt_id=call.prompt_id,
                    tier=tier,
                    model_id=call.model_id,
                    cost_usd=call.cost_usd,
                    latency_ms=call.latency_ms,
                    quality_score=call.quality_score,
                    complexity_score=call.complexity_score,
                    total_tokens=call.total_tokens,
                    pareto_status=p_point.pareto_status.value if p_point else None,
                    verdict="PASS" if p_point and p_point.pareto_status.value == "FRONTIER" else "WARN",
                )

        for rule in routing.rules:
            # Log a sample routing decision per rule
            audit.log_routing_decision(
                task_type=task_type,
                prompt_id=f"sample_{task_type}_{rule.condition.complexity_band.value}",
                complexity_score=(rule.condition.complexity_min + rule.condition.complexity_max) / 2,
                selected_tier=rule.recommended_tier,
                routing_rule_id=rule.rule_id,
                cost_savings_pct=rule.cost_savings_pct,
                quality_delta=rule.quality_delta,
                verdict=rule.verdict,
            )

    elapsed = time.time() - start
    audit.log_pipeline_end(duration_s=elapsed, n_calls=total_calls)

    stats = audit.stats()
    print(f"      {stats['total_events']} events logged → {audit.log_path}")

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    print("\n[6/6] Generating report...")
    reporter = ReportGenerator()
    report = reporter.generate(runs, pareto_results, routing_recs)
    report_path = reporter.save_json(report, Path("inferencelens_report.json"))
    print(f"      Report saved → {report_path}")
    print(f"\n  Global verdict: {report.global_verdict}")
    print(f"  Dominated configs found: {report.dominated_configs_found}")
    print(f"  Estimated avg savings via routing: {report.total_savings_opportunity_pct:.1f}%")
    if report.top_routing_rule:
        print(f"  Top routing rule: {report.top_routing_rule}")

    # ------------------------------------------------------------------
    # Terminal tables
    # ------------------------------------------------------------------
    print("\n" + reporter.render_all(report))

    # Cost summary
    print("Cost Summary by Task Type & Tier:")
    print("─" * 55)
    for task_type in TASK_TYPES:
        summaries = cost_tracker.summarize(task_type=task_type)
        print(f"\n  {task_type.capitalize()}:")
        for s in summaries:
            print(f"    {s.tier:<10} avg_cost=${s.avg_cost_usd:.5f}  avg_latency={s.avg_latency_ms:.0f}ms  calls={s.n_calls}")

    total_elapsed = time.time() - start
    print(f"\n{'=' * 65}")
    print(f"  Pipeline complete in {total_elapsed:.2f}s")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    run_demo()
