"""
InferenceLens — Report Generator
Generates InferenceLensReport with:
  - Pareto curve data per task type
  - Routing rules with verdicts
  - Per-task-type breakdown
  - PASS / WARN / FAIL routing efficiency verdict
  - Terminal-formatted table output
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ..config import MODEL_CONFIGS
from .pareto import ParetoResult, ConfigPoint, ParetoStatus
from .router import RoutingRecommendation, RoutingRule, AuditVerdict
from .profiler import ProfileRun


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------

class TaskReport(BaseModel):
    task_type: str
    pareto_result: ParetoResult
    routing_recommendation: RoutingRecommendation
    overall_verdict: str          # PASS | WARN | FAIL
    summary_lines: List[str]      # human-readable summary bullets
    savings_opportunity_usd: float
    n_dominated_configs: int


class InferenceLensReport(BaseModel):
    report_id: str
    generated_at: str
    task_reports: Dict[str, TaskReport]
    global_verdict: str
    total_savings_opportunity_pct: float
    dominated_configs_found: bool
    top_routing_rule: Optional[str] = None
    pareto_curve_data: Dict[str, List[Dict]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Assembles the final InferenceLensReport from profiling + Pareto + routing data.
    Also renders terminal-formatted tables.
    """

    def generate(
        self,
        runs: Dict[str, ProfileRun],
        pareto_results: Dict[str, ParetoResult],
        routing_recommendations: Dict[str, RoutingRecommendation],
    ) -> InferenceLensReport:
        import uuid as _uuid

        task_reports: Dict[str, TaskReport] = {}
        global_verdicts: List[str] = []

        for task_type in runs:
            run = runs[task_type]
            pareto = pareto_results[task_type]
            routing = routing_recommendations[task_type]

            # Compute per-task stats
            n_dominated = len(pareto.dominated_tiers)
            large_cost = pareto.point_for("large")
            savings_opp = 0.0
            if large_cost and large_cost.avg_cost_usd > 0:
                best_frontier = min(
                    [p for p in pareto.points if p.pareto_status == ParetoStatus.FRONTIER],
                    key=lambda p: p.avg_cost_usd,
                    default=large_cost,
                )
                savings_opp = (large_cost.avg_cost_usd - best_frontier.avg_cost_usd) * run.n_prompts

            # Overall task verdict
            rule_verdicts = [r.verdict for r in routing.rules]
            if any(v == AuditVerdict.FAIL.value for v in rule_verdicts):
                task_verdict = AuditVerdict.FAIL.value
            elif any(v == AuditVerdict.WARN.value for v in rule_verdicts):
                task_verdict = AuditVerdict.WARN.value
            else:
                task_verdict = AuditVerdict.PASS.value

            global_verdicts.append(task_verdict)

            summary_lines = self._build_summary(pareto, routing, task_verdict)

            # Pareto curve data (for API / charting)
            curve_data = [
                {
                    "tier": p.tier,
                    "cost_usd": p.avg_cost_usd,
                    "quality": p.avg_quality,
                    "latency_ms": p.avg_latency_ms,
                    "status": p.pareto_status.value,
                }
                for p in sorted(pareto.points, key=lambda x: x.avg_cost_usd)
            ]

            task_reports[task_type] = TaskReport(
                task_type=task_type,
                pareto_result=pareto,
                routing_recommendation=routing,
                overall_verdict=task_verdict,
                summary_lines=summary_lines,
                savings_opportunity_usd=round(savings_opp, 4),
                n_dominated_configs=n_dominated,
            )

        # Global verdict: worst of all tasks
        if any(v == AuditVerdict.FAIL.value for v in global_verdicts):
            global_verdict = AuditVerdict.FAIL.value
        elif any(v == AuditVerdict.WARN.value for v in global_verdicts):
            global_verdict = AuditVerdict.WARN.value
        else:
            global_verdict = AuditVerdict.PASS.value

        # Aggregate savings
        all_savings_pct = [r.estimated_avg_savings_pct for r in routing_recommendations.values()]
        avg_savings = sum(all_savings_pct) / len(all_savings_pct) if all_savings_pct else 0.0

        # Top routing rule (highest savings, PASS verdict)
        all_rules = [
            rule
            for rec in routing_recommendations.values()
            for rule in rec.rules
            if rule.verdict == AuditVerdict.PASS.value
        ]
        top_rule = max(all_rules, key=lambda r: r.cost_savings_pct, default=None)

        # Build pareto_curve_data for all tasks
        pareto_curve_data = {}
        for task_type, tr in task_reports.items():
            pareto_curve_data[task_type] = [
                {
                    "tier": p.tier,
                    "cost_usd": p.avg_cost_usd,
                    "quality": p.avg_quality,
                    "latency_ms": p.avg_latency_ms,
                    "status": p.pareto_status.value,
                }
                for p in sorted(tr.pareto_result.points, key=lambda x: x.avg_cost_usd)
            ]

        return InferenceLensReport(
            report_id=f"rpt_{_uuid.uuid4().hex[:8]}",
            generated_at=datetime.now(timezone.utc).isoformat(),
            task_reports=task_reports,
            global_verdict=global_verdict,
            total_savings_opportunity_pct=round(avg_savings, 1),
            dominated_configs_found=any(t.n_dominated_configs > 0 for t in task_reports.values()),
            top_routing_rule=top_rule.human_readable if top_rule else None,
            pareto_curve_data=pareto_curve_data,
        )

    def _build_summary(
        self,
        pareto: ParetoResult,
        routing: RoutingRecommendation,
        verdict: str,
    ) -> List[str]:
        lines = []
        dominated = [p for p in pareto.points if p.pareto_status == ParetoStatus.DOMINATED]
        if dominated:
            names = ", ".join(p.tier for p in dominated)
            lines.append(f"Dominated configs detected: {names} — these are strictly outperformed by alternatives.")
        frontier = [p for p in pareto.points if p.pareto_status == ParetoStatus.FRONTIER]
        lines.append(f"Pareto frontier: {len(frontier)} configs ({', '.join(p.tier for p in frontier)})")
        lines.append(f"Recommended default: {pareto.recommended_default}")
        lines.append(f"Estimated avg savings via routing: {routing.estimated_avg_savings_pct:.1f}%")
        lines.append(f"Estimated avg quality delta: {routing.estimated_avg_quality_delta:+.4f}")
        lines.append(f"Routing audit verdict: {verdict}")
        return lines

    # --- Terminal rendering --------------------------------------------------

    def render_terminal(self, report: InferenceLensReport, task_type: str) -> str:
        """Render a terminal-formatted table for a specific task type."""
        tr = report.task_reports.get(task_type)
        if tr is None:
            return f"No report found for task_type={task_type!r}"

        pareto = tr.pareto_result
        routing = tr.routing_recommendation
        verdict = tr.overall_verdict

        verdict_icon = {"PASS": "✓", "WARN": "△", "FAIL": "✗"}.get(verdict, "?")
        header = f"InferenceLens Report — {task_type.capitalize()} Task"
        sep = "═" * 63

        lines = [
            "",
            header,
            sep,
            f"{'Config':<16} {'Cost/call':>10} {'Latency':>10} {'Quality':>8} {'Pareto':>14}",
            "─" * 63,
        ]

        for point in sorted(pareto.points, key=lambda p: p.avg_cost_usd, reverse=True):
            status_str = (
                f"FRONTIER ✓" if point.pareto_status == ParetoStatus.FRONTIER
                else f"DOMINATED ✗"
            )
            lines.append(
                f"{point.tier:<16} "
                f"${point.avg_cost_usd:>9.4f} "
                f"{int(point.avg_latency_ms):>8}ms "
                f"{point.avg_quality:>8.2f} "
                f"{status_str:>14}"
            )

        lines.append("─" * 63)

        # Top routing rules
        for rule in routing.rules:
            lines.append(f"  {rule.human_readable}")
            lines.append(f"    Verdict: {rule.verdict}")

        lines += [sep, f"Global verdict: {verdict} {verdict_icon}", sep, ""]
        return "\n".join(lines)

    def render_all(self, report: InferenceLensReport) -> str:
        """Render all task type tables."""
        parts = []
        for task_type in report.task_reports:
            parts.append(self.render_terminal(report, task_type))
        return "\n".join(parts)

    def save_json(self, report: InferenceLensReport, path: Optional[Path] = None) -> Path:
        if path is None:
            path = Path("inferencelens_report.json")
        with open(path, "w") as f:
            f.write(report.model_dump_json(indent=2))
        return path
