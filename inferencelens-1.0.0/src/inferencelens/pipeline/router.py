"""
InferenceLens — RoutingRecommender
Generates routing rules based on complexity signals and Pareto analysis.
Rule format: "IF task_type=X AND complexity<Y THEN use=Z (saves N% cost, quality_delta=D)"
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..config import MODEL_CONFIGS, ROUTING_THRESHOLDS, RoutingThresholds
from .pareto import ParetoResult, ConfigPoint, ParetoStatus


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ComplexityBand(str, Enum):
    LOW = "low"        # complexity < threshold_low
    MEDIUM = "medium"  # threshold_low <= complexity < threshold_high
    HIGH = "high"      # complexity >= threshold_high


class RoutingCondition(BaseModel):
    task_type: str
    complexity_band: ComplexityBand
    complexity_min: float
    complexity_max: float


class RoutingRule(BaseModel):
    rule_id: str
    condition: RoutingCondition
    recommended_tier: str
    model_id: str
    cost_savings_pct: float           # % savings vs large model
    quality_delta: float              # quality diff vs large (can be negative)
    routing_efficiency: float
    verdict: str                      # PASS | WARN | FAIL
    rationale: str
    human_readable: str               # the "IF...THEN..." string


class RoutingRecommendation(BaseModel):
    task_type: str
    rules: List[RoutingRule]
    default_tier: str
    estimated_avg_savings_pct: float
    estimated_avg_quality_delta: float


class AuditVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


# ---------------------------------------------------------------------------
# RoutingRecommender
# ---------------------------------------------------------------------------

class RoutingRecommender:
    """
    Generates routing rules for each task type and complexity band,
    leveraging Pareto analysis to ensure only non-dominated configs
    are ever recommended.
    """

    def __init__(self, thresholds: RoutingThresholds = ROUTING_THRESHOLDS) -> None:
        self.thresholds = thresholds

    # --- Complexity classification -------------------------------------------

    def classify_complexity(self, complexity_score: float) -> ComplexityBand:
        if complexity_score < self.thresholds.complexity_low:
            return ComplexityBand.LOW
        elif complexity_score < self.thresholds.complexity_high:
            return ComplexityBand.MEDIUM
        else:
            return ComplexityBand.HIGH

    # --- Rule generation -----------------------------------------------------

    def generate_rules(self, pareto_result: ParetoResult) -> RoutingRecommendation:
        """Generate routing rules for a task type based on Pareto analysis."""
        task_type = pareto_result.task_type
        rules: List[RoutingRule] = []

        # Get sorted frontier points (best quality first)
        frontier_points = sorted(
            [p for p in pareto_result.points if p.pareto_status == ParetoStatus.FRONTIER],
            key=lambda p: p.avg_quality,
            reverse=True,
        )
        large_point = pareto_result.point_for("large")

        # Band definitions
        bands = [
            (ComplexityBand.LOW, 0.0, self.thresholds.complexity_low),
            (ComplexityBand.MEDIUM, self.thresholds.complexity_low, self.thresholds.complexity_high),
            (ComplexityBand.HIGH, self.thresholds.complexity_high, 1.0),
        ]

        for band, cmin, cmax in bands:
            tier, point = self._select_tier_for_band(band, frontier_points, large_point)
            if point is None:
                continue

            savings_pct = point.cost_savings_vs_large * 100
            quality_delta = point.quality_delta_vs_large
            verdict = self._compute_verdict(point, savings_pct, quality_delta)
            rationale = self._build_rationale(tier, band, savings_pct, quality_delta, point)
            human_readable = self._format_rule(task_type, band, cmin, cmax, tier, savings_pct, quality_delta)

            rule_id = f"{task_type}_{band.value}"
            rules.append(RoutingRule(
                rule_id=rule_id,
                condition=RoutingCondition(
                    task_type=task_type,
                    complexity_band=band,
                    complexity_min=cmin,
                    complexity_max=cmax,
                ),
                recommended_tier=tier,
                model_id=MODEL_CONFIGS[tier].model_id,
                cost_savings_pct=round(savings_pct, 1),
                quality_delta=round(quality_delta, 4),
                routing_efficiency=point.routing_efficiency,
                verdict=verdict,
                rationale=rationale,
                human_readable=human_readable,
            ))

        # Estimate aggregate impact
        avg_savings = sum(r.cost_savings_pct for r in rules) / len(rules) if rules else 0.0
        avg_delta = sum(r.quality_delta for r in rules) / len(rules) if rules else 0.0

        return RoutingRecommendation(
            task_type=task_type,
            rules=rules,
            default_tier=pareto_result.recommended_default,
            estimated_avg_savings_pct=round(avg_savings, 1),
            estimated_avg_quality_delta=round(avg_delta, 4),
        )

    def _select_tier_for_band(
        self,
        band: ComplexityBand,
        frontier_points: List[ConfigPoint],
        large_point: Optional[ConfigPoint],
    ) -> Tuple[str, Optional[ConfigPoint]]:
        """
        Select the appropriate tier for each complexity band:
        - LOW complexity → cheapest frontier point that meets quality floor
        - MEDIUM complexity → middle frontier point by quality rank
        - HIGH complexity → highest quality frontier point
        """
        quality_floor = self.thresholds.quality_floor

        if band == ComplexityBand.LOW:
            # Cheapest that still exceeds quality floor
            eligible = [p for p in frontier_points if p.avg_quality >= quality_floor]
            if not eligible:
                eligible = frontier_points
            # Sort by cost ascending (cheapest wins)
            eligible.sort(key=lambda p: p.avg_cost_usd)
            point = eligible[0] if eligible else (frontier_points[-1] if frontier_points else None)

        elif band == ComplexityBand.MEDIUM:
            # Middle quality
            if len(frontier_points) >= 3:
                point = frontier_points[len(frontier_points) // 2]
            elif len(frontier_points) >= 2:
                point = frontier_points[1]
            else:
                point = frontier_points[0] if frontier_points else None

        else:  # HIGH
            # Best quality on frontier (or large if it's on frontier)
            point = frontier_points[0] if frontier_points else None

        tier = point.tier if point else "large"
        return tier, point

    def _compute_verdict(
        self,
        point: ConfigPoint,
        savings_pct: float,
        quality_delta: float,
    ) -> str:
        """PASS / WARN / FAIL based on routing quality."""
        if point.pareto_status == ParetoStatus.DOMINATED:
            return AuditVerdict.FAIL.value
        if abs(quality_delta) > 0.10:  # >10% quality drop
            return AuditVerdict.WARN.value
        if savings_pct < self.thresholds.cost_savings_min * 100 and point.tier != "large":
            return AuditVerdict.WARN.value
        return AuditVerdict.PASS.value

    def _build_rationale(
        self,
        tier: str,
        band: ComplexityBand,
        savings_pct: float,
        quality_delta: float,
        point: ConfigPoint,
    ) -> str:
        parts = [
            f"Selected {tier} (Pareto rank #{point.pareto_rank}) for {band.value}-complexity tasks.",
            f"Cost savings vs large: {savings_pct:.1f}%.",
            f"Quality delta vs large: {quality_delta:+.4f}.",
            f"Routing efficiency score: {point.routing_efficiency:.4f}.",
        ]
        if point.pareto_status == ParetoStatus.DOMINATED:
            parts.append("WARNING: Selected config is dominated — review Pareto analysis.")
        return " ".join(parts)

    def _format_rule(
        self,
        task_type: str,
        band: ComplexityBand,
        cmin: float,
        cmax: float,
        tier: str,
        savings_pct: float,
        quality_delta: float,
    ) -> str:
        model = MODEL_CONFIGS[tier].display_name
        if cmax >= 1.0:
            complexity_clause = f"complexity>={cmin:.2f}"
        else:
            complexity_clause = f"complexity<{cmax:.2f}"
        return (
            f"IF task_type={task_type} AND {complexity_clause} "
            f"THEN use={tier} [{model}] "
            f"(saves {savings_pct:.0f}% cost, quality_delta={quality_delta:+.2f})"
        )

    # --- Routing at inference time -------------------------------------------

    def route(
        self,
        task_type: str,
        complexity_score: float,
        recommendation: RoutingRecommendation,
    ) -> Tuple[str, RoutingRule]:
        """
        Given a live complexity score, return the recommended tier and rule.
        """
        band = self.classify_complexity(complexity_score)
        for rule in recommendation.rules:
            if rule.condition.complexity_band == band:
                return rule.recommended_tier, rule
        # Fallback: default tier
        default_rule = recommendation.rules[0] if recommendation.rules else None
        return recommendation.default_tier, default_rule

    def generate_all(
        self, pareto_results: Dict[str, ParetoResult]
    ) -> Dict[str, RoutingRecommendation]:
        return {tt: self.generate_rules(pr) for tt, pr in pareto_results.items()}
