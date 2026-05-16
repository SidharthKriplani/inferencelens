"""
InferenceLens — ParetoAnalyzer
Finds the cost-quality Pareto frontier across model configurations.
A config is DOMINATED if another config achieves higher quality at lower cost.
A config is on the FRONTIER if no other config dominates it.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from config import MODEL_CONFIGS
from pipeline.profiler import ProfileRun


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ParetoStatus(str, Enum):
    FRONTIER = "FRONTIER"
    DOMINATED = "DOMINATED"


class ConfigPoint(BaseModel):
    tier: str
    model_id: str
    avg_cost_usd: float
    avg_quality: float
    avg_latency_ms: float
    pareto_status: ParetoStatus
    dominated_by: Optional[str] = None       # tier that dominates this one
    routing_efficiency: float = 0.0          # quality / normalized_cost
    cost_savings_vs_large: float = 0.0       # % cheaper than large model
    quality_delta_vs_large: float = 0.0      # quality diff vs large (negative = worse)
    pareto_rank: int = 0                     # 1 = best frontier point (lowest cost+highest quality)


class ParetoResult(BaseModel):
    task_type: str
    points: List[ConfigPoint]
    frontier_tiers: List[str]
    dominated_tiers: List[str]
    recommended_default: str                 # highest efficiency frontier point

    def point_for(self, tier: str) -> Optional[ConfigPoint]:
        return next((p for p in self.points if p.tier == tier), None)

    def large_point(self) -> Optional[ConfigPoint]:
        return self.point_for("large")


# ---------------------------------------------------------------------------
# ParetoAnalyzer
# ---------------------------------------------------------------------------

class ParetoAnalyzer:
    """
    Identifies which model configurations lie on the Pareto frontier
    (cost vs. quality). Configs that are strictly dominated — i.e., another
    config has both lower cost AND higher quality — are flagged as DOMINATED.
    """

    COST_NORMALIZATION_EPSILON: float = 1e-6

    def analyze(self, run: ProfileRun) -> ParetoResult:
        """
        Compute Pareto frontier for a ProfileRun.
        Returns ParetoResult with per-config classification.
        """
        # Aggregate per-tier
        tier_stats: Dict[str, Tuple[float, float, float]] = {}  # tier → (avg_cost, avg_quality, avg_latency)
        for tier in run.tiers_profiled:
            tier_stats[tier] = (
                run.avg_cost(tier),
                run.avg_quality(tier),
                run.avg_latency(tier),
            )

        # Determine Pareto status
        points: List[ConfigPoint] = []
        large_cost = tier_stats.get("large", (1e-9, 1.0, 1000))[0]
        large_quality = tier_stats.get("large", (1e-9, 1.0, 1000))[1]

        for tier, (cost, quality, latency) in tier_stats.items():
            status = ParetoStatus.FRONTIER
            dominated_by: Optional[str] = None

            # Check if any other config strictly dominates this one
            for other_tier, (other_cost, other_quality, _) in tier_stats.items():
                if other_tier == tier:
                    continue
                # Dominated: other is cheaper AND better quality
                if other_cost < cost and other_quality >= quality:
                    status = ParetoStatus.DOMINATED
                    dominated_by = other_tier
                    break
                # Also dominated if same cost but strictly better quality
                if abs(other_cost - cost) < 1e-9 and other_quality > quality:
                    status = ParetoStatus.DOMINATED
                    dominated_by = other_tier
                    break

            cfg = MODEL_CONFIGS[tier]
            # Routing efficiency: quality / log(cost+epsilon) — normalized to [0,1]
            # Use raw ratio with epsilon guard
            normalized_cost = max(cost, self.COST_NORMALIZATION_EPSILON)
            # We scale by 1000 so local ($0) gets efficiency from quality alone
            routing_efficiency = quality / (normalized_cost * 1000 + 0.001)

            cost_savings = 1.0 - (cost / max(large_cost, 1e-9)) if large_cost > 0 else 0.0
            quality_delta = quality - large_quality

            points.append(ConfigPoint(
                tier=tier,
                model_id=cfg.model_id,
                avg_cost_usd=round(cost, 6),
                avg_quality=round(quality, 4),
                avg_latency_ms=round(latency, 1),
                pareto_status=status,
                dominated_by=dominated_by,
                routing_efficiency=round(routing_efficiency, 4),
                cost_savings_vs_large=round(cost_savings, 4),
                quality_delta_vs_large=round(quality_delta, 4),
            ))

        # Assign Pareto ranks within frontier (sort by quality desc, then cost asc)
        frontier_points = [p for p in points if p.pareto_status == ParetoStatus.FRONTIER]
        frontier_points.sort(key=lambda p: (-p.avg_quality, p.avg_cost_usd))
        for rank, fp in enumerate(frontier_points, start=1):
            fp.pareto_rank = rank

        # Sort all points by cost desc for display
        points.sort(key=lambda p: p.avg_cost_usd, reverse=True)

        frontier_tiers = [p.tier for p in points if p.pareto_status == ParetoStatus.FRONTIER]
        dominated_tiers = [p.tier for p in points if p.pareto_status == ParetoStatus.DOMINATED]

        # Recommended default: highest routing efficiency on frontier
        frontier_sorted = sorted(
            [p for p in points if p.pareto_status == ParetoStatus.FRONTIER],
            key=lambda p: p.routing_efficiency,
            reverse=True,
        )
        recommended = frontier_sorted[0].tier if frontier_sorted else (frontier_tiers[0] if frontier_tiers else "medium")

        return ParetoResult(
            task_type=run.task_type,
            points=points,
            frontier_tiers=frontier_tiers,
            dominated_tiers=dominated_tiers,
            recommended_default=recommended,
        )

    def analyze_all(self, runs: Dict[str, ProfileRun]) -> Dict[str, ParetoResult]:
        return {task_type: self.analyze(run) for task_type, run in runs.items()}
