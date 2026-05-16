"""
InferenceLens — InferenceProfiler
Runs a batch of tasks against multiple model configurations,
logs tokens / latency / cost per call, and returns ProfileRun results.
In synthetic/offline mode, uses pre-computed dataset values.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from config import MODEL_CONFIGS, TASK_TYPES
from pipeline.cost_tracker import CostTracker, InferenceRecord
from pipeline.evaluator import QualityEvaluator, QualityResult
from data.generator import PromptRecord


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ProfiledCall(BaseModel):
    call_id: str
    prompt_id: str
    task_type: str
    tier: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    quality_score: float
    complexity_score: float
    fallback_used: bool = False


class ProfileRun(BaseModel):
    run_id: str
    timestamp: str
    task_type: str
    n_prompts: int
    tiers_profiled: List[str]
    calls: List[ProfiledCall]

    def calls_for_tier(self, tier: str) -> List[ProfiledCall]:
        return [c for c in self.calls if c.tier == tier]

    def avg_cost(self, tier: str) -> float:
        calls = self.calls_for_tier(tier)
        if not calls:
            return 0.0
        return sum(c.cost_usd for c in calls) / len(calls)

    def avg_quality(self, tier: str) -> float:
        calls = self.calls_for_tier(tier)
        if not calls:
            return 0.0
        return sum(c.quality_score for c in calls) / len(calls)

    def avg_latency(self, tier: str) -> float:
        calls = self.calls_for_tier(tier)
        if not calls:
            return 0.0
        return sum(c.latency_ms for c in calls) / len(calls)


# ---------------------------------------------------------------------------
# InferenceProfiler
# ---------------------------------------------------------------------------

class InferenceProfiler:
    """
    Profiles inference across model configs using synthetic dataset.

    Args:
        cost_tracker: shared CostTracker instance
        evaluator: shared QualityEvaluator instance
        tiers: list of model tiers to profile (default: all 4)
    """

    def __init__(
        self,
        cost_tracker: Optional[CostTracker] = None,
        evaluator: Optional[QualityEvaluator] = None,
        tiers: Optional[List[str]] = None,
    ) -> None:
        self.cost_tracker = cost_tracker or CostTracker()
        self.evaluator = evaluator or QualityEvaluator(use_simulated=True)
        self.tiers = tiers or list(MODEL_CONFIGS.keys())

    def profile_task(
        self,
        records: List[PromptRecord],
        task_type: str,
    ) -> ProfileRun:
        """
        Profile all records for a single task type across all tiers.
        Uses simulated quality + token counts from the dataset.
        """
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        calls: List[ProfiledCall] = []
        task_records = [r for r in records if r.task_type == task_type]

        for record in task_records:
            for tier in self.tiers:
                cfg = MODEL_CONFIGS[tier]

                # Use pre-computed simulated values
                prompt_tokens = int(record.expected_tokens.get(tier, 200) * 0.75)
                completion_tokens = int(record.expected_tokens.get(tier, 200) * 0.25)
                latency_ms = record.simulated_latency_ms.get(tier, cfg.avg_latency_ms)
                sim_quality = record.simulated_quality.get(tier, 0.7)
                cost_usd = record.simulated_cost_usd.get(tier, 0.0)

                # Log to cost tracker
                inf_record = self.cost_tracker.log(
                    task_type=task_type,
                    tier=tier,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    prompt_id=record.prompt_id,
                )

                # Evaluate quality (using simulated score)
                qual_result = self.evaluator.evaluate(
                    prompt_id=record.prompt_id,
                    task_type=task_type,
                    tier=tier,
                    predicted=f"[simulated output for {tier}]",
                    ground_truth=record.ground_truth,
                    simulated_quality=sim_quality,
                )

                calls.append(ProfiledCall(
                    call_id=inf_record.record_id,
                    prompt_id=record.prompt_id,
                    task_type=task_type,
                    tier=tier,
                    model_id=cfg.model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    quality_score=qual_result.primary_score,
                    complexity_score=record.complexity_score,
                    fallback_used=qual_result.fallback_used,
                ))

        return ProfileRun(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_type=task_type,
            n_prompts=len(task_records),
            tiers_profiled=self.tiers,
            calls=calls,
        )

    def profile_all(self, records: List[PromptRecord]) -> Dict[str, ProfileRun]:
        """Profile all task types. Returns dict keyed by task_type."""
        runs: Dict[str, ProfileRun] = {}
        for task_type in TASK_TYPES:
            runs[task_type] = self.profile_task(records, task_type)
        return runs
