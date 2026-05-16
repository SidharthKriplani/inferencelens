"""
InferenceLens — CostTracker
Logs token usage, estimated cost (USD), and latency per inference call.
Supports aggregation by model tier and task type.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from config import MODEL_CONFIGS, ModelCostConfig


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class InferenceRecord(BaseModel):
    record_id: str
    timestamp: str
    task_type: str
    tier: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    prompt_id: Optional[str] = None

    @property
    def cost_per_token(self) -> float:
        return self.cost_usd / max(self.total_tokens, 1)


class TierSummary(BaseModel):
    tier: str
    model_id: str
    n_calls: int
    total_tokens: int
    total_cost_usd: float
    avg_cost_usd: float
    avg_latency_ms: float
    avg_tokens_per_call: float
    task_type: Optional[str] = None


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """
    Tracks and aggregates per-call inference costs.

    Usage:
        tracker = CostTracker()
        record = tracker.log(
            task_type="summarization",
            tier="medium",
            prompt_tokens=250,
            completion_tokens=80,
            latency_ms=390,
        )
        summary = tracker.summarize(task_type="summarization")
    """

    def __init__(self) -> None:
        self._records: List[InferenceRecord] = []
        self._counter: int = 0

    # --- Logging -------------------------------------------------------------

    def log(
        self,
        task_type: str,
        tier: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        prompt_id: Optional[str] = None,
    ) -> InferenceRecord:
        """Record a single inference call."""
        cfg = MODEL_CONFIGS.get(tier)
        if cfg is None:
            raise ValueError(f"Unknown tier: {tier!r}. Valid tiers: {list(MODEL_CONFIGS.keys())}")

        self._counter += 1
        record = InferenceRecord(
            record_id=f"inf_{self._counter:05d}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_type=task_type,
            tier=tier,
            model_id=cfg.model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cfg.estimate_cost(prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
            prompt_id=prompt_id,
        )
        self._records.append(record)
        return record

    # --- Queries -------------------------------------------------------------

    def records(
        self,
        task_type: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> List[InferenceRecord]:
        out = self._records
        if task_type:
            out = [r for r in out if r.task_type == task_type]
        if tier:
            out = [r for r in out if r.tier == tier]
        return out

    def summarize(
        self,
        task_type: Optional[str] = None,
    ) -> List[TierSummary]:
        """Return per-tier aggregated statistics."""
        grouped: Dict[str, List[InferenceRecord]] = defaultdict(list)
        for r in self.records(task_type=task_type):
            grouped[r.tier].append(r)

        summaries: List[TierSummary] = []
        for tier, recs in grouped.items():
            cfg = MODEL_CONFIGS[tier]
            n = len(recs)
            total_tokens = sum(r.total_tokens for r in recs)
            total_cost = sum(r.cost_usd for r in recs)
            avg_cost = total_cost / n if n else 0.0
            avg_latency = sum(r.latency_ms for r in recs) / n if n else 0.0
            avg_tokens = total_tokens / n if n else 0.0
            summaries.append(TierSummary(
                tier=tier,
                model_id=cfg.model_id,
                n_calls=n,
                total_tokens=total_tokens,
                total_cost_usd=round(total_cost, 6),
                avg_cost_usd=round(avg_cost, 6),
                avg_latency_ms=round(avg_latency, 1),
                avg_tokens_per_call=round(avg_tokens, 1),
                task_type=task_type,
            ))

        # Sort by cost descending
        summaries.sort(key=lambda s: s.avg_cost_usd, reverse=True)
        return summaries

    def total_cost_usd(self, task_type: Optional[str] = None) -> float:
        return sum(r.cost_usd for r in self.records(task_type=task_type))

    def cost_breakdown_by_tier(
        self, task_type: Optional[str] = None
    ) -> Dict[str, float]:
        breakdown: Dict[str, float] = defaultdict(float)
        for r in self.records(task_type=task_type):
            breakdown[r.tier] += r.cost_usd
        return dict(breakdown)

    def clear(self) -> None:
        self._records.clear()
        self._counter = 0
