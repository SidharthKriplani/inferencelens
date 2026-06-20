"""
InferenceLens — AuditLogger
JSONL audit trail of all inference decisions and routing choices.
Provides queryable log with filtering and replay capability.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field

from ..config import AUDIT_LOG_PATH


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class AuditEventType(str, Enum):
    INFERENCE_CALL = "inference_call"
    ROUTING_DECISION = "routing_decision"
    QUALITY_EVALUATION = "quality_evaluation"
    PARETO_ANALYSIS = "pareto_analysis"
    REPORT_GENERATED = "report_generated"
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"


class AuditVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Audit event model
# ---------------------------------------------------------------------------

class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:10]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: AuditEventType
    task_type: Optional[str] = None
    prompt_id: Optional[str] = None
    tier: Optional[str] = None
    model_id: Optional[str] = None
    verdict: Optional[AuditVerdict] = None

    # Inference fields
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    quality_score: Optional[float] = None
    complexity_score: Optional[float] = None
    total_tokens: Optional[int] = None

    # Routing fields
    routing_rule_id: Optional[str] = None
    cost_savings_pct: Optional[float] = None
    quality_delta: Optional[float] = None

    # Pareto fields
    pareto_status: Optional[str] = None
    pareto_rank: Optional[int] = None
    dominated_by: Optional[str] = None

    # Freeform metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    def to_jsonl(self) -> str:
        return self.model_dump_json()


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Append-only JSONL audit logger.

    All inference decisions and routing choices are recorded with full context,
    enabling post-hoc analysis and compliance reporting.
    """

    def __init__(self, log_path: Optional[Union[str, Path]] = None) -> None:
        self.log_path = Path(log_path or AUDIT_LOG_PATH)
        self._buffer: List[AuditEvent] = []
        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Logging helpers -----------------------------------------------------

    def _write(self, event: AuditEvent) -> None:
        self._buffer.append(event)
        with open(self.log_path, "a") as f:
            f.write(event.to_jsonl() + "\n")

    def log_pipeline_start(self, task_types: List[str], n_records: int) -> AuditEvent:
        event = AuditEvent(
            event_type=AuditEventType.PIPELINE_START,
            metadata={"task_types": task_types, "n_records": n_records},
            notes="InferenceLens pipeline started",
        )
        self._write(event)
        return event

    def log_pipeline_end(self, duration_s: float, n_calls: int) -> AuditEvent:
        event = AuditEvent(
            event_type=AuditEventType.PIPELINE_END,
            metadata={"duration_seconds": round(duration_s, 2), "total_calls": n_calls},
            notes="InferenceLens pipeline completed",
        )
        self._write(event)
        return event

    def log_inference(
        self,
        task_type: str,
        prompt_id: str,
        tier: str,
        model_id: str,
        cost_usd: float,
        latency_ms: int,
        quality_score: float,
        complexity_score: float,
        total_tokens: int,
        pareto_status: Optional[str] = None,
        verdict: Optional[str] = None,
    ) -> AuditEvent:
        _verdict = AuditVerdict(verdict) if verdict else None
        event = AuditEvent(
            event_type=AuditEventType.INFERENCE_CALL,
            task_type=task_type,
            prompt_id=prompt_id,
            tier=tier,
            model_id=model_id,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            quality_score=quality_score,
            complexity_score=complexity_score,
            total_tokens=total_tokens,
            pareto_status=pareto_status,
            verdict=_verdict,
        )
        self._write(event)
        return event

    def log_routing_decision(
        self,
        task_type: str,
        prompt_id: str,
        complexity_score: float,
        selected_tier: str,
        routing_rule_id: str,
        cost_savings_pct: float,
        quality_delta: float,
        verdict: str,
    ) -> AuditEvent:
        event = AuditEvent(
            event_type=AuditEventType.ROUTING_DECISION,
            task_type=task_type,
            prompt_id=prompt_id,
            tier=selected_tier,
            complexity_score=complexity_score,
            routing_rule_id=routing_rule_id,
            cost_savings_pct=cost_savings_pct,
            quality_delta=quality_delta,
            verdict=AuditVerdict(verdict),
        )
        self._write(event)
        return event

    def log_pareto_analysis(
        self,
        task_type: str,
        tier: str,
        pareto_status: str,
        pareto_rank: int,
        dominated_by: Optional[str],
        routing_efficiency: float,
    ) -> AuditEvent:
        event = AuditEvent(
            event_type=AuditEventType.PARETO_ANALYSIS,
            task_type=task_type,
            tier=tier,
            pareto_status=pareto_status,
            pareto_rank=pareto_rank,
            dominated_by=dominated_by,
            metadata={"routing_efficiency": routing_efficiency},
        )
        self._write(event)
        return event

    def log_report_generated(self, task_type: str, report_path: str) -> AuditEvent:
        event = AuditEvent(
            event_type=AuditEventType.REPORT_GENERATED,
            task_type=task_type,
            metadata={"report_path": report_path},
        )
        self._write(event)
        return event

    # --- Query interface -----------------------------------------------------

    def read_all(self) -> List[AuditEvent]:
        """Read all events from the JSONL log file."""
        if not self.log_path.exists():
            return []
        events = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(AuditEvent.model_validate_json(line))
                    except Exception:
                        pass
        return events

    def filter(
        self,
        event_type: Optional[AuditEventType] = None,
        task_type: Optional[str] = None,
        tier: Optional[str] = None,
        verdict: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        events = self.read_all()
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if task_type:
            events = [e for e in events if e.task_type == task_type]
        if tier:
            events = [e for e in events if e.tier == tier]
        if verdict:
            events = [e for e in events if e.verdict and e.verdict.value == verdict]
        if limit:
            events = events[-limit:]
        return events

    def stats(self) -> Dict[str, Any]:
        """High-level stats from audit log."""
        events = self.read_all()
        inference_events = [e for e in events if e.event_type == AuditEventType.INFERENCE_CALL]
        routing_events = [e for e in events if e.event_type == AuditEventType.ROUTING_DECISION]

        verdict_counts: Dict[str, int] = {}
        for e in inference_events + routing_events:
            if e.verdict:
                verdict_counts[e.verdict.value] = verdict_counts.get(e.verdict.value, 0) + 1

        return {
            "total_events": len(events),
            "inference_calls": len(inference_events),
            "routing_decisions": len(routing_events),
            "verdict_counts": verdict_counts,
            "total_cost_usd": round(sum(e.cost_usd or 0.0 for e in inference_events), 6),
        }

    def clear(self) -> None:
        """Truncate the log file (for testing)."""
        with open(self.log_path, "w") as f:
            pass
        self._buffer.clear()
