"""
InferenceLens — Inference Cost/Quality Tradeoff Auditor

Finds the Pareto frontier across model configurations, flags dominated
routing decisions, and outputs IF/THEN routing rules with PASS/WARN/FAIL verdicts.

Quick start:
    from inferencelens.pipeline import (
        InferenceProfiler,
        ParetoAnalyzer,
        RoutingRecommender,
        ReportGenerator,
    )
    from inferencelens.config import MODEL_CONFIGS, ROUTING_THRESHOLDS
"""

from inferencelens.pipeline import (
    InferenceProfiler,
    ProfileRun,
    ProfiledCall,
    QualityEvaluator,
    QualityResult,
    CostTracker,
    InferenceRecord,
    TierSummary,
    ParetoAnalyzer,
    ParetoResult,
    ConfigPoint,
    ParetoStatus,
    RoutingRecommender,
    RoutingRecommendation,
    RoutingRule,
    AuditLogger,
    AuditEvent,
    AuditEventType,
    ReportGenerator,
    InferenceLensReport,
    TaskReport,
)
from inferencelens.config import (
    MODEL_CONFIGS,
    ModelCostConfig,
    ROUTING_THRESHOLDS,
    RoutingThresholds,
    QUALITY_BASELINES,
    TASK_TYPES,
)

__version__ = "1.0.0"

__all__ = [
    # Pipeline
    "InferenceProfiler", "ProfileRun", "ProfiledCall",
    "QualityEvaluator", "QualityResult",
    "CostTracker", "InferenceRecord", "TierSummary",
    "ParetoAnalyzer", "ParetoResult", "ConfigPoint", "ParetoStatus",
    "RoutingRecommender", "RoutingRecommendation", "RoutingRule",
    "AuditLogger", "AuditEvent", "AuditEventType",
    "ReportGenerator", "InferenceLensReport", "TaskReport",
    # Config
    "MODEL_CONFIGS", "ModelCostConfig",
    "ROUTING_THRESHOLDS", "RoutingThresholds",
    "QUALITY_BASELINES", "TASK_TYPES",
]
