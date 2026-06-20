"""
InferenceLens — Configuration
Central config for model configs, cost tables, and system settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
AUDIT_LOG_PATH = ROOT_DIR / "audit_log.jsonl"

# ---------------------------------------------------------------------------
# Model cost table (USD per 1K tokens — input / output)
# ---------------------------------------------------------------------------

class ModelCostConfig(BaseModel):
    model_id: str
    display_name: str
    input_cost_per_1k: float   # USD
    output_cost_per_1k: float  # USD
    avg_latency_ms: int        # simulated baseline latency
    tier: str                  # "large" | "medium" | "small" | "local"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000) * self.input_cost_per_1k + \
               (output_tokens / 1000) * self.output_cost_per_1k


MODEL_CONFIGS: Dict[str, ModelCostConfig] = {
    "large": ModelCostConfig(
        model_id="gpt-4o",
        display_name="gpt-4o (large)",
        input_cost_per_1k=0.005,
        output_cost_per_1k=0.015,
        avg_latency_ms=1240,
        tier="large",
    ),
    "medium": ModelCostConfig(
        model_id="gpt-4o-mini",
        display_name="gpt-4o-mini (medium)",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        avg_latency_ms=380,
        tier="medium",
    ),
    "small": ModelCostConfig(
        model_id="gpt-3.5-turbo",
        display_name="gpt-3.5-turbo (small)",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        avg_latency_ms=210,
        tier="small",
    ),
    "local": ModelCostConfig(
        model_id="ollama/mistral",
        display_name="ollama/mistral (local)",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        avg_latency_ms=890,
        tier="local",
    ),
}

# ---------------------------------------------------------------------------
# Quality baselines (simulated — seeded RNG in generator.py)
# ---------------------------------------------------------------------------

QUALITY_BASELINES: Dict[str, Dict[str, float]] = {
    "summarization": {
        "large": 0.91,
        "medium": 0.88,
        "small": 0.79,
        "local": 0.74,
    },
    "classification": {
        "large": 0.94,
        "medium": 0.91,
        "small": 0.86,
        "local": 0.78,
    },
    "extraction": {
        "large": 0.89,
        "medium": 0.85,
        "small": 0.77,
        "local": 0.70,
    },
}

# ---------------------------------------------------------------------------
# Routing thresholds
# ---------------------------------------------------------------------------

class RoutingThresholds(BaseModel):
    complexity_low: float = 0.35
    complexity_high: float = 0.70
    quality_floor: float = 0.75          # minimum acceptable quality
    cost_savings_min: float = 0.20       # must save at least 20% to route down
    confidence_threshold: float = 0.80  # below this → escalate

ROUTING_THRESHOLDS = RoutingThresholds()

# ---------------------------------------------------------------------------
# Evaluation weights
# ---------------------------------------------------------------------------

EVAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "summarization": {"rouge": 0.6, "semantic": 0.4},
    "classification": {"f1": 0.7, "semantic": 0.3},
    "extraction": {"exact_match": 0.5, "semantic": 0.5},
}

# ---------------------------------------------------------------------------
# Audit settings
# ---------------------------------------------------------------------------

AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", str(ROOT_DIR / "audit_log.jsonl"))
RANDOM_SEED = 42
N_SAMPLES_PER_TASK = 100
TASK_TYPES: List[str] = ["summarization", "classification", "extraction"]
