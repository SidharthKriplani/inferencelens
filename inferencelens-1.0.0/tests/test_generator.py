"""Tests for synthetic data generator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.generator import SyntheticDataGenerator, PromptRecord
from config import TASK_TYPES, MODEL_CONFIGS, N_SAMPLES_PER_TASK


def test_generator_produces_correct_count():
    gen = SyntheticDataGenerator(seed=42)
    records = gen.generate()
    assert len(records) == len(TASK_TYPES) * N_SAMPLES_PER_TASK


def test_generator_is_deterministic():
    gen1 = SyntheticDataGenerator(seed=42)
    gen2 = SyntheticDataGenerator(seed=42)
    r1 = gen1.generate()
    r2 = gen2.generate()
    assert r1[0].prompt_id == r2[0].prompt_id
    assert r1[0].complexity_score == r2[0].complexity_score


def test_task_type_distribution():
    gen = SyntheticDataGenerator(seed=42)
    records = gen.generate()
    for tt in TASK_TYPES:
        count = sum(1 for r in records if r.task_type == tt)
        assert count == N_SAMPLES_PER_TASK, f"Expected {N_SAMPLES_PER_TASK} for {tt}, got {count}"


def test_all_tiers_present():
    gen = SyntheticDataGenerator(seed=42)
    records = gen.generate()
    for r in records[:10]:
        for tier in MODEL_CONFIGS:
            assert tier in r.simulated_quality
            assert tier in r.simulated_cost_usd
            assert tier in r.simulated_latency_ms


def test_complexity_in_range():
    gen = SyntheticDataGenerator(seed=42)
    records = gen.generate()
    for r in records:
        assert 0.0 <= r.complexity_score <= 1.0


def test_quality_in_range():
    gen = SyntheticDataGenerator(seed=42)
    records = gen.generate()
    for r in records:
        for tier, q in r.simulated_quality.items():
            assert 0.0 <= q <= 1.0, f"Quality {q} out of range for {r.prompt_id}/{tier}"


def test_cost_non_negative():
    gen = SyntheticDataGenerator(seed=42)
    records = gen.generate()
    for r in records:
        assert r.simulated_cost_usd.get("local", 0.0) == 0.0
        for tier, cost in r.simulated_cost_usd.items():
            assert cost >= 0.0


def test_large_quality_highest():
    """Large model should have highest quality on average."""
    gen = SyntheticDataGenerator(seed=42)
    records = gen.generate()
    avg_quality = {tier: 0.0 for tier in MODEL_CONFIGS}
    for r in records:
        for tier, q in r.simulated_quality.items():
            avg_quality[tier] += q
    n = len(records)
    avg_quality = {k: v / n for k, v in avg_quality.items()}
    assert avg_quality["large"] > avg_quality["medium"]
    assert avg_quality["medium"] > avg_quality["small"]
    assert avg_quality["small"] > avg_quality["local"]
