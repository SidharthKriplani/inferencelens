"""
InferenceLens — QualityEvaluator
Measures output quality via:
  - Semantic similarity (cosine on TF-IDF vectors — no heavy deps)
  - Task-specific metrics:
      summarization → ROUGE-1 F1
      classification → macro F1
      extraction → field-level exact match + partial credit
Graceful degradation: if a metric fails, falls back to length-based proxy.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from config import EVAL_WEIGHTS


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class QualityResult(BaseModel):
    prompt_id: str
    task_type: str
    tier: str
    primary_score: float          # main weighted composite (0–1)
    rouge1_f1: Optional[float] = None
    f1_score: Optional[float] = None
    exact_match: Optional[float] = None
    semantic_similarity: float = 0.0
    fallback_used: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (no scikit-learn required)
# ---------------------------------------------------------------------------

class _TFIDFVectorizer:
    """Minimal TF-IDF implementation for two-document cosine similarity."""

    @staticmethod
    def cosine_similarity(a: str, b: str) -> float:
        try:
            tok_a = _tokenize(a)
            tok_b = _tokenize(b)
            if not tok_a or not tok_b:
                return 0.0
            vocab = set(tok_a) | set(tok_b)
            cnt_a = Counter(tok_a)
            cnt_b = Counter(tok_b)
            # TF (raw count / doc length)
            vec_a = {w: cnt_a[w] / len(tok_a) for w in vocab}
            vec_b = {w: cnt_b[w] / len(tok_b) for w in vocab}
            dot = sum(vec_a[w] * vec_b[w] for w in vocab)
            norm_a = math.sqrt(sum(v**2 for v in vec_a.values()))
            norm_b = math.sqrt(sum(v**2 for v in vec_b.values()))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return round(dot / (norm_a * norm_b), 4)
        except Exception:
            return 0.0


_vectorizer = _TFIDFVectorizer()


# ---------------------------------------------------------------------------
# ROUGE-1 F1
# ---------------------------------------------------------------------------

def _rouge1_f1(hypothesis: str, reference: str) -> float:
    try:
        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)
        if not ref_tokens:
            return 0.0
        hyp_counter = Counter(hyp_tokens)
        ref_counter = Counter(ref_tokens)
        overlap = sum((hyp_counter & ref_counter).values())
        precision = overlap / len(hyp_tokens) if hyp_tokens else 0.0
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Classification F1
# ---------------------------------------------------------------------------

def _macro_f1(predicted: str, ground_truth: str) -> float:
    """Single-label classification F1 (binary match at label level)."""
    try:
        pred = predicted.strip().lower()
        ref = ground_truth.strip().lower()
        return 1.0 if pred == ref else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Extraction exact match
# ---------------------------------------------------------------------------

def _extraction_score(predicted: str, ground_truth: str) -> float:
    """
    Field-level extraction accuracy.
    Parses both as JSON dicts; awards 1 pt per matching field value,
    partial credit (0.5) for approximate matches.
    """
    try:
        pred_dict = _parse_json_fuzzy(predicted)
        ref_dict = _parse_json_fuzzy(ground_truth)
        if not ref_dict:
            return 0.0
        scores = []
        for key, ref_val in ref_dict.items():
            pred_val = pred_dict.get(key, "")
            if _normalize(str(pred_val)) == _normalize(str(ref_val)):
                scores.append(1.0)
            elif _normalize(str(ref_val)) in _normalize(str(pred_val)) or \
                 _normalize(str(pred_val)) in _normalize(str(ref_val)):
                scores.append(0.5)
            else:
                scores.append(0.0)
        return round(sum(scores) / len(scores), 4) if scores else 0.0
    except Exception:
        return 0.0


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _parse_json_fuzzy(text: str) -> Dict[str, Any]:
    """Try JSON parse; fall back to key:value pattern extraction."""
    try:
        return json.loads(text)
    except Exception:
        pairs: Dict[str, Any] = {}
        for m in re.finditer(r'"?([\w_]+)"?\s*[=:]\s*"?([^",\n}]+)"?', text):
            pairs[m.group(1).strip()] = m.group(2).strip()
        return pairs


# ---------------------------------------------------------------------------
# Length-based fallback proxy
# ---------------------------------------------------------------------------

def _length_proxy(predicted: str, ground_truth: str) -> float:
    """Fallback: overlap-based length ratio (0–1)."""
    try:
        pred_len = len(_tokenize(predicted))
        ref_len = len(_tokenize(ground_truth))
        if ref_len == 0:
            return 0.5
        ratio = pred_len / ref_len
        return round(min(ratio, 1.0) * 0.5 + 0.5 * _vectorizer.cosine_similarity(predicted, ground_truth), 4)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# QualityEvaluator
# ---------------------------------------------------------------------------

class QualityEvaluator:
    """
    Evaluates predicted output against ground truth using task-specific metrics.

    Args:
        use_simulated: If True, use pre-computed simulated quality scores from
                       the dataset (for demo/offline mode without actual model calls).
    """

    def __init__(self, use_simulated: bool = True) -> None:
        self.use_simulated = use_simulated

    def evaluate(
        self,
        prompt_id: str,
        task_type: str,
        tier: str,
        predicted: str,
        ground_truth: str,
        simulated_quality: Optional[float] = None,
    ) -> QualityResult:
        """
        Compute quality score.

        If use_simulated=True and simulated_quality is provided, returns that
        score directly (with semantic sim computed as cross-check).
        Otherwise computes metrics from scratch.
        """
        if self.use_simulated and simulated_quality is not None:
            sem_sim = _vectorizer.cosine_similarity(predicted, ground_truth)
            return QualityResult(
                prompt_id=prompt_id,
                task_type=task_type,
                tier=tier,
                primary_score=simulated_quality,
                semantic_similarity=sem_sim,
                fallback_used=False,
                notes="simulated_quality_used",
            )

        return self._compute_metrics(prompt_id, task_type, tier, predicted, ground_truth)

    def _compute_metrics(
        self,
        prompt_id: str,
        task_type: str,
        tier: str,
        predicted: str,
        ground_truth: str,
    ) -> QualityResult:
        fallback = False
        notes = ""
        sem_sim = _vectorizer.cosine_similarity(predicted, ground_truth)
        weights = EVAL_WEIGHTS.get(task_type, {"semantic": 1.0})

        try:
            if task_type == "summarization":
                rouge = _rouge1_f1(predicted, ground_truth)
                primary = weights.get("rouge", 0.6) * rouge + weights.get("semantic", 0.4) * sem_sim
                return QualityResult(
                    prompt_id=prompt_id, task_type=task_type, tier=tier,
                    primary_score=round(primary, 4),
                    rouge1_f1=rouge, semantic_similarity=sem_sim,
                )

            elif task_type == "classification":
                f1 = _macro_f1(predicted, ground_truth)
                primary = weights.get("f1", 0.7) * f1 + weights.get("semantic", 0.3) * sem_sim
                return QualityResult(
                    prompt_id=prompt_id, task_type=task_type, tier=tier,
                    primary_score=round(primary, 4),
                    f1_score=f1, semantic_similarity=sem_sim,
                )

            elif task_type == "extraction":
                em = _extraction_score(predicted, ground_truth)
                primary = weights.get("exact_match", 0.5) * em + weights.get("semantic", 0.5) * sem_sim
                return QualityResult(
                    prompt_id=prompt_id, task_type=task_type, tier=tier,
                    primary_score=round(primary, 4),
                    exact_match=em, semantic_similarity=sem_sim,
                )

            else:
                fallback = True
                notes = f"unknown_task_type:{task_type}"
                primary = _length_proxy(predicted, ground_truth)
                return QualityResult(
                    prompt_id=prompt_id, task_type=task_type, tier=tier,
                    primary_score=primary,
                    semantic_similarity=sem_sim,
                    fallback_used=True, notes=notes,
                )

        except Exception as exc:
            primary = _length_proxy(predicted, ground_truth)
            return QualityResult(
                prompt_id=prompt_id, task_type=task_type, tier=tier,
                primary_score=primary,
                semantic_similarity=sem_sim,
                fallback_used=True,
                notes=f"exception:{exc}",
            )

    def batch_evaluate(
        self,
        records: List[Dict],
    ) -> List[QualityResult]:
        """Evaluate a list of dicts with keys: prompt_id, task_type, tier, predicted, ground_truth, simulated_quality."""
        results = []
        for rec in records:
            results.append(self.evaluate(
                prompt_id=rec["prompt_id"],
                task_type=rec["task_type"],
                tier=rec["tier"],
                predicted=rec.get("predicted", ""),
                ground_truth=rec.get("ground_truth", ""),
                simulated_quality=rec.get("simulated_quality"),
            ))
        return results
