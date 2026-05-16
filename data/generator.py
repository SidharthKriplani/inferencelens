"""
InferenceLens — Synthetic Data Generator
Generates 300 prompts (100 per task type) with ground truth outputs and
simulated quality scores / token counts per model config.
All generation is deterministic (seeded RNG).
"""

from __future__ import annotations

import json
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import (
    MODEL_CONFIGS,
    QUALITY_BASELINES,
    RANDOM_SEED,
    N_SAMPLES_PER_TASK,
    TASK_TYPES,
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PromptRecord:
    prompt_id: str
    task_type: str        # summarization | classification | extraction
    prompt: str
    ground_truth: str
    complexity_score: float                      # 0–1
    expected_tokens: Dict[str, int]              # tier → token count
    simulated_quality: Dict[str, float]          # tier → quality score
    simulated_latency_ms: Dict[str, int]         # tier → latency
    simulated_cost_usd: Dict[str, float]         # tier → cost

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "complexity_score": self.complexity_score,
            "expected_tokens": self.expected_tokens,
            "simulated_quality": self.simulated_quality,
            "simulated_latency_ms": self.simulated_latency_ms,
            "simulated_cost_usd": self.simulated_cost_usd,
        }


# ---------------------------------------------------------------------------
# Source material pools (compact but realistic)
# ---------------------------------------------------------------------------

_SUMMARY_ARTICLES = [
    ("tech", "A new study from MIT found that transformer-based models trained on diverse corpora "
     "outperform fine-tuned specialists by a 12% margin on zero-shot tasks. Researchers attribute "
     "this to emergent reasoning capabilities that arise only after scale thresholds are crossed. "
     "The paper, published in Nature Machine Intelligence, suggests compute efficiency curves "
     "are flattening, raising questions about the cost-effectiveness of continued scaling."),
    ("finance", "The Federal Reserve signaled a pause in rate hikes following three consecutive "
     "months of cooling inflation data. Markets responded with a 2.3% rally in the S&P 500 as "
     "investors reassessed their rate-path assumptions. Economists warn that services inflation "
     "remains sticky and premature easing could re-accelerate consumer prices into 2025."),
    ("health", "Clinical trial results for a novel mRNA-based treatment for Type 1 diabetes showed "
     "a 61% reduction in insulin dependence among participants over 18 months. The therapy works "
     "by reprogramming regulatory T-cells to tolerate beta-cell antigens. FDA Breakthrough Device "
     "Designation was granted, placing the treatment on an expedited review track."),
    ("policy", "The EU AI Act entered enforcement for high-risk system categories, requiring conformity "
     "assessments, bias audits, and human oversight mechanisms for AI deployed in credit scoring, "
     "hiring, and critical infrastructure. Non-compliant providers face fines up to 3% of global "
     "turnover. Member states have 12 months to establish national market surveillance authorities."),
    ("science", "Astronomers using the James Webb Space Telescope detected water vapor signatures "
     "in the atmosphere of exoplanet K2-18b, a sub-Neptune 120 light-years away. The finding "
     "revives debate about habitability conditions in the 'Hycean world' class of planets. "
     "Spectroscopic analysis also revealed traces of dimethyl sulfide, a compound produced "
     "exclusively by marine phytoplankton on Earth."),
]

_CLASSIFICATION_EXAMPLES = [
    ("This product completely changed my morning routine. Fast shipping, great quality — 5 stars!", "positive"),
    ("Arrived damaged. Customer service took two weeks to respond and offered no refund.", "negative"),
    ("Decent enough for the price. Nothing special but does what it says on the tin.", "neutral"),
    ("Absolutely love it. Bought three for family members already. Highly recommend!", "positive"),
    ("Not what was advertised. Photo looks nothing like the actual item.", "negative"),
    ("Works fine. Packaging could be better but product itself is acceptable.", "neutral"),
    ("Game changer. I've tried dozens of competitors and this is by far the best.", "positive"),
    ("Stopped working after two weeks. Very disappointed given the price point.", "negative"),
    ("Average quality. You get what you pay for at this price range.", "neutral"),
    ("Exceeded expectations in every way. Fast delivery and pristine condition.", "positive"),
]

_EXTRACTION_TEMPLATES = [
    ("Invoice #{inv} dated {date}. Bill to: {company}. Total amount due: ${amount}. "
     "Payment terms: Net {terms} days. Contact: {email}.",
     {"invoice_number": "{inv}", "date": "{date}", "company": "{company}",
      "amount": "${amount}", "payment_terms": "Net {terms} days", "email": "{email}"}),
    ("Contract between {party_a} (Client) and {party_b} (Vendor) effective {date}. "
     "Scope: {scope}. Term: {duration} months. Value: ${value}. Governed by laws of {state}.",
     {"client": "{party_a}", "vendor": "{party_b}", "effective_date": "{date}",
      "contract_value": "${value}", "duration_months": "{duration}"}),
    ("Job posting for {role} at {company}. Location: {city}. Salary: {salary}. "
     "Required experience: {years} years. Apply by: {deadline}. Contact: {contact}.",
     {"role": "{role}", "company": "{company}", "location": "{city}",
      "salary_range": "{salary}", "application_deadline": "{deadline}"}),
]

_COMPANIES = ["Acme Corp", "TechNova Inc", "BlueSky LLC", "Meridian Partners", "Apex Solutions",
               "Quantum Dynamics", "Stellar Analytics", "Vertex Systems", "Cascade Ventures"]
_ROLES = ["Senior ML Engineer", "Data Scientist", "AI Product Manager", "Platform Engineer",
          "MLOps Lead", "Research Scientist", "Applied AI Engineer"]
_CITIES = ["San Francisco", "New York", "Austin", "Seattle", "Boston", "Chicago", "London"]
_STATES = ["California", "New York", "Texas", "Washington", "Massachusetts"]
_SCOPES = ["cloud infrastructure migration", "ML pipeline development", "data warehouse modernization",
           "API integration services", "model monitoring platform"]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    def __init__(self, seed: int = RANDOM_SEED, n_per_task: int = N_SAMPLES_PER_TASK):
        self.seed = seed
        self.n_per_task = n_per_task
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    # --- Summarization -------------------------------------------------------

    def _make_summarization_record(self, idx: int) -> PromptRecord:
        article_tuple = self._SUMMARY_ARTICLES[idx % len(self._SUMMARY_ARTICLES)]
        domain, article_text = article_tuple

        # vary length via repetition / trimming
        multiplier = self.rng.randint(1, 3)
        body = " ".join([article_text] * multiplier)
        prompt = f"Summarize the following article in 2–3 sentences:\n\n{body}"
        ground_truth = f"[{domain.upper()}] {article_text[:120].rstrip(',')}..."

        complexity = min(1.0, 0.3 + (multiplier - 1) * 0.25 + self.np_rng.uniform(-0.05, 0.05))
        return self._build_record(f"sum_{idx:03d}", "summarization", prompt, ground_truth, complexity)

    # --- Classification ------------------------------------------------------

    def _make_classification_record(self, idx: int) -> PromptRecord:
        text, label = self._CLASSIFICATION_EXAMPLES[idx % len(self._CLASSIFICATION_EXAMPLES)]
        prompt = (f"Classify the sentiment of the following customer review.\n"
                  f"Output exactly one word: positive, negative, or neutral.\n\nReview: {text}")
        ground_truth = label
        # short prompts → low complexity
        complexity = 0.2 + self.np_rng.uniform(-0.05, 0.15)
        complexity = float(np.clip(complexity, 0.1, 0.6))
        return self._build_record(f"cls_{idx:03d}", "classification", prompt, ground_truth, complexity)

    # --- Extraction ----------------------------------------------------------

    def _make_extraction_record(self, idx: int) -> PromptRecord:
        template_tuple = self._EXTRACTION_TEMPLATES[idx % len(self._EXTRACTION_TEMPLATES)]
        template, gt_template = template_tuple

        inv = f"INV-{self.rng.randint(1000, 9999)}"
        date = f"2024-{self.rng.randint(1,12):02d}-{self.rng.randint(1,28):02d}"
        company = self.rng.choice(self._COMPANIES)
        amount = f"{self.rng.randint(500, 50000):,}"
        terms = self.rng.choice([15, 30, 45, 60])
        email = f"billing@{company.lower().replace(' ', '')}.com"
        party_a = self.rng.choice(self._COMPANIES)
        party_b = self.rng.choice([c for c in self._COMPANIES if c != party_a])
        duration = self.rng.choice([3, 6, 12, 24])
        value = f"{self.rng.randint(10000, 500000):,}"
        state = self.rng.choice(self._STATES)
        scope = self.rng.choice(self._SCOPES)
        role = self.rng.choice(self._ROLES)
        city = self.rng.choice(self._CITIES)
        salary = f"${self.rng.randint(80, 250)}K–${self.rng.randint(100, 300)}K"
        years = self.rng.randint(2, 8)
        deadline = f"2024-{self.rng.randint(1,12):02d}-{self.rng.randint(1,28):02d}"
        contact = f"careers@{company.lower().replace(' ', '')}.com"

        def fill(t: str) -> str:
            return (t.replace("{inv}", inv).replace("{date}", date)
                    .replace("{company}", company).replace("{amount}", amount)
                    .replace("{terms}", str(terms)).replace("{email}", email)
                    .replace("{party_a}", party_a).replace("{party_b}", party_b)
                    .replace("{duration}", str(duration)).replace("{value}", value)
                    .replace("{state}", state).replace("{scope}", scope)
                    .replace("{role}", role).replace("{city}", city)
                    .replace("{salary}", salary).replace("{years}", str(years))
                    .replace("{deadline}", deadline).replace("{contact}", contact))

        doc = fill(template)
        gt = {k: fill(v) for k, v in gt_template.items()}
        prompt = f"Extract the following fields as JSON from the document below:\n{list(gt.keys())}\n\nDocument: {doc}"
        ground_truth = json.dumps(gt)
        complexity = 0.5 + self.np_rng.uniform(-0.1, 0.2)
        complexity = float(np.clip(complexity, 0.3, 0.85))
        return self._build_record(f"ext_{idx:03d}", "extraction", prompt, ground_truth, complexity)

    # --- Common builder ------------------------------------------------------

    def _build_record(self, pid: str, task_type: str, prompt: str,
                      ground_truth: str, complexity: float) -> PromptRecord:
        base_quality = QUALITY_BASELINES[task_type]
        tiers = list(MODEL_CONFIGS.keys())
        sim_quality: Dict[str, float] = {}
        sim_latency: Dict[str, int] = {}
        sim_cost: Dict[str, float] = {}
        sim_tokens: Dict[str, int] = {}

        prompt_tokens = len(prompt.split()) * 1.3  # rough token estimate

        for tier in tiers:
            cfg = MODEL_CONFIGS[tier]
            # quality varies slightly with complexity (harder tasks expose model gaps more)
            q_base = base_quality[tier]
            q_noise = self.np_rng.uniform(-0.02, 0.02)
            complexity_penalty = (complexity - 0.5) * 0.05  # harder → larger gap vs large
            if tier != "large":
                q = float(np.clip(q_base + q_noise - complexity_penalty * 0.3, 0.0, 1.0))
            else:
                q = float(np.clip(q_base + q_noise, 0.0, 1.0))
            sim_quality[tier] = round(q, 4)

            # tokens
            inp_tokens = int(prompt_tokens)
            out_tokens = self.rng.randint(30, 200)
            sim_tokens[tier] = inp_tokens + out_tokens

            # cost
            sim_cost[tier] = round(cfg.estimate_cost(inp_tokens, out_tokens), 6)

            # latency
            lat_base = cfg.avg_latency_ms
            lat_jitter = self.rng.randint(-50, 100)
            sim_latency[tier] = max(50, lat_base + lat_jitter)

        return PromptRecord(
            prompt_id=pid,
            task_type=task_type,
            prompt=prompt,
            ground_truth=ground_truth,
            complexity_score=round(float(complexity), 4),
            expected_tokens=sim_tokens,
            simulated_quality=sim_quality,
            simulated_latency_ms=sim_latency,
            simulated_cost_usd=sim_cost,
        )

    # patch class-level references to module-level lists
    _SUMMARY_ARTICLES = _SUMMARY_ARTICLES
    _CLASSIFICATION_EXAMPLES = _CLASSIFICATION_EXAMPLES
    _EXTRACTION_TEMPLATES = _EXTRACTION_TEMPLATES
    _COMPANIES = _COMPANIES
    _ROLES = _ROLES
    _CITIES = _CITIES
    _STATES = _STATES
    _SCOPES = _SCOPES

    # --- Public API ----------------------------------------------------------

    def generate(self) -> List[PromptRecord]:
        records: List[PromptRecord] = []
        for i in range(self.n_per_task):
            records.append(self._make_summarization_record(i))
        for i in range(self.n_per_task):
            records.append(self._make_classification_record(i))
        for i in range(self.n_per_task):
            records.append(self._make_extraction_record(i))
        return records

    def generate_and_save(self, output_path: Optional[Path] = None) -> Path:
        records = self.generate()
        if output_path is None:
            output_path = Path(__file__).parent / "synthetic_dataset.jsonl"
        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")
        print(f"Saved {len(records)} records to {output_path}")
        return output_path


def load_dataset(path: Optional[Path] = None) -> List[PromptRecord]:
    if path is None:
        path = Path(__file__).parent / "synthetic_dataset.jsonl"
    if not path.exists():
        print("Dataset not found — generating...")
        SyntheticDataGenerator().generate_and_save(path)
    records = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            records.append(PromptRecord(**d))
    return records


if __name__ == "__main__":
    gen = SyntheticDataGenerator()
    gen.generate_and_save()
