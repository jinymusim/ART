"""SVAMP — simple arithmetic word problems.

Uses ``ChilleD/SVAMP`` (train 700 / test 300). Each row has a ``Body`` (context),
a ``Question``, and a numeric ``Answer``. The prompt reuses the GSM8K
``Answer: <number>`` contract so the same numeric extractor scores it.

Paper: Patel et al. NAACL 2021, "Are NLP Models really able to Solve Simple Math
Word Problems?" (arXiv:2103.07191).
"""

from datasets import load_dataset

from reasoning_with_art.datasets.base import BenchmarkDataset, register_dataset
from reasoning_with_art.datasets.gsm8k import GSM8K_ANSWER_SUFFIX


@register_dataset("svamp")
class SVAMPDataset(BenchmarkDataset):
    @property
    def name(self) -> str:
        return "svamp"

    def load(self, num_samples: int | None = None, split: str = "test") -> list[dict]:
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split: {split}, expected 'train' or 'test'")

        ds = load_dataset("ChilleD/SVAMP", split=split)

        items: list[dict] = []
        for row in ds:
            body = (row.get("Body") or "").strip()
            question = (row.get("Question") or "").strip()
            problem = f"{body} {question}".strip()

            items.append(
                {
                    "question": problem + GSM8K_ANSWER_SUFFIX,
                    "ground_truth": str(row["Answer"]).strip(),
                    "metadata": {"type": row.get("Type", ""), "equation": row.get("Equation", "")},
                }
            )
            if num_samples and len(items) >= num_samples:
                break
        return items

    def score(self, example: dict, model_output: str) -> float:
        from reasoning_with_art.datasets.evaluation.exact_match import (
            exact_match,
            extract_numeric,
        )

        predicted = extract_numeric(model_output, format="Answer:")
        if predicted is None:
            return 0.0
        return 1.0 if exact_match(predicted, example["ground_truth"]) else 0.0

    def train_reward(self, example: dict, model_output: str) -> float:
        from reasoning_with_art.datasets.evaluation.reward_shaping import numeric_reward

        return numeric_reward(model_output, example["ground_truth"], format="Answer:")
