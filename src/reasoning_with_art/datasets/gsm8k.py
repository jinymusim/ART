import re
from typing import cast

from datasets import load_dataset

from reasoning_with_art.datasets.base import BenchmarkDataset, register_dataset

GSM8K_ANSWER_SUFFIX = "\n\nEnd your response with: Answer: <number>"


@register_dataset("gsm8k")
class GSM8KDataset(BenchmarkDataset):
    @property
    def name(self) -> str:
        return "gsm8k"

    def load(self, num_samples: int | None = None, split: str = "test") -> list[dict]:
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split: {split}, expected 'train' or 'test'")
        ds = load_dataset("openai/gsm8k", "main", split=split)
        items = []
        for row in ds:
            sample = cast(dict[str, str], row)
            # Ground truth is the number after ####
            full_answer = sample["answer"]
            gt_match = re.search(r"####\s*(.+)", full_answer)
            ground_truth = gt_match.group(1).strip() if gt_match else full_answer

            # Gold CoT trace reformatted to end in "Answer: <number>" instead of
            # the dataset's "#### <number>" marker, matching the eval scorer
            # (extract_numeric(format="Answer:")) and GSM8K_ANSWER_SUFFIX. The
            # "####" is always the final line, so this rewrite leaves the
            # chain-of-thought body intact.
            answer_trace = re.sub(r"####\s*(.+?)\s*$", r"Answer: \1", full_answer.strip())

            items.append(
                {
                    "question": sample["question"] + GSM8K_ANSWER_SUFFIX,
                    "ground_truth": ground_truth,
                    "metadata": {"full_answer": full_answer, "answer_trace": answer_trace},
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
