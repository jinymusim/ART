"""Shaped GRPO training rewards for the math benchmarks.

The eval ``score`` on every dataset is strictly binary (0/1) and stays that way —
reported accuracy must remain honest. These functions back the *training* reward
(``BenchmarkDataset.train_reward``) instead.

Why shape it: GRPO turns rewards into group-centred advantages
``adv = (r - group_mean) / (group_std + eps)``. When every completion in a group
earns the *same* reward (e.g. all wrong → all 0.0, common early in training), the
group variance is zero, the advantage is zero, and that group contributes no
gradient. Crediting answer-format compliance and numeric proximity gives groups
intra-group variance even before any completion is fully correct, so learning
gets a denser, faster signal.

Invariants every reward here upholds:
  * output is in ``[0, 1]``;
  * a fully-correct answer scores exactly ``1.0`` and strictly dominates every
    partial-credit value, so shaping never inverts the true objective.

Weights are intentionally hardcoded constants (reward shaping is dataset-intrinsic,
not a swept hyperparameter).
"""

# Correctly-formatted but wrong answer: floor reward for emitting a parseable answer.
# (Unparseable output scores 0 in eval, so teaching the format directly lifts accuracy.)
FORMAT_FLOOR = 0.1
# Numeric proximity bands (relative error -> reward) for formatted-but-wrong math answers.
NEAR_REL, NEAR_REWARD = 0.01, 0.4
MID_REL, MID_REWARD = 0.10, 0.2
# Tiny credit when the model produced *something* on-topic (a bare number / a bare
# choice letter) but not in the required answer format.
LOOSE_CREDIT = 0.01


def numeric_reward(model_output: str, ground_truth: str, format: str = "Answer:") -> float:
    """Shaped reward for numeric-answer benchmarks (gsm8k, svamp). Returns [0, 1].
    """
    from reasoning_with_art.datasets.evaluation.exact_match import (
        exact_match,
        extract_numeric,
        extract_numeric_loose,
        relative_error,
    )

    predicted = extract_numeric(model_output, format=format)
    if predicted is None:
        # No formatted answer; tiny credit if any number appears at all.
        return LOOSE_CREDIT if extract_numeric_loose(model_output) is not None else 0.0
    if exact_match(predicted, ground_truth):
        return 1.0
    rel = relative_error(predicted, ground_truth)
    if rel is not None and rel <= NEAR_REL:
        return NEAR_REWARD
    if rel is not None and rel <= MID_REL:
        return MID_REWARD
    return FORMAT_FLOOR


