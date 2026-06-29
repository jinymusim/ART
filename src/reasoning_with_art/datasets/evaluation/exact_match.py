"""
Generic answer extraction and comparison utilities.

Provides extract_numeric / exact_match (GSM8K, SVAMP).
"""

import re
import string
import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_numeric(model_output: str, format: str = "Answer:") -> str | None:
    """Extract the final numeric answer from model output.

    Only accepts answers in the explicit format (e.g. "Answer: <number>").
    Returns the last occurrence of the format pattern, or None if not found.
    """
    pattern = r"(?i)" + re.escape(format) + r"\s*\$?\s*(-?[\d,\s]+(?:\.\d+)?)\s*\$?"
    matches = re.findall(pattern, model_output)
    if matches:
        return re.sub(r"[,\s]", "", matches[-1])
    return None


def extract_numeric_loose(model_output: str) -> str | None:
    """Extract the last number appearing anywhere in the output (no prefix required).

    Looser companion to ``extract_numeric``: used only by training-reward shaping
    to grant a tiny credit when the model produced *some* number but not in the
    required ``Answer: <number>`` format. Returns the cleaned numeric string
    (commas/spaces removed) or None if no number is present.
    """
    matches = re.findall(r"-?\d[\d,\s]*(?:\.\d+)?", model_output)
    if matches:
        return re.sub(r"[,\s]", "", matches[-1])
    return None


def extract_answer_text(model_output: str, format: str = "Answer:") -> str | None:
    """Extract a free-form answer string from model output (DROP, bAbI).

    Returns the text following the last ``format`` marker, up to the end of that
    line (so a trailing rationale on later lines is ignored). Surrounding quotes,
    markdown emphasis and a trailing period are stripped. Returns None when the
    marker is absent.
    """
    pattern = r"(?i)" + re.escape(format) + r"[ \t]*(.+)"
    matches = re.findall(pattern, model_output)
    if not matches:
        return None
    answer = matches[-1].strip().strip("\"'`*").strip()
    answer = answer.rstrip(".").strip()
    return answer or None


# ---------------------------------------------------------------------------
# Normalisation & comparison
# ---------------------------------------------------------------------------


def contains_valid_letter(model_output: str, valid_letters: str = "ABCD") -> bool:
    """Loosely detect a parenthesised choice letter (e.g. ``(B)``) anywhere.

    Used only by training-reward shaping to grant a tiny credit when the model
    gestured at a valid option but did not emit the required
    ``The correct answer is (X)`` format.
    """
    return re.search(r"\(\s*[" + valid_letters + r"]\s*\)", model_output, re.IGNORECASE) is not None


def relative_error(predicted: str, ground_truth: str) -> float | None:
    """Relative error ``|p - g| / max(|g|, eps)`` between two numeric strings.

    Returns None if either side does not parse as a number. Used by numeric
    training-reward shaping to award graded partial credit for near-misses.
    """
    try:
        p = float(predicted)
        g = float(ground_truth)
    except (ValueError, OverflowError):
        return None
    return abs(p - g) / max(abs(g), 1e-9)


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    text = answer.strip().lower()
    text = text.rstrip(".")
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"(\d)[,\s](\d)", r"\1\2", text)
    text = " ".join(text.split())
    return text


def numeric_equal(a: str, b: str) -> bool:
    """Check if two strings represent the same number."""
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, OverflowError):
        return False


def exact_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth after normalization."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    return numeric_equal(pred_norm, gt_norm)


