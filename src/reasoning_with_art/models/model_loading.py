"""Resolve the right HF auto-class for a multimodal model, with a fallback.

Qwen2-VL registers under ``AutoModelForImageTextToText``
in recent transformers, but some ports historically lived under
``AutoModelForVision2Seq``. Try the modern class first, fall back, and raise a
clear error listing what was tried if neither maps.

transformers is imported lazily inside the function so this module stays safe to
import before CUDA_VISIBLE_DEVICES is set (matching the repo's lazy-import rule).
"""

import logging

logger = logging.getLogger(__name__)


def load_image_text_model(model_id: str, **from_pretrained_kwargs):
    """Load a VLM via AutoModelForImageTextToText, falling back to
    AutoModelForVision2Seq. Returns the loaded model or raises RuntimeError."""
    import transformers

    errors: list[str] = []
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **from_pretrained_kwargs)
            logger.info(f"Loaded {model_id} via {name} -> {type(model).__name__}")
            return model
        except Exception as exc:  # noqa: BLE001 - report every failed attempt
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not load {model_id} via AutoModelForImageTextToText or AutoModelForVision2Seq. Tried:\n  " + "\n  ".join(errors))


def load_processor(model_id: str, **kwargs):
    """Load a processor via AutoProcessor.from_pretrained, with custom overrides for specific models."""
    from transformers import AutoProcessor

    kwargs.setdefault("trust_remote_code", True)
    return AutoProcessor.from_pretrained(model_id, **kwargs)
