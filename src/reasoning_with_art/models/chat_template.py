"""Shared chat-template helpers.

``enable_thinking`` is a Qwen3-family ``apply_chat_template`` kwarg. Other VL
families (Qwen2-VL) do not define it; forwarding
it is at best silently ignored and at worst raises. We forward it only when the
processor's chat template actually references it, detected once per processor.

This module deliberately imports nothing heavy (no torch/vllm/transformers), so
it is safe to import at module top regardless of the CUDA_VISIBLE_DEVICES /
lazy-import ordering the optimizers and clients rely on.
"""

import logging

logger = logging.getLogger(__name__)

# Keyed by id(processor); processors are long-lived singletons within a run.
_THINKING_SUPPORT_CACHE: dict[int, bool] = {}


def _chat_template_string(processor) -> str | None:
    """Best-effort extraction of the Jinja chat-template string.

    The attribute lives in different places across HF processor/tokenizer
    classes, so check the processor itself and its inner tokenizer.
    """
    for obj in (processor, getattr(processor, "tokenizer", None)):
        if obj is None:
            continue
        tmpl = getattr(obj, "chat_template", None)
        if isinstance(tmpl, str):
            return tmpl
    return None


def template_supports_thinking(processor) -> bool:
    """True iff the processor's chat template references ``enable_thinking``
    (the Qwen3 family). Cached per processor instance."""
    key = id(processor)
    cached = _THINKING_SUPPORT_CACHE.get(key)
    if cached is not None:
        return cached
    tmpl = _chat_template_string(processor)
    supported = bool(tmpl) and "enable_thinking" in tmpl
    _THINKING_SUPPORT_CACHE[key] = supported
    if not supported:
        logger.debug(
            "Processor %s chat template does not reference enable_thinking; the kwarg will not be forwarded.",
            type(processor).__name__,
        )
    return supported


def apply_chat_template(processor, conversation, *, enable_thinking=None, **kwargs):
    """``processor.apply_chat_template`` that forwards ``enable_thinking`` only
    when the template supports it. ``conversation`` is passed positionally so
    this works whether the underlying param is named ``conversation`` or
    ``messages``; all other kwargs pass through unchanged.
    """
    if enable_thinking is not None and template_supports_thinking(processor):
        kwargs["enable_thinking"] = enable_thinking
    return processor.apply_chat_template(conversation, **kwargs)
