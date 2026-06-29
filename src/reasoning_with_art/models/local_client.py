import hashlib
import io
import json
import logging
import re

import torch
from PIL import Image
from vllm import LLM
from vllm import SamplingParams
from reasoning_with_art.tools.instrumentation import (
    get_physical_device_index,
    measure_inference,
)
from reasoning_with_art.models.base import ModelClient, register_model
from reasoning_with_art.models.chat_template import apply_chat_template

logger = logging.getLogger(__name__)

# vLLM's layerwise reloader emits a WARNING per submodule on every wake_up()
logging.getLogger("vllm.model_executor.model_loader.reload.layerwise").setLevel(logging.ERROR)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def select_dtype_for_hardware() -> torch.dtype:
    """Return float16 or bfloat16 based on the minimum compute capability of visible GPUs.

    bfloat16 has native hardware support on Ampere (sm_80) and newer.
    Older GPUs (Volta sm_70, Turing sm_75) emulate bfloat16 at float32 speed,
    so float16 is the correct choice there.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return torch.float16
    min_major = min(torch.cuda.get_device_properties(i).major for i in range(torch.cuda.device_count()))
    chosen = torch.bfloat16 if min_major >= 8 else torch.float16
    logger.info(f"GPU min compute capability: sm_{min_major}x → using {chosen}")
    return chosen


def _dtype_str_for_hardware() -> str:
    """Return dtype as string ('float16'/'bfloat16') for vLLM configuration."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return "float16"
    min_major = min(torch.cuda.get_device_properties(i).major for i in range(torch.cuda.device_count()))
    return "bfloat16" if min_major >= 8 else "float16"


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output (e.g. Qwen3.5 reasoning)."""
    return _THINK_RE.sub("", text).strip()


@register_model("local")
class LocalClient(ModelClient):
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 32768,
        max_model_len: int | None = None,
        batch_size: int = 8,
        sampling_params: dict | None = None,
        vllm_kwargs: dict | None = None,
        enable_thinking: bool = False,
    ):
        self._model_id = model_id
        self._device = device
        self._dtype = dtype
        self._max_new_tokens = max_new_tokens
        self._max_model_len = max_model_len
        self._batch_size = batch_size
        self._sampling_params = sampling_params or {}
        self._vllm_kwargs = vllm_kwargs or {}
        self._enable_thinking = enable_thinking
        self._llm = None
        self._processor = None
        self._effective_max_model_len: int | None = None
        self._engine_kwargs_hash: str | None = None
        # Populated after every generate_batch call; read by pipeline for MLflow logging.
        self.last_batch_stats: dict | None = None
        # Resolved engine state (enforce_eager, CUDA-graph capture sizes, etc.)
        # populated once the engine has loaded; read by the benchmark harness.
        self.engine_info: dict = {}
        # Per-image preflight token budget, measured lazily from the processor
        # on first use (model-specific: Qwen ~64). See
        # _measure_image_token_allowance.
        self._image_token_allowance: int | None = None

    @property
    def name(self) -> str:
        return f"local/{self._model_id}"

    def _load_model(self):
        if self._llm is not None:
            return

        logger.info(f"Loading model with vLLM: {self._model_id}")

        dtype = _dtype_str_for_hardware() if self._dtype == "auto" else self._dtype

        llm_kwargs = dict(
            model=self._model_id,
            dtype=dtype,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
        )
        if self._max_model_len is not None:
            llm_kwargs["max_model_len"] = self._max_model_len
        # Override/extend with user-provided vLLM engine kwargs
        llm_kwargs.update(self._vllm_kwargs)

        # Fingerprint the effective engine config so the benchmark harness can
        # verify that two cells were run with comparable engine settings.
        self._engine_kwargs_hash = hashlib.md5(json.dumps({k: str(v) for k, v in sorted(llm_kwargs.items())}, sort_keys=True).encode()).hexdigest()[:8]

        self._llm = LLM(**llm_kwargs)

        self._effective_max_model_len = self._llm.llm_engine.model_config.max_model_len

        # Capture resolved engine state for the benchmark harness to log.
        try:
            mc = self._llm.llm_engine.model_config
            self.engine_info["enforce_eager"] = bool(getattr(mc, "enforce_eager", False))
        except Exception as exc:
            logger.debug(f"Could not read enforce_eager: {exc}")
        try:
            vc = getattr(self._llm.llm_engine, "vllm_config", None)
            comp = getattr(vc, "compilation_config", None) if vc is not None else None
            if comp is not None:
                sizes = getattr(comp, "cudagraph_capture_sizes", None) or getattr(comp, "capture_sizes", None)
                if sizes:
                    self.engine_info["cudagraph_capture_sizes"] = list(sizes)
                level = getattr(comp, "level", None)
                if level is not None:
                    self.engine_info["compilation_level"] = str(level)
        except Exception as exc:
            logger.debug(f"Could not read compilation_config: {exc}")
        logger.info(
            f"vLLM model ready — dtype={dtype}, "
            f"enable_thinking={self._enable_thinking}, "
            f"max_new_tokens={self._max_new_tokens}, "
            f"effective_max_model_len={self._effective_max_model_len}, "
            f"engine_info={self.engine_info}"
        )

    def _build_sampling_params(self, max_new_tokens: int | None = None):
        """Build vLLM SamplingParams directly from config.

        ``max_new_tokens`` overrides the client default for this call (used by
        the eval pipeline to apply per-benchmark output budgets without
        rebuilding the engine).
        """
        kwargs = {"max_tokens": max_new_tokens or self._max_new_tokens}
        kwargs.update(self._sampling_params)
        return SamplingParams(**kwargs)

    def _get_processor(self):
        """Lazy-load the HF processor used to render the chat template."""
        if self._processor is None:
            from reasoning_with_art.models.model_loading import load_processor

            self._processor = load_processor(self._model_id)
        return self._processor

    def _render_prompt(self, question: str, has_image: bool) -> str:
        """Render the chat template with a list-content user message.

        Text-only and multimodal calls go through the same Jinja branch — only
        difference is whether an image placeholder is in the content list.
        """
        content = []
        if has_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        return apply_chat_template(
            self._get_processor(),
            messages,
            enable_thinking=self._enable_thinking,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Conservative fallback for the per-image token budget when it cannot be
    # measured from the processor (used only if _measure_image_token_allowance
    # raises). Large enough to cover multi-tile preprocessing without being so
    # large it rejects every prompt.
    _IMAGE_TOKEN_ALLOWANCE_FALLBACK = 256
    # Margin reserved below max_model_len so accepted prompts have room for
    # at least a few output tokens.
    _PREFLIGHT_MARGIN = 32

    def _measure_image_token_allowance(self) -> int:
        """Measure the per-image token contribution once, model-agnostically.

        Image preprocessing inserts a model-specific number of placeholder
        tokens (Qwen-VL ~64 at 256px). The chat-template
        *string* only carries a
        single placeholder, so we must run the full processor with an image and
        diff against the text-only token count. Any mm_processor_kwargs the
        engine uses (e.g. crop_to_patches) are mirrored so the count matches.
        """
        try:
            processor = self._get_processor()
            mm_kwargs = self._vllm_kwargs.get("mm_processor_kwargs") or {}
            dummy = Image.new("RGB", (256, 256), color=(128, 128, 128))
            img_msgs = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": "x"}],
                }
            ]
            txt_msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
            img_rendered = apply_chat_template(
                processor,
                img_msgs,
                enable_thinking=self._enable_thinking,
                tokenize=False,
                add_generation_prompt=True,
            )
            txt_rendered = apply_chat_template(
                processor,
                txt_msgs,
                enable_thinking=self._enable_thinking,
                tokenize=False,
                add_generation_prompt=True,
            )
            n_text = self._count_prompt_tokens(txt_rendered)
            full = processor(text=img_rendered, images=[dummy], return_tensors="pt", **mm_kwargs)
            n_full = full["input_ids"].shape[-1]
            measured = max(0, n_full - n_text)
            # Cross-check against image_seq_length if the processor exposes it.
            seq = getattr(processor, "image_seq_length", None)
            allowance = max(measured, int(seq) if seq else 0)
            if allowance <= 0:
                raise ValueError("measured allowance was 0")
            return allowance
        except Exception as exc:
            logger.warning(f"Could not measure image-token allowance ({exc}); using fallback {self._IMAGE_TOKEN_ALLOWANCE_FALLBACK}.")
            return self._IMAGE_TOKEN_ALLOWANCE_FALLBACK

    def _get_image_token_allowance(self) -> int:
        if self._image_token_allowance is None:
            self._image_token_allowance = self._measure_image_token_allowance()
            logger.info(f"  Image-token allowance: {self._image_token_allowance}")
        return self._image_token_allowance

    def image_token_allowance(self) -> int:
        """Public accessor for the measured per-image vision-token count.

        Used to size the W3 random-string control so its token budget matches the
        W2 image / soft-prompt budget for this model. Ensures the model is loaded.
        """
        self._load_model()
        return self._get_image_token_allowance()

    def _count_prompt_tokens(self, rendered: str) -> int:
        tokenizer = self._get_processor().tokenizer
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    def generate_batch(
        self,
        prompts: list[str],
        images: list[bytes | None] | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate responses for a batch of prompts using vLLM.

        Prompts that would exceed the engine's max_model_len are filtered out
        pre-flight (empty string returned for those indices), so one overlong
        prompt cannot poison the rest of a batch.

        ``max_new_tokens`` overrides the client default output budget for this
        call (per-benchmark eval override); None uses the client default.
        """
        self._load_model()
        images = images or [None] * len(prompts)

        cap = self._effective_max_model_len
        results: list[str | None] = [None] * len(prompts)
        kept_requests: list[dict] = []
        kept_indices: list[int] = []
        for i, (prompt, img_bytes) in enumerate(zip(prompts, images)):
            has_image = img_bytes is not None
            rendered = self._render_prompt(prompt, has_image=has_image)
            prompt_tokens = self._count_prompt_tokens(rendered)
            image_tokens = self._get_image_token_allowance() if has_image else 0
            budget = prompt_tokens + image_tokens + self._PREFLIGHT_MARGIN
            if cap is not None and budget >= cap:
                logger.warning(
                    f"  Skipping row {i}: prompt_tokens={prompt_tokens}"
                    + (f" +image_tokens~{image_tokens}" if has_image else "")
                    + f" +margin={self._PREFLIGHT_MARGIN} >= max_model_len={cap}"
                )
                results[i] = ""
                continue
            if has_image:
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                kept_requests.append({"prompt": rendered, "multi_modal_data": {"image": pil}})
            else:
                kept_requests.append({"prompt": rendered})
            kept_indices.append(i)

        sampling_params = self._build_sampling_params(max_new_tokens=max_new_tokens)
        effective_max_tokens = max_new_tokens or self._max_new_tokens
        n_images = sum(1 for img in images if img is not None)
        n_skipped = len(prompts) - len(kept_requests)
        logger.info(
            f"  Generating batch: {len(kept_requests)}/{len(prompts)} items"
            + (f" ({n_skipped} skipped pre-flight)" if n_skipped else "")
            + f", max_tokens={effective_max_tokens}, images={n_images}"
        )

        if not kept_requests:
            self.last_batch_stats = None
            return [r or "" for r in results]

        phys_device = get_physical_device_index(logical_index=0)
        with measure_inference(physical_device_index=phys_device) as _stats:
            outputs = self._llm.generate(
                kept_requests,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            # Populate token counts inside the context so tok_per_s is derived correctly.
            _stats["gen_tokens"] = sum(len(o.outputs[0].token_ids) for o in outputs)
            try:
                _stats["prompt_tokens"] = sum(len(o.prompt_token_ids) for o in outputs)
            except Exception:
                pass  # prompt_token_ids may not always be populated

        # ── Per-request TTFT and decode-per-token from vLLM RequestOutput.metrics.
        ttfts_ms: list[float] = []
        decode_ms_per_tok: list[float] = []
        for o in outputs:
            m = getattr(o, "metrics", None)
            if m is None:
                continue
            arrival = getattr(m, "arrival_time", None)
            first_tok = getattr(m, "first_token_time", None)
            last_tok = getattr(m, "last_token_time", None) or getattr(m, "finished_time", None)
            if arrival is not None and first_tok is not None:
                ttfts_ms.append((first_tok - arrival) * 1000.0)
            n_dec = len(o.outputs[0].token_ids)
            if first_tok is not None and last_tok is not None and n_dec > 1:
                decode_ms_per_tok.append((last_tok - first_tok) * 1000.0 / (n_dec - 1))
        if ttfts_ms:
            _stats["ttft_ms_mean"] = sum(ttfts_ms) / len(ttfts_ms)
        if decode_ms_per_tok:
            _stats["decode_ms_per_tok_mean"] = sum(decode_ms_per_tok) / len(decode_ms_per_tok)

        total_gen = _stats.get("gen_tokens", 0)
        elapsed_s = (_stats.get("latency_ms") or 0) / 1000.0 or 1.0
        logger.info(
            f"  Done: generated {total_gen} tokens in {elapsed_s:.1f}s "
            f"({total_gen / elapsed_s:.1f} tok/s)" + (f", lat={_stats['latency_ms']:.0f}ms" if "latency_ms" in _stats else "")
        )

        # Attach batch-level metadata and store for the pipeline to consume.
        _stats.update(
            {
                "batch_size": len(kept_requests),
                "n_images": n_images,
                "n_skipped": n_skipped,
                "engine_kwargs_hash": self._engine_kwargs_hash or "",
            }
        )
        self.last_batch_stats = _stats

        for idx, output in zip(kept_indices, outputs):
            results[idx] = strip_thinking(output.outputs[0].text)

        return [r or "" for r in results]
