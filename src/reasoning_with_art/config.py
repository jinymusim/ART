import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# MLflow experiment name. All evaluation runs land here so the MLflow UI can
# compare modalities/methods side by side. The per-config `description` field
# provides the human-readable label via the run's mlflow.note.content tag.
# ════════════════════════════════════════════════════════════════════════════
MLFLOW_EVAL_EXPERIMENT = "reasoning-with-art/eval"


# ════════════════════════════════════════════════════════════════════════════
# ModalityConfig — what goes into the prompt alongside the question.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ModalityConfig:
    # "none" or "image"
    type: str = "none"

    # ── Image-only ────────────────────────────────────────────────────────────
    # Literal path to a PNG (single-cell).
    path: str | None = None
    # Per-cell template
    path_template: str | None = None
    # Explicit per-benchmark image map {benchmark_name: image_path}.
    images: dict[str, str] | None = None
    # "none" or "learned"
    strategy: str = "learned"
    # Image resolution (height, width). Must match the image_size the artifact was
    # trained at. Default 256×256; configurable.
    image_size: list[int] = field(default_factory=lambda: [256, 256])


# ════════════════════════════════════════════════════════════════════════════
# ModelConfig — one model entry in a sweep. Mixes API-only, local-only, and
# shared fields. The pipeline picks the relevant subset per backend.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    # ── Identity ────────────────────────────────────────────────────────────
    name: str = ""
    backend: str = "local"  # only "local" backend is supported

    # ── Local backend (HuggingFace + vLLM) ──────────────────────────────────
    model_id: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 8  # inference batch size (local only)

    # ── Inference (shared) ──────────────────────────────────────────────────
    max_tokens: int = 32768
    # Whether the model reasons in <think> blocks before answering.
    enable_thinking: bool = False

    # ── Sampling (local — vLLM SamplingParams) ──────────────────────────────
    # Passed directly to vllm.SamplingParams(**sampling_params).
    sampling_params: dict = field(default_factory=dict)

    # ── vLLM engine kwargs (local) ──────────────────────────────────────────
    # vLLM engine's total sequence length cap (prompt + completion). 
    max_model_len: int | None = None
    # Passed directly to vllm.LLM(**vllm_kwargs).
    vllm_kwargs: dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# ExperimentConfig — eval entry point (run-experiment).
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ExperimentConfig:
    # Human-readable label for the run (becomes the MLflow run's mlflow.note.content tag).
    description: str = ""
    benchmarks: list[str] = field(default_factory=lambda: ["gsm8k"])
    models: list[ModelConfig] = field(default_factory=list)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    num_samples: dict[str, int] | None = None
    # Per-benchmark overrides of token/batch settings, keyed by benchmark name.
    benchmark_settings: dict[str, dict] | None = None
    split: str = "test"  # "train" or "test"
    mlflow_tracking_uri: str = "file:///bhome/chudobm/ART/mlruns"
    gpu_ids: list[int] | None = None
    config_name: str = ""


# Slug + template helpers — used by the pipeline to resolve {model_slug} and
# {benchmark_slug} placeholders in path templates.
# ════════════════════════════════════════════════════════════════════════════
def model_slug(m: ModelConfig) -> str:
    """e.g. ModelConfig(model_id='Qwen/Qwen3.5-0.8B') -> 'qwen3.5-0.8b'."""
    base = m.model_id
    return base.split("/")[-1].lower()


def benchmark_slug(b: str) -> str:
    """e.g. 'openai/gsm8k' -> 'openai_gsm8k'."""
    return b.replace("/", "_")


# Inner keys allowed in a benchmark_settings entry.
_BENCHMARK_OVERRIDE_KEYS = {
    "max_new_tokens",
    "max_prompt_tokens",
    "batch_size",
    "temperature",
}


def benchmark_overrides(settings: dict | None, benchmark: str) -> dict:
    """Per-benchmark override dict (empty if none). Mirrors the num_samples lookup."""
    return (settings or {}).get(benchmark, {})


def _validate_benchmark_settings(benchmark_settings: dict | None, active_benchmarks: list[str]) -> None:
    """Shared validation for the benchmark_settings field on both config types.

    Raises on a non-dict structure or an unknown inner key. Orphaned top-level
    keys (settings for a benchmark not in the active list) only warn, so a
    benchmark can be commented out without deleting its settings entry.
    """
    if benchmark_settings is None:
        return
    if not isinstance(benchmark_settings, dict):
        raise ValueError(
            f"benchmark_settings must be a dict[benchmark, dict], "
            f"got {type(benchmark_settings).__name__}. Example:\n"
            "  benchmark_settings:\n"
            "    gsm8k: {max_new_tokens: 2048}\n"
            "    svamp: {max_new_tokens: 512, batch_size: 8}"
        )
    for bench, ov in benchmark_settings.items():
        if not isinstance(ov, dict):
            raise ValueError(f"benchmark_settings[{bench!r}] must be a dict of overrides, got {type(ov).__name__}")
        unknown = set(ov) - _BENCHMARK_OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"benchmark_settings[{bench!r}] has unknown keys {sorted(unknown)}; allowed: {sorted(_BENCHMARK_OVERRIDE_KEYS)}")
    orphaned = set(benchmark_settings) - set(active_benchmarks)
    if orphaned:
        logger.warning(f"benchmark_settings has entries for inactive benchmarks {sorted(orphaned)} (not in {sorted(active_benchmarks)}); they will be ignored.")


def derive_eval_method(modality_cfg: "ModalityConfig") -> str:
    """Classify an eval-side modality cell into a comparable `method`.

    Used as the value of the MLflow `method` tag so the UI can filter
    baseline vs. learned-image runs across one shared experiment:
      - "baseline"      — no image
      - "learned_image" — an optimized PNG loaded via path_template
    """
    if modality_cfg.type == "image":
        return "learned_image"
    return "baseline"


def _render_template(template: str, model: ModelConfig, benchmark: str) -> str:
    """Substitute {model_slug}, {benchmark_slug}. Literal strings pass through."""
    try:
        return template.format(
            model_slug=model_slug(model),
            benchmark_slug=benchmark_slug(benchmark),
        )
    except KeyError as e:
        raise ValueError(f"Unknown placeholder {e} in template: {template!r}. Supported placeholders: {{model_slug}}, {{benchmark_slug}}") from e


# ════════════════════════════════════════════════════════════════════════════
# YAML loaders — parse a config file into the dataclasses above.
# ════════════════════════════════════════════════════════════════════════════
def _resolve_tracking_uri(raw: dict) -> str:
    """Resolve the MLflow tracking URI.

    The standard ``MLFLOW_TRACKING_URI`` env var takes precedence over the YAML
    value so SLURM scripts can redirect the SQLite DB to node-local disk (SQLite
    on BeeGFS hits ``sqlite3.OperationalError: disk I/O error`` because the
    network filesystem can't honor POSIX file locking). Falls back to the YAML
    value, then the default.
    """
    return os.environ.get("MLFLOW_TRACKING_URI") or raw.get("mlflow_tracking_uri", "file:///bhome/chudobm/ART/mlruns")


def load_config(path: str) -> ExperimentConfig:
    """Load experiment config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = []
    for m in raw.get("models", []):
        models.append(ModelConfig(**m))

    modality_raw = raw.get("modality", {})
    modality = ModalityConfig(**modality_raw) if modality_raw else ModalityConfig()

    benchmarks = raw.get("benchmarks", ["gsm8k"])
    num_samples = raw.get("num_samples")
    if num_samples is not None:
        if not isinstance(num_samples, dict):
            raise ValueError(
                f"num_samples must be a dict[str, int] keyed by benchmark name, got {type(num_samples).__name__}. Example: num_samples: {{gsm8k: 1000}}"
            )
        unknown = set(num_samples) - set(benchmarks)
        if unknown:
            raise ValueError(f"num_samples keys {sorted(unknown)} are not in benchmarks {benchmarks}")

    benchmark_settings = raw.get("benchmark_settings")
    _validate_benchmark_settings(benchmark_settings, benchmarks)

    return ExperimentConfig(
        description=raw.get("description", raw.get("experiment_name", "")),
        benchmarks=benchmarks,
        models=models,
        modality=modality,
        num_samples=num_samples,
        benchmark_settings=benchmark_settings,
        split=raw.get("split", "test"),
        mlflow_tracking_uri=_resolve_tracking_uri(raw),
        gpu_ids=raw.get("gpu_ids"),
        config_name=Path(path).stem,
    )
