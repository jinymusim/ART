import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import yaml

# Ensure all dataset classes are registered via import side-effects
import reasoning_with_art.datasets.gsm8k  # noqa: F401
import reasoning_with_art.datasets.svamp  # noqa: F401
from reasoning_with_art.config import (
    MLFLOW_EVAL_EXPERIMENT,
    ExperimentConfig,
    ModalityConfig,
    _render_template,
    benchmark_overrides,
    benchmark_slug,
    derive_eval_method,
    model_slug,
)
from reasoning_with_art.datasets.base import DATASET_REGISTRY
from reasoning_with_art.modalities.image import ImageProvider
from reasoning_with_art.models.local_client import LocalClient

logger = logging.getLogger(__name__)


def _pct(data: list[float], p: float) -> float:
    """Linear-interpolation percentile on a list (does not require pre-sorting)."""
    if not data:
        return 0.0
    sd = sorted(data)
    pos = (len(sd) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sd) - 1)
    return sd[lo] + (sd[hi] - sd[lo]) * (pos - lo)


def _resolve_image_path(modality_cfg, model_cfg, benchmark_name: str) -> str | None:
    """Return the per-cell image checkpoint path.

    An explicit per-benchmark `images` map (if set) takes precedence: it lets each
    benchmark use a different PNG whose path doesn't follow a {benchmark_slug} pattern.
    """
    images_map = getattr(modality_cfg, "images", None)
    if images_map:
        if benchmark_name not in images_map:
            raise ValueError(f"modality.images has no entry for benchmark {benchmark_name!r}. Mapped benchmarks: {sorted(images_map)}")
        return images_map[benchmark_name]
    if modality_cfg.path_template:
        return _render_template(
            modality_cfg.path_template,
            model_cfg,
            benchmark_name,
        )
    return modality_cfg.path


def _build_model(model_cfg):
    """Build a model client."""
    if model_cfg.backend == "local":
        return LocalClient(
            model_id=model_cfg.model_id,
            device=model_cfg.device,
            dtype=model_cfg.dtype,
            max_new_tokens=model_cfg.max_tokens,
            max_model_len=model_cfg.max_model_len,
            batch_size=model_cfg.batch_size,
            sampling_params=model_cfg.sampling_params,
            vllm_kwargs=model_cfg.vllm_kwargs,
            enable_thinking=model_cfg.enable_thinking,
        )
    else:
        raise ValueError(f"Unknown model backend: {model_cfg.backend}")


def _build_image_provider(modality_cfg, image_path: str | None) -> ImageProvider:
    if modality_cfg.type == "image":
        h, w = modality_cfg.image_size
        return ImageProvider(strategy=modality_cfg.strategy, image_path=image_path, width=w, height=h)
    return ImageProvider(strategy="none")


def run(config: ExperimentConfig) -> dict:
    """Run the full evaluation pipeline. Returns results dict."""
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EVAL_EXPERIMENT)

    all_results = {}
    results_dir = Path("results") / config.config_name
    results_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in config.models:

        # Built lazily on the first valid benchmark, since the initial artifact path
        # may itself be benchmark-specific.
        shared_model = None

        for benchmark_name in config.benchmarks:
            if benchmark_name not in DATASET_REGISTRY:
                logger.warning(f"Unknown benchmark: {benchmark_name}, skipping")
                continue


            if shared_model is None:
                shared_model = _build_model(model_cfg)
            model = shared_model


            cell_image_path = _resolve_image_path(config.modality, model_cfg, benchmark_name)
            image_provider = _build_image_provider(config.modality, cell_image_path)
            if cell_image_path:
                logger.info(f"[{model_cfg.name}][{benchmark_name}] Using image: {cell_image_path}")

            dataset = DATASET_REGISTRY[benchmark_name]()
            logger.info(f"Loading dataset: {benchmark_name} (split={config.split})")
            benchmark_num_samples = (config.num_samples or {}).get(benchmark_name)
            examples = dataset.load(num_samples=benchmark_num_samples, split=config.split)
            logger.info(f"Loaded {len(examples)} examples from {benchmark_name}")

            # Per-benchmark token/batch overrides (fall back to model defaults).
            # Applied at generate-time / loop-time so the vLLM engine is not
            # rebuilt per benchmark. max_prompt_tokens is optimize-only (eval
            # does no prompt filtering beyond the engine max_model_len preflight).
            ov = benchmark_overrides(config.benchmark_settings, benchmark_name)
            cell_max_new_tokens = ov.get("max_new_tokens", model_cfg.max_tokens)
            cell_batch_size = ov.get("batch_size", model_cfg.batch_size)

            method = derive_eval_method(config.modality)
            run_name = f"{method}/{model_slug(model_cfg)}/{benchmark_slug(benchmark_name)}"

            with mlflow.start_run(run_name=run_name):
                # Log full config as artifact
                config_artifact = results_dir / "config.yaml"
                config_artifact.write_text(yaml.dump(asdict(config), default_flow_style=False))
                mlflow.log_artifact(str(config_artifact))

                tags = {
                    "method": method,
                    "model": model_cfg.model_id,
                    "model_backend": model_cfg.backend,
                    "benchmark": benchmark_name,
                    "modality": config.modality.type,
                    "split": config.split,
                    "config_name": config.config_name,
                }
                if config.description:
                    # MLflow renders this as the run's "Description" in the UI.
                    tags["mlflow.note.content"] = config.description
                mlflow.set_tags(tags)

                # Log the dataset as a tracked input (replaces the `benchmark`
                # string param — the Datasets filter in the MLflow UI now works).
                try:
                    mlflow.log_input(
                        dataset.as_mlflow_dataset(examples, split=config.split),
                        context="evaluation",
                    )
                except Exception as e:
                    logger.warning(f"mlflow.log_input failed for {benchmark_name}: {e}")

                mlflow.log_params(
                    {
                        "modality_strategy": config.modality.strategy,
                        "image_path": cell_image_path or "",
                        "num_samples": len(examples),
                        "max_tokens": cell_max_new_tokens,
                    }
                )

                per_example = []
                total_score = 0.0
                batch_size = cell_batch_size

                # Accumulated across batches for aggregate GPU metrics at run end.
                _batch_latencies_ms: list[float] = []
                _batch_toks_per_s: list[float] = []
                _batch_sm_util_p95: list[float] = []

                for batch_start in range(0, len(examples), batch_size):
                    batch = examples[batch_start : batch_start + batch_size]
                    batch_indices = list(range(batch_start, batch_start + len(batch)))

                    prompts = [ex["question"] for ex in batch]
                    images = [image_provider.get_image(i) for i in batch_indices]

                    start = time.time()
                    try:
                        responses = model.generate_batch(prompts, images=images, max_new_tokens=cell_max_new_tokens)
                    except Exception as e:
                        logger.error(f"Batch generation failed at {batch_start}: {e}")
                        responses = [""] * len(batch)
                    elapsed = time.time() - start
                    per_item_time = round(elapsed / len(batch), 2)

                    # Collect GPU stats from LocalClient instrumentation (if present).
                    # Uses getattr so API models and other backends are unaffected.
                    batch_stats: dict = getattr(model, "last_batch_stats", None) or {}
                    gpu_extras: dict = {}
                    if "latency_ms" in batch_stats:
                        gpu_extras["batch_latency_ms"] = round(batch_stats["latency_ms"], 2)
                        _batch_latencies_ms.append(batch_stats["latency_ms"])
                    if "tok_per_s_total" in batch_stats:
                        gpu_extras["batch_tok_per_s"] = round(batch_stats["tok_per_s_total"], 1)
                        _batch_toks_per_s.append(batch_stats["tok_per_s_total"])
                    if "peak_vram_delta_bytes" in batch_stats:
                        gpu_extras["batch_peak_vram_mb"] = round(batch_stats["peak_vram_delta_bytes"] / 1024**2, 2)
                    if "sm_util_p95" in batch_stats:
                        gpu_extras["batch_sm_util_p95"] = round(batch_stats["sm_util_p95"], 1)
                        _batch_sm_util_p95.append(batch_stats["sm_util_p95"])
                    if "sm_util_mean" in batch_stats:
                        gpu_extras["batch_sm_util_mean"] = round(batch_stats["sm_util_mean"], 1)

                    # Log batch-level GPU metrics once per batch (at batch_start step).
                    if gpu_extras:
                        mlflow.log_metrics(gpu_extras, step=batch_start)

                    for i, (example, response) in enumerate(zip(batch, responses)):
                        idx = batch_start + i
                        score = dataset.score(example, response)
                        total_score += score

                        per_example.append(
                            {
                                "idx": idx,
                                "question": example["question"],
                                "ground_truth": example["ground_truth"],
                                "task": example.get("metadata", {}).get("task", ""),
                                "model_output": response,
                                "score": score,
                                "time_s": per_item_time,
                                # GPU extras replicated across each example in the batch
                                # (same convention as time_s, which is already batch-averaged).
                                **gpu_extras,
                            }
                        )

                        # Log step-level metrics for MLflow charts
                        mlflow.log_metrics(
                            {
                                "running_score": total_score / (idx + 1),
                                "example_score": score,
                                "example_time_s": per_item_time,
                            },
                            step=idx,
                        )

                    processed = batch_start + len(batch)
                    if processed % 10 < batch_size or processed == len(examples):
                        logger.info(f"  [{model.name}][{benchmark_name}] {processed}/{len(examples)} — running score: {total_score / processed:.4f}")

                avg_score = total_score / len(examples) if examples else 0.0
                avg_time = sum(e["time_s"] for e in per_example) / len(per_example) if per_example else 0.0

                # Aggregate GPU metrics across all batches.
                agg_gpu: dict = {}
                if _batch_latencies_ms:
                    agg_gpu["latency_ms_p50"] = round(_pct(_batch_latencies_ms, 50), 2)
                    agg_gpu["latency_ms_p95"] = round(_pct(_batch_latencies_ms, 95), 2)
                    agg_gpu["latency_ms_mean"] = round(sum(_batch_latencies_ms) / len(_batch_latencies_ms), 2)
                if _batch_toks_per_s:
                    agg_gpu["tok_per_s_mean"] = round(sum(_batch_toks_per_s) / len(_batch_toks_per_s), 1)
                if _batch_sm_util_p95:
                    agg_gpu["sm_util_p95_mean"] = round(sum(_batch_sm_util_p95) / len(_batch_sm_util_p95), 1)

                mlflow.log_metrics(
                    {
                        "score": avg_score,
                        "total_score": total_score,
                        "num_total": len(examples),
                        "avg_time_per_example_s": avg_time,
                        **agg_gpu,
                    }
                )

                # Save per-example results as artifact
                artifact_path = results_dir / f"{run_name.replace('/', '_')}.json"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_text(json.dumps(per_example, indent=2))
                mlflow.log_artifact(str(artifact_path))

                result_key = f"{model.name}__{benchmark_name}"
                all_results[result_key] = {
                    "score": avg_score,
                    "total_score": total_score,
                    "total": len(examples),
                }

                logger.info(f"  [{model.name}][{benchmark_name}] Final score: {avg_score:.4f} ({total_score:.1f}/{len(examples)})")

    return all_results


def print_summary(results: dict):
    """Print a formatted summary table of results."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Benchmark':<20} {'Score':>10}")
    print("-" * 70)
    for key, res in results.items():
        parts = key.split("__", 1)
        model_name = parts[0] if parts else key
        bench_name = parts[1] if len(parts) > 1 else ""
        score_str = f"{res['score']:.4f} ({res['total_score']:.1f}/{res['total']})"
        print(f"{model_name:<30} {bench_name:<20} {score_str:>10}")
    print("=" * 70 + "\n")
