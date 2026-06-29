"""
Process-isolated GPU instrumentation for vLLM inference benchmarking.

Three backends, all optional with graceful fallback:

  CudaEventTimer        — torch.cuda.Event timing (GPU-side, stream-local).
                          Not affected by other processes on the same device.

  PeakMemoryRegion      — torch allocator peak-memory delta.
                          Process-local by construction.

  NvmlProcessSampler    — background thread polling NVML per-process utilisation
                          via nvmlDeviceGetProcessUtilization, filtered to our PID.
                          Works on shared clusters because the NVML API is PID-scoped.

All three are bundled by measure_inference(), which yields a dict that is
populated with results on exit.  Any metric that cannot be measured (CUDA absent,
NVML not installed, driver doesn't expose per-process util) is simply absent from
the dict.

Invariants:
  - No project-internal imports.
  - Every backend degrades silently to "metric not present" rather than raising.
  - The measure_inference() context manager is re-entrant-safe (each call is
    independent).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ── NVML lazy-init ────────────────────────────────────────────────────────────

_nvml_available: bool | None = None
_nvml_lock = threading.Lock()


def _ensure_nvml() -> bool:
    """Try to initialise pynvml (once per process).  Returns True on success."""
    global _nvml_available
    if _nvml_available is not None:
        return _nvml_available
    with _nvml_lock:
        if _nvml_available is not None:
            return _nvml_available
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml_available = True
            logger.debug("pynvml initialised — per-process NVML metrics enabled.")
        except Exception as exc:
            logger.debug(f"pynvml unavailable ({exc}); per-process GPU util will be skipped.")
            _nvml_available = False
    return _nvml_available


# ── Physical device index helper ──────────────────────────────────────────────


def get_physical_device_index(logical_index: int = 0) -> int:
    """Map a logical CUDA device index to its physical NVML index.

    When CUDA_VISIBLE_DEVICES="2,3", logical device 0 is physical GPU 2.
    NVML always uses the physical index, so this conversion is required to
    poll the right device on shared clusters.

    UUID-based CUDA_VISIBLE_DEVICES values (used in some container setups) are
    passed through as-is and fall back to returning ``logical_index``.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible or cuda_visible.upper() == "NODEVICEFILES":
        return logical_index
    parts = [p.strip() for p in cuda_visible.split(",")]
    if logical_index < len(parts):
        part = parts[logical_index]
        try:
            return int(part)
        except ValueError:
            # UUID or MIG device — NVML lookup by UUID would be needed; skip
            logger.debug(f"CUDA_VISIBLE_DEVICES contains non-integer '{part}'; NVML will use device index 0.")
    return logical_index


# ── CudaEventTimer ────────────────────────────────────────────────────────────


class CudaEventTimer:
    """GPU-side latency timer using torch.cuda.Event.

    On entry the current CUDA stream is synchronised and a start event is
    recorded.  On exit a stop event is recorded, the stop event is
    synchronised (blocking until all GPU work enqueued so far finishes), and
    elapsed_ms is populated.

    The elapsed time covers only GPU-side work; CPU scheduling overhead
    between event record calls is excluded.

    Usage::

        timer = CudaEventTimer()
        with timer:
            outputs = llm.generate(requests)
        print(timer.elapsed_ms)   # None if CUDA unavailable
    """

    def __init__(self) -> None:
        self.elapsed_ms: float | None = None
        self._start = None
        self._stop = None

    def __enter__(self) -> CudaEventTimer:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self._start = torch.cuda.Event(enable_timing=True)
                self._stop = torch.cuda.Event(enable_timing=True)
                self._start.record()
        except Exception as exc:
            logger.debug(f"CudaEventTimer.__enter__ failed: {exc}")
        return self

    def __exit__(self, *args) -> bool:
        try:
            import torch

            if self._start is not None and torch.cuda.is_available():
                self._stop.record()
                self._stop.synchronize()  # blocks until GPU work is done
                self.elapsed_ms = self._start.elapsed_time(self._stop)
        except Exception as exc:
            logger.debug(f"CudaEventTimer.__exit__ failed: {exc}")
        return False


# ── PeakMemoryRegion ──────────────────────────────────────────────────────────


class PeakMemoryRegion:
    """Peak CUDA memory delta for THIS process during a code region.

    torch.cuda allocator bookkeeping is process-local, so these readings are
    not contaminated by other jobs on the same physical GPU.

    Usage::

        mem = PeakMemoryRegion()
        with mem:
            do_work()
        print(mem.peak_bytes, mem.delta_bytes)
    """

    def __init__(self) -> None:
        self.baseline_bytes: int = 0
        self.peak_bytes: int = 0  # peak since region start (absolute)
        self.delta_bytes: int = 0  # peak − baseline

    def __enter__(self) -> PeakMemoryRegion:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self.baseline_bytes = torch.cuda.memory_allocated()
        except Exception as exc:
            logger.debug(f"PeakMemoryRegion.__enter__ failed: {exc}")
        return self

    def __exit__(self, *args) -> bool:
        try:
            import torch

            if torch.cuda.is_available():
                self.peak_bytes = torch.cuda.max_memory_allocated()
                self.delta_bytes = max(0, self.peak_bytes - self.baseline_bytes)
        except Exception as exc:
            logger.debug(f"PeakMemoryRegion.__exit__ failed: {exc}")
        return False


# ── NvmlProcessSampler ────────────────────────────────────────────────────────


def _percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    if not data:
        return 0.0
    n = len(data)
    idx = (n - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


class NvmlProcessSampler:
    """Background thread polling NVML for *this process's* GPU utilisation.

    Uses ``nvmlDeviceGetProcessUtilization``, which returns per-PID samples.
    Filtering to ``os.getpid()`` ensures measurements are not inflated by
    other jobs sharing the same physical GPU.

    ``device_index`` should be the *physical* NVML device index (use
    ``get_physical_device_index()`` to convert from a logical CUDA index).

    Usage::

        sampler = NvmlProcessSampler(device_index=2)
        sampler.start()
        do_work()
        result = sampler.stop()   # dict | None

    ``result`` keys (all optional):
      ``sm_util_mean``, ``sm_util_max``, ``sm_util_p50``, ``sm_util_p95``,
      ``mem_util_mean``, ``process_vram_bytes``, ``power_mw_mean``
    """

    _POLL_INTERVAL_S: float = 0.05  # 20 Hz

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._pid = os.getpid()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples: list[tuple[int, int]] = []  # (sm_util%, mem_util%)
        self._handle = None
        self._available = False
        self._vram_baseline_bytes: int | None = None
        self._first_sample_logged = False

    def _get_process_vram_bytes(self) -> int | None:
        """Return current per-PID VRAM in bytes via NVML (None if unavailable)."""
        if not self._available or self._handle is None:
            return None
        try:
            import pynvml

            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                if proc.pid == self._pid:
                    return int(proc.usedGpuMemory)
            return 0  # process not yet visible to NVML on this device
        except Exception:
            return None

    def start(self) -> None:
        if not _ensure_nvml():
            return
        try:
            import pynvml

            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            self._available = True
            self._stop_event.clear()
            self._samples.clear()
            self._first_sample_logged = False
            self._vram_baseline_bytes = self._get_process_vram_bytes()
            self._thread = threading.Thread(target=self._run, daemon=True, name="NvmlSampler")
            self._thread.start()
        except Exception as exc:
            logger.debug(f"NvmlProcessSampler.start() failed: {exc}")
            self._available = False

    def _run(self) -> None:
        try:
            import pynvml
        except ImportError:
            return

        last_ts_us = int(time.time() * 1_000_000)  # Prime NVML timestamp.
        first_poll = True  # Discard pre-start() historical samples.

        while not self._stop_event.is_set():
            try:
                samples = pynvml.nvmlDeviceGetProcessUtilization(self._handle, last_ts_us)
                new_last = last_ts_us
                for s in samples:
                    if s.pid == self._pid and not first_poll:
                        self._samples.append((s.smUtil, s.memUtil))
                        if not self._first_sample_logged:
                            logger.debug(f"NVML per-PID samples confirmed (pid={self._pid}, sm={s.smUtil}%, mem={s.memUtil}%)")
                            self._first_sample_logged = True
                    if s.timeStamp > new_last:
                        new_last = s.timeStamp
                last_ts_us = new_last + 1
                first_poll = False
            except Exception:
                # Transient NVML errors are common; just keep polling.
                pass
            self._stop_event.wait(self._POLL_INTERVAL_S)

    def stop(self) -> dict | None:
        """Stop sampling and return collected statistics (or None)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        if not self._available:
            return None

        result: dict = {}

        # Per-process VRAM from NVML: PID-scoped (immune to other jobs) and includes vLLM CuMemAllocator allocations PyTorch misses.
        end_vram = self._get_process_vram_bytes()
        if end_vram is not None:
            result["process_vram_bytes"] = end_vram
            if self._vram_baseline_bytes is not None:
                result["process_vram_delta_bytes"] = max(0, end_vram - self._vram_baseline_bytes)

        # Device-wide power — intentionally labelled as such in the paper.
        try:
            import pynvml

            result["power_mw_mean"] = pynvml.nvmlDeviceGetPowerUsage(self._handle)
        except Exception:
            pass

        if not self._samples:
            return result or None

        sm_vals = [float(s[0]) for s in self._samples]
        mem_vals = [float(s[1]) for s in self._samples]
        sm_sorted = sorted(sm_vals)

        result.update(
            {
                "sm_util_mean": sum(sm_vals) / len(sm_vals),
                "sm_util_max": max(sm_vals),
                "sm_util_p50": _percentile(sm_sorted, 50),
                "sm_util_p95": _percentile(sm_sorted, 95),
                "mem_util_mean": sum(mem_vals) / len(mem_vals),
            }
        )
        return result


# ── measure_inference — main public API ───────────────────────────────────────


@contextmanager
def measure_inference(physical_device_index: int = 0):
    """Bundle CudaEventTimer, PeakMemoryRegion, and NvmlProcessSampler.

    Yields a mutable ``stats`` dict.  The caller may write keys inside the
    block (e.g. ``stats["gen_tokens"] = …``); those are preserved in the final
    dict.  After the ``with`` block exits, the following keys may be present:

    From CudaEventTimer (if CUDA available):
      ``latency_ms``                — GPU-side elapsed time in milliseconds.

    From PeakMemoryRegion (if CUDA available):
      ``peak_vram_bytes``           — peak allocator bytes (absolute).
      ``peak_vram_delta_bytes``     — peak − baseline allocator bytes.

    From NvmlProcessSampler (if pynvml available and driver supports per-PID util):
      ``sm_util_mean/max/p50/p95`` — SM utilisation % for our PID.
      ``mem_util_mean``            — memory-controller util % for our PID.
      ``process_vram_bytes``       — VRAM used by our PID at region exit (NVML).
      ``process_vram_delta_bytes`` — NVML VRAM delta (end − baseline at start).
                                     Primary VRAM metric on shared GPUs: includes
                                     vLLM's CuMemAllocator allocations that the
                                     ``peak_vram_*`` torch metrics cannot see.
      ``power_mw_mean``            — device-wide power in mW (tagged as such).

    Derived (if gen_tokens / prompt_tokens written by caller, and CUDA available):
      ``tok_per_s_total``          — (gen + prompt) tokens / elapsed_s.
      ``tok_per_s_decode``         — gen_tokens / elapsed_s.

    Example::

        with measure_inference(physical_device_index=2) as stats:
            outputs = llm.generate(requests, ...)
            stats["gen_tokens"] = sum(len(o.outputs[0].token_ids) for o in outputs)
            stats["prompt_tokens"] = sum(len(o.prompt_token_ids) for o in outputs)

        print(stats.get("latency_ms"), stats.get("sm_util_p95"))
    """
    stats: dict = {}

    timer = CudaEventTimer()
    mem = PeakMemoryRegion()
    nvml = NvmlProcessSampler(device_index=physical_device_index)

    mem.__enter__()
    nvml.start()
    timer.__enter__()

    try:
        yield stats
    finally:
        timer.__exit__(None, None, None)
        nvml_result = nvml.stop()
        mem.__exit__(None, None, None)

        # ── latency ────────────────────────────────────────────────────────
        if timer.elapsed_ms is not None:
            stats["latency_ms"] = timer.elapsed_ms

        # ── memory ─────────────────────────────────────────────────────────
        if mem.peak_bytes:
            stats["peak_vram_bytes"] = mem.peak_bytes
            stats["peak_vram_delta_bytes"] = mem.delta_bytes

        # ── NVML per-process metrics ────────────────────────────────────────
        if nvml_result:
            stats.update(nvml_result)

        # ── derived throughput ──────────────────────────────────────────────
        elapsed_s = (timer.elapsed_ms or 0) / 1000.0
        if elapsed_s > 0:
            gen_tok = stats.get("gen_tokens", 0)
            prompt_tok = stats.get("prompt_tokens", 0)
            total_tok = gen_tok + prompt_tok
            if total_tok:
                stats["tok_per_s_total"] = total_tok / elapsed_s
            if gen_tok:
                stats["tok_per_s_decode"] = gen_tok / elapsed_s
