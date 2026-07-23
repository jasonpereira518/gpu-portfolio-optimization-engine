"""Timing harness.

The rules this encodes, each of which corresponds to a way GPU benchmarks are
commonly (and usually unintentionally) inflated:

1. **Warm-up is reported, not hidden.** The first call pays CUDA context
   creation, kernel JIT, and memory-pool growth — often 100x the steady-state
   cost. It is recorded as its own field rather than averaged in or quietly
   dropped.
2. **Device synchronization before stopping the clock.** CUDA is asynchronous;
   timing without a sync measures kernel *launch* rate.
3. **Variance is reported.** A mean with no spread hides a bimodal
   distribution, and GPU timings are frequently bimodal.
4. **Stages are timed separately.** A single end-to-end number lets a fast
   stage carry a slow one and conceals where the win actually is.
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Timing:
    """Result of timing one function under one configuration."""

    stage: str
    backend: str
    n_assets: int
    n_days: int
    warmup_s: float
    mean_s: float
    std_s: float
    min_s: float
    median_s: float
    n_runs: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cv(self) -> float:
        """Coefficient of variation — the honest way to flag an unstable timing."""
        return self.std_s / self.mean_s if self.mean_s else 0.0

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        extra = row.pop("extra")
        row["cv"] = self.cv
        row.update({f"extra_{k}": v for k, v in extra.items()})
        return row


def _sync() -> None:
    """Block until all queued GPU work is done. No-op without CuPy."""
    try:
        import cupy

        cupy.cuda.Stream.null.synchronize()
    except Exception:
        pass


def benchmark_stage(
    fn: Callable,
    *args,
    stage: str,
    backend: str,
    n_assets: int,
    n_days: int,
    n_runs: int = 5,
    extra: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[Any, Timing]:
    """Time ``fn`` ``n_runs`` times after one separately-recorded warm-up call."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    _sync()
    warmup = time.perf_counter() - t0

    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        _sync()
        times.append(time.perf_counter() - t0)

    timing = Timing(
        stage=stage,
        backend=backend,
        n_assets=n_assets,
        n_days=n_days,
        warmup_s=warmup,
        mean_s=float(statistics.fmean(times)),
        std_s=float(statistics.stdev(times)) if len(times) > 1 else 0.0,
        min_s=float(min(times)),
        median_s=float(statistics.median(times)),
        n_runs=n_runs,
        extra=extra or {},
    )
    return result, timing


def environment_metadata() -> dict[str, str]:
    """Capture what the numbers depend on, so the table means something later."""
    meta = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        meta["numpy"] = np.__version__
        meta["blas"] = str(np.__config__.show(mode="dicts")["Build Dependencies"]["blas"]["name"])
    except Exception:
        meta["blas"] = "unknown"

    for module in ("cupy", "cudf", "cuml", "cuopt", "cvxpy"):
        try:
            meta[module] = __import__(module).__version__
        except Exception:
            meta[module] = "not installed"

    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        meta["gpu"] = gpu.stdout.strip()
    except Exception:
        meta["gpu"] = "none detected"

    return meta


def speedup_table(timings: list[Timing]) -> "Any":
    """Pivot CPU/GPU timings into a per-stage, per-size speedup table.

    Reports median rather than mean: with n_runs=5 a single scheduling hiccup
    moves the mean substantially, and the median is what survives that.
    """
    import pandas as pd

    df = pd.DataFrame([t.as_row() for t in timings])
    if df.empty:
        return df

    pivot = df.pivot_table(
        index=["stage", "n_assets", "n_days"], columns="backend", values="median_s"
    )
    if "cpu" in pivot.columns and "gpu" in pivot.columns:
        pivot["speedup"] = pivot["cpu"] / pivot["gpu"]
    return pivot.reset_index()
