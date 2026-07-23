"""GPU feature and risk pipeline: cuDF + cuML.

Mirrors ``pipeline.cpu_baseline`` function for function. The estimators are
implemented the same way on both sides deliberately — calling sklearn on CPU
and cuML on GPU would compare two different algorithms and attribute the
difference to hardware.

One deliberate deviation: covariance estimation runs in float64 on GPU. cuDF's
default float32 gives a ~2x memory-bandwidth win, but a covariance matrix built
in float32 loses roughly 7 significant digits, which is enough to push a
near-singular matrix indefinite and change the optimizer's answer. Precision is
matched to the CPU baseline so that parity is a real test; ``dtype`` is exposed
so the float32 tradeoff can be measured rather than guessed at.
"""

from __future__ import annotations

import functools
import importlib

import numpy as np

from pipeline.risk_model import TRADING_DAYS, RiskModel


class GpuUnavailable(RuntimeError):
    """Raised when RAPIDS is not importable, with a message that says what to do."""


@functools.lru_cache(maxsize=1)
def load_rapids():
    """Import cudf/cupy/cuml once, or explain how to get them."""
    try:
        cudf = importlib.import_module("cudf")
        cupy = importlib.import_module("cupy")
    except ImportError as exc:  # pragma: no cover - exercised only off-GPU
        raise GpuUnavailable(
            "RAPIDS (cudf/cupy) is not installed. This module only runs on a CUDA "
            "machine. Install with:\n"
            "  pip install --extra-index-url=https://pypi.nvidia.com \\\n"
            "      cudf-cu13==26.6.* cuml-cu13==26.6.*\n"
            "On a CPU-only host, use pipeline.cpu_baseline instead."
        ) from exc
    try:
        cuml = importlib.import_module("cuml")
    except ImportError:
        cuml = None  # cuML is only needed for the PCA factor estimator
    return cudf, cupy, cuml


def rapids_available() -> bool:
    try:
        load_rapids()
        return True
    except GpuUnavailable:
        return False


def to_gpu(prices, dtype: str = "float64"):
    """pandas DataFrame -> cuDF DataFrame (or pass through if already cuDF)."""
    cudf, _, _ = load_rapids()
    if isinstance(prices, cudf.DataFrame):
        return prices.astype(dtype)
    return cudf.from_pandas(prices).astype(dtype)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def compute_returns_gpu(price_gdf, method: str = "simple"):
    cudf, cupy, _ = load_rapids()
    if method == "simple":
        returns = price_gdf.pct_change()
    elif method == "log":
        returns = price_gdf.log().diff()
    else:
        raise ValueError(f"unknown return method {method!r}")
    return returns.dropna(how="all").dropna(axis=1, how="all")


def rolling_features_gpu(price_gdf, windows: tuple[int, ...] = (21, 63, 252)) -> dict:
    """Same rolling vol / momentum set as the CPU path."""
    returns = compute_returns_gpu(price_gdf)
    out: dict = {}
    for w in windows:
        out[f"vol_{w}"] = returns.rolling(w).std() * np.sqrt(TRADING_DAYS)
        out[f"mom_{w}"] = price_gdf.pct_change(w)
    out["returns"] = returns
    return out


# --------------------------------------------------------------------------
# Covariance estimators (CuPy — this is the dense linear algebra that GPUs win)
# --------------------------------------------------------------------------

def _returns_matrix(returns_gdf):
    """cuDF frame -> demeaned CuPy array, plus the raw array."""
    _, cupy, _ = load_rapids()
    x = returns_gdf.to_cupy().astype(cupy.float64)
    return x - x.mean(axis=0, keepdims=True)


def sample_covariance_gpu(returns_gdf) -> np.ndarray:
    _, cupy, _ = load_rapids()
    xc = _returns_matrix(returns_gdf)
    t = xc.shape[0]
    # ddof=1 to match pandas' DataFrame.cov(), which uses the unbiased estimator.
    cov = (xc.T @ xc) / (t - 1)
    return cupy.asnumpy(cov) * TRADING_DAYS


def ledoit_wolf_covariance_gpu(returns_gdf) -> np.ndarray:
    """Ledoit-Wolf shrinkage, GPU version of the CPU closed form.

    The beta^2 term is the interesting part: the CPU version loops over T
    observations forming outer products. Written as a loop on GPU that would be
    T kernel launches. The identity

        sum_t ||x_t x_t' - S||_F^2 = sum_t ||x_t x_t'||_F^2 - T*||S||_F^2
                                   = sum_t (x_t'x_t)^2 - T*||S||_F^2

    collapses it to one squared-norm reduction over the rows — no n x n
    intermediates at all, which is what makes this tractable at n=3,000.
    """
    _, cupy, _ = load_rapids()
    xc = _returns_matrix(returns_gdf)
    t, n = xc.shape

    sample = (xc.T @ xc) / t

    mu = cupy.trace(sample) / n
    delta_sq = cupy.sum((sample - mu * cupy.eye(n)) ** 2) / n

    row_sq_norms = cupy.sum(xc * xc, axis=1)  # x_t' x_t for each t
    beta_sq = (cupy.sum(row_sq_norms**2) - t * cupy.sum(sample**2)) / (n * t**2)
    beta_sq = cupy.maximum(beta_sq, 0.0)
    beta_sq = cupy.minimum(beta_sq, delta_sq)

    shrink = 0.0 if float(delta_sq) == 0.0 else beta_sq / delta_sq
    shrunk = shrink * mu * cupy.eye(n) + (1.0 - shrink) * sample
    return cupy.asnumpy(shrunk) * TRADING_DAYS


def pca_factor_covariance_gpu(returns_gdf, n_factors: int = 10) -> np.ndarray:
    """Factor covariance Sigma = B F B' + D via cuML PCA (CuPy SVD fallback)."""
    _, cupy, cuml = load_rapids()
    xc = _returns_matrix(returns_gdf)
    t, n = xc.shape
    k = n_factors
    if k > min(t, n) - 1 or k < 1:
        raise ValueError(
            f"cannot fit {n_factors} factors to a {t}x{n} return matrix "
            f"(max {min(t, n) - 1})"
        )

    if cuml is not None:
        from cuml.decomposition import PCA

        pca = PCA(n_components=k, output_type="cupy")
        scores = pca.fit_transform(xc)  # (t, k)
        loadings = cupy.asarray(pca.components_).T  # (n, k)
        factor_var = scores.var(axis=0)
    else:
        _, s, vt = cupy.linalg.svd(xc, full_matrices=False)
        loadings = vt[:k].T
        factor_var = (s[:k] ** 2) / t

    systematic = (loadings * factor_var) @ loadings.T
    total_var = cupy.sum(xc * xc, axis=0) / t
    idio_var = cupy.maximum(total_var - cupy.diag(systematic), 0.0)

    cov = systematic + cupy.diag(idio_var)
    return cupy.asnumpy(cov) * TRADING_DAYS


COV_ESTIMATORS_GPU = {
    "sample": sample_covariance_gpu,
    "ledoit_wolf": ledoit_wolf_covariance_gpu,
    "pca_factor": pca_factor_covariance_gpu,
}


def build_risk_model_gpu(
    prices,
    estimator: str = "ledoit_wolf",
    n_factors: int = 10,
    dtype: str = "float64",
    synchronize: bool = True,
) -> RiskModel:
    """prices (pandas or cuDF) -> RiskModel computed on GPU.

    ``synchronize`` forces a device sync before returning. CUDA calls are
    asynchronous, so timing a GPU function without a sync measures how fast
    Python can enqueue kernels, not how fast they run — the single most common
    way GPU benchmarks come out fraudulently good. Every timed call site sets
    this True.
    """
    _, cupy, _ = load_rapids()

    price_gdf = to_gpu(prices, dtype=dtype)
    returns_gdf = compute_returns_gpu(price_gdf).dropna(axis=1, how="any")
    if returns_gdf.shape[1] == 0:
        raise ValueError("no usable returns after dropping incomplete columns")

    exp_returns = cupy.asnumpy(returns_gdf.mean().to_cupy()).astype(np.float64) * TRADING_DAYS

    if estimator == "pca_factor":
        cov = pca_factor_covariance_gpu(returns_gdf, n_factors=n_factors)
    elif estimator in COV_ESTIMATORS_GPU:
        cov = COV_ESTIMATORS_GPU[estimator](returns_gdf)
    else:
        raise ValueError(f"unknown estimator {estimator!r}; choose from {list(COV_ESTIMATORS_GPU)}")

    if synchronize:
        cupy.cuda.Stream.null.synchronize()

    return RiskModel(
        exp_returns=exp_returns,
        cov=cov,
        tickers=[str(c) for c in returns_gdf.columns],
        estimator=estimator,
        backend="gpu",
    )
