# Reproducible GPU benchmark environment.
#
# The base image is pinned by digest and the GPU stack by release, because the
# benchmark numbers in the README are only meaningful alongside the stack that
# produced them.
#
#   docker build -t gpu-portfolio-engine .
#   docker run --gpus all -it --rm -v $(pwd):/workspace gpu-portfolio-engine
#
# Requires the NVIDIA Container Toolkit on the host (Docker Desktop's WSL2
# backend provides it on Windows). On a CPU-only machine, skip this entirely
# and use `make venv` — the CPU baseline, tests, and backtest all run natively.

# nvidia/cuda:13.0.3-runtime-ubuntu24.04 (Ubuntu 24.04 ships Python 3.12).
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04@sha256:76f46f3e649824ebd3685d3d86c42b0ea62af04f59242f419f56cd5804a72efb

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# CPU dependencies first: they change rarely, so this layer stays cached across
# rebuilds while the (much larger) GPU layer below is being iterated on.
COPY requirements.txt .
RUN python3 -m pip install --break-system-packages -r requirements.txt

# GPU stack from NVIDIA's index, from the same pin file a venv install uses.
COPY requirements-gpu.txt .
RUN python3 -m pip install --break-system-packages \
        --extra-index-url=https://pypi.nvidia.com -r requirements-gpu.txt \
    && python3 -m pip check

COPY . .

# `docker build` has no GPU, so importing cuDF here would fail for the wrong
# reason. Record the resolved versions instead; GPU checks happen at run time.
RUN python3 -c "import importlib.metadata as m; print({p: m.version(p) for p in ('cudf-cu13', 'cuml-cu13', 'cuopt-cu13', 'cupy-cuda13x')})"

CMD ["python3", "-m", "benchmarks.run_benchmarks", "--sizes", "50", "500", "3000", "--runs", "5"]
