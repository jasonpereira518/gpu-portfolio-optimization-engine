# Reproducible GPU benchmark environment.
#
# Every version below is pinned, because the benchmark numbers in the README
# are only meaningful alongside the stack that produced them.
#
#   docker build -t gpu-portfolio-engine .
#   docker run --gpus all -it --rm -v $(pwd):/workspace gpu-portfolio-engine
#
# Requires the NVIDIA Container Toolkit on the host. On a CPU-only machine,
# skip this entirely and use `make venv` — the CPU baseline, tests, and
# backtest all run natively.

FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# CPU dependencies first: they change rarely, so this layer stays cached across
# rebuilds while the (much larger) GPU layer below is being iterated on.
COPY requirements.txt .
RUN python3 -m pip install --break-system-packages -r requirements.txt

# GPU stack from NVIDIA's index. cu12 wheels to match the CUDA 12.9 base image
# above — if you change the base image's CUDA major version, change these too.
RUN python3 -m pip install --break-system-packages \
        --extra-index-url=https://pypi.nvidia.com \
        cudf-cu12==26.6.* \
        cuml-cu12==26.6.* \
        cupy-cuda12x \
        cuopt-cu12==26.2.*

COPY . .

# Fail the build if the GPU stack is not actually importable, rather than
# discovering it at benchmark time.
RUN python3 -c "import cudf, cupy, cuopt; print('cudf', cudf.__version__, '| cuopt', cuopt.__version__)"

CMD ["python3", "-m", "benchmarks.run_benchmarks", "--sizes", "50", "500", "3000", "--runs", "5"]
