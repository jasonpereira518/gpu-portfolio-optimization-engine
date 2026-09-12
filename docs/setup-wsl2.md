# GPU bring-up on Windows 11 + WSL2 (GeForce RTX)

The GPU half of this project runs on Linux. On a Windows machine with a
GeForce RTX card, that means WSL2. This is the sequence that takes a clean
machine to parity-checked GPU numbers, in the order that keeps each step's
failure easy to diagnose.

> **Verified on:** _not yet — fill in after the first successful run_
> (GPU, driver, Windows build, WSL kernel, Ubuntu release). Until this line is
> filled in, treat every command below as the documented procedure, not a
> tested one.

## What WSL2 does and does not give you

Worth knowing before you start, because it bounds what this setup can claim:

| Works under WSL2 + GeForce | Does not |
|---|---|
| CUDA, cuDF/cuML, cuOpt (cuOpt lists Windows 11 + WSL2; needs ≥16 GB system RAM) | DCGM / dcgm-exporter (bare metal or full passthrough only) |
| Nsight Systems CUDA + NVTX tracing | CUDA MPS (Linux-native only) |
| NVML basics (`nvidia-smi`, `nvidia-ml-py`) — some fields report N/A | NVIDIA GPU Operator (data-center Linux platforms) |
| Docker with `--gpus all` via Docker Desktop's WSL2 backend | Representative absolute PCIe bandwidth (the GPU is paravirtualized) |

Anything in the right-hand column is described in this repo, never claimed as
measured.

## 1. Windows side

1. **Driver.** Install the current NVIDIA Game Ready or Studio driver on
   **Windows only** — never install a Linux display driver inside WSL; WSL
   exposes the Windows driver to Linux. You want R580 or newer (CUDA 13
   support; also what current NIM containers require). Check in PowerShell:

   ```powershell
   nvidia-smi
   ```

   The header shows the driver version and the highest CUDA version it supports.

2. **WSL2 + Ubuntu 24.04:**

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

   ```powershell
   wsl --update
   ```

3. **Give WSL enough memory.** cuOpt needs at least 16 GB of system RAM
   (64 GB recommended), and WSL defaults to half the host's. Create
   `%UserProfile%\.wslconfig`:

   ```ini
   [wsl2]
   memory=32GB      # as much as the host can spare; leave headroom for Windows
   swap=16GB
   ```

   Then restart WSL so it takes effect:

   ```powershell
   wsl --shutdown
   ```

## 2. Linux side (inside Ubuntu)

1. **Confirm the GPU is visible:**

   ```bash
   nvidia-smi
   ```

2. **CUDA toolkit for WSL** — only needed for Nsight Systems (`nsys`); the
   Python wheels bring their own CUDA runtime. Use NVIDIA's WSL-Ubuntu
   repository, whose packages deliberately omit the driver:

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   ```

   ```bash
   sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
   ```

   ```bash
   sudo apt-get install -y cuda-toolkit-13-0
   ```

   Nsight Systems should then be available (`nsys --version`); if not, install
   `nsight-systems-cli` from the same repository.

3. **Clone into the Linux filesystem** (e.g. `~/src`), not under `/mnt/c` —
   file I/O across the Windows boundary is slow enough to distort timings.

4. **Python environment:**

   ```bash
   sudo apt-get install -y python3-venv
   ```

   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```

   ```bash
   .venv/bin/pip install --extra-index-url=https://pypi.nvidia.com -r requirements-gpu.txt
   ```

   `requirements-gpu.txt` pins cuDF, cuML and cuOpt to one release. If your
   driver reports CUDA 12.x rather than 13.x, swap the `-cu13` / `cuda13x`
   suffixes for `-cu12` / `cuda12x` first.

5. **Smoke test** — each line should print, not raise:

   ```bash
   .venv/bin/python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
   ```

   ```bash
   .venv/bin/python -c "import cudf, cuml, cuopt; print(cudf.__version__, cuml.__version__, cuopt.__version__)"
   ```

6. **Docker (optional, for the containerized path).** In Docker Desktop, use
   the WSL2 backend and enable integration for the Ubuntu distro, then:

   ```bash
   docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi
   ```

## 3. Benchmark hygiene on a desktop GPU

A GeForce card on Windows also drives the display and boosts its clocks
opportunistically, so run-to-run variance is higher than on a data-center GPU.
To keep numbers honest:

- Close anything else using the GPU (browsers with hardware acceleration,
  games, video).
- Use the Windows "Best performance" power mode and keep the machine plugged in.
- Optionally lock clocks from an **administrator** PowerShell for the duration
  of a sweep, and unlock afterwards. Not every GeForce driver accepts every
  combination; if it refuses, skip it — clocks are logged either way.

  ```powershell
  nvidia-smi --query-supported-clocks=graphics,memory --format=csv
  ```

  ```powershell
  nvidia-smi -lgc <mhz>,<mhz>
  ```

  ```powershell
  nvidia-smi -rgc
  ```

- **Run the CPU column natively on Windows once too.** WSL2 can slow
  multi-threaded CPU floating point, which would flatter the GPU. Use the
  faster of the two CPU results as the baseline (from a Windows Python 3.12
  venv with `requirements.txt` only):

  ```powershell
  python -m benchmarks.run_benchmarks --backends cpu --out benchmarks/results/<gpu>-windows-cpu
  ```

## 4. Run order — parity before speed, always

```bash
.venv/bin/python -m pytest tests/ -q
```

The GPU tests that skip on a CPU-only machine now execute. Then:

```bash
.venv/bin/python -m pipeline.parity_tests --n 500 --days 2520
```

Only when every parity check passes:

```bash
.venv/bin/python -m benchmarks.run_benchmarks --sizes 50 500 3000 --days 2520 --runs 5 --out benchmarks/results/<gpu>-wsl2
```

```bash
.venv/bin/python -m backtest.run_backtest --source synthetic --n 500 --frequency QE
```

The PCA factor estimator is the one stage that runs cuML, so it gets its own
parity check, then its own sweep:

```bash
.venv/bin/python -m pipeline.parity_tests --n 500 --days 2520 --estimator pca_factor
```

```bash
.venv/bin/python -m benchmarks.run_benchmarks --sizes 50 500 3000 --days 2520 --runs 5 --estimator pca_factor --out benchmarks/results/<gpu>-wsl2-pca
```

Stage 2, the lot-rounding MIP against greedy rounding. Each of the 18 cases
has a fully-invested solve that can run to its 60 s time limit, so allow up
to ~20 minutes:

```bash
.venv/bin/python -m benchmarks.run_lot_rounding --sizes 50 200 500 --out benchmarks/results/<gpu>-wsl2-lots
```

Commit the results directory with its `environment.json`; the README's tables
are generated from those files, never typed.

## Troubleshooting

- **`nvidia-smi` missing inside WSL:** the Windows driver predates WSL2 GPU
  support, or WSL itself needs `wsl --update`.
- **`libcuda.so` not found:** `/usr/lib/wsl/lib` must be on the library path;
  it is by default, so something has overridden `LD_LIBRARY_PATH`.
- **Killed or out-of-memory during import or the n=3000 sweep:** the
  `.wslconfig` memory limit is too low for cuOpt.
- **A wheel refuses to install:** pins from different releases cannot co-exist;
  keep every NVIDIA package on the same release, as `requirements-gpu.txt` does.
