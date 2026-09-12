"""The dashboard shows the benchmark sweeps committed under benchmarks/results/."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
RTX_4060 = "NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 592.82"


def test_benchmark_section_labels_each_committed_gpu_sweep_from_its_own_files():
    """Sweeps are committed one directory per host; a page that reads only
    benchmarks/results/ itself reports no results with two of them committed."""
    at = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not at.exception
    labels = [m.value for m in at.markdown]
    assert f"**{RTX_4060}** — `ledoit_wolf` covariance — `benchmarks/results/rtx4060-wsl2/`" in labels
    assert f"**{RTX_4060}** — `pca_factor` covariance — `benchmarks/results/rtx4060-wsl2-pca/`" in labels
    # colab-tesla-t4/ timed only the CPU column on a host that had a GPU: its
    # environment names a Tesla T4, but none of its numbers came from one.
    assert not any("colab-tesla-t4" in label for label in labels)
    assert len(at.dataframe) == len(at.expander) == len(labels)  # a table and its environment each
