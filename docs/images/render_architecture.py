"""Regenerate docs/images/architecture.png.

    python docs/images/render_architecture.py

The README diagram has no other source, so edit the boxes here. Colours say what
a box *is*, not which hardware it ran on: every stage has a CPU and a GPU
implementation behind the same RiskModel / PortfolioSpec contract.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "architecture.png"

STYLES = {  # kind -> (edge, fill, title colour)
    "plain": ("#3a3f47", "#f7f8fa", "#3a3f47"),
    "data": ("#1f5fa8", "#eaf1fb", "#1f5fa8"),
    "optimize": ("#1a7f37", "#e9f6ee", "#1a7f37"),
}

# (column, row, kind, title, lines). Row 0 runs left to right, row 1 right to left.
BOXES = [
    (0, 0, "plain", "Price data", ["yfinance", "(Parquet cache)", "synthetic", "(k-factor)"]),
    (1, 0, "data", "pandas / cuDF", ["returns,", "rolling vol,", "momentum"]),
    (2, 0, "data", "NumPy / CuPy / cuML", ["covariance:", "sample /", "Ledoit-Wolf /", "PCA factor"]),
    (3, 0, "optimize", "CVXPY / cuOpt", ["QP:", "mean-variance", "weights"]),
    (3, 1, "optimize", "MIP (opt.)", ["lot rounding,", "turnover", "limits"]),
    (2, 1, "plain", "backtest", ["rolling", "rebalance, P&L", "vs 1/N"]),
    (1, 1, "plain", "benchmark", ["table +", "parity report"]),
    (0, 1, "plain", "dashboard", ["(Streamlit)", "CPU solves +", "GPU tables"]),
]
W, H, GAP_X, GAP_Y = 2.0, 1.55, 0.35, 0.9


def origin(col: int, row: int) -> tuple[float, float]:
    return col * (W + GAP_X), -row * (H + GAP_Y)


fig, ax = plt.subplots(figsize=(11.9, 6.2), dpi=200)
ax.set_xlim(-0.3, 4 * W + 3 * GAP_X + 0.3)
ax.set_ylim(-(H + GAP_Y) - 0.95, H + 1.1)
ax.axis("off")

ax.text(2 * W + 1.5 * GAP_X, H + 0.85, "GPU Portfolio & Risk Decision Engine — architecture",
        ha="center", fontsize=15, fontweight="bold")
ax.text(2 * W + 1.5 * GAP_X, H + 0.45,
        "every stage has a CPU and a GPU implementation behind the same RiskModel / PortfolioSpec contract",
        ha="center", fontsize=9.5, style="italic", color="#555c66")

for col, row, kind, title, lines in BOXES:
    edge, fill, title_colour = STYLES[kind]
    x, y = origin(col, row)
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=0.08",
                                linewidth=1.6, edgecolor=edge, facecolor=fill))
    ax.text(x + W / 2, y + H - 0.28, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=title_colour)
    for i, line in enumerate(lines):
        ax.text(x + W / 2, y + H - 0.58 - i * 0.22, line, ha="center", va="center", fontsize=8.8,
                color="#14181f")


def arrow(start: tuple[float, float], end: tuple[float, float], colour: str = "#3a3f47") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.8,
                                 color=colour, shrinkA=0, shrinkB=0))


mid = H / 2
for col in range(3):  # top row, left to right
    x, y = origin(col, 0)
    arrow((x + W + 0.04, y + mid), (x + W + GAP_X - 0.04, y + mid))
x, y = origin(3, 0)  # QP down to MIP
arrow((x + W / 2, y - 0.04), (x + W / 2, y - GAP_Y + 0.04), STYLES["optimize"][0])
for col in (3, 2, 1):  # bottom row, right to left
    x, y = origin(col, 1)
    arrow((x - 0.04, y + mid), (x - GAP_X + 0.04, y + mid))

# Legend: what the colours mean.
lx, ly = 0.0, -(H + GAP_Y) - 0.6  # just below the bottom row of boxes
for x_offset, kind, label in [
    (0.0, "data", "features and risk model (CPU | GPU libraries)"),
    (4.1, "optimize", "optimizers (CVXPY | cuOpt)"),
    (6.9, "plain", "inputs and consumers"),
]:
    edge, fill, _ = STYLES[kind]
    ax.add_patch(FancyBboxPatch((lx + x_offset, ly), 0.3, 0.2, boxstyle="round,pad=0,rounding_size=0.04",
                                linewidth=1.4, edgecolor=edge, facecolor=fill))
    ax.text(lx + x_offset + 0.4, ly + 0.1, label, va="center", fontsize=8.2, color="#3a3f47")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
