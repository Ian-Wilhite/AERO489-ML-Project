"""
Plot g-limit distribution for Dataset 2 (2a + 2b combined).
Output: figures/v2/fig_glimit_combined.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

FIGURES_DIR = Path("figures/v2")
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# ── Load ──────────────────────────────────────────────────────────────────────

y = np.concatenate([
    pl.read_parquet("features/v2a/features_scalar.parquet")["g_limit"].to_numpy(),
    pl.read_parquet("features/v2b/features_scalar.parquet")["g_limit"].to_numpy(),
])

COLOR = "#2c7bb6"

# ── Figure ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))

bins = np.linspace(y.min(), y.max(), 35)
ax.hist(y, bins=bins, color=COLOR, alpha=0.55, density=True,
        edgecolor="white", linewidth=0.4)

kde_x = np.linspace(y.min(), y.max(), 400)
ax.plot(kde_x, stats.gaussian_kde(y)(kde_x), color=COLOR, linewidth=2.2)

ax.axvline(np.median(y), color="black", linewidth=1.2, linestyle="--", alpha=0.65,
           label=f"Median = {np.median(y):.3f} g")

stats_txt = (
    f"n = {len(y)}\n"
    f"μ = {np.mean(y):.3f} g\n"
    f"σ = {np.std(y):.3f} g\n"
    f"IQR = {np.percentile(y,75) - np.percentile(y,25):.3f} g\n"
    f"Range [{y.min():.2f}, {y.max():.2f}] g"
)
ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
        fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="none"))

ax.set_xlabel("g-limit (g)")
ax.set_ylabel("Density")
ax.set_title("g-limit Distribution — Dataset 2", fontweight="bold")
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
out = FIGURES_DIR / "fig_glimit_combined.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
