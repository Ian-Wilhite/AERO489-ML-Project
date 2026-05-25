#!/usr/bin/env python3
"""
Strain gauge rank analysis — two-panel composite figure.

Left panel  : per-rank (percentile-ordered) boxplots, median growth curve,
              cubic spline, R²_raw and R²_spline⁻¹ on a dual right axis.
Right panel : fixed-node (unordered) boxplots, per-node R² on a dual right axis.
              Nodes sorted by their mean strain so spread is visually comparable.

The side-by-side contrast shows that within-simulation rank ordering alone
creates the structured, monotone signal seen in the left panel.

Output: figures/v2/strain_gauge_rank_analysis.png
"""

from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
SCALAR_PARQUET = ROOT / "features/v2" / "features_scalar.parquet"
TARGET         = "g_limit"
OUT            = ROOT / "figures/v2" / "strain_gauge_rank_analysis.png"

# ── 1. Load ────────────────────────────────────────────────────────────────────
df        = pl.read_parquet(SCALAR_PARQUET)
node_cols = sorted(c for c in df.columns if c.startswith("Node_at_"))
gauges    = df.select(node_cols).to_numpy().astype(float)   # (N, G) raw, unordered
g_lim     = df[TARGET].to_numpy().astype(float)

N, G = gauges.shape
print(f"Loaded  N={N} simulations  G={G} gauges")

# ── 2. Rank matrix ─────────────────────────────────────────────────────────────
ranked  = np.sort(gauges, axis=1)       # (N, G)  col k = k-th lowest within sim
medians = np.median(ranked, axis=0)     # (G,)
ranks   = np.arange(1, G + 1)

# ── 3. Cubic spline through the median growth curve ───────────────────────────
cs     = CubicSpline(ranks, medians)
cs_min = float(cs(ranks[0]))
cs_max = float(cs(ranks[-1]))

# ── 4. Inverse spline: strain → spline-rank position ──────────────────────────
def inv_spline(y_arr):
    y_arr = np.asarray(y_arr, dtype=float)
    out   = np.empty(y_arr.size)
    lo, hi = float(ranks[0]), float(ranks[-1])
    for i, y in enumerate(y_arr.flat):
        yc = float(np.clip(y, cs_min * (1 + 1e-9), cs_max * (1 - 1e-9)))
        try:
            out.flat[i] = brentq(lambda r: float(cs(r)) - yc, lo, hi)
        except ValueError:
            out.flat[i] = np.nan
    return out

# ── 5. Per-rank R² (left panel) ───────────────────────────────────────────────
# R²_raw: 24 discrete points, one per gauge rank (step chart)
r2_raw = np.empty(G)
for ki in range(G):
    Xr = ranked[:, ki].reshape(-1, 1)
    r2_raw[ki] = r2_score(g_lim, LinearRegression().fit(Xr, g_lim).predict(Xr))

# R²_spline: evaluated on a fine percentile grid → smooth line.
# Fit a CubicSpline through each simulation's sorted profile (same method as cs),
# vectorised by passing ranked.T so scipy fits all N at once.
sim_cs = CubicSpline(ranks.astype(float), ranked.T)  # evaluates to shape (N,) at scalar p

P_FINE      = 300
p_fine      = np.linspace(float(ranks[0]), float(ranks[-1]), P_FINE)
r2_spl_fine = np.empty(P_FINE)

for pi, p in enumerate(p_fine):
    strain_p = sim_cs(p)                                      # (N,) per-simulation strain

    Xi    = inv_spline(strain_p).reshape(-1, 1)
    valid = np.isfinite(Xi.ravel())
    if valid.sum() > 5:
        r2_spl_fine[pi] = r2_score(
            g_lim[valid],
            LinearRegression().fit(Xi[valid], g_lim[valid]).predict(Xi[valid]),
        )
    else:
        r2_spl_fine[pi] = np.nan

print("R²_raw   :", np.round(r2_raw, 3))
print("R²_spline (fine, min/med/max):",
      np.round(np.nanmin(r2_spl_fine), 3),
      np.round(np.nanmedian(r2_spl_fine), 3),
      np.round(np.nanmax(r2_spl_fine), 3))

# ── 6. Per-node R² (right panel) — unordered fixed spatial nodes ──────────────
r2_node = np.empty(G)
for ni in range(G):
    Xn = gauges[:, ni].reshape(-1, 1)
    r2_node[ni] = r2_score(g_lim, LinearRegression().fit(Xn, g_lim).predict(Xn))

# Pool all node strain values into one distribution for the single boxplot.
pooled_strain = gauges.ravel()   # N*G values

print("R²_node:", np.round(r2_node, 3))
print(f"Pooled strain  n={len(pooled_strain)}  "
      f"median={np.median(pooled_strain):.3e}  IQR={np.percentile(pooled_strain,75)-np.percentile(pooled_strain,25):.3e}")
print(f"R²_node        n={G}  "
      f"median={np.median(r2_node):.3f}  IQR={np.percentile(r2_node,75)-np.percentile(r2_node,25):.3f}")

# ── 7. Figure layout ───────────────────────────────────────────────────────────
BLUE_BOX  = dict(facecolor="#aecde8", edgecolor="#4a90d9", alpha=0.65)
GREY_BOX  = dict(facecolor="#d0d0d0", edgecolor="#888888", alpha=0.65)
R2_BOX    = dict(facecolor="#f4a0a8", edgecolor="#c0392b", alpha=0.75)
FLIER_B   = dict(marker=".", markersize=2.5, alpha=0.25, color="#4a90d9")
FLIER_G   = dict(marker=".", markersize=2.5, alpha=0.25, color="#888888")
FLIER_R   = dict(marker=".", markersize=4.0, alpha=0.5,  color="#c0392b")
MED_PROPS = dict(linewidth=1.5)

fig = plt.figure(figsize=(20, 7))
gs  = GridSpec(1, 2, figure=fig, width_ratios=[3.6, 1], wspace=0.32)

ax1 = fig.add_subplot(gs[0])   # left: ranked strain (main y)
ax2 = ax1.twinx()              # left: R²
ax3 = fig.add_subplot(gs[1])   # right: single-column summary (strain, left y)
ax4 = ax3.twinx()              # right: R² (right y)

# ── 8. Left panel ─────────────────────────────────────────────────────────────

# Boxplots
ax1.boxplot(
    [ranked[:, k] for k in range(G)],
    positions=ranks, widths=0.55, patch_artist=True, showfliers=True,
    flierprops=FLIER_B, medianprops={**MED_PROPS, "color": "#003566"},
    boxprops=BLUE_BOX, whiskerprops=dict(color="#4a90d9", lw=0.9),
    capprops=dict(color="#4a90d9", lw=0.9), zorder=2,
)

# Median growth curve
ax1.plot(ranks, medians, color="#003566", lw=2.0, zorder=4)

# Cubic spline
xf = np.linspace(ranks[0], ranks[-1], 600)
ax1.plot(xf, cs(xf), color="#e76f51", lw=2.2, ls="--", zorder=5)

# R² step lines
ax2.step(ranks, r2_raw, where="mid", color="#e63946", lw=2.0, zorder=6)
ax2.plot(p_fine, r2_spl_fine, color="#2a9d8f", lw=2.0, zorder=7)
ax2.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.55)

# Formatting
pct_labels = [f"{r}\n({100*(r-1)//(G-1)}%)" for r in ranks]
ax1.set_xticks(ranks)
ax1.set_xticklabels(pct_labels, fontsize=7.5)
ax1.set_xlim(0.5, G + 0.5)
ax1.set_xlabel("Gauge rank at failure   (rank · percentile within simulation)", fontsize=11)
ax1.set_ylabel("Strain at failure", fontsize=11, color="#003566")
ax1.tick_params(axis="y", labelcolor="#003566")
ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax1.set_title("Rank-ordered (percentile)", fontsize=11, pad=6)

r2_min_l = min(-0.05, float(np.nanmin(r2_spl_fine)) - 0.05)
ax2.set_ylim(r2_min_l, 1.08)
ax2.set_ylabel("R²", fontsize=11, color="#c0392b")
ax2.tick_params(axis="y", labelcolor="#c0392b")

# Legend (left panel only)
legend_elems = [
    Patch(**{**BLUE_BOX, "label": f"Strain dist. — ranked  (n={N} sims)"}),
    Line2D([0], [0], color="#003566", lw=2,   label="Median growth curve"),
    Line2D([0], [0], color="#e76f51", lw=2.2, ls="--", label="Cubic spline"),
    Line2D([0], [0], color="#e63946", lw=2,   label="R² — raw strain → g-limit"),
    Line2D([0], [0], color="#2a9d8f", lw=2,   label="R² — spline⁻¹(strain) → g-limit  (continuous)"),
]
ax1.legend(handles=legend_elems, loc="upper left", fontsize=8.5, framealpha=0.92)

# ── 9. Right panel — single pooled strain box + R² box ────────────────────────
# Position 1: pooled strain (all nodes × all sims), left axis
# Position 2: per-node R² values (24 values), right axis

ax3.boxplot(
    [pooled_strain],
    positions=[1], widths=0.5, patch_artist=True, showfliers=True,
    flierprops=FLIER_G, medianprops={**MED_PROPS, "color": "#444444"},
    boxprops=GREY_BOX, whiskerprops=dict(color="#888888", lw=0.9),
    capprops=dict(color="#888888", lw=0.9), zorder=2,
)

ax4.boxplot(
    [r2_node],
    positions=[1], widths=0.5, patch_artist=True, showfliers=True,
    flierprops=FLIER_R, medianprops={**MED_PROPS, "color": "#7b0000"},
    boxprops=R2_BOX, whiskerprops=dict(color="#c0392b", lw=0.9),
    capprops=dict(color="#c0392b", lw=0.9), zorder=3,
)

# Formatting
ax3.set_xlim(0.4, 1.6)
ax3.set_xticks([1])
ax3.set_xticklabels(
    [f"Unordered nodes\n({G} nodes × {N} sims)"],
    fontsize=9,
)
ax3.set_xlabel("Unordered spatial nodes", fontsize=10)
ax3.set_ylabel("Strain at failure", fontsize=11, color="#444444")
ax3.tick_params(axis="y", labelcolor="#444444")
ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax3.set_title("Unordered — summary", fontsize=11, pad=6)

# Match strain y-limit to left panel for direct comparison
ax3.set_ylim(ax1.get_ylim())

ax4.set_ylim(-0.05, 1.08)
ax4.set_ylabel("R²", fontsize=11, color="#c0392b")
ax4.tick_params(axis="y", labelcolor="#c0392b")

# Legend (right panel)
legend_r = [
    Patch(**{**GREY_BOX, "label": f"Pooled strain  (n={N*G:,})"}),
    Patch(**{**R2_BOX,   "label": f"Per-node R²  (n={G} nodes)"}),
]
ax3.legend(handles=legend_r, loc="upper left", fontsize=8.5, framealpha=0.92)

# ── 10. Save ───────────────────────────────────────────────────────────────────
fig.suptitle(
    "Strain Gauge Analysis: Within-simulation rank ordering vs. fixed spatial nodes",
    fontsize=13, y=1.01,
)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT}")
