#!/usr/bin/env python3
"""
Strain gauge rank analysis — 5 interpolation/fitting methods compared.

Generates one figure per method in figures-v2/spline_comparison/:
  fig_01_logistic.png  — four-parameter logistic curve fit
  fig_02_cubic.png     — natural cubic spline through medians
  fig_03_pchip.png     — PCHIP monotone-preserving interpolant
  fig_04_akima.png     — Akima locally-fitted spline
  fig_05_bspline.png   — least-squares cubic B-spline (3 interior knots)

Shared across all methods:
  - boxplot data (24 rank-ordered strain distributions)
  - r2_raw step chart (discrete, 24 points)
  - r2_node boxplot (right panel)
  - per-simulation cubic splines for fine strain evaluation

Only the *median curve* and its inverse (used to build the continuous
R²_spline line) differ between methods.
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
from scipy.interpolate import (
    CubicSpline, PchipInterpolator, Akima1DInterpolator, make_lsq_spline,
)
from scipy.optimize import brentq, curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
SCALAR_PARQUET = ROOT / "features-v2" / "features_scalar.parquet"
TARGET         = "g_limit"
OUT_DIR        = ROOT / "figures-v2" / "spline_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load ────────────────────────────────────────────────────────────────────
df        = pl.read_parquet(SCALAR_PARQUET)
node_cols = sorted(c for c in df.columns if c.startswith("Node_at_"))
gauges    = df.select(node_cols).to_numpy().astype(float)
g_lim     = df[TARGET].to_numpy().astype(float)
N, G      = gauges.shape
print(f"Loaded  N={N} simulations  G={G} gauges")

# ── 2. Rank matrix ─────────────────────────────────────────────────────────────
ranked  = np.sort(gauges, axis=1)          # (N, G)
medians = np.median(ranked, axis=0)        # (G,)
ranks   = np.arange(1, G + 1, dtype=float)
int_ranks  = np.arange(1, G + 1)
pct_labels = [f"{r}\n({100*(r-1)//(G-1)}%)" for r in int_ranks]

# ── 3. Shared R² quantities ────────────────────────────────────────────────────
# Per-rank R²  (same across all methods)
r2_raw = np.array([
    r2_score(g_lim,
             LinearRegression().fit(ranked[:, k:k+1], g_lim)
                               .predict(ranked[:, k:k+1]))
    for k in range(G)
])

# Per-node R² (right panel, unordered)
r2_node = np.array([
    r2_score(g_lim,
             LinearRegression().fit(gauges[:, n:n+1], g_lim)
                               .predict(gauges[:, n:n+1]))
    for n in range(G)
])
pooled_strain = gauges.ravel()

# ── 4. Per-simulation cubic splines (fine evaluation, fixed across methods) ────
sim_cs = CubicSpline(ranks, ranked.T)

P_FINE = 300
p_fine = np.linspace(ranks[0], ranks[-1], P_FINE)

print("Pre-computing per-simulation strains at fine positions ...", end="", flush=True)
all_strains = np.stack([sim_cs(p) for p in p_fine])   # (P_FINE, N)
print(" done")


# ── 5. Helpers ─────────────────────────────────────────────────────────────────
def make_brentq_inv(fn, y_lo, y_hi):
    """Return a vectorized inverse of the monotone fn: strain → rank position."""
    lo   = float(ranks[0])
    hi   = float(ranks[-1])
    span = y_hi - y_lo
    eps  = 1e-12 * span

    def inv(y_arr):
        y_arr = np.asarray(y_arr, dtype=float)
        out   = np.empty(y_arr.size)
        for i, y in enumerate(y_arr.flat):
            yc = float(np.clip(y, y_lo + eps, y_hi - eps))
            try:
                out.flat[i] = brentq(lambda r: float(fn(r)) - yc, lo, hi)
            except ValueError:
                out.flat[i] = np.nan
        return out
    return inv


def compute_r2_spline(inv_fn):
    """R²_spline at P_FINE positions using pre-computed per-sim strains."""
    r2 = np.empty(P_FINE)
    for pi in range(P_FINE):
        Xi    = inv_fn(all_strains[pi]).reshape(-1, 1)
        valid = np.isfinite(Xi.ravel())
        if valid.sum() > 5:
            r2[pi] = r2_score(
                g_lim[valid],
                LinearRegression().fit(Xi[valid], g_lim[valid]).predict(Xi[valid]),
            )
        else:
            r2[pi] = np.nan
    return r2


# ── 6. Figure style constants ──────────────────────────────────────────────────
BLUE_BOX  = dict(facecolor="#aecde8", edgecolor="#4a90d9", alpha=0.65)
GREY_BOX  = dict(facecolor="#d0d0d0", edgecolor="#888888", alpha=0.65)
R2_BOX    = dict(facecolor="#f4a0a8", edgecolor="#c0392b", alpha=0.75)
FLIER_B   = dict(marker=".", markersize=2.5, alpha=0.25, color="#4a90d9")
FLIER_G   = dict(marker=".", markersize=2.5, alpha=0.25, color="#888888")
FLIER_R   = dict(marker=".", markersize=4.0, alpha=0.5,  color="#c0392b")
MED_PROPS = dict(linewidth=1.5)


def make_figure(title, curve_fn, curve_label, inv_fn, out_path):
    print(f"  Computing R²_spline ...", end="", flush=True)
    r2_spl = compute_r2_spline(inv_fn)
    print(f"  min={np.nanmin(r2_spl):.3f}  med={np.nanmedian(r2_spl):.3f}"
          f"  max={np.nanmax(r2_spl):.3f}")

    fig = plt.figure(figsize=(20, 7))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[3.6, 1], wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(gs[1])
    ax4 = ax3.twinx()

    # Left panel — boxplots
    ax1.boxplot(
        [ranked[:, k] for k in range(G)],
        positions=ranks, widths=0.55, patch_artist=True, showfliers=True,
        flierprops=FLIER_B, medianprops={**MED_PROPS, "color": "#003566"},
        boxprops=BLUE_BOX, whiskerprops=dict(color="#4a90d9", lw=0.9),
        capprops=dict(color="#4a90d9", lw=0.9), zorder=2,
    )
    ax1.plot(ranks, medians, color="#003566", lw=2.0, zorder=4)
    xf = np.linspace(ranks[0], ranks[-1], 600)
    ax1.plot(xf, curve_fn(xf), color="#e76f51", lw=2.2, ls="--", zorder=5)

    ax2.step(ranks, r2_raw, where="mid", color="#e63946", lw=2.0, zorder=6)
    ax2.plot(p_fine, r2_spl, color="#2a9d8f", lw=2.0, zorder=7)
    ax2.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.55)

    ax1.set_xticks(int_ranks)
    ax1.set_xticklabels(pct_labels, fontsize=7.5)
    ax1.set_xlim(0.5, G + 0.5)
    ax1.set_xlabel("Gauge rank at failure   (rank · percentile within simulation)", fontsize=11)
    ax1.set_ylabel("Strain at failure", fontsize=11, color="#003566")
    ax1.tick_params(axis="y", labelcolor="#003566")
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax1.set_title(f"Rank-ordered — {title}", fontsize=11, pad=6)

    ax2.set_ylim(min(-0.05, float(np.nanmin(r2_spl)) - 0.05), 1.08)
    ax2.set_ylabel("R²", fontsize=11, color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    ax1.legend(handles=[
        Patch(**{**BLUE_BOX, "label": f"Strain dist. — ranked  (n={N} sims)"}),
        Line2D([0], [0], color="#003566", lw=2,   label="Median growth curve"),
        Line2D([0], [0], color="#e76f51", lw=2.2, ls="--", label=curve_label),
        Line2D([0], [0], color="#e63946", lw=2,   label="R² — raw strain → g-limit"),
        Line2D([0], [0], color="#2a9d8f", lw=2,   label="R² — curve⁻¹(strain) → g-limit"),
    ], loc="upper left", fontsize=8.5, framealpha=0.92)

    # Right panel
    ax3.boxplot(
        [pooled_strain], positions=[1], widths=0.5, patch_artist=True, showfliers=True,
        flierprops=FLIER_G, medianprops={**MED_PROPS, "color": "#444444"},
        boxprops=GREY_BOX, whiskerprops=dict(color="#888888", lw=0.9),
        capprops=dict(color="#888888", lw=0.9), zorder=2,
    )
    ax4.boxplot(
        [r2_node], positions=[1], widths=0.5, patch_artist=True, showfliers=True,
        flierprops=FLIER_R, medianprops={**MED_PROPS, "color": "#7b0000"},
        boxprops=R2_BOX, whiskerprops=dict(color="#c0392b", lw=0.9),
        capprops=dict(color="#c0392b", lw=0.9), zorder=3,
    )
    ax3.set_xlim(0.4, 1.6)
    ax3.set_xticks([1])
    ax3.set_xticklabels([f"Unordered nodes\n({G} nodes × {N} sims)"], fontsize=9)
    ax3.set_xlabel("Unordered spatial nodes", fontsize=10)
    ax3.set_ylabel("Strain at failure", fontsize=11, color="#444444")
    ax3.tick_params(axis="y", labelcolor="#444444")
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax3.set_title("Unordered — summary", fontsize=11, pad=6)
    ax3.set_ylim(ax1.get_ylim())
    ax4.set_ylim(-0.05, 1.08)
    ax4.set_ylabel("R²", fontsize=11, color="#c0392b")
    ax4.tick_params(axis="y", labelcolor="#c0392b")
    ax3.legend(handles=[
        Patch(**{**GREY_BOX, "label": f"Pooled strain  (n={N*G:,})"}),
        Patch(**{**R2_BOX,   "label": f"Per-node R²  (n={G} nodes)"}),
    ], loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle(f"Strain Gauge Analysis — {title}", fontsize=13, y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Method 1: Logistic ─────────────────────────────────────────────────────────
print("\n[1/11] Logistic")

def _logistic(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

p0 = [float(medians[-1] - medians[0]), 0.3, float(G / 2), float(medians[0])]
try:
    popt_log, _ = curve_fit(_logistic, ranks, medians, p0=p0, maxfev=20000)
    print(f"  L={popt_log[0]:.3e}  k={popt_log[1]:.4f}  x0={popt_log[2]:.3f}  b={popt_log[3]:.3e}")
except RuntimeError:
    popt_log = p0
    print("  Warning: logistic fit did not converge, using initial guess")

log_fn = lambda x: _logistic(x, *popt_log)
y0l, y1l = float(log_fn(ranks[0])), float(log_fn(ranks[-1]))
inv_log = make_brentq_inv(log_fn, min(y0l, y1l), max(y0l, y1l))
make_figure("Logistic", log_fn, "Logistic curve", inv_log, OUT_DIR / "fig_01_logistic.png")

# ── Method 2: Cubic Spline ──────────────────────────────────────────────────────
print("\n[2/5] Cubic Spline")
cs_cub = CubicSpline(ranks, medians)
y0c, y1c = float(cs_cub(ranks[0])), float(cs_cub(ranks[-1]))
inv_cub = make_brentq_inv(cs_cub, min(y0c, y1c), max(y0c, y1c))
make_figure("Cubic Spline", cs_cub, "Cubic spline", inv_cub, OUT_DIR / "fig_02_cubic.png")

# ── Method 3: PCHIP ────────────────────────────────────────────────────────────
print("\n[3/5] PCHIP")
cs_pch = PchipInterpolator(ranks, medians)
y0p, y1p = float(cs_pch(ranks[0])), float(cs_pch(ranks[-1]))
inv_pch = make_brentq_inv(cs_pch, min(y0p, y1p), max(y0p, y1p))
make_figure("PCHIP", cs_pch, "PCHIP interpolant", inv_pch, OUT_DIR / "fig_03_pchip.png")

# ── Method 4: Akima ────────────────────────────────────────────────────────────
print("\n[4/5] Akima")
cs_aki = Akima1DInterpolator(ranks, medians)
y0a, y1a = float(cs_aki(ranks[0])), float(cs_aki(ranks[-1]))
inv_aki = make_brentq_inv(cs_aki, min(y0a, y1a), max(y0a, y1a))
make_figure("Akima", cs_aki, "Akima spline", inv_aki, OUT_DIR / "fig_04_akima.png")

# ── Method 5: B-Spline (LSQ regression spline) ────────────────────────────────
print("\n[5/5] B-Spline (LSQ, 3 interior knots)")
k_deg   = 3
t_knots = np.concatenate([
    [ranks[0]] * (k_deg + 1),
    np.quantile(ranks, [0.25, 0.5, 0.75]),
    [ranks[-1]] * (k_deg + 1),
])
cs_bspl  = make_lsq_spline(ranks, medians, t_knots, k=k_deg)
y0b, y1b = float(cs_bspl(ranks[0])), float(cs_bspl(ranks[-1]))
inv_bspl = make_brentq_inv(cs_bspl, min(y0b, y1b), max(y0b, y1b))
make_figure(
    "B-Spline (LSQ)", cs_bspl,
    "B-spline regression (k=3, 3 interior knots)",
    inv_bspl, OUT_DIR / "fig_05_bspline.png",
)

print(f"\n✓ All 5 figures saved to {OUT_DIR}")
