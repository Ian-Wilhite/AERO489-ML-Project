#!/usr/bin/env python3
"""
Strain gauge rank analysis — 5 interpolation/fitting methods compared.

Generates one figure per method in figures/v2/spline_comparison/:
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
SCALAR_PARQUET = ROOT / "features/v2" / "features_scalar.parquet"
TARGET         = "g_limit"
OUT_DIR        = ROOT / "figures/v2" / "spline_comparison"
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
print("\n[2/11] Cubic Spline")
cs_cub = CubicSpline(ranks, medians)
y0c, y1c = float(cs_cub(ranks[0])), float(cs_cub(ranks[-1]))
inv_cub = make_brentq_inv(cs_cub, min(y0c, y1c), max(y0c, y1c))
make_figure("Cubic Spline", cs_cub, "Cubic spline", inv_cub, OUT_DIR / "fig_02_cubic.png")

# ── Method 3: PCHIP ────────────────────────────────────────────────────────────
print("\n[3/11] PCHIP")
cs_pch = PchipInterpolator(ranks, medians)
y0p, y1p = float(cs_pch(ranks[0])), float(cs_pch(ranks[-1]))
inv_pch = make_brentq_inv(cs_pch, min(y0p, y1p), max(y0p, y1p))
make_figure("PCHIP", cs_pch, "PCHIP interpolant", inv_pch, OUT_DIR / "fig_03_pchip.png")

# ── Method 4: Akima ────────────────────────────────────────────────────────────
print("\n[4/11] Akima")
cs_aki = Akima1DInterpolator(ranks, medians)
y0a, y1a = float(cs_aki(ranks[0])), float(cs_aki(ranks[-1]))
inv_aki = make_brentq_inv(cs_aki, min(y0a, y1a), max(y0a, y1a))
make_figure("Akima", cs_aki, "Akima spline", inv_aki, OUT_DIR / "fig_04_akima.png")

# ── Method 5: B-Spline (LSQ regression spline) ────────────────────────────────
print("\n[5/11] B-Spline (LSQ, 3 interior knots)")
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

# ── Method 6: Gompertz ────────────────────────────────────────────────────────
print("\n[6/11] Gompertz")

def _gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))

p0 = [float(medians[-1] * 1.5), 10.0, 0.2]
try:
    popt_gom, _ = curve_fit(_gompertz, ranks, medians, p0=p0, maxfev=20000,
                            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    print(f"  a={popt_gom[0]:.3e}  b={popt_gom[1]:.4f}  c={popt_gom[2]:.4f}")
except RuntimeError:
    popt_gom = p0
    print("  Warning: Gompertz fit did not converge, using initial guess")

gom_fn = lambda x: _gompertz(x, *popt_gom)
y0g, y1g = float(gom_fn(ranks[0])), float(gom_fn(ranks[-1]))
inv_gom = make_brentq_inv(gom_fn, min(y0g, y1g), max(y0g, y1g))
make_figure("Gompertz", gom_fn, "Gompertz curve", inv_gom, OUT_DIR / "fig_06_gompertz.png")

# ── Method 7: Richards (Generalized Logistic) ─────────────────────────────────
print("\n[7/11] Richards (Generalized Logistic)")

def _richards(x, L, k, x0, v, b):
    return L / (1 + np.exp(-k * (x - x0))) ** (1.0 / v) + b

p0 = [float(medians[-1] - medians[0]), 0.2, float(G * 3), 0.5, float(medians[0])]
try:
    popt_ric, _ = curve_fit(_richards, ranks, medians, p0=p0, maxfev=20000,
                            bounds=([0, 0, 0, 0.01, -np.inf],
                                    [np.inf, np.inf, np.inf, np.inf, np.inf]))
    print(f"  L={popt_ric[0]:.3e}  k={popt_ric[1]:.4f}  x0={popt_ric[2]:.3f}"
          f"  v={popt_ric[3]:.4f}  b={popt_ric[4]:.3e}")
except RuntimeError:
    popt_ric = p0
    print("  Warning: Richards fit did not converge, using initial guess")

ric_fn = lambda x: _richards(x, *popt_ric)
y0r, y1r = float(ric_fn(ranks[0])), float(ric_fn(ranks[-1]))
inv_ric = make_brentq_inv(ric_fn, min(y0r, y1r), max(y0r, y1r))
make_figure("Richards", ric_fn, "Richards / generalized logistic", inv_ric,
            OUT_DIR / "fig_07_richards.png")

# ── Method 8: Weibull CDF ─────────────────────────────────────────────────────
print("\n[8/11] Weibull CDF")

def _weibull(x, L, lam, k, b):
    return L * (1.0 - np.exp(-(x / lam) ** k)) + b

p0 = [float(medians[-1] - medians[0]), float(G / 2), 2.5, float(medians[0])]
try:
    popt_wei, _ = curve_fit(_weibull, ranks, medians, p0=p0, maxfev=20000,
                            bounds=([0, 1e-6, 1e-6, -np.inf],
                                    [np.inf, np.inf, np.inf, np.inf]))
    print(f"  L={popt_wei[0]:.3e}  lam={popt_wei[1]:.4f}  k={popt_wei[2]:.4f}  b={popt_wei[3]:.3e}")
except RuntimeError:
    popt_wei = p0
    print("  Warning: Weibull fit did not converge, using initial guess")

wei_fn = lambda x: _weibull(x, *popt_wei)
y0w, y1w = float(wei_fn(ranks[0])), float(wei_fn(ranks[-1]))
inv_wei = make_brentq_inv(wei_fn, min(y0w, y1w), max(y0w, y1w))
make_figure("Weibull CDF", wei_fn, "Weibull CDF", inv_wei, OUT_DIR / "fig_08_weibull.png")

# ── Method 9: Power Law ───────────────────────────────────────────────────────
print("\n[9/11] Power Law")

def _powerlaw(x, a, b, c):
    return a * x ** b + c

p0 = [1e-4, 3.0, float(medians[0])]
try:
    popt_pow, _ = curve_fit(_powerlaw, ranks, medians, p0=p0, maxfev=20000,
                            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
    print(f"  a={popt_pow[0]:.3e}  b={popt_pow[1]:.4f}  c={popt_pow[2]:.3e}")
except RuntimeError:
    popt_pow = p0
    print("  Warning: Power law fit did not converge, using initial guess")

pow_fn = lambda x: _powerlaw(x, *popt_pow)
y0pw, y1pw = float(pow_fn(ranks[0])), float(pow_fn(ranks[-1]))
inv_pow = make_brentq_inv(pow_fn, min(y0pw, y1pw), max(y0pw, y1pw))
make_figure("Power Law", pow_fn, "Power law  y = a·xᵇ + c", inv_pow,
            OUT_DIR / "fig_09_powerlaw.png")

# ── Method 10: Exponential ────────────────────────────────────────────────────
print("\n[10/11] Exponential")

def _exponential(x, a, b, c):
    return a * np.exp(b * x) + c

p0 = [1e-6, 0.4, float(medians[0])]
try:
    popt_exp, _ = curve_fit(_exponential, ranks, medians, p0=p0, maxfev=20000)
    print(f"  a={popt_exp[0]:.3e}  b={popt_exp[1]:.4f}  c={popt_exp[2]:.3e}")
except RuntimeError:
    popt_exp = p0
    print("  Warning: Exponential fit did not converge, using initial guess")

exp_fn = lambda x: _exponential(x, *popt_exp)
y0e, y1e = float(exp_fn(ranks[0])), float(exp_fn(ranks[-1]))
inv_exp = make_brentq_inv(exp_fn, min(y0e, y1e), max(y0e, y1e))
make_figure("Exponential", exp_fn, "Exponential  y = a·eᵇˣ + c", inv_exp,
            OUT_DIR / "fig_10_exponential.png")

# ── Method 11: Hill ───────────────────────────────────────────────────────────
print("\n[11/11] Hill")

def _hill(x, L, K, n, b):
    return L * x ** n / (K ** n + x ** n) + b

p0 = [float(medians[-1] - medians[0]), float(G / 2), 3.0, float(medians[0])]
try:
    popt_hil, _ = curve_fit(_hill, ranks, medians, p0=p0, maxfev=20000,
                            bounds=([0, 1e-6, 1e-6, -np.inf],
                                    [np.inf, np.inf, np.inf, np.inf]))
    print(f"  L={popt_hil[0]:.3e}  K={popt_hil[1]:.4f}  n={popt_hil[2]:.4f}  b={popt_hil[3]:.3e}")
except RuntimeError:
    popt_hil = p0
    print("  Warning: Hill fit did not converge, using initial guess")

hil_fn = lambda x: _hill(x, *popt_hil)
y0h, y1h = float(hil_fn(ranks[0])), float(hil_fn(ranks[-1]))
inv_hil = make_brentq_inv(hil_fn, min(y0h, y1h), max(y0h, y1h))
make_figure("Hill", hil_fn, "Hill function  y = L·xⁿ/(Kⁿ+xⁿ) + b", inv_hil,
            OUT_DIR / "fig_11_hill.png")

print(f"\n✓ All 11 figures saved to {OUT_DIR}")
