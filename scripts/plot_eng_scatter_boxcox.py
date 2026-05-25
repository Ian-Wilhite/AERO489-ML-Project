"""
Remake fig_04 scatter plots using per-feature optimal Box-Cox transform.
For each engineered feature: sweep λ ∈ [-3, 3], pick λ* that maximises R² vs g_limit,
apply x^λ* (or log(x) at λ≈0), then scatter against g_limit.
Output: figures/v2/fig_04_engineered_scatter_boxcox.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

FIGURES_DIR = Path("figures/v2")
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

# ── Data ──────────────────────────────────────────────────────────────────────

df = pl.read_parquet("features/v2/features_scalar.parquet").to_pandas()
y  = df["g_limit"].to_numpy()

ENGINEERED = [
    "tip_deflection_slope",
    "tip_per_g_at_failure",
    "avg_strain_at_failure",
    "avg_strain_slope",
    "strain_energy_at_failure",
    "strain_energy_slope",
    "max_vm_stress_at_failure",
    "k_spring",
    "inv_tip_per_g_at_failure",
    "inv_tip_deflection_slope",
    "inv_max_vm_stress_at_failure",
    "sqrt_strain_energy_at_failure",
]

ENG_TITLES = {
    "tip_deflection_slope":           "Tip Deflection Slope",
    "tip_per_g_at_failure":           "Tip Deflection per g",
    "avg_strain_at_failure":          "Avg Strain at Failure",
    "avg_strain_slope":               "Avg Strain Slope",
    "strain_energy_at_failure":       "Strain Energy at Failure",
    "strain_energy_slope":            "Strain Energy Slope",
    "max_vm_stress_at_failure":       "Max VM Stress at Failure",
    "k_spring":                       "Effective Stiffness k",
    "inv_tip_per_g_at_failure":       "Inv. Tip Defl. per g",
    "inv_tip_deflection_slope":       "Inv. Tip Defl. Slope",
    "inv_max_vm_stress_at_failure":   "Inv. Max VM Stress",
    "sqrt_strain_energy_at_failure":  "Sqrt Strain Energy",
}

GROUP_MAP = {
    "tip_deflection_slope":           "tip",
    "tip_per_g_at_failure":           "tip",
    "avg_strain_at_failure":          "strain_avg",
    "avg_strain_slope":               "strain_avg",
    "strain_energy_at_failure":       "strain_eng",
    "strain_energy_slope":            "strain_eng",
    "max_vm_stress_at_failure":       "meta",
    "k_spring":                       "tip",
    "inv_tip_per_g_at_failure":       "derived",
    "inv_tip_deflection_slope":       "derived",
    "inv_max_vm_stress_at_failure":   "derived",
    "sqrt_strain_energy_at_failure":  "derived",
}

COLORS = {
    "tip":        "#d7191c",
    "strain_avg": "#fdae61",
    "strain_eng": "#1a9641",
    "meta":       "#756bb1",
    "derived":    "#17becf",
}

GROUP_LABELS = {
    "tip":        "Tip Deflection",
    "strain_avg": "Avg Strain",
    "strain_eng": "Strain Energy",
    "meta":       "Metadata / Stress",
    "derived":    "Inverse / Sqrt Transforms",
}

legend_handles = [
    mpatches.Patch(fc=COLORS[g], label=GROUP_LABELS[g])
    for g in ["tip", "strain_avg", "strain_eng", "meta", "derived"]
]

LAMBDA_GRID = np.linspace(-15, 15, 3000)

def _minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

def apply_boxcox(x_raw, lam):
    if abs(lam) < 1e-4:
        return np.log(x_raw)
    return np.power(x_raw, lam)

def best_lambda(x_raw):
    """Return (λ*, R²*) maximising R² of x^λ vs y."""
    best_lam, best_r2 = 1.0, -np.inf
    for lam in LAMBDA_GRID:
        xt = apply_boxcox(x_raw, lam)
        mask = np.isfinite(xt) & np.isfinite(y)
        if mask.sum() < 5:
            continue
        xm, ym = xt[mask].reshape(-1, 1), y[mask]
        r2 = r2_score(ym, LinearRegression().fit(xm, ym).predict(xm))
        if r2 > best_r2:
            best_r2 = r2
            best_lam = lam
    return best_lam, best_r2

def lambda_label(lam):
    if abs(lam) < 1e-4:
        return "log(x)"
    if abs(lam - 0.5) < 0.03:
        return "√x"
    if abs(lam - 1.0) < 0.03:
        return "x (raw)"
    if abs(lam - 2.0) < 0.03:
        return "x²"
    if abs(lam + 1.0) < 0.03:
        return "1/x"
    return f"x^{lam:.2f}"


# ── Figure ────────────────────────────────────────────────────────────────────

NCOLS, NROWS = 4, 3
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, NROWS * 3.5))
axes = axes.flatten()

for i, feat in enumerate(ENGINEERED):
    ax    = axes[i]
    color = COLORS[GROUP_MAP[feat]]
    x_raw = df[feat].to_numpy().astype(float)

    lam, best_r2 = best_lambda(x_raw)
    xt = apply_boxcox(x_raw, lam)

    mask = np.isfinite(xt) & np.isfinite(y)
    xm, ym = xt[mask], y[mask]

    ax.scatter(xm, ym, color=color, alpha=0.35, s=8, linewidths=0)

    xfit = np.linspace(xm.min(), xm.max(), 200)
    slope, intercept, *_ = stats.linregress(xm, ym)
    ax.plot(xfit, slope * xfit + intercept, color=color, linewidth=1.6)

    r, pval = stats.pearsonr(xm, ym)
    pstr = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
    ann = f"λ* = {lam:.2f}  ({lambda_label(lam)})\nr = {r:.3f}\nR² = {best_r2:.3f}\n{pstr}"
    ax.text(0.97, 0.97, ann, transform=ax.transAxes,
            fontsize=7, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"))

    ax.set_title(ENG_TITLES.get(feat, feat), pad=3)
    ax.set_xlabel(f"Box-Cox transform  (λ*={lam:.2f})", fontsize=8)
    ax.set_ylabel("g-limit", fontsize=8)
    ax.tick_params(labelsize=7)

fig.legend(handles=legend_handles, loc="lower right", ncol=2, fontsize=9,
           framealpha=0.9, title="Feature Group")
fig.suptitle(
    "AERO 489 — Engineered Features vs g-limit  |  Box-Cox Optimal Transform (Dataset 2)",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
out = FIGURES_DIR / "fig_04_engineered_scatter_boxcox.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
