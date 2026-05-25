"""
Regenerate engineered feature distribution and scatter figures for Dataset 2.
  fig_02_engineered_dist.png   — 4×3 grid of distributions
  fig_04_engineered_scatter.png — 4×3 grid of scatter vs g-limit
Output: figures/v2/
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

ENG_LABELS = {
    "tip_deflection_slope":           "Tip Deflection Slope (m/N)",
    "tip_per_g_at_failure":           "Tip Deflection per g at Failure (m/g)",
    "avg_strain_at_failure":          "Avg Strain at Failure (m/m)",
    "avg_strain_slope":               "Avg Strain Slope ((m/m)/N)",
    "strain_energy_at_failure":       "Strain Energy Proxy at Failure (Pa)",
    "strain_energy_slope":            "Strain Energy Slope (Pa/N)",
    "max_vm_stress_at_failure":       "Max von Mises Stress at Failure (Pa)",
    "k_spring":                       "Effective Stiffness k (N/m)",
    "inv_tip_per_g_at_failure":       "Inv. Tip Deflection per g (g/m)",
    "inv_tip_deflection_slope":       "Inv. Tip Deflection Slope (N/m)",
    "inv_max_vm_stress_at_failure":   "Inv. Max VM Stress at Failure (1/Pa)",
    "sqrt_strain_energy_at_failure":  "Sqrt Strain Energy at Failure (Pa¹ᐟ²)",
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

NCOLS, NROWS = 4, 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def dist_panel(ax, vals, color, title, xlabel):
    finite = vals[np.isfinite(vals)]
    ax.hist(finite, bins=30, color=color, alpha=0.55, density=True,
            edgecolor="white", linewidth=0.4)
    kde_x = np.linspace(finite.min(), finite.max(), 300)
    ax.plot(kde_x, stats.gaussian_kde(finite)(kde_x), color=color, linewidth=1.8)
    ax.axvline(np.median(finite), color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    stats_txt = (
        f"μ={np.mean(finite):.3g}\n"
        f"σ={np.std(finite):.3g}\n"
        f"IQR={np.percentile(finite,75)-np.percentile(finite,25):.3g}"
    )
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"))
    ax.set_title(title, pad=3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)


def scatter_panel(ax, x, y, color, title, xlabel):
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    ax.scatter(xm, ym, color=color, alpha=0.35, s=8, linewidths=0)
    xfit = np.linspace(xm.min(), xm.max(), 200)
    slope, intercept, *_ = stats.linregress(xm, ym)
    ax.plot(xfit, slope * xfit + intercept, color=color, linewidth=1.6)
    r, pval = stats.pearsonr(xm, ym)
    r2 = r2_score(ym, LinearRegression().fit(xm.reshape(-1,1), ym).predict(xm.reshape(-1,1)))
    pstr = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
    ax.text(0.97, 0.97, f"r = {r:.3f}\nR² = {r2:.3f}\n{pstr}",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="none"))
    ax.set_title(title, pad=3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("g-limit", fontsize=8)
    ax.tick_params(labelsize=7)


# ── Fig 2: Distributions ──────────────────────────────────────────────────────

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, NROWS * 3.2))
axes = axes.flatten()

for i, feat in enumerate(ENGINEERED):
    color = COLORS[GROUP_MAP[feat]]
    label = ENG_LABELS.get(feat, feat)
    dist_panel(axes[i], df[feat].to_numpy(), color, feat.replace("_", " "), label)

fig.legend(handles=legend_handles, loc="lower right", ncol=2, fontsize=9,
           framealpha=0.9, title="Feature Group")
fig.suptitle("AERO 489 — Engineered Feature Distributions (Dataset 2)", fontsize=13, fontweight="bold")
fig.tight_layout()
out2 = FIGURES_DIR / "fig_02_engineered_dist.png"
fig.savefig(out2, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out2}")


# ── Fig 4: Scatter vs g-limit ─────────────────────────────────────────────────

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, NROWS * 3.5))
axes = axes.flatten()

for i, feat in enumerate(ENGINEERED):
    color = COLORS[GROUP_MAP[feat]]
    label = ENG_LABELS.get(feat, feat)
    scatter_panel(axes[i], df[feat].to_numpy(), y, color, feat.replace("_", " "), label)

fig.legend(handles=legend_handles, loc="lower right", ncol=2, fontsize=9,
           framealpha=0.9, title="Feature Group")
fig.suptitle("AERO 489 — Engineered Features vs g-limit (Dataset 2)", fontsize=13, fontweight="bold")
fig.tight_layout()
out4 = FIGURES_DIR / "fig_04_engineered_scatter.png"
fig.savefig(out4, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out4}")
