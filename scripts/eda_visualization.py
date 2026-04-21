"""
AERO 489 — EDA Visualization
Produces 6 figures showing variance, inter-variable relations, and univariate
predictive value for all raw and engineered features.

Figures saved to figures/:
  fig_01_target_dist.png         — g_limit distribution
  fig_02_engineered_dist.png     — Engineered feature distributions (7 panels)
  fig_03_raw_strain_dist.png     — Per-node strain-at-failure distributions (24 panels)
  fig_04_engineered_scatter.png  — Engineered features vs g_limit with regression
  fig_05_raw_strain_scatter.png  — Node strains vs g_limit (4×6 grid)
  fig_06_predictive_summary.png  — All features ranked by |Pearson r|
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── Config ───────────────────────────────────────────────────────────────────

FEATURES_PATH = Path("features/features_scalar.parquet")
FIGURES_DIR   = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

TARGET = "g_limit"

ENGINEERED = [
    "tip_deflection_at_failure",
    "tip_deflection_slope",
    "tip_per_g_at_failure",
    "avg_strain_at_failure",
    "avg_strain_slope",
    "strain_energy_at_failure",
    "strain_energy_slope",
    "max_vm_stress_at_failure",
    "n_steps",
]

# Human-readable labels for engineered features
ENG_LABELS = {
    "tip_deflection_at_failure": "Tip Deflection\nat Failure (m)",
    "tip_deflection_slope":      "Tip Deflection\nSlope (m/N)",
    "tip_per_g_at_failure":      "Tip Deflection\nper g at Failure (m/g)",
    "avg_strain_at_failure":     "Avg Strain\nat Failure (m/m)",
    "avg_strain_slope":          "Avg Strain\nSlope ((m/m)/N)",
    "strain_energy_at_failure":  "Strain Energy Proxy\nat Failure (Pa)",
    "strain_energy_slope":       "Strain Energy\nSlope (Pa/N)",
    "max_vm_stress_at_failure":  "Max von Mises Stress\nat Failure (Pa)",
    "n_steps":                   "Converged Load\nSteps (#)",
}

# Group color palette (consistent across all figures)
COLORS = {
    "target":     "#2c7bb6",
    "tip":        "#d7191c",
    "strain_avg": "#fdae61",
    "strain_eng": "#1a9641",
    "meta":       "#756bb1",
    "node":       "#636363",
}

GROUP_MAP = {
    "tip_deflection_at_failure":  "tip",
    "tip_deflection_slope":       "tip",
    "tip_per_g_at_failure":       "tip",
    "avg_strain_at_failure":      "strain_avg",
    "avg_strain_slope":           "strain_avg",
    "strain_energy_at_failure":   "strain_eng",
    "strain_energy_slope":        "strain_eng",
    "max_vm_stress_at_failure":   "meta",
    "n_steps":                    "meta",
}

GROUP_LABELS = {
    "tip":        "Tip Deflection",
    "strain_avg": "Avg Strain",
    "strain_eng": "Strain Energy",
    "meta":       "Metadata / Stress",
    "node":       "Node Strain at Failure",
}

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi":   150,
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def pearson(x, y):
    """Return (r, p) Pearson correlation, skipping NaNs."""
    mask = np.isfinite(x) & np.isfinite(y)
    return stats.pearsonr(x[mask], y[mask])


def univariate_r2(x, y):
    """R² of a simple OLS fit of y ~ x."""
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask].reshape(-1, 1), y[mask]
    return r2_score(ym, LinearRegression().fit(xm, ym).predict(xm))


def dist_panel(ax, vals, color, title, xlabel):
    """Histogram + KDE + variance stats on one axes."""
    finite = vals[np.isfinite(vals)]
    ax.hist(finite, bins=30, color=color, alpha=0.55, density=True, edgecolor="white", linewidth=0.4)
    kde_x = np.linspace(finite.min(), finite.max(), 300)
    kde   = stats.gaussian_kde(finite)
    ax.plot(kde_x, kde(kde_x), color=color, linewidth=1.8)
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


def scatter_panel(ax, x, y, color, title, xlabel, r, pval, r2):
    """Scatter + regression line + stats annotation."""
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    ax.scatter(xm, ym, color=color, alpha=0.35, s=8, linewidths=0)
    xfit = np.linspace(xm.min(), xm.max(), 200)
    slope, intercept, *_ = stats.linregress(xm, ym)
    ax.plot(xfit, slope * xfit + intercept, color=color, linewidth=1.6)
    pstr = f"p<0.001" if pval < 0.001 else f"p={pval:.3f}"
    ax.text(0.97, 0.97,
            f"r = {r:.3f}\nR² = {r2:.3f}\n{pstr}",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="none"))
    ax.set_title(title, pad=3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("g-limit", fontsize=8)
    ax.tick_params(labelsize=7)


# ── Load data ─────────────────────────────────────────────────────────────────

df_pl = pl.read_parquet(FEATURES_PATH)
df    = df_pl.to_pandas()

node_cols = [c for c in df.columns if c.startswith("Strain_Node_") and c.endswith("_failure")]
y = df[TARGET].to_numpy()


# ── Figure 1: g_limit distribution ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
dist_panel(ax, y, COLORS["target"], "g-limit Distribution (Target Variable)", "g-limit")
ax.set_ylabel("Density")
fig.suptitle("AERO 489 — Target Variable: g-limit", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_01_target_dist.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_01_target_dist.png")


# ── Figure 2: Engineered feature distributions ───────────────────────────────

ncols = 3
nrows = int(np.ceil(len(ENGINEERED) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2))
axes = axes.flatten()

for i, feat in enumerate(ENGINEERED):
    grp   = GROUP_MAP.get(feat, "meta")
    color = COLORS[grp]
    label = ENG_LABELS.get(feat, feat)
    dist_panel(axes[i], df[feat].to_numpy(), color, feat.replace("_", " "), label)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

# Legend
from matplotlib.patches import Patch
handles = [Patch(fc=COLORS[g], label=GROUP_LABELS[g])
           for g in ["tip", "strain_avg", "strain_eng", "meta"]]
fig.legend(handles=handles, loc="lower right", ncol=2, fontsize=9,
           framealpha=0.9, title="Feature Group")
fig.suptitle("AERO 489 — Engineered Feature Distributions", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_02_engineered_dist.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_02_engineered_dist.png")


# ── Figure 3: Per-node strain distributions ───────────────────────────────────

ncols = 6
nrows = 4  # 24 nodes
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(node_cols):
    node_id = col.replace("Strain_Node_", "").replace("_failure", "")
    dist_panel(axes[i], df[col].to_numpy(), COLORS["node"],
               f"Node {node_id}", "Strain (m/m)")

fig.suptitle("AERO 489 — Per-Node Strain at Failure Distributions", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_03_raw_strain_dist.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_03_raw_strain_dist.png")


# ── Figure 4: Engineered features vs g_limit ─────────────────────────────────

ncols = 3
nrows = int(np.ceil(len(ENGINEERED) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
axes = axes.flatten()

for i, feat in enumerate(ENGINEERED):
    grp   = GROUP_MAP.get(feat, "meta")
    color = COLORS[grp]
    label = ENG_LABELS.get(feat, feat)
    x     = df[feat].to_numpy()
    r, pv = pearson(x, y)
    r2    = univariate_r2(x, y)
    scatter_panel(axes[i], x, y, color, feat.replace("_", " "), label, r, pv, r2)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.legend(handles=handles, loc="lower right", ncol=2, fontsize=9,
           framealpha=0.9, title="Feature Group")
fig.suptitle("AERO 489 — Engineered Features vs g-limit (Univariate)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_04_engineered_scatter.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_04_engineered_scatter.png")


# ── Figure 5: Node strains vs g_limit ────────────────────────────────────────

ncols = 6
nrows = 4
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
axes = axes.flatten()

for i, col in enumerate(node_cols):
    node_id = col.replace("Strain_Node_", "").replace("_failure", "")
    x       = df[col].to_numpy()
    r, pv   = pearson(x, y)
    r2      = univariate_r2(x, y)
    scatter_panel(axes[i], x, y, COLORS["node"],
                  f"Node {node_id}", "Strain (m/m)", r, pv, r2)

fig.suptitle("AERO 489 — Per-Node Strain at Failure vs g-limit", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_05_raw_strain_scatter.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_05_raw_strain_scatter.png")


# ── Figure 6: Predictive value summary (ranked |r|) ──────────────────────────

all_features = ENGINEERED + node_cols
records = []
for feat in all_features:
    x     = df[feat].to_numpy()
    r, pv = pearson(x, y)
    r2    = univariate_r2(x, y)
    grp   = GROUP_MAP.get(feat, "node")
    label = feat.replace("Strain_Node_", "N").replace("_failure", "").replace("_", " ")
    records.append({"feat": feat, "label": label, "r": r, "abs_r": abs(r), "r2": r2, "group": grp})

records.sort(key=lambda d: d["abs_r"], reverse=True)

labels  = [d["label"]  for d in records]
abs_rs  = [d["abs_r"]  for d in records]
r2s     = [d["r2"]     for d in records]
bar_colors = [COLORS[d["group"]] for d in records]

n = len(records)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

x_idx = np.arange(n)

ax1.bar(x_idx, abs_rs, color=bar_colors, edgecolor="white", linewidth=0.3)
ax1.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax1.axhline(0.8, color="gray", linewidth=0.8, linestyle=":",  alpha=0.6)
ax1.set_ylabel("|Pearson r|")
ax1.set_ylim(0, 1.05)
ax1.set_title("Univariate Correlation with g-limit  (|Pearson r|)")
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
ax1.grid(axis="y", alpha=0.3)

ax2.bar(x_idx, r2s, color=bar_colors, edgecolor="white", linewidth=0.3)
ax2.axhline(0.25, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax2.axhline(0.64, color="gray", linewidth=0.8, linestyle=":",  alpha=0.6)
ax2.set_ylabel("Univariate R²")
ax2.set_ylim(0, 1.05)
ax2.set_title("Univariate R² (simple OLS) with g-limit")
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
ax2.grid(axis="y", alpha=0.3)

ax2.set_xticks(x_idx)
ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)

# Legend
all_groups = ["tip", "strain_avg", "strain_eng", "meta", "node"]
handles_all = [Patch(fc=COLORS[g], label=GROUP_LABELS[g]) for g in all_groups]
fig.legend(handles=handles_all, loc="upper right", ncol=1, fontsize=9,
           framealpha=0.9, title="Feature Group", bbox_to_anchor=(1.0, 0.98))

fig.suptitle("AERO 489 — Univariate Predictive Value of All Features", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_06_predictive_summary.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_06_predictive_summary.png")

print("\nAll figures written to figures/")
