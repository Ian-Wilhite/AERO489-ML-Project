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

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── Config ───────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1",
                   help="Dataset version tag, e.g. v1, v2a, v2b")
    return p.parse_args()

_args = _parse_args()
FEATURES_PATH = Path(f"features-{_args.version}/features_scalar.parquet")
FIGURES_DIR   = Path(f"figures-{_args.version}")
FIGURES_DIR.mkdir(exist_ok=True)
print(f"Version: {_args.version}  |  features: {FEATURES_PATH}  |  figures: {FIGURES_DIR}")

TARGET = "g_limit"

ENGINEERED = [
    "tip_deflection_slope",
    "tip_per_g_at_failure",
    "avg_strain_at_failure",
    "avg_strain_slope",
    "strain_energy_at_failure",
    "strain_energy_slope",
    "max_vm_stress_at_failure",
    "k_spring",
    # Feature D: inverse and sqrt transforms
    "inv_tip_per_g_at_failure",
    "inv_tip_deflection_slope",
    "inv_max_vm_stress_at_failure",
    "sqrt_strain_energy_at_failure",
]

# Human-readable labels for engineered features
ENG_LABELS = {
    "tip_deflection_at_failure":      "Tip Deflection\nat Failure (m)",
    "tip_deflection_slope":           "Tip Deflection\nSlope (m/N)",
    "tip_per_g_at_failure":           "Tip Deflection\nper g at Failure (m/g)",
    "avg_strain_at_failure":          "Avg Strain\nat Failure (m/m)",
    "avg_strain_slope":               "Avg Strain\nSlope ((m/m)/N)",
    "strain_energy_at_failure":       "Strain Energy Proxy\nat Failure (Pa)",
    "strain_energy_slope":            "Strain Energy\nSlope (Pa/N)",
    "max_vm_stress_at_failure":       "Max von Mises Stress\nat Failure (Pa)",
    "k_spring":                       "Effective Stiffness\nk (N/m)",
    "n_steps":                        "Converged Load\nSteps (#)",
    "inv_tip_per_g_at_failure":       "Inv. Tip Defl.\nper g (g/m)",
    "inv_tip_deflection_slope":       "Inv. Tip Defl.\nSlope (N/m)",
    "inv_max_vm_stress_at_failure":   "Inv. Max VM Stress\nat Failure (1/Pa)",
    "sqrt_strain_energy_at_failure":  "Sqrt Strain Energy\nat Failure (Pa^0.5)",
}

# Group color palette (consistent across all figures)
COLORS = {
    "target":     "#2c7bb6",
    "tip":        "#d7191c",
    "strain_avg": "#fdae61",
    "strain_eng": "#1a9641",
    "meta":       "#756bb1",
    "derived":    "#17becf",
    "node":       "#636363",
}

GROUP_MAP = {
    "tip_deflection_at_failure":      "tip",
    "tip_deflection_slope":           "tip",
    "tip_per_g_at_failure":           "tip",
    "avg_strain_at_failure":          "strain_avg",
    "avg_strain_slope":               "strain_avg",
    "strain_energy_at_failure":       "strain_eng",
    "strain_energy_slope":            "strain_eng",
    "max_vm_stress_at_failure":       "meta",
    "k_spring":                       "tip",
    "n_steps":                        "meta",
    "inv_tip_per_g_at_failure":       "derived",
    "inv_tip_deflection_slope":       "derived",
    "inv_max_vm_stress_at_failure":   "derived",
    "sqrt_strain_energy_at_failure":  "derived",
}

GROUP_LABELS = {
    "tip":        "Tip Deflection",
    "strain_avg": "Avg Strain",
    "strain_eng": "Strain Energy",
    "meta":       "Metadata / Stress",
    "derived":    "Inverse / Sqrt Transforms",
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

_TRANSFORM_PREFIXES = ("ln_", "log10_", "exp_", "pow10_")
_non_node = set(ENGINEERED + ["g_limit", "sim_id", "n_steps", "RF_failure",
                               "tip_deflection_at_failure", "max_vm_stress_at_failure"])
node_cols = [
    c for c in df.columns
    if c.endswith("_failure")
    and c not in _non_node
    and not any(c.startswith(p) for p in _TRANSFORM_PREFIXES)
]
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
           for g in ["tip", "strain_avg", "strain_eng", "meta", "derived"]]
fig.legend(handles=handles, loc="lower right", ncol=2, fontsize=9,
           framealpha=0.9, title="Feature Group")
fig.suptitle("AERO 489 — Engineered Feature Distributions", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_02_engineered_dist.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_02_engineered_dist.png")


# ── Figure 3: Per-node strain distributions ───────────────────────────────────

ncols = 6
nrows = int(np.ceil(len(node_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(node_cols):
    node_id = col.replace("Strain_Node_", "N").replace("_failure", "").replace("Node_at_", "")
    dist_panel(axes[i], df[col].to_numpy(), COLORS["node"],
               f"Node {node_id[:30]}", "Strain (m/m)")

for j in range(len(node_cols), len(axes)):
    axes[j].set_visible(False)

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


# ── Figure 4b: log10(Engineered features) vs g_limit ─────────────────────────

LOG_ENGINEERED = [f"log10_{f}" for f in ENGINEERED]

LOG_LABELS = {f"log10_{f}": f"log₁₀({ENG_LABELS.get(f, f).replace(chr(10), ' ')})"
              for f in ENGINEERED}

ncols = 3
nrows = int(np.ceil(len(LOG_ENGINEERED) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
axes = axes.flatten()

for i, feat in enumerate(LOG_ENGINEERED):
    base_feat = feat[len("log10_"):]
    grp   = GROUP_MAP.get(base_feat, "meta")
    color = COLORS[grp]
    label = LOG_LABELS.get(feat, feat)
    x     = df[feat].to_numpy()
    r, pv = pearson(x, y)
    r2    = univariate_r2(x, y)
    scatter_panel(axes[i], x, y, color, feat.replace("_", " "), label, r, pv, r2)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.legend(handles=handles, loc="lower right", ncol=2, fontsize=9,
           framealpha=0.9, title="Feature Group")
fig.suptitle("AERO 489 — log₁₀(Engineered Features) vs g-limit (Univariate)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_04b_log_scatter.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_04b_log_scatter.png")


# ── Figure 5: Node strains vs g_limit ────────────────────────────────────────

ncols = 6
nrows = int(np.ceil(len(node_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
axes = axes.flatten()

for i, col in enumerate(node_cols):
    node_id = col.replace("Strain_Node_", "N").replace("_failure", "").replace("Node_at_", "")
    x       = df[col].to_numpy()
    r, pv   = pearson(x, y)
    r2      = univariate_r2(x, y)
    scatter_panel(axes[i], x, y, COLORS["node"],
                  f"Node {node_id[:30]}", "Strain (m/m)", r, pv, r2)

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
    label = feat.replace("Strain_Node_", "N").replace("Node_at_", "").replace("_failure", "").replace("_", " ")
    label = label[:35]
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
all_groups = ["tip", "strain_avg", "strain_eng", "meta", "derived", "node"]
handles_all = [Patch(fc=COLORS[g], label=GROUP_LABELS[g]) for g in all_groups]
fig.legend(handles=handles_all, loc="upper right", ncol=1, fontsize=9,
           framealpha=0.9, title="Feature Group", bbox_to_anchor=(1.0, 0.98))

fig.suptitle("AERO 489 — Univariate Predictive Value of All Features", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_06_predictive_summary.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_06_predictive_summary.png")


# ── Figure 7: R² heatmap — feature × transform ───────────────────────────────

import seaborn as sns

def _minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

TRANSFORMS = {
    "log10":  lambda x: np.log10(x),
    "ln":     lambda x: np.log(x),
    "raw":    lambda x: x,
    "e^x":    lambda x: np.exp(_minmax(x)),
    "10^x":   lambda x: np.power(10.0, _minmax(x)),
}

def _r2_transform(x_raw, fn):
    """Apply transform, return R² vs y, or NaN if result is invalid."""
    xt = fn(x_raw)
    mask = np.isfinite(xt) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    xm, ym = xt[mask].reshape(-1, 1), y[mask]
    return r2_score(ym, LinearRegression().fit(xm, ym).predict(xm))

# Short display labels for the 12 base features
FEAT_SHORT = {
    "tip_deflection_slope":          "tip_defl_slope",
    "tip_per_g_at_failure":          "tip_per_g",
    "avg_strain_at_failure":         "avg_strain",
    "avg_strain_slope":              "avg_strain_slope",
    "strain_energy_at_failure":      "strain_energy",
    "strain_energy_slope":           "strain_energy_slope",
    "max_vm_stress_at_failure":      "max_vm_stress",
    "k_spring":                      "k_spring",
    "inv_tip_per_g_at_failure":      "inv_tip_per_g",
    "inv_tip_deflection_slope":      "inv_tip_defl_slope",
    "inv_max_vm_stress_at_failure":  "inv_max_vm_stress",
    "sqrt_strain_energy_at_failure": "sqrt_strain_energy",
}

r2_data = {}
for feat in ENGINEERED:
    x_raw = df[feat].to_numpy().astype(float)
    row = {t_name: _r2_transform(x_raw, t_fn) for t_name, t_fn in TRANSFORMS.items()}
    r2_data[FEAT_SHORT.get(feat, feat)] = row

r2_df = pl.DataFrame({
    "feature": list(r2_data.keys()),
    **{t: [r2_data[f][t] for f in r2_data] for t in TRANSFORMS},
}).to_pandas().set_index("feature")

# Sort rows by best R² across any transform
r2_df = r2_df.loc[r2_df.max(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    r2_df.astype(float),
    ax=ax,
    cmap="YlGnBu",
    vmin=0, vmax=1,
    annot=True, fmt=".3f",
    linewidths=0.5,
    linecolor="white",
    mask=r2_df.isna(),
    cbar_kws={"label": "Univariate R²  (OLS, higher = more predictive)"},
)

# Star the best transform per feature
for i, feat in enumerate(r2_df.index):
    row_vals = r2_df.loc[feat]
    if row_vals.notna().any():
        best_col = row_vals.idxmax()
        j = list(r2_df.columns).index(best_col)
        ax.add_patch(plt.Rectangle(
            (j, i), 1, 1,
            fill=False, edgecolor="crimson", lw=2.5, zorder=5,
        ))

ax.set_title(
    "Univariate R² — Feature × Transform\n"
    "(e^x and 10^x applied to min-max normalised feature; red border = best per feature)",
    fontsize=12,
)
ax.set_xlabel("Transform")
ax.set_ylabel("Feature")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_07_transform_r2_heatmap.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_07_transform_r2_heatmap.png")

# ── Figure 8: Box-Cox λ sweep ─────────────────────────────────────────────────
# x^λ (λ≠0) or log(x) (λ→0) swept from -3 to +3
# Shows which power transform maximises R² vs g_limit for each feature.

LAMBDA_GRID = np.linspace(-3, 3, 400)

# Reference λ values to annotate
LAMBDA_REF = {
    -2:   ("x⁻²",  "#d62728"),
    -1:   ("1/x",   "#ff7f0e"),
    -0.5: ("1/√x",  "#9467bd"),
     0:   ("log",   "#2ca02c"),
     0.5: ("√x",    "#17becf"),
     1:   ("raw",   "#7f7f7f"),
     2:   ("x²",    "#bcbd22"),
}

def _boxcox_r2_curve(x_raw):
    """Return R² at each λ in LAMBDA_GRID."""
    curve = np.empty(len(LAMBDA_GRID))
    for k, lam in enumerate(LAMBDA_GRID):
        if abs(lam) < 1e-4:
            xt = np.log(x_raw)
        else:
            xt = np.power(x_raw, lam)
        mask = np.isfinite(xt) & np.isfinite(y)
        if mask.sum() < 5:
            curve[k] = np.nan
            continue
        xm, ym = xt[mask].reshape(-1, 1), y[mask]
        curve[k] = r2_score(ym, LinearRegression().fit(xm, ym).predict(xm))
    return curve

ncols, nrows = 4, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows))
axes = axes.flatten()

for i, feat in enumerate(ENGINEERED):
    ax = axes[i]
    x_raw = df[feat].to_numpy().astype(float)
    curve = _boxcox_r2_curve(x_raw)

    grp   = GROUP_MAP.get(feat, "meta")
    color = COLORS[grp]
    ax.plot(LAMBDA_GRID, curve, color=color, lw=2)

    # Reference lines
    for lam_val, (lam_label, lam_color) in LAMBDA_REF.items():
        ax.axvline(lam_val, color=lam_color, lw=0.8, ls="--", alpha=0.6)

    # Mark optimal λ
    best_idx = int(np.nanargmax(curve))
    best_lam = LAMBDA_GRID[best_idx]
    best_r2  = curve[best_idx]
    ax.scatter([best_lam], [best_r2], color="crimson", s=60, zorder=5)
    ax.annotate(f"λ*={best_lam:.2f}\nR²={best_r2:.3f}",
                xy=(best_lam, best_r2),
                xytext=(8, -18), textcoords="offset points",
                fontsize=7.5, color="crimson",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

    ax.set_title(FEAT_SHORT.get(feat, feat), fontsize=9, fontweight="bold")
    ax.set_xlabel("λ", fontsize=8)
    ax.set_ylabel("R²", fontsize=8)
    ax.set_xlim(-3, 3)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)

for j in range(len(ENGINEERED), nrows * ncols):
    axes[j].set_visible(False)

# Shared legend for reference λ lines
from matplotlib.lines import Line2D
ref_handles = [
    Line2D([0],[0], color=c, lw=1.2, ls="--", label=f"λ={v} ({lbl})")
    for v, (lbl, c) in LAMBDA_REF.items()
]
fig.legend(handles=ref_handles, loc="lower right", ncol=2, fontsize=8,
           framealpha=0.9, title="Reference transforms",
           bbox_to_anchor=(0.98, 0.02))

fig.suptitle(
    "Box-Cox Power Sweep — R²(λ) vs g-limit\n"
    "x^λ for λ≠0, log(x) at λ=0  |  red dot = optimal λ per feature",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_08_boxcox_sweep.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_08_boxcox_sweep.png")

print("\nAll figures written to figures/")
