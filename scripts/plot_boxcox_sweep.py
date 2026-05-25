"""
Box-Cox power λ sweep for all engineered features using combined Dataset 2.
Sweeps λ ∈ [-15, 15], plots R²(λ) vs g_limit in a 4×3 grid.
Output: figures/v2/fig_08_boxcox_sweep.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import polars as pl
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

FEAT_SHORT = {
    "tip_deflection_slope":           "tip_defl_slope",
    "tip_per_g_at_failure":           "tip_per_g",
    "avg_strain_at_failure":          "avg_strain",
    "avg_strain_slope":               "avg_strain_slope",
    "strain_energy_at_failure":       "strain_energy",
    "strain_energy_slope":            "strain_energy_slope",
    "max_vm_stress_at_failure":       "max_vm_stress",
    "k_spring":                       "k_spring",
    "inv_tip_per_g_at_failure":       "inv_tip_per_g",
    "inv_tip_deflection_slope":       "inv_tip_defl_slope",
    "inv_max_vm_stress_at_failure":   "inv_max_vm_stress",
    "sqrt_strain_energy_at_failure":  "sqrt_strain_energy",
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

LAMBDA_GRID = np.linspace(-15, 15, 3000)

LAMBDA_REF = {
    -2:   ("x⁻²",  "#d62728"),
    -1:   ("1/x",  "#ff7f0e"),
    -0.5: ("1/√x", "#9467bd"),
     0:   ("log",  "#2ca02c"),
     0.5: ("√x",   "#17becf"),
     1:   ("raw",  "#7f7f7f"),
     2:   ("x²",   "#bcbd22"),
}

# ── Sweep ─────────────────────────────────────────────────────────────────────

def r2_curve(x_raw):
    curve = np.empty(len(LAMBDA_GRID))
    for k, lam in enumerate(LAMBDA_GRID):
        xt = np.log(x_raw) if abs(lam) < 1e-4 else np.power(x_raw, lam)
        mask = np.isfinite(xt) & np.isfinite(y)
        if mask.sum() < 5:
            curve[k] = np.nan
            continue
        xm, ym = xt[mask].reshape(-1, 1), y[mask]
        curve[k] = r2_score(ym, LinearRegression().fit(xm, ym).predict(xm))
    return curve

# ── Figure ────────────────────────────────────────────────────────────────────

NCOLS, NROWS = 4, 3
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(5.5 * NCOLS, 4 * NROWS))
axes = axes.flatten()

for i, feat in enumerate(ENGINEERED):
    ax    = axes[i]
    color = COLORS[GROUP_MAP[feat]]
    x_raw = df[feat].to_numpy().astype(float)
    curve = r2_curve(x_raw)

    ax.plot(LAMBDA_GRID, curve, color=color, lw=2)

    for lam_val, (lam_label, lam_color) in LAMBDA_REF.items():
        ax.axvline(lam_val, color=lam_color, lw=0.8, ls="--", alpha=0.6)

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
    ax.set_xlim(-15, 15)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)

ref_handles = [
    Line2D([0], [0], color=c, lw=1.2, ls="--", label=f"λ={v} ({lbl})")
    for v, (lbl, c) in LAMBDA_REF.items()
]
fig.legend(handles=ref_handles, loc="lower right", ncol=2, fontsize=8,
           framealpha=0.9, title="Reference transforms",
           bbox_to_anchor=(0.98, 0.02))

fig.suptitle(
    "Box-Cox Power Sweep — R²(λ) vs g-limit  |  Dataset 2\n"
    "x^λ for λ≠0, log(x) at λ=0  |  red dot = optimal λ per feature",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
out = FIGURES_DIR / "fig_08_boxcox_sweep.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
