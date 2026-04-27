"""
AERO 489 — Linear model standardized coefficient plot.

Loads results/linear_regression.json and plots the standardized coefficients
(i.e. change in g-limit per 1-std increase in each Box-Cox feature).
Saved to figures-v2/lr_coefficients.png.

Usage
-----
    .venv/bin/python scripts/plot_lr_coefficients.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
RESULTS    = ROOT / "results" / "linear_regression.json"
FIGURES    = ROOT / "figures-v2"
FIGURES.mkdir(exist_ok=True)

# Optimal λ values used when building each boxcox column
OPT_LAMBDA = {
    "boxcox_tip_deflection_slope":           -6.984,
    "boxcox_tip_per_g_at_failure":           -6.984,
    "boxcox_avg_strain_at_failure":          +0.526,
    "boxcox_avg_strain_slope":               +2.068,
    "boxcox_strain_energy_at_failure":       +0.285,
    "boxcox_strain_energy_slope":            +0.165,
    "boxcox_max_vm_stress_at_failure":      -11.840,
    "boxcox_k_spring":                       +6.984,
    "boxcox_inv_tip_per_g_at_failure":       +6.984,
    "boxcox_inv_tip_deflection_slope":       +6.984,
    "boxcox_inv_max_vm_stress_at_failure":  +11.840,
    "boxcox_sqrt_strain_energy_at_failure":  +0.566,
}

# Short readable labels (base quantity + λ)
LABELS = {
    "boxcox_tip_deflection_slope":          "Tip defl. slope  (λ=−6.98)",
    "boxcox_tip_per_g_at_failure":          "Tip defl. / g    (λ=−6.98)",
    "boxcox_avg_strain_at_failure":         "Avg strain        (λ=+0.53)",
    "boxcox_avg_strain_slope":              "Avg strain slope  (λ=+2.07)",
    "boxcox_strain_energy_at_failure":      "Strain energy     (λ=+0.29)",
    "boxcox_strain_energy_slope":           "Strain energy sl. (λ=+0.17)",
    "boxcox_max_vm_stress_at_failure":      "Max VM stress     (λ=−11.84)",
    "boxcox_k_spring":                      "k_spring          (λ=+6.98)",
    "boxcox_inv_tip_per_g_at_failure":      "1 / tip per g     (λ=+6.98)",
    "boxcox_inv_tip_deflection_slope":      "1 / tip defl. sl. (λ=+6.98)",
    "boxcox_inv_max_vm_stress_at_failure":  "1 / max VM stress (λ=+11.84)",
    "boxcox_sqrt_strain_energy_at_failure": "√(strain energy)  (λ=+0.57)",
}

# Feature group colours
GROUP_COLOR = {
    "boxcox_tip_deflection_slope":          "#d7191c",
    "boxcox_tip_per_g_at_failure":          "#d7191c",
    "boxcox_avg_strain_at_failure":         "#fdae61",
    "boxcox_avg_strain_slope":              "#fdae61",
    "boxcox_strain_energy_at_failure":      "#1a9641",
    "boxcox_strain_energy_slope":           "#1a9641",
    "boxcox_max_vm_stress_at_failure":      "#756bb1",
    "boxcox_k_spring":                      "#d7191c",
    "boxcox_inv_tip_per_g_at_failure":      "#d7191c",
    "boxcox_inv_tip_deflection_slope":      "#d7191c",
    "boxcox_inv_max_vm_stress_at_failure":  "#756bb1",
    "boxcox_sqrt_strain_energy_at_failure": "#1a9641",
}

d    = json.load(open(RESULTS))
coef = d["coefficients"]
m    = d["metrics"]

features = list(coef.keys())
values   = np.array([coef[f] for f in features])

# Sort by absolute value descending
order    = np.argsort(np.abs(values))[::-1]
features = [features[i] for i in order]
values   = values[order]
labels   = [LABELS.get(f, f) for f in features]
colors   = [GROUP_COLOR.get(f, "#888888") for f in features]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.barh(range(len(values)), values, color=colors, alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", lw=1.0)

# Value labels
for i, (bar, v) in enumerate(zip(bars, values)):
    pad = 0.05 if v >= 0 else -0.05
    ha  = "left" if v >= 0 else "right"
    ax.text(v + pad, i, f"{v:+.3f}", va="center", ha=ha, fontsize=8.5)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Standardized coefficient  [g per 1-std increase in Box-Cox feature]", fontsize=10)
ax.set_title(
    f"Linear Regression — Standardized Coefficients on Box-Cox Features\n"
    f"adj-R²={m['adj_r2']:.4f}  RMSE={m['rmse']:.4f}g  cv-R²={d['cv_r2']:.4f}",
    fontsize=11, fontweight="bold",
)

# Legend
from matplotlib.patches import Patch
legend_els = [
    Patch(fc="#d7191c", label="Tip deflection family"),
    Patch(fc="#fdae61", label="Avg strain family"),
    Patch(fc="#1a9641", label="Strain energy family"),
    Patch(fc="#756bb1", label="VM stress family"),
]
ax.legend(handles=legend_els, fontsize=9, loc="lower right")
ax.grid(axis="x", alpha=0.25)

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
fig.tight_layout()
out = FIGURES / "lr_coefficients.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# Print table to console
print(f"\n{'Feature':<45} {'λ*':>7}  {'Coef':>8}  {'|Coef|':>8}")
print("─" * 75)
for f, v in zip(features, values):
    lam = OPT_LAMBDA.get(f, float("nan"))
    print(f"{LABELS.get(f,f):<45} {lam:>7.3f}  {v:>+8.4f}  {abs(v):>8.4f}")
