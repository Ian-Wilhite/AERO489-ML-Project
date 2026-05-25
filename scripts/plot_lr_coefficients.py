"""
AERO 489 — Linear model standardized coefficient plot.

Loads results/linear_regression.json and plots the standardized coefficients
(change in g-limit per 1-std increase in each feature) for the current
GREEDY-6 feature set: boxcox_strain_energy_slope, ranked_strain_p23,
ranked_strain_p24, gompertz_c, gompertz_log_b, boxcox_k_spring.

Saved to figures/v2/lr_coefficients.png.

Usage
-----
    .venv/bin/python scripts/plot_lr_coefficients.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "linear_regression.json"
FIGURES = ROOT / "figures/v2"
FIGURES.mkdir(exist_ok=True)

# Human-readable labels
LABELS = {
    "boxcox_strain_energy_slope": "Strain energy slope  (Box-Cox)",
    "boxcox_k_spring":            "Structural stiffness k  (Box-Cox)",
    "ranked_strain_p23":          "Gauge slope — rank 23/24  (~96th pct)",
    "ranked_strain_p24":          "Gauge slope — rank 24/24  (maximum)",
    "gompertz_c":                 "Gompertz c  (rank-profile growth rate)",
    "gompertz_log_b":             "Gompertz log b  (initial suppression)",
}

# Colour by feature family
GROUP_COLOR = {
    "boxcox_strain_energy_slope": "#1a9641",   # strain energy — green
    "boxcox_k_spring":            "#d7191c",   # stiffness/tip — red
    "ranked_strain_p23":          "#2166ac",   # ranked gauge — blue
    "ranked_strain_p24":          "#2166ac",
    "gompertz_c":                 "#756bb1",   # Gompertz shape — purple
    "gompertz_log_b":             "#756bb1",
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
fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(range(len(values)), values, color=colors, alpha=0.85, edgecolor="white", height=0.6)
ax.axvline(0, color="black", lw=0.9)

x_range = max(abs(values)) * 1.35
for i, (bar, v) in enumerate(zip(bars, values)):
    pad = x_range * 0.03 if v >= 0 else -x_range * 0.03
    ha  = "left" if v >= 0 else "right"
    ax.text(v + pad, i, f"{v:+.3f}", va="center", ha=ha, fontsize=9)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlim(-x_range, x_range)
ax.set_xlabel("Standardized coefficient  [g per 1-σ increase in feature]", fontsize=10)
ax.set_title(
    f"Linear Regression — Standardized Coefficients  (GREEDY-6 feature set)\n"
    f"adj-R²={m['adj_r2']:.4f}   RMSE={m['rmse']:.3f} g   CV-R²={d['cv_r2']:.4f}",
    fontsize=11, fontweight="bold",
)

legend_els = [
    Patch(fc="#1a9641", alpha=0.85, label="Strain energy (Box-Cox)"),
    Patch(fc="#d7191c", alpha=0.85, label="Stiffness (Box-Cox)"),
    Patch(fc="#2166ac", alpha=0.85, label="Ranked gauge slopes"),
    Patch(fc="#756bb1", alpha=0.85, label="Gompertz shape params"),
]
ax.legend(handles=legend_els, fontsize=9, loc="lower right")
ax.grid(axis="x", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
out = FIGURES / "lr_coefficients.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

print(f"\n{'Feature':<45}  {'Coef':>8}  {'|Coef|':>8}")
print("─" * 65)
for f, v in zip(features, values):
    print(f"{LABELS.get(f, f):<45}  {v:>+8.4f}  {abs(v):>8.4f}")
