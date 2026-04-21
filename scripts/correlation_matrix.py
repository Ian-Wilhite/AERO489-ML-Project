"""
AERO 489 — Correlation matrix for all scalar features + target (g_limit).
Reads features/features_scalar.parquet and saves the heatmap to figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

PARQUET = Path("features/features_scalar.parquet")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ── Load and drop non-feature columns ────────────────────────────────────────
df = pl.read_parquet(PARQUET).drop("sim_id")

# Shorten Strain_Node_XXX_failure → SN_XXX for readability
rename = {
    c: "SN_" + c.split("_Node_")[1].replace("_failure", "")
    for c in df.columns
    if c.startswith("Strain_Node_")
}
df = df.rename(rename)

corr = df.to_pandas().corr(method="pearson")
n = len(corr)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 14))

mask = np.zeros_like(corr, dtype=bool)  # show full matrix (no triangle mask)

sns.heatmap(
    corr,
    ax=ax,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.3,
    linecolor="white",
    annot=(n <= 15),       # only annotate if small enough to be readable
    fmt=".2f",
    square=True,
    cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    xticklabels=corr.columns,
    yticklabels=corr.columns,
)

ax.set_title("Pearson Correlation Matrix — All Scalar Features", fontsize=14, pad=12)
ax.tick_params(axis="x", labelsize=7, rotation=45)
ax.tick_params(axis="y", labelsize=7, rotation=0)
plt.tight_layout()

out_path = OUT_DIR / "correlation_matrix.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
