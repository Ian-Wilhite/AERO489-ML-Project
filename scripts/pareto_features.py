#!/usr/bin/env python3
"""
Pareto front: OLS R² vs. number of features (Box-Cox + ranked strain pool).

Greedy forward selection on BOXCOX_COLS + RANKED_STRAIN_COLS with StandardScaler.
Shows both in-sample and 5-fold CV R² so the optimal feature-count knee is visible.
Saves to figures/v2/pareto_r2_vs_features.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

from data_utils import BOXCOX_COLS, RANKED_STRAIN_COLS, BOXCOX_COLS_LR

ROOT    = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "features/v2" / "features_scalar.parquet"
OUT     = ROOT / "figures/v2" / "pareto_r2_vs_features.png"


def _r2_insample(df, cols, g):
    X  = df.select(cols).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    m  = LinearRegression().fit(Xs, g)
    return r2_score(g, m.predict(Xs))


def _r2_cv(df, cols, g, cv=5):
    X  = df.select(cols).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    return cross_val_score(LinearRegression(), Xs, g, cv=cv, scoring="r2").mean()


def greedy_forward(df, candidates, g):
    selected  = []
    remaining = list(candidates)
    order, r2_is, r2_cv = [], [], []

    for _ in range(len(candidates)):
        best_col, best_r2 = None, -np.inf
        for col in remaining:
            trial = selected + [col]
            r2 = _r2_insample(df, trial, g)
            if r2 > best_r2:
                best_r2, best_col = r2, col
        selected.append(best_col)
        remaining.remove(best_col)
        order.append(best_col)
        r2_is.append(best_r2)
        r2_cv.append(_r2_cv(df, selected, g))
        print(f"  k={len(selected):2d}  +{best_col:<48}  is={best_r2:.4f}  cv={r2_cv[-1]:.4f}")

    return order, np.array(r2_is), np.array(r2_cv)


def main():
    df = pl.read_parquet(PARQUET).drop_nulls()
    g  = df["g_limit"].to_numpy()

    candidates = BOXCOX_COLS + RANKED_STRAIN_COLS  # 17 total
    print(f"Greedy forward selection on {len(candidates)} candidate features …")
    order, r2_is, r2_cv = greedy_forward(df, candidates, g)

    # Named reference points ──────────────────────────────────────────────────
    nc_boxcox = len(BOXCOX_COLS)
    nc_lr     = len(BOXCOX_COLS_LR)
    nc_ranked = len(RANKED_STRAIN_COLS)
    nc_all    = nc_boxcox + nc_ranked
    named = {
        f"ranked_strain_p23\n(best single, k=1)": ["ranked_strain_p23"],
        f"RANKED_STRAIN\n(k={nc_ranked})":         RANKED_STRAIN_COLS,
        f"BOXCOX_LR\n(k={nc_lr})":                 BOXCOX_COLS_LR,
        f"BOXCOX_ALL\n(k={nc_boxcox})":            BOXCOX_COLS,
        f"BOXCOX_ALL + RANKED\n(k={nc_all})":      BOXCOX_COLS + RANKED_STRAIN_COLS,
    }
    named_pts = {}
    for label, cols in named.items():
        is_ = _r2_insample(df, cols, g)
        cv_ = _r2_cv(df, cols, g)
        named_pts[label] = (len(cols), is_, cv_)
        print(f"  {label.replace(chr(10),' '):<40}  k={len(cols):2d}  is={is_:.4f}  cv={cv_:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ks = np.arange(1, len(candidates) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ks, r2_is, "b-o",  ms=5, lw=1.4, label="Greedy fwd. selection — in-sample",  alpha=0.85, zorder=3)
    ax.plot(ks, r2_cv, "b--s", ms=5, lw=1.4, label="Greedy fwd. selection — 5-fold CV", alpha=0.70, zorder=3)

    # Mark CV peak
    k_opt = int(np.argmax(r2_cv)) + 1
    ax.axvline(k_opt, color="steelblue", lw=0.9, ls=":", alpha=0.6)
    ax.annotate(f"CV peak\n(k={k_opt})", xy=(k_opt, r2_cv[k_opt - 1]),
                xytext=(k_opt + 0.4, r2_cv[k_opt - 1] - 0.004),
                fontsize=8, color="steelblue")

    # Named feature sets
    palette = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3", "#377eb8"]
    for (label, (n, is_, cv_)), color in zip(named_pts.items(), palette):
        ax.scatter(n, is_, color=color, s=110, zorder=6, edgecolors="white", lw=0.8)
        ax.scatter(n, cv_, color=color, marker="s", s=80, zorder=6, alpha=0.75)
        ax.annotate(label, (n, is_), textcoords="offset points",
                    xytext=(5, 3), fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"))

    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_ylabel("OLS R²  (StandardScaled)", fontsize=12)
    ax.set_title("Feature-Count Pareto Front — Box-Cox + Rank-ordered Strain Pool\n"
                 "Greedy forward selection (circles = in-sample, squares = 5-fold CV)", fontsize=11)
    ax.set_xlim(0, len(candidates) + 1)
    ax.set_ylim(max(0.0, min(r2_cv.min(), r2_is.min()) - 0.02), 1.005)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUT}")
    plt.close(fig)

    print("\nGreedy selection order (in full):")
    for i, (col, is_, cv_) in enumerate(zip(order, r2_is, r2_cv), 1):
        print(f"  {i:2d}. {col:<50}  is={is_:.4f}  cv={cv_:.4f}")


if __name__ == "__main__":
    main()
