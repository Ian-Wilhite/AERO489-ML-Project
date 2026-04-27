"""
AERO 489 — Statistical significance tests + conservative prediction rate.

Produces:
  figures-v2/wilcoxon_pvalue.png    — 7×7 heatmap of Wilcoxon p-values (|error| pairs)
  figures-v2/wilcoxon_wins.png      — win/tie/loss matrix (which model is better)
  figures-v2/conservative_rate.png  — bar chart: fraction of conservative predictions per model

Console output:
  - Friedman test result (is there any significant difference across all models?)
  - Wilcoxon pairwise table with Bonferroni-corrected significance stars
  - Conservative prediction rate table

Usage
-----
    cd /home/ianw/Github/Courses/AERO489-Proj
    .venv/bin/python scripts/significance_tests.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

sns.set_theme(style="whitegrid", font_scale=1.05)

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures-v2"
FIGURES_DIR.mkdir(exist_ok=True)

ALPHA = 0.05

MODEL_META = [
    ("linear_regression",           "Linear Reg.",    "#7f7f7f"),
    ("polynomial_regression",        "Poly Reg.",      "#bcbd22"),
    ("gaussian_process_regression",  "GPR",            "#9467bd"),
    ("random_forest",                "Random Forest",  "#8c564b"),
    ("feedforward_nn",               "Feedforward NN", "#1f77b4"),
    ("pinn",                         "PINN",           "#2ca02c"),
    ("deep_learning_lstm",           "LSTM",           "#d62728"),
]

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})


def _load():
    records = []
    for name, label, color in MODEL_META:
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            print(f"  [WARN] {path} not found — skipping")
            continue
        with open(path) as f:
            d = json.load(f)
        d["label"]   = label
        d["color"]   = color
        d["y_pred"]  = np.array(d["y_pred_test"])
        d["y_true"]  = np.array(d["y_true_test"])
        d["abs_err"] = np.abs(d["y_pred"] - d["y_true"])
        d["resid"]   = d["y_pred"] - d["y_true"]
        records.append(d)
    return records


def _save(fig, name):
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Friedman test ─────────────────────────────────────────────────────────────
def run_friedman(models):
    print("\n── Friedman Test (H₀: all models perform equally) ──")
    # Requires equal-length test sets (same observations); use only matching models
    sizes = [len(r["abs_err"]) for r in models]
    modal_size = max(set(sizes), key=sizes.count)
    matched = [r for r in models if len(r["abs_err"]) == modal_size]
    excluded = [r["label"] for r in models if len(r["abs_err"]) != modal_size]
    if excluded:
        print(f"  [NOTE] Excluding {excluded} — different test-set size ({modal_size} vs {set(sizes)-{modal_size}})")
    abs_errors = [r["abs_err"] for r in matched]
    stat, p = friedmanchisquare(*abs_errors)
    print(f"  χ²({len(matched)-1}) = {stat:.4f},  p = {p:.4e}  (n={modal_size} obs, {len(matched)} models)")
    if p < ALPHA:
        print(f"  → Reject H₀ at α={ALPHA}: models are NOT equivalent")
    else:
        print(f"  → Fail to reject H₀ at α={ALPHA}: no significant difference detected")
    return stat, p


# ── Wilcoxon pairwise ─────────────────────────────────────────────────────────
def run_wilcoxon(models):
    n = len(models)
    labels = [r["label"] for r in models]

    p_matrix   = np.ones((n, n))
    stat_matrix = np.zeros((n, n))

    pairs = list(combinations(range(n), 2))
    n_pairs = len(pairs)
    alpha_bonf = ALPHA / n_pairs  # Bonferroni correction

    print(f"\n── Wilcoxon Signed-Rank Tests ({n_pairs} pairs, Bonferroni α={alpha_bonf:.4f}) ──")
    print(f"  {'Model A':<20} vs {'Model B':<20}  stat      p-value    sig")
    print("  " + "─" * 68)

    for i, j in pairs:
        a = models[i]["abs_err"]
        b = models[j]["abs_err"]
        if len(a) == len(b):
            # Paired Wilcoxon signed-rank (same test observations)
            try:
                stat, p = wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
                test_name = "W"
            except ValueError:
                stat, p = 0.0, 1.0
                test_name = "W"
        else:
            # Unpaired Mann-Whitney U (different test set sizes)
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            test_name = "U"
        p_matrix[i, j] = p
        p_matrix[j, i] = p
        stat_matrix[i, j] = stat
        sig = "***" if p < alpha_bonf else ("*" if p < ALPHA else "")
        print(f"  {labels[i]:<20} vs {labels[j]:<20}  {test_name}={stat:8.1f}  {p:.4e}  {sig}")

    # fill diagonal
    np.fill_diagonal(p_matrix, np.nan)

    return p_matrix, labels, alpha_bonf


def plot_wilcoxon_heatmap(p_matrix, labels, alpha_bonf):
    fig, ax = plt.subplots(figsize=(9, 7))

    log_p = -np.log10(p_matrix)  # higher = more significant
    mask  = np.eye(len(labels), dtype=bool)

    sns.heatmap(
        log_p, ax=ax, mask=mask,
        cmap="YlOrRd", vmin=0, vmax=4,
        annot=p_matrix, fmt=".3f",
        linewidths=0.5, linecolor="white",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "-log10(p-value)  [higher = more significant]"},
    )

    # mark diagonal as grey
    for k in range(len(labels)):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True,
                                    facecolor="#cccccc", edgecolor="white", zorder=2))

    sig_threshold = -np.log10(alpha_bonf)
    ax.set_title(
        f"Wilcoxon Signed-Rank p-values (two-sided, Bonferroni a={alpha_bonf:.4f})\n"
        f"Colour threshold = -log10({alpha_bonf:.4f}) ~ {sig_threshold:.1f}",
        fontsize=11
    )
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    _save(fig, "wilcoxon_pvalue.png")


def plot_win_matrix(models, labels, alpha_bonf):
    """For each pair (i,j): colour by which model has lower median |error|, with significance."""
    n = len(models)
    # win_matrix[i,j]: +1 if model i significantly beats j, -1 if j beats i, 0 tie
    win_matrix = np.zeros((n, n))

    pairs = list(combinations(range(n), 2))
    for i, j in pairs:
        a = models[i]["abs_err"]
        b = models[j]["abs_err"]
        try:
            _, p = wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
        except ValueError:
            p = 1.0
        if p < alpha_bonf:
            if np.median(a) < np.median(b):
                win_matrix[i, j] = 1   # i beats j
                win_matrix[j, i] = -1
            else:
                win_matrix[i, j] = -1
                win_matrix[j, i] = 1

    # count wins per model
    wins = (win_matrix == 1).sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              gridspec_kw={"width_ratios": [1.6, 1]})

    # left: win/loss heatmap
    ax = axes[0]
    cmap = matplotlib.colors.ListedColormap(["#d6604d", "#f7f7f7", "#4393c3"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    mask = np.eye(n, dtype=bool)
    sns.heatmap(win_matrix, ax=ax, mask=mask,
                cmap=cmap, norm=norm,
                annot=False, linewidths=0.5, linecolor="white",
                xticklabels=labels, yticklabels=labels, cbar=False)
    for k in range(n):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True,
                                    facecolor="#cccccc", edgecolor="white", zorder=2))

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#4393c3", label="Row beats column"),
                  Patch(facecolor="#d6604d", label="Row loses to column"),
                  Patch(facecolor="#f7f7f7", label="No sig. difference")]
    ax.legend(handles=legend_els, loc="upper left", bbox_to_anchor=(0, -0.22),
              fontsize=9, ncol=3)
    ax.set_title("Win/Loss Matrix (Bonferroni-corrected Wilcoxon)", fontsize=11)
    plt.sca(ax)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    # right: win count bar
    ax2 = axes[1]
    colors = [r["color"] for r in models]
    ax2.barh(labels, wins, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xlabel("Number of models significantly beaten")
    ax2.set_title("Win Count (out of 6)")
    ax2.set_xlim(0, n - 1)
    for i, v in enumerate(wins):
        ax2.text(v + 0.05, i, str(int(v)), va="center", fontsize=10)
    ax2.invert_yaxis()

    fig.suptitle("Pairwise Statistical Comparison — Wilcoxon Signed-Rank",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, "wilcoxon_wins.png")


# ── Conservative prediction rate ─────────────────────────────────────────────
def compute_conservative_rates(models):
    print("\n── Conservative Prediction Rate (ŷ ≤ y_true) ──")
    print(f"  {'Model':<22}  conserv. rate  median resid [g]")
    print("  " + "─" * 52)
    rates = []
    for r in models:
        rate = float(np.mean(r["resid"] <= 0))
        med  = float(np.median(r["resid"]))
        rates.append(rate)
        print(f"  {r['label']:<22}  {rate*100:>6.1f}%        {med:+.4f}")
    return rates


def plot_conservative_rate(models, rates):
    labels = [r["label"] for r in models]
    colors = [r["color"] for r in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, [r * 100 for r in rates], color=colors,
                  alpha=0.85, edgecolor="white")
    ax.axhline(50, color="black", lw=1.0, ls="--", label="50% (unbiased)")
    ax.axhline(95, color="darkorange", lw=1.2, ls=":", label="95% target (notional)")

    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Conservative predictions [%]  (ŷ ≤ y_true)")
    ax.set_title("Conservative Prediction Rate — How often does each model stay on the safe side?",
                 fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "conservative_rate.png")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading results...")
    models = _load()
    if len(models) < 2:
        sys.exit("Need at least 2 models.")
    print(f"Loaded {len(models)} models: {[r['label'] for r in models]}\n")

    run_friedman(models)

    p_matrix, labels, alpha_bonf = run_wilcoxon(models)
    plot_wilcoxon_heatmap(p_matrix, labels, alpha_bonf)
    plot_win_matrix(models, labels, alpha_bonf)

    rates = compute_conservative_rates(models)
    plot_conservative_rate(models, rates)

    print(f"\nDone → {FIGURES_DIR}/")
