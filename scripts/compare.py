"""
AERO 489 — Model comparison: ranked table + figures.

Loads all results/{model_name}.json files produced by train_classical.py and
train_modern.py, then outputs:

  1. Console table ranked by test adj-R² (all six proposal metrics per model)
  2. figures/comparison_bar.png  — grouped bar chart: R², MAE, RMSE per model
  3. figures/comparison_mos.png  — max_overpredict and MOS@1% per model
  4. figures/residuals.png       — per-model residual scatter (y_pred vs y_true)

The success criteria from §6.2 are highlighted:
    adj-R² > 0.8  |  RMSE < 0.75g  |  MAE < 0.5g  |  MOS < 0.25g

Usage
-----
    python compare.py
    python compare.py --results-dir results/  # default
"""

import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_root    = _scripts.parent
for _p in [str(_scripts), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = _root / "results"
FIGURES_DIR = _root / "figures/v2"

# §6.2 success thresholds
THRESHOLDS = {
    "adj_r2":          (0.80,  ">="),
    "rmse":            (0.75,  "<="),
    "mae":             (0.50,  "<="),
    "mos_01":          (0.25,  "<="),
}

# Display order for the table
METRIC_COLS = ["adj_r2", "mae", "rmse", "max_overpredict", "mos_01", "inference_time_ms"]
METRIC_LABELS = {
    "adj_r2":           "Adj. R²",
    "mae":              "MAE [g]",
    "rmse":             "RMSE [g]",
    "max_overpredict":  "Max over-pred [g]",
    "mos_01":           "MOS@1% [g]",
    "inference_time_ms": "Infer. [ms]",
}


SHORT_NAMES = {
    "linear_regression":          "Lin. Reg.",
    "polynomial_regression":      "Poly Reg.",
    "gaussian_process_regression": "GPR",
    "random_forest":              "Rand. Forest",
    "feedforward_nn":             "FFNN",
    "pinn":                       "PINN",
    "deep_learning_lstm":         "LSTM",
}


def _label(record: dict) -> str:
    return SHORT_NAMES.get(record["model"], record["model"])


def load_results(results_dir: Path) -> list[dict]:
    """Load all JSON result files; sort by adj_r2 descending."""
    records = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        records.append(data)
    records.sort(key=lambda r: r["metrics"]["adj_r2"], reverse=True)
    return records


def print_table(records: list[dict]) -> None:
    """Print ranked comparison table to stdout with threshold pass/fail markers."""
    col_w = 20
    name_w = 22

    header = f"{'Model':<{name_w}}" + "".join(f"{METRIC_LABELS[c]:>{col_w}}" for c in METRIC_COLS)
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in records:
        m = r["metrics"]
        name = _label(r)
        row = f"{name:<{name_w}}"
        for c in METRIC_COLS:
            val = m[c]
            cell = f"{val:.4f}"
            if c in THRESHOLDS:
                thresh, op = THRESHOLDS[c]
                passed = (val >= thresh) if op == ">=" else (val <= thresh)
                cell += " ✓" if passed else " ✗"
            row += f"{cell:>{col_w}}"
        print(row)

    print(sep)
    footer = f"{'Threshold':<{name_w}}"
    for c in METRIC_COLS:
        if c in THRESHOLDS:
            thresh, op = THRESHOLDS[c]
            cell = f"{op}{thresh}"
        else:
            cell = "—"
        footer += f"{cell:>{col_w}}"
    print(footer)
    print(sep)


def plot_bar_chart(records: list[dict]) -> None:
    """Three-panel grouped bar chart: Adj. R², MAE, RMSE per model."""
    labels = [_label(r) for r in records]
    adj_r2 = [r["metrics"]["adj_r2"] for r in records]
    mae    = [r["metrics"]["mae"]    for r in records]
    rmse   = [r["metrics"]["rmse"]   for r in records]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        (axes[0], adj_r2, "Adj. R²",  0.80, ">=", "steelblue"),
        (axes[1], mae,    "MAE [g]",   0.50, "<=", "darkorange"),
        (axes[2], rmse,   "RMSE [g]",  0.75, "<=", "seagreen"),
    ]
    for ax, vals, ylabel, thresh, op, color in panels:
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor="white")
        ax.axhline(thresh, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Threshold {op}{thresh}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Model Comparison — Accuracy Metrics", fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "comparison_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_mos_chart(records: list[dict]) -> None:
    """Side-by-side bars: max_overpredict and MOS@1% per model."""
    labels       = [_label(r) for r in records]
    max_over     = [r["metrics"]["max_overpredict"] for r in records]
    mos          = [r["metrics"]["mos_01"]          for r in records]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - w / 2, max_over, w, label="Max over-predict [g]", color="tomato",   alpha=0.85)
    ax.bar(x + w / 2, mos,      w, label="MOS@1% [g]",           color="steelblue", alpha=0.85)
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1.2, label="MOS threshold (0.25 g)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("g-units [g]")
    ax.set_title("Safety Margins: Max Over-Prediction and MOS@1%", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "comparison_mos.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_residuals(records: list[dict]) -> None:
    """2×4 grid of y_pred vs y_true scatter plots, one per model."""
    n   = len(records)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes_flat = axes.flatten()

    for i, r in enumerate(records):
        ax = axes_flat[i]
        y_true = np.array(r["y_true_test"])
        y_pred = np.array(r["y_pred_test"])
        residuals = y_pred - y_true

        colors = np.where(residuals > 0, "tomato", "steelblue")
        ax.scatter(y_true, y_pred, c=colors, s=20, alpha=0.7, edgecolors="none")

        lo = min(y_true.min(), y_pred.min()) - 0.2
        hi = max(y_true.max(), y_pred.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True g_limit [g]", fontsize=9)
        ax.set_ylabel("Predicted g_limit [g]", fontsize=9)
        ax.set_title(_label(r), fontweight="bold", fontsize=10)

        rmse_val = r["metrics"]["rmse"]
        mae_val  = r["metrics"]["mae"]
        ax.text(0.05, 0.93, f"RMSE={rmse_val:.3f} g\nMAE={mae_val:.3f} g",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Residual Scatter: Predicted vs True g_limit\n"
                 "(red = over-predict, blue = under-predict)", fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "comparison_residuals.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def main(args) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    if not results_dir.is_absolute():
        results_dir = (_root / results_dir).resolve()

    if not list(results_dir.glob("*.json")):
        print(f"No result JSON files found in {results_dir}.")
        print("Run train_classical.py and train_modern.py first.")
        return

    records = load_results(results_dir)
    print_table(records)
    plot_bar_chart(records)
    plot_mos_chart(records)
    plot_residuals(records)
    print(f"\nFigures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    main(parser.parse_args())
