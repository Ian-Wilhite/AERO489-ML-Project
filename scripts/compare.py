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

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

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


def load_results(results_dir: Path) -> list[dict]:
    """Load all JSON result files; sort by adj_r2 descending."""
    records = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        records.append(data)

    # TODO: sort records by metrics["adj_r2"] descending
    raise NotImplementedError


def print_table(records: list[dict]) -> None:
    """Print ranked comparison table to stdout with threshold pass/fail markers."""
    # TODO: print header row with model names and metric columns
    # TODO: for each record, format metric values; append ✓ or ✗ for columns
    #       that have a threshold in THRESHOLDS
    # TODO: print a footer row showing the success threshold values
    raise NotImplementedError


def plot_bar_chart(records: list[dict]) -> None:
    """Grouped bar chart comparing R², MAE, RMSE across models."""
    # TODO: extract model names and (adj_r2, mae, rmse) per model
    # TODO: plot grouped bars; add horizontal threshold lines (0.80 for R², etc.)
    # TODO: save to FIGURES_DIR / "comparison_bar.png"
    raise NotImplementedError


def plot_mos_chart(records: list[dict]) -> None:
    """Bar chart of max_overpredict and MOS@1% per model."""
    # TODO: extract max_overpredict and mos_01 per model
    # TODO: plot side-by-side bars; add threshold line at 0.25g for MOS
    # TODO: save to FIGURES_DIR / "comparison_mos.png"
    raise NotImplementedError


def plot_residuals(records: list[dict]) -> None:
    """Residual scatter: y_pred vs y_true for each model in a grid."""
    # TODO: create subplot grid (2 rows × 4 cols for 7 models)
    # TODO: for each model: scatter(y_true, y_pred); add y=x diagonal
    #       colour points by residual sign (red = overpredict, blue = underpredict)
    # TODO: annotate with RMSE and MAE from stored metrics
    # TODO: save to FIGURES_DIR / "residuals.png"
    raise NotImplementedError


def main(args) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    results_dir = Path(args.results_dir)

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
