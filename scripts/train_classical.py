"""
AERO 489 — Train and evaluate classical ML models (Models 1–4).

Runs each model on the same 80/20 train/test split from data_utils.py, evaluates
all six proposal metrics, and saves results to results/{model_name}.json.

Usage
-----
    python train_classical.py                   # all four models
    python train_classical.py --models lr poly  # subset by short name

Short names: lr, poly, gpr, rf
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data_utils import load_scalar, ENGINEERED_COLS, ALL_SCALAR_COLS
from evaluate import score

# ── Model registry ────────────────────────────────────────────────────────────
# (name, class, feature_set)
# Linear and poly use only engineered features (simpler + interpretable baseline)
# GPR and RF use the full 31-column set

def _build_registry() -> dict:
    from models.linear_reg   import LinearReg
    from models.poly_reg     import PolyReg
    from models.gpr          import GPR
    from models.random_forest import RandomForest

    return {
        "lr":   (LinearReg(),    ENGINEERED_COLS),
        "poly": (PolyReg(),      ENGINEERED_COLS),
        "gpr":  (GPR(),          ALL_SCALAR_COLS),
        "rf":   (RandomForest(), ALL_SCALAR_COLS),
    }


RESULTS_DIR = Path("results")


def train_and_save(short_name: str, model, feature_cols: list[str]) -> None:
    print(f"\n{'─'*60}")
    print(f"  {model.name}")
    print(f"{'─'*60}")

    # TODO: call load_scalar(feature_cols=feature_cols)
    # TODO: record wall-clock time for model.fit(X_train, y_train)
    # TODO: call score(model, X_test, y_test, n_features=len(feature_cols))
    # TODO: print metrics table (adj_r2, mae, rmse, max_overpredict, mos_01,
    #       inference_time_ms) + CV R² if available (model.cv_r2_)
    # TODO: if model has feature_importances_ (RandomForest), print top-10
    # TODO: build results dict:
    #   {
    #     "model":          model.name,
    #     "feature_set":    feature_cols,
    #     "n_features":     len(feature_cols),
    #     "train_time_s":   ...,
    #     "cv_r2":          model.cv_r2_ if hasattr else None,
    #     "metrics":        metrics_dict,
    #     "y_pred_test":    y_pred_test.tolist(),
    #     "y_true_test":    y_test.tolist(),
    #   }
    # TODO: RESULTS_DIR.mkdir(exist_ok=True)
    # TODO: write JSON to RESULTS_DIR / f"{model.name}.json"
    raise NotImplementedError


def main(args) -> None:
    registry = _build_registry()
    to_run = list(registry.keys()) if not args.models else args.models

    for key in to_run:
        if key not in registry:
            print(f"Unknown model '{key}'. Valid: {list(registry.keys())}")
            continue
        model, feature_cols = registry[key]
        train_and_save(key, model, feature_cols)

    print("\nAll classical models complete. Results saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="*",
        help="Subset of models to run: lr poly gpr rf"
    )
    main(parser.parse_args())
