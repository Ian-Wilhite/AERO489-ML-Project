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

import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_root    = _scripts.parent
for _p in [str(_scripts), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import time

import numpy as np

from data_utils import load_scalar, ENGINEERED_COLS, ALL_SCALAR_COLS, BOXCOX_COLS, BOXCOX_COLS_LR, GREEDY_8_COLS, RF_NN_COLS
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
        "lr":   (LinearReg(),    GREEDY_8_COLS),
        "poly": (PolyReg(),      BOXCOX_COLS),
        "gpr":  (GPR(),          BOXCOX_COLS),
        "rf":   (RandomForest(), RF_NN_COLS),
    }


RESULTS_DIR = _root / "results"


def train_and_save(short_name: str, model, feature_cols: list[str]) -> None:
    print(f"\n{'─'*60}")
    print(f"  {model.name}")
    print(f"{'─'*60}")

    X_train, X_test, y_train, y_test = load_scalar(feature_cols=feature_cols)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0

    metrics = score(model, X_test, y_test, n_features=len(feature_cols))
    cv_r2 = getattr(model, "cv_r2_", None)

    print(f"  adj_r2            : {metrics['adj_r2']:.4f}")
    print(f"  mae               : {metrics['mae']:.4f} g")
    print(f"  rmse              : {metrics['rmse']:.4f} g")
    print(f"  max_overpredict   : {metrics['max_overpredict']:.4f} g")
    print(f"  mos_01            : {metrics['mos_01']:.4f} g")
    print(f"  inference_time_ms : {metrics['inference_time_ms']:.3f} ms")
    if cv_r2 is not None:
        print(f"  cv_r2 (10-fold)   : {cv_r2:.4f}")
    print(f"  train_time_s      : {train_time_s:.2f} s")

    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        top10 = np.argsort(fi)[::-1][:10]
        print("\n  Top-10 feature importances:")
        for rank, idx in enumerate(top10, 1):
            print(f"    {rank:2d}. {feature_cols[idx]:40s}  {fi[idx]:.4f}")

    y_pred_test = model.predict(X_test)

    # Extract linear model coefficients if available (for interpretability plots)
    coef_dict = None
    pipeline = getattr(model, "_pipeline", None)
    if pipeline is not None:
        lr_step = pipeline.named_steps.get("lr") or pipeline.named_steps.get("ridge")
        if lr_step is not None and hasattr(lr_step, "coef_"):
            coef_dict = dict(zip(feature_cols, lr_step.coef_.tolist()))

    results = {
        "model":        model.name,
        "feature_set":  feature_cols,
        "n_features":   len(feature_cols),
        "train_time_s": train_time_s,
        "cv_r2":        cv_r2,
        "metrics":      metrics,
        "y_pred_test":  y_pred_test.tolist(),
        "y_true_test":  y_test.tolist(),
    }
    if coef_dict is not None:
        results["coefficients"] = coef_dict

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{model.name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")


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
