"""
AERO 489 — Train and evaluate modern ML models (Models 5–7).

Model 5 (FeedforwardNN) and Model 6 (PINN) use the same scalar parquet and
80/20 split as the classical models.  Model 7 (DeepLearning / LSTM) uses the
time_series parquet with a sim-id-level split.

Usage
-----
    python train_modern.py                     # all three models
    python train_modern.py --models nn pinn    # subset by short name

Short names: nn, pinn, lstm
"""

import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_root    = _scripts.parent
for p in [str(_scripts), str(_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_scalar, load_timeseries, ALL_SCALAR_COLS
from evaluate import score

RESULTS_DIR = _root / "results"
FIGURES_DIR = _root / "figures-v2"


def _build_registry() -> dict:
    from models.feedforward_nn import FeedforwardNN
    from models.pinn           import PINN
    from models.deep_learning  import DeepLearning

    return {
        "nn":   (FeedforwardNN(), ALL_SCALAR_COLS, "scalar"),
        "pinn": (PINN(),          ALL_SCALAR_COLS, "scalar"),
        "lstm": (DeepLearning(),  None,            "timeseries"),
    }


def _save_loss_curve(model, tag: str) -> None:
    if not getattr(model, "train_loss_history_", None):
        return
    FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(model.train_loss_history_, label="Train MSE", lw=1.5)
    ax.plot(model.val_loss_history_,   label="Val MSE",   lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model.name} — Training Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / f"{model.name}_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Loss curve → {out}")


def train_scalar(model, feature_cols: list[str]) -> dict:
    X_train, X_test, y_train, y_test = load_scalar(feature_cols=feature_cols)
    print(f"  Data: {X_train.shape[0]} train / {X_test.shape[0]} test  |  {len(feature_cols)} features")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    metrics = score(model, X_test, y_test, n_features=len(feature_cols))
    y_pred  = model.predict(X_test)

    print(f"  Train time : {train_time:.1f}s")
    print(f"  adj_R²     : {metrics['adj_r2']:.4f}")
    print(f"  RMSE       : {metrics['rmse']:.4f} g")
    print(f"  MAE        : {metrics['mae']:.4f} g")
    print(f"  MOS@1%     : {metrics['mos_01']:.4f} g")
    print(f"  Inference  : {metrics['inference_time_ms']:.2f} ms")

    _save_loss_curve(model, model.name)

    return {
        "model":        model.name,
        "feature_set":  feature_cols,
        "n_features":   len(feature_cols),
        "train_time_s": train_time,
        "metrics":      metrics,
        "y_pred_test":  y_pred.tolist(),
        "y_true_test":  y_test.tolist(),
    }


def train_timeseries(model) -> dict:
    X_train, X_test, y_train, y_test, sl_train, sl_test = load_timeseries()
    n_channels = X_train.shape[2]
    print(f"  Data: {len(y_train)} train / {len(y_test)} test sims  |  {n_channels} channels  |  max_len={X_train.shape[1]}")

    t0 = time.time()
    model.fit(X_train, sl_train, y_train)
    train_time = time.time() - t0

    # Wrap predict so score() can call model.predict(X_test) transparently
    class _Wrapper:
        def __init__(self, m, sl): self.m, self.sl = m, sl
        def predict(self, X):      return self.m.predict(X, self.sl)

    metrics = score(_Wrapper(model, sl_test), X_test, y_test, n_features=n_channels)
    y_pred  = model.predict(X_test, sl_test)

    print(f"  Train time : {train_time:.1f}s")
    print(f"  adj_R²     : {metrics['adj_r2']:.4f}")
    print(f"  RMSE       : {metrics['rmse']:.4f} g")
    print(f"  MAE        : {metrics['mae']:.4f} g")
    print(f"  MOS@1%     : {metrics['mos_01']:.4f} g")
    print(f"  Inference  : {metrics['inference_time_ms']:.2f} ms")

    _save_loss_curve(model, model.name)

    return {
        "model":        model.name,
        "n_features":   n_channels,
        "train_time_s": train_time,
        "metrics":      metrics,
        "y_pred_test":  y_pred.tolist(),
        "y_true_test":  y_test.tolist(),
    }


def main(args) -> None:
    registry = _build_registry()
    to_run   = list(registry.keys()) if not args.models else args.models

    RESULTS_DIR.mkdir(exist_ok=True)

    for key in to_run:
        if key not in registry:
            print(f"Unknown model '{key}'. Valid: {list(registry.keys())}")
            continue

        model, feature_cols, data_type = registry[key]
        print(f"\n{'─'*60}")
        print(f"  {model.name}")
        print(f"{'─'*60}")

        result = train_scalar(model, feature_cols) if data_type == "scalar" else train_timeseries(model)

        out = RESULTS_DIR / f"{model.name}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out}")

    print("\nAll modern models complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Subset: nn pinn lstm")
    main(parser.parse_args())
