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

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data_utils import load_scalar, load_timeseries, ALL_SCALAR_COLS
from evaluate import score

RESULTS_DIR = Path("results")


def _build_registry() -> dict:
    from models.feedforward_nn import FeedforwardNN
    from models.pinn           import PINN
    from models.deep_learning  import DeepLearning

    return {
        "nn":   (FeedforwardNN(), ALL_SCALAR_COLS, "scalar"),
        "pinn": (PINN(),          ALL_SCALAR_COLS, "scalar"),
        "lstm": (DeepLearning(),  None,            "timeseries"),
    }


def train_scalar(model, feature_cols: list[str]) -> dict:
    """Train and evaluate a scalar-input model (Models 5–6)."""
    # TODO: call load_scalar(feature_cols=feature_cols)
    # TODO: time model.fit(X_train, y_train)
    # TODO: call score(model, X_test, y_test, n_features=len(feature_cols))
    # TODO: print loss curves (model.train_loss_history_, model.val_loss_history_)
    #       — save figure to figures/{model.name}_loss.png
    # TODO: return results dict (same schema as train_classical.py)
    raise NotImplementedError


def train_timeseries(model) -> dict:
    """Train and evaluate the LSTM sequence model (Model 7)."""
    # TODO: call load_timeseries() → X_train, X_test, y_train, y_test
    # TODO: time model.fit(X_train, y_train)
    #       Note: LSTM fit may need seq_lens — decide interface in deep_learning.py
    # TODO: call score(model, X_test, y_test, n_features=X_train.shape[-1])
    #       — n_features is number of channels, not flattened sequence length;
    #         adjusted R² penalisation is approximate here
    # TODO: print and save loss curves
    # TODO: return results dict
    raise NotImplementedError


def main(args) -> None:
    registry = _build_registry()
    to_run = list(registry.keys()) if not args.models else args.models

    RESULTS_DIR.mkdir(exist_ok=True)

    for key in to_run:
        if key not in registry:
            print(f"Unknown model '{key}'. Valid: {list(registry.keys())}")
            continue

        model, feature_cols, data_type = registry[key]
        print(f"\n{'─'*60}")
        print(f"  {model.name}")
        print(f"{'─'*60}")

        if data_type == "scalar":
            result = train_scalar(model, feature_cols)
        else:
            result = train_timeseries(model)

        # TODO: write result to RESULTS_DIR / f"{model.name}.json"

    print("\nAll modern models complete. Results saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="*",
        help="Subset of models to run: nn pinn lstm"
    )
    main(parser.parse_args())
