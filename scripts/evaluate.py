"""
AERO 489 — Evaluation metrics module.

All six metrics required by the proposal (§5.3):
  1. adjusted_r2        — penalises extra features
  2. mae                — mean absolute error [g]
  3. rmse               — root mean square error [g]
  4. max_overpredict    — worst-case safety violation: max(y_pred - y_true) [g]
  5. mos_01             — required margin of safety so ≤1% of predictions exceed
                          actual g_limit after subtracting MOS [g]
  6. inference_time_ms  — median inference latency over N repeated calls [ms]

Usage
-----
    from evaluate import score
    metrics = score(model, X_test, y_test, n_features=X_train.shape[1])
    # returns dict with all six keys
"""

import time
import numpy as np


def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """R² adjusted for number of predictors."""
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def max_overpredict(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum amount by which the model overpredicts g_limit.

    A positive value means the model predicted a higher (less conservative)
    g_limit than the FEA result — a safety-critical violation.
    """
    return float(np.max(y_pred - y_true))


def conservative_prediction_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that are on the safe (conservative) side.

    Conservative = y_pred <= y_true (model does not overstate the g-limit).
    """
    return float(np.mean(y_pred <= y_true))


def mos_01(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Required margin of safety (MOS) so that ≤1% of predictions exceed true g_limit.

    Returns the scalar MOS such that:
        P(y_pred - MOS > y_true) ≤ 0.01

    Equivalently: the 99th percentile of (y_pred - y_true).
    A pilot would use the adjusted prediction  y_pred - MOS  as their g_limit.
    """
    return float(max(0.0, np.percentile(y_pred - y_true, 99)))


def inference_time_ms(model, X: np.ndarray, n_reps: int = 100) -> float:
    """Median inference latency in milliseconds over n_reps repeated predict calls.

    Runs a warm-up call first to avoid cold-start JIT effects.
    """
    model.predict(X)  # warm-up
    times_ms = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model.predict(X)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times_ms))


def score(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_features: int,
    n_inference_reps: int = 100,
) -> dict:
    """Compute all six proposal metrics for a fitted model.

    Parameters
    ----------
    model       : any object with a .predict(X) method
    X_test      : feature matrix for the held-out test set
    y_test      : true g_limit values
    n_features  : number of input features (for adjusted R²)

    Returns
    -------
    dict with keys: adj_r2, mae, rmse, max_overpredict, mos_01, inference_time_ms
    """
    y_pred = model.predict(X_test)
    return {
        "adj_r2":           adjusted_r2(y_test, y_pred, n_features),
        "mae":              mae(y_test, y_pred),
        "rmse":             rmse(y_test, y_pred),
        "max_overpredict":  max_overpredict(y_test, y_pred),
        "mos_01":           mos_01(y_test, y_pred),
        "inference_time_ms": inference_time_ms(model, X_test, n_inference_reps),
    }
