#!/usr/bin/env python3
"""
Add rank-ordered strain gauge features to features_scalar.parquet.

New columns (5 total):
  ranked_strain_p04   — 4th-lowest gauge SLOPE  (low-end R² peak)
  ranked_strain_p23   — 23rd-lowest gauge slope  (global R² peak)
  ranked_strain_p24   — maximum gauge slope       (rank 24)
  gompertz_log_b      — Gompertz shape parameter log(b)  (initial suppression)
  gompertz_c          — Gompertz shape parameter c        (growth rate across ranks)

Gauge slopes ({Node_col}_slope) are the per-gauge strain rate per Newton of applied
load — computable from pre-failure flight data without knowing the failure moment.

Gompertz model fitted per simulation on the rank-normalized slope profile:
  y_norm(x) = exp(-b * exp(-c * x))   where  y_norm = slope / max_slope
Amplitude is encoded separately in ranked_strain_p24, so only b and c are fitted.
This normalization avoids the identifiability issue that makes a, b, c individually
unstable when all three are free (a=6e49 observed on the median curve).

Usage:
  python scripts/add_ranked_features.py [--version v2]
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent


def _fit_gompertz_linearized(ranks: np.ndarray, y_norm: np.ndarray):
    """
    Linearized Gompertz fit via double-log transform.

    log(-log(y_norm)) = log(b) - c * rank

    Fits by OLS on the transformed scale, avoiding the numerical instability of
    nonlinear curve_fit (which pushes b → 1e49 when rank-1 strain is near zero).

    Returns (log_b, c).  log_b = log(b) is stored directly as the feature so that
    the log-scale value is well-conditioned (range ~0 to ~10 across the dataset).
    """
    # Exclude ranks where y_norm ≥ 1 or ≤ 0 (double-log undefined)
    valid = (y_norm > 1e-10) & (y_norm < 1.0 - 1e-10)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    w = np.log(-np.log(y_norm[valid]))
    r = ranks[valid]
    slope, intercept = np.polyfit(r, w, 1)   # w = intercept + slope * rank
    log_b = float(intercept)                  # log(b) = intercept
    c     = float(-slope)                     # c = -slope (positive for increasing curve)
    return log_b, c


def add_ranked_features(parquet_path: Path) -> None:
    df = pl.read_parquet(parquet_path)
    # Detect per-gauge slope columns produced by feature_engineering.py
    node_cols = sorted(c for c in df.columns
                       if c.startswith("Node_at_") and c.endswith("_slope"))
    G = len(node_cols)
    if G == 0:
        raise ValueError("No Node_at_* columns found — wrong parquet?")

    gauges = df.select(node_cols).to_numpy().astype(float)  # (N, G)
    N = gauges.shape[0]
    ranks = np.arange(1, G + 1, dtype=float)

    p04_vals = np.empty(N)
    p23_vals = np.empty(N)
    p24_vals = np.empty(N)
    gb_vals  = np.empty(N)
    gc_vals  = np.empty(N)

    # Percentile-proportional indices (preserve semantics across gauge counts)
    # p04 ≈ 17th pct, p23 ≈ 96th pct, p24 = max
    idx_p04 = max(0, round(G * 4  / 24) - 1)  # ~17th percentile
    idx_p23 = max(0, round(G * 23 / 24) - 1)  # ~96th percentile
    idx_p24 = G - 1                             # maximum
    print(f"  G={G} gauges  |  p04→rank{idx_p04+1}  p23→rank{idx_p23+1}  p24→rank{idx_p24+1}")

    n_fail = 0
    for i in range(N):
        sorted_s = np.sort(gauges[i])
        p04_vals[i] = sorted_s[idx_p04]
        p23_vals[i] = sorted_s[idx_p23]
        p24_vals[i] = sorted_s[idx_p24]  # max

        # Shift slopes so minimum = 0 before Gompertz normalization.
        # Gauge slopes can be negative (near-zero-load or compression nodes);
        # Gompertz requires a non-negative, normalized [0,1] profile.
        s_shifted = sorted_s - sorted_s[0]
        amp = s_shifted[-1]
        if amp < 1e-20:
            gb_vals[i] = gc_vals[i] = float("nan")
            n_fail += 1
            continue

        y_norm = s_shifted / amp
        log_b, c = _fit_gompertz_linearized(ranks, y_norm)
        if not np.isfinite(log_b) or not np.isfinite(c):
            n_fail += 1
        gb_vals[i] = log_b   # store log(b) — well-conditioned, range ~0–10
        gc_vals[i] = c

    print(f"  N={N}  Gompertz failures={n_fail}")
    print(f"  gompertz_log_b  median={np.nanmedian(gb_vals):.4f}  "
          f"min={np.nanmin(gb_vals):.4f}  max={np.nanmax(gb_vals):.4f}")
    print(f"  gompertz_c      median={np.nanmedian(gc_vals):.4f}  "
          f"min={np.nanmin(gc_vals):.4f}  max={np.nanmax(gc_vals):.4f}")

    df = df.with_columns([
        pl.Series("ranked_strain_p04",    p04_vals),
        pl.Series("ranked_strain_p23",    p23_vals),
        pl.Series("ranked_strain_p24",    p24_vals),
        pl.Series("gompertz_log_b",       gb_vals),   # log(b): initial suppression on log scale
        pl.Series("gompertz_c",           gc_vals),   # growth rate across ranks
    ])

    df.write_parquet(parquet_path)
    print(f"  Written → {parquet_path}  shape={df.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    args = ap.parse_args()

    parquet = ROOT / f"features-{args.version}" / "features_scalar.parquet"
    if not parquet.exists():
        raise FileNotFoundError(parquet)

    print(f"Adding ranked strain features to {parquet}")
    add_ranked_features(parquet)
    print("Done.")


if __name__ == "__main__":
    main()
