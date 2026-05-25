#!/usr/bin/env python3
"""
AERO 489 — Add transform-family columns to features_scalar.parquet.

For each of the 6 slope-based base features, computes:
  ln_{feat}      — natural log transform
  log10_{feat}   — base-10 log transform
  exp_{feat}     — exp of min-max normalised values → range [1, e]
  pow10_{feat}   — 10^(min-max normalised) → range [1, 10]
  boxcox_{feat}  — optimal power transform x^λ*, λ* chosen to maximise R²(g_limit)

All transforms are stored back into the same parquet file.

Box-Cox lambda values are fit on the full dataset (all simulations) to find
the univariate R²-maximising power per feature.  The train/test split happens
downstream in data_utils.load_scalar().

Usage:
  python scripts/add_transforms.py [--version v2]
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent

BASE_FEATURES = [
    "tip_deflection_slope",
    "avg_strain_slope",
    "strain_energy_slope",
    "k_spring",
    "inv_tip_deflection_slope",
]

LAMBDA_GRID = np.arange(-15.0, 15.5, 0.5)


def _power_transform(x: np.ndarray, lam: float) -> np.ndarray:
    """Apply x^λ with special case ln(x) when λ≈0."""
    if abs(lam) < 1e-6:
        return np.log(x)
    return np.sign(x) * (np.abs(x) ** lam)


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns zeros if range is degenerate."""
    lo, hi = x.min(), x.max()
    rng = hi - lo
    if rng < 1e-30:
        return np.zeros_like(x)
    return (x - lo) / rng


def _best_lambda(x: np.ndarray, y: np.ndarray) -> float:
    """Find λ* ∈ LAMBDA_GRID maximising |R(g_limit, x^λ)|."""
    # Shift x to be strictly positive before power transform
    x_pos = x - x.min() + 1e-10 * (x.max() - x.min() + 1e-30)
    best_r2, best_lam = -1.0, 1.0
    for lam in LAMBDA_GRID:
        xt = _power_transform(x_pos, lam)
        if not np.all(np.isfinite(xt)):
            continue
        try:
            r, _ = pearsonr(xt, y)
        except Exception:
            continue
        if np.isfinite(r) and r ** 2 > best_r2:
            best_r2 = r ** 2
            best_lam = float(lam)
    return best_lam


def add_transforms(parquet_path: Path) -> None:
    df = pl.read_parquet(parquet_path)
    y  = df["g_limit"].to_numpy().astype(float)

    new_cols: dict[str, np.ndarray] = {}

    for feat in BASE_FEATURES:
        if feat not in df.columns:
            print(f"  [WARN] {feat} not found — skipping transforms")
            continue

        x = df[feat].to_numpy().astype(float)

        # Shift to strictly positive for log/exp transforms
        x_pos = x - x.min() + 1e-10 * (x.max() - x.min() + 1e-30)

        # Natural log and log10
        new_cols[f"ln_{feat}"]    = np.log(x_pos)
        new_cols[f"log10_{feat}"] = np.log10(x_pos)

        # exp and pow10 on min-max normalised → [1, e] and [1, 10]
        x_norm = _minmax_norm(x_pos)
        new_cols[f"exp_{feat}"]   = np.exp(x_norm)
        new_cols[f"pow10_{feat}"] = 10.0 ** x_norm

        # Box-Cox: optimal power transform maximising R²(g_limit)
        lam = _best_lambda(x, y)
        x_boxcox = _power_transform(x_pos, lam)
        new_cols[f"boxcox_{feat}"] = x_boxcox
        print(f"  {feat:<30s}  λ*={lam:+.1f}  R²={pearsonr(x_boxcox, y)[0]**2:.4f}")

    # Append all new columns
    df = df.with_columns([
        pl.Series(name, vals) for name, vals in new_cols.items()
    ])

    df.write_parquet(parquet_path)
    print(f"  Written → {parquet_path}  shape={df.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    args = ap.parse_args()

    parquet = ROOT / f"features-{args.version}" / "features_scalar.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} not found — run feature_engineering.py first")

    print(f"Adding transform columns to {parquet}")
    print(f"Base features: {BASE_FEATURES}")
    print(f"Lambda grid: [{LAMBDA_GRID[0]}, {LAMBDA_GRID[-1]}] step 0.5")
    add_transforms(parquet)
    print("Done.")


if __name__ == "__main__":
    main()
