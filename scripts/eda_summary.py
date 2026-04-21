"""
AERO 489 — EDA Summary
Prints a 5-number summary table and Pearson correlation with g_limit
for every feature column in features/features_scalar.parquet.
"""

from pathlib import Path
import polars as pl
import numpy as np

PARQUET = Path("features/features_scalar.parquet")
TARGET  = "g_limit"
EXCLUDE = {"sim_id", TARGET}


def five_number_summary(series: pl.Series) -> dict:
    a = series.drop_nulls().to_numpy()
    return {
        "min":    float(np.min(a)),
        "Q1":     float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "Q3":     float(np.percentile(a, 75)),
        "max":    float(np.max(a)),
        "mean":   float(np.mean(a)),
        "std":    float(np.std(a, ddof=1)),
    }


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main() -> None:
    df = pl.read_parquet(PARQUET)
    target = df[TARGET].to_numpy()

    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    # ── Header ────────────────────────────────────────────────────────────────
    col_w = 36
    fmt   = f"{{:<{col_w}}} {{:>10}} {{:>10}} {{:>10}} {{:>10}} {{:>10}} {{:>10}} {{:>10}} {{:>8}}"
    header = fmt.format("Feature", "Min", "Q1", "Median", "Q3", "Max", "Mean", "Std", "r (g_limit)")
    sep    = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    rows = []
    for col in feature_cols:
        s    = df[col]
        stat = five_number_summary(s)
        r    = pearson_r(s.to_numpy(), target)
        rows.append((col, stat, r))

    # Sort by |r| descending
    rows.sort(key=lambda t: abs(t[2]) if np.isfinite(t[2]) else -1, reverse=True)

    for col, stat, r in rows:
        r_str = f"{r:+.4f}" if np.isfinite(r) else "   NaN"
        print(fmt.format(
            col[:col_w],
            f"{stat['min']:.4g}",
            f"{stat['Q1']:.4g}",
            f"{stat['median']:.4g}",
            f"{stat['Q3']:.4g}",
            f"{stat['max']:.4g}",
            f"{stat['mean']:.4g}",
            f"{stat['std']:.4g}",
            r_str,
        ))

    print(sep)

    # ── g_limit summary ───────────────────────────────────────────────────────
    print(f"\nTarget ({TARGET}) — 5-number summary:")
    ts = five_number_summary(df[TARGET])
    print(f"  min={ts['min']:.4f}  Q1={ts['Q1']:.4f}  median={ts['median']:.4f}"
          f"  Q3={ts['Q3']:.4f}  max={ts['max']:.4f}"
          f"  mean={ts['mean']:.4f}  std={ts['std']:.4f}")

    print(f"\nN simulations : {len(df)}")
    print(f"N features    : {len(feature_cols)}")


if __name__ == "__main__":
    main()