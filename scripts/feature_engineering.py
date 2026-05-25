"""
AERO 489 — Feature Engineering Script
Reads all ABAQUS simulation CSVs and builds two polars DataFrames:
  1. features_scalar.parquet  — one row per simulation (classical ML input)
  2. time_series.parquet      — all load steps from all sims (for deep/recurrent models)

Target: g_limit — max allowable g-load before structural failure (Max_VM_Stress
        exceeds material yield strength).

Engineered features — all slope-based, derived from onboard sensors only:
  Sensors available in deployment: IMU (g-load), tip-deflection camera, 24 strain gauges.
  Load axis: Total_RF [N] = g_load × AIRCRAFT_MASS_KG × G_ACCEL / 2

  A.  Tip deflection slope vs. applied load  [m/N]
      → tip_deflection_slope, k_spring (= 1/tip_slope), inv_tip_deflection_slope
      (camera + IMU)
  B.  Average strain slope vs. applied load  [strain/N]
      → avg_strain_slope
      (mean of 24 gauges + IMU)
  C.  Strain energy proxy slope  ½·E·d(Σεᵢ²)/dRF  [Pa/N]
      → strain_energy_slope
      (0.5·E·sum(gauge²) computable onboard from gauges + IMU)
  D.  Per-gauge strain slope  [strain/N] for each of the 24 structural nodes
      → {Node_col}_slope  (one per gauge + IMU)

All features are rates of change per Newton, computable at any point during
normal flight without knowing when (or if) failure will occur.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import polars as pl

# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1",
                   help="Dataset version tag, e.g. v1, v2a, v2b, v2 (merges v2a+v2b)")
    return p.parse_args()

# ── Parameters — set in main() after arg parsing ─────────────────────────────
DATA_DIR: Path
OUT_DIR:  Path
STRAIN_COLS:  list[str]
HEADER_COLS:  list[str]
N_NAMED_COLS: int

G_ACCEL          = 9.81        # m/s²
AIRCRAFT_MASS_KG = 16_500.0    # A-10 operational weight estimate (kg) — update with actual
YOUNGS_MODULUS   = 71.0e9      # Pa  (7075-T6 Al) — update if different material

BASE_COLS = ["Time", "Total_RF", "Max_VM_Stress", "Tip_Deflection"]


def _detect_columns(data_dir: Path) -> list[str]:
    """Read the header of the first simulation file and return all column names."""
    first = next(iter(sorted(data_dir.glob("simulation_data_*.csv"))), None)
    if first is None:
        raise FileNotFoundError(f"No simulation_data_*.csv files found in {data_dir}")
    with open(first) as f:
        for line in f:
            if line.strip():
                return next(csv.reader([line.strip()]))
    raise ValueError(f"Could not read header from {first}")

# ── CSV reader ───────────────────────────────────────────────────────────────

def read_sim_csv(path: Path) -> pl.DataFrame | None:
    """Parse one simulation CSV.

    Format quirks handled:
    - Blank lines between every data row
    - Header has 28 named columns; data rows may have more (extra ABAQUS outputs)
    - Only the first N_NAMED_COLS values are used
    """
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    if len(lines) < 2:
        return None

    rows = []
    for line in lines[1:]:  # skip header row
        vals = line.split(",")
        if len(vals) < N_NAMED_COLS:
            continue
        try:
            rows.append([float(v) for v in vals[:N_NAMED_COLS]])
        except ValueError:
            continue

    if not rows:
        return None

    return pl.DataFrame(rows, schema=HEADER_COLS, orient="row")


# ── Per-simulation feature extraction ────────────────────────────────────────

def extract_features(sim_id: int, raw: pl.DataFrame) -> dict | None:
    """Compute scalar engineered features from one simulation's time series.

    All features are slope-based (rate of change per Newton of applied load),
    computable from any pre-failure flight data without knowing the failure moment.
    """
    df = raw.filter(pl.col("Total_RF") > 0)
    if len(df) < 2:
        return None

    RF         = df["Total_RF"].to_numpy()          # N
    tip        = df["Tip_Deflection"].to_numpy()    # m
    strain_mat = df.select(STRAIN_COLS).to_numpy()  # (steps, n_gauges)

    last = df.row(-1, named=True)

    # ── Target ──────────────────────────────────────────────────────────────
    RF_fail = last["Total_RF"]
    g_limit = 2.0 * RF_fail / (AIRCRAFT_MASS_KG * G_ACCEL)

    # ── Feature A: Tip deflection slope [m/N] ────────────────────────────
    tip_slope = float(np.polyfit(RF, tip, 1)[0])
    k_spring  = 1.0 / abs(tip_slope) if abs(tip_slope) > 1e-12 else float("nan")

    # ── Feature B: Average strain slope [strain/N] ───────────────────────
    avg_strain       = strain_mat.mean(axis=1)
    avg_strain_slope = float(np.polyfit(RF, avg_strain, 1)[0])

    # ── Feature C: Strain energy proxy slope [Pa/N] ──────────────────────
    # Proxy: U_proxy = 0.5 * E * sum(eps_i²)  [Pa, without per-node volumes]
    strain_sq_sum = (strain_mat ** 2).sum(axis=1)
    strain_energy = 0.5 * YOUNGS_MODULUS * strain_sq_sum
    se_slope      = float(np.polyfit(RF, strain_energy, 1)[0])

    row: dict = {
        "sim_id":               sim_id,
        "n_steps":              len(df),
        "RF_failure":           float(RF_fail),
        "g_limit":              float(g_limit),
        # Feature A — tip deflection
        "tip_deflection_slope": tip_slope,
        "k_spring":             k_spring,
        "inv_tip_deflection_slope": 1.0 / tip_slope if abs(tip_slope) > 1e-12 else float("nan"),
        # Feature B — average strain
        "avg_strain_slope":     avg_strain_slope,
        # Feature C — strain energy proxy
        "strain_energy_slope":  se_slope,
    }

    # Feature E: per-gauge strain slopes [strain/N]
    for k, col in enumerate(STRAIN_COLS):
        row[f"{col}_slope"] = float(np.polyfit(RF, strain_mat[:, k], 1)[0])

    return row


# ── Time-series dataframe builder ─────────────────────────────────────────────

def build_time_series_row(sim_id: int, df: pl.DataFrame) -> pl.DataFrame:
    """Return a dataframe with one row per load step, annotated with sim_id and g_limit."""
    df = df.filter(pl.col("Total_RF") > 0)
    if len(df) == 0:
        return pl.DataFrame()

    RF_fail = df["Total_RF"].max()
    g_limit  = 2.0 * RF_fail / (AIRCRAFT_MASS_KG * G_ACCEL)

    return df.with_columns([
        pl.lit(sim_id).alias("sim_id"),
        pl.lit(float(g_limit)).alias("g_limit"),
        (pl.col("Total_RF") / float(RF_fail)).alias("load_fraction"),
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global DATA_DIR, OUT_DIR, STRAIN_COLS, HEADER_COLS, N_NAMED_COLS

    args = _parse_args()
    OUT_DIR = Path("features") / args.version

    # Support --version v2 by merging training_data/v2a + training_data/v2b
    if args.version == "v2":
        data_dirs = [Path("training_data/v2a"), Path("training_data/v2b")]
    else:
        data_dirs = [Path("training_data") / args.version]

    # Detect columns from the first available data directory
    DATA_DIR = data_dirs[0]
    all_cols     = _detect_columns(DATA_DIR)
    STRAIN_COLS  = [c for c in all_cols if c not in BASE_COLS]
    HEADER_COLS  = BASE_COLS + STRAIN_COLS
    N_NAMED_COLS = len(HEADER_COLS)
    print(f"Version: {args.version}  |  data dirs: {[str(d) for d in data_dirs]}  |  out: {OUT_DIR}")
    print(f"Detected {len(STRAIN_COLS)} strain/node columns")

    OUT_DIR.mkdir(exist_ok=True)

    scalar_rows: list[dict] = []
    ts_frames:   list[pl.DataFrame] = []
    skipped = 0
    sim_id_offset = 0  # offset to avoid sim_id collisions when merging dirs

    for data_dir in data_dirs:
        files = sorted(data_dir.glob("simulation_data_*.csv"))
        print(f"\nProcessing {len(files)} files from {data_dir} ...")

        for path in files:
            m = re.search(r"simulation_data_(\d+)\.csv", path.name)
            raw_id = int(m.group(1)) if m else -1
            sim_id = raw_id + sim_id_offset

            raw = read_sim_csv(path)
            if raw is None:
                skipped += 1
                continue

            feat = extract_features(sim_id, raw)
            if feat is not None:
                scalar_rows.append(feat)

            ts = build_time_series_row(sim_id, raw)
            if len(ts) > 0:
                ts_frames.append(ts)

        # Offset ensures unique sim_ids when merging multiple directories
        if scalar_rows:
            sim_id_offset = max(r["sim_id"] for r in scalar_rows) + 1

    # ── Scalar feature DataFrame ──────────────────────────────────────────
    scalar_df = pl.DataFrame(scalar_rows)
    scalar_df.write_parquet(OUT_DIR / "features_scalar.parquet")
    print(f"\nScalar features  → {OUT_DIR/'features_scalar.parquet'}")
    print(f"  Shape : {scalar_df.shape}")
    print(f"  g_limit range: {scalar_df['g_limit'].min():.3f} – {scalar_df['g_limit'].max():.3f} g")
    print(scalar_df.select(["sim_id","n_steps","g_limit",
                             "avg_strain_slope","strain_energy_slope"]).head(5))

    # ── Time-series DataFrame ─────────────────────────────────────────────
    ts_df = pl.concat(ts_frames)
    ts_df.write_parquet(OUT_DIR / "time_series.parquet")
    print(f"\nTime-series data → {OUT_DIR/'time_series.parquet'}")
    print(f"  Shape : {ts_df.shape}")
    print(f"  Sims  : {ts_df['sim_id'].n_unique()}, "
          f"total load steps: {len(ts_df)}")

    if skipped:
        print(f"\nSkipped {skipped} files (empty or malformed)")


if __name__ == "__main__":
    main()
