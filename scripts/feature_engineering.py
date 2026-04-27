"""
AERO 489 — Feature Engineering Script
Reads all ABAQUS simulation CSVs and builds two polars DataFrames:
  1. features_scalar.parquet  — one row per simulation (classical ML input)
  2. time_series.parquet      — all load steps from all sims (for deep/recurrent models)

Target: g_limit — max allowable g-load before structural failure (Max_VM_Stress
        exceeds material yield strength).

Engineered features (per proposal §4.1):
  A.  Tip deflection per g-unit at each load step  [m / (m/s²)]
      → tip_deflection_slope, tip_deflection_per_g_at_failure
  B.  Average Strain Change Factor  [mm/mm per N]
      → avg_strain_slope, avg_strain_at_failure
  C.  Strain energy proxy  ½·E·Σεᵢ²  [Pa = J/m³, without per-node volumes]
      → strain_energy_at_failure, strain_energy_slope
      NOTE: multiply by average node volume (m³) to get true strain energy in J.
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
                   help="Dataset version tag, e.g. v1, v2a, v2b")
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
    """Compute scalar engineered features from one simulation's time series."""
    df = raw.filter(pl.col("Total_RF") > 0)
    if len(df) < 2:
        return None

    RF  = df["Total_RF"].to_numpy()          # N
    tip = df["Tip_Deflection"].to_numpy()    # m
    strain_mat = df.select(STRAIN_COLS).to_numpy()  # (steps, 24)

    last = df.row(-1, named=True)

    # ── Target ──────────────────────────────────────────────────────────────
    RF_fail  = last["Total_RF"]
    g_limit  = 2.0 * RF_fail / (AIRCRAFT_MASS_KG * G_ACCEL)

    # ── Feature A: Tip deflection / acceleration ─────────────────────────
    # acceleration  a = Total_RF / (m/2)  (one wing carries half the load)
    # tip_per_g    = Tip_Deflection / (Total_RF / (AIRCRAFT_MASS_KG * G_ACCEL / 2))
    g_loading = RF / (AIRCRAFT_MASS_KG * G_ACCEL / 2.0)   # g-units at each step
    tip_per_g = tip / np.where(g_loading > 0, g_loading, np.nan)

    tip_slope          = float(np.polyfit(RF, tip, 1)[0])
    tip_per_g_at_fail  = float(tip[-1] / g_loading[-1]) if g_loading[-1] > 0 else float("nan")

    # ── Feature B: Average Strain Change Factor ───────────────────────────
    avg_strain       = strain_mat.mean(axis=1)
    avg_strain_slope = float(np.polyfit(RF, avg_strain, 1)[0])

    # ── Feature C: Strain energy proxy (no node volumes) ─────────────────
    # True: U = 0.5 * E * sum(eps_i² * V_i)
    # Proxy: U_proxy = 0.5 * E * sum(eps_i²)  [Pa·m^0 without volumes]
    strain_sq_sum  = (strain_mat ** 2).sum(axis=1)
    strain_energy  = 0.5 * YOUNGS_MODULUS * strain_sq_sum
    se_slope       = float(np.polyfit(RF, strain_energy, 1)[0])

    k_spring = 1.0 / abs(tip_slope) if abs(tip_slope) > 1e-12 else float("nan")

    se_fail  = float(strain_energy[-1])
    vm_fail  = float(last["Max_VM_Stress"])

    row: dict = {
        "sim_id":                     sim_id,
        "n_steps":                    len(df),
        "RF_failure":                 float(RF_fail),
        "g_limit":                    float(g_limit),
        # Feature A
        "tip_deflection_at_failure":  float(last["Tip_Deflection"]),
        "tip_deflection_slope":       tip_slope,
        "tip_per_g_at_failure":       tip_per_g_at_fail,
        "k_spring":                   k_spring,
        # Feature B
        "avg_strain_at_failure":      float(avg_strain[-1]),
        "avg_strain_slope":           avg_strain_slope,
        # Feature C
        "strain_energy_at_failure":   se_fail,
        "strain_energy_slope":        se_slope,
        # Max stress
        "max_vm_stress_at_failure":   vm_fail,
        # Derived features D: inverse and sqrt transforms
        "inv_tip_per_g_at_failure":       1.0 / tip_per_g_at_fail if abs(tip_per_g_at_fail) > 1e-12 else float("nan"),
        "inv_tip_deflection_slope":       1.0 / tip_slope         if abs(tip_slope)          > 1e-12 else float("nan"),
        "inv_max_vm_stress_at_failure":   1.0 / vm_fail           if abs(vm_fail)            > 1e-12 else float("nan"),
        "sqrt_strain_energy_at_failure":  float(np.sqrt(max(se_fail, 0.0))),
    }

    # Individual gauge values at failure — useful as raw inputs for ML
    for col in STRAIN_COLS:
        row[f"{col}_failure"] = float(last[col])

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
    DATA_DIR = Path(f"training_data-{args.version}")
    OUT_DIR  = Path(f"features-{args.version}")

    all_cols     = _detect_columns(DATA_DIR)
    STRAIN_COLS  = [c for c in all_cols if c not in BASE_COLS]
    HEADER_COLS  = BASE_COLS + STRAIN_COLS
    N_NAMED_COLS = len(HEADER_COLS)
    print(f"Version: {args.version}  |  data: {DATA_DIR}  |  out: {OUT_DIR}")
    print(f"Detected {len(STRAIN_COLS)} strain/node columns")

    OUT_DIR.mkdir(exist_ok=True)

    files = sorted(DATA_DIR.glob("simulation_data_*.csv"))
    print(f"Found {len(files)} simulation files")

    scalar_rows: list[dict] = []
    ts_frames:   list[pl.DataFrame] = []
    skipped = 0

    for path in files:
        m = re.search(r"simulation_data_(\d+)\.csv", path.name)
        sim_id = int(m.group(1)) if m else -1

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

    # ── Scalar feature DataFrame ──────────────────────────────────────────
    scalar_df = pl.DataFrame(scalar_rows)
    scalar_df.write_parquet(OUT_DIR / "features_scalar.parquet")
    print(f"\nScalar features  → {OUT_DIR/'features_scalar.parquet'}")
    print(f"  Shape : {scalar_df.shape}")
    print(f"  g_limit range: {scalar_df['g_limit'].min():.3f} – {scalar_df['g_limit'].max():.3f} g")
    print(scalar_df.select(["sim_id","n_steps","g_limit","tip_deflection_at_failure",
                             "avg_strain_slope","strain_energy_at_failure"]).head(5))

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
