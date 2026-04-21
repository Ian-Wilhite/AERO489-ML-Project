"""
AERO 489 — Data loading and train/test split utilities.

Single source of truth for:
  - Feature set definitions (column name constants)
  - Loading features_scalar.parquet  → scalar ML inputs (Models 1–6)
  - Loading time_series.parquet      → sequential inputs (Model 7 LSTM)
  - Reproducible 80/20 train/test split (seed=42, split by sim_id for time series)

All model scripts should import from here so every model trains and evaluates
on identical partitions.
"""

from pathlib import Path
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR    = Path("features")
SCALAR_PARQUET  = FEATURES_DIR / "features_scalar.parquet"
TS_PARQUET      = FEATURES_DIR / "time_series.parquet"

TARGET = "g_limit"

# ── Feature set constants ─────────────────────────────────────────────────────

# 7 hand-crafted features (proposal §4.1 A/B/C + n_steps)
ENGINEERED_COLS = [
    "tip_deflection_slope",
    "tip_per_g_at_failure",
    "avg_strain_at_failure",
    "avg_strain_slope",
    "strain_energy_at_failure",
    "strain_energy_slope",
    "n_steps",
]

# 24 individual strain gauge readings at failure step
RAW_GAUGE_COLS = [
    "Strain_Node_107_failure", "Strain_Node_151_failure", "Strain_Node_172_failure",
    "Strain_Node_180_failure", "Strain_Node_192_failure", "Strain_Node_201_failure",
    "Strain_Node_250_failure", "Strain_Node_257_failure", "Strain_Node_266_failure",
    "Strain_Node_281_failure", "Strain_Node_305_failure", "Strain_Node_327_failure",
    "Strain_Node_344_failure", "Strain_Node_368_failure", "Strain_Node_375_failure",
    "Strain_Node_407_failure", "Strain_Node_415_failure", "Strain_Node_447_failure",
    "Strain_Node_463_failure", "Strain_Node_472_failure", "Strain_Node_488_failure",
    "Strain_Node_502_failure", "Strain_Node_506_failure", "Strain_Node_510_failure",
]

# 31-column combined set — default for classical models and NN
ALL_SCALAR_COLS = ENGINEERED_COLS + RAW_GAUGE_COLS

# 26 time-series channels per load step (24 gauges + tip_deflection + load_fraction)
TS_FEATURE_COLS = [
    c.replace("_failure", "")
    for c in RAW_GAUGE_COLS
] + ["Tip_Deflection", "load_fraction"]

MAX_SEQ_LEN = 29  # maximum load steps across all simulations (for LSTM padding)

# ── Scalar loader ─────────────────────────────────────────────────────────────

def load_scalar(
    feature_cols: list[str] = ALL_SCALAR_COLS,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load scalar features and return an 80/20 train/test split.

    Returns
    -------
    X_train, X_test : np.ndarray  shape (n_train, n_features), (n_test, n_features)
    y_train, y_test : np.ndarray  shape (n_train,), (n_test,)
    """
    # TODO: verify SCALAR_PARQUET exists; raise a clear error if feature_engineering.py
    #       has not been run yet
    df = pl.read_parquet(SCALAR_PARQUET).drop_nulls()

    X = df.select(feature_cols).to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    return train_test_split(X, y, test_size=test_size, random_state=seed)


# ── Time-series loader ────────────────────────────────────────────────────────

def load_timeseries(
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load time-series data and split by sim_id (no simulation leaks across folds).

    Returns
    -------
    X_train, X_test : np.ndarray  shape (n_sims, MAX_SEQ_LEN, n_channels)
                      Zero-padded to MAX_SEQ_LEN; actual length stored in seq_lens.
    y_train, y_test : np.ndarray  shape (n_sims,)

    Notes
    -----
    The split is performed on sim_ids before building the padded array so that
    every load step from a given simulation lands in only train or only test.
    """
    # TODO: load TS_PARQUET with polars
    # TODO: get sorted list of unique sim_ids
    # TODO: split sim_ids 80/20 with train_test_split(sim_ids, ..., random_state=seed)
    # TODO: for each split, group by sim_id → build padded array shape
    #       (n_sims, MAX_SEQ_LEN, len(TS_FEATURE_COLS)) and target vector y
    # TODO: return X_train, X_test, y_train, y_test
    raise NotImplementedError("load_timeseries not yet implemented")
