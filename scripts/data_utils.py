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
FEATURES_DIR   = Path(__file__).resolve().parent.parent / "features-v2"
SCALAR_PARQUET = FEATURES_DIR / "features_scalar.parquet"
TS_PARQUET     = FEATURES_DIR / "time_series.parquet"

TARGET = "g_limit"

# ── Feature set constants ─────────────────────────────────────────────────────

# 7 hand-crafted features (proposal §4.1 A/B/C); n_steps removed (sim artifact)
ENGINEERED_COLS = [
    "tip_deflection_slope",
    "tip_per_g_at_failure",
    "avg_strain_at_failure",
    "avg_strain_slope",
    "strain_energy_at_failure",
    "strain_energy_slope",
    "k_spring",
]

# Non-gauge columns that should never be treated as raw gauge readings
_META_COLS = {
    "sim_id", "n_steps", "RF_failure", "g_limit",
    "tip_deflection_at_failure", "max_vm_stress_at_failure",
} | set(ENGINEERED_COLS)


def _read_gauge_cols() -> list[str]:
    """Auto-detect node/_failure columns from the scalar parquet header."""
    try:
        schema = pl.read_parquet_schema(SCALAR_PARQUET)
        return [c for c in schema if c.endswith("_failure") and c not in _META_COLS]
    except Exception:
        return []


def _read_ts_feature_cols() -> list[str]:
    """Auto-detect time-series feature channels from the TS parquet header."""
    try:
        schema = pl.read_parquet_schema(TS_PARQUET)
        skip = {"sim_id", "g_limit", "Time", "Total_RF", "Max_VM_Stress"}
        return [c for c in schema if c not in skip]
    except Exception:
        return []


RAW_GAUGE_COLS  = _read_gauge_cols()
ALL_SCALAR_COLS = ENGINEERED_COLS + RAW_GAUGE_COLS

TS_FEATURE_COLS = _read_ts_feature_cols()
MAX_SEQ_LEN     = 29  # verified from v2 data


# ── Scalar loader ─────────────────────────────────────────────────────────────

def load_scalar(
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load scalar features and return an 80/20 train/test split.

    Returns
    -------
    X_train, X_test : np.ndarray  shape (n_train, n_features), (n_test, n_features)
    y_train, y_test : np.ndarray  shape (n_train,), (n_test,)
    """
    if feature_cols is None:
        feature_cols = ALL_SCALAR_COLS
    if not SCALAR_PARQUET.exists():
        raise FileNotFoundError(
            f"{SCALAR_PARQUET} not found — run feature_engineering.py first."
        )
    df = pl.read_parquet(SCALAR_PARQUET).drop_nulls()
    X  = df.select(feature_cols).to_numpy().astype(float)
    y  = df[TARGET].to_numpy().astype(float)
    return train_test_split(X, y, test_size=test_size, random_state=seed)


# ── Time-series loader ────────────────────────────────────────────────────────

def load_timeseries(
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load time-series data, split by sim_id, and return padded arrays.

    Returns
    -------
    X_train, X_test  : np.ndarray  shape (n_sims, MAX_SEQ_LEN, n_channels)
    y_train, y_test  : np.ndarray  shape (n_sims,)
    sl_train, sl_test: np.ndarray  shape (n_sims,)  actual step counts before padding
    """
    if not TS_PARQUET.exists():
        raise FileNotFoundError(
            f"{TS_PARQUET} not found — run feature_engineering.py first."
        )
    df = pl.read_parquet(TS_PARQUET)

    feat_cols = TS_FEATURE_COLS
    n_ch      = len(feat_cols)
    max_len   = MAX_SEQ_LEN

    sim_ids = sorted(df["sim_id"].unique().to_list())
    ids_train, ids_test = train_test_split(sim_ids, test_size=test_size, random_state=seed)

    def _build(ids: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n   = len(ids)
        X   = np.zeros((n, max_len, n_ch), dtype=np.float32)
        y   = np.zeros(n, dtype=np.float32)
        sl  = np.zeros(n, dtype=np.int64)
        for i, sid in enumerate(ids):
            grp    = df.filter(pl.col("sim_id") == sid)
            steps  = grp.select(feat_cols).to_numpy().astype(np.float32)
            n_steps = min(len(steps), max_len)
            X[i, :n_steps] = steps[:n_steps]
            y[i]  = float(grp["g_limit"][0])
            sl[i] = n_steps
        return X, y, sl

    X_tr, y_tr, sl_tr = _build(ids_train)
    X_te, y_te, sl_te = _build(ids_test)
    return X_tr, X_te, y_tr, y_te, sl_tr, sl_te
