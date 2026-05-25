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
FEATURES_DIR   = Path(__file__).resolve().parent.parent / "features/v2"
SCALAR_PARQUET = FEATURES_DIR / "features_scalar.parquet"
TS_PARQUET     = FEATURES_DIR / "time_series.parquet"

TARGET = "g_limit"

# ── Feature set constants ─────────────────────────────────────────────────────

# Slope-based features — rates of change per Newton of applied load.
# All computable from pre-failure flight data (no at-failure state required).
_BASE_ENGINEERED = [
    "tip_deflection_slope",      # tip displacement rate [m/N]  — camera + IMU
    "avg_strain_slope",          # mean gauge strain rate [strain/N]  — gauges + IMU
    "strain_energy_slope",       # strain energy proxy rate [Pa/N]  — gauges + IMU
    "k_spring",                  # structural stiffness = 1/tip_slope [N/m]  — derived
    "inv_tip_deflection_slope",  # = 1/tip_slope (alternative stiffness form)  — derived
]

# log/exp transforms for all 6 base slope features
_LN_ENGINEERED    = [f"ln_{c}"    for c in _BASE_ENGINEERED]
_LOG10_ENGINEERED = [f"log10_{c}" for c in _BASE_ENGINEERED]

# exp/pow10 applied to min-max normalised features -> [1,e] and [1,10], no overflow
_EXP_ENGINEERED   = [f"exp_{c}"   for c in _BASE_ENGINEERED]
_POW10_ENGINEERED = [f"pow10_{c}" for c in _BASE_ENGINEERED]

# Box-Cox optimal power transform (x^λ*) per feature — maximises univariate R² with g_limit
_BOXCOX_ENGINEERED = [f"boxcox_{c}" for c in _BASE_ENGINEERED]

# 6 Box-Cox features — one optimally-transformed column per base slope feature.
# Best feature set for linear models and GPR: pre-linearised, no collinear pairs.
BOXCOX_COLS = _BOXCOX_ENGINEERED

# Reduced Box-Cox set for linear regression — drops features that are mathematically
# redundant (inv_tip_deflection_slope = 1/tip_slope, so their boxcox transforms are
# equivalent up to sign; keep k_spring which is the more physically interpretable form).
_BOXCOX_LR_DROP = {
    "boxcox_inv_tip_deflection_slope",  # = 1/tip_slope → equivalent to boxcox_k_spring
    "boxcox_tip_deflection_slope",      # same information as boxcox_k_spring
}
BOXCOX_COLS_LR = [c for c in BOXCOX_COLS if c not in _BOXCOX_LR_DROP]

# Rank-ordered strain gauge slope features (5 scalars).
# Gauge slopes (strain rate per Newton) are sorted and three percentile positions
# are extracted.  Gompertz shape params encode the rank-profile shape compactly.
RANKED_STRAIN_COLS = [
    "ranked_strain_p04",   # ~17th-percentile gauge slope  (low-end sensitivity)
    "ranked_strain_p23",   # ~96th-percentile gauge slope  (near-peak sensitivity)
    "ranked_strain_p24",   # maximum gauge slope
    "gompertz_log_b",      # log(b): Gompertz initial suppression on log scale
    "gompertz_c",          # Gompertz growth-rate across slope ranks
]

ENGINEERED_COLS = (
    _BASE_ENGINEERED
    + _LN_ENGINEERED
    + _LOG10_ENGINEERED
    + _EXP_ENGINEERED
    + _POW10_ENGINEERED
    + _BOXCOX_ENGINEERED
)

# Non-gauge columns that should never be treated as raw gauge slope readings
_META_COLS = {
    "sim_id", "n_steps", "RF_failure", "g_limit",
} | set(ENGINEERED_COLS) | set(RANKED_STRAIN_COLS)


def _read_gauge_cols() -> list[str]:
    """Auto-detect per-gauge slope columns from the scalar parquet header."""
    try:
        schema = pl.read_parquet_schema(SCALAR_PARQUET)
        return [c for c in schema
                if c.startswith("Node_at_") and c.endswith("_slope")
                and c not in _META_COLS]
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
ALL_SCALAR_COLS = ENGINEERED_COLS + RAW_GAUGE_COLS + RANKED_STRAIN_COLS

# ── Curated model-specific feature sets ───────────────────────────────────────

# Greedy-forward-selection (Box-Cox + ranked strain pool, CV peaks at k=1).
# CV is flat in the 0.84–0.85 range across all k; k=4 chosen as a practical
# multi-feature set that is still interpretable for Linear Reg.
GREEDY_8_COLS = [
    "boxcox_strain_energy_slope",   # k=1  CV=0.661
    "ranked_strain_p23",            # k=2  CV=0.844
    "ranked_strain_p24",            # k=3  CV=0.856
    "gompertz_c",                   # k=4  CV=0.879
    "gompertz_log_b",               # k=5  CV=0.885
    "boxcox_k_spring",              # k=6  CV=0.894 ← CV peak
]

# Physics-accessible feature set for PINN: base slope features so the
# physics-residual lookup by column name is stable.
# Ranked strain appended at the end adds gauge-level signal.
PINN_COLS = _BASE_ENGINEERED + RANKED_STRAIN_COLS  # 11 features

# Best compact set for RF and FFNN: Box-Cox pre-linearised features plus
# rank-ordered gauge signal.  No redundant ln/log10/exp/pow10 transform families.
RF_NN_COLS = BOXCOX_COLS + RANKED_STRAIN_COLS  # 11 features

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
