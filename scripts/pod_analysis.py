"""
AERO 489 — POD / PCA Component Analysis
Determines how many modes are needed to approximate g_limit, then builds
the best Principal Component Regression (PCR) model for two feature sets:
  (1) ORIGINAL — 24 raw strain-gauge readings + tip deflection + max VM stress
  (2) ENGINEERED — 7 hand-crafted features (slopes, per-g, strain energy)

Outputs
-------
  pod_variance.png   — cumulative POD energy per mode (both sets)
  pod_r2_vs_k.png    — cross-validated R² vs number of modes (both sets)
  Printed summary tables
"""

import argparse
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold

# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1",
                   help="Dataset version tag, e.g. v1, v2a, v2b")
    return p.parse_args()

_args = _parse_args()
PARQUET     = Path(f"features-{_args.version}/features_scalar.parquet")
FIGURES_DIR = Path(f"figures-{_args.version}")
FIGURES_DIR.mkdir(exist_ok=True)
print(f"Version: {_args.version}  |  parquet: {PARQUET}  |  figures: {FIGURES_DIR}")

# ── Feature set definitions ───────────────────────────────────────────────────

ENGINEERED_COLS = [
    "tip_deflection_slope",
    "tip_per_g_at_failure",
    "avg_strain_at_failure",
    "avg_strain_slope",
    "strain_energy_at_failure",
    "strain_energy_slope",
    "n_steps",
]

_NON_NODE = {
    "sim_id", "n_steps", "RF_failure", "g_limit",
    "tip_deflection_at_failure", "max_vm_stress_at_failure",
} | set(ENGINEERED_COLS)

TARGET   = "g_limit"
CV_FOLDS = 10
ENERGY_THRESHOLDS = [0.90, 0.95, 0.99]


# ── Helpers ───────────────────────────────────────────────────────────────────

def cumulative_energy(X: np.ndarray) -> np.ndarray:
    """Fraction of variance explained by k PCA modes (on standardized data)."""
    Xs = StandardScaler().fit_transform(X)
    _, s, _ = np.linalg.svd(Xs, full_matrices=False)
    lam = s ** 2
    return np.cumsum(lam) / lam.sum()


def modes_needed(cum_energy: np.ndarray, threshold: float) -> int:
    idx = np.searchsorted(cum_energy, threshold)
    return int(min(idx + 1, len(cum_energy)))


def pcr_cv_r2(X: np.ndarray, y: np.ndarray, max_k: int) -> np.ndarray:
    """Cross-validated R² for PCR with k = 1..max_k components."""
    kf  = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    r2s = np.full(max_k, np.nan)
    for k in range(1, max_k + 1):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=k)),
            ("reg",    LinearRegression()),
        ])
        scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        r2s[k - 1] = scores.mean()
    return r2s


def best_pcr(X: np.ndarray, y: np.ndarray, k: int):
    """Fit a full-data PCR with k components; return model + predicted y."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=k)),
        ("reg",    LinearRegression()),
    ])
    pipe.fit(X, y)
    return pipe, pipe.predict(X)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def print_pod_table(name: str, cum_energy: np.ndarray, cv_r2: np.ndarray) -> None:
    n = len(cum_energy)
    col_w = 5
    print(f"\n{'─'*60}")
    print(f"  {name}  ({n} features)")
    print(f"{'─'*60}")
    print(f"  {'Mode':>5}  {'Cumul. Energy':>14}  {'CV R²':>8}")
    print(f"  {'─'*5}  {'─'*14}  {'─'*8}")
    for k in range(n):
        marker = ""
        for thr in ENERGY_THRESHOLDS:
            if abs(cum_energy[k] - thr) < 0.01 or (k > 0 and cum_energy[k - 1] < thr <= cum_energy[k]):
                marker = f"  ← {thr:.0%} threshold"
                break
        r2_str = f"{cv_r2[k]:+.4f}" if k < len(cv_r2) and np.isfinite(cv_r2[k]) else "     —"
        print(f"  {k+1:>5}  {cum_energy[k]:>14.4f}  {r2_str:>8}{marker}")
    print(f"{'─'*60}")

    print(f"\n  Modes to reach energy thresholds:")
    for thr in ENERGY_THRESHOLDS:
        k = modes_needed(cum_energy, thr)
        r2 = cv_r2[k - 1] if k - 1 < len(cv_r2) else float("nan")
        print(f"    {thr:.0%} → {k:2d} mode(s)   CV R² = {r2:+.4f}")

    best_k = int(np.nanargmax(cv_r2)) + 1
    print(f"\n  Best CV R² = {cv_r2[best_k-1]:+.4f}  at k = {best_k}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pl.read_parquet(PARQUET).drop_nulls()
    y  = df[TARGET].to_numpy()

    # Auto-detect node/_failure columns present in this parquet
    strain_fail_cols = [c for c in df.columns if c.endswith("_failure") and c not in _NON_NODE]
    original_cols    = strain_fail_cols + ["tip_deflection_at_failure", "max_vm_stress_at_failure"]
    n_raw = len(strain_fail_cols)

    MEGA_COLS = original_cols + ENGINEERED_COLS

    sets = {
        f"ORIGINAL ({n_raw+2} raw)": np.array(df.select(original_cols).to_numpy(),  dtype=float),
        "ENGINEERED (7 feat)":       np.array(df.select(ENGINEERED_COLS).to_numpy(), dtype=float),
        f"MEGA ({n_raw+9} combined)": np.array(df.select(MEGA_COLS).to_numpy(),      dtype=float),
    }

    fig_var, axes_var = plt.subplots(1, 3, figsize=(18, 5))
    fig_r2,  axes_r2  = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    results = {}
    for i, (name, X) in enumerate(sets.items()):
        max_k = X.shape[1]

        # POD energy
        cum_e = cumulative_energy(X)

        # CV R²
        print(f"\nComputing CV R² for {name} ...")
        cv_r2 = pcr_cv_r2(X, y, max_k)

        print_pod_table(name, cum_e, cv_r2)

        best_k = int(np.nanargmax(cv_r2)) + 1
        _, y_pred = best_pcr(X, y, best_k)
        train_r2  = float(1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))

        print(f"\n  Best PCR fit (k={best_k}, training set):")
        print(f"    R²   = {train_r2:.4f}")
        print(f"    RMSE = {rmse(y, y_pred):.4f} g")

        results[name] = {"cum_e": cum_e, "cv_r2": cv_r2, "best_k": best_k}

        # ── Variance plot ──────────────────────────────────────────────────
        ax = axes_var[i]
        ks = np.arange(1, max_k + 1)
        ax.plot(ks, cum_e, "o-", color=colors[i], lw=2, ms=5)
        for thr in ENERGY_THRESHOLDS:
            k_thr = modes_needed(cum_e, thr)
            ax.axhline(thr, ls="--", lw=0.8, color="gray")
            ax.axvline(k_thr, ls=":", lw=0.8, color="gray")
            ax.text(k_thr + 0.1, thr - 0.02, f"{k_thr} modes\n({thr:.0%})",
                    fontsize=7, color="gray", va="top")
        ax.set_xlabel("Number of POD modes", fontsize=11)
        ax.set_ylabel("Cumulative energy (variance explained)", fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.set_xlim(1, max_k)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # ── R² vs k plot (axes linked after loop) ─────────────────────────
        ax2 = axes_r2[i]
        ax2.plot(ks, cv_r2, "s-", color=colors[i], lw=2, ms=5, label=f"CV R² ({CV_FOLDS}-fold)")
        ax2.axvline(best_k, ls="--", lw=1, color="k",
                    label=f"Best k = {best_k}  (R²={cv_r2[best_k-1]:.3f})")
        ax2.set_xlabel("Number of PCR modes k", fontsize=11)
        ax2.set_ylabel(f"{CV_FOLDS}-fold CV R²", fontsize=11)
        ax2.set_title(name, fontsize=12)
        ax2.set_xlim(1, max_k)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    # ── Align R² y-axes across all three panels ───────────────────────────
    all_r2_vals = np.concatenate([r["cv_r2"][np.isfinite(r["cv_r2"])] for r in results.values()])
    y_lo = max(0.0, float(np.nanmin(all_r2_vals)) - 0.02)
    y_hi = min(1.0, float(np.nanmax(all_r2_vals)) + 0.02)
    for ax2 in axes_r2:
        ax2.set_ylim(y_lo, y_hi)

    for fig, path in [(fig_var, FIGURES_DIR / "pod_variance.png"), (fig_r2, FIGURES_DIR / "pod_r2_vs_k.png")]:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        print(f"\nSaved → {path}")

    # ── Side-by-side best-k comparison ────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  COMPARISON SUMMARY")
    print("═" * 60)
    print(f"  {'Feature set':<24}  {'Best k':>6}  {'CV R²':>8}  {'90% E':>7}  {'95% E':>7}  {'99% E':>7}")
    print(f"  {'─'*24}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")
    for name, res in results.items():
        bk   = res["best_k"]
        br2  = res["cv_r2"][bk - 1]
        e90  = modes_needed(res["cum_e"], 0.90)
        e95  = modes_needed(res["cum_e"], 0.95)
        e99  = modes_needed(res["cum_e"], 0.99)
        print(f"  {name:<24}  {bk:>6}  {br2:>+8.4f}  {e90:>7}  {e95:>7}  {e99:>7}")
    print("═" * 60)


if __name__ == "__main__":
    main()