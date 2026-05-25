"""
AERO 489 — PINN physics-model × lambda ablation study.

Trains one PINN per (physics_model, lambda) combination and records test metrics.

3 physics models × 10 lambda values = 30 runs.
Results saved to:
  results/pinn_ablation.csv
  figures/v2/pinn_ablation.png
"""

import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_root    = _scripts.parent
for p in [str(_scripts), str(_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import load_scalar, PINN_COLS
from evaluate import adjusted_r2, rmse, mae, mos_01

PHYSICS_MODELS = {
    "hooke_strain": "Hooke (avg strain)",
    "energy_quad":  "Strain energy (RF² ∝ U)",
    "energy_rate":  "Energy rate (RF ∝ dU/dF)",
}

LAMBDAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

RESULTS_DIR = _root / "results"
FIGURES_DIR = _root / "figures/v2"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def run_ablation() -> pd.DataFrame:
    from models.pinn import PINN

    X_train, X_test, y_train, y_test = load_scalar(feature_cols=PINN_COLS)
    n_feat = len(PINN_COLS)

    records = []
    total   = len(PHYSICS_MODELS) * len(LAMBDAS)
    run_idx = 0

    for phys_model, phys_label in PHYSICS_MODELS.items():
        for lam in LAMBDAS:
            run_idx += 1
            print(f"\n[{run_idx:2d}/{total}]  {phys_label}  λ={lam}")

            model = PINN(
                physics_model=phys_model,
                lambda_physics=lam,
                epochs=500,
                patience=40,
                seed=42,
            )
            t0 = time.time()
            model.fit(X_train, y_train, feature_cols=PINN_COLS)
            elapsed = time.time() - t0

            y_pred  = model.predict(X_test)
            adj_r2  = adjusted_r2(y_test, y_pred, n_feat)
            rmse_v  = rmse(y_test, y_pred)
            mae_v   = mae(y_test, y_pred)
            mos_v   = mos_01(y_test, y_pred)
            n_ep    = len(model.train_loss_history_)

            print(f"    epochs={n_ep}  adj_R²={adj_r2:.4f}  RMSE={rmse_v:.4f}  MOS={mos_v:.4f}  ({elapsed:.0f}s)")

            records.append({
                "physics_model": phys_model,
                "physics_label": phys_label,
                "lambda":        lam,
                "adj_r2":        adj_r2,
                "rmse":          rmse_v,
                "mae":           mae_v,
                "mos_01":        mos_v,
                "n_epochs":      n_ep,
                "train_time_s":  elapsed,
            })

    return pd.DataFrame(records)


def plot_ablation(df: pd.DataFrame) -> None:
    colors  = ["#1f77b4", "#d62728", "#2ca02c"]
    metrics = [
        ("adj_r2", "Adjusted R²",  True),
        ("rmse",   "RMSE (g)",     False),
        ("mos_01", "MOS @ 1% (g)", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (col, ylabel, higher_better) in zip(axes, metrics):
        for color, (phys, label) in zip(colors, PHYSICS_MODELS.items()):
            sub = df[df["physics_model"] == phys].sort_values("lambda")
            ax.plot(sub["lambda"], sub[col], "o-", color=color, label=label, lw=2, ms=6)

        ax.set_xscale("log")
        ax.set_xlabel("λ (physics weight)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.grid(alpha=0.3)

        # Reference line: plain NN (no physics, loaded from results if available)
        nn_json = _root / "results" / "feedforward_nn.json"
        if nn_json.exists():
            nn_val = json.load(open(nn_json))["metrics"][col if col != "adj_r2" else "adj_r2"]
            ax.axhline(nn_val, ls="--", lw=1.2, color="gray", alpha=0.8, label="Plain NN")

        arrow = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{ylabel}  ({arrow})", fontsize=11)

    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("PINN — Physics model × λ ablation  (test set)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "pinn_ablation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure → {out}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "═" * 72)
    print("  ABLATION SUMMARY  (sorted by adj_R²)")
    print("═" * 72)
    top = df.sort_values("adj_r2", ascending=False).head(10)
    print(f"  {'Physics model':<28}  {'λ':>6}  {'adj_R²':>8}  {'RMSE':>7}  {'MOS@1%':>7}")
    print(f"  {'─'*28}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}")
    for _, r in top.iterrows():
        print(f"  {r['physics_label']:<28}  {r['lambda']:>6.3f}  {r['adj_r2']:>8.4f}  {r['rmse']:>7.4f}  {r['mos_01']:>7.4f}")
    print("═" * 72)

    for phys, label in PHYSICS_MODELS.items():
        sub = df[df["physics_model"] == phys]
        best = sub.loc[sub["adj_r2"].idxmax()]
        print(f"  Best λ for {label}: λ={best['lambda']}  adj_R²={best['adj_r2']:.4f}")


if __name__ == "__main__":
    print(f"Running {len(PHYSICS_MODELS)} physics models × {len(LAMBDAS)} λ values = {len(PHYSICS_MODELS)*len(LAMBDAS)} experiments\n")

    df = run_ablation()
    df.to_csv(RESULTS_DIR / "pinn_ablation.csv", index=False)
    print(f"\nCSV → {RESULTS_DIR / 'pinn_ablation.csv'}")

    plot_ablation(df)
    print_summary(df)
