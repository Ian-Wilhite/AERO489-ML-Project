"""
AERO 489 — Model-specific diagnostic figures for GPR, PolyReg, and RandomForest.

Generates 8 figures in figures/v2/:
  GPR  : gpr_calibration.png, gpr_uncertainty_vs_glimit.png, gpr_length_scales.png
  Poly : poly_coefficients.png, poly_pdp.png
  RF   : rf_feature_importance.png, rf_pdp.png, rf_oob_curve.png

Usage:
    .venv/bin/python scripts/plot_model_diagnostics.py
    .venv/bin/python scripts/plot_model_diagnostics.py --models gpr poly rf
"""

import argparse
import json
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_root    = _scripts.parent
for _p in [str(_scripts), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats as sp_stats

from data_utils import load_scalar, BOXCOX_COLS

RESULTS = _root / "results"
FIGURES = _root / "figures/v2"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

# Short labels for Box-Cox features (used across all three model sections)
SHORT = {
    "boxcox_tip_deflection_slope":          "tip_slope",
    "boxcox_tip_per_g_at_failure":          "tip/g",
    "boxcox_avg_strain_at_failure":         "avg_strain",
    "boxcox_avg_strain_slope":              "strain_slope",
    "boxcox_strain_energy_at_failure":      "SE",
    "boxcox_strain_energy_slope":           "SE_slope",
    "boxcox_max_vm_stress_at_failure":      "VM_stress",
    "boxcox_k_spring":                      "k_spring",
    "boxcox_inv_tip_per_g_at_failure":      "1/tip/g",
    "boxcox_inv_tip_deflection_slope":      "1/tip_slope",
    "boxcox_inv_max_vm_stress_at_failure":  "1/VM",
    "boxcox_sqrt_strain_energy_at_failure": "√SE",
}

LONG = {
    "boxcox_tip_deflection_slope":          "Tip defl. slope",
    "boxcox_tip_per_g_at_failure":          "Tip defl. / g",
    "boxcox_avg_strain_at_failure":         "Avg strain",
    "boxcox_avg_strain_slope":              "Avg strain slope",
    "boxcox_strain_energy_at_failure":      "Strain energy",
    "boxcox_strain_energy_slope":           "Strain energy slope",
    "boxcox_max_vm_stress_at_failure":      "Max VM stress",
    "boxcox_k_spring":                      "k_spring",
    "boxcox_inv_tip_per_g_at_failure":      "1 / tip-per-g",
    "boxcox_inv_tip_deflection_slope":      "1 / tip defl. slope",
    "boxcox_inv_max_vm_stress_at_failure":  "1 / max VM stress",
    "boxcox_sqrt_strain_energy_at_failure": "√(strain energy)",
}

GROUP_COLOR = {
    "boxcox_tip_deflection_slope":          "#d7191c",
    "boxcox_tip_per_g_at_failure":          "#d7191c",
    "boxcox_avg_strain_at_failure":         "#fdae61",
    "boxcox_avg_strain_slope":              "#fdae61",
    "boxcox_strain_energy_at_failure":      "#1a9641",
    "boxcox_strain_energy_slope":           "#1a9641",
    "boxcox_max_vm_stress_at_failure":      "#756bb1",
    "boxcox_k_spring":                      "#d7191c",
    "boxcox_inv_tip_per_g_at_failure":      "#d7191c",
    "boxcox_inv_tip_deflection_slope":      "#d7191c",
    "boxcox_inv_max_vm_stress_at_failure":  "#756bb1",
    "boxcox_sqrt_strain_energy_at_failure": "#1a9641",
}


# ═══════════════════════════════════════════════════════════════════════════════
# GPR
# ═══════════════════════════════════════════════════════════════════════════════

def make_gpr_figures():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    print("\n── GPR figures ──────────────────────────────────────────────────")
    X_train, X_test, y_train, y_test = load_scalar(feature_cols=BOXCOX_COLS)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Isotropic RBF kernel — same as the trained model
    print("  Fitting isotropic GPR (calibration + uncertainty)...")
    k1 = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
    gpr1 = GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=5, normalize_y=True)
    gpr1.fit(X_tr_s, y_train)
    y_pred, y_std = gpr1.predict(X_te_s, return_std=True)

    _plot_gpr_calibration(y_test, y_pred, y_std)
    _plot_gpr_uncertainty_vs_glimit(y_test, y_pred, y_std)

    # Anisotropic Matérn — one length scale per feature
    print("  Fitting anisotropic Matérn GPR (length scales)...")
    n_feat = X_train.shape[1]
    k2 = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_feat), nu=2.5) + WhiteKernel(1e-3)
    gpr2 = GaussianProcessRegressor(kernel=k2, n_restarts_optimizer=5, normalize_y=True)
    gpr2.fit(X_tr_s, y_train)
    _plot_gpr_length_scales(gpr2, BOXCOX_COLS)


def _plot_gpr_calibration(y_true, y_pred, y_std):
    k = 2.0
    lo = y_pred - k * y_std
    hi = y_pred + k * y_std
    in_ci = (y_true >= lo) & (y_true <= hi)
    coverage = in_ci.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: predicted vs true with ±2σ error bars
    ax = axes[0]
    colors_pts = np.where(in_ci, "#2166ac", "#d73027")
    ax.errorbar(y_true, y_pred, yerr=k * y_std, fmt="none",
                ecolor="gray", alpha=0.35, lw=0.9, zorder=1)
    ax.scatter(y_true, y_pred, c=colors_pts, s=30, zorder=3, alpha=0.9)
    lims = [min(y_true.min(), y_pred.min()) - 0.15,
            max(y_true.max(), y_pred.max()) + 0.15]
    ax.plot(lims, lims, "k--", lw=1.0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True $g$-limit [g]")
    ax.set_ylabel("Predicted $g$-limit [g]")
    ax.set_title(f"GPR: Predicted ± 2σ vs. True\nEmpirical 95% CI coverage = {coverage:.1%}")
    ax.legend(handles=[
        Patch(fc="#2166ac", label=f"Inside CI ({in_ci.sum()}/{len(in_ci)})"),
        Patch(fc="#d73027", label=f"Outside CI ({(~in_ci).sum()}/{len(in_ci)})"),
    ], fontsize=8, loc="upper left")

    # Right: reliability diagram
    ax2 = axes[1]
    cl_vals = np.linspace(0.0, 0.999, 60)
    empirical = []
    for cl in cl_vals:
        z = sp_stats.norm.ppf(0.5 + cl / 2.0)
        cov = ((y_true >= y_pred - z * y_std) & (y_true <= y_pred + z * y_std)).mean()
        empirical.append(cov)
    empirical = np.array(empirical)

    ax2.plot(cl_vals, empirical, "o-", color="#2166ac", ms=3, lw=1.5, label="GPR")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect calibration")
    ax2.fill_between(cl_vals, cl_vals, empirical, alpha=0.15, color="#2166ac")
    ax2.set_xlabel("Nominal confidence level")
    ax2.set_ylabel("Empirical coverage")
    ax2.set_title("Calibration Reliability Diagram")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out = FIGURES / "gpr_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_gpr_uncertainty_vs_glimit(y_true, y_pred, y_std):
    abs_err = np.abs(y_pred - y_true)
    fig, ax = plt.subplots(figsize=(7, 5))

    sc = ax.scatter(y_true, y_std, c=abs_err, cmap="RdYlGn_r",
                    s=40, alpha=0.85, edgecolors="none", vmin=0)
    fig.colorbar(sc, ax=ax, label="|Prediction error| [g]")

    # Median σ per bin
    bins = np.linspace(y_true.min(), y_true.max(), 8)
    med, edges, _ = sp_stats.binned_statistic(y_true, y_std, statistic="median", bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, med, "k-", lw=2, label="Median σ per bin")

    ax.set_xlabel("True $g$-limit [g]")
    ax.set_ylabel("Posterior std deviation σ [g]")
    ax.set_title("GPR Predictive Uncertainty vs. True $g$-limit\n"
                 "(colour = |error|; lower σ ⟹ tighter prediction)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = FIGURES / "gpr_uncertainty_vs_glimit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_gpr_length_scales(gpr, feature_cols):
    # kernel_ structure: (ConstantKernel * Matern) + WhiteKernel
    matern = gpr.kernel_.k1.k2
    ls = np.array(matern.length_scale)

    labels = [LONG.get(f, f) for f in feature_cols]
    colors = [GROUP_COLOR.get(f, "#888888") for f in feature_cols]

    # Sort ascending: smaller length scale = more influential
    order = np.argsort(ls)
    ls_s  = ls[order]
    lbl_s = [labels[i] for i in order]
    col_s = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(ls_s)), ls_s, color=col_s, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(lbl_s)))
    ax.set_yticklabels(lbl_s, fontsize=9)
    ax.set_xlabel("Matérn kernel length scale (standardized units)\n"
                  "[smaller ⟹ shorter correlation length ⟹ feature more influential]")
    ax.set_title("GPR — Anisotropic Matérn Per-Feature Length Scales\n"
                 "(features standardized before GP fit)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[
        Patch(fc="#d7191c", label="Tip deflection family"),
        Patch(fc="#fdae61", label="Avg strain family"),
        Patch(fc="#1a9641", label="Strain energy family"),
        Patch(fc="#756bb1", label="VM stress family"),
    ], fontsize=9, loc="lower right")

    fig.tight_layout()
    out = FIGURES / "gpr_length_scales.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# POLYNOMIAL REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def _poly_term_type(raw_name: str) -> str:
    parts = raw_name.split(" ")
    if len(parts) == 1:
        return "linear"
    return "quadratic" if len(set(parts)) == 1 else "interaction"


def _shorten_poly_name(raw_name: str) -> str:
    parts = raw_name.split(" ")
    return " × ".join(SHORT.get(p, p) for p in parts)


TERM_COLORS = {"linear": "#2166ac", "quadratic": "#d73027", "interaction": "#66c2a5"}


def make_poly_figures():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.inspection import partial_dependence

    print("\n── Poly figures ─────────────────────────────────────────────────")
    X_train, X_test, y_train, y_test = load_scalar(feature_cols=BOXCOX_COLS)

    print("  Fitting PolyReg degree=2...")
    pipeline = Pipeline([
        ("scaler1", StandardScaler()),
        ("poly",    PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler2", StandardScaler()),
        ("ridge",   Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    raw_names = pipeline.named_steps["poly"].get_feature_names_out(BOXCOX_COLS)
    coefs     = pipeline.named_steps["ridge"].coef_

    _plot_poly_coefficients(raw_names, coefs)
    _plot_poly_pdp(pipeline, X_train, coefs, raw_names)


def _plot_poly_coefficients(raw_names, coefs, top_n=25):
    short  = [_shorten_poly_name(n) for n in raw_names]
    types  = [_poly_term_type(n)    for n in raw_names]
    colors = [TERM_COLORS[t]        for t in types]

    order    = np.argsort(np.abs(coefs))[::-1][:top_n]
    top_c    = coefs[order]
    top_s    = [short[i]  for i in order]
    top_col  = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(range(top_n), top_c, color=top_col, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)

    for i, (_, v) in enumerate(zip(bars, top_c)):
        pad = 0.02 if v >= 0 else -0.02
        ax.text(v + pad, i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_s, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized coefficient  [g per 1-std increase in polynomial feature]")
    ax.set_title(f"Polynomial Regression — Top-{top_n} Standardized Ridge Coefficients\n"
                 "(degree=2 on Box-Cox features; coefficients on double-scaled inputs)")
    ax.legend(handles=[Patch(fc=c, label=t.capitalize()) for t, c in TERM_COLORS.items()],
              fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    out = FIGURES / "poly_coefficients.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_poly_pdp(pipeline, X_train, coefs, raw_names):
    from sklearn.inspection import partial_dependence

    n_base = len(BOXCOX_COLS)
    # Find top-4 original features by their linear-term coefficient magnitude
    linear_coefs = np.abs(coefs[:n_base])
    top4_idx = np.argsort(linear_coefs)[::-1][:4]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feat_idx in zip(axes.flat, top4_idx):
        feat_name = BOXCOX_COLS[feat_idx]
        pdp = partial_dependence(pipeline, X_train, features=[feat_idx], kind="average")
        grid = pdp["grid_values"][0]
        avg  = pdp["average"][0]
        ax.plot(grid, avg, color="#2166ac", lw=2)
        ax.fill_between(grid, avg.min(), avg, alpha=0.12, color="#2166ac")
        ax.set_xlabel(LONG.get(feat_name, feat_name), fontsize=9)
        ax.set_ylabel("Partial dependence [g]", fontsize=9)
        ax.set_title(f"PDP: {LONG.get(feat_name, feat_name)}", fontsize=10)
        ax.grid(alpha=0.2)

    fig.suptitle("Polynomial Regression — Partial Dependence Plots\n"
                 "(top-4 features by linear-term coefficient magnitude)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "poly_pdp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_ENGINEERED_SET = {
    "tip_deflection_slope", "tip_per_g_at_failure", "avg_strain_at_failure",
    "avg_strain_slope", "strain_energy_at_failure", "strain_energy_slope", "k_spring",
    "inv_tip_per_g_at_failure", "inv_tip_deflection_slope",
    "inv_max_vm_stress_at_failure", "sqrt_strain_energy_at_failure",
}

_RANKED_SET = {"ranked_strain_p04", "ranked_strain_p23", "ranked_strain_p24",
               "gompertz_log_b", "gompertz_c"}


def _classify_rf_feature(fname: str) -> tuple[str, str]:
    if fname in _BASE_ENGINEERED_SET:
        return "Engineered", "#2166ac"
    if any(fname.startswith(p) for p in ("ln_", "log10_", "exp_", "pow10_", "boxcox_")):
        return "Transformed", "#4daf4a"
    if fname in _RANKED_SET:
        return "Ranked strain", "#984ea3"
    return "Raw gauge", "#ff7f00"


def _shorten_rf_name(fname: str, gauge_map: dict) -> str:
    if fname in gauge_map:
        return gauge_map[fname]
    return fname.replace("_at_failure", "").replace("_failure", "").replace("_", " ")


def make_rf_figures():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.inspection import partial_dependence

    print("\n── RF figures ───────────────────────────────────────────────────")
    rf_json         = json.load(open(RESULTS / "random_forest.json"))
    rf_feature_cols = rf_json["feature_set"]

    # Build short "Gauge N" labels for raw node columns
    gauge_n = 1
    gauge_map = {}
    for f in rf_feature_cols:
        if f not in _BASE_ENGINEERED_SET and f not in _RANKED_SET and not any(
            f.startswith(p) for p in ("ln_", "log10_", "exp_", "pow10_", "boxcox_")
        ):
            gauge_map[f] = f"Gauge {gauge_n}"
            gauge_n += 1

    X_train, X_test, y_train, y_test = load_scalar(feature_cols=rf_feature_cols)

    PARAM_GRID = {
        "n_estimators":     [100, 300, 500],
        "max_depth":        [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    }
    print("  Running GridSearchCV (27 configs × 5-fold)...")
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        PARAM_GRID, cv=5, scoring="r2", n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    best = gs.best_params_
    print(f"  Best params: {best}")

    rf = RandomForestRegressor(**best, random_state=42)
    rf.fit(X_train, y_train)
    fi = rf.feature_importances_

    _plot_rf_feature_importance(fi, rf_feature_cols, gauge_map)
    _plot_rf_pdp(rf, X_train, rf_feature_cols, fi, gauge_map)
    _plot_rf_oob_curve(X_train, y_train, best)


def _plot_rf_feature_importance(fi, feature_cols, gauge_map, top_n=20):
    order     = np.argsort(fi)[::-1][:top_n]
    top_fi    = fi[order]
    top_names = [feature_cols[i] for i in order]

    groups, colors = zip(*[_classify_rf_feature(n) for n in top_names])
    short_names    = [_shorten_rf_name(n, gauge_map) for n in top_names]

    fig, ax = plt.subplots(figsize=(10, 7))
    # Plot in reverse order so rank-1 appears at top
    ax.barh(range(top_n), top_fi[::-1], color=list(reversed(colors)),
            alpha=0.85, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(list(reversed(short_names)), fontsize=8.5)
    ax.set_xlabel("Mean Decrease in Impurity (Gini importance)")
    ax.set_title(f"Random Forest — Top-{top_n} Feature Importances")
    ax.grid(axis="x", alpha=0.25)

    seen = {}
    for g, c in zip(groups, colors):
        seen[g] = c
    ax.legend(handles=[Patch(fc=c, label=g) for g, c in seen.items()],
              fontsize=9, loc="lower right")

    fig.tight_layout()
    out = FIGURES / "rf_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_rf_pdp(rf, X_train, feature_cols, fi, gauge_map, top_n=4):
    from sklearn.inspection import partial_dependence

    top_idx = np.argsort(fi)[::-1][:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feat_idx in zip(axes.flat, top_idx):
        feat_name = feature_cols[feat_idx]
        pdp  = partial_dependence(rf, X_train, features=[feat_idx], kind="average")
        grid = pdp["grid_values"][0]
        avg  = pdp["average"][0]
        short = _shorten_rf_name(feat_name, gauge_map)
        ax.plot(grid, avg, color="#d73027", lw=2)
        ax.fill_between(grid, avg.min(), avg, alpha=0.12, color="#d73027")
        ax.set_xlabel(short, fontsize=9)
        ax.set_ylabel("Partial dependence [g]", fontsize=9)
        ax.set_title(f"PDP: {short}", fontsize=10)
        ax.grid(alpha=0.2)

    fig.suptitle("Random Forest — Partial Dependence Plots\n"
                 "(top-4 features by Gini importance)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "rf_pdp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_rf_oob_curve(X_train, y_train, best_params):
    from sklearn.ensemble import RandomForestRegressor

    max_depth        = best_params.get("max_depth", None)
    min_samples_leaf = best_params.get("min_samples_leaf", 1)

    print("  Computing OOB curve (warm_start, 10–500 trees)...")
    rf_oob = RandomForestRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
        warm_start=True,
        random_state=42,
    )
    n_values   = list(range(10, 501, 10))
    oob_scores = []
    for n in n_values:
        rf_oob.n_estimators = n
        rf_oob.fit(X_train, y_train)
        oob_scores.append(rf_oob.oob_score_)
        if n % 100 == 0:
            print(f"    n={n:4d}  OOB R²={rf_oob.oob_score_:.4f}")

    best_n   = n_values[int(np.argmax(oob_scores))]
    best_oob = max(oob_scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, oob_scores, color="#d73027", lw=2)
    ax.axvline(best_n, color="gray", lw=1, ls="--")
    ax.axhline(best_oob, color="gray", lw=1, ls="--",
               label=f"Peak OOB R² = {best_oob:.4f} at n={best_n}")
    ax.set_xlabel("Number of trees (n_estimators)")
    ax.set_ylabel("OOB $R^2$")
    ax.set_title(f"Random Forest — OOB Score vs. Number of Trees\n"
                 f"(max_depth={max_depth}, min_samples_leaf={min_samples_leaf})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=max(0, min(oob_scores) - 0.02))

    fig.tight_layout()
    out = FIGURES / "rf_oob_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", choices=["gpr", "poly", "rf"],
                        help="Subset of model groups to run (default: all)")
    args   = parser.parse_args()
    to_run = set(args.models) if args.models else {"gpr", "poly", "rf"}

    if "gpr"  in to_run: make_gpr_figures()
    if "poly" in to_run: make_poly_figures()
    if "rf"   in to_run: make_rf_figures()

    print("\nAll done. Figures saved to figures/v2/")


if __name__ == "__main__":
    main()
