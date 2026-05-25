"""
AERO 489 — Full model comparison plots (all 7 models + PINN ablation).

Produces:
  Fig 01  pred_vs_true.png          — 2×4 predicted vs true scatter (all models)
  Fig 02  residual_dist.png         — overlaid residual KDEs (all models)
  Fig 03  abs_error_cdf.png         — P(|error| ≤ x) CDF (all models)
  Fig 04  overpredict_cdf.png       — P(overpredict > x) tail risk (all models)
  Fig 05  mos_sensitivity.png       — required MOS vs confidence level (all models)
  Fig 06  error_vs_glimit.png       — residual vs true g-limit, 2×4 grid
  Fig 07  safety_bars.png           — max_overpredict + MOS@1% bar chart
  Fig 08  pareto_r2_vs_time.png     — R² vs inference time (all models, real data)
  Fig 09  pareto_r2_vs_interp.png   — R² vs interpretability (all models)
  Fig 10  nn_vs_pinn_per_sample.png — per-sample NN vs PINN absolute error
  Fig 11  ablation_heatmap.png      — PINN λ × physics model heatmaps (adj_r2, RMSE, MOS)
  Fig 12  ablation_lambda_curves.png— λ sweep line plots with plain-NN reference

Usage
-----
    cd /home/ianw/Github/Courses/AERO489-Proj
    .venv/bin/python scripts/comparison_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures/v2"
FIGURES_DIR.mkdir(exist_ok=True)

# ── model metadata ────────────────────────────────────────────────────────────
# (json_name, display_label, color, interp_score, group)
#   interp_score: 1=none → 5=fully explicit  (for Pareto interp axis)
MODEL_META = [
    ("linear_regression",        "Linear Reg.",   "#7f7f7f", 5.3, "classical"),
    ("polynomial_regression",    "Poly Reg.",     "#bcbd22", 4.7, "classical"),
    ("gaussian_process_regression", "GPR",        "#9467bd", 4.0, "classical"),
    ("random_forest",            "Random Forest", "#8c564b", 2.0, "classical"),
    ("feedforward_nn",           "Feedforward NN","#1f77b4", 2.5, "modern"),
    ("pinn",                     "PINN",          "#2ca02c", 3.0, "modern"),
    ("deep_learning_lstm",       "LSTM",          "#d62728", 1.0, "modern"),
]
INTERP_TICKS = {1.0: "None", 2.0: "Partial", 2.5: "Partial", 3.0: "Physics-\ninformed",
                4.0: "Probabil-\nistic", 4.7: "Explicit", 5.3: "Explicit"}
INTERP_AXIS  = {1.0: "None", 2.0: "Partial", 3.0: "Physics-\ninformed",
                4.0: "Probabilistic", 5.0: "Fully\nexplicit"}

THRESHOLD_MOS  = 0.25
THRESHOLD_RMSE = 0.75
THRESHOLD_R2   = 0.80

plt.rcParams.update({
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── load results ──────────────────────────────────────────────────────────────
def _load(name: str, label: str, color: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        print(f"  [WARN] {path} not found — skipping")
        return None
    with open(path) as f:
        d = json.load(f)
    d["label"]  = label
    d["color"]  = color
    d["y_pred"] = np.array(d["y_pred_test"])
    d["y_true"] = np.array(d["y_true_test"])
    d["resid"]  = d["y_pred"] - d["y_true"]
    return d

MODELS = [r for r in [
    _load(name, label, color) for name, label, color, *_ in MODEL_META
] if r is not None]

# Ordered for legend consistency
CLASSICAL = [r for r in MODELS if r["model"] in
             {"linear_regression","polynomial_regression","gaussian_process_regression","random_forest"}]
MODERN    = [r for r in MODELS if r["model"] in
             {"feedforward_nn","pinn","deep_learning_lstm"}]

if not MODELS:
    sys.exit("No result JSONs found.")

print(f"Loaded {len(MODELS)} models: {[r['label'] for r in MODELS]}\n")


# ── helpers ───────────────────────────────────────────────────────────────────
def _threshold_line(ax, val, label, color="darkorange", ls="--", axis="h"):
    fn = ax.axhline if axis == "h" else ax.axvline
    fn(val, color=color, lw=1.3, ls=ls, label=label, zorder=1)


def _save(fig, name: str):
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Fig 01 — Predicted vs True (2×4 grid) ────────────────────────────────────
def fig_pred_vs_true():
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = axes.flatten()

    all_vals = np.concatenate([r["y_true"] for r in MODELS] + [r["y_pred"] for r in MODELS])
    lims = (all_vals.min() - 0.1, all_vals.max() + 0.1)
    diag = np.array(lims)

    for i, r in enumerate(MODELS):
        ax = axes[i]
        over = r["resid"] > 0
        ax.scatter(r["y_true"][~over], r["y_pred"][~over],
                   c="#4393c3", s=22, alpha=0.65, linewidths=0, label="Under-pred.")
        ax.scatter(r["y_true"][over],  r["y_pred"][over],
                   c="#d6604d", s=22, alpha=0.65, linewidths=0, label="Over-pred.")
        ax.plot(diag, diag, "k--", lw=1.1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect("equal")
        m = r["metrics"]
        ax.set_title(f"{r['label']}\nadj-R²={m['adj_r2']:.3f}  RMSE={m['rmse']:.3f} g", fontsize=10)
        ax.set_xlabel("True g-limit [g]", fontsize=9)
        ax.set_ylabel("Predicted g-limit [g]", fontsize=9)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, markerscale=1.2)

    # Hide unused panels
    for j in range(len(MODELS), nrows * ncols):
        axes[j].set_visible(False)

    fig.suptitle("Predicted vs True g-limit — All Models", fontweight="bold", fontsize=14)
    fig.tight_layout()
    _save(fig, "pred_vs_true.png")


# ── Fig 02 — Residual KDE (all models) ────────────────────────────────────────
def fig_residual_dist():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, group, title in zip(axes,
                                [CLASSICAL, MODERN],
                                ["Classical Models", "Modern Models"]):
        for r in group:
            sns.kdeplot(r["resid"], ax=ax, color=r["color"], lw=2,
                        fill=True, alpha=0.10, label=r["label"], bw_adjust=0.6)
        ax.axvline(0,              color="black",      lw=1.0, ls="--")
        ax.axvline( THRESHOLD_MOS, color="darkorange", lw=1.2, ls=":",
                    label=f"±{THRESHOLD_MOS}g MOS threshold")
        ax.axvline(-THRESHOLD_MOS, color="darkorange", lw=1.2, ls=":")
        ax.set_xlabel("Residual: predicted − true g-limit [g]")
        ax.set_ylabel("Density")
        ax.set_title(f"Residual Distribution — {title}")
        ax.legend(fontsize=9)

    fig.suptitle("Over-predict (right of zero) is the safety-critical direction",
                 fontsize=11, style="italic")
    fig.tight_layout()
    _save(fig, "residual_dist.png")


# ── Fig 03 — Absolute error CDF ───────────────────────────────────────────────
def fig_abs_error_cdf():
    fig, ax = plt.subplots(figsize=(8, 5))

    for r in MODELS:
        xs = np.sort(np.abs(r["resid"]))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ls = "-" if r["model"] in {"feedforward_nn","pinn","deep_learning_lstm"} else "--"
        ax.plot(xs, ys, color=r["color"], lw=2, ls=ls, label=r["label"])

    for thr in [0.25, 0.50, 0.75]:
        ax.axvline(thr, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.text(thr + 0.01, 0.02, f"{thr}g", fontsize=8, color="grey", va="bottom")

    ax.set_xlabel("|Error| [g]")
    ax.set_ylabel("Cumulative probability  P(|error| ≤ x)")
    ax.set_title("Absolute Error CDF — Fraction of predictions within tolerance\n"
                 "(solid = modern, dashed = classical)")
    ax.set_xlim(0); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "abs_error_cdf.png")


# ── Fig 04 — Overprediction tail CDF ─────────────────────────────────────────
def fig_overpredict_cdf():
    fig, ax = plt.subplots(figsize=(8, 5))

    for r in MODELS:
        over  = r["resid"][r["resid"] > 0]
        if len(over) == 0:
            continue
        n_tot = len(r["resid"])
        xs = np.sort(over)[::-1]
        ys = np.arange(1, len(xs) + 1) / n_tot
        ls = "-" if r["model"] in {"feedforward_nn","pinn","deep_learning_lstm"} else "--"
        ax.plot(np.concatenate([[0], xs]),
                np.concatenate([[len(over) / n_tot], ys]),
                color=r["color"], lw=2, ls=ls, label=r["label"])

    _threshold_line(ax, THRESHOLD_MOS, f"{THRESHOLD_MOS}g MOS threshold", "darkorange", "--", "v")
    ax.axhline(0.01, color="grey", lw=0.8, ls=":", alpha=0.7, label="1% exceedance")

    ax.set_xlabel("Overprediction magnitude [g]")
    ax.set_ylabel("P(overpredict > x)")
    ax.set_title("Overprediction Tail Risk — Right tail of safety-critical errors\n"
                 "(solid = modern, dashed = classical)")
    ax.set_xlim(0); ax.set_ylim(0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "overpredict_cdf.png")


# ── Fig 05 — MOS sensitivity curve ────────────────────────────────────────────
def fig_mos_sensitivity():
    fig, ax = plt.subplots(figsize=(8, 5))
    cls = np.linspace(0.80, 0.999, 300)

    for r in MODELS:
        mos_vals = [max(0.0, np.percentile(r["resid"], cl * 100)) for cl in cls]
        ls = "-" if r["model"] in {"feedforward_nn","pinn","deep_learning_lstm"} else "--"
        ax.plot(cls * 100, mos_vals, color=r["color"], lw=2, ls=ls, label=r["label"])

    _threshold_line(ax, THRESHOLD_MOS, f"{THRESHOLD_MOS}g target (proposal §6.2)", "darkorange")
    ax.axvline(99, color="grey", lw=0.8, ls=":", alpha=0.6, label="99% confidence")

    ax.set_xlabel("Confidence level [%]")
    ax.set_ylabel("Required MOS [g]")
    ax.set_title("Required Margin of Safety vs Confidence Level\n"
                 "(solid = modern, dashed = classical)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "mos_sensitivity.png")


# ── Fig 06 — Error vs true g-limit (2×4 grid) ────────────────────────────────
def fig_error_vs_glimit():
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = axes.flatten()

    for i, r in enumerate(MODELS):
        ax = axes[i]
        ax.scatter(r["y_true"], r["resid"],
                   color=r["color"], s=18, alpha=0.55, linewidths=0)
        ax.axhline(0, color="black", lw=1.0, ls="--")
        ax.axhline( THRESHOLD_MOS, color="darkorange", lw=0.9, ls=":", alpha=0.8)
        ax.axhline(-THRESHOLD_MOS, color="darkorange", lw=0.9, ls=":", alpha=0.8)
        ax.set_xlabel("True g-limit [g]", fontsize=9)
        ax.set_title(r["label"], fontsize=10)
        if i % ncols == 0:
            ax.set_ylabel("Residual [g]", fontsize=9)

    for j in range(len(MODELS), nrows * ncols):
        axes[j].set_visible(False)

    fig.suptitle("Residual vs True g-limit — checks for systematic bias by damage severity\n"
                 "Orange lines: ±0.25g", fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "error_vs_glimit.png")


# ── Fig 07 — Safety bar chart (all models) ───────────────────────────────────
def fig_safety_bars():
    names  = [r["label"]                    for r in MODELS]
    max_op = [r["metrics"]["max_overpredict"] for r in MODELS]
    mos    = [r["metrics"]["mos_01"]          for r in MODELS]
    colors = [r["color"]                    for r in MODELS]

    x     = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))

    bars1 = ax.bar(x - width/2, max_op, width, label="Max over-predict [g]",
                   color=colors, alpha=0.90, edgecolor="white")
    bars2 = ax.bar(x + width/2, mos,    width, label="MOS@1% [g]",
                   color=colors, alpha=0.50, edgecolor="white", hatch="//")

    _threshold_line(ax, THRESHOLD_MOS, f"MOS target {THRESHOLD_MOS}g (§6.2)", "darkorange")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("g [g]")
    ax.set_title("Worst-case Safety Metrics — Lower is safer")
    ax.legend()
    fig.tight_layout()
    _save(fig, "safety_bars.png")


# ── Fig 08 — Pareto: R² vs inference time ─────────────────────────────────────
def fig_pareto_time(subset=None, suffix=""):
    """subset: list of model key strings to include; None = all."""
    models = [r for r in MODELS if subset is None or r["model"] in subset]
    fig, ax = plt.subplots(figsize=(8, 5.5))

    times = [r["metrics"]["inference_time_ms"] for r in models]
    r2s   = [r["metrics"]["adj_r2"]            for r in models]
    x_max = max(times) * 6
    y_min = min(r2s) - 0.08

    from matplotlib.patches import Rectangle
    for r in models:
        m = r["metrics"]
        x0 = m["inference_time_ms"]
        rect = Rectangle((x0, y_min), x_max - x0, m["adj_r2"] - y_min,
                          linewidth=0, facecolor=r["color"], alpha=0.08, zorder=1)
        ax.add_patch(rect)
        ax.plot([x0, x0, x_max], [m["adj_r2"], y_min, y_min],
                color=r["color"], linewidth=0.7, linestyle="--", alpha=0.35, zorder=2)

    for r in models:
        m   = r["metrics"]
        mk  = "o" if r["model"] in {"feedforward_nn","pinn","deep_learning_lstm"} else "s"
        ax.scatter(m["inference_time_ms"], m["adj_r2"],
                   color=r["color"], s=130, zorder=5, marker=mk,
                   edgecolors="white", linewidths=0.8)
        offsets = {
            "Linear Reg.":   (6, -10),
            "Poly Reg.":     (6,   4),
            "GPR":           (-55,  4),
            "Random Forest": (6,   4),
            "Feedforward NN":(6,  -10),
            "PINN":          (6,   4),
            "LSTM":          (6,   4),
        }
        dx, dy = offsets.get(r["label"], (6, 4))
        ax.annotate(r["label"], (m["inference_time_ms"], m["adj_r2"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, color=r["color"])

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker="s", color="w", markerfacecolor="grey", ms=9, label="Classical"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="grey", ms=9, label="Modern"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="lower right")

    ax.set_xscale("log")
    ax.set_xlabel("Inference time [ms]  (log scale, lower = faster)")
    ax.set_ylabel("Adjusted R²  (higher = better)")
    ax.set_title("Pareto Front — Accuracy vs Inference Speed")
    fig.tight_layout()
    _save(fig, f"pareto_r2_vs_time{suffix}.png")


# ── Fig 09 — Pareto: R² vs interpretability ───────────────────────────────────
def fig_pareto_interp(subset=None, suffix=""):
    """subset: list of model key strings to include; None = all."""
    meta_map = {name: (interp, group) for name, _, _, interp, group in MODEL_META}
    models   = [r for r in MODELS if subset is None or r["model"] in subset]
    fig, ax  = plt.subplots(figsize=(9, 5.5))

    r2s   = [r["metrics"]["adj_r2"] for r in models]
    x_min = 0.5
    y_min = min(r2s) - 0.08

    from matplotlib.patches import Rectangle
    for r in models:
        interp, group = meta_map[r["model"]]
        rect = Rectangle((x_min, y_min), interp - x_min, r["metrics"]["adj_r2"] - y_min,
                          linewidth=0, facecolor=r["color"], alpha=0.08, zorder=1)
        ax.add_patch(rect)
        ax.plot([x_min, interp, interp], [r["metrics"]["adj_r2"], r["metrics"]["adj_r2"], y_min],
                color=r["color"], linewidth=0.7, linestyle="--", alpha=0.35, zorder=2)

    for r in models:
        interp, group = meta_map[r["model"]]
        mk = "o" if group == "modern" else "s"
        ax.scatter(interp, r["metrics"]["adj_r2"],
                   color=r["color"], s=130, zorder=5, marker=mk,
                   edgecolors="white", linewidths=0.8)
        offsets = {
            "Linear Reg.":   ( 0,  8),
            "Poly Reg.":     ( 0, -14),
            "GPR":           ( 0,  8),
            "Random Forest": ( 0,  8),
            "Feedforward NN":(-20,-14),
            "PINN":          ( 0,  8),
            "LSTM":          ( 0,  8),
        }
        dx, dy = offsets.get(r["label"], (0, 8))
        ax.annotate(r["label"], (interp, r["metrics"]["adj_r2"]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha="center", fontsize=9, color=r["color"])

    xtick_positions = [1.0, 2.0, 3.0, 4.0, 5.0]
    xtick_labels    = ["None", "Partial", "Physics-\ninformed", "Probabilistic", "Fully\nexplicit"]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(0.5, 5.8)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker="s", color="w", markerfacecolor="grey", ms=9, label="Classical"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="grey", ms=9, label="Modern"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="lower right")

    ax.set_xlabel("Physical interpretability  (right = more interpretable)")
    ax.set_ylabel("Adjusted R²")
    ax.set_title("Pareto Front — Accuracy vs Interpretability")
    fig.tight_layout()
    _save(fig, f"pareto_r2_vs_interp{suffix}.png")


# ── Fig 10 — Per-sample NN vs PINN comparison ────────────────────────────────
def fig_nn_vs_pinn():
    nn   = next((r for r in MODELS if r["model"] == "feedforward_nn"), None)
    pinn = next((r for r in MODELS if r["model"] == "pinn"),           None)
    if nn is None or pinn is None:
        print("  [SKIP] nn_vs_pinn — need both results")
        return

    nn_err   = np.abs(nn["resid"])
    pinn_err = np.abs(pinn["resid"])
    diff     = pinn_err - nn_err

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    lim = max(nn_err.max(), pinn_err.max()) * 1.05
    sc  = ax.scatter(nn_err, pinn_err, c=diff, cmap="RdBu_r",
                     vmin=-0.03, vmax=0.03, s=30, alpha=0.75, linewidths=0)
    plt.colorbar(sc, ax=ax, label="PINN err − NN err [g]")
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="Equal performance")
    ax.set_xlabel("Feedforward NN |error| [g]")
    ax.set_ylabel("PINN |error| [g]")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_title("Per-sample error (above diagonal = PINN worse)")
    ax.legend(fontsize=9)

    ax = axes[1]
    sns.histplot(diff, bins=40, ax=ax, color="#2ca02c", alpha=0.7, edgecolor="white", kde=True)
    ax.axvline(0,          color="black", lw=1.0, ls="--")
    ax.axvline(diff.mean(), color="red",  lw=1.5, label=f"Mean: {diff.mean():.4f} g")
    nn_wins = int(np.sum(diff > 0))
    ax.set_xlabel("PINN |error| − NN |error| [g]\n(positive = PINN worse on this sample)")
    ax.set_ylabel("Count")
    ax.set_title(f"NN wins on {nn_wins}/{len(diff)} samples ({nn_wins/len(diff)*100:.0f}%)")
    ax.legend(fontsize=9)

    fig.suptitle("Feedforward NN vs PINN — Per-sample absolute error", fontweight="bold")
    fig.tight_layout()
    _save(fig, "nn_vs_pinn_per_sample.png")


# ── Fig 11 — PINN ablation heatmaps ──────────────────────────────────────────
def fig_ablation_heatmap():
    csv_path = RESULTS_DIR / "pinn_ablation.csv"
    if not csv_path.exists():
        print("  [SKIP] ablation_heatmap — run pinn_ablation.py first")
        return

    df = pd.read_csv(csv_path)
    phys_order   = ["hooke_strain", "energy_quad", "energy_rate"]
    phys_labels  = ["Hooke\n(avg strain)", "Energy quad\n(RF^2 ~ U)", "Energy rate\n(RF ~ dU/dF)"]
    lambda_vals  = sorted(df["lambda"].unique())

    metrics = [
        ("adj_r2", "Adjusted R²",  True,  "Blues",   0.86, 0.99),
        ("rmse",   "RMSE [g]",     False, "Reds_r",  0.11, 0.65),
        ("mos_01", "MOS@1% [g]",   False, "Reds_r",  0.30, 1.20),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (col, ylabel, higher_better, cmap, vmin, vmax) in zip(axes, metrics):
        pivot = df.pivot(index="lambda", columns="physics_model", values=col)
        pivot = pivot.reindex(index=lambda_vals, columns=phys_order)
        pivot.columns = phys_labels

        sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    annot=True, fmt=".3f", linewidths=0.5,
                    cbar_kws={"label": ylabel})

        # mark best cell
        if higher_better:
            best_idx = np.unravel_index(pivot.values.argmax(), pivot.shape)
        else:
            best_idx = np.unravel_index(pivot.values.argmin(), pivot.shape)
        ax.add_patch(plt.Rectangle(
            (best_idx[1], best_idx[0]), 1, 1,
            fill=False, edgecolor="gold", lw=2.5, zorder=10
        ))

        arrow = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{ylabel}  ({arrow})", fontsize=11)
        ax.set_xlabel("Physics residual model")
        ax.set_ylabel("λ (physics weight)")
        ax.set_yticklabels([f"{v:.3g}" for v in lambda_vals], rotation=0)

    # Plain-NN reference line annotation
    nn = next((r for r in MODELS if r["model"] == "feedforward_nn"), None)
    if nn:
        nn_r2   = nn["metrics"]["adj_r2"]
        nn_rmse = nn["metrics"]["rmse"]
        nn_mos  = nn["metrics"]["mos_01"]
        fig.text(0.5, -0.04,
                 f"Plain NN baseline (no physics):  adj-R²={nn_r2:.4f}  RMSE={nn_rmse:.4f} g  MOS={nn_mos:.4f} g",
                 ha="center", fontsize=10, color="dimgrey",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("PINN Ablation — Physics model × λ (gold border = best per metric)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, "ablation_heatmap.png")


# ── Fig 12 — PINN ablation lambda sweep line plots ────────────────────────────
def fig_ablation_lambda_curves():
    csv_path = RESULTS_DIR / "pinn_ablation.csv"
    if not csv_path.exists():
        print("  [SKIP] ablation_lambda_curves — run pinn_ablation.py first")
        return

    df = pd.read_csv(csv_path)
    PHYSICS_META = {
        "hooke_strain": ("Hooke (avg strain)",     "#1f77b4"),
        "energy_quad":  ("Energy quad (RF^2 ~ U)", "#d62728"),
        "energy_rate":  ("Energy rate (RF ~ dU/dF)","#2ca02c"),
    }
    metrics = [
        ("adj_r2", "Adjusted R²",  True),
        ("rmse",   "RMSE [g]",     False),
        ("mos_01", "MOS@1% [g]",   False),
    ]

    nn = next((r for r in MODELS if r["model"] == "feedforward_nn"), None)
    nn_vals = {
        "adj_r2": nn["metrics"]["adj_r2"] if nn else None,
        "rmse":   nn["metrics"]["rmse"]   if nn else None,
        "mos_01": nn["metrics"]["mos_01"] if nn else None,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (col, ylabel, higher_better) in zip(axes, metrics):
        for phys, (label, color) in PHYSICS_META.items():
            sub = df[df["physics_model"] == phys].sort_values("lambda")
            ax.plot(sub["lambda"], sub[col], "o-", color=color,
                    label=label, lw=2, ms=6)

        if nn_vals[col] is not None:
            ax.axhline(nn_vals[col], ls="--", lw=1.3, color="grey", alpha=0.8, label="Plain NN")

        if col == "mos_01":
            ax.axhline(THRESHOLD_MOS, ls=":", lw=1.2, color="darkorange",
                       alpha=0.9, label=f"{THRESHOLD_MOS}g target")

        ax.set_xscale("log")
        ax.set_xlabel("λ (physics weight)")
        ax.set_ylabel(ylabel)
        arrow = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{ylabel}  ({arrow})")

    axes[0].legend(fontsize=9)
    fig.suptitle("PINN — Physics model × λ ablation  (test set)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, "ablation_lambda_curves.png")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...\n")
    fig_pred_vs_true()
    fig_residual_dist()
    fig_abs_error_cdf()
    fig_overpredict_cdf()
    fig_mos_sensitivity()
    fig_error_vs_glimit()
    fig_safety_bars()
    fig_pareto_time()
    fig_pareto_interp()
    _top3 = {"linear_regression", "polynomial_regression", "gaussian_process_regression"}
    fig_pareto_time(subset=_top3, suffix="_top3")
    fig_pareto_interp(subset=_top3, suffix="_top3")
    fig_nn_vs_pinn()
    fig_ablation_heatmap()
    fig_ablation_lambda_curves()
    print(f"\nAll figures → {FIGURES_DIR}/")
