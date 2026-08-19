import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import time
import pandas as pd
import numpy as np
from typing import Any, Literal
from .config import COLORS

def plot_component_curves(
    aglm_cva: "CVAAccurateGLM",  # noqa: F821
    train: pd.DataFrame,
    metrics_table: pd.DataFrame,
    #exposure_col: str,
    offset_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str]
) -> plt.Figure:
    
    feature_cols=numeric_cols+categorical_cols
    aglm_cols = offset_col + feature_cols
    
    best_model = aglm_cva.best_model

    ref: dict = {col: float(train[col].median()) for col in numeric_cols}
    ref.update({col: train[col].mode().iloc[0] for col in categorical_cols})
    ref["Exposure"]    = 1.0
    ref["LogExposure"] = 0.0

    ref_df  = pd.DataFrame([ref])[aglm_cols]
    mu_ref  = float(best_model.predict(ref_df)[0])
    log_ref = np.log(max(mu_ref, 1e-12))

    def component(sweep_df: pd.DataFrame) -> np.ndarray:
        mu = best_model.predict(sweep_df[aglm_cols])
        return np.log(np.maximum(mu, 1e-12)) - log_ref

    ncols   = 3
    n_plots = len(feature_cols) + 1
    nrows   = int(np.ceil(n_plots / ncols))

    fig = plt.figure(figsize=(5.5 * ncols, 4.0 * nrows))
    fig.suptitle(
        "Figure 2 - AGLM-Lvar Component Curves\n"
        "log contribution to claim frequency | freMTPL2freq",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs   = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.40)
    axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]
    ax_idx = 0

    for col in numeric_cols:
        ax  = axes[ax_idx]; ax_idx += 1
        lo  = float(train[col].quantile(0.01))
        hi  = float(train[col].quantile(0.99))
        grd = np.linspace(lo, hi, 150)
        rows = [{**ref, col: v} for v in grd]
        swp  = pd.DataFrame(rows)[aglm_cols]
        cmp  = component(swp)
        ax.plot(grd, cmp, lw=2.0, color=COLORS["AGLM-Lvar"])
        ax.fill_between(grd, cmp, alpha=0.12, color=COLORS["AGLM-Lvar"])
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        rug = train[col].sample(min(1200, len(train)), random_state=0).values
        ax.plot(rug, np.full_like(rug, cmp.min()), "|",
                color="grey", alpha=0.18, ms=3)
        ax.set(xlabel=col, ylabel="log contribution", title=col)
        ax.title.set_fontweight("bold")

    for col in categorical_cols:
        ax   = axes[ax_idx]; ax_idx += 1
        lvls = sorted(train[col].unique())
        rows = [{**ref, col: lv} for lv in lvls]
        swp  = pd.DataFrame(rows)[aglm_cols]
        cmp  = component(swp)
        bar_colors = [COLORS["AGLM-Lvar"] if c >= 0 else COLORS["GBM"] for c in cmp]
        ax.bar(range(len(lvls)), cmp, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(lvls)))
        ax.set_xticklabels(lvls, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set(ylabel="log contribution", title=col)
        ax.title.set_fontweight("bold")

    # Metrics summary inset
    ax = axes[ax_idx]; ax_idx += 1
    ax.axis("off")
    rows_tbl = [
        [r["Model"], f"{r['Poisson Deviance']:.5f}", f"{r['AUC']:.4f}"]
        for _, r in metrics_table.iterrows()
    ]
    tbl = ax.table(
        cellText=rows_tbl,
        colLabels=["Model", "Poisson Dev", "AUC"],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.05, 1.55)
    best_idx = metrics_table["Poisson Deviance"].idxmin()
    for j in range(3):
        tbl[(best_idx + 1, j)].set_facecolor("#D6EAF8")
        tbl[(best_idx + 1, j)].set_text_props(fontweight="bold")
    ax.set_title("Model Metrics", fontsize=10, fontweight="bold")

    for ax in axes[ax_idx:]:
        ax.set_visible(False)

    return fig


# ---------------------------------------------------------------------------
# Figure 3 - Multi-metric comparison (2x2 subplots)
# ---------------------------------------------------------------------------

def plot_metrics_comparison(metrics_table: pd.DataFrame) -> plt.Figure:
    metric_cfg = [
        ("Poisson Deviance", "Mean Poisson Deviance - lower is better",  True),
        ("MSE",              "MSE on Claim Frequency - lower is better", True),
        ("MAE",              "MAE on Claim Frequency - lower is better", True),
        ("AUC",              "ROC-AUC (any claim) - higher is better",   False),
    ]
    # Map new model names to paper Table 8 reference values
    paper = {
        "AGLM-Lvar": 0.3111920,
        "GLM":       0.3201199,
        "AGLM-Lin":  0.3201245,
        "GAM":       0.3171236,
        "GBM":       0.3123919,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Model Comparison - freMTPL2freq (N={N_SAMPLE or 678_013:,})\n"
        "Diamond = Fujita et al. Table 8 reference (full N=678k, Poisson Deviance only)",
        fontsize=12, fontweight="bold",
    )

    for ax, (metric, xlabel, lower_better) in zip(axes.flat, metric_cfg):
        srt    = metrics_table.sort_values(metric, ascending=lower_better)
        models = srt["Model"].tolist()
        values = srt[metric].tolist()
        bcs    = [COLORS.get(m, "#888") for m in models]

        bars = ax.barh(models, values, color=bcs,
                       alpha=0.88, edgecolor="white", height=0.55)
        best_val = min(values) if lower_better else max(values)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + abs(best_val) * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.5f}", va="center", fontsize=8.5)

        if metric == "Poisson Deviance":
            for m_name, pdv in paper.items():
                if m_name in models:
                    ax.plot(pdv, models.index(m_name), marker="D",
                            ms=6, color="black", alpha=0.45, zorder=5)

        ref_line = min(values) if lower_better else max(values)
        ax.axvline(ref_line, color="black", lw=1.0, ls=":",
                   label=f"Best = {ref_line:.5f}")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.legend(fontsize=8)
        margin = abs(best_val) * 0.03
        if lower_better:
            ax.set_xlim(left=min(values) - margin)
        else:
            ax.set_xlim(right=max(values) + margin)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Figure 4 - Calibration (average predicted vs actual frequency)
# ---------------------------------------------------------------------------

def plot_calibration(metrics_table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))

    actual = metrics_table["Avg Actual (freq)"].iloc[0]
    models = metrics_table["Model"].tolist()
    preds  = metrics_table["Avg Pred (freq)"].tolist()
    bcs    = [COLORS.get(m, "#888") for m in models]

    bars = ax.bar(models, preds, color=bcs, alpha=0.85,
                  edgecolor="white", width=0.55)
    ax.axhline(actual, color="black", lw=1.5, ls="--",
               label=f"Actual mean freq = {actual:.5f}")

    for bar, val in zip(bars, preds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + actual * 0.003,
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Exposure-weighted Mean Predicted Frequency", fontsize=10)
    ax.set_title(
        "Figure 4 - Prediction Calibration\n"
        "Exposure-weighted mean predicted vs actual claim frequency",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig