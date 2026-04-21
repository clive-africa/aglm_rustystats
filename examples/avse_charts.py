"""
ave_charts.py
=============
Actual vs Expected (AvE) lift charts for a fitted Poisson frequency model.

For each input variable produces a dual-axis panel showing:
  - Bar chart : total Exposure per band (right axis, grey)
  - Line      : actual claim frequency  = sum(ClaimNb) / sum(Exposure)  (blue)
  - Line      : predicted claim frequency = sum(y_pred) / sum(Exposure) (red)

Banding rules:
  - Continuous numeric   : equal-frequency quantile bands (max 40)
  - Categorical ≤ 40 lvls: one bar per level, sorted by actual frequency
  - Categorical > 40 lvls: top-39 levels by exposure + "Other" bucket

Usage
-----
Call `plot_ave_charts(train, y_pred, feature_cols)` after fitting.
`y_pred` should be the model's predicted number of claims (response scale,
already multiplied by Exposure if the model predicts a rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Styling ───────────────────────────────────────────────────────────────────
ACTUAL_COLOR  = "#1A3C6E"   # deep blue
PRED_COLOR    = "#C0392B"   # red
EXPO_COLOR    = "#BDC3C7"   # light grey
GRID_ALPHA    = 0.3

MAX_BANDS     = 40          # maximum number of bands per variable

plt.rcParams.update({
    "font.family":          "sans-serif",
    "axes.spines.top":      False,
    "axes.grid":            True,
    "grid.alpha":           GRID_ALPHA,
    "grid.linestyle":       "--",
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
})


# ─────────────────────────────────────────────────────────────────────────────
# Banding helpers
# ─────────────────────────────────────────────────────────────────────────────

def band_numeric(series: pd.Series, max_bands: int = MAX_BANDS) -> pd.Series:
    """
    Equal-frequency quantile banding for a continuous variable.
    Returns a categorical Series whose labels are the band midpoints
    (floats) so the x-axis is ordered correctly.
    """
    n_unique = series.nunique()
    n_bands  = min(max_bands, n_unique)

    # pd.qcut collapses duplicate edges automatically with duplicates="drop"
    banded = pd.qcut(series, q=n_bands, duplicates="drop")

    # Use interval midpoint as the band label (numeric, sortable)
    midpoints = banded.apply(
        lambda iv: iv.mid if pd.notna(iv) else np.nan
    )
    return midpoints


def band_categorical(series: pd.Series,
                     exposure: pd.Series,
                     max_bands: int = MAX_BANDS) -> pd.Series:
    """
    For categoricals with > max_bands levels, keep the top (max_bands-1)
    levels by total exposure and roll the rest into "Other".
    Returns the series with the same index, values are strings.
    """
    n_levels = series.nunique()
    if n_levels <= max_bands:
        return series.astype(str)

    # Top levels by exposure
    top = (
        pd.Series(exposure.values, index=series.values)
        .groupby(level=0)
        .sum()
        .nlargest(max_bands - 1)
        .index
    )
    out = series.astype(str).copy()
    out[~out.isin(top)] = "Other"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-variable aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_bands(
    band_col: pd.Series,
    actual: pd.Series,
    predicted: pd.Series,
    exposure: pd.Series,
    is_numeric: bool,
) -> pd.DataFrame:
    """
    Aggregate actual counts, predicted counts, and exposure by band.

    Returns a DataFrame with columns:
        band        : band label
        actual_freq : sum(ClaimNb) / sum(Exposure)
        pred_freq   : sum(y_pred)  / sum(Exposure)
        exposure    : sum(Exposure)
    sorted by band (numeric order for continuous, exposure desc for categorical).
    """
    df = pd.DataFrame({
        "band":      band_col,
        "actual":    actual.values,
        "predicted": predicted.values,
        "exposure":  exposure.values,
    })

    agg = df.groupby("band", observed=True).agg(
        actual_sum    = ("actual",    "sum"),
        predicted_sum = ("predicted", "sum"),
        exposure_sum  = ("exposure",  "sum"),
    ).reset_index()

    agg["actual_freq"] = agg["actual_sum"]    / agg["exposure_sum"].clip(lower=1e-9)
    agg["pred_freq"]   = agg["predicted_sum"] / agg["exposure_sum"].clip(lower=1e-9)

    if is_numeric:
        agg = agg.sort_values("band")
    else:
        # Sort: "Other" last, remainder by exposure descending
        other = agg[agg["band"] == "Other"]
        rest  = agg[agg["band"] != "Other"].sort_values("exposure_sum", ascending=False)
        agg   = pd.concat([rest, other], ignore_index=True)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Single-panel plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_one_variable(
    ax_freq: plt.Axes,
    agg: pd.DataFrame,
    col_name: str,
    is_numeric: bool,
    overall_actual: float,
    overall_pred: float,
) -> None:
    """Draw one AvE panel onto ax_freq (frequency) with a shared exposure bar axis."""

    ax_expo = ax_freq.twinx()          # right axis for exposure bars
    ax_expo.spines["right"].set_visible(True)

    x      = np.arange(len(agg))
    labels = agg["band"].astype(str).tolist()

    # ── Exposure bars (background) ────────────────────────────────────────
    ax_expo.bar(x, agg["exposure_sum"], color=EXPO_COLOR, alpha=0.55,
                width=0.75, zorder=1, label="Exposure")
    ax_expo.set_ylabel("Exposure", color="grey", fontsize=8)
    ax_expo.tick_params(axis="y", labelcolor="grey", labelsize=7)
    ax_expo.set_ylim(bottom=0,
                     top=agg["exposure_sum"].max() * 3.0)   # push bars to bottom third

    # ── Frequency lines (foreground) ──────────────────────────────────────
    ax_freq.plot(x, agg["actual_freq"], color=ACTUAL_COLOR,
                 lw=2.0, marker="o", ms=4, zorder=3, label="Actual")
    ax_freq.plot(x, agg["pred_freq"],  color=PRED_COLOR,
                 lw=2.0, marker="s", ms=4, zorder=3, label="Predicted",
                 linestyle="--")

    # Overall average reference lines
    ax_freq.axhline(overall_actual, color=ACTUAL_COLOR, lw=0.8,
                    ls=":", alpha=0.6)
    ax_freq.axhline(overall_pred,   color=PRED_COLOR,   lw=0.8,
                    ls=":", alpha=0.6)

    # ── Axes formatting ───────────────────────────────────────────────────
    ax_freq.set_xticks(x)
    if is_numeric:
        # Format midpoint numbers cleanly
        try:
            formatted = [f"{float(lb):.1f}" for lb in labels]
        except ValueError:
            formatted = labels
    else:
        formatted = labels

    ax_freq.set_xticklabels(formatted, rotation=45, ha="right", fontsize=7)
    ax_freq.set_ylabel("Claim Frequency", fontsize=8)
    ax_freq.set_title(col_name, fontsize=10, fontweight="bold", pad=6)
    ax_freq.set_zorder(ax_expo.get_zorder() + 1)   # frequency lines on top
    ax_freq.patch.set_visible(False)                # transparent so bars show through

    # Frequency y-axis: start at 0, give 10% headroom above max line
    freq_max = max(agg["actual_freq"].max(), agg["pred_freq"].max())
    ax_freq.set_ylim(bottom=0, top=freq_max * 1.15)
    ax_freq.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def plot_ave_charts(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    feature_cols: list[str],
    actual_col: str   = "ClaimNb",
    exposure_col: str = "Exposure",
    max_bands: int    = MAX_BANDS,
    ncols: int        = 3,
    fig_title: str    = "Actual vs Expected — Claim Frequency by Variable",
    figsize_per_panel: tuple[float, float] = (5.5, 4.2),
) -> plt.Figure:
    """
    Plot Actual vs Expected lift charts for every feature in `feature_cols`.

    Parameters
    ----------
    df           : DataFrame containing actual counts, exposure, and features.
                   Must already be aligned with `y_pred` (same row order).
    y_pred       : 1-D array of predicted claim counts (response scale,
                   i.e. predicted number of claims, NOT rate per year).
    feature_cols : List of column names to plot (numeric and/or categorical).
    actual_col   : Column name for observed claim counts.
    exposure_col : Column name for exposure (in policy-years).
    max_bands    : Maximum number of bands per variable (default 40).
    ncols        : Number of subplot columns (default 3).
    fig_title    : Overall figure title.
    figsize_per_panel : (width, height) per subplot panel in inches.

    Returns
    -------
    matplotlib Figure — call fig.savefig(...) or plt.show() as needed.
    """
    y_pred   = np.asarray(y_pred, dtype=float)
    actual   = df[actual_col].values.astype(float)
    exposure = df[exposure_col].values.astype(float)

    # Overall portfolio averages (for reference lines)
    overall_actual = actual.sum()   / exposure.sum()
    overall_pred   = y_pred.sum()   / exposure.sum()

    n_panels = len(feature_cols)
    nrows    = int(np.ceil(n_panels / ncols))
    fig_w    = figsize_per_panel[0] * ncols
    fig_h    = figsize_per_panel[1] * nrows

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(fig_w, fig_h),
                              squeeze=False)

    fig.suptitle(
        f"{fig_title}\n"
        f"Overall actual freq = {overall_actual:.4f}  |  "
        f"Overall predicted freq = {overall_pred:.4f}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for panel_idx, col in enumerate(feature_cols):
        row = panel_idx // ncols
        col_idx = panel_idx % ncols
        ax = axes[row][col_idx]

        series = df[col]

        # ── Determine variable type and band ──────────────────────────────
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric:
            band_col = band_numeric(series, max_bands=max_bands)
        else:
            band_col = band_categorical(series, pd.Series(exposure), max_bands)

        # ── Aggregate ─────────────────────────────────────────────────────
        agg = aggregate_bands(
            band_col      = band_col,
            actual        = pd.Series(actual),
            predicted     = pd.Series(y_pred),
            exposure      = pd.Series(exposure),
            is_numeric    = is_numeric,
        )

        # ── Plot ──────────────────────────────────────────────────────────
        _plot_one_variable(
            ax_freq        = ax,
            agg            = agg,
            col_name       = col,
            is_numeric     = is_numeric,
            overall_actual = overall_actual,
            overall_pred   = overall_pred,
        )

    # ── Hide unused panels ────────────────────────────────────────────────
    for panel_idx in range(n_panels, nrows * ncols):
        row = panel_idx // ncols
        col_idx = panel_idx % ncols
        axes[row][col_idx].set_visible(False)

    # ── Shared legend (top-right of figure) ───────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=ACTUAL_COLOR, lw=2, marker="o", ms=5,
               label="Actual frequency"),
        Line2D([0], [0], color=PRED_COLOR,   lw=2, marker="s", ms=5,
               linestyle="--", label="Predicted frequency"),
        plt.Rectangle((0, 0), 1, 1, fc=EXPO_COLOR, alpha=0.6,
                       label="Exposure (bars)"),
        Line2D([0], [0], color=ACTUAL_COLOR, lw=0.9, ls=":",
               label="Portfolio average (actual)"),
        Line2D([0], [0], color=PRED_COLOR,   lw=0.9, ls=":",
               label="Portfolio average (predicted)"),
    ]
    fig.legend(
        handles        = legend_handles,
        loc            = "lower center",
        ncol           = 3,
        fontsize       = 9,
        frameon        = True,
        bbox_to_anchor = (0.5, -0.04),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: one-call wrapper for the experiment script
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_models_ave(
    df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    feature_cols: list[str],
    actual_col: str   = "ClaimNb",
    exposure_col: str = "Exposure",
    max_bands: int    = MAX_BANDS,
    output_prefix: str = "ave",
) -> None:
    """
    Produce and save one AvE figure per model.

    Parameters
    ----------
    df          : DataFrame (test set).
    predictions : Dict mapping model name → predicted claim count array.
    feature_cols: Variables to plot.
    output_prefix: Filename prefix; files saved as {prefix}_{model_name}.png
    """
    for model_name, y_pred in predictions.items():
        print(f"  Plotting AvE for {model_name}…")
        fig = plot_ave_charts(
            df            = df,
            y_pred        = y_pred,
            feature_cols  = feature_cols,
            actual_col    = actual_col,
            exposure_col  = exposure_col,
            max_bands     = max_bands,
            fig_title     = f"Actual vs Expected — {model_name}",
        )
        fname = f"{output_prefix}_{model_name.replace(' ', '_')}.png"
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        print(f"    Saved: {fname}")
        plt.close(fig)