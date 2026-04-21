"""
Plotting utilities for fitted AGLM models.

Python port of R/plot-aglm.R from kkondo1981/aglm.

:func:`plot_aglm` produces a per-variable contribution plot.  For each
original variable a curve (numeric) or bar chart (categorical) shows the
model's predicted linear predictor as a function of that variable alone,
with all other variables held at their median / most-frequent level.

This visual representation of how variables are used by the model is one of
the key interpretability features of AGLM (as emphasised in the original R
package documentation).

Reference:
  Fujita, Tanaka, Kondo & Iwasawa (2020).
  AGLM: A Hybrid Modeling Method of GLM and Data Science Techniques.
  Actuarial Colloquium Paris 2020.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .model import AccurateGLM, CVAAccurateGLM
from .input import AGLMInput, VarInfo


# ---------------------------------------------------------------------------
# plot_aglm — main plotting entry point
# ---------------------------------------------------------------------------

def plot_aglm(
    model: Union[AccurateGLM, CVAAccurateGLM],
    which_vars: Optional[List[Union[int, str]]] = None,
    n_grid: int = 200,
    ncols: int = 3,
    figsize: Optional[tuple] = None,
    show_residuals: bool = True,
    show_cv_curve: bool = True,
    link_scale: bool = True,
    title_prefix: str = "",
) -> plt.Figure:
    """Plot per-variable contributions of a fitted AGLM model.

    For each variable the function sweeps a grid of values through the model
    while keeping all other variables at their reference level (median for
    numeric, most-frequent for categorical) and plots the resulting linear
    predictor (or response, controlled by ``link_scale``).

    Args:
        model:          Fitted :class:`~aglm.model.AccurateGLM` or
                        :class:`~aglm.model.CVAAccurateGLM` object.
        which_vars:     Subset of variable names or 0-based indices to plot.
                        If ``None`` all original (non-interaction) variables are shown.
        n_grid:         Number of grid points for numeric sweeps.
        ncols:          Columns in the subplot grid.
        figsize:        Figure size ``(width, height)``.  Auto-sized if ``None``.
        show_residuals: If ``True`` add a residuals-vs-fitted subplot.
        show_cv_curve:  If ``True`` and the model has CV results, add a
                        lambda-path subplot.
        link_scale:     If ``True`` plot on the link (linear predictor) scale;
                        if ``False`` plot on the response scale.
        title_prefix:   Optional string prepended to the overall figure title.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    # Resolve CVAAccurateGLM → best model
    if isinstance(model, CVAAccurateGLM):
        model = model.best_model

    aglm_input: AGLMInput = model.aglm_input

    # ---- Select variables to plot ----------------------------------------
    orig_vars = [vi for vi in aglm_input.vars_info if vi.type in ("quan", "qual")]

    if which_vars is not None:
        col_names = list(aglm_input.data.columns)
        if all(isinstance(v, int) for v in which_vars):
            orig_vars = [orig_vars[i] for i in which_vars]
        else:
            orig_vars = [vi for vi in orig_vars if vi.name in which_vars]

    # ---- Compute reference row -------------------------------------------
    ref_df = _reference_row(aglm_input.data)

    # ---- Determine subplot count -----------------------------------------
    has_cv = show_cv_curve and model.cv_results is not None
    has_resid = show_residuals and hasattr(model, "_y")
    n_var_plots = len(orig_vars)
    n_total = n_var_plots + int(has_resid) + int(has_cv)
    nrows = max(1, int(np.ceil(n_total / ncols)))

    if figsize is None:
        figsize = (5.5 * ncols, 4.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).ravel()

    suptitle = f"{title_prefix}AGLM — {model.family.capitalize()} | α={model.alpha} λ={model.lambda_:.3g}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    # ---- Per-variable plots ----------------------------------------------
    for ax_idx, vi in enumerate(orig_vars):
        ax = axes[ax_idx]
        col_data = aglm_input.data.iloc[:, vi.data_col_idx]

        if vi.type == "quan":
            _plot_numeric_var(
                ax, vi, col_data, ref_df, model, aglm_input, n_grid, link_scale
            )
        else:
            _plot_categorical_var(
                ax, vi, col_data, ref_df, model, aglm_input, link_scale
            )

    # ---- Residuals vs Fitted ---------------------------------------------
    if has_resid:
        ax = axes[n_var_plots]
        _plot_residuals(ax, model)

    # ---- CV lambda path --------------------------------------------------
    if has_cv:
        ax = axes[n_var_plots + int(has_resid)]
        _plot_cv_path(ax, model)

    # ---- Hide unused axes ------------------------------------------------
    for ax in axes[n_total:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sub-plot helpers
# ---------------------------------------------------------------------------

def _predict_at(
    model: AccurateGLM,
    df_single_row: pd.DataFrame,
    link_scale: bool,
) -> float:
    """Return the model output for a single-row DataFrame."""
    X = model.aglm_input.transform(df_single_row)
    if link_scale:
        return _linear_predictor(model, X)
    else:
        pred = model.predict(df_single_row)
        return float(pred[0])


def _linear_predictor(model: AccurateGLM, X: np.ndarray) -> float:
    """Extract the linear predictor (η = Xβ + intercept) for a single-row X.

    * Gaussian backend is sklearn ElasticNet — its ``predict()`` is already
      on the linear (identity link) scale.
    * Binomial / Poisson backends are :class:`StatsmodelsGLMEstimator` — use
      ``predict_linear()`` which returns η directly.
    """
    if model.family == "gaussian":
        return float(model.backend_model.predict(X)[0])
    # StatsmodelsGLMEstimator exposes predict_linear() for the link scale.
    if hasattr(model.backend_model, "predict_linear"):
        return float(model.backend_model.predict_linear(X)[0])
    # Fallback: derive η from response-scale prediction via the inverse link.
    mu = float(model.backend_model.predict(X)[0])
    if model.family == "poisson":
        return float(np.log(max(mu, 1e-15)))
    return mu


def _reference_row(data: pd.DataFrame) -> pd.DataFrame:
    """Build a single-row reference DataFrame (median / mode per column)."""
    ref = {}
    for col in data.columns:
        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            ref[col] = [float(series.median())]
        else:
            ref[col] = [series.mode().iloc[0]]
    return pd.DataFrame(ref)


def _plot_numeric_var(
    ax: plt.Axes,
    vi: VarInfo,
    col_data: pd.Series,
    ref_df: pd.DataFrame,
    model: AccurateGLM,
    aglm_input: AGLMInput,
    n_grid: int,
    link_scale: bool,
) -> None:
    x_min, x_max = float(col_data.min()), float(col_data.max())
    x_grid = np.linspace(x_min, x_max, n_grid)
    contributions = []

    for xval in x_grid:
        tmp = ref_df.copy()
        tmp.iloc[0, vi.data_col_idx] = xval
        contributions.append(_predict_at(model, tmp, link_scale))

    ax.plot(x_grid, contributions, linewidth=2.0, color="#1f77b4")
    ax.fill_between(x_grid, contributions, alpha=0.12, color="#1f77b4")

    # Rug plot for data density
    rug_vals = col_data.values
    y_rug = ax.get_ylim()[0] if contributions else 0
    ax.plot(rug_vals, np.full_like(rug_vals, np.min(contributions)),
            "|", color="grey", alpha=0.3, markersize=4)

    scale_label = "Linear predictor" if link_scale else "Response"
    ax.set_xlabel(vi.name, fontsize=10)
    ax.set_ylabel(scale_label, fontsize=9)
    ax.set_title(vi.name, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")


def _plot_categorical_var(
    ax: plt.Axes,
    vi: VarInfo,
    col_data: pd.Series,
    ref_df: pd.DataFrame,
    model: AccurateGLM,
    aglm_input: AGLMInput,
    link_scale: bool,
) -> None:
    levels = vi.ud_info["levels"] if vi.ud_info else sorted(col_data.unique().tolist())
    contributions = []

    for lv in levels:
        tmp = ref_df.copy()
        tmp.iloc[0, vi.data_col_idx] = lv
        contributions.append(_predict_at(model, tmp, link_scale))

    colors = ["#2196F3" if c >= 0 else "#EF5350" for c in contributions]
    bars = ax.bar(range(len(levels)), contributions, color=colors, alpha=0.82, edgecolor="white")

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=40, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    scale_label = "Linear predictor" if link_scale else "Response"
    ax.set_ylabel(scale_label, fontsize=9)
    ax.set_title(vi.name, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")


def _plot_residuals(ax: plt.Axes, model: AccurateGLM) -> None:
    fitted = model.predict()
    resid = model._y - fitted

    ax.scatter(fitted, resid, alpha=0.4, s=12, color="#555555")
    ax.axhline(0, color="#e53935", linewidth=1.2, linestyle="--")

    # Loess-like trend using rolling average
    try:
        order = np.argsort(fitted)
        window = max(1, len(fitted) // 20)
        smooth = pd.Series(resid[order]).rolling(window, center=True).mean()
        ax.plot(np.sort(fitted), smooth.values, color="#43a047", linewidth=1.5)
    except Exception:
        pass

    ax.set_xlabel("Fitted values", fontsize=10)
    ax.set_ylabel("Residuals", fontsize=10)
    ax.set_title("Residuals vs Fitted", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")


def _plot_cv_path(ax: plt.Axes, model: AccurateGLM) -> None:
    cv = model.cv_results
    lam = cv.get("lambda_grid")
    scores = cv.get("mean_cv_score")

    if lam is None or scores is None:
        ax.set_visible(False)
        return

    ax.semilogx(lam, scores, linewidth=2, color="#7b1fa2")
    ax.axvline(model.lambda_, color="#e53935", linewidth=1.5,
               linestyle="--", label=f"Best λ = {model.lambda_:.4g}")

    ax.set_xlabel("λ (regularisation strength)", fontsize=10)
    ax.set_ylabel("CV score (higher = better)", fontsize=9)
    ax.set_title("Cross-validation path", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")


# ---------------------------------------------------------------------------
# Convenience: plot_cva_alpha — show CV scores across alpha values
# ---------------------------------------------------------------------------

def plot_cva_alpha(
    cva_model: CVAAccurateGLM,
    figsize: tuple = (7, 4),
) -> plt.Figure:
    """Bar chart of best CV scores across alpha values for a CVA model.

    Args:
        cva_model: Fitted :class:`~aglm.model.CVAAccurateGLM`.
        figsize:   Figure size.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    alphas = sorted(cva_model.cv_scores.keys())
    scores = [cva_model.cv_scores[a] for a in alphas]
    colors = [
        "#e53935" if a == cva_model.best_alpha else "#1e88e5"
        for a in alphas
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([str(round(a, 3)) for a in alphas], scores, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(
        cva_model.cv_scores[cva_model.best_alpha],
        color="#e53935", linestyle="--", linewidth=1.2,
        label=f"Best α = {cva_model.best_alpha}",
    )
    ax.set_xlabel("α (elastic-net mixing)", fontsize=11)
    ax.set_ylabel("Best CV score", fontsize=11)
    ax.set_title("CVA — cross-validation score by α", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    fig.tight_layout()
    return fig
