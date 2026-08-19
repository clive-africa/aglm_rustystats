import numpy as np
from sklearn.metrics import roc_auc_score
import inspect
from typing import Callable, Any
import warnings
import pandas as pd


from .glm import fit_glm, predict_glm, fit_reg_glm, predict_reg_glm
from .aglm import (
    fit_aglm_linear,
    predict_aglm_linear,
    fit_aglm_lvar,
    predict_aglm_lvar,
)
from .gam import fit_gam, predict_gam
from .xgb import fit_xgb, predict_xgb
from .cat import fit_catboost, predict_catboost
from .lasso import fit_derivative_lasso, predict_derivative_lasso
from .lgbm import fit_lgbm, predict_lgbm

# ---------------------------------------------------------------------------
# Generic Model Fitting 
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Callable] = {
    "GLM":        fit_glm,
    "RegGLM":     fit_reg_glm,
    "AGLM-Lin":   fit_aglm_linear,
    "AGLM-Lvar":  fit_aglm_lvar,
    "GAM":        fit_gam,
    "GBM":        fit_lgbm,
    "XGB":        fit_xgb,
    "CatBoost":   fit_catboost,
    "DerivLasso": fit_derivative_lasso,
}

def call_fit(fn, _used: set[str] | None = None, **kwargs):
    accepted = inspect.signature(fn).parameters
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    if _used is not None:
        _used.update(filtered)
    return fn(**filtered)

def fit_models(models: list[str], **common) -> dict:
    results = {}
    used: set[str] = set()
    for name in models:
        results[name] = call_fit(MODEL_REGISTRY[name], _used=used, **common)

    unused = common.keys() - used
    if unused:
        warnings.warn(f"fit_models: arguments never consumed by any model: {sorted(unused)}")

    return results

PREDICT_REGISTRY: dict[str, Callable] = {
    "GLM":        predict_glm,
    "RegGLM":     predict_reg_glm,
    "AGLM-Lin":   predict_aglm_linear,
    "AGLM-Lvar":  predict_aglm_lvar,
    "GAM":        predict_gam,
    "GBM":        predict_lgbm,
    "XGB":        predict_xgb,
    "CatBoost":   predict_catboost,
    "DerivLasso": predict_derivative_lasso,
}

def make_predict_fns(results: dict) -> dict[str, Callable[[pd.DataFrame], np.ndarray]]:
    predict_fns = {}
    for name, fitted in results.items():
        predict_fn = PREDICT_REGISTRY[name]
        model = fitted["model"] if isinstance(fitted, dict) and "model" in fitted else fitted
        predict_fns[name] = lambda df, _fn=predict_fn, _model=model: _fn(_model, df)
    return predict_fns

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance - 2x convention (matches R GLM / sklearn / paper)."""
    y  = np.asarray(y_true, float)
    mu = np.maximum(np.asarray(y_pred, float), 1e-12)
    return float(2.0 * np.mean(
        np.where(y > 0, y * np.log(y / mu), 0.0) - y + mu
    ))


def _weighted_gini(y_true, y_pred, exposure):
    order  = np.argsort(y_pred)
    y_s    = y_true[order]
    e_s    = exposure[order]
    cum_e  = np.cumsum(e_s) / e_s.sum()
    cum_l  = np.cumsum(y_s * e_s) / (y_s * e_s).sum()
    return float(1 - 2 * np.trapezoid(cum_l, cum_e))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v_s = v[order]
    w_s = w[order]
    cdf = np.cumsum(w_s) / np.sum(w_s)
    return float(v_s[np.searchsorted(cdf, 0.5, side="left")])

def compute_metrics(
    name: str,
    y_count: np.ndarray,
    y_pred_count: np.ndarray,
    exposure: np.ndarray,
) -> dict:
    """Compute the full suite of comparison metrics for one model."""
    y   = np.asarray(y_count,       float)
    mu  = np.maximum(np.asarray(y_pred_count, float), 1e-12)
    exp = np.asarray(exposure,      float)

    freq_true = y  / np.maximum(exp, 1e-9)
    freq_pred = mu / np.maximum(exp, 1e-9)

    dev = _poisson_deviance(y, mu)
    mse = float(np.mean((freq_true - freq_pred) ** 2))
    mae = float(np.mean(np.abs(freq_true - freq_pred)))

    binary = (y > 0).astype(int)
    auc = (float(roc_auc_score(binary, freq_pred))
           if 0 < binary.sum() < len(binary) else float("nan"))

    gini = _weighted_gini(y, mu, exp)

    w          = exp / exp.sum()
    avg_pred   = float(np.sum(freq_pred * w))
    med_pred   = _weighted_median(freq_pred, w)

    avg_actual = float(np.sum(freq_true * w))
    med_actual = _weighted_median(freq_true, w)

    return {
        "Model":             name,
        "Poisson Deviance":  dev,
        "MSE":               mse,
        "MAE":               mae,
        "AUC":               auc,
        "Gini":              gini,
        "Avg Pred (freq)":   avg_pred,
        "Median Pred (freq)": med_pred,
        "Avg Actual (freq)": avg_actual,
        "Median Actual (freq)": med_actual,
    }

def model_metrics(
    test: pd.DataFrame,
    predict_fns: dict[str, Callable[[pd.DataFrame], np.ndarray]],
    models: list[str],
    common: dict[str, Any]
) -> list[dict]:
    """Compute metrics for multiple models."""

    rows = []
    for name in models.keys():
        fn=predict_fns[name]
        mu = fn(test)
        row = compute_metrics(
            name, test[common['response_col']].values, mu, test[common['response_col']].values
        )
        rows.append(row)
        print(
            f"  {name:12s}: dev={row['Poisson Deviance']:.6f}  "
            f"mse={row['MSE']:.8f}  mae={row['MAE']:.6f}  "
            f"auc={row['AUC']:.4f}  gini={row['Gini']:.4f} "
            f"avg_pred={row['Avg Pred (freq)']:.5f} "
            f"med_pred={row['Median Pred (freq)']:.5f}"
        )

    return rows
    