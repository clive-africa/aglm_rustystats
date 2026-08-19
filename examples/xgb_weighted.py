"""
xgb_weighted.py
===============
XGBoost variant that uses sample weights (= Exposure) instead of a log-offset.

Training target  : ClaimNb / Exposure  (the frequency rate)
Sample weight    : Exposure
Prediction       : predicted_rate × Exposure → counts  (same units as other models)

Motivation
----------
The standard XGBoost "offset" approach sets log(Exposure) as a base_margin, which
works but can interact poorly with max_delta_step (clamps updates to ±0.7 for
count:poisson), causing slow convergence and early stopping to trigger before the
model is fully trained.  The weighted-rate approach mirrors the LightGBM training
setup and avoids both issues.

Usage
-----
    from xgb_weighted import fit_xgb_weighted, predict_xgb_weighted

    res = fit_xgb_weighted(
        train, response_col="ClaimNb", exposure_col="Exposure",
        numeric_cols=[...], categorical_cols=[...], family="poisson",
    )
    counts = predict_xgb_weighted(res, test)   # → np.ndarray of expected counts
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
import xgboost as xgb

# ── Objective / metric mapping ──────────────────────────────────────────────
_OBJ: dict[str, str] = {
    "poisson": "count:poisson",
    "gamma":   "reg:gamma",
    "tweedie": "reg:tweedie",
}
_METRIC: dict[str, str] = {
    "poisson": "poisson-nloglik",
    "gamma":   "gamma-nloglik",
    "tweedie": "tweedie-nloglik",
}


def _make_X(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Build feature matrix with categoricals cast to the `category` dtype."""
    x = df[numeric_cols + categorical_cols].copy()
    for col in categorical_cols:
        x[col] = x[col].astype("category")
    return x


# ── Public API ───────────────────────────────────────────────────────────────

def fit_xgb_weighted(
    train:            pd.DataFrame,
    response_col:     str,
    exposure_col:     str,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    family:           str,
    cv_folds:         int = 5,
) -> dict:
    """
    Fit an XGBoost model using rate-as-label + exposure-as-weight.

    Parameters
    ----------
    train            : training DataFrame
    response_col     : claim count column  (e.g. "ClaimNb")
    exposure_col     : exposure column     (e.g. "Exposure")
    numeric_cols     : list of numeric feature column names
    categorical_cols : list of categorical feature column names
    family           : "poisson", "gamma", or "tweedie"
    cv_folds         : number of cross-validation folds for n_estimators selection

    Returns
    -------
    dict with keys: model, numeric_cols, categorical_cols, exposure_col,
                    best_n, timer
    """
    obj    = _OBJ.get(family,    "count:poisson")
    metric = _METRIC.get(family, "poisson-nloglik")
    print(
        f"\n[XGBoost-W] {obj}  (rate label + exposure weights, no offset) ...",
        flush=True,
    )
    t0 = time.time()

    # Label = frequency rate; weight = exposure so that the loss is
    # exposure-weighted, mirroring an offset model without the convergence issues.
    freq = train[response_col].values / np.maximum(train[exposure_col].values, 1e-9)
    w    = train[exposure_col].values

    dtrain = xgb.DMatrix(
        _make_X(train, numeric_cols, categorical_cols),
        label=freq,
        weight=w,
        enable_categorical=True,
    )

    params: dict = {
        "objective":        obj,
        "eval_metric":      metric,
        "learning_rate":    0.01,
        "max_depth":        4,
        "min_child_weight": 50,
        "subsample":        0.9,
        "colsample_bytree": 0.8,
        "seed":             42,
        "nthread":          4,
        "verbosity":        0,
    }

    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round=500,
        nfold=cv_folds,
        stratified=False,
        early_stopping_rounds=20,
        callbacks=[xgb.callback.EvaluationMonitor(show_stdv=False, period=999)],
    )

    # Identify the true optimum, not the last row (which is early_stopping_rounds
    # past the best, or exactly best + 1 when the CV loop ended early).
    metric_col = f"test-{metric}-mean"
    if metric_col in cv_result.columns:
        best_n = int(cv_result[metric_col].idxmin()) + 1
    else:
        best_n = max(1, len(cv_result) - 20)   # fallback: subtract early-stop rounds

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=best_n,
        verbose_eval=False,
    )

    elapsed = time.time() - t0
    print(f"  best n_estimators={best_n}   elapsed {elapsed:.1f}s")

    return {
        "model":            booster,
        "numeric_cols":     numeric_cols,
        "categorical_cols": categorical_cols,
        "exposure_col":     exposure_col,
        "best_n":           best_n,
        "timer":            elapsed,
    }


def predict_xgb_weighted(res: dict, df: pd.DataFrame) -> np.ndarray:
    """
    Return expected claim **counts** for every row in *df*.

    The model predicts the frequency rate; multiplying by Exposure gives counts
    in the same units as ClaimNb (or whichever response_col was used at fit time).
    """
    dtest = xgb.DMatrix(
        _make_X(df, res["numeric_cols"], res["categorical_cols"]),
        enable_categorical=True,
    )
    rate = res["model"].predict(dtest)
    return rate * df[res["exposure_col"]].values
