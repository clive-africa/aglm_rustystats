"""
model_benchmark.py
==================
Generic insurance pricing model benchmarker.

Fits up to nine models on any tabular dataset and produces:
  - A printed metrics table on the hold-out test set
  - An interactive self-contained HTML report

Supported tasks
---------------
  "frequency"  — Poisson GLM family; response is a count (e.g. ClaimNb),
                 exposure is time-at-risk (e.g. Exposure year fraction).
  "severity"   — Gamma GLM family; response is a positive amount
                 (e.g. ClaimAmount / ClaimNb for records with claims > 0),
                 exposure is the number of claims or a similar denominator.

Models
------
  1. GLM          — plain GLM (rustystats IRLS)
  2. RegGLM       — regularised GLM (rustystats elastic-net CV)
  3. AGLM-Lin     — AGLM without basis expansion (cva_aglm, elastic-net CV)
  4. AGLM-Lvar    — Full AGLM with L-variable basis (cva_aglm)
  5. GAM          — PoissonGAM / GammaGAM (pygam)
  6. GBM          — LightGBM (CV for n_estimators)
  7. CatBoost     — CatBoost regressor (native categoricals)
  8. XGBoost      — XGBoost (CV for n_estimators)
  9. DerivLasso   — Derivative (fused) lasso GLM via CVXPY

Usage (standalone)
------------------
    python model_benchmark.py                    # full freMTPL2freq, frequency
    python model_benchmark.py --n 50000          # subsample
    python model_benchmark.py --task severity    # (requires ClaimAmount column)
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import warnings
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import rustystats as rs
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
import xgboost as xgb
import cvxpy as cp
import polars as pl

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from aglm import cva_aglm  # noqa: E402
from xgb_weighted import fit_xgb_weighted, predict_xgb_weighted  # noqa: E402
from benchmark_report import BenchmarkReport  # noqa: E402

# ---------------------------------------------------------------------------
# Global defaults (can be overridden in main())
# ---------------------------------------------------------------------------

NBIN_MAX   = 40
N_ALPHAS   = 10
ALPHA_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CV_FOLDS   = 5

COLORS = {
    "GLM":        "#6C8EAD",
    "RegGLM":     "#3A7CA5",
    "AGLM-Lin":   "#A8C0D6",
    "AGLM-Lvar":  "#1A3C6E",
    "GAM":        "#5C8A5C",
    "GBM":        "#C0392B",
    "CatBoost":   "#8E44AD",
    "XGBoost":    "#E74C3C",
    "DerivLasso": "#E67E22",
}

# Family auto-selected by task; can be overridden
TASK_FAMILY: dict[str, str] = {
    "frequency": "poisson",
    "severity":  "gamma",
}

# XGBoost objective & metric per family
_XGB_OBJ = {
    "poisson": "count:poisson",
    "gamma":   "reg:gamma",
    "tweedie": "reg:tweedie",
}
_XGB_METRIC = {
    "poisson": "poisson-nloglik",
    "gamma":   "gamma-nloglik",
    "tweedie": "tweedie-nloglik",
}

# CatBoost loss per family
_CB_LOSS = {
    "poisson": "Poisson",
    "gamma":   "Tweedie:variance_power=2.0",
    "tweedie": "Tweedie:variance_power=1.5",
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance (2× convention, matches R/sklearn/paper)."""
    y  = np.asarray(y_true, float)
    mu = np.maximum(np.asarray(y_pred, float), 1e-12)
    return float(2.0 * np.mean(
        np.where(y > 0, y * np.log(y / mu), 0.0) - y + mu
    ))


def _gamma_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Gamma deviance (2× convention)."""
    y  = np.maximum(np.asarray(y_true, float),  1e-12)
    mu = np.maximum(np.asarray(y_pred, float), 1e-12)
    return float(2.0 * np.mean(np.log(mu / y) + y / mu - 1.0))


def _weighted_gini(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   exposure: np.ndarray) -> float:
    order  = np.argsort(y_pred)
    y_s    = y_true[order]
    e_s    = exposure[order]
    cum_e  = np.cumsum(e_s) / e_s.sum()
    cum_l  = np.cumsum(y_s * e_s) / (y_s * e_s).sum()
    return float(1.0 - 2.0 * np.trapezoid(cum_l, cum_e))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    v, w = v[mask], w[mask]
    order = np.argsort(v)
    cdf   = np.cumsum(w[order]) / np.sum(w[order])
    return float(v[order][np.searchsorted(cdf, 0.5, side="left")])


def compute_metrics(
    name:          str,
    y_count:       np.ndarray,
    y_pred_count:  np.ndarray,
    exposure:      np.ndarray,
    task:          str = "frequency",
) -> dict:
    """Full metric suite for one model.  All predictions and actuals are in
    *count/amount* space; rates are derived internally for MSE/MAE/averages."""
    y   = np.asarray(y_count,      float)
    mu  = np.maximum(np.asarray(y_pred_count, float), 1e-12)
    exp = np.asarray(exposure,     float)

    rate_true = y  / np.maximum(exp, 1e-9)
    rate_pred = mu / np.maximum(exp, 1e-9)

    if task == "frequency":
        dev = _poisson_deviance(y, mu)
    else:
        dev = _gamma_deviance(y, mu)

    mse  = float(np.mean((rate_true - rate_pred) ** 2))
    mae  = float(np.mean(np.abs(rate_true - rate_pred)))
    rmse = float(np.sqrt(mse))

    # AUC only meaningful for frequency (binary: any claim)
    if task == "frequency":
        binary = (y > 0).astype(int)
        auc = (float(roc_auc_score(binary, rate_pred))
               if 0 < binary.sum() < len(binary) else float("nan"))
    else:
        auc = float("nan")

    gini = _weighted_gini(y, mu, exp)

    w          = exp / exp.sum()
    avg_pred   = float(np.sum(rate_pred * w))
    med_pred   = _weighted_median(rate_pred, w)
    avg_actual = float(np.sum(rate_true * w))
    med_actual = _weighted_median(rate_true, w)

    return {
        "Model":        name,
        "Deviance":     dev,      # Poisson or Gamma depending on task
        "MSE":          mse,
        "MAE":          mae,
        "RMSE":         rmse,
        "AUC":          auc,
        "Gini":         gini,
        "Avg Pred":     avg_pred,
        "Median Pred":  med_pred,
        "Avg Actual":   avg_actual,
        "Median Actual": med_actual,
    }


# ---------------------------------------------------------------------------
# Shared feature utilities
# ---------------------------------------------------------------------------

def _aglm_features(
    df:             pd.DataFrame,
    numeric_cols:   list[str],
    categorical_cols: list[str],
    exposure_col:   str,
) -> pd.DataFrame:
    """Build the feature DataFrame for cva_aglm: log-exposure prepended."""
    x = df[numeric_cols + categorical_cols].copy()
    x.insert(0, "__log_exposure__",
             np.log(np.maximum(df[exposure_col].values, 1e-9)))
    return x


def load_and_split(
    df:          pd.DataFrame,
    seed:        int | None = 42,
    sample_size: int | None = None,
    claim_col:   str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stratified train/test split (75/25) with a printed summary table."""
    if sample_size is not None:
        df = df.sample(sample_size, random_state=seed).reset_index(drop=True)

    if claim_col is not None:
        strat = (df[claim_col] > 0).astype(int)
        train, test = train_test_split(
            df, test_size=0.25, random_state=seed, stratify=strat
        )
    else:
        train, test = train_test_split(df, test_size=0.25, random_state=seed)

    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    header = f"{'Dataset':<12} {'Records':>10} {'% of Total':>11}"
    if claim_col is not None:
        header += f" {'Claim Rate':>12}"
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    summary: dict = {}
    for label, subset, total in [("Original", df, len(df)),
                                   ("Train",   train, len(df)),
                                   ("Test",    test,  len(df))]:
        n   = len(subset)
        pct = n / total * 100
        entry: dict = {"n": n, "pct_of_total": round(pct, 1)}
        row = f"{label:<12} {n:>10,} {pct:>10.1f}%"
        if claim_col is not None:
            cr = subset[claim_col].mean()
            entry["claim_rate"] = round(cr, 6)
            row += f" {cr:>11.4f}"
        summary[label] = entry
        print(row)
    print(sep)
    return train, test, summary


# ---------------------------------------------------------------------------
# 1/9  GLM — plain GLM (rustystats IRLS)
# ---------------------------------------------------------------------------

def fit_glm(
    train:            pd.DataFrame,
    response_col:     str,
    exposure_col:     str,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    family:           str,
) -> dict:
    print(f"\n[GLM] Plain {family} GLM (rustystats IRLS) ...", flush=True)
    t0 = time.time()

    terms = {col: {"type": "linear"}      for col in numeric_cols}
    terms.update({col: {"type": "categorical"} for col in categorical_cols})

    result = rs.glm_dict(
        response=response_col,
        terms=terms,
        data=train,
        family=family,
        offset=exposure_col,
        weights=None,
        seed=42,
    ).fit()

    elapsed = time.time() - t0
    print(f"  ok {elapsed:.1f}s | deviance={result.deviance:.4f} "
          f"| converged={result.converged} | iters={result.iterations}")
    return {
        "model":      result,
        "timer":      elapsed,
        "deviance":   result.deviance,
        "converged":  result.converged,
    }


def predict_glm(res: dict, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(res["model"].predict(df))


# ---------------------------------------------------------------------------
# 2/9  RegGLM — regularised GLM (rustystats elastic-net CV)
# ---------------------------------------------------------------------------

def fit_reg_glm(
    train:            pd.DataFrame,
    response_col:     str,
    exposure_col:     str,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    family:           str,
    cv_folds:         int = CV_FOLDS,
) -> dict:
    print(f"\n[RegGLM] Regularised {family} GLM (rustystats elastic-net CV) ...",
          flush=True)
    t0 = time.time()

    terms = {col: {"type": "linear"}      for col in numeric_cols}
    terms.update({col: {"type": "categorical"} for col in categorical_cols})

    result = rs.glm_dict(
        response=response_col,
        terms=terms,
        data=train,
        family=family,
        offset=exposure_col,
        weights=None,
        seed=42,
    ).fit(
        regularization="elastic_net",
        selection="min",
        cv=cv_folds,
        cv_seed=42,
    )

    elapsed = time.time() - t0
    print(f"  alpha={result.alpha:.6f}  "
          f"non-zero={result.n_nonzero()}/{len(result.params)}")
    print(f"  ok {elapsed:.1f}s")
    return {
        "model": result,
        "timer": elapsed,
        "alpha": result.alpha,
    }


def predict_reg_glm(res: dict, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(res["model"].predict(df))


# ---------------------------------------------------------------------------
# 3/9  AGLM-Lin — AGLM without basis expansion
# ---------------------------------------------------------------------------

def fit_aglm_linear(
    train:            pd.DataFrame,
    response_col:     str,
    exposure_col:     str,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    family:           str,
    alpha_grid:       np.ndarray = ALPHA_GRID,
    cv_folds:         int        = CV_FOLDS,
    n_alphas:         int        = N_ALPHAS,
    nbin_max:         int        = NBIN_MAX,
) -> dict:
    print("\n[AGLM-Lin] AGLM without basis expansion (cva_aglm elastic-net CV) ...",
          flush=True)
    t0 = time.time()

    x = _aglm_features(train, numeric_cols, categorical_cols, exposure_col)
    y = train[response_col].values.astype(float)

    model = cva_aglm(
        x, y,
        alpha_grid=alpha_grid,
        nfolds=cv_folds,
        family=family,
        lambda_grid=np.logspace(-3, 1, n_alphas),
        add_linear_columns=True,
        use_lvar=False,
        od_type_of_quantitatives="N",
        add_od_columns_of_qualitatives=False,
        nbin_max=nbin_max,
    )

    bm     = model.best_model
    coef_s = bm.coef(with_names=True)
    n_nz   = int((coef_s.abs() > 1e-6).sum())
    elapsed = time.time() - t0

    print(f"  best alpha={model.best_alpha:.2f}  lambda={bm.lambda_:.5f}  "
          f"nnz={n_nz}/{len(coef_s)}")
    print(f"  ok {elapsed:.1f}s")
    return {
        "model":          model,
        "timer":          elapsed,
        "numeric_cols":   numeric_cols,
        "categorical_cols": categorical_cols,
        "exposure_col":   exposure_col,
        "best_alpha":     model.best_alpha,
        "best_lambda":    bm.lambda_,
    }


def predict_aglm_linear(res: dict, df: pd.DataFrame) -> np.ndarray:
    x = _aglm_features(df, res["numeric_cols"], res["categorical_cols"],
                        res["exposure_col"])
    return res["model"].best_model.predict(x)


# ---------------------------------------------------------------------------
# 4/9  AGLM-Lvar — Full AGLM with L-variable basis
# ---------------------------------------------------------------------------

def fit_aglm_lvar(
    train:            pd.DataFrame,
    response_col:     str,
    exposure_col:     str,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    family:           str,
    alpha_grid:       np.ndarray = ALPHA_GRID,
    cv_folds:         int        = CV_FOLDS,
    n_alphas:         int        = N_ALPHAS,
    nbin_max:         int        = NBIN_MAX,
) -> dict:
    print(f"\n[AGLM-Lvar] Full AGLM with L-variable basis (nbin={nbin_max}) ...",
          flush=True)
    t0 = time.time()

    x = _aglm_features(train, numeric_cols, categorical_cols, exposure_col)
    y = train[response_col].values.astype(float)

    model = cva_aglm(
        x, y,
        alpha_grid=alpha_grid,
        nfolds=cv_folds,
        family=family,
        lambda_grid=np.logspace(-3, 1, n_alphas),
        add_linear_columns=True,
        use_lvar=True,
        add_od_columns_of_qualitatives=True,
        nbin_max=nbin_max,
    )

    bm     = model.best_model
    coef_s = bm.coef(with_names=True)
    n_nz   = int((coef_s.abs() > 1e-6).sum())
    elapsed = time.time() - t0

    print(f"  best alpha={model.best_alpha:.2f}  lambda={bm.lambda_:.5f}  "
          f"nnz={n_nz}/{len(coef_s)}  (total basis: {len(coef_s)})")
    print(f"  ok {elapsed:.1f}s")
    return {
        "model":          model,
        "timer":          elapsed,
        "numeric_cols":   numeric_cols,
        "categorical_cols": categorical_cols,
        "exposure_col":   exposure_col,
        "best_alpha":     model.best_alpha,
        "best_lambda":    bm.lambda_,
        "n_basis":        len(coef_s),
    }


def predict_aglm_lvar(res: dict, df: pd.DataFrame) -> np.ndarray:
    x = _aglm_features(df, res["numeric_cols"], res["categorical_cols"],
                        res["exposure_col"])
    return res["model"].best_model.predict(x)


# ---------------------------------------------------------------------------
# 5/9  GAM — PoissonGAM / GammaGAM (pygam)
# ---------------------------------------------------------------------------

def fit_gam(
    train:            pd.DataFrame,
    response_col:     str,
    exposure_col:     str,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    family:           str,
) -> dict:
    from pygam import GammaGAM, PoissonGAM, f, s
    gam_cls = PoissonGAM if family == "poisson" else GammaGAM
    print(f"\n[GAM] {gam_cls.__name__} spline + factor terms (pygam) ...",
          flush=True)
    t0 = time.time()

    cat_maps: dict[str, dict] = {}

    def _gam_mat(df: pd.DataFrame) -> np.ndarray:
        parts = [df[numeric_cols].values.astype(float)]
        for col in categorical_cols:
            if col not in cat_maps:
                cats = sorted(df[col].unique())
                cat_maps[col] = {c: i for i, c in enumerate(cats)}
            parts.append(
                np.array([cat_maps[col].get(v, 0) for v in df[col]],
                         float).reshape(-1, 1)
            )
        return np.hstack(parts)

    x      = _gam_mat(train)
    n_num  = len(numeric_cols)
    n_cat  = len(categorical_cols)

    # Build term spec dynamically
    if n_num > 0:
        terms = s(0)
        for i in range(1, n_num):
            terms += s(i)
        for i in range(n_num, n_num + n_cat):
            terms += f(i)
    else:
        terms = f(0)
        for i in range(1, n_cat):
            terms += f(i)

    gam = gam_cls(terms)
    gam.gridsearch(
        x, train[response_col].values,
        weights=train[exposure_col].values,
        progress=False,
    )

    elapsed = time.time() - t0
    print(f"  ok {elapsed:.1f}s")
    return {
        "model":       gam,
        "cat_maps":    cat_maps,
        "gam_mat":     _gam_mat,
        "exposure_col": exposure_col,
        "timer":       elapsed,
    }


def predict_gam(res: dict, df: pd.DataFrame) -> np.ndarray:
    return res["model"].predict(res["gam_mat"](df)) * df[res["exposure_col"]].values


# ---------------------------------------------------------------------------
# 6/9  GBM — LightGBM (CV for n_estimators)
# ---------------------------------------------------------------------------

def _lgb_X(
    df:               pd.DataFrame,
    numeric_cols:     list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    x = df[numeric_cols + categorical_cols].copy()
    for col in categorical_cols:
        x[col] = x[col].astype("category")
    return x


def fit_gbm(
    train:            pd.DataFrame,
    response_col:     str,
    exposure_col:     str,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    family:           str,
    cv_folds:         int = CV_FOLDS,
) -> dict:
    print(f"\n[GBM] LightGBM {family} (log-exposure offset, CV for n_estimators) ...",
          flush=True)
    t0 = time.time()

    freq   = train[response_col].values / np.maximum(train[exposure_col].values, 1e-9)
    w      = train[exposure_col].values

    dtrain = lgb.Dataset(
        _lgb_X(train, numeric_cols, categorical_cols),
        label=freq, weight=w, free_raw_data=False,
    )
    params = {
        "objective":        family,
        "metric":           family,
        "learning_rate":    0.01,
        "num_leaves":       63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq":     5,
        "verbose":          -1,
        "n_jobs":           -1,
    }
    cv_res = lgb.cv(
        params, dtrain,
        num_boost_round=500, nfold=cv_folds,
        stratified=False,
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    # Pick the valid metric key dynamically
    metric_key = next(
        (k for k in cv_res if k.startswith("valid ") and k.endswith("-mean")),
        None,
    )
    best_n = len(cv_res[metric_key]) if metric_key else 100

    booster = lgb.train(
        params, dtrain,
        num_boost_round=best_n,
        callbacks=[lgb.log_evaluation(-1)],
    )

    elapsed = time.time() - t0
    print(f"  best n_estimators={best_n}  ok {elapsed:.1f}s")
    return {
        "model":          booster,
        "numeric_cols":   numeric_cols,
        "categorical_cols": categorical_cols,
        "exposure_col":   exposure_col,
        "best_n":         best_n,
        "timer":          elapsed,
    }


def predict_gbm(res: dict, df: pd.DataFrame) -> np.ndarray:
    return (res["model"].predict(
                _lgb_X(df, res["numeric_cols"], res["categorical_cols"]))
            * df[res["exposure_col"]].values)


# ---------------------------------------------------------------------------
# 7/9  CatBoost — native categorical support
# ---------------------------------------------------------------------------

def fit_catboost(
    train:            pd.DataFrame,
    response_col:     str,
    exposure_col:     str,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    family:           str,
    cv_folds:         int = CV_FOLDS,
) -> dict:
    from catboost import CatBoostRegressor, Pool, cv as cb_cv
    loss_fn = _CB_LOSS.get(family, "Poisson")
    print(f"\n[CatBoost] {loss_fn} regressor (native categoricals) ...",
          flush=True)
    t0 = time.time()

    feature_cols = numeric_cols + categorical_cols
    x = train[feature_cols].copy()
    for col in categorical_cols:
        x[col] = x[col].astype(str)

    log_exp = np.log(np.maximum(train[exposure_col].values, 1e-9))
    pool = Pool(
        data=x,
        label=train[response_col].values.astype(float),
        baseline=log_exp,
        cat_features=categorical_cols,
    )

    model = CatBoostRegressor(
        loss_function=loss_fn,
        iterations=600,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3.0,
        min_data_in_leaf=50,
        subsample=0.9,
        colsample_bylevel=0.8,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=30,
    )
    cv_result = cb_cv(
        pool=pool,
        params=model.get_params(),
        fold_count=cv_folds,
        shuffle=True,
        partition_random_seed=42,
        plot=False,
        verbose=False,
    )

    # Find the test metric column dynamically
    test_col = next(
        (c for c in cv_result.columns
         if c.startswith("test-") and c.endswith("-mean")),
        None,
    )
    best_iter = (int(cv_result[test_col].idxmin()) + 1
                 if test_col else 200)
    print(f"  Best iteration from CV: {best_iter}")

    model.set_params(iterations=best_iter, early_stopping_rounds=None)
    model.fit(pool)

    elapsed = time.time() - t0
    print(f"  ok {elapsed:.1f}s")
    return {
        "model":          model,
        "feature_cols":   feature_cols,
        "categorical_cols": categorical_cols,
        "exposure_col":   exposure_col,
        "best_iter":      best_iter,
        "timer":          elapsed,
    }


def predict_catboost(res: dict, df: pd.DataFrame) -> np.ndarray:
    from catboost import Pool
    x = df[res["feature_cols"]].copy()
    for col in res["categorical_cols"]:
        x[col] = x[col].astype(str)
    log_exp = np.log(np.maximum(df[res["exposure_col"]].values, 1e-9))
    pool    = Pool(data=x, baseline=log_exp, cat_features=res["categorical_cols"])
    return res["model"].predict(pool)


# ---------------------------------------------------------------------------
# 8/9  XGBoost — CV for n_estimators, log-offset
# ---------------------------------------------------------------------------

def _xgb_X(
    df:               pd.DataFrame,
    numeric_cols:     list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    x = df[numeric_cols + categorical_cols].copy()
    for col in categorical_cols:
        x[col] = x[col].astype("category")
    return x


def fit_xgb(
    train:            pd.DataFrame,
    response_col:     str,
    exposure_col:     str,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    family:           str,
    cv_folds:         int = CV_FOLDS,
) -> dict:
    obj    = _XGB_OBJ.get(family, "count:poisson")
    metric = _XGB_METRIC.get(family, "poisson-nloglik")
    print(f"\n[XGBoost] {obj} (log-exposure offset, CV for n_estimators) ...",
          flush=True)
    t0 = time.time()

    log_exp = np.log(np.maximum(train[exposure_col].values, 1e-9))
    dtrain  = xgb.DMatrix(
        _xgb_X(train, numeric_cols, categorical_cols),
        label=train[response_col].values,
        enable_categorical=True,
    )
    dtrain.set_base_margin(log_exp)

    params = {
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
        params, dtrain,
        num_boost_round=500,
        nfold=cv_folds,
        stratified=False,
        early_stopping_rounds=20,
        callbacks=[xgb.callback.EvaluationMonitor(show_stdv=False, period=999)],
    )
    best_n = len(cv_result)

    booster = xgb.train(
        params, dtrain,
        num_boost_round=best_n,
        verbose_eval=False,
    )

    elapsed = time.time() - t0
    print(f"  best n_estimators={best_n}  ok {elapsed:.1f}s")
    return {
        "model":          booster,
        "numeric_cols":   numeric_cols,
        "categorical_cols": categorical_cols,
        "exposure_col":   exposure_col,
        "best_n":         best_n,
        "timer":          elapsed,
    }


def predict_xgb(res: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict counts: set log-offset at test time so predictions are on the
    same scale as response_col."""
    log_exp = np.log(np.maximum(df[res["exposure_col"]].values, 1e-9))
    dtest   = xgb.DMatrix(
        _xgb_X(df, res["numeric_cols"], res["categorical_cols"]),
        enable_categorical=True,
    )
    dtest.set_base_margin(log_exp)
    return res["model"].predict(dtest)


# ---------------------------------------------------------------------------
# 9/9  DerivLasso — derivative (fused) lasso GLM via CVXPY
# ---------------------------------------------------------------------------

def _fit_feature_engineer(
    df:               pd.DataFrame,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    n_bins:           int = 40,
    exposure_col:     str | None = None,
    other_label:      str = "Other",
) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    """
    Fit quantile bin edges and category top-N sets on *training* data.

    Numeric columns are quantile-binned; the outer edges are set to ±inf so
    that test values outside the training range always fall into the first or
    last bin rather than producing NaN.

    Categorical columns are capped to the top-(n_bins-1) levels by exposure
    (or frequency); all other levels map to *other_label*.

    Returns
    -------
    out          : transformed DataFrame
    binned       : list of new binned column names  (e.g. "VehAge_bin")
    capped       : list of new capped column names  (e.g. "VehBrand_cap")
    artifacts    : dict containing bin edges and category sets — pass to
                   _apply_feature_engineer at predict time
    """
    out = df.copy()
    binned:    list[str] = []
    capped:    list[str] = []
    bin_edges: dict[str, np.ndarray | None] = {}
    cat_tops:  dict[str, set] = {}

    for col in numeric_cols:
        new_col = f"{col}_bin"
        try:
            _, edges = pd.qcut(
                out[col].astype(float), q=n_bins,
                duplicates="drop", retbins=True,
            )
            # Make outer edges unbounded so every future value maps to a bin
            edges[0]  = -np.inf
            edges[-1] = np.inf
            bin_edges[col] = edges
            out[new_col] = pd.cut(
                out[col].astype(float), bins=edges, include_lowest=True
            ).astype(str)
        except ValueError:
            # Fewer unique values than n_bins — fall back to raw string
            bin_edges[col] = None
            out[new_col] = out[col].astype(str)
        binned.append(new_col)

    for col in categorical_cols:
        new_col = f"{col}_cap"
        series  = out[col].astype(str)
        if exposure_col is not None:
            by_exp = (out.groupby(series)[exposure_col].sum()
                      .sort_values(ascending=False))
        else:
            by_exp = series.value_counts()
        top = set(by_exp.head(n_bins - 1).index)
        cat_tops[col] = top
        out[new_col]  = series.where(series.isin(top), other=other_label)
        capped.append(new_col)

    artifacts = {
        "bin_edges":   bin_edges,
        "cat_tops":    cat_tops,
        "other_label": other_label,
    }
    return out, binned, capped, artifacts


def _apply_feature_engineer(
    df:               pd.DataFrame,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    artifacts:        dict,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Apply pre-fitted bin edges and category caps to *any* data (train or test).

    Values outside the training range are absorbed by the ±inf outer edges.
    Unseen categorical levels map to 'Other'.
    """
    out         = df.copy()
    bin_edges   = artifacts["bin_edges"]
    cat_tops    = artifacts["cat_tops"]
    other_label = artifacts.get("other_label", "Other")
    binned: list[str] = []
    capped: list[str] = []

    for col in numeric_cols:
        new_col = f"{col}_bin"
        edges   = bin_edges.get(col)
        if edges is not None:
            out[new_col] = pd.cut(
                out[col].astype(float), bins=edges, include_lowest=True
            ).astype(str)
            # Guard: NaN can still arise if edges somehow don't cover a value
            out[new_col] = out[new_col].fillna(other_label)
        else:
            out[new_col] = out[col].astype(str)
        binned.append(new_col)

    for col in categorical_cols:
        new_col = f"{col}_cap"
        series  = out[col].astype(str)
        top     = cat_tops.get(col, set())
        out[new_col] = series.where(series.isin(top), other=other_label)
        capped.append(new_col)

    return out, binned, capped


class _GenericDLDesignMatrix:
    """Generic design matrix for the fused-lasso GLM.

    *ordered_cols* — quantile-binned numeric columns (full dummies, no drop_first)
    *nominal_cols* — capped categorical columns (drop_first)
    """

    def __init__(self, ordered_cols: list[str], nominal_cols: list[str]):
        self.ordered_cols   = ordered_cols
        self.nominal_cols   = nominal_cols
        self.feature_groups_: dict[str, list[int]] = {}
        self.feature_names_:  list[str]            = []
        self.col_order_:      dict[str, list[str]] = {}
        self.n_features_:     int                  = 0

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        blocks, col = [np.ones((n, 1))], 1
        self.feature_groups_["intercept"] = [0]
        self.feature_names_.append("intercept")

        for var in self.ordered_cols:
            dum = pd.get_dummies(df[var], prefix=var,
                                  drop_first=False, dtype=float)
            k   = dum.shape[1]
            blocks.append(dum.values)
            self.feature_groups_[var] = list(range(col, col + k))
            self.feature_names_.extend(dum.columns.tolist())
            self.col_order_[var] = dum.columns.tolist()
            col += k

        for var in self.nominal_cols:
            dum = pd.get_dummies(df[var], prefix=var,
                                  drop_first=True, dtype=float)
            k   = dum.shape[1]
            blocks.append(dum.values)
            self.feature_groups_[var] = list(range(col, col + k))
            self.feature_names_.extend(dum.columns.tolist())
            self.col_order_[var] = dum.columns.tolist()
            col += k

        self.n_features_ = col
        X = np.hstack(blocks)
        print(f"  Design matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = [np.ones((len(df), 1))]
        for var in self.ordered_cols + self.nominal_cols:
            dum = pd.get_dummies(df[var], prefix=var,
                                  drop_first=False, dtype=float)
            for c in self.col_order_[var]:
                if c not in dum.columns:
                    dum[c] = 0.0
            blocks.append(dum[self.col_order_[var]].values)
        return np.hstack(blocks)


class _DerivativeLassoGLM:
    """Poisson/Gamma GLM with first-difference lasso penalty, solved via CVXPY."""

    def __init__(self, lam: float = 0.05, family: str = "poisson"):
        self.lam    = lam
        self.family = family
        self.coef_: np.ndarray | None = None
        self.status_: str = ""

    def fit(
        self,
        X:              np.ndarray,
        y:              np.ndarray,
        exposure:       np.ndarray | None = None,
        ordered_groups: dict | None       = None,
    ) -> "_DerivativeLassoGLM":
        n, p  = X.shape
        exp_  = np.ones(n) if exposure is None else np.asarray(exposure, float)
        log_e = np.log(np.maximum(exp_, 1e-12))

        beta = cp.Variable(p)
        eta  = X @ beta + log_e

        if self.family == "gamma":
            # Gamma deviance: 2 * sum(log(mu/y) + y/mu - 1)
            mu   = cp.exp(eta)
            loss = 2.0 * cp.sum(eta + cp.multiply(y, cp.exp(-eta)) - 1.0)
        else:
            # Poisson NLL: sum(exp(eta) - y * eta)
            loss = cp.sum(cp.exp(eta) - cp.multiply(y, eta))

        diffs = []
        if ordered_groups:
            for idx in ordered_groups.values():
                if len(idx) > 1:
                    b_v = beta[idx]
                    for j in range(len(idx) - 1):
                        diffs.append(cp.abs(b_v[j + 1] - b_v[j]))
        penalty = cp.sum(cp.hstack(diffs)) if diffs else cp.Constant(0)

        prob = cp.Problem(cp.Minimize(loss + self.lam * penalty))
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-4)

        self.status_ = prob.status
        if beta.value is None:
            raise RuntimeError(f"DerivLasso solver failed: {prob.status}")
        self.coef_ = beta.value
        return self

    def predict(self,
                X:        np.ndarray,
                exposure: np.ndarray | None = None) -> np.ndarray:
        exp_  = np.ones(len(X)) if exposure is None else np.asarray(exposure, float)
        return np.exp(X @ self.coef_ + np.log(np.maximum(exp_, 1e-12)))


def fit_derivative_lasso(
    train:            pd.DataFrame,
    response_col:     str,
    exposure_col:     str,
    numeric_cols:     list[str],
    categorical_cols: list[str],
    family:           str,
    n_bins:           int = NBIN_MAX,
    cv_folds:         int = CV_FOLDS,
) -> dict:
    print(f"\n[DerivLasso] Fused lasso {family} GLM via CVXPY ...", flush=True)
    t0 = time.time()

    eng, binned, capped, artifacts = _fit_feature_engineer(
        train, numeric_cols, categorical_cols,
        n_bins=n_bins, exposure_col=exposure_col,
    )

    dm             = _GenericDLDesignMatrix(binned, capped)
    X              = dm.fit_transform(eng)
    y              = train[response_col].values.astype(float)
    exp_           = train[exposure_col].values.astype(float)
    ordered_groups = {v: dm.feature_groups_[v] for v in binned}

    lam_grid = [0.005, 0.02, 0.05, 0.10, 0.25, 0.50]
    kf       = KFold(n_splits=cv_folds, shuffle=True, random_state=1)
    print(f"  Lambda CV over {lam_grid} ({cv_folds}-fold) ...")

    cv_devs: list[float] = []
    for lam in lam_grid:
        fold_devs = []
        for tr_i, va_i in kf.split(X):
            m = _DerivativeLassoGLM(lam=lam, family=family)
            m.fit(X[tr_i], y[tr_i], exposure=exp_[tr_i],
                  ordered_groups=ordered_groups)
            pred = m.predict(X[va_i], exp_[va_i])
            if family == "gamma":
                fold_devs.append(_gamma_deviance(y[va_i], pred))
            else:
                fold_devs.append(_poisson_deviance(y[va_i], pred))
        mean_dev = float(np.mean(fold_devs))
        cv_devs.append(mean_dev)
        print(f"    lambda={lam:.3f}  cv_dev={mean_dev:.6f}")

    best_lam = lam_grid[int(np.argmin(cv_devs))]
    print(f"  Best lambda: {best_lam}")

    best_m = _DerivativeLassoGLM(lam=best_lam, family=family)
    best_m.fit(X, y, exposure=exp_, ordered_groups=ordered_groups)
    print(f"  Solver status: {best_m.status_}")

    elapsed = time.time() - t0
    print(f"  ok {elapsed:.1f}s")
    return {
        "model":          best_m,
        "dm":             dm,
        "artifacts":      artifacts,   # fitted bin edges + category sets
        "numeric_cols":   numeric_cols,
        "categorical_cols": categorical_cols,
        "exposure_col":   exposure_col,
        "best_lam":       best_lam,
        "ordered_groups": ordered_groups,
        "timer":          elapsed,
    }


def predict_derivative_lasso(res: dict, df: pd.DataFrame) -> np.ndarray:
    eng, _, _ = _apply_feature_engineer(
        df, res["numeric_cols"], res["categorical_cols"],
        artifacts=res["artifacts"],   # uses training bin edges — no re-fitting
    )
    X   = res["dm"].transform(eng)
    exp = df[res["exposure_col"]].values.astype(float)
    return res["model"].predict(X, exp)


# ---------------------------------------------------------------------------
# HTML report — generated by benchmark_report.BenchmarkReport
# ---------------------------------------------------------------------------

# (Removed: _HTML_TEMPLATE and generate_html_report moved to benchmark_report.py)



# ---------------------------------------------------------------------------
# (HTML template and generate_html_report removed — see benchmark_report.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(
    df:               pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols:     list[str],
    response_col:     str,
    exposure_col:     str,
    task:             Literal["frequency", "severity"] = "frequency",
    family:           str | None = None,
    sample_size:      int | None = None,
    seed:             int        = 42,
    nbin_max:         int        = NBIN_MAX,
    n_alphas:         int        = N_ALPHAS,
    alpha_grid:       np.ndarray = ALPHA_GRID,
    cv_folds:         int        = CV_FOLDS,
    run_gam:           bool       = True,
    run_gbm:           bool       = True,
    run_catboost:      bool       = True,
    run_xgb:           bool       = True,
    run_xgb_weighted:  bool       = True,
    run_deriv_lasso:   bool       = True,
    run_shap:          bool       = True,
    html_output:       str | pathlib.Path | None = None,
    dataset_title:     str        = "Dataset",
) -> dict:
    """
    Fit all models, compute metrics, print a summary table, and write an
    interactive HTML report (benchmark_report.BenchmarkReport).

    Parameters
    ----------
    df               : input DataFrame (already cleaned)
    categorical_cols : categorical feature column names
    numeric_cols     : numeric feature column names
    response_col     : target column (counts for frequency, amounts for severity)
    exposure_col     : exposure column (policy time for frequency, claim count
                       for severity)
    task             : "frequency" or "severity"
    family           : GLM family; defaults to "poisson" / "gamma" based on task
    sample_size      : subsample the data before splitting (None = use all)
    seed             : random seed
    nbin_max         : max bins for AGLM / DerivLasso
    n_alphas         : lambda grid size for AGLM
    alpha_grid       : elastic-net alpha candidates
    cv_folds         : cross-validation folds
    run_*            : toggle individual models on/off
    run_xgb_weighted : XGBoost variant using rate label + exposure weights (no offset)
    run_shap         : compute SHAP values for tree models (needs shap installed)
    html_output      : path for the HTML report (None = auto-name)
    dataset_title    : title string for the HTML report

    Returns
    -------
    dict with keys: "metrics_table", "predictions", "train_predictions",
                    "train_times", "models", "model_objects", "train", "test"
    """
    if family is None:
        family = TASK_FAMILY.get(task, "poisson")

    print("=" * 70)
    print(f"Model Benchmark  —  task={task}  family={family}")
    print(f"  Features: {numeric_cols + categorical_cols}")
    print(f"  Response: {response_col}   Exposure: {exposure_col}")
    print("=" * 70)
    t_total = time.time()

    # ── split ────────────────────────────────────────────────────────────────
    train, test, summary = load_and_split(
        df, seed=seed, sample_size=sample_size,
        claim_col=response_col if task == "frequency" else None,
    )

    print()
    print("─" * 70)
    print(f"Fitting models  nbin={nbin_max}  n_alphas={n_alphas}"
          f"  cv={cv_folds}  alpha_grid={alpha_grid.tolist()}")
    print("─" * 70)

    # ── fit all models ───────────────────────────────────────────────────────
    glm_r       = fit_glm(train, response_col, exposure_col,
                          numeric_cols, categorical_cols, family)
    reg_glm_r   = fit_reg_glm(train, response_col, exposure_col,
                               numeric_cols, categorical_cols, family, cv_folds)
    aglm_lin_r  = fit_aglm_linear(train, response_col, exposure_col,
                                   numeric_cols, categorical_cols, family,
                                   alpha_grid, cv_folds, n_alphas, nbin_max)
    aglm_lvar_r = fit_aglm_lvar(train, response_col, exposure_col,
                                  numeric_cols, categorical_cols, family,
                                  alpha_grid, cv_folds, n_alphas, nbin_max)

    gam_r       = (fit_gam(train, response_col, exposure_col,
                           numeric_cols, categorical_cols, family)
                   if run_gam         else None)
    gbm_r       = (fit_gbm(train, response_col, exposure_col,
                           numeric_cols, categorical_cols, family, cv_folds)
                   if run_gbm         else None)
    catboost_r  = (fit_catboost(train, response_col, exposure_col,
                                numeric_cols, categorical_cols, family, cv_folds)
                   if run_catboost    else None)
    xgb_r       = (fit_xgb(train, response_col, exposure_col,
                           numeric_cols, categorical_cols, family, cv_folds)
                   if run_xgb          else None)
    xgb_w_r     = (fit_xgb_weighted(train, response_col, exposure_col,
                                    numeric_cols, categorical_cols, family, cv_folds)
                   if run_xgb_weighted else None)
    deriv_r     = (fit_derivative_lasso(train, response_col, exposure_col,
                                        numeric_cols, categorical_cols, family,
                                        nbin_max, cv_folds)
                   if run_deriv_lasso  else None)

    # ── prediction dispatch ──────────────────────────────────────────────────
    predict_fns: dict[str, Any] = {
        "GLM":       (glm_r,      predict_glm),
        "RegGLM":    (reg_glm_r,  predict_reg_glm),
        "AGLM-Lin":  (aglm_lin_r, predict_aglm_linear),
        "AGLM-Lvar": (aglm_lvar_r,predict_aglm_lvar),
    }
    if gam_r:      predict_fns["GAM"]        = (gam_r,      predict_gam)
    if gbm_r:      predict_fns["GBM"]        = (gbm_r,      predict_gbm)
    if catboost_r: predict_fns["CatBoost"]   = (catboost_r, predict_catboost)
    if xgb_r:      predict_fns["XGBoost"]    = (xgb_r,      predict_xgb)
    if xgb_w_r:    predict_fns["XGBoost-W"]  = (xgb_w_r,   predict_xgb_weighted)
    if deriv_r:    predict_fns["DerivLasso"] = (deriv_r,    predict_derivative_lasso)

    # ── compute metrics ──────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("Hold-out test metrics")
    print("─" * 70)

    rows:              list[dict]       = []
    predictions:       dict[str, Any]  = {}
    train_predictions: dict[str, Any]  = {}
    train_times:       dict[str, float] = {}

    for name, (res, pred_fn) in predict_fns.items():
        mu = pred_fn(res, test)
        predictions[name]       = mu
        train_predictions[name] = pred_fn(res, train)   # needed for profit matrix
        train_times[name]       = float(res.get("timer", 0))
        row = compute_metrics(name, test[response_col].values, mu,
                              test[exposure_col].values, task)
        rows.append(row)
        dev_label = "Poisson Dev" if task == "frequency" else "Gamma Dev"
        print(f"  {name:<12}: {dev_label}={row['Deviance']:.6f}  "
              f"mse={row['MSE']:.2e}  mae={row['MAE']:.6f}  "
              f"auc={row['AUC']:.4f}  gini={row['Gini']:.4f}  "
              f"avg_pred={row['Avg Pred']:.5f}")

    metrics_table = (
        pd.DataFrame(rows)
        .sort_values("Deviance")
        .reset_index(drop=True)
    )

    deviance_label = "Poisson Deviance" if task == "frequency" else "Gamma Deviance"
    print()
    print("─" * 120)
    hdr = (f"  {'Model':<12}  {deviance_label:>16}  {'MSE':>12}  {'MAE':>10}  "
           f"{'RMSE':>10}  {'AUC':>7}  {'Gini':>7}  {'Avg Pred':>10}  {'Avg Actual':>10}")
    print(hdr)
    print("  " + "─" * 116)
    for _, r in metrics_table.iterrows():
        auc_str = f"{r['AUC']:>7.4f}" if not np.isnan(r['AUC']) else f"{'—':>7}"
        print(
            f"  {r['Model']:<12}  {r['Deviance']:>16.7f}  "
            f"{r['MSE']:>12.4e}  {r['MAE']:>10.7f}  "
            f"{r['RMSE']:>10.7f}  {auc_str}  {r['Gini']:>7.4f}  "
            f"{r['Avg Pred']:>10.6f}  {r['Avg Actual']:>10.6f}"
        )
    print("─" * 120)

    # ── Collect model objects for SHAP ──────────────────────────────────────
    model_objects: dict[str, dict] = {}
    local_res = {
        "GLM":        glm_r,
        "RegGLM":     reg_glm_r,
        "AGLM-Lin":   aglm_lin_r,
        "AGLM-Lvar":  aglm_lvar_r,
    }
    if gam_r:      local_res["GAM"]        = gam_r
    if gbm_r:      local_res["GBM"]        = gbm_r
    if catboost_r: local_res["CatBoost"]   = catboost_r
    if xgb_r:      local_res["XGBoost"]    = xgb_r
    if xgb_w_r:    local_res["XGBoost-W"]  = xgb_w_r
    if deriv_r:    local_res["DerivLasso"] = deriv_r
    for name in predict_fns:
        if name in local_res:
            model_objects[name] = local_res[name]

    # ── HTML report ──────────────────────────────────────────────────────────
    if html_output is None:
        html_output = pathlib.Path(__file__).parent / "model_benchmark_report.html"
    print("\nGenerating HTML report ...")

    results_for_report = {
        "metrics_table": metrics_table,
        "predictions":   predictions,
        "train_times":   train_times,
        "models":        list(predict_fns.keys()),
    }
    BenchmarkReport(
        results          = results_for_report,
        train            = train,
        test             = test,
        numeric_cols     = numeric_cols,
        categorical_cols = categorical_cols,
        response_col     = response_col,
        exposure_col     = exposure_col,
        task             = task,
        family           = family,
        title            = dataset_title,
    ).generate(
        output_path   = html_output,
        model_objects = model_objects if run_shap else None,
        train_preds   = train_predictions,
    )

    print(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} min")
    print("=" * 70)

    return {
        "metrics_table":     metrics_table,
        "predictions":       predictions,
        "train_predictions": train_predictions,
        "train_times":       train_times,
        "models":            list(predict_fns.keys()),
        "model_objects":     model_objects,
        "train":             train,
        "test":              test,
    }


# ---------------------------------------------------------------------------
# Run configuration — edit these values and press Run (F5) in VS Code
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── What to model ────────────────────────────────────────────────────────
    TASK        = "frequency"   # "frequency" (Poisson) or "severity" (Gamma)
    SAMPLE_SIZE = None        # int to subsample, or None for the full ~678k rows

    # ── Cross-validation & regularisation ────────────────────────────────────
    CV_FOLDS_RUN  = 5
    NBIN_MAX_RUN  = 40
    N_ALPHAS_RUN  = 10

    # ── Toggle models on/off (set False to skip slow ones during development) ─
    RUN_GAM          = True
    RUN_GBM          = True
    RUN_CATBOOST     = True
    RUN_XGB          = True
    RUN_XGB_WEIGHTED = True    # XGBoost-W: rate label + exposure weights (no offset)
    RUN_DERIV_LASSO  = True
    RUN_SHAP         = True    # SHAP for tree models (needs: pip install shap)

    # ── Dataset columns (freMTPL2freq defaults — change for your own data) ───
    NUMERIC_COLS     = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    CATEGORICAL_COLS = ["VehBrand", "VehGas", "Area", "Region"]
    RESPONSE_COL     = "ClaimNb"
    EXPOSURE_COL     = "Exposure"

    # ────────────────────────────────────────────────────────────────────────
    # Data loading — freMTPL2freq from OpenML
    # Replace the block below with your own pd.read_csv(...) if using
    # a different dataset.
    # ────────────────────────────────────────────────────────────────────────
    print("Loading freMTPL2freq from OpenML ...")
    raw = fetch_openml(data_id=41214, as_frame=True, parser="pandas")["data"]

    for col in ["ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype("Int64")
    for col in ["Exposure", "Density"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    for col in ["VehBrand", "VehGas", "Area", "Region"]:
        raw[col] = raw[col].astype(str).str.strip()

    before = len(raw)
    raw = raw.dropna(subset=["ClaimNb", "VehPower", "VehAge", "DrivAge",
                               "BonusMalus", "Exposure", "Density",
                               "VehBrand", "VehGas", "Area", "Region"])
    print(f"  Dropped {before - len(raw):,} rows with missing values")
    raw["Exposure"] = raw["Exposure"].clip(lower=1e-4)
    raw["ClaimNb"]  = raw["ClaimNb"].astype(int).clip(upper=4)
    raw = raw.drop(columns=["IDpol"], errors="ignore")

    # ── Run ──────────────────────────────────────────────────────────────────
    main(
        df               = raw,
        categorical_cols = CATEGORICAL_COLS,
        numeric_cols     = NUMERIC_COLS,
        response_col     = RESPONSE_COL,
        exposure_col     = EXPOSURE_COL,
        task             = TASK,
        sample_size      = SAMPLE_SIZE,
        nbin_max         = NBIN_MAX_RUN,
        n_alphas         = N_ALPHAS_RUN,
        alpha_grid       = ALPHA_GRID,
        cv_folds         = CV_FOLDS_RUN,
        run_gam          = RUN_GAM,
        run_gbm          = RUN_GBM,
        run_catboost     = RUN_CATBOOST,
        run_xgb          = RUN_XGB,
        run_xgb_weighted = RUN_XGB_WEIGHTED,
        run_deriv_lasso  = RUN_DERIV_LASSO,
        run_shap         = RUN_SHAP,
        dataset_title    = "freMTPL2freq",
    )
