"""
fremtpl2_experiment.py
======================
Replication of Fujita et al. (2020) Section 5 numerical experiments.

Fits eight models on the freMTPL2freq motor-insurance frequency dataset:

  1. GLM          — plain Poisson GLM (statsmodels IRLS, log-exposure offset)
  2. RegGLM       — regularised Poisson GLM; sklearn PoissonRegressor + GridSearchCV
                    over L2 penalty strength.  Equivalent to R's cv.glmnet(family=
                    "poisson", alpha=0) on a standard one-hot design matrix.
  3. AGLM-Lin     — AGLM without basis expansion (linear + one-hot only).
                    Uses the RustyStats IRLS engine with elastic-net CV over both
                    lambda and alpha.  Comparable to RegGLM but with L1/L2 mixing.
  4. AGLM-Lvar    — Full AGLM with L-variable basis (Fujita et al. Table 8 "AGLM").
                    Augments the design matrix with |x - tk| tent functions before
                    elastic-net regularisation.
  5. GAM          — PoissonGAM, spline + factor terms (pygam)
  6. GBM          — LightGBM Poisson (lgb.cv for n_estimators)
  7. CatBoost     — CatBoost Poisson regressor (native ordered categorical encoding)
  8. DerivLasso   — Derivative (fused) Lasso GLM via CVXPY (Akur8-style)

Metrics reported
----------------
  - Poisson Deviance  — Eq. 11 from Fujita et al. (2x convention)
  - MSE               — mean squared error on claim frequency (ClaimNb / Exposure)
  - MAE               — mean absolute error on claim frequency
  - AUC               — ROC-AUC of predicted frequency vs binary (any claim)
  - Avg Pred (freq)   — exposure-weighted mean predicted frequency
  - Avg Actual (freq) — exposure-weighted mean actual frequency (same for all)
"""

from __future__ import annotations


import pathlib
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rustystats as rs
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
import xgboost as xgb
import cvxpy as cp


# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV, KFold, train_test_split
# from sklearn.linear_model import PoissonRegressor
# import statsmodels.api as sm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from aglm import cv_aglm, cva_aglm
from generate_fremtpl2freq import make_fremtpl2freq


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

N_SAMPLE   = None     # None = use full OpenML dataset; set e.g. 50_000 for dev
NBIN_MAX   = 40       # max bins per numeric variable in the AGLM basis
N_ALPHAS   = 10       # lambda grid size for cva_aglm
ALPHA_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CV_FOLDS   = 5

# Maximum rows passed to the CVXPY-based DerivLasso (solver time scales ~O(n^2))
DERIV_LASSO_MAX_N = 20_000


#AGLM_COLS = ["LogExposure"] + FEATURE_COLS

COLORS = {
    "GLM":        "#6C8EAD",
    "RegGLM":     "#3A7CA5",   # proper regularised GLM (sklearn)
    "AGLM-Lin":   "#A8C0D6",   # AGLM without basis expansion
    "AGLM-Lvar":  "#1A3C6E",   # full AGLM with L-variable basis
    "GAM":        "#5C8A5C",
    "GBM":        "#C0392B",
    "CatBoost":   "#8E44AD",
    "DerivLasso": "#E67E22",
}

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


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


# def _log_exposure(df: pd.DataFrame) -> np.ndarray:
#     return np.log(np.maximum(df["Exposure"].values, 1e-9))


def _aglm_features(df: pd.DataFrame, numeric_cols, categorical_cols, exposure_col) -> pd.DataFrame:
    """Feature DataFrame for cva_aglm - prepends LogExposure as near-offset."""
    features_cols=numeric_cols + categorical_cols
    x = df[features_cols].copy()
    x.insert(0, "__log_exposure__", np.log(np.maximum(df[exposure_col].values, 1e-9)))
    return x

# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_and_split(
    df: pd.DataFrame,
    seed: int | None = 42,
    sample_size: int | None = None,
    claim_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

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

    # ── summary table ────────────────────────────────────────────────────────
    # Produce a nice summary table for the users to understadn some high level metrics of the split
    header = f"{'Dataset':<12} {'Records':>10} {'% of Total':>11}"
    if claim_col is not None:
        header += f" {'Claim Rate':>12}"
    sep = "─" * len(header)

    rows = [
        ("Original", df,    len(df)),
        ("Train",    train, len(df)),
        ("Test",     test,  len(df)),
    ]

    summary = {}

    print(sep)
    print(header)
    print(sep)
    for label, subset, total in rows:
        n   = len(subset)
        pct = n / total * 100
        entry = {"n": n, "pct_of_total": round(pct, 1)}
        if claim_col is not None:
            cr = subset[claim_col].mean()
            entry["claim_rate"] = round(cr, 6)
        summary[label] = entry

        row = f"{label:<12} {n:>10,} {pct:>10.1f}%"
        if claim_col is not None:
            row += f" {cr:>11.4f}"
        print(row)
    print(sep)

    return train, test, summary

# ---------------------------------------------------------------------------
# 1/8  GLM - plain Poisson (IRLS, log-exposure offset)
# ---------------------------------------------------------------------------


def fit_glm(train: pd.DataFrame, response_col: str, offset_col: str, 
            numeric_cols: list[str], categorical_cols: list[str], 
            family: Literal["poisson", "gamma", "tweedie"] ) -> dict[str, Any]:  # noqa: F821
    
    print("\n[GLM - Plain Poisson (rustystats) ...", flush=True)
    
    #Just some basic statistics measurements to help understand the process
    timer = time.time()

    terms = {col: {"type": "linear"}      for col in numeric_cols}
    terms.update({col: {"type": "categorical"} for col in categorical_cols})

    result = rs.glm_dict(
        response=response_col,
        terms=terms,
        data=train,
        family=family,
        offset=offset_col,
        weights=None,
        seed=42,
    ).fit()

    timer=time.time()-timer

    # Get the summary of the model
    res={}
    res['model']=result
    res['timer']=timer
    res['deviance']=result.deviance
    res['converged']=result.converged
    res['iterations']=result.iterations

    print(f"  ok {timer:.1f}s | deviance = {result.deviance:.4f} | "
          f"converged = {result.converged} | iterations = {result.iterations}")

    return res


def predict_glm(result: "rs.GLMModel", df: pd.DataFrame) -> np.ndarray:
    # predict() picks up the offset column ("Exposure") automatically from df
    return np.asarray(result.predict(df))



# ---------------------------------------------------------------------------
# 2/8  RegGLM - regularised Poisson GLM
#
# Uses sklearn's PoissonRegressor (coordinate descent, log link, Poisson
# deviance loss) inside a StandardScaler pipeline. GridSearchCV over L2
# penalty strength selects the optimal regularisation level, equivalent to
# R's cv.glmnet(X, y, family="poisson", alpha=0).
#
# Features: standard one-hot encoding of categoricals + continuous numerics
# + log(Exposure) as an additional column (near-offset; coefficient -> 1.0
# as n -> inf). This keeps the feature space identical to the plain GLM.
# ---------------------------------------------------------------------------

def fit_reg_glm(train: pd.DataFrame, response_col: str, offset_col: str, 
            numeric_cols: list[str], categorical_cols: list[str], 
            family: Literal["poisson", "gamma", "tweedie"],
            cv_folds: int|None=CV_FOLDS) -> dict[str, Any]:
    
    print("\nRegGLM - Regularised Poisson GLM"
          " (rustystats elastic_net CV) ...", flush=True)
    
    timer = time.time()

    terms = {col: {"type": "linear"}          for col in numeric_cols}
    terms.update({col: {"type": "categorical"} for col in categorical_cols})

    result = rs.glm_dict(
        response=response_col,
        terms=terms,
        data=train,
        family=family,
        offset=offset_col,
        weights=None,
        seed=42,
    ).fit(
        regularization="elastic_net",
        selection="min",
        cv=cv_folds,
        cv_seed=42,
    )

    timer = time.time() - timer

    res={}
    res['model']=result
    res['timer']=timer
    res['alpha']=result.alpha
    res['non-zero_features']=result.n_nonzero()/len(result.params)

    print(f"  Selected alpha={result.alpha:.6f}  "
          f"non-zero features={result.n_nonzero()}/{len(result.params)}")
    print(f"  ok {timer:.1f}s")
    
    return res


def predict_reg_glm(result: "rs.GLMModel", df: pd.DataFrame) -> np.ndarray:
    return np.asarray(result.predict(df))


# ---------------------------------------------------------------------------
# 3/8  AGLM-Lin - AGLM without basis expansion (linear + one-hot only)
#
# Uses the RustyStats Rust IRLS engine via cva_aglm with all basis expansion
# disabled (no O-dummies, no L-variables for numerics or categoricals). The
# feature space matches RegGLM but the regularisation path searches over both
# lambda (strength) and alpha (L1/L2 mixing), making this the closest match
# to R's cva.glmnet(family="poisson") with a full alpha grid.
# ---------------------------------------------------------------------------

def fit_aglm_linear(train: pd.DataFrame,
                    response_col: str, exposure_col: str, 
                    numeric_cols: list[str], categorical_cols: list[str], 
                    family: Literal["poisson", "gamma", "tweedie"],
                    alpha_grid,
                    cv_folds,
                    n_alphas,
                    nbin_max) -> dict[str, Any]:  # noqa: F821
    
    print("\nAGLM-Lin - AGLM without basis expansion"
          " (cva_aglm / RustyStats, elastic-net CV) ...", flush=True)
    timer = time.time()

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
        od_type_of_quantitatives="N",          # no O-dummy basis for numerics
        add_od_columns_of_qualitatives=False,   # no O-dummy basis for categoricals
        nbin_max=nbin_max,
    )

    timer=time.time() - timer

    bm     = model.best_model
    coef_s = bm.coef(with_names=True)
    n_nz   = int((coef_s.abs() > 1e-6).sum())

    res={}
    res['model']=model
    res['coeficients']=bm.coef(with_names=True)
    res['non_zero_features']=(coef_s.abs() > 1e-6).sum()/len(coef_s)
    res['alpha']=model.best_alpha
    res['lambda']=model.best_model.lambda_
    res['timer']=timer

    

    print(f"  Best alpha={model.best_alpha:.2f}  lambda={bm.lambda_:.5f}  "
          f"nnz={n_nz}/{len(coef_s)}")
    print(f"  ok {timer:.1f}s")
    return res


def predict_aglm_linear(model: "CVAAccurateGLM", df: pd.DataFrame) -> np.ndarray:  # noqa: F821
    return model.best_model.predict(_aglm_features(df))


# ---------------------------------------------------------------------------
# 4/8  AGLM-Lvar - Full AGLM with L-variable basis (Fujita et al. Table 8)
#
# Augments the design matrix with |x - tk| tent functions at up to NBIN_MAX
# data-driven knots per numeric variable, then applies elastic-net via the
# RustyStats Rust IRLS engine with full (alpha, lambda) CV. This is the
# primary AGLM model from the paper.
# ---------------------------------------------------------------------------

def fit_aglm_lvar(train: pd.DataFrame,
                    response_col: str, exposure_col: str, 
                    numeric_cols: list[str], categorical_cols: list[str], 
                    family: Literal["poisson", "gamma", "tweedie"],
                    alpha_grid,
                    cv_folds,
                    n_alphas,
                    nbin_max) -> dict[str, Any]: 
    print(f"\nAGLM-Lvar - full AGLM with L-variable basis"
          f"  nbin={NBIN_MAX} (cva_aglm / RustyStats) ...", flush=True)
    
    timer = time.time()

    x = _aglm_features(train, numeric_cols, categorical_cols, exposure_col)
    y = train[response_col].values.astype(float)

    model = cva_aglm(
        x, y,
        alpha_grid=alpha_grid,
        nfolds=cv_folds,
        family=family,
        lambda_grid=np.logspace(-3, 1, n_alphas),
        add_linear_columns=True,
        use_lvar=True,                         # L-variable |x - tk| basis
        add_od_columns_of_qualitatives=True,
        nbin_max=nbin_max,
    )

    timer = time.time() - timer

    bm     = model.best_model
    coef_s = bm.coef(with_names=True)
    n_nz   = int((coef_s.abs() > 1e-6).sum())
    total  = len(coef_s)

    res={}
    res['model']=model
    res['best_model']=model.best_model
    res['best_alpha']=model.best_alpha
    res['lambda']=bm.lambda_
    res['basis_columns']=total
    res['non_zero_features']=n_nz/total
    res['timer']=timer

    print(f"  Best alpha={model.best_alpha:.2f}  lambda={bm.lambda_:.5f}  "
          f"nnz={n_nz}/{total}  (augmented matrix: {total} basis columns)")
    print(f"  ok {timer:.1f}s")
    return res


def predict_aglm_lvar(model: "CVAAccurateGLM", df: pd.DataFrame) -> np.ndarray:  # noqa: F821
    return model.best_model.predict(_aglm_features(df))


# ---------------------------------------------------------------------------
# 5/8  GAM - PoissonGAM with spline + factor terms (pygam)
# ---------------------------------------------------------------------------

def fit_gam(train: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], response_col, exposure_col) -> dict:
    from pygam import PoissonGAM, f, s
    print("\nGAM - PoissonGAM spline + factor terms (pygam) ...", flush=True)
    
    timer = time.time()

    cat_maps: dict = {}

    def _gam_mat(df: pd.DataFrame) -> np.ndarray:
        parts = [df[numeric_cols].values.astype(float)]
        for col in categorical_cols:
            if col not in cat_maps:
                cats = sorted(df[col].unique())
                cat_maps[col] = {c: i for i, c in enumerate(cats)}
            parts.append(
                np.array([cat_maps[col].get(v, 0) for v in df[col]], float
                          ).reshape(-1, 1)
            )
        return np.hstack(parts)

    x     = _gam_mat(train)
    terms = s(0) + s(1) + s(2) + s(3) + s(4)
    for i in range(5, x.shape[1]):
        terms += f(i)

    gam = PoissonGAM(terms)
    gam.gridsearch(
        x, train[response_col].values,
        weights=train[exposure_col].values,
        progress=False,
    )

    timer = time.time() - timer

    res={}
    res['model']=gam
    res['cat_maps']=cat_maps
    res['gam_mat']=_gam_mat
    res['exposure_col']=exposure_col
    res['timer']=timer

    print(f"  ok {timer:.1f}s")
    return res


def predict_gam(m: dict, df: pd.DataFrame) -> np.ndarray:
    return m["model"].predict(m["gam_mat"](df)) * df[m['exposure_col']].values


# ---------------------------------------------------------------------------
# 6/8  GBM - LightGBM Poisson with CV for n_estimators
# ---------------------------------------------------------------------------



def _lgb_X(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Shared feature prep for LightGBM — categorical dtype + column selection."""
    feature_cols= numeric_cols + categorical_cols
    x = df[feature_cols].copy()
    for col in feature_cols:
        x[col] = x[col].astype("category")
    return x


def fit_gbm(train: pd.DataFrame,  response_col: str, exposure_col: str, numeric_cols: list[str], categorical_cols: list[str], family: str, cv_folds: int) -> lgb.Booster:
    print("\nGBM - LightGBM Poisson (log-exposure offset, CV for n_estimators) ...",
          flush=True)
    
    timer      = time.time()
    #log_exp = np.log(np.maximum(train[exposure_col].values, 1e-9))

    freq = train[response_col].values / np.maximum(train[exposure_col].values, 1e-9)
    w    = train[exposure_col].values

    dtrain = lgb.Dataset(
        _lgb_X(train, numeric_cols=numeric_cols, categorical_cols=categorical_cols,),
        label=freq,
        weight=w,
        free_raw_data=False,
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

    cv = lgb.cv(
        params, dtrain,
        num_boost_round=500, nfold=cv_folds,
        stratified=False,
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    # Get the toal time it takes to train the model
    timer=time.time() - timer

    best_n  = len(cv["valid poisson-mean"])
    booster = lgb.train(
        params, dtrain,
        num_boost_round=best_n,
        callbacks=[lgb.log_evaluation(-1)],
    )

    res={}
    res['model']=booster
    res['best_n_estimators']=best_n
    res['time']=timer
    res['exposure_col']=exposure_col

    print(f"  Best n_estimators={best_n}  ok {timer:.1f}s")
    return res


def predict_gbm_weights(res: dict[str,Any], df: pd.DataFrame) -> np.ndarray:
    return res['model'].predict(_lgb_X(df)) * df[res['exposure_col']].values


# ---------------------------------------------------------------------------
# 7/8  CatBoost - Poisson regressor with native categorical support
# ---------------------------------------------------------------------------

def fit_catboost(train: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str], response_col: str, exposure_col: str, family: Literal["poisson", "gamma", "tweedie"], cv_folds: int) -> "CatBoostRegressor":
    from catboost import CatBoostRegressor, Pool, cv as cb_cv
    print("\nCatBoost - Poisson regressor (native categoricals, log-exposure offset) ...",
          flush=True)
    
    timer = time.time()

    feature_cols = numeric_cols + categorical_cols

    x= train[feature_cols].copy()
    for col in categorical_cols:
        x[col] = x[col].astype(str)

    log_exp = np.log(np.maximum(train[exposure_col].values, 1e-9))

    pool = Pool(
        data=x,
        label=train[response_col].values.astype(float),
        baseline=log_exp,                          # proper log-exposure offset
        cat_features=categorical_cols,
    )

    # We need a bi of a hack based on the family to get CatBoost to work
    # with different error functions
    if family=="poisson":
        family = "Poisson"
    elif family=="tweedie":
        family = "Tweedie:variance_power=1.5"
    elif family=="gamma":
        family = "Tweedie:variance_power=2.0"

    model = CatBoostRegressor(
        loss_function=family,
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

    timer=time.time() - timer
    best_iter = int(cv_result["test-Poisson-mean"].idxmin()) + 1
    print(f"  Best iteration from CV: {best_iter}")

    model.set_params(iterations=best_iter, early_stopping_rounds=None)
    model.fit(pool)

    res={}
    res['model']=model
    res['best_iteration']=best_iter
    res['timer']=timer
    res['cv_result']=cv_result["test-Poisson-mean"].min()
    res['exposure_col']=exposure_col
    res['categorical_cols']=categorical_cols
    res['feature_cols']=feature_cols

    print(f"  ok {timer:.1f}s")
    return res


def predict_catboost(
    res: dict[str, Any],
    df: pd.DataFrame,
) -> np.ndarray:
    from catboost import Pool
    x = df[res['feature_cols']].copy()
    for col in res['categorical_cols']:
        x[col] = x[col].astype(str)

    log_exp = np.log(np.maximum(df[res['exposure_col']].values, 1e-9))

    pool = Pool(
        data=x,
        baseline=log_exp,                          # must pass at predict time too
        cat_features=res['categorical_cols'],
    )
    return res['model'].predict(pool)                     # no * Exposure — offset handles it

def fit_xgb(train: pd.DataFrame, response_col, exposure_col, numeric_cols: list[str], categorical_cols: list[str], family: str, cv_folds: int) -> "xgb.Booster":  # noqa: F821
    
    print("\nXGBoost - Poisson (log-exposure offset, CV for n_estimators) ...",
          flush=True)
    
    timer = time.time()

    feature_cols=numeric_cols + categorical_cols

    x = train[feature_cols].copy()
    for col in categorical_cols:
        x[col] = x[col].astype("category")

    log_exp = np.log(np.maximum(train[exposure_col].values, 1e-9))

    dtrain = xgb.DMatrix(
        x,
        label=train[response_col].values,
        enable_categorical=True,

    )
    dtrain.set_base_margin(log_exp)       # log-exposure as proper offset

    params = {
        "objective":        "count:poisson",
        "eval_metric":      "poisson-nloglik",
        "learning_rate":    0.01,
        "max_depth":        4,
        "min_child_weight": 50,           # min obs per leaf — mirrors lgb min_data_in_leaf
        "subsample":        0.9,
        "colsample_bytree": 0.8,
        "seed":             42,
        "nthread":          4,
        "verbosity":        0
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
    print(f"  Best n_estimators={best_n}  "
          f"cv_poisson_nloglik={cv_result['test-poisson-nloglik-mean'].iloc[-1]:.6f}")

    booster = xgb.train(
        params, dtrain,
        num_boost_round=best_n,
        verbose_eval=False,
    )

    timer=time.time() - timer

    res={}
    res['model']=booster
    res['timer']=timer
    res['best_n']=best_n
    res['cv_poisson_nloglik']=cv_result['test-poisson-nloglik-mean'].iloc[-1]

    print(f"  ok {timer:.1f}s")
    return res


def predict_xgb(booster: "xgb.Booster", df: pd.DataFrame) -> np.ndarray:  # noqa: F821
    

    X = df[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")

    log_exp = np.log(np.maximum(df["Exposure"].values, 1e-9))

    dtest = xgb.DMatrix(X, enable_categorical=True)
    #dtest.set_base_margin(log_exp)        # must pass offset at predict time too

    return booster.predict(dtest)*df["Exposure"].values         # returns predicted counts directly


# ---------------------------------------------------------------------------
# 8/8  DerivLasso - Derivative (fused) Lasso GLM via CVXPY (Akur8-style)
#
# Penalises first-differences between adjacent ordered-category coefficients:
#   min_beta  sum[exp(eta) - y*eta]  +  lambda * sum_v sum_j |beta_{v,j+1} - beta_{v,j}|
# Produces piecewise-constant relativities. Solver: CLARABEL (SCS fallback).
# Training is capped at DERIV_LASSO_MAX_N rows for feasible solve times.
# ---------------------------------------------------------------------------

def _engineer_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_bins: int = 40,
    exposure_col: str | None = None,
    other_label: str = "Other",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Generic feature engineering:
      - Numeric cols  → quantile-binned into n_bins ordered categories (_bin suffix)
      - Categorical cols → top n_bins levels by exposure, rest → other_label (_cap suffix)

    Returns
    -------
    out                 : transformed DataFrame (original cols preserved)
    new_numeric_bins    : list of new binned column names
    new_categorical_caps: list of new capped column names
    """
    out = df.copy()
    new_numeric_bins: list[str]     = []
    new_categorical_caps: list[str] = []

    # ── numeric → quantile bins ───────────────────────────────────────────────
    for col in numeric_cols:
        new_col = f"{col}_bin"
        try:
            out[new_col] = pd.qcut(
                out[col].astype(float),
                q=n_bins,
                duplicates="drop",   # handles low-cardinality / skewed cols
            ).astype(str)
        except ValueError:
            # Fewer unique values than requested bins — fall back to all unique values
            out[new_col] = out[col].astype(str)
        new_numeric_bins.append(new_col)

    # ── categorical → top-N by exposure, rest → Other ────────────────────────
    for col in categorical_cols:
        new_col = f"{col}_cap"
        series  = out[col].astype(str)

        if exposure_col is not None:
            # Sum exposure per category, keep top n_bins
            exposure_by_cat = (
                out.groupby(series)[exposure_col]
                .sum()
                .sort_values(ascending=False)
            )
        else:
            # Fall back to record count
            exposure_by_cat = series.value_counts()

        top_levels = set(exposure_by_cat.head(n_bins-1).index)
        out[new_col] = series.where(series.isin(top_levels), other=other_label)
        new_categorical_caps.append(new_col)

    return out, new_numeric_bins, new_categorical_caps

def _engineer_features_dl(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Bin continuous predictors into ordered categorical groups."""
    out = df.copy()
    out["DrivAgeBin"] = pd.cut(
        out["DrivAge"],
        bins=[17, 22, 26, 30, 40, 50, 60, 70, 101],
        labels=["18-22", "23-26", "27-30", "31-40",
                "41-50", "51-60", "61-70", "71+"],
        include_lowest=True, ordered=True,
    )
    out["BonusMalusBin"] = pd.cut(
        out["BonusMalus"],
        bins=[49, 60, 70, 80, 100, 120, 150, 231],
        labels=["50-60", "61-70", "71-80", "81-100",
                "101-120", "121-150", "151+"],
        include_lowest=True, ordered=True,
    )
    out["VehAgeBin"] = pd.cut(
        out["VehAge"],
        bins=[0, 1, 3, 6, 10, 15, 101],
        labels=["0-1", "2-3", "4-6", "7-10", "11-15", "16+"],
        include_lowest=True, ordered=True,
    )
    out["VehPowerBin"] = pd.cut(
        out["VehPower"],
        bins=[3, 5, 7, 9, 11, 15],
        labels=["4-5", "6-7", "8-9", "10-11", "12+"],
        include_lowest=True, ordered=True,
    )
    out["DensityBin"] = pd.qcut(
        np.log1p(out["Density"].astype(float)),
        q=6,
        labels=["D1", "D2", "D3", "D4", "D5", "D6"],
        duplicates="drop",
    )
    return out


class _DLDesignMatrix:
    """Design matrix builder with fit_transform / transform pair."""

    DL_ORDERED = ["DrivAgeBin", "BonusMalusBin", "VehAgeBin",
                  "VehPowerBin", "DensityBin"]
    DL_NOMINAL = ["VehGas", "Area"]

    def __init__(self):
        self.feature_groups_: dict = {}
        self.feature_names_:  list = []
        self.col_order_:      dict = {}
        self.n_features_:     int  = 0

    def fit_transform(self, df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> np.ndarray:
        n = len(df)
        blocks, col = [], 0

        blocks.append(np.ones((n, 1)))
        self.feature_groups_["intercept"] = [col]
        self.feature_names_.append("intercept")
        col += 1

        for var in numeric_cols:
            dummies = pd.get_dummies(df[var], prefix=var,
                                     drop_first=False, dtype=float)
            k = dummies.shape[1]
            blocks.append(dummies.values)
            self.feature_groups_[var] = list(range(col, col + k))
            self.feature_names_.extend(dummies.columns.tolist())
            self.col_order_[var] = dummies.columns.tolist()
            col += k

        for var in categorical_cols:
            dummies = pd.get_dummies(df[var], prefix=var,
                                     drop_first=True, dtype=float)
            k = dummies.shape[1]
            blocks.append(dummies.values)
            self.feature_groups_[var] = list(range(col, col + k))
            self.feature_names_.extend(dummies.columns.tolist())
            self.col_order_[var] = dummies.columns.tolist()
            col += k

        self.n_features_ = col
        X = np.hstack(blocks)
        print(f"  Design matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the encoding learned in fit_transform to new data."""
        blocks = [np.ones((len(df), 1))]
        for var in self.DL_ORDERED:
            dummies = pd.get_dummies(df[var], prefix=var,
                                     drop_first=False, dtype=float)
            for c in self.col_order_[var]:
                if c not in dummies.columns:
                    dummies[c] = 0.0
            blocks.append(dummies[self.col_order_[var]].values)
        for var in self.DL_NOMINAL:
            dummies = pd.get_dummies(df[var], prefix=var,
                                     drop_first=False, dtype=float)
            for c in self.col_order_[var]:
                if c not in dummies.columns:
                    dummies[c] = 0.0
            blocks.append(dummies[self.col_order_[var]].values)
        return np.hstack(blocks)


class _DerivativeLassoGLM:
    """Poisson GLM with first-difference lasso penalty, solved via CVXPY."""

    def __init__(self, lam: float = 0.05):
        self.lam            = lam
        self.coef_:          np.ndarray | None = None
        self.solve_status_:  str = ""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray | None = None,
        ordered_groups: dict | None = None,
    ) -> "_DerivativeLassoGLM":
        

        n, p  = X.shape
        exp_  = np.ones(n) if exposure is None else np.asarray(exposure, float)
        log_e = np.log(np.maximum(exp_, 1e-12))

        beta  = cp.Variable(p)
        eta   = X @ beta + log_e
        loss  = cp.sum(cp.exp(eta) - cp.multiply(y, eta))

        diff_list = []
        if ordered_groups:
            for var, idx in ordered_groups.items():
                if len(idx) > 1:
                    b_v = beta[idx]
                    for j in range(len(idx) - 1):
                        diff_list.append(cp.abs(b_v[j + 1] - b_v[j]))

        penalty = (cp.sum(cp.hstack(diff_list)) if diff_list
                   else cp.Constant(0))

        prob = cp.Problem(cp.Minimize(loss + self.lam * penalty))
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-4)

        self.solve_status_ = prob.status
        if beta.value is None:
            raise RuntimeError(f"DerivLasso solver failed: {prob.status}")
        self.coef_ = beta.value
        return self

    def predict(self, X: np.ndarray,
                exposure: np.ndarray | None = None) -> np.ndarray:
        exp_  = np.ones(len(X)) if exposure is None else np.asarray(exposure, float)
        log_e = np.log(np.maximum(exp_, 1e-12))
        return np.exp(X @ self.coef_ + log_e)


def fit_derivative_lasso(train: pd.DataFrame, response_col: str, exposure_col: str, numeric_cols: list[str], categorical_cols: list[str], num_bins: int, cv_folds: int) -> dict:
    print("\n[8/8] DerivLasso - derivative fused lasso Poisson (CVXPY) ...",
          flush=True)
    t0 = time.time()

    # if len(train) > DERIV_LASSO_MAX_N:
    #     strat = (train[response_col] > 0).astype(int)
    #     sub, _ = train_test_split(
    #         train, train_size=DERIV_LASSO_MAX_N,
    #         random_state=42, stratify=strat,
    #     )
    #     sub = sub.reset_index(drop=True)
    #     print(f"  Subsampled {DERIV_LASSO_MAX_N:,} rows for CVXPY solver "
    #           f"(full train = {len(train):,})")
    # else:
    #   sub = train.copy()

    sub=train.copy()

    sub_eng = _engineer_features_dl(sub, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    dm      = _DLDesignMatrix()

    X       = dm.fit_transform(sub_eng, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    y       = sub[response_col].values.astype(float)
    exp_    = sub[exposure_col].values.astype(float)

    ordered_groups = {v: dm.feature_groups_[v] for v in dm.DL_ORDERED}

    lam_grid = [0.005, 0.02, 0.05, 0.10, 0.25, 0.50]
    kf       = KFold(n_splits=cv_folds, shuffle=True, random_state=1)
    print(f"  Lambda CV over {lam_grid} ({cv_folds}-fold) ...")

    cv_devs = []
    for lam in lam_grid:
        fold_devs = []
        for tr_idx, va_idx in kf.split(X):
            m = _DerivativeLassoGLM(lam=lam)
            m.fit(X[tr_idx], y[tr_idx],
                  exposure=exp_[tr_idx],
                  ordered_groups=ordered_groups)
            fold_devs.append(
                _poisson_deviance(y[va_idx], m.predict(X[va_idx], exp_[va_idx]))
            )
        mean_dev = float(np.mean(fold_devs))
        cv_devs.append(mean_dev)
        print(f"    lambda={lam:.3f}  cv_dev={mean_dev:.6f}")

    best_lam = lam_grid[int(np.argmin(cv_devs))]
    print(f"  Best lambda by CV deviance: {best_lam}")

    best_model = _DerivativeLassoGLM(lam=best_lam)
    best_model.fit(X, y, exposure=exp_, ordered_groups=ordered_groups)
    print(f"  Solver status: {best_model.solve_status_}")
    print(f"  ok {time.time() - t0:.1f}s")

    return {"model": best_model, "dm": dm,
            "best_lam": best_lam, "ordered_groups": ordered_groups}


def predict_derivative_lasso(fitted: dict, df: pd.DataFrame) -> np.ndarray:
    X   = fitted["dm"].transform(_engineer_features_dl(df))
    exp = df["Exposure"].values.astype(float)
    return fitted["model"].predict(X, exp)


# ---------------------------------------------------------------------------
# Figure 2 - AGLM-Lvar component curves
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    response_col: str,
    exposure_col: str,
    offset_col: str,
    sample_size: int | None = None,
    num_bin: int = NBIN_MAX,
    n_alphas: int = N_ALPHAS,
    alpha_grid: np.ndarray | None = ALPHA_GRID,
    cv_folds: int = CV_FOLDS,
    run_gam: bool = True,
    run_gbm: bool = True,
    run_catboost: bool = True,
    run_xgb: bool = True,
    run_deriv_lasso: bool = True,
) -> None:
    
    
    print("=" * 70)
    print("Fujita et al. (2020) - Section 5 Numerical Experiments")
    print("  Models: GLM | RegGLM | AGLM-Lin | AGLM-Lvar |")
    print("          GAM | GBM | CatBoost | DerivLasso")
    print("=" * 70)
    t_total = time.time()

    train, test, summ_stats = load_and_split(df=df)
    #ref_levels  = {col: sorted(train[col].unique()) for col in categorical_cols}

    print()
    print("─" * 70)
    print(f"Fitting models  | nbin={NBIN_MAX}  n_alphas={N_ALPHAS}"
          f"  alpha_grid={ALPHA_GRID.tolist()}  cv={CV_FOLDS}")
    print("─" * 70)

    times={}

    glm_m = fit_glm(train:=train, response_col:=response_col, offset_col:=offset_col, numeric_cols:=numeric_cols, categorical_cols:=categorical_cols, family:='poisson' ) #, ref_levels)   # sklearn GLM with log-exposure offset
    reg_glm_m          = fit_reg_glm(train:= train, response_col:=response_col, offset_col:=offset_col, numeric_cols:=numeric_cols, categorical_cols:=categorical_cols, family:='poisson', cv_folds=cv_folds)
    aglm_lin_m         = fit_aglm_linear(train:=train, response_col=response_col, exposure_col=exposure_col, numeric_cols=numeric_cols, categorical_cols=categorical_cols, family='poisson',alpha_grid=alpha_grid,cv_folds=cv_folds,n_alphas=n_alphas, nbin_max=num_bin )            # AGLM, no basis
    aglm_lvar_m        = fit_aglm_lvar(train=train,response_col=response_col, exposure_col=exposure_col, numeric_cols=numeric_cols, categorical_cols=categorical_cols, family='poisson',alpha_grid=alpha_grid,cv_folds=cv_folds,n_alphas=n_alphas, nbin_max=num_bin )              # AGLM, L-variable basis
    gam_m              = fit_gam(train=train, numeric_cols=numeric_cols, categorical_cols=categorical_cols,response_col=response_col, exposure_col=exposure_col)              if run_gam         else None
    gbm_m              = fit_gbm(train=train, response_col=response_col,exposure_col=exposure_col, numeric_cols=numeric_cols, categorical_cols=categorical_cols, family='poisson',  cv_folds=cv_folds)              if run_gbm         else None
    xgb_m              = fit_xgb(train=train, response_col=response_col, exposure_col=exposure_col, family='poisson', numeric_cols=numeric_cols, categorical_cols=categorical_cols, cv_folds=cv_folds)              if run_xgb         else None
    catboost_m         = fit_catboost(train=train, response_col=response_col, categorical_cols=categorical_cols,numeric_cols=numeric_cols, exposure_col=exposure_col, family='poisson', cv_folds=cv_folds)         if run_catboost    else None
    deriv_lasso_m      = fit_derivative_lasso(train=train, response_col=response_col,exposure_col=exposure_col,cv_folds=cv_folds) if run_deriv_lasso else None

    print()
    print("─" * 70)
    print("Full Metrics Table - Hold-out Test Set")
    print("─" * 70)

    predict_fns: dict = {
        "GLM":       lambda df: predict_glm(glm_m['model'], df), #, glm_ncols, df, ref_levels),
        "RegGLM":    lambda df: predict_reg_glm(reg_glm_m['model'], df),
        "AGLM-Lin":  lambda df: predict_aglm_linear(aglm_lin_m['model'], df),
        "AGLM-Lvar": lambda df: predict_aglm_lvar(aglm_lvar_m['model'], df),
    }
    if gam_m         is not None: predict_fns["GAM"]        = lambda df: predict_gam(gam_m, df)
    if gbm_m         is not None: predict_fns["GBM"]        = lambda df: predict_gbm_weights(gbm_m, df)
    if xgb_m         is not None: predict_fns["XGB"]        = lambda df: predict_xgb(xgb_m, df)
    if catboost_m    is not None: predict_fns["CatBoost"]   = lambda df: predict_catboost(catboost_m, df)
    if deriv_lasso_m is not None: predict_fns["DerivLasso"] = lambda df: predict_derivative_lasso(deriv_lasso_m, df)

    rows = []
    for name, fn in predict_fns.items():
        mu  = fn(test)
        row = compute_metrics(name, test[response_col].values, mu, test[exposure_col].values)
        rows.append(row)
        print(f"  {name:12s}: dev={row['Poisson Deviance']:.6f}  "
              f"mse={row['MSE']:.8f}  mae={row['MAE']:.6f}  "
              f"auc={row['AUC']:.4f}  gini={row['Gini']:.4f} "
              f"avg_pred={row['Avg Pred (freq)']:.5f} "
              f"med_pred={row['Median Pred (freq)']:.5f}")

    metrics_table = (
        pd.DataFrame(rows)
        .sort_values("Poisson Deviance")
        .reset_index(drop=True)
    )

    print()
    print("─" * 132)
    hdr = (f"  {'Model':<12}  {'Poisson Dev':>13}  {'MSE':>12}  "
           f"{'MAE':>10}  {'AUC':>7}  {'Gini':>7} {'Avg Pred':>10}  {'Med Pred':>10}  "
           f"{'Avg Actual':>10}  {'Med Actual':>10}")
    print(hdr)
    print("  " + "─" * 128)
    for _, r in metrics_table.iterrows():
        print(
            f"  {r['Model']:<12}  {r['Poisson Deviance']:>13.7f}  "
            f"{r['MSE']:>12.9f}  {r['MAE']:>10.7f}  "
            f"{r['AUC']:>7.4f} {r['Gini']:>7.4f}  {r['Avg Pred (freq)']:>10.6f}  "
            f"{r['Median Pred (freq)']:>10.6f}  {r['Avg Actual (freq)']:>10.6f}  "
            f"{r['Median Actual (freq)']:>10.6f}"
        )
    print("─" * 132)

    print()
    print("Fujita et al. Table 8 reference values (full N=678k, Poisson Deviance):")
    print(pd.DataFrame({
        "Model":          ["AGLM-Lvar", "GBM",     "GAM",     "GLM",     "AGLM-Lin"],
        "Paper deviance": [0.3111920,   0.3123919, 0.3171236, 0.3201199, 0.3201245],
    }).to_string(index=False))

    print("\nGenerating figures ...")

    fig2 = plot_component_curves(aglm_lvar_m, train, metrics_table)
    fig2.savefig("fremtpl2_figure2_component_curves.png", dpi=130, bbox_inches="tight")
    print("  Saved: fremtpl2_figure2_component_curves.png")
    plt.close(fig2)

    fig3 = plot_metrics_comparison(metrics_table)
    fig3.savefig("fremtpl2_figure3_metrics_comparison.png", dpi=130, bbox_inches="tight")
    print("  Saved: fremtpl2_figure3_metrics_comparison.png")
    plt.close(fig3)

    fig4 = plot_calibration(metrics_table)
    fig4.savefig("fremtpl2_figure4_calibration.png", dpi=130, bbox_inches="tight")
    print("  Saved: fremtpl2_figure4_calibration.png")
    plt.close(fig4)

    print(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} minutes")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # We use teh french third party dataset for our testing
    print("Loading freMTPL2freq from OpenML ...")
    df = fetch_openml(data_id=41214, as_frame=True, parser="pandas")["data"]

    for col in ["ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["Exposure", "Density"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["VehBrand", "VehGas", "Area", "Region"]:
        df[col] = df[col].astype(str).str.strip()

    before = len(df)
    df = df.dropna(subset=["ClaimNb", "VehPower", "VehAge", "DrivAge",
                               "BonusMalus", "Exposure", "Density",
                               "VehBrand", "VehGas", "Area", "Region"])
    print(f"  Dropped {before - len(df):,} rows with missing values")

    df["Exposure"] = df["Exposure"].clip(lower=1e-4)
    df["ClaimNb"]  = df["ClaimNb"].astype(int).clip(upper=4)
    df = df.drop(columns=["IDpol"], errors="ignore")

    # Define our numeric and categorical columns
    NUMERIC_COLS     = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    CATEGORICAL_COLS = ["VehBrand", "VehGas", "Area", "Region"]


    # The main model fitting routines happen here, with all models and metrics computed and printed, and figures generated
    res=main (df=df, categorical_cols=CATEGORICAL_COLS, numeric_cols=NUMERIC_COLS, 
              response_col='ClaimNb', exposure_col='Exposure', offset_col='Exposure', sample_size=None, 
              num_bin=NBIN_MAX,
               n_alphas=N_ALPHAS, alpha_grid=ALPHA_GRID,
               cv_folds=CV_FOLDS, run_gam=True, run_gbm=True, run_catboost=True, run_xgb=True, run_deriv_lasso=True)


    def main(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    response_col: str,
    offset_col: str,
    sample_size: int | None = None,
    num_bin: int = NBIN_MAX,
    n_alphas: int = N_ALPHAS,
    alpha_grid: np.ndarray | None = ALPHA_GRID,
    cv_folds: int = CV_FOLDS,
    run_gam: bool = True,
    run_gbm: bool = True,
    run_catboost: bool = True,
    run_xgb: bool = True,
    run_deriv_lasso: bool = True,
) -> None:
    