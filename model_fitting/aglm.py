from __future__ import annotations
import pandas as pd
import rustystats as rs
import time
import numpy as np
from aglm import cv_aglm, cva_aglm

def _aglm_features(df: pd.DataFrame, numeric_cols, categorical_cols, exposure_col) -> pd.DataFrame:
    """Feature DataFrame for cva_aglm - prepends LogExposure as near-offset."""
    features_cols=numeric_cols + categorical_cols
    x = df[features_cols].copy()
    x.insert(0, "__log_exposure__", np.log(np.maximum(df[exposure_col].values, 1e-9)))
    return x

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