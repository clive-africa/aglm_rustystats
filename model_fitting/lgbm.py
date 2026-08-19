import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Any

def _lgb_X(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Shared feature prep for LightGBM — categorical dtype + column selection."""
    feature_cols= numeric_cols + categorical_cols
    x = df[feature_cols].copy()
    for col in feature_cols:
        x[col] = x[col].astype("category")
    return x


def fit_lgbm(train: pd.DataFrame,  response_col: str, exposure_col: str, numeric_cols: list[str], categorical_cols: list[str], family: str, cv_folds: int) -> lgb.Booster:
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


def predict_lgbm(res: dict[str,Any], df: pd.DataFrame) -> np.ndarray:
    return res['model'].predict(_lgb_X(df)) * df[res['exposure_col']].values
