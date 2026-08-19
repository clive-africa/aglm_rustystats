import numpy as np
import time
import pandas as pd
import xgboost as xgb

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

    return booster.predict(dtest)*df["Exposure"].values         # returns predicted 
