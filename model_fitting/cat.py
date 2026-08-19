import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv as cb_cv

def fit_catboost(train: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str], response_col: str, exposure_col: str, family: Literal["poisson", "gamma", "tweedie"], cv_folds: int) -> "CatBoostRegressor":
    
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