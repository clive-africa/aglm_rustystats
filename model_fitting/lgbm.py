"""LightGBM fitting/prediction on Polars frames, via Arrow (no numpy round-trip)."""

import time
from typing import Any, TypedDict

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa

Categories = dict[str, list[str]]


class LgbmResult(TypedDict):
    model: lgb.Booster
    best_n_estimators: int
    time: float
    response_col: str
    exposure_col: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    categories: Categories
    params: dict[str, Any]


def _fit_categories(df: pl.DataFrame, categorical_cols: list[str]) -> Categories:
    """Level sets for each categorical, learned from the training frame only."""
    levels = df.select(
        pl.col(c).cast(pl.Utf8).drop_nulls().unique().sort().implode() for c in categorical_cols
    ).row(0)
    return dict(zip(categorical_cols, levels))


def _lgb_table(
    df: pl.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    categories: Categories,
) -> pa.Table:
    """Build the LightGBM design matrix as an Arrow table.

    Categorical codes are pinned by a polars Enum fitted on train and reused at
    predict time, so codes stay in sync across calls and any level unseen in
    training becomes null -- which LightGBM treats as missing.

    LightGBM's Arrow reader wants signed integers, hence the Int32 cast on the
    Enum's physical (UInt32) representation.
    """
    missing = [c for c in numeric_cols + categorical_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")

    return (
        df.select(
            *(pl.col(c).cast(pl.Float64) for c in numeric_cols),
            *(
                pl.col(c)
                .cast(pl.Utf8)
                .cast(pl.Enum(categories[c]), strict=False)
                .to_physical()
                .cast(pl.Int32)
                for c in categorical_cols
            ),
        )
        .rechunk()
        .to_arrow()
    )


def _best_rounds(cv: dict[str, list[float]]) -> int:
    """Rounds kept by lgb.cv -- metric-agnostic, so gamma/tweedie work too."""
    mean_keys = [k for k in cv if k.endswith("-mean")]
    if not mean_keys:
        raise RuntimeError(f"no mean metric in cv output: {list(cv)}")
    return len(cv[mean_keys[0]])


def fit_lgbm(
    train: pl.DataFrame,
    response_col: str,
    exposure_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    family: str,
    cv_folds: int,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 20,
    seed: int = 42,
    extra_params: dict[str, Any] | None = None,
) -> LgbmResult:
    """Fit a weighted-rate GBM: response/exposure as label, exposure as weight.

    Equivalent to a log-exposure offset for Poisson, and the same shape works for
    gamma severity (response=total loss, exposure=claim count) and tweedie pure
    premium. Pass e.g. extra_params={"tweedie_variance_power": 1.5} as needed.
    """
    print(f"\nGBM - LightGBM {family} (exposure-weighted rate, CV for n_estimators) ...", flush=True)
    timer = time.perf_counter()

    n_before = train.height
    train = train.filter(pl.col(exposure_col) > 0)
    if dropped := n_before - train.height:
        print(f"  dropped {dropped:,} rows with non-positive {exposure_col}")
    if train.is_empty():
        raise ValueError(f"no rows with positive {exposure_col}")

    label, weight = (
        train.select(
            (pl.col(response_col) / pl.col(exposure_col)).cast(pl.Float64).alias("y"),
            pl.col(exposure_col).cast(pl.Float64).alias("w"),
        )
        .to_numpy()
        .T
    )

    categories = _fit_categories(train, categorical_cols)
    x_train = _lgb_table(train, numeric_cols, categorical_cols, categories)

    # Feature names come from the Arrow schema, so categoricals are named, not
    # positional -- no index arithmetic to fall out of step with the column order.
    dtrain = lgb.Dataset(
        x_train,
        label=label,
        weight=weight,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )

    params: dict[str, Any] = {
        "objective": family,
        "metric": family,
        "learning_rate": 0.01,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "min_sum_hessian_in_leaf": 1e-3,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "seed": seed,
        "verbose": -1,
        "n_jobs": -1,
        **(extra_params or {}),
    }

    cv = lgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=cv_folds,
        stratified=False,
        seed=seed,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    best_n = _best_rounds(cv)

    booster = lgb.train(params, dtrain, num_boost_round=best_n)
    elapsed = time.perf_counter() - timer

    print(f"  Best n_estimators={best_n}  ok {elapsed:.1f}s")
    return LgbmResult(
        model=booster,
        best_n_estimators=best_n,
        time=elapsed,
        response_col=response_col,
        exposure_col=exposure_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        categories=categories,
        params=params,
    )


def predict_lgbm(res: LgbmResult, df: pl.DataFrame, scale_by_exposure: bool = True) -> np.ndarray:
    """Predicted rate, scaled to expected counts/amounts by exposure by default."""
    x = _lgb_table(df, res["numeric_cols"], res["categorical_cols"], res["categories"])
    rate = res.predict(x)
    if not scale_by_exposure:
        return rate
    return rate * df[res["exposure_col"]].to_numpy()


# import time
# import pandas as pd
# import polars as pl
# import numpy as np
# import lightgbm as lgb
# from typing import Any

# def _lgb_X(df: pl.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pl.DataFrame:
#     """Shared feature prep for LightGBM — categorical dtype + column selection."""
#     feature_cols= numeric_cols + categorical_cols
#     x = df.select(feature_cols)
#     x = x.with_columns(pl.col(categorical_cols).cast(pl.Categorical))

#     return x


# def fit_lgbm(train: pd.DataFrame,  response_col: str, exposure_col: str, numeric_cols: list[str], categorical_cols: list[str], family: str, cv_folds: int) -> lgb.Booster:
#     print("\nGBM - LightGBM Poisson (log-exposure offset, CV for n_estimators) ...",
#           flush=True)
    
#     timer      = time.time()
#     #log_exp = np.log(np.maximum(train[exposure_col].values, 1e-9))

#     freq = train[response_col].values / np.maximum(train[exposure_col].values, 1e-9)
#     w    = train[exposure_col].values

#     dtrain = lgb.Dataset(
#         _lgb_X(train, numeric_cols=numeric_cols, categorical_cols=categorical_cols,),
#         label=freq,
#         weight=w,
#         free_raw_data=False,
#     )

#     params = {
#         "objective":        family,
#         "metric":           family,
#         "learning_rate":    0.01,
#         "num_leaves":       63,
#         "min_data_in_leaf": 50,
#         "feature_fraction": 0.8,
#         "bagging_fraction": 0.9,
#         "bagging_freq":     5,
#         "verbose":          -1,
#         "n_jobs":           -1,
#     }

#     cv = lgb.cv(
#         params, dtrain,
#         num_boost_round=500, nfold=cv_folds,
#         stratified=False,
#         callbacks=[
#             lgb.early_stopping(20, verbose=False),
#             lgb.log_evaluation(-1),
#         ],
#     )

#     # Get the toal time it takes to train the model
#     timer=time.time() - timer

#     best_n  = len(cv["valid poisson-mean"])
#     booster = lgb.train(
#         params, dtrain,
#         num_boost_round=best_n,
#         callbacks=[lgb.log_evaluation(-1)],
#     )

#     res={}
#     res['model']=booster
#     res['best_n_estimators']=best_n
#     res['time']=timer
#     res['exposure_col']=exposure_col

#     print(f"  Best n_estimators={best_n}  ok {timer:.1f}s")
#     return res


# def predict_lgbm(res: dict[str,Any], df: pd.DataFrame) -> np.ndarray:
#     return res['model'].predict(_lgb_X(df)) * df[res['exposure_col']].values
