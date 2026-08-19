


"""CatBoost fitting/prediction on Polars frames, mirroring fit_lgbm / fit_xgb.

CatBoost differs from the other two backends in one important way: it consumes
categoricals as STRINGS and hashes them internally, computing target statistics
per level. There is therefore no Enum/code-alignment step here -- train and
predict cannot fall out of sync, and levels unseen in training are handled
natively via the prior rather than being treated as missing. The one thing
CatBoost will not accept is a null in a categorical column, so nulls are mapped
to an explicit sentinel level (see _cb_frame).

Exposure handling follows the XGBoost module, selected by `use_offset`:

  use_offset=False (default)  label = response/exposure, weight = exposure.
                              Matches fit_lgbm/fit_xgb, so a benchmark compares
                              algorithms rather than parameterisations.

  use_offset=True             label = response, baseline = log(exposure) + intercept.
                              The classical actuarial offset.
"""

import time
import warnings
from typing import Any, Literal, TypedDict

import catboost as cb
import numpy as np
import polars as pl

Family = Literal["poisson", "gamma", "tweedie"]

# CatBoost has no native Gamma loss. Gamma is the Tweedie limit at p=2, but
# CatBoost enforces 1 < variance_power < 2 strictly and raises on exactly 2.0,
# so gamma is approximated just inside the boundary.
_GAMMA_VARIANCE_POWER = 1.99

_LOSS: dict[str, str] = {
    "poisson": "Poisson",
    "tweedie": "Tweedie:variance_power=1.5",
    "gamma": f"Tweedie:variance_power={_GAMMA_VARIANCE_POWER}",
}

MISSING_LEVEL = "__missing__"


class CatBoostResult(TypedDict):
    model: cb.CatBoostRegressor
    best_n_estimators: int
    time: float
    cv_metric: str
    cv_score: float
    response_col: str
    exposure_col: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    use_offset: bool
    offset_intercept: float
    loss_function: str
    params: dict[str, Any]


def _cb_frame(
    df: pl.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> pl.DataFrame:
    """Design matrix as a Polars frame: floats for numerics, strings for categoricals.

    CatBoost accepts a Polars frame directly, so no pandas/numpy round-trip is
    needed. It does reject nulls in categorical columns outright, hence the
    fill_null to an explicit sentinel -- which also makes "missing" a level the
    model can learn from, rather than something silently coerced to the string
    "nan" by an .astype(str) on a pandas frame.
    """
    missing = [c for c in numeric_cols + categorical_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")

    return df.select(
        *(pl.col(c).cast(pl.Float64) for c in numeric_cols),
        *(pl.col(c).cast(pl.Utf8).fill_null(MISSING_LEVEL) for c in categorical_cols),
    )


def _best_iteration(cv_result: "Any") -> tuple[str, int, float]:
    """Loss-agnostic read of the CV frame.

    The metric column is named after the loss, so it is
    'test-Poisson-mean' for Poisson but 'test-Tweedie:variance_power=1.5-mean'
    for tweedie/gamma -- hence the lookup rather than a hardcoded key.
    Unlike xgb.cv, CatBoost's cv does not truncate at the best iteration, so the
    minimum must be located rather than taking the last row.
    """
    cols = [c for c in cv_result.columns if c.startswith("test-") and c.endswith("-mean")]
    if not cols:
        raise RuntimeError(f"no test metric in cv output: {list(cv_result.columns)}")
    col = cols[0]
    return (
        col.removeprefix("test-").removesuffix("-mean"),
        int(cv_result[col].idxmin()) + 1,
        float(cv_result[col].min()),
    )


def fit_catboost(
    train: pl.DataFrame,
    response_col: str,
    exposure_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    family: Family,
    cv_folds: int,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 30,
    seed: int = 42,
    use_offset: bool = False,
    extra_params: dict[str, Any] | None = None,
) -> CatBoostResult:
    """Fit a CatBoost model with CV-selected iteration count.

    Argument order matches fit_lgbm/fit_xgb (response before features), which
    differs from the original signature -- update call sites accordingly.
    """
    if family not in _LOSS:
        raise ValueError(f"family must be one of {list(_LOSS)}, got {family!r}")
    if family == "gamma":
        warnings.warn(
            "CatBoost has no native Gamma loss; approximating with "
            f"Tweedie:variance_power={_GAMMA_VARIANCE_POWER}. For a true Gamma fit, "
            "use the LightGBM or XGBoost backend.",
            stacklevel=2,
        )
    loss = _LOSS[family]

    mode = "log-exposure offset" if use_offset else "exposure-weighted rate"
    print(f"\nCatBoost - {loss} (native categoricals, {mode}) ...", flush=True)
    timer = time.perf_counter()

    n_before = train.height
    train = train.filter(pl.col(exposure_col) > 0)
    if dropped := n_before - train.height:
        print(f"  dropped {dropped:,} rows with non-positive {exposure_col}")
    if train.is_empty():
        raise ValueError(f"no rows with positive {exposure_col}")

    exposure = train[exposure_col].cast(pl.Float64).to_numpy()
    response = train[response_col].cast(pl.Float64).to_numpy()

    if use_offset:
        # CatBoost sets boost_from_average=False for Poisson, and a baseline
        # replaces the starting raw score entirely. A bare log(exposure)
        # baseline therefore starts the model at a fitted rate of 1.0 and burns
        # most of the iteration budget boosting down to the portfolio mean.
        # Folding the log-rate in starts it at the right level; it must be
        # reapplied identically at predict time.
        intercept = float(np.log(response.sum() / exposure.sum()))
        label, weight, baseline = response, None, np.log(exposure) + intercept
    else:
        intercept = 0.0
        label, weight, baseline = response / exposure, exposure, None

    x_train = _cb_frame(train, numeric_cols, categorical_cols)
    pool = cb.Pool(
        data=x_train, label=label, weight=weight, baseline=baseline, cat_features=categorical_cols
    )

    params: dict[str, Any] = {
        "loss_function": loss,
        "iterations": num_boost_round,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "min_data_in_leaf": 50,
        "subsample": 0.9,
        "colsample_bylevel": 0.8,
        "random_seed": seed,
        "early_stopping_rounds": early_stopping_rounds,
        "logging_level": "Silent",  # cb.cv ignores verbose=False and prints per fold
        **(extra_params or {}),
    }

    cv_result = cb.cv(
        pool=pool,
        params=params,
        fold_count=cv_folds,
        shuffle=True,
        partition_random_seed=seed,
        plot=False,
        verbose=False,
    )
    metric, best_n, score = _best_iteration(cv_result)

    # Refit on the full pool for exactly best_n iterations. early_stopping_rounds
    # must be cleared -- with no eval set it has nothing to watch.
    final_params = {**params, "iterations": best_n, "early_stopping_rounds": None}
    model = cb.CatBoostRegressor(**final_params)
    model.fit(pool)
    elapsed = time.perf_counter() - timer

    print(f"  Best n_estimators={best_n}  cv_{metric}={score:.6f}  ok {elapsed:.1f}s")
    return CatBoostResult(
        model=model,
        best_n_estimators=best_n,
        time=elapsed,
        cv_metric=metric,
        cv_score=score,
        response_col=response_col,
        exposure_col=exposure_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        use_offset=use_offset,
        offset_intercept=intercept,
        loss_function=loss,
        params=params,
    )


def predict_catboost(
    res: CatBoostResult, df: pl.DataFrame, scale_by_exposure: bool = True
) -> np.ndarray:
    """Expected counts/amounts, or the underlying rate if scale_by_exposure=False.

    CatBoostRegressor.predict returns the response-scale value for log-link
    losses (equivalent to prediction_type="Exponent"), NOT the raw log-scale
    score -- verified against RawFormulaVal. So an offset-trained model returns
    expected counts directly and must not be multiplied by exposure again.
    """
    x = _cb_frame(df, res["numeric_cols"], res["categorical_cols"])
    exposure = df[res["exposure_col"]].cast(pl.Float64).to_numpy()

    if res["use_offset"]:
        baseline = np.full(df.height, res["offset_intercept"])
        if scale_by_exposure:
            baseline = baseline + np.log(np.maximum(exposure, 1e-12))
        pool = cb.Pool(data=x, baseline=baseline, cat_features=res["categorical_cols"])
        return res["model"].predict(pool)

    pool = cb.Pool(data=x, cat_features=res["categorical_cols"])
    rate = res["model"].predict(pool)
    return rate * exposure if scale_by_exposure else rate


# import time
# import numpy as np
# import pandas as pd
# from catboost import CatBoostRegressor, Pool, cv as cb_cv
# from typing import Literal, Any

# def fit_catboost(train: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str], response_col: str, exposure_col: str, family: Literal["poisson", "gamma", "tweedie"], cv_folds: int) -> "CatBoostRegressor":
    
#     print("\nCatBoost - Poisson regressor (native categoricals, log-exposure offset) ...",
#           flush=True)
    
#     timer = time.time()

#     feature_cols = numeric_cols + categorical_cols

#     x= train[feature_cols].copy()
#     for col in categorical_cols:
#         x[col] = x[col].astype(str)

#     log_exp = np.log(np.maximum(train[exposure_col].values, 1e-9))

#     pool = Pool(
#         data=x,
#         label=train[response_col].values.astype(float),
#         baseline=log_exp,                          # proper log-exposure offset
#         cat_features=categorical_cols,
#     )

#     # We need a bi of a hack based on the family to get CatBoost to work
#     # with different error functions
#     if family=="poisson":
#         family = "Poisson"
#     elif family=="tweedie":
#         family = "Tweedie:variance_power=1.5"
#     elif family=="gamma":
#         family = "Tweedie:variance_power=2.0"

#     model = CatBoostRegressor(
#         loss_function=family,
#         iterations=600,
#         learning_rate=0.02,
#         depth=6,
#         l2_leaf_reg=3.0,
#         min_data_in_leaf=50,
#         subsample=0.9,
#         colsample_bylevel=0.8,
#         random_seed=42,
#         verbose=0,
#         early_stopping_rounds=30,
#     )

#     cv_result = cb_cv(
#         pool=pool,
#         params=model.get_params(),
#         fold_count=cv_folds,
#         shuffle=True,
#         partition_random_seed=42,
#         plot=False,
#         verbose=False,
#     )

#     timer=time.time() - timer
#     best_iter = int(cv_result["test-Poisson-mean"].idxmin()) + 1
#     print(f"  Best iteration from CV: {best_iter}")

#     model.set_params(iterations=best_iter, early_stopping_rounds=None)
#     model.fit(pool)

#     res={}
#     res['model']=model
#     res['best_iteration']=best_iter
#     res['timer']=timer
#     res['cv_result']=cv_result["test-Poisson-mean"].min()
#     res['exposure_col']=exposure_col
#     res['categorical_cols']=categorical_cols
#     res['feature_cols']=feature_cols

#     print(f"  ok {timer:.1f}s")
#     return res


# def predict_catboost(
#     res: dict[str, Any],
#     df: pd.DataFrame,
# ) -> np.ndarray:
#     from catboost import Pool
#     x = df[res['feature_cols']].copy()
#     for col in res['categorical_cols']:
#         x[col] = x[col].astype(str)

#     log_exp = np.log(np.maximum(df[res['exposure_col']].values, 1e-9))

#     pool = Pool(
#         data=x,
#         baseline=log_exp,                          # must pass at predict time too
#         cat_features=res['categorical_cols'],
#     )
#     return res['model'].predict(pool)                     # no * Exposure — offset handles it