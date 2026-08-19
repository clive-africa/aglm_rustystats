"""XGBoost fitting/prediction on Polars frames, mirroring the fit_lgbm/predict_lgbm API.

Two parameterisations of the exposure relationship are supported, selected by
`use_offset`:

  use_offset=False (default)  label = response/exposure, weight = exposure.
                              Identical to fit_lgbm, so the two model families
                              are directly comparable in a benchmark.

  use_offset=True             label = response, base_margin = log(exposure).
                              The classical actuarial offset. Only valid for
                              log-link objectives (poisson, gamma, tweedie).

The two are near-equivalent for Poisson, but they are NOT interchangeable at
predict time -- see predict_xgb.
"""

import time
from typing import Any, TypedDict

import numpy as np
import polars as pl
import xgboost as xgb

Categories = dict[str, list[str]]


class XgbResult(TypedDict):
    model: xgb.Booster
    best_n_estimators: int
    time: float
    cv_metric: str
    cv_score: float
    response_col: str
    exposure_col: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    categories: Categories
    use_offset: bool
    offset_intercept: float
    params: dict[str, Any]


def _fit_categories(df: pl.DataFrame, categorical_cols: list[str]) -> Categories:
    """Level sets for each categorical, learned from the training frame only."""
    levels = df.select(
        pl.col(c).cast(pl.Utf8).drop_nulls().unique().sort().implode() for c in categorical_cols
    ).row(0)
    return dict(zip(categorical_cols, levels))


def _xgb_frame(
    df: pl.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    categories: Categories,
) -> pl.DataFrame:
    """Build the design matrix as an all-numeric Polars frame.

    XGBoost rejects Polars string columns (large_string is unsupported), so
    categoricals are encoded to integer codes here rather than handed over as
    strings. Codes are pinned by a Polars Enum fitted on train and reused at
    predict time, so they stay in sync across calls; any level unseen in
    training becomes null, which XGBoost treats as missing.

    The resulting frame is passed to DMatrix as-is -- once every column is
    numeric, XGBoost consumes a Polars frame directly and reads the feature
    names off it.
    """
    missing = [c for c in numeric_cols + categorical_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")

    return df.select(
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


def _make_dmatrix(
    x: pl.DataFrame,
    n_numeric: int,
    n_categorical: int,
    label: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    base_margin: np.ndarray | None = None,
) -> xgb.DMatrix:
    """DMatrix with explicit feature types -- 'q' quantitative, 'c' categorical.

    feature_types is positional, which is why _xgb_frame always emits numerics
    first and categoricals second.
    """
    return xgb.DMatrix(
        x,
        label=label,
        weight=weight,
        base_margin=base_margin,
        feature_types=["q"] * n_numeric + ["c"] * n_categorical,
        enable_categorical=True,
    )


def _best_score(cv: "Any") -> tuple[str, float]:
    """Objective-agnostic read of the CV result: works for poisson/gamma/tweedie.

    xgb.cv trims its output to the best iteration when early stopping fires, so
    the last row is the best row and len(cv) is the round count to refit on.
    """
    mean_cols = [c for c in cv.columns if c.startswith("test-") and c.endswith("-mean")]
    if not mean_cols:
        raise RuntimeError(f"no test metric in cv output: {list(cv.columns)}")
    col = mean_cols[0]
    return col.removeprefix("test-").removesuffix("-mean"), float(cv[col].iloc[-1])


def fit_xgb(
    train: pl.DataFrame,
    response_col: str,
    exposure_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    family: str,
    cv_folds: int,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 20,
    seed: int = 42,
    use_offset: bool = False,
    extra_params: dict[str, Any] | None = None,
) -> XgbResult:
    """Fit an XGBoost model with CV-selected n_estimators.

    `family` is an XGBoost objective string: "count:poisson", "reg:gamma",
    "reg:tweedie". For tweedie also pass
    extra_params={"tweedie_variance_power": 1.5}.
    """
    if family=="poisson":
        family="count:poisson"
    elif family=="gamma":
        family="reg:gamma"


    mode = "log-exposure offset" if use_offset else "exposure-weighted rate"
    print(f"\nXGBoost - {family} ({mode}, CV for n_estimators) ...", flush=True)
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
        # Setting base_margin OVERRIDES base_score, so an offset model has no
        # intercept and would have to boost its way from a fitted rate of 1.0
        # down to the true rate -- at learning_rate=0.01 that alone eats
        # hundreds of rounds. Folding the portfolio log-rate into the margin
        # starts the model at the right level. It must be reapplied at predict.
        intercept = float(np.log(response.sum() / exposure.sum()))
        label, weight, base_margin = response, None, np.log(exposure) + intercept
    else:
        intercept = 0.0
        label, weight, base_margin = response / exposure, exposure, None

    categories = _fit_categories(train, categorical_cols)
    x_train = _xgb_frame(train, numeric_cols, categorical_cols, categories)
    dtrain = _make_dmatrix(
        x_train, len(numeric_cols), len(categorical_cols), label, weight, base_margin
    )

    # Defaults chosen to line up with the LightGBM module as closely as XGBoost
    # allows, so a benchmark compares algorithms rather than hyperparameters:
    #   num_leaves=63     -> grow_policy="lossguide" + max_leaves=63, max_depth=0
    #   feature_fraction  -> colsample_bytree
    #   bagging_fraction  -> subsample (applied every round; no bagging_freq analogue)
    #   reg_lambda        -> forced to 0; XGBoost defaults to 1, LightGBM to 0
    # min_data_in_leaf has NO XGBoost equivalent. See the min_child_weight note below.
    params: dict[str, Any] = {
        "objective": family,
        "tree_method": "hist",  # required for categorical support
        "grow_policy": "lossguide",
        "max_leaves": 63,
        "max_depth": 0,  # 0 = unlimited, controlled by max_leaves instead
        "learning_rate": 0.01,
        # min_child_weight is a minimum sum of HESSIANS, not a row count. For
        # count:poisson the hessian is the predicted mean, so at a frequency of
        # ~0.15 a value of 50 implies roughly 300+ exposure-years per leaf --
        # far stricter than LightGBM's min_data_in_leaf=50 rows. Tune on your
        # own book rather than assuming the two numbers mean the same thing.
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
        "seed": seed,
        "nthread": -1,
        "verbosity": 0,
        **(extra_params or {}),
    }

    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=cv_folds,
        stratified=False,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    best_n = len(cv_result)
    metric, score = _best_score(cv_result)

    booster = xgb.train(params, dtrain, num_boost_round=best_n, verbose_eval=False)
    elapsed = time.perf_counter() - timer

    print(f"  Best n_estimators={best_n}  cv_{metric}={score:.6f}  ok {elapsed:.1f}s")
    return XgbResult(
        model=booster,
        best_n_estimators=best_n,
        time=elapsed,
        cv_metric=metric,
        cv_score=score,
        response_col=response_col,
        exposure_col=exposure_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        categories=categories,
        use_offset=use_offset,
        offset_intercept=intercept,
        params=params,
    )


def predict_xgb(res: XgbResult, df: pl.DataFrame, scale_by_exposure: bool = True) -> np.ndarray:
    """Expected counts/amounts, or the underlying rate if scale_by_exposure=False.

    The offset MUST be reapplied at predict time exactly as it was at fit time.
    Omitting base_margin for an offset-trained model does not return a rate --
    XGBoost silently substitutes base_score, so the prediction carries a
    spurious intercept. Both branches below therefore go through the same
    exposure handling that training used.
    """
    x = _xgb_frame(df, res["numeric_cols"], res["categorical_cols"], res["categories"])
    exposure = df[res["exposure_col"]].cast(pl.Float64).to_numpy()

    if res["use_offset"]:
        # log(exposure)+intercept reproduces training and yields expected counts
        # directly; the intercept alone isolates the rate.
        margin = np.full(df.height, res["offset_intercept"])
        if scale_by_exposure:
            margin = margin + np.log(np.maximum(exposure, 1e-12))
        dtest = _make_dmatrix(
            x, len(res["numeric_cols"]), len(res["categorical_cols"]), base_margin=margin
        )
        return res["model"].predict(dtest)

    dtest = _make_dmatrix(x, len(res["numeric_cols"]), len(res["categorical_cols"]))
    rate = res.predict(dtest)
    return rate * exposure if scale_by_exposure else rate

# import numpy as np
# import time
# import pandas as pd
# import xgboost as xgb

# def fit_xgb(train: pd.DataFrame, response_col, exposure_col, numeric_cols: list[str], categorical_cols: list[str], family: str, cv_folds: int) -> "xgb.Booster":  # noqa: F821
    
#     print("\nXGBoost - Poisson (log-exposure offset, CV for n_estimators) ...",
#           flush=True)
    
#     timer = time.time()

#     feature_cols=numeric_cols + categorical_cols

#     x = train[feature_cols].copy()
#     for col in categorical_cols:
#         x[col] = x[col].astype("category")

#     log_exp = np.log(np.maximum(train[exposure_col].values, 1e-9))

#     dtrain = xgb.DMatrix(
#         x,
#         label=train[response_col].values,
#         enable_categorical=True,

#     )
#     dtrain.set_base_margin(log_exp)       # log-exposure as proper offset

#     params = {
#         "objective":        "count:poisson",
#         "eval_metric":      "poisson-nloglik",
#         "learning_rate":    0.01,
#         "max_depth":        4,
#         "min_child_weight": 50,           # min obs per leaf — mirrors lgb min_data_in_leaf
#         "subsample":        0.9,
#         "colsample_bytree": 0.8,
#         "seed":             42,
#         "nthread":          4,
#         "verbosity":        0
#     }

#     cv_result = xgb.cv(
#         params, dtrain,
#         num_boost_round=500,
#         nfold=cv_folds,
#         stratified=False,
#         early_stopping_rounds=20,
#         callbacks=[xgb.callback.EvaluationMonitor(show_stdv=False, period=999)],
#     )


#     best_n = len(cv_result)
#     print(f"  Best n_estimators={best_n}  "
#           f"cv_poisson_nloglik={cv_result['test-poisson-nloglik-mean'].iloc[-1]:.6f}")

#     booster = xgb.train(
#         params, dtrain,
#         num_boost_round=best_n,
#         verbose_eval=False,
#     )

#     timer=time.time() - timer

#     res={}
#     res['model']=booster
#     res['timer']=timer
#     res['best_n']=best_n
#     res['cv_poisson_nloglik']=cv_result['test-poisson-nloglik-mean'].iloc[-1]

#     print(f"  ok {timer:.1f}s")
#     return res


# def predict_xgb(booster: "xgb.Booster", df: pd.DataFrame) -> np.ndarray:  # noqa: F821
    

#     X = df[FEATURE_COLS].copy()
#     for col in CATEGORICAL_COLS:
#         X[col] = X[col].astype("category")

#     log_exp = np.log(np.maximum(df["Exposure"].values, 1e-9))

#     dtest = xgb.DMatrix(X, enable_categorical=True)
#     #dtest.set_base_margin(log_exp)        # must pass offset at predict time too

#     return booster.predict(dtest)*df["Exposure"].values         # returns predicted 
