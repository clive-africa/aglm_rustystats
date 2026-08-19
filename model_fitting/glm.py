"""GLM and regularised GLM via rustystats, on Polars frames.

rustystats is already Polars-native (`Requires: numpy, polars`) and accepts
pl.DataFrame or pl.LazyFrame directly -- the pandas type hints in the original
were never accurate. Passing a pandas frame to `.predict()` fails with the
opaque "'Series' object has no attribute 'estimated_size'"; passing one to
`glm_dict(data=...)` fails with "'DataFrame' object has no attribute 'schema'".
_require_polars below turns both into a message that says what is wrong.

LazyFrames are passed through untouched where possible: rustystats collects only
the columns the model needs, which is worth keeping for Parquet/CSV scans.
"""

import time
from typing import Any, Literal, TypedDict

import numpy as np
import polars as pl
import rustystats as rs

Family = Literal["poisson", "gamma", "tweedie"]
Frame = pl.DataFrame | pl.LazyFrame


class GlmResult(TypedDict):
    model: rs.GLMModel
    time: float
    deviance: float
    converged: bool
    iterations: int
    n_params: int
    n_nonzero: int
    alpha: float | None
    response_col: str
    exposure_col: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    family: str
    regularization: str | None


def _require_polars(df: Any, argname: str) -> Frame:
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(
            f"{argname} must be a polars DataFrame or LazyFrame, got "
            f"{type(df).__module__}.{type(df).__qualname__}. rustystats is "
            "polars-native; convert with pl.from_pandas(df)."
        )
    return df


def _build_terms(numeric_cols: list[str], categorical_cols: list[str]) -> dict[str, dict[str, str]]:
    return {
        **{c: {"type": "linear"} for c in numeric_cols},
        **{c: {"type": "categorical"} for c in categorical_cols},
    }


def _drop_zero_exposure(df: Frame, exposure_col: str) -> Frame:
    """Non-positive exposure would make log(exposure) -inf in the offset."""
    if isinstance(df, pl.LazyFrame):
        return df.filter(pl.col(exposure_col) > 0)
    kept = df.filter(pl.col(exposure_col) > 0)
    if dropped := df.height - kept.height:
        print(f"  dropped {dropped:,} rows with non-positive {exposure_col}")
    if kept.is_empty():
        raise ValueError(f"no rows with positive {exposure_col}")
    return kept


def _n_nonzero(model: rs.GLMModel) -> int:
    n = model.n_nonzero
    return int(n() if callable(n) else n)


def _fit(
    train: Frame,
    response_col: str,
    exposure_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    family: Family,
    label: str,
    regularization: str | None = None,
    cv_folds: int | None = None,
    selection: str = "min",
    var_power: float = 1.5,
    seed: int = 42,
    fit_kwargs: dict[str, Any] | None = None,
) -> GlmResult:
    """Shared fit path for the plain and regularised GLMs."""
    train = _require_polars(train, "train")
    print(f"\n{label} - {family} GLM (rustystats) ...", flush=True)
    timer = time.perf_counter()

    train = _drop_zero_exposure(train, exposure_col)

    spec = rs.glm_dict(
        response=response_col,
        terms=_build_terms(numeric_cols, categorical_cols),
        data=train,
        family=family,
        # exposure=, NOT offset=. rustystats adds log(exposure) to the linear
        # predictor for exposure=, while offset= is added VERBATIM on the link
        # scale -- the docs are explicit that an offset is never treated as raw
        # exposure. Passing a raw exposure column as offset= fits
        # mu = exp(X.beta) * exp(exposure) instead of mu = exp(X.beta) * exposure,
        # which is a materially different (and wrong) model.
        exposure=exposure_col,
        var_power=var_power,  # only consulted for family="tweedie"
        weights=None,
        seed=seed,
    )

    model = spec.fit(**(fit_kwargs or {}))
    elapsed = time.perf_counter() - timer

    n_params, n_nz = len(model.params), _n_nonzero(model)
    alpha = float(model.alpha) if regularization else None

    if regularization:
        print(f"  Selected alpha={alpha:.6f}  non-zero features={n_nz}/{n_params}")
    print(
        f"  ok {elapsed:.1f}s | deviance={model.deviance:.4f} | "
        f"converged={model.converged} | iterations={model.iterations}"
    )

    return GlmResult(
        model=model,
        time=elapsed,
        deviance=float(model.deviance),
        converged=bool(model.converged),
        iterations=int(model.iterations),
        n_params=n_params,
        n_nonzero=n_nz,
        alpha=alpha,
        response_col=response_col,
        exposure_col=exposure_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        family=family,
        regularization=regularization,
    )


def fit_glm(
    train: Frame,
    response_col: str,
    exposure_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    family: Family,
    var_power: float = 1.5,
    seed: int = 42,
) -> GlmResult:
    """Unregularised GLM with a log(exposure) offset."""
    return _fit(
        train,
        response_col,
        exposure_col,
        numeric_cols,
        categorical_cols,
        family,
        label="GLM",
        var_power=var_power,
        seed=seed,
    )


def fit_reg_glm(
    train: Frame,
    response_col: str,
    exposure_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    family: Family,
    cv_folds: int | None,
    regularization: str = "elastic_net",
    selection: str = "min",
    var_power: float = 1.5,
    seed: int = 42,
) -> GlmResult:
    """Regularised GLM with the penalty strength chosen by cross-validation."""
    return _fit(
        train,
        response_col,
        exposure_col,
        numeric_cols,
        categorical_cols,
        family,
        label="RegGLM",
        regularization=regularization,
        cv_folds=cv_folds,
        selection=selection,
        var_power=var_power,
        seed=seed,
        fit_kwargs={
            "regularization": regularization,
            "selection": selection,
            "cv": cv_folds,
            "cv_seed": seed,
        },
    )


def predict_glm(res: GlmResult, df: Frame, scale_by_exposure: bool = True) -> np.ndarray:
    """Expected counts/amounts, or the underlying rate if scale_by_exposure=False.

    rustystats reads the exposure column off `df` by name and reapplies the
    log(exposure) offset itself, so predict already returns expected counts --
    do not multiply by exposure again. Verified: predicted total equals the
    observed total on the training frame.
    """
    df = _require_polars(df, "df")
    mu = np.asarray(res.predict(df))
    if scale_by_exposure:
        return mu
    exposure = (
        df.select(res["exposure_col"]).collect() if isinstance(df, pl.LazyFrame) else df
    )[res["exposure_col"]].cast(pl.Float64).to_numpy()
    return mu / np.maximum(exposure, 1e-12)


# fit_reg_glm shares predict_glm -- both return an rs.GLMModel in res["model"].
def predict_reg_glm(res: GlmResult, df: Frame, scale_by_exposure: bool = True) -> np.ndarray:
    return predict_glm(res, df, scale_by_exposure)

# from __future__ import annotations
# import pandas as pd
# import rustystats as rs
# import time
# import numpy as np
# from typing import Any, Literal


# def fit_glm(train: pd.DataFrame, response_col: str, offset_col: str, 
#             numeric_cols: list[str], categorical_cols: list[str], 
#             family: Literal["poisson", "gamma", "tweedie"] ) -> dict[str, Any]:  # noqa: F821
    
#     print("\n[GLM - Plain Poisson (rustystats) ...", flush=True)
    
#     #Just some basic statistics measurements to help understand the process
#     timer = time.time()

#     terms = {col: {"type": "linear"}      for col in numeric_cols}
#     terms.update({col: {"type": "categorical"} for col in categorical_cols})

#     result = rs.glm_dict(
#         response=response_col,
#         terms=terms,
#         data=train,
#         family=family,
#         offset=offset_col,
#         weights=None,
#         seed=42,
#     ).fit()

#     timer=time.time()-timer

#     # Get the summary of the model
#     res={}
#     res['model']=result
#     res['timer']=timer
#     res['deviance']=result.deviance
#     res['converged']=result.converged
#     res['iterations']=result.iterations

#     print(f"  ok {timer:.1f}s | deviance = {result.deviance:.4f} | "
#           f"converged = {result.converged} | iterations = {result.iterations}")

#     return res


# def predict_glm(result: rs.GLMModel, df: pl.DataFrame) -> np.ndarray:
#     # predict() picks up the offset column ("Exposure") automatically from df
#     return np.asarray(result.predict(df))

# def fit_reg_glm(train: pd.DataFrame, response_col: str, offset_col: str, 
#             numeric_cols: list[str], categorical_cols: list[str], 
#             family: Literal["poisson", "gamma", "tweedie"],
#             cv_folds: int|None) -> dict[str, Any]:
    
#     print("\nRegGLM - Regularised Poisson GLM"
#           " (rustystats elastic_net CV) ...", flush=True)
    
#     timer = time.time()

#     terms = {col: {"type": "linear"}          for col in numeric_cols}
#     terms.update({col: {"type": "categorical"} for col in categorical_cols})

#     result = rs.glm_dict(
#         response=response_col,
#         terms=terms,
#         data=train,
#         family=family,
#         offset=offset_col,
#         weights=None,
#         seed=42,
#     ).fit(
#         regularization="elastic_net",
#         selection="min",
#         cv=cv_folds,
#         cv_seed=42,
#     )

#     timer = time.time() - timer

#     res={}
#     res['model']=result
#     res['timer']=timer
#     res['alpha']=result.alpha
#     res['non-zero_features']=result.n_nonzero()/len(result.params)

#     print(f"  Selected alpha={result.alpha:.6f}  "
#           f"non-zero features={result.n_nonzero()}/{len(result.params)}")
#     print(f"  ok {timer:.1f}s")
    
#     return res


# def predict_reg_glm(result: "rs.GLMModel", df: pd.DataFrame) -> np.ndarray:
#     return np.asarray(result.predict(df))