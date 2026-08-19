from __future__ import annotations
import pandas as pd
import rustystats as rs
import time
import numpy as np
from Typing import Any, Literal


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


def predict_glm(result: rs.GLMModel, df: pd.DataFrame) -> np.ndarray:
    # predict() picks up the offset column ("Exposure") automatically from df
    return np.asarray(result.predict(df))

def fit_reg_glm(train: pd.DataFrame, response_col: str, offset_col: str, 
            numeric_cols: list[str], categorical_cols: list[str], 
            family: Literal["poisson", "gamma", "tweedie"],
            cv_folds: int|None) -> dict[str, Any]:
    
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