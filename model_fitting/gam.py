import pandas as pd
import numpy as np
import time



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