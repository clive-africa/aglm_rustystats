import pandas as pd
import numpy as np
import time
import cvxpy as cp



def _engineer_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_bins: int = 40,
    exposure_col: str | None = None,
    other_label: str = "Other",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Generic feature engineering:
      - Numeric cols  → quantile-binned into n_bins ordered categories (_bin suffix)
      - Categorical cols → top n_bins levels by exposure, rest → other_label (_cap suffix)

    Returns
    -------
    out                 : transformed DataFrame (original cols preserved)
    new_numeric_bins    : list of new binned column names
    new_categorical_caps: list of new capped column names
    """
    out = df.copy()
    new_numeric_bins: list[str]     = []
    new_categorical_caps: list[str] = []

    # ── numeric → quantile bins ───────────────────────────────────────────────
    for col in numeric_cols:
        new_col = f"{col}_bin"
        try:
            out[new_col] = pd.qcut(
                out[col].astype(float),
                q=n_bins,
                duplicates="drop",   # handles low-cardinality / skewed cols
            ).astype(str)
        except ValueError:
            # Fewer unique values than requested bins — fall back to all unique values
            out[new_col] = out[col].astype(str)
        new_numeric_bins.append(new_col)

    # ── categorical → top-N by exposure, rest → Other ────────────────────────
    for col in categorical_cols:
        new_col = f"{col}_cap"
        series  = out[col].astype(str)

        if exposure_col is not None:
            # Sum exposure per category, keep top n_bins
            exposure_by_cat = (
                out.groupby(series)[exposure_col]
                .sum()
                .sort_values(ascending=False)
            )
        else:
            # Fall back to record count
            exposure_by_cat = series.value_counts()

        top_levels = set(exposure_by_cat.head(n_bins-1).index)
        out[new_col] = series.where(series.isin(top_levels), other=other_label)
        new_categorical_caps.append(new_col)

    return out, new_numeric_bins, new_categorical_caps

def _engineer_features_dl(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Bin continuous predictors into ordered categorical groups."""
    out = df.copy()
    out["DrivAgeBin"] = pd.cut(
        out["DrivAge"],
        bins=[17, 22, 26, 30, 40, 50, 60, 70, 101],
        labels=["18-22", "23-26", "27-30", "31-40",
                "41-50", "51-60", "61-70", "71+"],
        include_lowest=True, ordered=True,
    )
    out["BonusMalusBin"] = pd.cut(
        out["BonusMalus"],
        bins=[49, 60, 70, 80, 100, 120, 150, 231],
        labels=["50-60", "61-70", "71-80", "81-100",
                "101-120", "121-150", "151+"],
        include_lowest=True, ordered=True,
    )
    out["VehAgeBin"] = pd.cut(
        out["VehAge"],
        bins=[0, 1, 3, 6, 10, 15, 101],
        labels=["0-1", "2-3", "4-6", "7-10", "11-15", "16+"],
        include_lowest=True, ordered=True,
    )
    out["VehPowerBin"] = pd.cut(
        out["VehPower"],
        bins=[3, 5, 7, 9, 11, 15],
        labels=["4-5", "6-7", "8-9", "10-11", "12+"],
        include_lowest=True, ordered=True,
    )
    out["DensityBin"] = pd.qcut(
        np.log1p(out["Density"].astype(float)),
        q=6,
        labels=["D1", "D2", "D3", "D4", "D5", "D6"],
        duplicates="drop",
    )
    return out


class _DLDesignMatrix:
    """Design matrix builder with fit_transform / transform pair."""

    DL_ORDERED = ["DrivAgeBin", "BonusMalusBin", "VehAgeBin",
                  "VehPowerBin", "DensityBin"]
    DL_NOMINAL = ["VehGas", "Area"]

    def __init__(self):
        self.feature_groups_: dict = {}
        self.feature_names_:  list = []
        self.col_order_:      dict = {}
        self.n_features_:     int  = 0

    def fit_transform(self, df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> np.ndarray:
        n = len(df)
        blocks, col = [], 0

        blocks.append(np.ones((n, 1)))
        self.feature_groups_["intercept"] = [col]
        self.feature_names_.append("intercept")
        col += 1

        for var in numeric_cols:
            dummies = pd.get_dummies(df[var], prefix=var,
                                     drop_first=False, dtype=float)
            k = dummies.shape[1]
            blocks.append(dummies.values)
            self.feature_groups_[var] = list(range(col, col + k))
            self.feature_names_.extend(dummies.columns.tolist())
            self.col_order_[var] = dummies.columns.tolist()
            col += k

        for var in categorical_cols:
            dummies = pd.get_dummies(df[var], prefix=var,
                                     drop_first=True, dtype=float)
            k = dummies.shape[1]
            blocks.append(dummies.values)
            self.feature_groups_[var] = list(range(col, col + k))
            self.feature_names_.extend(dummies.columns.tolist())
            self.col_order_[var] = dummies.columns.tolist()
            col += k

        self.n_features_ = col
        X = np.hstack(blocks)
        print(f"  Design matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the encoding learned in fit_transform to new data."""
        blocks = [np.ones((len(df), 1))]
        for var in self.DL_ORDERED:
            dummies = pd.get_dummies(df[var], prefix=var,
                                     drop_first=False, dtype=float)
            for c in self.col_order_[var]:
                if c not in dummies.columns:
                    dummies[c] = 0.0
            blocks.append(dummies[self.col_order_[var]].values)
        for var in self.DL_NOMINAL:
            dummies = pd.get_dummies(df[var], prefix=var,
                                     drop_first=False, dtype=float)
            for c in self.col_order_[var]:
                if c not in dummies.columns:
                    dummies[c] = 0.0
            blocks.append(dummies[self.col_order_[var]].values)
        return np.hstack(blocks)


class _DerivativeLassoGLM:
    """Poisson GLM with first-difference lasso penalty, solved via CVXPY."""

    def __init__(self, lam: float = 0.05):
        self.lam            = lam
        self.coef_:          np.ndarray | None = None
        self.solve_status_:  str = ""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray | None = None,
        ordered_groups: dict | None = None,
    ) -> "_DerivativeLassoGLM":
        

        n, p  = X.shape
        exp_  = np.ones(n) if exposure is None else np.asarray(exposure, float)
        log_e = np.log(np.maximum(exp_, 1e-12))

        beta  = cp.Variable(p)
        eta   = X @ beta + log_e
        loss  = cp.sum(cp.exp(eta) - cp.multiply(y, eta))

        diff_list = []
        if ordered_groups:
            for var, idx in ordered_groups.items():
                if len(idx) > 1:
                    b_v = beta[idx]
                    for j in range(len(idx) - 1):
                        diff_list.append(cp.abs(b_v[j + 1] - b_v[j]))

        penalty = (cp.sum(cp.hstack(diff_list)) if diff_list
                   else cp.Constant(0))

        prob = cp.Problem(cp.Minimize(loss + self.lam * penalty))
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-4)

        self.solve_status_ = prob.status
        if beta.value is None:
            raise RuntimeError(f"DerivLasso solver failed: {prob.status}")
        self.coef_ = beta.value
        return self

    def predict(self, X: np.ndarray,
                exposure: np.ndarray | None = None) -> np.ndarray:
        exp_  = np.ones(len(X)) if exposure is None else np.asarray(exposure, float)
        log_e = np.log(np.maximum(exp_, 1e-12))
        return np.exp(X @ self.coef_ + log_e)


def fit_derivative_lasso(train: pd.DataFrame, response_col: str, exposure_col: str, numeric_cols: list[str], categorical_cols: list[str], num_bins: int, cv_folds: int) -> dict:
    print("\n[8/8] DerivLasso - derivative fused lasso Poisson (CVXPY) ...",
          flush=True)
    t0 = time.time()

    # if len(train) > DERIV_LASSO_MAX_N:
    #     strat = (train[response_col] > 0).astype(int)
    #     sub, _ = train_test_split(
    #         train, train_size=DERIV_LASSO_MAX_N,
    #         random_state=42, stratify=strat,
    #     )
    #     sub = sub.reset_index(drop=True)
    #     print(f"  Subsampled {DERIV_LASSO_MAX_N:,} rows for CVXPY solver "
    #           f"(full train = {len(train):,})")
    # else:
    #   sub = train.copy()

    sub=train.copy()

    sub_eng = _engineer_features_dl(sub, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    dm      = _DLDesignMatrix()

    X       = dm.fit_transform(sub_eng, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    y       = sub[response_col].values.astype(float)
    exp_    = sub[exposure_col].values.astype(float)

    ordered_groups = {v: dm.feature_groups_[v] for v in dm.DL_ORDERED}

    lam_grid = [0.005, 0.02, 0.05, 0.10, 0.25, 0.50]
    kf       = KFold(n_splits=cv_folds, shuffle=True, random_state=1)
    print(f"  Lambda CV over {lam_grid} ({cv_folds}-fold) ...")

    cv_devs = []
    for lam in lam_grid:
        fold_devs = []
        for tr_idx, va_idx in kf.split(X):
            m = _DerivativeLassoGLM(lam=lam)
            m.fit(X[tr_idx], y[tr_idx],
                  exposure=exp_[tr_idx],
                  ordered_groups=ordered_groups)
            fold_devs.append(
                _poisson_deviance(y[va_idx], m.predict(X[va_idx], exp_[va_idx]))
            )
        mean_dev = float(np.mean(fold_devs))
        cv_devs.append(mean_dev)
        print(f"    lambda={lam:.3f}  cv_dev={mean_dev:.6f}")

    best_lam = lam_grid[int(np.argmin(cv_devs))]
    print(f"  Best lambda by CV deviance: {best_lam}")

    best_model = _DerivativeLassoGLM(lam=best_lam)
    best_model.fit(X, y, exposure=exp_, ordered_groups=ordered_groups)
    print(f"  Solver status: {best_model.solve_status_}")
    print(f"  ok {time.time() - t0:.1f}s")

    return {"model": best_model, "dm": dm,
            "best_lam": best_lam, "ordered_groups": ordered_groups}


def predict_derivative_lasso(fitted: dict, df: pd.DataFrame) -> np.ndarray:
    X   = fitted["dm"].transform(_engineer_features_dl(df))
    exp = df["Exposure"].values.astype(float)
    return fitted["model"].predict(X, exp)