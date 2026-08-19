"""
fremtpl2_experiment.py
======================
Replication of Fujita et al. (2020) Section 5 numerical experiments.

Fits eight models on the freMTPL2freq motor-insurance frequency dataset:

  1. GLM          — plain Poisson GLM (statsmodels IRLS, log-exposure offset)
  2. RegGLM       — regularised Poisson GLM; sklearn PoissonRegressor + GridSearchCV
                    over L2 penalty strength.  Equivalent to R's cv.glmnet(family=
                    "poisson", alpha=0) on a standard one-hot design matrix.
  3. AGLM-Lin     — AGLM without basis expansion (linear + one-hot only).
                    Uses the RustyStats IRLS engine with elastic-net CV over both
                    lambda and alpha.  Comparable to RegGLM but with L1/L2 mixing.
  4. AGLM-Lvar    — Full AGLM with L-variable basis (Fujita et al. Table 8 "AGLM").
                    Augments the design matrix with |x - tk| tent functions before
                    elastic-net regularisation.
  5. GAM          — PoissonGAM, spline + factor terms (pygam)
  6. GBM          — LightGBM Poisson (lgb.cv for n_estimators)
  7. CatBoost     — CatBoost Poisson regressor (native ordered categorical encoding)
  8. DerivLasso   — Derivative (fused) Lasso GLM via CVXPY (Akur8-style)

Metrics reported
----------------
  - Poisson Deviance  — Eq. 11 from Fujita et al. (2x convention)
  - MSE               — mean squared error on claim frequency (ClaimNb / Exposure)
  - MAE               — mean absolute error on claim frequency
  - AUC               — ROC-AUC of predicted frequency vs binary (any claim)
  - Avg Pred (freq)   — exposure-weighted mean predicted frequency
  - Avg Actual (freq) — exposure-weighted mean actual frequency (same for all)
"""

from __future__ import annotations


import pathlib
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml

from sklearn.model_selection import KFold, train_test_split

from plots.plots import plot_component_curves, plot_metrics_comparison, plot_calibration

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

N_SAMPLE = None  # None = use full OpenML dataset; set e.g. 50_000 for dev
NBIN_MAX = 40  # max bins per numeric variable in the AGLM basis
N_ALPHAS = 10  # lambda grid size for cva_aglm
ALPHA_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def load_and_split(
    df: pd.DataFrame,
    seed: int | None = 42,
    sample_size: int | None = None,
    claim_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    if sample_size is not None:
        df = df.sample(sample_size, random_state=seed).reset_index(drop=True)

    if claim_col is not None:
        strat = (df[claim_col] > 0).astype(int)
        train, test = train_test_split(
            df, test_size=0.25, random_state=seed, stratify=strat
        )
    else:
        train, test = train_test_split(df, test_size=0.25, random_state=seed)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # ── summary table ────────────────────────────────────────────────────────
    # Produce a nice summary table for the users to understadn some high level metrics of the split
    header = f"{'Dataset':<12} {'Records':>10} {'% of Total':>11}"
    if claim_col is not None:
        header += f" {'Claim Rate':>12}"
    sep = "─" * len(header)

    rows = [
        ("Original", df, len(df)),
        ("Train", train, len(df)),
        ("Test", test, len(df)),
    ]

    summary = {}

    print(sep)
    print(header)
    print(sep)
    for label, subset, total in rows:
        n = len(subset)
        pct = n / total * 100
        entry = {"n": n, "pct_of_total": round(pct, 1)}
        if claim_col is not None:
            cr = subset[claim_col].mean()
            entry["claim_rate"] = round(cr, 6)
        summary[label] = entry

        row = f"{label:<12} {n:>10,} {pct:>10.1f}%"
        if claim_col is not None:
            row += f" {cr:>11.4f}"
        print(row)
    print(sep)

    return train, test, summary


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    response_col: str,
    exposure_col: str,
    offset_col: str,
    sample_size: int | None = None,
    num_bin: int = NBIN_MAX,
    n_alphas: int = N_ALPHAS,
    alpha_grid: np.ndarray | None = ALPHA_GRID,
    cv_folds: int = CV_FOLDS,
    run_gam: bool = True,
    run_gbm: bool = True,
    run_catboost: bool = True,
    run_xgb: bool = True,
    run_deriv_lasso: bool = True,
) -> None:

    print("=" * 70)
    print("Fujita et al. (2020) - Section 5 Numerical Experiments")
    print("  Models: GLM | RegGLM | AGLM-Lin | AGLM-Lvar |")
    print("          GAM | GBM | CatBoost | DerivLasso")
    print("=" * 70)
    t_total = time.time()

    train, test, summ_stats = load_and_split(df=df)

    print()
    print("─" * 70)
    print(
        f"Fitting models  | nbin={NBIN_MAX}  n_alphas={N_ALPHAS}"
        f"  alpha_grid={ALPHA_GRID.tolist()}  cv={CV_FOLDS}"
    )
    print("─" * 70)

    times = {}

    glm_m = fit_glm(
        train := train,
        response_col := response_col,
        offset_col := offset_col,
        numeric_cols := numeric_cols,
        categorical_cols := categorical_cols,
        family := "poisson",
    )  # , ref_levels)   # sklearn GLM with log-exposure offset
    reg_glm_m = fit_reg_glm(
        train := train,
        response_col := response_col,
        offset_col := offset_col,
        numeric_cols := numeric_cols,
        categorical_cols := categorical_cols,
        family := "poisson",
        cv_folds=cv_folds,
    )
    aglm_lin_m = fit_aglm_linear(
        train := train,
        response_col=response_col,
        exposure_col=exposure_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        family="poisson",
        alpha_grid=alpha_grid,
        cv_folds=cv_folds,
        n_alphas=n_alphas,
        nbin_max=num_bin,
    )  # AGLM, no basis
    aglm_lvar_m = fit_aglm_lvar(
        train=train,
        response_col=response_col,
        exposure_col=exposure_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        family="poisson",
        alpha_grid=alpha_grid,
        cv_folds=cv_folds,
        n_alphas=n_alphas,
        nbin_max=num_bin,
    )  # AGLM, L-variable basis
    gam_m = (
        fit_gam(
            train=train,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            response_col=response_col,
            exposure_col=exposure_col,
        )
        if run_gam
        else None
    )
    gbm_m = (
        fit_gbm(
            train=train,
            response_col=response_col,
            exposure_col=exposure_col,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            family="poisson",
            cv_folds=cv_folds,
        )
        if run_gbm
        else None
    )
    xgb_m = (
        fit_xgb(
            train=train,
            response_col=response_col,
            exposure_col=exposure_col,
            family="poisson",
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            cv_folds=cv_folds,
        )
        if run_xgb
        else None
    )
    catboost_m = (
        fit_catboost(
            train=train,
            response_col=response_col,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            exposure_col=exposure_col,
            family="poisson",
            cv_folds=cv_folds,
        )
        if run_catboost
        else None
    )
    deriv_lasso_m = (
        fit_derivative_lasso(
            train=train,
            response_col=response_col,
            exposure_col=exposure_col,
            cv_folds=cv_folds,
        )
        if run_deriv_lasso
        else None
    )

    print()
    print("─" * 70)
    print("Full Metrics Table - Hold-out Test Set")
    print("─" * 70)

    predict_fns: dict = {
        "GLM": lambda df: predict_glm(
            glm_m["model"], df
        ),  # , glm_ncols, df, ref_levels),
        "RegGLM": lambda df: predict_reg_glm(reg_glm_m["model"], df),
        "AGLM-Lin": lambda df: predict_aglm_linear(aglm_lin_m["model"], df),
        "AGLM-Lvar": lambda df: predict_aglm_lvar(aglm_lvar_m["model"], df),
    }
    if gam_m is not None:
        predict_fns["GAM"] = lambda df: predict_gam(gam_m, df)
    if gbm_m is not None:
        predict_fns["GBM"] = lambda df: predict_gbm_weights(gbm_m, df)
    if xgb_m is not None:
        predict_fns["XGB"] = lambda df: predict_xgb(xgb_m, df)
    if catboost_m is not None:
        predict_fns["CatBoost"] = lambda df: predict_catboost(catboost_m, df)
    if deriv_lasso_m is not None:
        predict_fns["DerivLasso"] = lambda df: predict_derivative_lasso(
            deriv_lasso_m, df
        )

    rows = []
    for name, fn in predict_fns.items():
        mu = fn(test)
        row = compute_metrics(
            name, test[response_col].values, mu, test[exposure_col].values
        )
        rows.append(row)
        print(
            f"  {name:12s}: dev={row['Poisson Deviance']:.6f}  "
            f"mse={row['MSE']:.8f}  mae={row['MAE']:.6f}  "
            f"auc={row['AUC']:.4f}  gini={row['Gini']:.4f} "
            f"avg_pred={row['Avg Pred (freq)']:.5f} "
            f"med_pred={row['Median Pred (freq)']:.5f}"
        )

    metrics_table = (
        pd.DataFrame(rows).sort_values("Poisson Deviance").reset_index(drop=True)
    )

    print()
    print("─" * 132)
    hdr = (
        f"  {'Model':<12}  {'Poisson Dev':>13}  {'MSE':>12}  "
        f"{'MAE':>10}  {'AUC':>7}  {'Gini':>7} {'Avg Pred':>10}  {'Med Pred':>10}  "
        f"{'Avg Actual':>10}  {'Med Actual':>10}"
    )
    print(hdr)
    print("  " + "─" * 128)
    for _, r in metrics_table.iterrows():
        print(
            f"  {r['Model']:<12}  {r['Poisson Deviance']:>13.7f}  "
            f"{r['MSE']:>12.9f}  {r['MAE']:>10.7f}  "
            f"{r['AUC']:>7.4f} {r['Gini']:>7.4f}  {r['Avg Pred (freq)']:>10.6f}  "
            f"{r['Median Pred (freq)']:>10.6f}  {r['Avg Actual (freq)']:>10.6f}  "
            f"{r['Median Actual (freq)']:>10.6f}"
        )
    print("─" * 132)

    print()
    print("Fujita et al. Table 8 reference values (full N=678k, Poisson Deviance):")
    print(
        pd.DataFrame(
            {
                "Model": ["AGLM-Lvar", "GBM", "GAM", "GLM", "AGLM-Lin"],
                "Paper deviance": [
                    0.3111920,
                    0.3123919,
                    0.3171236,
                    0.3201199,
                    0.3201245,
                ],
            }
        ).to_string(index=False)
    )

    print("\nGenerating figures ...")

    fig2 = plot_component_curves(aglm_lvar_m, train, metrics_table)
    fig2.savefig("fremtpl2_figure2_component_curves.png", dpi=130, bbox_inches="tight")
    print("  Saved: fremtpl2_figure2_component_curves.png")
    plt.close(fig2)

    fig3 = plot_metrics_comparison(metrics_table)
    fig3.savefig(
        "fremtpl2_figure3_metrics_comparison.png", dpi=130, bbox_inches="tight"
    )
    print("  Saved: fremtpl2_figure3_metrics_comparison.png")
    plt.close(fig3)

    fig4 = plot_calibration(metrics_table)
    fig4.savefig("fremtpl2_figure4_calibration.png", dpi=130, bbox_inches="tight")
    print("  Saved: fremtpl2_figure4_calibration.png")
    plt.close(fig4)

    print(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} minutes")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # We use teh french third party dataset for our testing
    print("Loading freMTPL2freq from OpenML ...")
    df = fetch_openml(data_id=41214, as_frame=True, parser="pandas")["data"]

    # Peform some basic data cleaning and type conversions for the dataset
    for col in ["ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["Exposure", "Density"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["VehBrand", "VehGas", "Area", "Region"]:
        df[col] = df[col].astype(str).str.strip()

    before = len(df)
    df = df.dropna(
        subset=[
            "ClaimNb",
            "VehPower",
            "VehAge",
            "DrivAge",
            "BonusMalus",
            "Exposure",
            "Density",
            "VehBrand",
            "VehGas",
            "Area",
            "Region",
        ]
    )
    print(f"  Dropped {before - len(df):,} rows with missing values")

    df["Exposure"] = df["Exposure"].clip(lower=1e-4)
    df["ClaimNb"] = df["ClaimNb"].astype(int).clip(upper=4)
    df = df.drop(columns=["IDpol"], errors="ignore")

    # Define our numeric and categorical columns
    NUMERIC_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    CATEGORICAL_COLS = ["VehBrand", "VehGas", "Area", "Region"]

    common = dict(
    train=df, response_col="ClaimNb", offset_col="Exposure",
    exposure_col="Exposure", numeric_cols=NUMERIC_COLS,
    categorical_cols=CATEGORICAL_COLS, family='poisson', cv_folds=CV_FOLDS,
    alpha_grid=ALPHA_GRID, n_alphas=N_ALPHAS, nbin_max=NBIN_MAX,
    )


    # The main model fitting routines happen here, with all models and metrics computed and printed, and figures generated
    res = main(
        df=df,
        categorical_cols=CATEGORICAL_COLS,
        numeric_cols=NUMERIC_COLS,
        response_col="ClaimNb",
        exposure_col="Exposure",
        offset_col="Exposure",
        sample_size=None,
        num_bin=NBIN_MAX,
        n_alphas=N_ALPHAS,
        alpha_grid=ALPHA_GRID,
        cv_folds=CV_FOLDS,
        run_gam=True,
        run_gbm=True,
        run_catboost=True,
        run_xgb=True,
        run_deriv_lasso=True,
    )
