"""
fremtpl2_experiment.py
======================
Replication of Fujita et al. (2020) Section 5 numerical experiments.

Fits five models on the freMTPL2freq motor-insurance frequency dataset and
compares their hold-out Poisson deviance against the paper's Table 8:

  1. GLM      — plain Poisson GLM (statsmodels IRLS, log-exposure offset)
  2. RegGLM   — regularised GLM, linear + one-hot only (aglm / RustyStats)
  3. AGLM     — augmented GLM, L-variable basis (aglm / RustyStats)
  4. GAM       — PoissonGAM, spline + factor terms (pygam)
  5. GBM       — LightGBM Poisson (lgb.cv for n_estimators)

Changes from the original irls_en_cv version
---------------------------------------------
The old ``irls_en_cv`` was a hand-rolled outer IRLS loop that called
``sklearn.ElasticNetCV / RidgeCV`` on the working-response at each step.
It has been replaced by ``aglm.cva_aglm``, which:

  * uses the Rust IRLS engine (RustyStats ``fit_glm_py``) — no outer loop needed
  * runs the full regularisation path + K-fold CV in parallel via Rayon
  * searches over both lambda (regularisation strength) and alpha (L1 mixing)
    via ``cva_aglm``

Offset handling
---------------
``cva_aglm`` does not yet accept an explicit offset argument.  The standard
workaround is to include ``log(Exposure)`` as a numeric predictor; the model
recovers a coefficient close to 1.0 asymptotically.  The plain GLM still uses
the proper statsmodels offset so its deviance is directly comparable to the
paper.  GAM and GBM use exposure as a weight / multiplier (same as before).

Usage
-----
    python fremtpl2_experiment.py [--n 50000] [--no-gam] [--no-gbm]

Dependencies
------------
    pip install rustystats pygam lightgbm scikit-learn statsmodels matplotlib
    # aglm package must be on sys.path (parent of the aglm/ folder)
"""

from __future__ import annotations

import argparse
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
import statsmodels.api as sm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from aglm import cv_aglm, cva_aglm
from generate_fremtpl2freq import make_fremtpl2freq


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

N_SAMPLE   = None     # None → use full OpenML dataset; set e.g. 50_000 for dev
NBIN_MAX   = 40       # max bins per numeric variable in the AGLM basis
N_ALPHAS   = 10       # lambda grid size for cv_aglm
ALPHA_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # L1-mixing values for cva_aglm
CV_FOLDS   = 5

NUMERIC_COLS     = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEGORICAL_COLS = ["VehBrand", "VehGas", "Area", "Region"]
FEATURE_COLS     = NUMERIC_COLS + CATEGORICAL_COLS

# Feature columns supplied to cva_aglm — includes LogExposure as first column
# to act as a near-offset (coefficient → 1.0 as n → ∞).
AGLM_COLS = ["LogExposure"] + FEATURE_COLS

COLORS = {
    "AGLM":   "#1A3C6E",
    "GLM":    "#6C8EAD",
    "RegGLM": "#A8C0D6",
    "GAM":    "#5C8A5C",
    "GBM":    "#C0392B",
}

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def poisson_deviance_eq11(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance — 2× convention used by R GLM / sklearn / paper.

    Paper Eq. 11 (without the 2) gives values that are exactly half of every
    R and Python implementation.  We follow the 2× convention so numbers are
    directly comparable to Table 8.
    """
    y  = np.asarray(y_true, float)
    mu = np.maximum(np.asarray(y_pred, float), 1e-12)
    return float(2.0 * np.mean(
        np.where(y > 0, y * np.log(y / mu), 0.0) - y + mu
    ))


def _log_exposure(df: pd.DataFrame) -> np.ndarray:
    return np.log(np.maximum(df["Exposure"].values, 1e-9))


def _aglm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature DataFrame supplied to cva_aglm / cv_aglm.

    Prepends ``LogExposure`` (log of Exposure) so the Poisson model can
    absorb the exposure offset.  All other columns are the standard
    freMTPL2freq predictors.
    """
    X = df[FEATURE_COLS].copy()
    X.insert(0, "LogExposure", np.log(np.maximum(df["Exposure"].values, 1e-9)))
    return X


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load freMTPL2freq from OpenML and produce a stratified 75/25 split."""
    print("Loading freMTPL2freq from OpenML …")
    raw = fetch_openml(data_id=41214, as_frame=True, parser="pandas")["data"]

    # Fix dtypes coming out of OpenML
    for col in ["ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype("Int64")
    for col in ["Exposure", "Density"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    for col in ["VehBrand", "VehGas", "Area", "Region"]:
        raw[col] = raw[col].astype(str).str.strip()

    before = len(raw)
    raw = raw.dropna(subset=["ClaimNb", "VehPower", "VehAge", "DrivAge",
                               "BonusMalus", "Exposure", "Density",
                               "VehBrand", "VehGas", "Area", "Region"])
    print(f"  Dropped {before - len(raw):,} rows with missing values")

    raw["Exposure"] = raw["Exposure"].clip(lower=1e-4)
    raw["ClaimNb"]  = raw["ClaimNb"].astype(int).clip(upper=4)
    raw = raw.drop(columns=["IDpol"], errors="ignore")

    if N_SAMPLE:
        raw = raw.sample(N_SAMPLE, random_state=42).reset_index(drop=True)

    print(f"  Policies: {len(raw):,}")
    strat = (raw["ClaimNb"] > 0).astype(int)
    train, test = train_test_split(
        raw, test_size=0.25, random_state=42, stratify=strat
    )
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    print(f"  Train {len(train):,} | Test {len(test):,} | "
          f"claim rate {train['ClaimNb'].mean():.4f}")
    return train, test


# ---------------------------------------------------------------------------
# One-hot helper (used only by the plain GLM)
# ---------------------------------------------------------------------------

def _one_hot(df: pd.DataFrame, ref_levels: dict) -> np.ndarray:
    """Build a plain one-hot feature matrix for the statsmodels GLM."""
    parts = [df[NUMERIC_COLS].values.astype(float)]
    for col in CATEGORICAL_COLS:
        for lv in ref_levels[col]:
            parts.append((df[col] == lv).values.astype(float).reshape(-1, 1))
    return np.hstack(parts)


# ---------------------------------------------------------------------------
# 1/5  GLM — plain Poisson (statsmodels IRLS, proper log-exposure offset)
# ---------------------------------------------------------------------------

def fit_glm(
    train: pd.DataFrame,
    ref_levels: dict,
) -> tuple[sm.GLMResultsWrapper, int]:
    print("\n[1/5] GLM — plain Poisson (statsmodels IRLS) …", flush=True)
    t0 = time.time()

    X   = sm.add_constant(_one_hot(train, ref_levels), has_constant="add")
    y   = train["ClaimNb"].values
    off = _log_exposure(train)

    res = sm.GLM(
        y, X, family=sm.families.Poisson(), offset=off
    ).fit(maxiter=30)

    print(f"  ✓ {time.time() - t0:.1f}s | deviance = {res.deviance:.4f}")
    return res, X.shape[1]


def predict_glm(
    res: sm.GLMResultsWrapper,
    ncols: int,
    df: pd.DataFrame,
    ref_levels: dict,
) -> np.ndarray:
    X   = sm.add_constant(_one_hot(df, ref_levels), has_constant="add")
    if X.shape[1] < ncols:
        X = np.hstack([X, np.zeros((len(X), ncols - X.shape[1]))])
    off = _log_exposure(df)
    return res.predict(X[:, :ncols], offset=off)


# ---------------------------------------------------------------------------
# 2/5  RegGLM — regularised Poisson, linear + one-hot only (aglm / RustyStats)
#
# Equivalent to the old irls_en_cv on a plain one-hot matrix, now replaced by
# cva_aglm with O-dummies disabled so the feature space is identical to a
# standard GLM design matrix.  cva_aglm searches the full (alpha, lambda)
# grid via the parallel Rust IRLS engine.
# ---------------------------------------------------------------------------

def fit_reg_glm(train: pd.DataFrame) -> "CVAAccurateGLM":  # noqa: F821
    print("\n[2/5] RegGLM — regularised Poisson, plain design matrix"
          " (aglm / RustyStats) …", flush=True)
    t0 = time.time()

    X = _aglm_features(train)
    y = train["ClaimNb"].values.astype(float)

    # Replaces:  irls_en_cv(X_onehot, y, log_offset, np.linspace(0,1,11))
    # Key flag:  od_type_of_quantitatives="N" suppresses O-dummy basis for
    #            numerics, keeping the design matrix equivalent to plain GLM.
    model = cva_aglm(
        X, y,
        alpha_grid=ALPHA_GRID,
        nfolds=CV_FOLDS,
        family="poisson",
        lambda_grid=np.logspace(-3, 1, N_ALPHAS),
        add_linear_columns=True,
        use_lvar=False,
        od_type_of_quantitatives="N",       # no O-dummy basis for numerics
        add_od_columns_of_qualitatives=False,  # no O-dummy basis for categoricals
        nbin_max=NBIN_MAX,
    )

    bm     = model.best_model
    coef_s = bm.coef(with_names=True)
    n_nz   = int((coef_s.abs() > 1e-6).sum())
    print(f"  Best α={model.best_alpha:.2f}  λ={bm.lambda_:.5f}  "
          f"nnz={n_nz}/{len(coef_s)}")
    print(f"  ✓ {time.time() - t0:.1f}s")
    return model


def predict_reg_glm(model: "CVAAccurateGLM", df: pd.DataFrame) -> np.ndarray:  # noqa: F821
    """Predict claim counts using the best RegGLM model."""
    X = _aglm_features(df)
    return model.best_model.predict(X)


# ---------------------------------------------------------------------------
# 3/5  AGLM — augmented Poisson, L-variable basis (aglm / RustyStats)
#
# Equivalent to the old irls_en_cv on the full AGLM augmented design matrix
# (built by aglm.input.new_input with use_lvar=True).  Now replaced by
# cva_aglm which handles the augmented matrix, fitting, and CV internally.
# ---------------------------------------------------------------------------

def fit_aglm(train: pd.DataFrame) -> "CVAAccurateGLM":  # noqa: F821
    print(f"\n[3/5] AGLM — augmented Poisson, L-variable basis"
          f"  nbin={NBIN_MAX} (aglm / RustyStats) …", flush=True)
    t0 = time.time()

    X = _aglm_features(train)
    y = train["ClaimNb"].values.astype(float)

    # Replaces:  build_aglm_matrix()  +  irls_en_cv(X_aug, y, log_offset, …)
    # The augmented design matrix (L-vars, O-dummies, linear terms) is built
    # internally by new_input() inside cva_aglm.
    model = cva_aglm(
        X, y,
        alpha_grid=ALPHA_GRID,
        nfolds=CV_FOLDS,
        family="poisson",
        lambda_grid=np.logspace(-3, 1, N_ALPHAS),
        add_linear_columns=True,
        use_lvar=True,                         # L-variable basis (vs O-dummies)
        add_od_columns_of_qualitatives=True,
        nbin_max=NBIN_MAX,
    )

    bm     = model.best_model
    coef_s = bm.coef(with_names=True)
    n_nz   = int((coef_s.abs() > 1e-6).sum())
    total  = len(coef_s)
    print(f"  Best α={model.best_alpha:.2f}  λ={bm.lambda_:.5f}  "
          f"nnz={n_nz}/{total}  (augmented matrix: {total} basis columns)")
    print(f"  ✓ {time.time() - t0:.1f}s")
    return model


def predict_aglm(model: "CVAAccurateGLM", df: pd.DataFrame) -> np.ndarray:  # noqa: F821
    """Predict claim counts using the best AGLM model."""
    X = _aglm_features(df)
    return model.best_model.predict(X)


# ---------------------------------------------------------------------------
# 4/5  GAM — PoissonGAM with spline + factor terms (pygam)
# ---------------------------------------------------------------------------

def fit_gam(train: pd.DataFrame) -> dict:
    from pygam import PoissonGAM, f, s
    print("\n[4/5] GAM — PoissonGAM spline + factor terms (pygam) …", flush=True)
    t0 = time.time()

    cat_maps: dict = {}

    def _gam_mat(df: pd.DataFrame) -> np.ndarray:
        parts = [df[NUMERIC_COLS].values.astype(float)]
        for col in CATEGORICAL_COLS:
            if col not in cat_maps:
                cats = sorted(df[col].unique())
                cat_maps[col] = {c: i for i, c in enumerate(cats)}
            parts.append(
                np.array([cat_maps[col].get(v, 0) for v in df[col]], float
                          ).reshape(-1, 1)
            )
        return np.hstack(parts)

    X     = _gam_mat(train)
    terms = s(0) + s(1) + s(2) + s(3) + s(4)
    for i in range(5, X.shape[1]):
        terms += f(i)

    gam = PoissonGAM(terms)
    gam.gridsearch(
        X, train["ClaimNb"].values,
        weights=train["Exposure"].values,
        progress=False,
    )
    print(f"  ✓ {time.time() - t0:.1f}s")
    return {"model": gam, "cat_maps": cat_maps, "gam_mat": _gam_mat}


def predict_gam(m: dict, df: pd.DataFrame) -> np.ndarray:
    return m["model"].predict(m["gam_mat"](df)) * df["Exposure"].values


# ---------------------------------------------------------------------------
# 5/5  GBM — LightGBM Poisson with CV for n_estimators
# ---------------------------------------------------------------------------

def fit_gbm(train: pd.DataFrame) -> "lgb.Booster":  # noqa: F821
    import lightgbm as lgb
    print("\n[5/5] GBM — LightGBM Poisson (CV for n_estimators) …", flush=True)
    t0 = time.time()

    X = train[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")

    dtrain = lgb.Dataset(
        X,
        label=train["ClaimNb"].values,
        weight=train["Exposure"].values,
        free_raw_data=False,
    )
    params = {
        "objective":      "poisson",
        "metric":         "poisson",
        "learning_rate":  0.01,
        "num_leaves":     63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq":   5,
        "verbose":        -1,
        "n_jobs":         4,
    }
    cv = lgb.cv(
        params, dtrain,
        num_boost_round=500, nfold=5,
        stratified=False,
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    best_n = len(cv["valid poisson-mean"])
    booster = lgb.train(
        params, dtrain,
        num_boost_round=best_n,
        callbacks=[lgb.log_evaluation(-1)],
    )
    print(f"  Best n_estimators={best_n}  ✓ {time.time() - t0:.1f}s")
    return booster


def predict_gbm(booster: "lgb.Booster", df: pd.DataFrame) -> np.ndarray:  # noqa: F821
    import lightgbm as lgb  # noqa: F401 — needed for category dtype handling
    X = df[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")
    return booster.predict(X) * df["Exposure"].values


# ---------------------------------------------------------------------------
# Figure 2 — AGLM component curves
# ---------------------------------------------------------------------------

def plot_component_curves(
    aglm_cva: "CVAAccurateGLM",  # noqa: F821
    train: pd.DataFrame,
    deviance_table: pd.DataFrame,
) -> plt.Figure:
    """Reproduce Figure 2 — per-variable log-contribution curves.

    Each panel sweeps one variable across its 1st–99th percentile range (or
    its unique levels for categoricals) while all other variables are held at
    their reference level (median / mode).  The y-axis is the log-contribution
    relative to the reference prediction:

        c(x_j) = log μ(x_j, x_{-j}=ref) − log μ(ref)

    This isolates the shape of each variable's effect on the linear predictor.
    """
    best_model = aglm_cva.best_model

    # Build reference row — unit exposure (LogExposure = log(1) = 0)
    ref: dict = {col: float(train[col].median()) for col in NUMERIC_COLS}
    ref.update({col: train[col].mode().iloc[0] for col in CATEGORICAL_COLS})
    ref["Exposure"]    = 1.0
    ref["LogExposure"] = 0.0      # log(1.0) = 0

    ref_df    = pd.DataFrame([ref])[AGLM_COLS]
    mu_ref    = float(best_model.predict(ref_df)[0])
    log_ref   = np.log(max(mu_ref, 1e-12))

    def component(sweep_df: pd.DataFrame) -> np.ndarray:
        """Log-contribution relative to reference for a sweep DataFrame."""
        mu = best_model.predict(sweep_df[AGLM_COLS])
        return np.log(np.maximum(mu, 1e-12)) - log_ref

    ncols   = 3
    n_plots = len(FEATURE_COLS) + 1          # variables + deviance table inset
    nrows   = int(np.ceil(n_plots / ncols))

    fig = plt.figure(figsize=(5.5 * ncols, 4.0 * nrows))
    fig.suptitle(
        "Figure 2 — AGLM Component Curves\n"
        "log contribution to claim frequency | freMTPL2freq",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs   = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.40)
    axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]

    ax_idx = 0

    # ---- Numeric variables -----------------------------------------------
    for col in NUMERIC_COLS:
        ax  = axes[ax_idx]; ax_idx += 1
        lo  = float(train[col].quantile(0.01))
        hi  = float(train[col].quantile(0.99))
        grd = np.linspace(lo, hi, 150)

        rows = [{**ref, col: v} for v in grd]
        swp  = pd.DataFrame(rows)[AGLM_COLS]
        cmp  = component(swp)

        ax.plot(grd, cmp, lw=2.0, color=COLORS["AGLM"])
        ax.fill_between(grd, cmp, alpha=0.12, color=COLORS["AGLM"])
        ax.axhline(0, color="grey", lw=0.8, ls="--")

        rug = train[col].sample(min(1200, len(train)), random_state=0).values
        ax.plot(rug, np.full_like(rug, cmp.min()), "|",
                color="grey", alpha=0.18, ms=3)

        ax.set(xlabel=col, ylabel="log contribution", title=col)
        ax.title.set_fontweight("bold")

    # ---- Categorical variables -------------------------------------------
    for col in CATEGORICAL_COLS:
        ax   = axes[ax_idx]; ax_idx += 1
        lvls = sorted(train[col].unique())

        rows = [{**ref, col: lv} for lv in lvls]
        swp  = pd.DataFrame(rows)[AGLM_COLS]
        cmp  = component(swp)

        bar_colors = [COLORS["AGLM"] if c >= 0 else COLORS["GBM"] for c in cmp]
        ax.bar(range(len(lvls)), cmp, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(lvls)))
        ax.set_xticklabels(lvls, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set(ylabel="log contribution", title=col)
        ax.title.set_fontweight("bold")

    # ---- Deviance table inset -------------------------------------------
    ax = axes[ax_idx]; ax_idx += 1
    ax.axis("off")
    rows_tbl = [
        [r["Model"], f"{r['Poisson deviance (test)']:.6f}"]
        for _, r in deviance_table.iterrows()
    ]
    tbl = ax.table(
        cellText=rows_tbl,
        colLabels=["Model", "Poisson Deviance"],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 1.9)
    best_idx = deviance_table["Poisson deviance (test)"].idxmin()
    for j in range(2):
        tbl[(best_idx + 1, j)].set_facecolor("#D6EAF8")
        tbl[(best_idx + 1, j)].set_text_props(fontweight="bold")
    ax.set_title("Table 8 — Poisson Deviance", fontsize=10, fontweight="bold")

    for ax in axes[ax_idx:]:
        ax.set_visible(False)

    return fig


# ---------------------------------------------------------------------------
# Figure 3 — deviance bar chart
# ---------------------------------------------------------------------------

def plot_deviance_bar(deviance_table: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart comparing model deviances against paper reference."""
    fig, ax = plt.subplots(figsize=(9, 5))

    models = deviance_table["Model"].tolist()
    devs   = deviance_table["Poisson deviance (test)"].tolist()
    bcs    = [COLORS.get(m, "#888") for m in models]

    bars = ax.barh(
        models[::-1], devs[::-1],
        color=bcs[::-1], alpha=0.88, edgecolor="white", height=0.55,
    )
    for bar, dev in zip(bars, devs[::-1]):
        ax.text(
            bar.get_width() + 2e-5,
            bar.get_y() + bar.get_height() / 2,
            f"{dev:.5f}", va="center", fontsize=9.5,
        )

    ax.axvline(min(devs), color="black", lw=1.2, ls=":",
               label=f"Best = {min(devs):.5f}")

    # Paper Table 8 reference diamonds (full N = 678k)
    paper = {
        "AGLM":   0.3111920,
        "GLM":    0.3201199,
        "RegGLM": 0.3201245,
        "GAM":    0.3171236,
        "GBM":    0.3123919,
    }
    ordered = models[::-1]
    for m, pdv in paper.items():
        if m in ordered:
            ax.plot(pdv, ordered.index(m), marker="D",
                    ms=6, color="black", alpha=0.45, zorder=5)

    ax.set_xlabel("Mean Poisson Deviance — lower is better", fontsize=11)
    ax.set_title(
        f"Table 8 — Model Comparison | freMTPL2freq "
        f"(N={N_SAMPLE or 678_013:,})\n◆ = paper reference (full N=678k)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_xlim(left=min(devs) * 0.97)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(run_gam: bool = True, run_gbm: bool = True) -> None:
    print("=" * 65)
    print("Fujita et al. (2020) — Section 5 Numerical Experiments")
    print("=" * 65)
    t_total = time.time()

    train, test = load_and_split()
    ref_levels  = {col: sorted(train[col].unique()) for col in CATEGORICAL_COLS}

    print()
    print("─" * 65)
    print(f"Fitting models  | nbin={NBIN_MAX}  n_alphas={N_ALPHAS}"
          f"  alpha_grid={ALPHA_GRID.tolist()}  cv={CV_FOLDS}")
    print("─" * 65)

    glm_res, glm_ncols = fit_glm(train, ref_levels)
    reg_glm_m          = fit_reg_glm(train)
    aglm_m             = fit_aglm(train)

    gam_m = fit_gam(train)  if run_gam  else None
    gbm_m = fit_gbm(train)  if run_gbm  else None

    print()
    print("─" * 65)
    print("Table 8 — Poisson Deviance on Hold-out Test Set (Eq. 11)")
    print("─" * 65)

    predict_fns: dict = {
        "AGLM":   lambda df: predict_aglm(aglm_m, df),
        "GLM":    lambda df: predict_glm(glm_res, glm_ncols, df, ref_levels),
        "RegGLM": lambda df: predict_reg_glm(reg_glm_m, df),
    }
    if gam_m is not None:
        predict_fns["GAM"] = lambda df: predict_gam(gam_m, df)
    if gbm_m is not None:
        predict_fns["GBM"] = lambda df: predict_gbm(gbm_m, df)

    rows = []
    for name, fn in predict_fns.items():
        mu  = fn(test)
        dev = poisson_deviance_eq11(test["ClaimNb"].values, mu)
        rows.append({"Model": name, "Poisson deviance (test)": dev})
        print(f"  {name:9s}: {dev:.7f}")

    deviance_table = (
        pd.DataFrame(rows)
        .sort_values("Poisson deviance (test)")
        .reset_index(drop=True)
    )
    print()
    print(deviance_table.to_string(index=False))
    print()
    print("Paper Table 8 (reference, full N=678k):")
    print(pd.DataFrame({
        "Model":         ["AGLM",     "GBM",      "GAM",      "GLM",      "RegGLM"],
        "Paper deviance": [0.3111920,  0.3123919,  0.3171236,  0.3201199,  0.3201245],
    }).to_string(index=False))

    print("\nGenerating figures …")

    fig2 = plot_component_curves(aglm_m, train, deviance_table)
    fig2.savefig("fremtpl2_figure2_component_curves.png",
                 dpi=130, bbox_inches="tight")
    print("  Saved: fremtpl2_figure2_component_curves.png")
    plt.close(fig2)

    fig3 = plot_deviance_bar(deviance_table)
    fig3.savefig("fremtpl2_figure3_deviance_comparison.png",
                 dpi=130, bbox_inches="tight")
    print("  Saved: fremtpl2_figure3_deviance_comparison.png")
    plt.close(fig3)

    print(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} minutes")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    res=main(True, True)

