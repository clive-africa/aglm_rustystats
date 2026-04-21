"""
synthetic_accident_experiment.py
=================================
Synthetic motor accident frequency dataset with known ground truth.

Data generating process
-----------------------
  Age:       Uniform integer in [20, 40]
  Gender:    50/50 Male / Female
  True rate: linear from 15% at age 20 to 5% at age 40
             Males 20% more likely than females
  Exposure:
    Males:   90% ~ U(0, 1),      10% ~ U(1/365, 1/12)
    Females: 60% ~ U(0, 1),      40% ~ U(1/365, 1/12)
  Claims:    Poisson(exposure * true_rate)

Models fitted (4)
-----------------
  GBM-Offset   — LightGBM Poisson, log-exposure as init_score (offset method)
  GBM-Weights  — LightGBM Poisson, frequency label + exposure weights
  XGB-Offset   — XGBoost Poisson, log-exposure as base_margin (offset method)
  XGB-Weights  — XGBoost Poisson, frequency label + exposure weights

Outputs
-------
  Console metrics table (Poisson Deviance, MSE, MAE, AUC, Avg Pred, Avg Actual)
  synthetic_figure_metrics.png     — 2x2 bar chart comparison
  synthetic_figure_calibration.png — avg pred vs actual per model
  synthetic_figure_age_curves.png  — recovered age effect vs true DGP
"""

from __future__ import annotations

import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_POLICIES  = 100_000
RANDOM_SEED = 42
CV_FOLDS    = 5

COLORS = {
    "GBM-Offset":  "#C0392B",
    "GBM-Weights": "#E67E22",
    "XGB-Offset":  "#2471A3",
    "XGB-Weights": "#1ABC9C",
}

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "font.size":         10,
})

FEATURE_COLS     = ["Age", "Gender"]
CATEGORICAL_COLS = ["Gender"]

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_dataset(n: int = N_POLICIES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic accident frequency data with known DGP.

    True claim frequency:
        base_rate(age)   = 0.15 - (age - 20) * 0.005   [linear 15% -> 5%]
        female_rate(age) = base_rate(age)
        male_rate(age)   = base_rate(age) * 1.2          [males 20% higher]

    Exposure:
        Males:   90% ~ U(0, 1),      10% ~ U(1/365, 1/12)
        Females: 60% ~ U(0, 1),      40% ~ U(1/365, 1/12)

    Claims ~ Poisson(exposure * true_rate)
    """
    rng = np.random.default_rng(seed)

    age    = rng.integers(20, 41, size=n)
    gender = rng.choice(["M", "F"], size=n)

    base_rate = 0.15 - (age - 20) * (0.10 / 20)          # 15% at 20, 5% at 40
    male_mask = gender == "M"
    true_rate = np.where(male_mask, base_rate * 1.2, base_rate)

    # --- Exposure ---
    exposure = np.empty(n)

    m_idx   = np.where(male_mask)[0]
    short_m = rng.random(len(m_idx)) < 0.10               # 10% short exposure
    exposure[m_idx[~short_m]] = rng.uniform(0,     1,      (~short_m).sum())
    exposure[m_idx[ short_m]] = rng.uniform(1/365, 1/12,   ( short_m).sum())

    f_idx   = np.where(~male_mask)[0]
    short_f = rng.random(len(f_idx)) < 0.40               # 40% short exposure
    exposure[f_idx[~short_f]] = rng.uniform(0,     1,      (~short_f).sum())
    exposure[f_idx[ short_f]] = rng.uniform(1/365, 1/12,   ( short_f).sum())

    exposure = np.maximum(exposure, 1e-6)

    claims = rng.poisson(exposure * true_rate)

    return pd.DataFrame({
        "Age":      age,
        "Gender":   gender,
        "Exposure": exposure,
        "TrueRate": true_rate,
        "ClaimNb":  claims,
    })


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance (2x convention, matches sklearn / R GLM / paper)."""
    y  = np.asarray(y_true, float)
    mu = np.maximum(np.asarray(y_pred, float), 1e-12)
    return float(2.0 * np.mean(
        np.where(y > 0, y * np.log(y / mu), 0.0) - y + mu
    ))


def compute_metrics(
    name: str,
    y_count: np.ndarray,
    y_pred_count: np.ndarray,
    exposure: np.ndarray,
) -> dict:
    y   = np.asarray(y_count,       float)
    mu  = np.maximum(np.asarray(y_pred_count, float), 1e-12)
    exp = np.asarray(exposure,      float)

    freq_true = y  / np.maximum(exp, 1e-9)
    freq_pred = mu / np.maximum(exp, 1e-9)

    dev = poisson_deviance(y, mu)
    mse = float(np.mean((freq_true - freq_pred) ** 2))
    mae = float(np.mean(np.abs(freq_true - freq_pred)))

    binary = (y > 0).astype(int)
    auc = (float(roc_auc_score(binary, freq_pred))
           if 0 < binary.sum() < len(binary) else float("nan"))

    w          = exp / exp.sum()
    avg_pred   = float(np.sum(freq_pred * w))
    avg_actual = float(np.sum(freq_true * w))

    return {
        "Model":             name,
        "Poisson Deviance":  dev,
        "MSE":               mse,
        "MAE":               mae,
        "AUC":               auc,
        "Avg Pred (freq)":   avg_pred,
        "Avg Actual (freq)": avg_actual,
    }


# ---------------------------------------------------------------------------
# LightGBM helpers
# ---------------------------------------------------------------------------

def _lgb_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    X["Gender"] = X["Gender"].astype("category")
    return X


_LGB_PARAMS = dict(
    objective        = "poisson",
    metric           = "poisson",
    learning_rate    = 0.05,
    num_leaves       = 31,
    min_data_in_leaf = 50,
    verbose          = -1,
    n_jobs           = 4,
)

_LGB_CB = [
    lgb.early_stopping(20, verbose=False),
    lgb.log_evaluation(-1),
]


def _lgb_cv_train(dtrain: lgb.Dataset) -> lgb.Booster:
    cv = lgb.cv(_LGB_PARAMS, dtrain, num_boost_round=500, nfold=CV_FOLDS,
                stratified=False, callbacks=_LGB_CB)
    best_n = len(cv["valid poisson-mean"])
    return lgb.train(_LGB_PARAMS, dtrain, num_boost_round=best_n,
                     callbacks=[lgb.log_evaluation(-1)]), best_n


# ---------------------------------------------------------------------------
# 1/4  GBM-Offset  — log-exposure as LightGBM init_score
# ---------------------------------------------------------------------------

def fit_gbm_offset(train: pd.DataFrame) -> lgb.Booster:
    print("\n[1/4] GBM-Offset  — LightGBM Poisson, log-exposure init_score ...",
          flush=True)
    t0      = time.time()
    log_exp = np.log(np.maximum(train["Exposure"].values, 1e-9))
    dtrain  = lgb.Dataset(_lgb_X(train), label=train["ClaimNb"].values,
                          init_score=log_exp, free_raw_data=False)
    booster, best_n = _lgb_cv_train(dtrain)
    print(f"  best_n={best_n}  ok {time.time() - t0:.1f}s")
    return booster


def predict_gbm_offset(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    log_exp = np.log(np.maximum(df["Exposure"].values, 1e-9))
    raw     = booster.predict(_lgb_X(df), raw_score=True)
    return np.exp(raw + log_exp)


# ---------------------------------------------------------------------------
# 2/4  GBM-Weights — frequency label + exposure weights
# ---------------------------------------------------------------------------

def fit_gbm_weights(train: pd.DataFrame) -> lgb.Booster:
    print("\n[2/4] GBM-Weights — LightGBM Poisson, frequency label + weights ...",
          flush=True)
    t0   = time.time()
    freq = train["ClaimNb"].values / np.maximum(train["Exposure"].values, 1e-9)
    w    = train["Exposure"].values
    dtrain = lgb.Dataset(_lgb_X(train), label=freq, weight=w,
                         free_raw_data=False)
    booster, best_n = _lgb_cv_train(dtrain)
    print(f"  best_n={best_n}  ok {time.time() - t0:.1f}s")
    return booster


def predict_gbm_weights(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return booster.predict(_lgb_X(df)) * df["Exposure"].values


# ---------------------------------------------------------------------------
# XGBoost helpers
# ---------------------------------------------------------------------------

def _xgb_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    X["Gender"] = X["Gender"].astype("category")
    return X


_XGB_PARAMS = dict(
    objective        = "count:poisson",
    eval_metric      = "poisson-nloglik",
    learning_rate    = 0.05,
    max_depth        = 4,
    min_child_weight = 50,
    seed             = 42,
    nthread          = 4,
    verbosity        = 0,
)

_XGB_CB = [xgb.callback.EvaluationMonitor(show_stdv=False, period=9999)]


def _xgb_cv_train(dtrain: xgb.DMatrix) -> tuple[xgb.Booster, int]:
    cv = xgb.cv(_XGB_PARAMS, dtrain, num_boost_round=500, nfold=CV_FOLDS,
                stratified=False, early_stopping_rounds=20, callbacks=_XGB_CB)
    best_n = len(cv)
    booster = xgb.train(_XGB_PARAMS, dtrain, num_boost_round=best_n,
                        verbose_eval=False)
    return booster, best_n


# ---------------------------------------------------------------------------
# 3/4  XGB-Offset  — log-exposure as XGBoost base_margin
# ---------------------------------------------------------------------------

def fit_xgb_offset(train: pd.DataFrame) -> xgb.Booster:
    print("\n[3/4] XGB-Offset  — XGBoost Poisson, log-exposure base_margin ...",
          flush=True)
    t0      = time.time()
    log_exp = np.log(np.maximum(train["Exposure"].values, 1e-9))
    dtrain  = xgb.DMatrix(_xgb_X(train), label=train["ClaimNb"].values,
                          enable_categorical=True)
    dtrain.set_base_margin(log_exp)
    booster, best_n = _xgb_cv_train(dtrain)
    print(f"  best_n={best_n}  ok {time.time() - t0:.1f}s")
    return booster


def predict_xgb_offset(booster: xgb.Booster, df: pd.DataFrame) -> np.ndarray:
    log_exp = np.log(np.maximum(df["Exposure"].values, 1e-9))
    dtest   = xgb.DMatrix(_xgb_X(df), enable_categorical=True)
    dtest.set_base_margin(log_exp)
    return booster.predict(dtest)


# ---------------------------------------------------------------------------
# 4/4  XGB-Weights — frequency label + exposure weights
# ---------------------------------------------------------------------------

def fit_xgb_weights(train: pd.DataFrame) -> xgb.Booster:
    print("\n[4/4] XGB-Weights — XGBoost Poisson, frequency label + weights ...",
          flush=True)
    t0   = time.time()
    freq = train["ClaimNb"].values / np.maximum(train["Exposure"].values, 1e-9)
    w    = train["Exposure"].values
    dtrain = xgb.DMatrix(_xgb_X(train), label=freq, weight=w,
                         enable_categorical=True)
    booster, best_n = _xgb_cv_train(dtrain)
    print(f"  best_n={best_n}  ok {time.time() - t0:.1f}s")
    return booster


def predict_xgb_weights(booster: xgb.Booster, df: pd.DataFrame) -> np.ndarray:
    dtest = xgb.DMatrix(_xgb_X(df), enable_categorical=True)
    return booster.predict(dtest) * df["Exposure"].values


# ---------------------------------------------------------------------------
# Figure 1 — Dataset overview
# ---------------------------------------------------------------------------

def plot_dataset_overview(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"Dataset Overview — Synthetic Accident Frequency (N={len(df):,})",
        fontsize=12, fontweight="bold",
    )

    # Exposure distribution by gender
    ax = axes[0]
    for g, col in [("M", COLORS["GBM-Offset"]), ("F", COLORS["XGB-Offset"])]:
        exp = df.loc[df["Gender"] == g, "Exposure"].values
        ax.hist(exp, bins=60, alpha=0.6, color=col,
                label=f"{'Male' if g=='M' else 'Female'} (n={len(exp):,})",
                density=True)
    ax.set(xlabel="Exposure (years)", ylabel="Density",
           title="Exposure Distribution by Gender")
    ax.legend()

    # True rate by age and gender
    ax = axes[1]
    ages = np.arange(20, 41)
    ax.plot(ages, 0.15 - (ages - 20) * 0.005,       "--", color=COLORS["XGB-Offset"],
            lw=2, label="Female (true DGP)")
    ax.plot(ages, (0.15 - (ages - 20) * 0.005)*1.2, "--", color=COLORS["GBM-Offset"],
            lw=2, label="Male (true DGP)")
    ax.set(xlabel="Age", ylabel="Claim Frequency",
           title="True DGP Rates by Age and Gender")
    ax.legend()

    # Observed claim rate by age (binned)
    ax = axes[2]
    df["AgeBin"] = pd.cut(df["Age"], bins=np.arange(19.5, 41.5, 2))
    obs = (df.groupby(["AgeBin", "Gender"], observed=True)
             .apply(lambda g: g["ClaimNb"].sum() / g["Exposure"].sum(),
                    include_groups=False)
             .reset_index(name="ObsFreq"))
    for g, col in [("M", COLORS["GBM-Offset"]), ("F", COLORS["XGB-Offset"])]:
        sub = obs[obs["Gender"] == g]
        mids = [iv.mid for iv in sub["AgeBin"]]
        ax.plot(mids, sub["ObsFreq"], "o-", color=col, lw=1.5, ms=5,
                label=f"{'Male' if g=='M' else 'Female'} (observed)")
    ax.set(xlabel="Age", ylabel="Observed Claim Frequency",
           title="Observed Frequency by Age and Gender")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Metrics comparison
# ---------------------------------------------------------------------------

def plot_metrics_comparison(metrics_table: pd.DataFrame) -> plt.Figure:
    metric_cfg = [
        ("Poisson Deviance", "Mean Poisson Deviance — lower is better",  True),
        ("MSE",              "MSE on Claim Frequency — lower is better", True),
        ("MAE",              "MAE on Claim Frequency — lower is better", True),
        ("AUC",              "ROC-AUC (any claim) — higher is better",   False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Model Comparison — Synthetic Accident Dataset (N={N_POLICIES:,})\n"
        "Offset method vs Sample Weights method | LightGBM and XGBoost",
        fontsize=12, fontweight="bold",
    )

    for ax, (metric, xlabel, lower_better) in zip(axes.flat, metric_cfg):
        srt    = metrics_table.sort_values(metric, ascending=lower_better)
        models = srt["Model"].tolist()
        values = srt[metric].tolist()
        bcs    = [COLORS.get(m, "#888") for m in models]

        bars = ax.barh(models, values, color=bcs,
                       alpha=0.88, edgecolor="white", height=0.55)
        best_val = min(values) if lower_better else max(values)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + abs(best_val) * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.5f}", va="center", fontsize=8.5)

        ref_line = min(values) if lower_better else max(values)
        ax.axvline(ref_line, color="black", lw=1.0, ls=":",
                   label=f"Best = {ref_line:.5f}")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.legend(fontsize=8)
        margin = abs(best_val) * 0.03
        ax.set_xlim(left=min(values) - margin) if lower_better else \
            ax.set_xlim(right=max(values) + margin)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Calibration
# ---------------------------------------------------------------------------

def plot_calibration(metrics_table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    actual = metrics_table["Avg Actual (freq)"].iloc[0]
    models = metrics_table["Model"].tolist()
    preds  = metrics_table["Avg Pred (freq)"].tolist()
    bcs    = [COLORS.get(m, "#888") for m in models]

    bars = ax.bar(models, preds, color=bcs, alpha=0.85,
                  edgecolor="white", width=0.55)
    ax.axhline(actual, color="black", lw=1.5, ls="--",
               label=f"Actual mean freq = {actual:.5f}")

    for bar, val in zip(bars, preds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + actual * 0.003,
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Exposure-weighted Mean Predicted Frequency", fontsize=10)
    ax.set_title(
        "Calibration — Exposure-weighted Mean Predicted vs Actual Frequency\n"
        "All models should hit the dashed line if correctly calibrated",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Recovered age curves vs true DGP
# ---------------------------------------------------------------------------

def plot_age_curves(predict_fns: dict, test: pd.DataFrame) -> plt.Figure:
    """
    Plot predicted frequency by age for each model against the true DGP.
    Uses a sweep over ages 20-40 with exposure=1 year (so predicted count
    = predicted frequency for both offset and weights models).
    """
    ages = np.arange(20, 41)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Recovered Age Effect — Predicted Frequency by Age vs True DGP\n"
        "Sweep: single year exposure (Exposure=1), each gender separately",
        fontsize=12, fontweight="bold",
    )

    for ax, (gender, label) in zip(axes, [("M", "Male"), ("F", "Female")]):
        mult       = 1.2 if gender == "M" else 1.0
        true_rates = mult * (0.15 - (ages - 20) * (0.10 / 20))
        ax.plot(ages, true_rates, "k--", lw=2.5, label="True DGP", zorder=5)

        sweep = pd.DataFrame({
            "Age":      ages,
            "Gender":   gender,
            "Exposure": 1.0,   # unit exposure -> count = frequency
            "ClaimNb":  0,
        })

        for name, fn in predict_fns.items():
            pred_counts = fn(sweep)
            ax.plot(ages, pred_counts, color=COLORS[name], lw=1.8,
                    marker="o", ms=4, label=name, alpha=0.85)

        ax.set_title(f"Gender = {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Age")
        ax.set_ylabel("Predicted Frequency (= count at Exposure=1)")
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Synthetic Accident Frequency Experiment")
    print("  Comparing: Offset method vs Sample Weights method")
    print("  Models:    LightGBM and XGBoost (4 total)")
    print("=" * 70)
    t_total = time.time()

    # --- Generate data ---
    print(f"\nGenerating dataset ({N_POLICIES:,} policies) ...")
    df = generate_dataset()

    print(f"  Claim rate (raw):           {df['ClaimNb'].mean():.4f}")
    print(f"  Mean exposure:              {df['Exposure'].mean():.4f}")
    print(f"  Policies with exp < 1/12:   {(df['Exposure'] < 1/12).mean():.2%}")
    print(f"  True DGP: age 20 M={0.15*1.2:.3f} F={0.150:.3f} | "
          f"age 40 M={0.05*1.2:.3f} F={0.050:.3f}")

    strat       = (df["ClaimNb"] > 0).astype(int)
    train, test = train_test_split(df, test_size=0.25, random_state=42, stratify=strat)
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    print(f"  Train {len(train):,} | Test {len(test):,}")

    # --- Fit all 4 models ---
    gbm_off_m  = fit_gbm_offset(train)
    gbm_wgt_m  = fit_gbm_weights(train)
    xgb_off_m  = fit_xgb_offset(train)
    xgb_wgt_m  = fit_xgb_weights(train)

    predict_fns: dict = {
        "GBM-Offset":  lambda df: predict_gbm_offset(gbm_off_m, df),
        "GBM-Weights": lambda df: predict_gbm_weights(gbm_wgt_m, df),
        "XGB-Offset":  lambda df: predict_xgb_offset(xgb_off_m, df),
        "XGB-Weights": lambda df: predict_xgb_weights(xgb_wgt_m, df),
    }

    # --- Evaluate on test set ---
    print("\n" + "─" * 80)
    print("Test Set Metrics")
    print("─" * 80)

    rows = []
    for name, fn in predict_fns.items():
        mu  = fn(test)
        row = compute_metrics(name, test["ClaimNb"].values, mu,
                              test["Exposure"].values)
        rows.append(row)
        print(f"  {name:<14}: dev={row['Poisson Deviance']:.6f}  "
              f"mse={row['MSE']:.8f}  mae={row['MAE']:.6f}  "
              f"auc={row['AUC']:.4f}  "
              f"avg_pred={row['Avg Pred (freq)']:.5f}  "
              f"avg_actual={row['Avg Actual (freq)']:.5f}")

    metrics_table = (
        pd.DataFrame(rows)
        .sort_values("Poisson Deviance")
        .reset_index(drop=True)
    )

    print()
    print("─" * 96)
    hdr = (f"  {'Model':<14}  {'Poisson Dev':>13}  {'MSE':>12}  "
           f"{'MAE':>10}  {'AUC':>7}  {'Avg Pred':>10}  {'Avg Actual':>10}")
    print(hdr)
    print("  " + "─" * 90)
    for _, r in metrics_table.iterrows():
        print(
            f"  {r['Model']:<14}  {r['Poisson Deviance']:>13.7f}  "
            f"{r['MSE']:>12.9f}  {r['MAE']:>10.7f}  "
            f"{r['AUC']:>7.4f}  "
            f"{r['Avg Pred (freq)']:>10.6f}  "
            f"{r['Avg Actual (freq)']:>10.6f}"
        )
    print("─" * 96)

    # Sanity: true DGP weighted frequency on test
    w          = test["Exposure"].values / test["Exposure"].values.sum()
    freq_true  = test["ClaimNb"].values / np.maximum(test["Exposure"].values, 1e-9)
    true_avg   = float(np.sum(freq_true * w))
    true_dgp   = float(np.sum(test["TrueRate"].values * w))
    print(f"\n  Reference — exposure-weighted mean on test set:")
    print(f"    Observed claim frequency : {true_avg:.5f}")
    print(f"    True DGP rate (TrueRate) : {true_dgp:.5f}  "
          f"[should be ≈ {true_avg:.5f}]")

    # --- Figures ---
    print("\nGenerating figures ...")

    fig1 = plot_dataset_overview(df)
    fig1.savefig("synthetic_figure1_dataset.png", dpi=130, bbox_inches="tight")
    print("  Saved: synthetic_figure1_dataset.png")
    plt.close(fig1)

    fig2 = plot_metrics_comparison(metrics_table)
    fig2.savefig("synthetic_figure2_metrics.png", dpi=130, bbox_inches="tight")
    print("  Saved: synthetic_figure2_metrics.png")
    plt.close(fig2)

    fig3 = plot_calibration(metrics_table)
    fig3.savefig("synthetic_figure3_calibration.png", dpi=130, bbox_inches="tight")
    print("  Saved: synthetic_figure3_calibration.png")
    plt.close(fig3)

    fig4 = plot_age_curves(predict_fns, test)
    fig4.savefig("synthetic_figure4_age_curves.png", dpi=130, bbox_inches="tight")
    print("  Saved: synthetic_figure4_age_curves.png")
    plt.close(fig4)

    print(f"\nTotal runtime: {(time.time() - t_total):.1f}s")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()