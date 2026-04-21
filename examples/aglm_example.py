"""
aglm_example.py
===============
End-to-end demonstration of the aglm package (RustyStats backend) on a
synthetic freMTPL2freq motor third-party liability frequency dataset.

Workflow
--------
1. Generate a synthetic freMTPL2freq dataset (10 000 policies).
2. Prepare features — log(Exposure) is included as a predictor to act as
   an approximate Poisson offset (the AGLM interface does not take an explicit
   offset argument; including log-exposure as a linear term is the standard
   workaround and recovers the correct offset coefficient asymptotically).
3. Fit three models via cv_aglm, stepping across the elastic-net mixing
   parameter: Ridge (α=0), Elastic Net (α=0.5), LASSO (α=1).
4. Compare in-sample deviances and selected lambda values.
5. Produce per-variable contribution plots for the best model.

Run
---
    python aglm_example.py

Requirements
------------
    pip install rustystats matplotlib
    # aglm package must be on sys.path (parent directory of the aglm/ folder)
"""

from __future__ import annotations

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — adjust if the aglm package lives elsewhere
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from generate_fremtpl2freq import make_fremtpl2freq
from aglm import cv_aglm, cva_aglm, plot_aglm, plot_cva_alpha

warnings.filterwarnings("ignore")


# ===========================================================================
# 1. Data generation
# ===========================================================================

print("=" * 60)
print("freMTPL2freq  —  AGLM example (RustyStats backend)")
print("=" * 60)

N_SAMPLE = 10_000
print(f"\n[1] Generating synthetic dataset  (n={N_SAMPLE:,}) ...")
df = make_fremtpl2freq(n=N_SAMPLE, seed=42)

# log(Exposure) acts as a near-offset: coefficient should converge to ~1.0
df["LogExposure"] = np.log(np.maximum(df["Exposure"], 1e-6))

print(f"    Policies : {len(df):,}")
print(f"    Claims   : {df['ClaimNb'].sum():,}")
print(f"    Freq (raw): {df['ClaimNb'].sum() / df['Exposure'].sum():.4f} claims/policy-year")
print(f"\n    ClaimNb distribution:")
for k, cnt in df["ClaimNb"].value_counts().sort_index().items():
    pct = 100 * cnt / len(df)
    print(f"      {k}  : {cnt:>6,}  ({pct:.1f}%)")


# ===========================================================================
# 2. Feature matrix
# ===========================================================================

print("\n[2] Preparing feature matrix ...")

NUMERIC_COLS = ["LogExposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEG_COLS   = ["VehBrand", "VehGas", "Area", "Region"]

X = df[NUMERIC_COLS + CATEG_COLS].copy()
y = df["ClaimNb"].values.astype(float)

print(f"    X shape  : {X.shape}")
print(f"    Columns  : {list(X.columns)}")


# ===========================================================================
# 3. Fit models
# ===========================================================================

print("\n[3] Fitting models via cv_aglm (Poisson, nfolds=5) ...")
print("    Using an explicit lambda_grid so runtime stays predictable.\n")

# A compact log-spaced grid — 15 points is enough to find the CV minimum.
LAMBDA_GRID = np.logspace(-3, 1, 15)

results = {}

for alpha, label in [(0.0, "Ridge"), (0.5, "ElasticNet"), (1.0, "LASSO")]:
    t0 = time.time()
    model = cv_aglm(
        X, y,
        alpha=alpha,
        nfolds=5,
        family="poisson",
        lambda_grid=LAMBDA_GRID,
        # Feature engineering — numeric vars get O-dummy basis + linear term;
        # categoricals get U-dummy (one-hot) encoding automatically.
        nbin_max=20,          # cap bins per numeric variable (speed)
        add_linear_columns=True,
        add_interaction_columns=False,
    )
    elapsed = time.time() - t0
    dev = model.deviance()
    results[label] = model

    print(f"    {label:12s}  α={alpha:.1f}  "
          f"best_λ={model.lambda_:.4f}  "
          f"deviance={dev:.2f}  "
          f"({elapsed:.1f}s)")


# ===========================================================================
# 4. Coefficient summary for the best model (lowest deviance)
# ===========================================================================

best_label = min(results, key=lambda k: results[k].deviance())
best_model = results[best_label]

print(f"\n[4] Coefficient summary — {best_label} model")
print(f"    Intercept : {best_model.intercept():.4f}")

coef_series = best_model.coef(with_names=True)
nonzero = coef_series[coef_series.abs() > 1e-6]
zero_count = (coef_series.abs() <= 1e-6).sum()

print(f"    Non-zero coefficients : {len(nonzero)} / {len(coef_series)}")
print(f"    Zero (penalised out)  : {zero_count}")
print(f"\n    Top 10 by |coef|:")
top10 = nonzero.abs().nlargest(10).index
for name in top10:
    print(f"      {name:45s}  {coef_series[name]:+.5f}")

print(f"\n    LogExposure coefficient : {coef_series.get('LogExposure', float('nan')):.4f}")
print("    (Should be close to 1.0 if the model has recovered the exposure offset)")


# ===========================================================================
# 5. Model comparison table
# ===========================================================================

print("\n[5] Model comparison")
print(f"    {'Model':12s}  {'α':>4}  {'best_λ':>10}  {'Deviance':>12}  {'Non-zero coefs':>15}")
print("    " + "-" * 60)

for label, model in results.items():
    alpha_val = model.alpha
    lam       = model.lambda_
    dev       = model.deviance()
    coef_s    = model.coef(with_names=True)
    nz        = (coef_s.abs() > 1e-6).sum()
    total     = len(coef_s)
    print(f"    {label:12s}  {alpha_val:>4.1f}  {lam:>10.4f}  {dev:>12.2f}  {nz:>6} / {total}")


# ===========================================================================
# 6. Plots
# ===========================================================================

print("\n[6] Generating contribution plots ...")

fig = plot_aglm(
    best_model,
    ncols=3,
    show_residuals=True,
    show_cv_curve=True,
    title_prefix=f"{best_label} — ",
    link_scale=True,       # plot on log scale (linear predictor for Poisson)
)
fig.savefig("aglm_contributions.png", dpi=120, bbox_inches="tight")
print("    Saved: aglm_contributions.png")

# CV path for the best model
if best_model.cv_results is not None:
    fig2, ax = plt.subplots(figsize=(6, 3.5))
    lam_path = best_model.cv_results["lambda_grid"]
    cv_scores = best_model.cv_results["mean_cv_score"]   # negated deviance
    ax.semilogx(lam_path, cv_scores, color="#7b1fa2", linewidth=2)
    ax.axvline(best_model.lambda_, color="#e53935", linestyle="--",
               label=f"Best λ = {best_model.lambda_:.4g}")
    ax.set_xlabel("λ", fontsize=11)
    ax.set_ylabel("CV score (−deviance)", fontsize=11)
    ax.set_title(f"CV path — {best_label}", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    fig2.tight_layout()
    fig2.savefig("aglm_cv_path.png", dpi=120, bbox_inches="tight")
    print("    Saved: aglm_cv_path.png")

plt.close("all")

print("\nDone.")
