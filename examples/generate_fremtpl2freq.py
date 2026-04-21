"""
generate_fremtpl2freq.py
========================
Generates a synthetic freMTPL2freq dataset that faithfully reproduces:
  - Table 5  : all 12 features (types, ordering, ranges)
  - Table 6  : ClaimNb distribution (96% zeros, ~4.7% ones, etc.)
  - Figure 1 : marginal distributions of numeric and categorical features

The synthetic Poisson mean is constructed from a realistic log-linear model so
that model comparisons (AGLM vs GLM vs GBM) produce meaningful, paper-like
results.

The experiment section (run via ``python generate_fremtpl2freq.py``) fits
three AGLM models (Ridge, Elastic Net, LASSO) using the RustyStats backend
and prints a comparison table mirroring the paper's benchmarks.

Usage
-----
    # Data generation only:
    from generate_fremtpl2freq import make_fremtpl2freq
    df = make_fremtpl2freq(n=678_013, seed=42)

    # Full experiment:
    python generate_fremtpl2freq.py [--n 30000] [--seed 42] [--nfolds 5]
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from numpy.random import default_rng


# ---------------------------------------------------------------------------
# Constants matching the paper
# ---------------------------------------------------------------------------

N_FULL = 678_013

VEH_BRANDS = ["B1", "B2", "B3", "B4", "B5", "B6",
               "B10", "B11", "B12", "B13", "B14"]
VEH_BRAND_PROBS = np.array([0.08, 0.22, 0.17, 0.08, 0.07, 0.04,
                              0.11, 0.05, 0.06, 0.07, 0.05])
VEH_BRAND_PROBS /= VEH_BRAND_PROBS.sum()

VEH_GAS = ["Diesel", "Regular"]
VEH_GAS_PROBS = [0.565, 0.435]

AREAS = ["A", "B", "C", "D", "E", "F"]
AREA_PROBS = np.array([0.145, 0.095, 0.255, 0.235, 0.13, 0.14])
AREA_PROBS /= AREA_PROBS.sum()

REGIONS = ["Al", "Aq", "Au", "Ba", "Bo", "Br", "Ce", "Ch",
           "Co", "Fr", "Ha", "Il", "La", "Li", "Mi", "No",
           "Pa", "Pi", "Po", "Pr", "Rh", "Wo"]
REGION_COUNTS = np.array([25000, 8000, 15000, 18000, 50000, 12000,
                           30000, 10000, 5000, 8000, 28000, 160000,
                           22000, 14000, 8000, 40000, 100000, 18000,
                           12000, 20000, 65000, 8013])
REGION_PROBS = REGION_COUNTS / REGION_COUNTS.sum()

NUMERIC_COLS = ["LogExposure", "VehPower", "VehAge",
                "DrivAge", "BonusMalus", "Density"]
CATEG_COLS   = ["VehBrand", "VehGas", "Area", "Region"]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_fremtpl2freq(n: int = N_FULL, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic freMTPL2freq dataset.

    Args:
        n:    Number of policies (default 678,013 to match the paper).
        seed: Random seed for reproducibility.

    Returns:
        ``pd.DataFrame`` with columns matching Table 5 of the paper.
    """
    rng = default_rng(seed)

    # Exposure (years)
    exposure = np.clip(rng.exponential(scale=0.5, size=n), 0.01, 1.0)
    over_one = rng.random(n) < 0.03
    exposure[over_one] = rng.uniform(1.0, 1.5, over_one.sum())

    # VehPower (integer 4–15, spiky distribution)
    veh_power_vals = np.arange(4, 17)
    veh_power_p = np.array([0.04, 0.18, 0.22, 0.08, 0.13, 0.10, 0.06,
                              0.07, 0.04, 0.03, 0.02, 0.02, 0.01])
    veh_power_p /= veh_power_p.sum()
    veh_power = rng.choice(veh_power_vals, size=n, p=veh_power_p)

    # VehAge (0–100, right-skewed)
    veh_age = np.clip(rng.exponential(scale=8.0, size=n).astype(int), 0, 100)

    # DrivAge (18–100, roughly normal peaking ~40–45)
    driv_age = np.clip(rng.normal(loc=44, scale=14, size=n), 18, 100).astype(int)

    # BonusMalus (50–350, spike at 50 then long tail)
    bm_base = np.clip(50 + rng.exponential(scale=18, size=n), 50, 350)
    malus_mask = rng.random(n) < 0.10
    bm_base[malus_mask] = rng.uniform(101, 230, malus_mask.sum())
    bonus_malus = np.round(bm_base).astype(int)

    # Categorical features
    veh_brand = rng.choice(VEH_BRANDS, size=n, p=VEH_BRAND_PROBS)
    veh_gas   = rng.choice(VEH_GAS,    size=n, p=VEH_GAS_PROBS)
    area      = rng.choice(AREAS,      size=n, p=AREA_PROBS)
    region    = rng.choice(REGIONS,    size=n, p=REGION_PROBS)

    # Density (right-skewed, higher in urban areas)
    density_raw = np.clip(rng.exponential(scale=1500, size=n), 1, 30_000)
    area_boost  = {"A": 0.3, "B": 0.5, "C": 0.8, "D": 1.0, "E": 1.5, "F": 3.0}
    density = np.clip(
        density_raw * np.array([area_boost[a] for a in area]), 1, 30_000
    ).astype(int)

    # Realistic non-linear log-linear Poisson mean (mirrors paper Figure 2)
    log_mu  = np.log(np.maximum(exposure, 1e-6))
    log_mu += 0.012 * (bonus_malus - 50) - 3e-5 * (bonus_malus - 50) ** 2
    log_mu += -0.04 * (driv_age - 25)    + 0.0006 * (driv_age - 25) ** 2
    log_mu -= 0.002 * np.maximum(driv_age - 60, 0) ** 1.5 * 0.05
    log_mu += -0.05 * np.log1p(veh_age)
    log_mu += 0.03 * (veh_power - 6) - 0.002 * (veh_power - 6) ** 2
    log_mu += 0.08 * np.log1p(density / 1000)

    area_fx   = {"A": 0.00, "B": -0.02, "C": -0.05, "D": -0.08, "E": 0.02, "F": 0.10}
    gas_fx    = {"Diesel": -0.05, "Regular": 0.00}
    brand_fx  = {b: rng.normal(0, 0.05) for b in VEH_BRANDS}
    brand_fx.update({"B2": -0.04, "B3": -0.02, "B4": 0.05})
    region_fx = {r: rng.normal(0, 0.08) for r in REGIONS}
    region_fx.update({"Il": 0.10, "Pa": 0.12, "Rh": 0.06})

    log_mu += np.array([area_fx[a]   for a in area])
    log_mu += np.array([gas_fx[g]    for g in veh_gas])
    log_mu += np.array([brand_fx[b]  for b in veh_brand])
    log_mu += np.array([region_fx[r] for r in region])
    log_mu += -2.136   # global intercept → ~5% claim rate

    mu       = np.exp(log_mu)
    claim_nb = np.minimum(rng.poisson(mu), 4)  # censor at 4 (paper convention)

    return pd.DataFrame({
        "IDpol":      np.arange(1, n + 1),
        "ClaimNb":    claim_nb,
        "Exposure":   np.round(exposure, 6),
        "VehPower":   veh_power,
        "VehAge":     veh_age,
        "DrivAge":    driv_age,
        "BonusMalus": bonus_malus,
        "VehBrand":   veh_brand,
        "VehGas":     veh_gas,
        "Area":       area,
        "Density":    density,
        "Region":     region,
    })


def print_data_summary(df: pd.DataFrame) -> None:
    """Print a summary matching Table 5 and Table 6 of the paper."""
    print("=" * 62)
    print("freMTPL2freq — Synthetic Dataset Summary")
    print("=" * 62)
    print(f"Total policies : {len(df):,}")
    print(f"Total claims   : {df['ClaimNb'].sum():,}")
    print(f"Total exposure : {df['Exposure'].sum():,.1f} policy-years")
    print(f"Claim freq (claims/policy)      : {df['ClaimNb'].mean():.4f}")
    print(f"Claim rate (claims/exposure-yr) : "
          f"{df['ClaimNb'].sum() / df['Exposure'].sum():.4f}")

    print("\nTable 6 — ClaimNb distribution:")
    vc = df["ClaimNb"].value_counts().sort_index()
    print(f"  {'ClaimNb':>8}  {'# policies':>12}  {'Sum Exposure':>14}")
    for k in vc.index:
        print(f"  {k:>8}  {vc[k]:>12,}  "
              f"{df[df['ClaimNb'] == k]['Exposure'].sum():>14,.1f}")

    print("\nNumeric feature ranges:")
    for col in ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]:
        s = df[col]
        print(f"  {col:12s}  min={s.min():.2f}  "
              f"median={s.median():.2f}  mean={s.mean():.2f}  max={s.max():.2f}")

    print("\nCategorical feature levels:")
    for col in ["VehBrand", "VehGas", "Area", "Region"]:
        print(f"  {col}: {sorted(df[col].unique())}")


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Build feature matrix X and response y from a freMTPL2freq DataFrame.

    ``log(Exposure)`` is included as a numeric predictor.  In a correctly
    specified Poisson model the AGLM will recover a coefficient close to 1.0,
    equivalent to the standard log-exposure offset.  This is the recommended
    workaround when the modelling framework does not accept an explicit offset.

    Args:
        df: DataFrame from :func:`make_fremtpl2freq`.

    Returns:
        Tuple ``(X, y)`` where X is a ``pd.DataFrame`` and y is a 1-D float
        ``np.ndarray`` of claim counts (ClaimNb).
    """
    X = df[["VehPower", "VehAge", "DrivAge",
            "BonusMalus", "Density",
            "VehBrand", "VehGas", "Area", "Region"]].copy()
    X.insert(0, "LogExposure", np.log(np.maximum(df["Exposure"], 1e-6)))
    y = df["ClaimNb"].values.astype(float)
    return X, y


# ---------------------------------------------------------------------------
# AGLM experiment (replaces previous irls_en_cv experiment)
# ---------------------------------------------------------------------------

def run_aglm_experiment(
    n: int = 30_000,
    seed: int = 42,
    nfolds: int = 5,
    nbin_max: int = 20,
    save_plots: bool = True,
) -> dict:
    """Fit Ridge, Elastic Net, and LASSO AGLM models on a freMTPL2freq sample.

    Previously this function called ``irls_en_cv`` directly.  It now uses
    :func:`~aglm.cv_aglm` backed by the RustyStats Rust IRLS engine, which
    runs CV folds in parallel via Rayon and supports all three families
    (Gaussian, Binomial, Poisson) through the same code path.

    Key differences from the old ``irls_en_cv`` approach
    -----------------------------------------------------
    * No manual fold loop — ``cv_aglm`` calls ``fit_cv_path_py`` which runs
      all folds × all lambda values in parallel C++ / Rust.
    * Explicit ``lambda_grid`` is passed to keep runtimes predictable; the
      Rust engine also supports auto-generation if ``lambda_grid=None``.
    * ``selection="min"`` picks the lambda with the lowest CV deviance
      (equivalent to the old default).  Pass ``selection="1se"`` for the
      one-standard-error rule (more conservative; recommended for production).
    * All three families use the same :class:`~aglm.model.RustyStatsEstimator`
      wrapper — no family-specific branching.

    Args:
        n:          Policies to generate (default 30 000; use 678 013 for
                    full-scale paper replication).
        seed:       Random seed.
        nfolds:     CV folds (default 5).
        nbin_max:   Max bins per numeric variable (controls feature-space size;
                    default 20 — increase for finer non-linear resolution).
        save_plots: Save per-variable contribution PNGs (default True).

    Returns:
        ``dict`` mapping ``"Ridge"``, ``"ElasticNet"``, ``"LASSO"`` to fitted
        :class:`~aglm.model.AccurateGLM` objects.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from aglm import cv_aglm, plot_aglm

    warnings.filterwarnings("ignore")

    # ------------------------------------------------------------------ data
    print("=" * 65)
    print("freMTPL2freq  —  AGLM Experiment  (RustyStats backend)")
    print("=" * 65)
    print(f"\n[1] Generating dataset  (n={n:,}, seed={seed}) ...")
    df = make_fremtpl2freq(n=n, seed=seed)
    print_data_summary(df)

    X, y = prepare_features(df)
    print(f"\n    Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} columns")
    print(f"    Numeric        : {[c for c in X.columns if X[c].dtype != object]}")
    print(f"    Categorical    : {[c for c in X.columns if X[c].dtype == object]}")

    # ------------------------------------------------------------------ models
    # 15-point log-spaced grid — enough to locate the CV minimum cleanly.
    # Supplying an explicit grid makes runtimes predictable across machines.
    LAMBDA_GRID = np.logspace(-3, 1, 15)

    print(f"\n[2] Fitting models  (Poisson, nfolds={nfolds}, "
          f"grid={len(LAMBDA_GRID)} lambdas) ...\n")
    print("    Note: LASSO/ElasticNet use relaxed CV-fold convergence")
    print("          (max_iter=5, tol=0.05) — sufficient for lambda selection.")
    print("          Final model is always fitted to full convergence.\n")

    model_specs = [
        (0.0, "Ridge",
         "all variables retained, smooth shrinkage"),
        (0.5, "ElasticNet",
         "grouped shrinkage + selection, l1_ratio=0.5"),
        (1.0, "LASSO",
         "sparse solution, automatic variable selection"),
    ]

    fitted: dict = {}

    for alpha, label, description in model_specs:
        print(f"  {label:12s} (α={alpha}, {description})")
        t0 = time.time()

        model = cv_aglm(
            X, y,
            alpha=alpha,
            nfolds=nfolds,
            family="poisson",
            lambda_grid=LAMBDA_GRID,
            selection="min",        # "1se" for more conservative selection
            nbin_max=nbin_max,
            add_linear_columns=True,
            add_od_columns_of_qualitatives=True,
            add_interaction_columns=False,
        )

        elapsed = time.time() - t0
        coef_s  = model.coef(with_names=True)
        n_nz    = int((coef_s.abs() > 1e-6).sum())
        print(f"    best_λ     = {model.lambda_:.5f}")
        print(f"    deviance   = {model.deviance():.4f}")
        print(f"    non-zero β = {n_nz} / {len(coef_s)}")
        print(f"    wall time  = {elapsed:.1f}s\n")

        fitted[label] = model

    # ------------------------------------------------------------------ compare
    print("[3] Model comparison\n")
    hdr = f"  {'Model':12s}  {'α':>4}  {'best_λ':>9}  {'Deviance':>12}  " \
          f"{'Non-zero':>9}  {'Coefs':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for label, model in fitted.items():
        coef_s = model.coef(with_names=True)
        nz     = int((coef_s.abs() > 1e-6).sum())
        total  = len(coef_s)
        print(f"  {label:12s}  {model.alpha:>4.1f}  {model.lambda_:>9.5f}  "
              f"{model.deviance():>12.4f}  {nz:>9}  {total:>7}")

    # ------------------------------------------------------------------ detail
    best_label = min(fitted, key=lambda k: fitted[k].deviance())
    best_model = fitted[best_label]
    coef_s     = best_model.coef(with_names=True)

    print(f"\n[4] Coefficient detail — {best_label} (lowest deviance)\n")
    print(f"  Intercept     : {best_model.intercept():+.4f}")

    log_exp = coef_s.get("LogExposure", float("nan"))
    print(f"  LogExposure β : {log_exp:+.4f}  "
          f"(ideal exposure offset → 1.0000;  gap = {log_exp - 1:+.4f})")

    nz_coefs = coef_s[coef_s.abs() > 1e-6]
    print(f"\n  Non-zero coefficients : {len(nz_coefs)}")
    print(f"  Top 15 by |β|:")
    for name in nz_coefs.abs().nlargest(15).index:
        print(f"    {name:50s}  {coef_s[name]:+.5f}")

    # ------------------------------------------------------------------ plots
    if save_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        print(f"\n[5] Saving contribution plots ...")
        for label, model in fitted.items():
            fig = plot_aglm(
                model,
                ncols=3,
                show_residuals=True,
                show_cv_curve=True,
                title_prefix=f"freMTPL2freq  {label} — ",
                link_scale=True,    # log scale for Poisson
            )
            fname = f"fremtpl2freq_aglm_{label.lower()}.png"
            fig.savefig(fname, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fname}")

    print("\nExperiment complete.")
    return fitted


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic freMTPL2freq data and run AGLM experiment.\n\n"
            "Examples:\n"
            "  python generate_fremtpl2freq.py                   # default 30k, 5 folds\n"
            "  python generate_fremtpl2freq.py --n 10000         # faster demo\n"
            "  python generate_fremtpl2freq.py --data-only --n 678013  # full dataset\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n",         type=int,  default=30_000,
                        help="Number of policies (default 30 000).")
    parser.add_argument("--seed",      type=int,  default=42,
                        help="Random seed (default 42).")
    parser.add_argument("--nfolds",    type=int,  default=5,
                        help="CV folds (default 5).")
    parser.add_argument("--nbin",      type=int,  default=20,
                        help="Max bins per numeric variable (default 20).")
    parser.add_argument("--no-plots",  action="store_true",
                        help="Skip saving contribution plot PNGs.")
    parser.add_argument("--data-only", action="store_true",
                        help="Generate and summarise data only; skip fitting.")
    args = parser.parse_args()

    if args.data_only:
        df  = make_fremtpl2freq(n=args.n, seed=args.seed)
        print_data_summary(df)
        out = f"freMTPL2freq_n{args.n}_seed{args.seed}.parquet"
        df.to_parquet(out, index=False)
        print(f"\nSaved: {out}")
    else:
        run_aglm_experiment(
            n=args.n,
            seed=args.seed,
            nfolds=args.nfolds,
            nbin_max=args.nbin,
            save_plots=not args.no_plots,
        )
