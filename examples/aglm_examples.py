"""
examples/aglm_examples.py
==========================
Python equivalents of the R aglm package example scripts.

Mirrors:
  examples/aglm-1.R        → Gaussian regression (Boston housing)
  examples/aglm-2.R        → Binomial regression
  examples/cv-aglm-1.R     → CV for lambda
  examples/cva-aglm-1.R    → CV for alpha + lambda
  examples/lvar-and-extrapolation.R → L-variable demo

Run:
    python examples/aglm_examples.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from aglm import aglm, cv_aglm, cva_aglm, plot_aglm, plot_cva_alpha


from sklearn.datasets import fetch_openml
# Fetch freMTPL2freq (OpenML ID: 41214)
df_freq = fetch_openml(data_id=41214, as_frame=True, parser='pandas')['data']
df_sev = fetch_openml(data_id=41215, as_frame=True, parser='pandas')['data']
# Merge df_freq and df_sev on 'IDpol' to get all rating factors
df_sev = pd.merge(df_sev, df_freq.drop(columns=["ClaimNb", "Exposure"]),  on="IDpol", how="left")


# ============================================================
# Example 1 — Gaussian regression (mirrors aglm-1.R)
# Equivalent dataset: sklearn diabetes (like R Boston housing)
# ============================================================

def example_gaussian():
    print("=" * 60)
    print("Example 1 — Gaussian regression")
    print("=" * 60)

    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=2018
    )

    # ---- Basic fit at a given lambda ----
    model = aglm(X_train, y_train, alpha=1.0, lambda_=0.1)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\naglm  alpha=1.0 lambda=0.10  RMSE: {rmse:.4f}")

    model2 = aglm(X_train, y_train, alpha=1.0, lambda_=1.0)
    y_pred2 = model2.predict(X_test)
    rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2))
    print(f"aglm  alpha=1.0 lambda=1.00  RMSE: {rmse2:.4f}")

    model3 = aglm(X_train, y_train, alpha=0.0, lambda_=0.1)
    y_pred3 = model3.predict(X_test)
    rmse3 = np.sqrt(mean_squared_error(y_test, y_pred3))
    print(f"aglm  alpha=0.0 lambda=0.10  RMSE: {rmse3:.4f}")

    print(f"\n{model}")

    # ---- Plot variable contributions ----
    fig = plot_aglm(model, ncols=4, show_cv_curve=False, title_prefix="Ex1 | ")
    fig.savefig("example1_contributions.png", dpi=120, bbox_inches="tight")
    print("\nSaved: example1_contributions.png")
    plt.close(fig)


# ============================================================
# Example 2 — Binomial regression (mirrors aglm-2.R)
# ============================================================

def example_binomial():
    print("\n" + "=" * 60)
    print("Example 2 — Binomial regression")
    print("=" * 60)

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # 0 = malignant, 1 = benign

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2018, stratify=y
    )

    model = aglm(X_train, y_train, alpha=1.0, lambda_=0.01, family="binomial")
    probs = model.predict(X_test)
    auc = roc_auc_score(y_test, probs)
    print(f"\naglm  alpha=1.0 lambda=0.01  AUC: {auc:.4f}")
    print(f"\n{model}")

    fig = plot_aglm(
        model, ncols=4, show_cv_curve=False,
        which_vars=list(X.columns[:8]),
        title_prefix="Ex2 | Binomial | ",
    )
    fig.savefig("example2_binomial.png", dpi=120, bbox_inches="tight")
    print("Saved: example2_binomial.png")
    plt.close(fig)


# ============================================================
# Example 3 — CV for lambda (mirrors cv-aglm-1.R)
# ============================================================

def example_cv():
    print("\n" + "=" * 60)
    print("Example 3 — cv_aglm: cross-validation for lambda")
    print("=" * 60)

    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=2018
    )

    model = cv_aglm(X_train, y_train, alpha=1.0, nfolds=10)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\ncv_aglm  best lambda: {model.lambda_:.6f}")
    print(f"         RMSE:        {rmse:.4f}")
    print(f"\n{model}")

    fig = plot_aglm(model, ncols=4, show_cv_curve=True, title_prefix="Ex3 | CV | ")
    fig.savefig("example3_cv.png", dpi=120, bbox_inches="tight")
    print("Saved: example3_cv.png")
    plt.close(fig)


# ============================================================
# Example 4 — CV for alpha + lambda (mirrors cva-aglm-1.R)
# ============================================================

def example_cva():
    print("\n" + "=" * 60)
    print("Example 4 — cva_aglm: cross-validation for alpha + lambda")
    print("=" * 60)

    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=2018
    )

    cva = cva_aglm(
        X_train, y_train,
        alpha_grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        nfolds=5,
    )

    print(f"\n{cva}")

    best = cva.best_model
    y_pred = best.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nBest model RMSE on test: {rmse:.4f}")

    fig_alpha = plot_cva_alpha(cva)
    fig_alpha.savefig("example4_cva_alpha.png", dpi=120, bbox_inches="tight")
    print("Saved: example4_cva_alpha.png")
    plt.close(fig_alpha)

    fig_best = plot_aglm(
        best, ncols=4, show_cv_curve=True,
        title_prefix=f"Ex4 | Best α={cva.best_alpha} | "
    )
    fig_best.savefig("example4_best_model.png", dpi=120, bbox_inches="tight")
    print("Saved: example4_best_model.png")
    plt.close(fig_best)


# ============================================================
# Example 5 — L-variable + extrapolation (mirrors R example)
# ============================================================

def example_lvar():
    print("\n" + "=" * 60)
    print("Example 5 — L-variables and extrapolation control")
    print("=" * 60)

    # Synthetic non-linear data
    np.random.seed(0)
    x = np.sort(np.random.uniform(-3, 3, 150))
    y = np.sin(x) + 0.3 * np.random.randn(150)
    X = pd.DataFrame({"x": x})

    X_train, X_test = X.iloc[:100], X.iloc[100:]
    y_train, y_test = y[:100], y[100:]

    model_od = aglm(X_train, y_train, alpha=1.0, lambda_=0.01,
                    use_lvar=False, extrapolation="default")
    model_lv = aglm(X_train, y_train, alpha=1.0, lambda_=0.01,
                    use_lvar=True,  extrapolation="flat")

    # Predict over a wide range including extrapolation region
    x_wide = np.linspace(-5, 5, 300)
    X_wide = pd.DataFrame({"x": x_wide})
    pred_od = model_od.predict(X_wide)
    pred_lv = model_lv.predict(X_wide)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(x, y, s=12, alpha=0.5, color="grey", label="Data")
    ax.plot(x_wide, pred_od, linewidth=2, label="O-dummies (default extrap.)", color="#1976D2")
    ax.plot(x_wide, pred_lv, linewidth=2, label="L-variables (flat extrap.)", color="#E53935",
            linestyle="--")
    ax.axvline(-3, color="grey", linestyle=":", linewidth=0.8, label="Training boundary")
    ax.axvline( 3, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("Predicted y")
    ax.set_title("L-variables vs O-dummies with extrapolation control")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("example5_lvar.png", dpi=120, bbox_inches="tight")
    print("Saved: example5_lvar.png")
    plt.close(fig)

    rmse_od = np.sqrt(mean_squared_error(y_test, model_od.predict(X_test)))
    rmse_lv = np.sqrt(mean_squared_error(y_test, model_lv.predict(X_test)))
    print(f"\nTest RMSE — O-dummies: {rmse_od:.4f}   L-variables: {rmse_lv:.4f}")


# ============================================================
# Example 6 — Categorical variable handling
# ============================================================

def example_categorical():
    print("\n" + "=" * 60)
    print("Example 6 — Categorical variable handling")
    print("=" * 60)

    np.random.seed(42)
    n = 300
    region = np.random.choice(["North", "South", "East", "West"], n)
    age = np.random.uniform(20, 70, n)
    # True effect: region baseline + non-linear age effect
    region_effect = {"North": 0.0, "South": 0.5, "East": -0.3, "West": 1.2}
    y = (
        np.array([region_effect[r] for r in region])
        + 0.02 * (age - 45) ** 2 / 100
        + 0.5 * np.random.randn(n)
    )

    X = pd.DataFrame({"age": age, "region": region})

    model = cv_aglm(X, y, nfolds=5, alpha=1.0)
    rmse = np.sqrt(mean_squared_error(y, model.predict()))
    print(f"\nIn-sample RMSE: {rmse:.4f}")
    print(f"Best lambda:    {model.lambda_:.6f}")

    fig = plot_aglm(model, ncols=2, show_cv_curve=True, title_prefix="Ex6 | Categorical | ")
    fig.savefig("example6_categorical.png", dpi=120, bbox_inches="tight")
    print("Saved: example6_categorical.png")
    plt.close(fig)


# ============================================================
# Run all examples
# ============================================================

if __name__ == "__main__":
    example_gaussian()
    example_binomial()
    example_cv()
    example_cva()
    example_lvar()
    example_categorical()
    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print("=" * 60)
