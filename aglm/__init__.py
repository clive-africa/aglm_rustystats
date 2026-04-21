"""
aglm — Accurate Generalized Linear Model
==========================================

Python port of the R package ``aglm`` by Kenji Kondo, Kazuhisa Takahashi,
and Hikari Banno.

Original R package: https://github.com/kkondo1981/aglm

AGLM is a regularised GLM that enriches the feature space through:

* **U-dummies** — one-hot encoding for unordered categorical variables.
* **O-dummies** — piecewise-linear or step-function basis for ordered /
  numeric variables (enables non-linear GLM fits that remain interpretable).
* **L-variables** — absolute-value basis functions (alternative to O-dummies
  with better extrapolation behaviour).
* **Pairwise interactions** — optional outer-product interaction columns.

The augmented design matrix is then passed to an elastic-net regularised GLM
(via scikit-learn) supporting Gaussian, Binomial, and Poisson families.

Reference
---------
Suguru Fujita, Toyoto Tanaka, Kenji Kondo and Hirokazu Iwasawa (2020).
*AGLM: A Hybrid Modeling Method of GLM and Data Science Techniques.*
Actuarial Colloquium Paris 2020.
https://www.institutdesactuaires.com/global/gene/link.php?doc_id=16273&fg=1

Quick-start
-----------
::

    import pandas as pd
    from aglm import cv_aglm, plot_aglm

    # Load data
    df = pd.read_csv("my_data.csv")
    X = df.drop(columns=["y"])
    y = df["y"].values

    # Fit with lambda chosen by 10-fold CV (LASSO by default)
    model = cv_aglm(X, y, alpha=1.0, nfolds=10, family="gaussian")
    print(model)

    # Plot per-variable contributions
    fig = plot_aglm(model)
    fig.savefig("aglm_contributions.png", dpi=150, bbox_inches="tight")

    # Predict on new data
    preds = model.predict(X_new)
"""

from .model import aglm, cv_aglm, cva_aglm, AccurateGLM, CVAAccurateGLM
from .plot import plot_aglm, plot_cva_alpha
from .binning import create_equal_width_bins, create_equal_freq_bins, execute_binning
from .dummies import get_u_dummy_mat, get_o_dummy_mat, get_lvar_mat
from .input import new_input, AGLMInput, VarInfo

__version__ = "0.1.0"
__author__ = "Python port of kkondo1981/aglm (https://github.com/kkondo1981/aglm)"
__license__ = "GPL-2.0"

__all__ = [
    # ---- Fitting functions ----
    "aglm",
    "cv_aglm",
    "cva_aglm",
    # ---- Model classes ----
    "AccurateGLM",
    "CVAAccurateGLM",
    # ---- Input / feature engineering ----
    "new_input",
    "AGLMInput",
    "VarInfo",
    # ---- Plotting ----
    "plot_aglm",
    "plot_cva_alpha",
    # ---- Binning utilities ----
    "create_equal_width_bins",
    "create_equal_freq_bins",
    "execute_binning",
    # ---- Dummy-variable utilities ----
    "get_u_dummy_mat",
    "get_o_dummy_mat",
    "get_lvar_mat",
]
