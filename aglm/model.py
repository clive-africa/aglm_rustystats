"""
AGLM fitting functions — RustyStats backend.

Python port of R/aglm.R, R/cv-aglm.R, and R/cva-aglm.R from kkondo1981/aglm,
with the scikit-learn / statsmodels backend replaced by **RustyStats**
(https://pricingfrontier.github.io/rustystats/).

Three fitting strategies mirror the R package:

  :func:`aglm`        — fit with given ``alpha`` and ``lambda_``
                         (solution at a single point in hyperparameter space).
  :func:`cv_aglm`     — fix ``alpha``; cross-validate to select ``lambda``.
  :func:`cva_aglm`    — cross-validate over both ``alpha`` and ``lambda``.

Backend
-------
All families (Gaussian, Binomial, Poisson) are fitted through the
Rust-accelerated IRLS engine exposed by ``rustystats._rustystats``:

* ``fit_glm_py``      — single elastic-net GLM fit (coordinate-descent IRLS).
* ``fit_cv_path_py``  — full regularisation path with parallel K-fold CV.

Hyperparameter convention (glmnet / AGLM)
-----------------------------------------
* ``alpha``   — elastic-net L1 mixing  (0 = Ridge, 1 = LASSO).
* ``lambda_`` — regularisation strength.

RustyStats uses the same convention with different parameter names:
* ``l1_ratio`` ↔ AGLM ``alpha``
* ``alpha``    ↔ AGLM ``lambda_``

This mapping is applied internally; the public API of this module always uses
the AGLM / glmnet naming.

Reference:
  Fujita, Tanaka, Kondo & Iwasawa (2020).
  AGLM: A Hybrid Modeling Method of GLM and Data Science Techniques.
  Actuarial Colloquium Paris 2020.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# RustyStats Rust core — the two functions we use
from rustystats._rustystats import fit_glm_py as _fit_glm_rust
from rustystats._rustystats import fit_cv_path_py as _fit_cv_path_rust

from .input import AGLMInput, VarInfo, new_input


# ---------------------------------------------------------------------------
# Supported families
# ---------------------------------------------------------------------------

_VALID_FAMILIES = {"gaussian", "binomial", "poisson"}

# Default link functions matching RustyStats expectations
_FAMILY_LINK = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones (intercept) to a design matrix.

    ``fit_glm_py`` does not add an intercept automatically; the caller must
    include it as the first column of ``X``.
    """
    n = X.shape[0]
    return np.column_stack([np.ones(n, dtype=X.dtype), X])


def _inverse_link(eta: np.ndarray, family: str) -> np.ndarray:
    """Apply the canonical inverse link to a linear predictor vector."""
    if family == "gaussian":
        return eta
    if family == "poisson":
        return np.exp(eta)
    if family == "binomial":
        return 1.0 / (1.0 + np.exp(-eta))
    raise ValueError(f"Unknown family: {family!r}")


def _regularization_type(aglm_alpha: float) -> str:
    """Map AGLM/glmnet alpha (L1 mixing) to a RustyStats regularization label."""
    if aglm_alpha <= 0.0:
        return "ridge"
    if aglm_alpha >= 1.0:
        return "lasso"
    return "elastic_net"


# ---------------------------------------------------------------------------
# RustyStatsEstimator — sklearn-compatible wrapper around fit_glm_py
# ---------------------------------------------------------------------------

class RustyStatsEstimator:
    """Thin wrapper around ``rustystats._rustystats.fit_glm_py``.

    Provides the sklearn-compatible interface (``fit``, ``predict``,
    ``coef_``, ``intercept_``, ``score``) expected by :class:`AccurateGLM`.

    Parameters
    ----------
    family :   GLM family — ``"gaussian"``, ``"binomial"``, or ``"poisson"``.
    lambda_ :  Regularisation strength  (AGLM/glmnet convention).
    alpha :    L1 mixing parameter       (AGLM/glmnet convention; 0=Ridge, 1=LASSO).
    max_iter : Maximum IRLS iterations.
    tol :      Convergence tolerance.
    """

    def __init__(
        self,
        family: str = "poisson",
        lambda_: float = 1e-3,
        alpha: float = 1.0,
        max_iter: int = 25,
        tol: float = 1e-8,
    ) -> None:
        self.family = family
        self.lambda_ = lambda_
        self.alpha = alpha          # L1 mixing (AGLM convention)
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RustyStatsEstimator":
        """Fit the GLM on the augmented AGLM design matrix ``X``.

        An intercept column is prepended internally — ``X`` should NOT
        already contain one.

        Args:
            X: 2-D float array, shape ``(n, p)`` — the AGLM design matrix
               *without* an intercept column.
            y: 1-D float response vector.

        Returns:
            ``self``
        """
        X_int = _add_intercept(np.asarray(X, dtype=float))
        y_arr = np.asarray(y, dtype=float).ravel()

        link = _FAMILY_LINK[self.family]

        self._result = _fit_glm_rust(
            y_arr,
            X_int,
            self.family,
            link,          # explicit canonical link
            1.5,           # var_power (Tweedie — unused for our families)
            1.0,           # theta      (NegBinomial  — unused)
            None,          # offset
            None,          # weights
            self.lambda_,  # alpha in RustyStats = lambda_ in AGLM
            self.alpha,    # l1_ratio in RustyStats = alpha in AGLM
            self.max_iter,
            self.tol,
        )

        params = np.asarray(self._result.params)  # shape (p+1,) — first is intercept
        self.intercept_ = float(params[0])
        self.coef_ = params[1:].reshape(1, -1)    # shape (1, p) — sklearn convention
        self._fitted_X = X                         # keep for score()
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return response-scale predictions for ``X``.

        Args:
            X: 2-D array, shape ``(n, p)`` — no intercept column.

        Returns:
            1-D ``np.ndarray`` of fitted values on the response scale.
        """
        eta = X @ self.coef_.ravel() + self.intercept_
        return _inverse_link(eta, self.family)

    # ------------------------------------------------------------------
    # predict_linear
    # ------------------------------------------------------------------

    def predict_linear(self, X: np.ndarray) -> np.ndarray:
        """Return the linear predictor η = X β + intercept.

        Args:
            X: 2-D array, shape ``(n, p)`` — no intercept column.

        Returns:
            1-D ``np.ndarray`` of linear predictor values.
        """
        return X @ self.coef_.ravel() + self.intercept_

    # ------------------------------------------------------------------
    # score  (negative deviance — higher = better, for CV scoring)
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return negative deviance on ``(X, y)`` — higher is better.

        Args:
            X: 2-D array, shape ``(n, p)`` — no intercept column.
            y: 1-D response vector.

        Returns:
            Negative deviance (scalar float).
        """
        mu = self.predict(X)
        y = np.asarray(y, dtype=float)
        eps = 1e-12

        if self.family == "gaussian":
            dev = float(np.sum((y - mu) ** 2))
        elif self.family == "binomial":
            mu = np.clip(mu, eps, 1.0 - eps)
            dev = float(-2.0 * np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)))
        elif self.family == "poisson":
            mu = np.clip(mu, eps, None)
            dev = float(2.0 * np.sum(
                np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
            ))
        else:
            raise ValueError(f"Unknown family: {self.family!r}")

        return -dev

    # ------------------------------------------------------------------
    # sklearn compatibility stubs
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        return {
            "family": self.family,
            "lambda_": self.lambda_,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def set_params(self, **params) -> "RustyStatsEstimator":
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# AccurateGLM — fitted model object
# ---------------------------------------------------------------------------

class AccurateGLM:
    """Fitted AGLM model.

    Returned by :func:`aglm` and :func:`cv_aglm`.

    Attributes:
        backend_model:  Fitted :class:`RustyStatsEstimator`.
        aglm_input:     :class:`~aglm.input.AGLMInput` object (variable metadata
                        + training data).
        family:         Distribution family used (``"gaussian"`` etc.).
        alpha:          Elastic-net L1 mixing parameter (0=Ridge, 1=LASSO).
        lambda_:        Regularisation strength used (best lambda from CV when
                        fitted with :func:`cv_aglm`).
        cv_results:     ``dict`` with cross-validation diagnostics (populated by
                        :func:`cv_aglm` and :func:`cva_aglm`; ``None`` for
                        :func:`aglm`).
    """

    def __init__(
        self,
        backend_model: RustyStatsEstimator,
        aglm_input: AGLMInput,
        family: str,
        alpha: float,
        lambda_: float,
        y: np.ndarray,
        cv_results: Optional[dict] = None,
    ) -> None:
        self.backend_model = backend_model
        self.aglm_input = aglm_input
        self.family = family
        self.alpha = alpha
        self.lambda_ = lambda_
        self._y = y
        self.cv_results = cv_results

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        x_new: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        predict_type: str = "response",
    ) -> np.ndarray:
        """Make predictions for new data.

        Args:
            x_new:        New feature matrix.  If ``None`` uses training data
                          (in-sample predictions).
            predict_type: ``"response"`` (default, on the response scale) or
                          ``"link"`` (linear predictor, before link function).

        Returns:
            1-D ``np.ndarray`` of predictions.
        """
        X = (
            self.aglm_input.get_design_matrix()
            if x_new is None
            else self.aglm_input.transform(x_new)
        )

        if predict_type == "link":
            return self.backend_model.predict_linear(X)
        return self.backend_model.predict(X)

    # ------------------------------------------------------------------
    # coef
    # ------------------------------------------------------------------

    def coef(self, with_names: bool = False) -> Union[np.ndarray, pd.Series]:
        """Return the fitted coefficient vector (excluding intercept).

        Args:
            with_names: If ``True`` return a :class:`pandas.Series` indexed
                        by feature name; otherwise a plain ``np.ndarray``.

        Returns:
            Coefficient array or Series.
        """
        m = self.backend_model
        if not hasattr(m, "coef_"):
            raise AttributeError("Model has not been fitted yet.")

        coef_arr = m.coef_.ravel()

        if with_names:
            feat_names = self.aglm_input.get_feature_names()
            if len(feat_names) == len(coef_arr):
                return pd.Series(coef_arr, index=feat_names, name="coef")
            warnings.warn(
                "Feature name count does not match coefficient count; "
                "returning unnamed array.",
                stacklevel=2,
            )
        return coef_arr

    # ------------------------------------------------------------------
    # intercept
    # ------------------------------------------------------------------

    def intercept(self) -> float:
        """Return the fitted intercept."""
        m = self.backend_model
        if hasattr(m, "intercept_"):
            ic = m.intercept_
            return float(ic[0] if hasattr(ic, "__len__") else ic)
        return 0.0

    # ------------------------------------------------------------------
    # deviance
    # ------------------------------------------------------------------

    def deviance(self) -> float:
        """Return model deviance (lower is better).

        * Gaussian:  residual sum of squares.
        * Binomial:  −2 × log-likelihood.
        * Poisson:   scaled deviance.
        """
        y = self._y
        mu = self.predict()

        if self.family == "gaussian":
            return float(np.sum((y - mu) ** 2))

        elif self.family == "binomial":
            eps = 1e-15
            mu = np.clip(mu, eps, 1 - eps)
            return float(
                -2.0 * np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu))
            )

        elif self.family == "poisson":
            eps = 1e-15
            mu = np.clip(mu, eps, None)
            return float(
                2.0 * np.sum(np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
            )

        raise ValueError(f"Unknown family: {self.family!r}")

    # ------------------------------------------------------------------
    # residuals
    # ------------------------------------------------------------------

    def residuals(self, residual_type: str = "response") -> np.ndarray:
        """Return residuals of the fitted model.

        Args:
            residual_type: ``"response"`` (y − μ), ``"pearson"`` (normalised
                           by variance), or ``"deviance"`` (Gaussian only).

        Returns:
            1-D ``np.ndarray``.
        """
        y = self._y
        mu = self.predict()

        if residual_type == "response":
            return y - mu

        if residual_type == "pearson":
            if self.family == "gaussian":
                return y - mu
            if self.family == "binomial":
                return (y - mu) / np.sqrt(np.maximum(mu * (1.0 - mu), 1e-15))
            if self.family == "poisson":
                return (y - mu) / np.sqrt(np.maximum(mu, 1e-15))

        if residual_type == "deviance":
            if self.family == "gaussian":
                return y - mu
            raise NotImplementedError(
                "Deviance residuals are only implemented for Gaussian family."
            )

        raise ValueError(
            f"residual_type must be 'response', 'pearson', or 'deviance', "
            f"got {residual_type!r}."
        )

    # ------------------------------------------------------------------
    # __repr__ / __str__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_coef = len(self.coef()) if hasattr(self.backend_model, "coef_") else "?"
        cv_info = (
            f", cv_lambda_path_len={len(self.cv_results['lambda_grid'])}"
            if self.cv_results else ""
        )
        return (
            f"AccurateGLM(family={self.family!r}, alpha={self.alpha:.4g}, "
            f"lambda={self.lambda_:.4g}, n_coef={n_coef}{cv_info})"
        )

    def __str__(self) -> str:
        lines = [
            "Accurate Generalized Linear Model",
            f"  Family   : {self.family}",
            f"  Alpha    : {self.alpha}",
            f"  Lambda   : {self.lambda_:.6g}",
            f"  Intercept: {self.intercept():.6g}",
            f"  Backend  : RustyStats (Rust IRLS)",
        ]
        if hasattr(self.backend_model, "coef_"):
            lines.append(f"  N coefs  : {len(self.coef())}")
        if self.cv_results:
            best_score = self.cv_results.get("best_cv_score")
            if best_score is not None:
                lines.append(f"  Best CV  : {best_score:.6g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CVAAccurateGLM — result of cva_aglm
# ---------------------------------------------------------------------------

@dataclass
class CVAAccurateGLM:
    """Result of :func:`cva_aglm`.

    Stores fitted models and CV diagnostics for every ``alpha`` value tested.

    Attributes:
        models:     ``{alpha: AccurateGLM}`` — best model for each alpha.
        cv_scores:  ``{alpha: float}`` — best CV score for each alpha.
        best_alpha: Alpha value with the best CV score.
        best_model: :class:`AccurateGLM` fitted at ``(best_alpha, best_lambda)``.
    """

    models: Dict[float, AccurateGLM]
    cv_scores: Dict[float, float]
    best_alpha: float
    best_model: AccurateGLM

    def __repr__(self) -> str:
        return (
            f"CVAAccurateGLM("
            f"best_alpha={self.best_alpha:.4g}, "
            f"best_model={self.best_model!r})"
        )

    def __str__(self) -> str:
        lines = ["CV results over alpha grid:", "  alpha    best_cv_score"]
        for a, s in sorted(self.cv_scores.items()):
            marker = " ← best" if a == self.best_alpha else ""
            lines.append(f"  {a:.4f}  {s:.6g}{marker}")
        lines.append(f"\nbest_alpha: {self.best_alpha}")
        lines.append(str(self.best_model))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _build_backend(family: str, lambda_: float, alpha: float) -> RustyStatsEstimator:
    """Construct a :class:`RustyStatsEstimator` for the given family and hyperparameters.

    All three families (Gaussian, Binomial, Poisson) use the same Rust IRLS
    engine — no family-specific branching is needed here.

    Args:
        family:  ``"gaussian"``, ``"binomial"``, or ``"poisson"``.
        lambda_: Regularisation strength (AGLM / glmnet convention).
        alpha:   L1 mixing parameter (0 = Ridge, 1 = LASSO).
    """
    if family not in _VALID_FAMILIES:
        raise ValueError(
            f"family must be one of {sorted(_VALID_FAMILIES)}, got {family!r}."
        )
    return RustyStatsEstimator(family=family, lambda_=lambda_, alpha=alpha)


# ---------------------------------------------------------------------------
# Shared input kwargs
# ---------------------------------------------------------------------------

_INPUT_KWARG_NAMES = frozenset({
    "qualitative_vars_ud_only",
    "qualitative_vars_both",
    "qualitative_vars_od_only",
    "quantitative_vars",
    "use_lvar",
    "extrapolation",
    "add_linear_columns",
    "add_od_columns_of_qualitatives",
    "add_interaction_columns",
    "od_type_of_quantitatives",
    "nbin_max",
    "bins_list",
    "bins_names",
})


def _split_kwargs(kwargs: dict):
    """Separate new_input kwargs from any remaining kwargs."""
    input_kw = {k: v for k, v in kwargs.items() if k in _INPUT_KWARG_NAMES}
    rest_kw  = {k: v for k, v in kwargs.items() if k not in _INPUT_KWARG_NAMES}
    return input_kw, rest_kw


# ---------------------------------------------------------------------------
# aglm — basic fit
# ---------------------------------------------------------------------------

def aglm(
    x: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    alpha: float = 1.0,
    lambda_: float = 1e-3,
    family: str = "gaussian",
    qualitative_vars_ud_only: Optional[List] = None,
    qualitative_vars_both: Optional[List] = None,
    qualitative_vars_od_only: Optional[List] = None,
    quantitative_vars: Optional[List] = None,
    use_lvar: bool = False,
    extrapolation: str = "default",
    add_linear_columns: bool = True,
    add_od_columns_of_qualitatives: bool = True,
    add_interaction_columns: bool = False,
    od_type_of_quantitatives: str = "C",
    nbin_max: Optional[int] = None,
    bins_list: Optional[List] = None,
    bins_names: Optional[List] = None,
) -> AccurateGLM:
    """Fit an AGLM model with given ``alpha`` and ``lambda_``.

    The function:

    1. Constructs the augmented design matrix via :func:`~aglm.input.new_input`.
    2. Fits a regularised GLM through the RustyStats Rust IRLS engine.

    Args:
        x:                             Feature matrix (``DataFrame`` or ``ndarray``).
                                       Numeric columns → quantitative;
                                       object / bool / categorical → qualitative.
        y:                             Response vector (1-D).
        alpha:                         Elastic-net L1 mixing (1.0 = LASSO, 0.0 = Ridge).
        lambda_:                       Regularisation strength.
        family:                        ``"gaussian"``, ``"binomial"``, or ``"poisson"``.
        qualitative_vars_ud_only:      Override columns to unordered categorical.
        qualitative_vars_both:         Override columns to ordered categorical (U + O dummies).
        qualitative_vars_od_only:      Override columns to O-dummies only.
        quantitative_vars:             Override columns to numeric.
        use_lvar:                      Use L-variables instead of O-dummies for numeric.
        extrapolation:                 ``"default"`` or ``"flat"``.
        add_linear_columns:            Include raw linear term for numeric columns.
        add_od_columns_of_qualitatives: Include O-dummies for ordered categoricals.
        add_interaction_columns:       Add pairwise interaction columns.
        od_type_of_quantitatives:      ``"C"`` (ramp) or ``"N"`` (none).
        nbin_max:                      Max bins for numeric discretisation.
        bins_list:                     Pre-computed breakpoints.
        bins_names:                    Names / indices for ``bins_list``.

    Returns:
        Fitted :class:`AccurateGLM` object.

    Example::

        import pandas as pd
        from aglm import aglm

        X = pd.DataFrame({"age": [25, 40, 60], "brand": ["A", "B", "A"]})
        y = [0.05, 0.08, 0.12]
        model = aglm(X, y, alpha=1.0, lambda_=0.01, family="poisson")
        print(model)
    """
    if family not in _VALID_FAMILIES:
        raise ValueError(
            f"family must be one of {sorted(_VALID_FAMILIES)}, got {family!r}."
        )

    y = np.asarray(y, dtype=float).ravel()

    aglm_input = new_input(
        x,
        qualitative_vars_ud_only=qualitative_vars_ud_only,
        qualitative_vars_both=qualitative_vars_both,
        qualitative_vars_od_only=qualitative_vars_od_only,
        quantitative_vars=quantitative_vars,
        use_lvar=use_lvar,
        extrapolation=extrapolation,
        add_linear_columns=add_linear_columns,
        add_od_columns_of_qualitatives=add_od_columns_of_qualitatives,
        add_interaction_columns=add_interaction_columns,
        od_type_of_quantitatives=od_type_of_quantitatives,
        nbin_max=nbin_max,
        bins_list=bins_list,
        bins_names=bins_names,
    )

    X_design = aglm_input.get_design_matrix()

    backend = _build_backend(family, lambda_, alpha)
    backend.fit(X_design, y)

    return AccurateGLM(
        backend_model=backend,
        aglm_input=aglm_input,
        family=family,
        alpha=alpha,
        lambda_=lambda_,
        y=y,
    )


# ---------------------------------------------------------------------------
# cv_aglm — cross-validation for lambda using RustyStats parallel CV path
# ---------------------------------------------------------------------------

def cv_aglm(
    x: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    alpha: float = 1.0,
    nfolds: int = 10,
    family: str = "gaussian",
    lambda_grid: Optional[np.ndarray] = None,
    n_alphas: int = 50,
    selection: str = "min",
    cv_seed: Optional[int] = None,
    qualitative_vars_ud_only: Optional[List] = None,
    qualitative_vars_both: Optional[List] = None,
    qualitative_vars_od_only: Optional[List] = None,
    quantitative_vars: Optional[List] = None,
    use_lvar: bool = False,
    extrapolation: str = "default",
    add_linear_columns: bool = True,
    add_od_columns_of_qualitatives: bool = True,
    add_interaction_columns: bool = False,
    od_type_of_quantitatives: str = "C",
    nbin_max: Optional[int] = None,
    bins_list: Optional[List] = None,
    bins_names: Optional[List] = None,
) -> AccurateGLM:
    """Fit AGLM with given ``alpha``; select ``lambda`` via K-fold CV.

    Uses ``rustystats._rustystats.fit_cv_path_py`` — a parallel Rust
    implementation that fits all folds × all alpha values simultaneously via
    Rayon, typically 5–20× faster than a pure-Python loop.

    ``selection="min"`` picks the lambda with the lowest CV deviance.
    ``selection="1se"`` picks the largest lambda within one standard error of
    the minimum (more conservative; recommended for production pricing models).

    Args:
        x:            Feature matrix.
        y:            Response vector.
        alpha:        Fixed elastic-net L1 mixing parameter.
        nfolds:       Number of CV folds.
        family:       ``"gaussian"``, ``"binomial"``, or ``"poisson"``.
        lambda_grid:  Sequence of lambda values to evaluate.  If ``None``
                      a log-spaced grid of ``n_alphas`` values is generated
                      automatically by the Rust engine.
        n_alphas:     Size of the auto-generated lambda grid (ignored when
                      ``lambda_grid`` is provided).
        selection:    ``"min"`` or ``"1se"`` — CV selection rule.
        cv_seed:      Random seed for reproducible fold assignment.
        **:           All feature-engineering arguments from :func:`aglm`.

    Returns:
        :class:`AccurateGLM` fitted at the CV-optimal ``lambda``.
    """
    if family not in _VALID_FAMILIES:
        raise ValueError(
            f"family must be one of {sorted(_VALID_FAMILIES)}, got {family!r}."
        )
    if selection not in ("min", "1se"):
        raise ValueError(f"selection must be 'min' or '1se', got {selection!r}.")

    y = np.asarray(y, dtype=float).ravel()

    aglm_input = new_input(
        x,
        qualitative_vars_ud_only=qualitative_vars_ud_only,
        qualitative_vars_both=qualitative_vars_both,
        qualitative_vars_od_only=qualitative_vars_od_only,
        quantitative_vars=quantitative_vars,
        use_lvar=use_lvar,
        extrapolation=extrapolation,
        add_linear_columns=add_linear_columns,
        add_od_columns_of_qualitatives=add_od_columns_of_qualitatives,
        add_interaction_columns=add_interaction_columns,
        od_type_of_quantitatives=od_type_of_quantitatives,
        nbin_max=nbin_max,
        bins_list=bins_list,
        bins_names=bins_names,
    )

    X_design = aglm_input.get_design_matrix()
    X_int = _add_intercept(X_design)   # Rust engine expects intercept in X
    link = _FAMILY_LINK[family]

    # ------------------------------------------------------------------
    # Build lambda grid
    # ------------------------------------------------------------------
    if lambda_grid is not None:
        alphas_list = sorted(np.asarray(lambda_grid, dtype=float).tolist())
    else:
        # Let Rust generate a log-spaced grid via fit_cv_path_py (pass None)
        alphas_list = None

    # ------------------------------------------------------------------
    # Run parallel CV path through Rust
    # ------------------------------------------------------------------
    # CV folds only need approximate deviance estimates, so we use relaxed
    # convergence settings.  LASSO coordinate-descent is more sensitive to
    # iteration count than Ridge, so we cap the grid size too.
    #
    # Rule of thumb:
    #   Ridge  (alpha=0):   20 iterations / 1e-4 tol — converges fast.
    #   Lasso  (alpha=1):   8  iterations / 1e-2 tol — prevent stalling.
    #   Enet   (0<alpha<1): interpolate linearly between the two.
    _is_ridge = float(alpha) <= 0.0
    _is_lasso = float(alpha) >= 1.0

    # CV folds only need approximate deviance.  Ridge converges in ~20 IRLS
    # steps; LASSO/Enet coordinate-descent is much slower, so we use very
    # relaxed settings and a smaller grid.
    cv_max_iter = 20 if _is_ridge else 5
    cv_tol      = 1e-4 if _is_ridge else 5e-2

    # For non-Ridge paths, cap the lambda grid to 15 points to keep runtime
    # bounded.  Explicit lambda_grid is passed through unchanged.
    if alphas_list is None and not _is_ridge:
        from rustystats.regularization_path import (
            compute_alpha_max,
            generate_alpha_path,
        )
        a_max = compute_alpha_max(X_int, y, float(alpha))
        alphas_list = generate_alpha_path(a_max, min(n_alphas, 15)).tolist()

    rust_cv = _fit_cv_path_rust(
        y,
        X_int,
        family,
        link,
        1.5,           # var_power  (unused for our families)
        1.0,           # theta      (unused)
        None,          # offset
        None,          # weights
        alphas_list,   # None → auto-generate inside Rust (Ridge / EN only)
        float(alpha),  # l1_ratio (RustyStats) = alpha (AGLM)
        nfolds,
        cv_max_iter,
        cv_tol,
        cv_seed,
    )

    # rust_cv keys: 'alphas', 'cv_deviance_mean', 'cv_deviance_se', 'best_alpha', 'best_cv_deviance'
    lambda_path  = np.asarray(rust_cv["alphas"])
    mean_devs    = np.asarray(rust_cv["cv_deviance_mean"])
    se_devs      = np.asarray(rust_cv["cv_deviance_se"])

    # ------------------------------------------------------------------
    # Select best lambda
    # ------------------------------------------------------------------
    if selection == "min":
        best_idx = int(np.argmin(mean_devs))
    else:
        # 1-SE rule: largest lambda (most regularised) within 1 SE of the minimum
        min_dev = mean_devs.min()
        min_se  = se_devs[int(np.argmin(mean_devs))]
        threshold = min_dev + min_se
        eligible  = np.where(mean_devs <= threshold)[0]
        best_idx  = int(eligible[np.argmax(lambda_path[eligible])])

    best_lambda = float(lambda_path[best_idx])
    best_cv_dev = float(mean_devs[best_idx])

    # ------------------------------------------------------------------
    # Final fit at the selected lambda on the full training set
    # ------------------------------------------------------------------
    best_backend = _build_backend(family, best_lambda, alpha)
    best_backend.fit(X_design, y)

    # CV scores stored as negative deviance (higher = better) for
    # compatibility with the plotting code that expects "higher is better"
    cv_results = {
        "lambda_grid":   lambda_path,
        "mean_cv_score": -mean_devs,
        "best_cv_score": -best_cv_dev,
    }

    return AccurateGLM(
        backend_model=best_backend,
        aglm_input=aglm_input,
        family=family,
        alpha=alpha,
        lambda_=best_lambda,
        y=y,
        cv_results=cv_results,
    )


# ---------------------------------------------------------------------------
# cva_aglm — cross-validation for both alpha and lambda
# ---------------------------------------------------------------------------

def cva_aglm(
    x: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    alpha_grid: Optional[np.ndarray] = None,
    nfolds: int = 10,
    family: str = "gaussian",
    lambda_grid: Optional[np.ndarray] = None,
    n_alphas: int = 50,
    selection: str = "min",
    cv_seed: Optional[int] = None,
    qualitative_vars_ud_only: Optional[List] = None,
    qualitative_vars_both: Optional[List] = None,
    qualitative_vars_od_only: Optional[List] = None,
    quantitative_vars: Optional[List] = None,
    use_lvar: bool = False,
    extrapolation: str = "default",
    add_linear_columns: bool = True,
    add_od_columns_of_qualitatives: bool = True,
    add_interaction_columns: bool = False,
    od_type_of_quantitatives: str = "C",
    nbin_max: Optional[int] = None,
    bins_list: Optional[List] = None,
    bins_names: Optional[List] = None,
) -> CVAAccurateGLM:
    """Fit AGLM with cross-validation over *both* ``alpha`` and ``lambda``.

    Calls :func:`cv_aglm` once per ``alpha`` value and returns the best model.
    Each ``cv_aglm`` call uses the parallel Rust CV path, so the full grid
    search is still fast.

    Args:
        x:           Feature matrix.
        y:           Response vector.
        alpha_grid:  Alpha (L1 mixing) values to search.  Defaults to
                     ``[0.0, 0.1, 0.2, …, 1.0]``.
        nfolds:      Number of CV folds.
        family:      ``"gaussian"``, ``"binomial"``, or ``"poisson"``.
        lambda_grid: Lambda values per alpha.  If ``None`` auto-generated.
        n_alphas:    Grid size when ``lambda_grid`` is ``None``.
        selection:   ``"min"`` or ``"1se"`` — CV selection rule.
        cv_seed:     Random seed for reproducible fold assignment.
        **:          All feature-engineering arguments from :func:`aglm`.

    Returns:
        :class:`CVAAccurateGLM` containing all fitted models and the best model.
    """
    if alpha_grid is None:
        alpha_grid = np.round(np.linspace(0.0, 1.0, 11), 4)

    fe_kwargs = dict(
        qualitative_vars_ud_only=qualitative_vars_ud_only,
        qualitative_vars_both=qualitative_vars_both,
        qualitative_vars_od_only=qualitative_vars_od_only,
        quantitative_vars=quantitative_vars,
        use_lvar=use_lvar,
        extrapolation=extrapolation,
        add_linear_columns=add_linear_columns,
        add_od_columns_of_qualitatives=add_od_columns_of_qualitatives,
        add_interaction_columns=add_interaction_columns,
        od_type_of_quantitatives=od_type_of_quantitatives,
        nbin_max=nbin_max,
        bins_list=bins_list,
        bins_names=bins_names,
    )

    models: Dict[float, AccurateGLM] = {}
    cv_scores: Dict[float, float] = {}

    for alpha_val in alpha_grid:
        a = float(alpha_val)
        m = cv_aglm(
            x, y,
            alpha=a,
            nfolds=nfolds,
            family=family,
            lambda_grid=lambda_grid,
            n_alphas=n_alphas,
            selection=selection,
            cv_seed=cv_seed,
            **fe_kwargs,
        )
        best_score = float(m.cv_results.get("best_cv_score", 0.0))
        models[a] = m
        cv_scores[a] = best_score

    best_alpha = max(cv_scores, key=cv_scores.__getitem__)

    return CVAAccurateGLM(
        models=models,
        cv_scores=cv_scores,
        best_alpha=best_alpha,
        best_model=models[best_alpha],
    )
