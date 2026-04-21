"""
Feature-engineering / dummy-variable construction for AGLM.

Python port of R/get-dummies.R from kkondo1981/aglm.

Three basis-expansion types are implemented, matching the R originals:

U-dummy  (Unordered)
    Standard one-hot encoding for *unordered* categorical variables.
    With ``drop_last=True`` (default) the last column is dropped to avoid
    perfect multicollinearity (treatment / reference-cell encoding).
    Column names: ``<var>_UD_1``, ``<var>_UD_2``, ...

O-dummy  (Ordered)
    Piecewise-linear or step-function encoding for *ordered* variables.

    Type ``"C"`` — continuous ramp (default for numeric variables):
        For bin *j* with boundaries [B0, B1]:
            d_j(x) = clip( (x − B0) / (B1 − B0),  0,  1 )
        The resulting basis spans arbitrary piecewise-linear functions on
        the bin grid, enabling the model to capture non-linear shapes while
        remaining interpretable (each coefficient is the slope change at
        the bin edge).

    Type ``"J"`` — jump / step dummies (for ordered categorical variables):
        d_k(x) = 1  if  bin_label(x) > k,  else 0
        Analogous to cumulative indicator variables for ordinal data.

    Column names: ``<var>_OD_1``, ``<var>_OD_2``, ...

L-variable
    Absolute-value basis functions centred at *interior* breakpoints:
        L_k(x) = |x − B_k|,   k = 1 … (n_breaks − 2)
    Together with a linear term these span piecewise-linear functions of
    arbitrary shape — equivalent to a free-knot linear spline.  They produce
    more stable extrapolation behaviour than O-dummies.
    Column names: ``<var>_LV_1``, ``<var>_LV_2``, ...

All functions follow the same return-value convention as the R originals:
a ``dict`` with an ``info`` section (always present) and a ``dummy_mat``
numpy array (omitted when ``only_info=True``).

Reference:
  Fujita, Tanaka, Kondo & Iwasawa (2020).
  AGLM: A Hybrid Modeling Method of GLM and Data Science Techniques.
  Actuarial Colloquium Paris 2020.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .binning import execute_binning


# ---------------------------------------------------------------------------
# U-dummy  (unordered categorical)
# ---------------------------------------------------------------------------

def get_u_dummy_mat(
    x_vec: np.ndarray,
    levels: Optional[List[str]] = None,
    drop_last: bool = True,
    only_info: bool = False,
) -> dict:
    """Create a U-dummy (one-hot) matrix for one categorical variable.

    Args:
        x_vec:     1-D array of integer, string, bool, or categorical values.
        levels:    Ordered list of category labels.  If ``None`` the sorted
                   unique values of ``x_vec`` are used.
        drop_last: If ``True`` the last column is dropped to avoid perfect
                   multicollinearity.
        only_info: If ``True`` return only metadata (``levels``, ``drop_last``)
                   and skip building the matrix.

    Returns:
        ``dict`` with keys:

        * ``"levels"``    – list of category labels (str).
        * ``"drop_last"`` – bool, same as input.
        * ``"dummy_mat"`` – ``np.ndarray`` of shape ``(n, n_levels - drop_last)``
          (absent when ``only_info=True``).
    """
    x_str = np.asarray(x_vec, dtype=str)

    if levels is None:
        levels = sorted(np.unique(x_str).tolist())

    if only_info:
        return {"levels": levels, "drop_last": drop_last}

    n = len(x_str)
    n_cols = len(levels) - int(drop_last)
    dummy_mat = np.zeros((n, n_cols), dtype=float)
    level_to_idx = {lv: i for i, lv in enumerate(levels)}

    for row, val in enumerate(x_str):
        col_idx = level_to_idx.get(val)
        if col_idx is not None and col_idx < n_cols:
            dummy_mat[row, col_idx] = 1.0

    return {"levels": levels, "drop_last": drop_last, "dummy_mat": dummy_mat}


# ---------------------------------------------------------------------------
# O-dummy  (ordered / numeric)
# ---------------------------------------------------------------------------

def get_o_dummy_mat(
    x_vec: np.ndarray,
    breaks: Optional[np.ndarray] = None,
    nbin_max: int = 100,
    dummy_type: str = "C",
    only_info: bool = False,
) -> dict:
    """Create an O-dummy matrix for one numeric or ordered-categorical variable.

    Args:
        x_vec:      1-D numeric (or integer-coded ordered) array.
        breaks:     Pre-computed monotone breakpoints.  If ``None`` computed
                    from ``x_vec`` via equal-frequency binning.
        nbin_max:   Maximum number of bins when ``breaks`` is ``None``.
        dummy_type: ``"C"`` for continuous ramp dummies (numeric variables) or
                    ``"J"`` for step dummies (ordered categoricals).
        only_info:  If ``True`` return only ``{"breaks": ...}``.

    Returns:
        ``dict`` with keys:

        * ``"breaks"``    – 1-D array of bin boundaries.
        * ``"dummy_mat"`` – ``np.ndarray``
          (absent when ``only_info=True``).
    """
    x_arr = np.asarray(x_vec, dtype=float)
    computed_breaks, labels = execute_binning(x_arr, breaks=breaks, nbin_max=nbin_max)

    if only_info:
        return {"breaks": computed_breaks}

    n = len(x_arr)

    if dummy_type == "C":
        # Continuous ramp: each basis function rises linearly across its bin.
        n_bins = len(computed_breaks) - 1
        B0 = computed_breaks[:-1]   # left edges  (n_bins,)
        B1 = computed_breaks[1:]    # right edges (n_bins,)
        width = B1 - B0             # bin widths  (n_bins,)

        # Broadcast: (n, 1) vs (n_bins,)
        X = x_arr[:, np.newaxis]
        with np.errstate(invalid="ignore", divide="ignore"):
            ramp = np.where(width > 0.0, (X - B0) / width, 0.0)
        dummy_mat = np.clip(ramp, 0.0, 1.0)

    elif dummy_type == "J":
        # Step / jump dummies for ordered categorical variables.
        n_breaks = len(computed_breaks)
        k_arr = np.arange(1, n_breaks + 1)           # (n_breaks,)
        dummy_mat = (labels[:, np.newaxis] > k_arr[np.newaxis, :]).astype(float)

    else:
        raise ValueError(
            f"dummy_type must be 'C' or 'J', got {dummy_type!r}."
        )

    return {"breaks": computed_breaks, "dummy_mat": dummy_mat}


# ---------------------------------------------------------------------------
# L-variable  (absolute-value spline basis)
# ---------------------------------------------------------------------------

def get_lvar_mat(
    x_vec: np.ndarray,
    breaks: Optional[np.ndarray] = None,
    nbin_max: int = 100,
    only_info: bool = False,
) -> dict:
    """Create an L-variable matrix (absolute-value spline basis) for one variable.

    Args:
        x_vec:    1-D numeric array.
        breaks:   Pre-computed breakpoints.  If ``None`` computed from ``x_vec``.
        nbin_max: Maximum bins when ``breaks`` is ``None``.
        only_info: If ``True`` return only ``{"breaks": ...}``.

    Returns:
        ``dict`` with keys:

        * ``"breaks"``    – 1-D array of bin boundaries.
        * ``"dummy_mat"`` – ``np.ndarray`` of shape ``(n, n_interior_knots)``
          or ``None`` if there are no interior knots
          (absent when ``only_info=True``).
    """
    x_arr = np.asarray(x_vec, dtype=float)
    computed_breaks, _ = execute_binning(x_arr, breaks=breaks, nbin_max=nbin_max)

    if only_info:
        return {"breaks": computed_breaks}

    # Interior breakpoints only (exclude first and last)
    interior = computed_breaks[1:-1]

    if len(interior) == 0:
        return {"breaks": computed_breaks, "dummy_mat": None}

    dummy_mat = np.abs(x_arr[:, np.newaxis] - interior[np.newaxis, :])
    return {"breaks": computed_breaks, "dummy_mat": dummy_mat}
