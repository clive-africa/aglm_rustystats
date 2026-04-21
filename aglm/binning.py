"""
Binning utilities for AGLM.

Python port of R/binning.R from kkondo1981/aglm.

Numeric variables are discretised into bins before auxiliary dummy/basis
columns are constructed.  Three public functions mirror the R originals:

  create_equal_width_bins(x_vec, nbin)   → evenly-spaced breakpoints
  create_equal_freq_bins(x_vec, nbin)    → quantile-based breakpoints
  execute_binning(x_vec, breaks, nbin_max) → assign observations to bins

Reference:
  Fujita, Tanaka, Kondo & Iwasawa (2020).
  AGLM: A Hybrid Modeling Method of GLM and Data Science Techniques.
  Actuarial Colloquium Paris 2020.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def create_equal_width_bins(x_vec: np.ndarray, nbin: int) -> np.ndarray:
    """Create equal-width bin breakpoints.

    Produces *nbin + 1* boundary values uniformly distributed between the
    minimum and maximum of ``x_vec``.

    Args:
        x_vec: 1-D numeric array.
        nbin:  Number of bins.

    Returns:
        1-D array of length ``nbin + 1`` containing the bin boundaries.
    """
    x_arr = np.asarray(x_vec, dtype=float)
    return np.linspace(x_arr.min(), x_arr.max(), nbin + 1)


def create_equal_freq_bins(x_vec: np.ndarray, nbin: int) -> np.ndarray:
    """Create equal-frequency (quantile) bin breakpoints.

    Returns the unique quantile boundaries so that each bin contains
    approximately the same number of observations.

    Args:
        x_vec: 1-D numeric array.
        nbin:  Target number of bins.

    Returns:
        1-D array of unique breakpoints (may be shorter than ``nbin + 1``
        when there are repeated values in ``x_vec``).
    """
    x_arr = np.asarray(x_vec, dtype=float)
    percentiles = np.linspace(0, 100, nbin + 1)
    return np.unique(np.percentile(x_arr, percentiles))


def execute_binning(
    x_vec: np.ndarray,
    breaks: Optional[np.ndarray] = None,
    nbin_max: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign a numeric vector to bins.

    If ``breaks`` is ``None`` the bin boundaries are determined from the data
    using equal-frequency binning with ``min(nbin_max, n_unique_values)`` bins.

    Observations that fall outside the range of ``breaks`` are assigned to the
    nearest edge bin (i.e. the function clamps values, not extrapolates).

    Args:
        x_vec:    1-D numeric array of values to bin.
        breaks:   Pre-computed monotone breakpoints.  If ``None``, computed
                  from ``x_vec``.
        nbin_max: Maximum number of bins when ``breaks`` is ``None``.

    Returns:
        Tuple ``(breaks, labels)`` where:

        * **breaks** – sorted unique breakpoint array of length *n_bins + 1*.
        * **labels** – 1-indexed integer array (length = len(x_vec)) giving
          the bin assignment of each observation.  Compatible with the R
          convention used in the original package.
    """
    x_arr = np.asarray(x_vec, dtype=float)

    if breaks is None:
        n_unique = len(np.unique(x_arr))
        nbin = min(nbin_max, max(n_unique, 1))
        breaks = create_equal_freq_bins(x_arr, nbin)

    breaks = np.sort(np.unique(np.asarray(breaks, dtype=float)))

    # Clamp to [min_break, max_break] so out-of-range values go to edge bins
    clipped = np.clip(x_arr, breaks[0], breaks[-1])

    # searchsorted with side='right' returns index of the right boundary;
    # bin index (0-based) = that index - 1, clamped to valid range.
    raw_idx = np.searchsorted(breaks, clipped, side="right") - 1
    labels_0based = np.clip(raw_idx, 0, len(breaks) - 2)

    # Convert to 1-indexed to match R convention
    labels = labels_0based + 1

    return breaks, labels
