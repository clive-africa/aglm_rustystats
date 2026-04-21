"""
AGLM input processing and design-matrix construction.

Python port of R/aglm-input.R from kkondo1981/aglm.

The :class:`AGLMInput` class:

* Inspects each column of the incoming ``DataFrame``.
* Decides whether it is *quantitative* (numeric) or *qualitative* (categorical).
* For quantitative variables:
    - Optional raw linear term.
    - O-dummy columns (type ``"C"``) or L-variable columns.
* For qualitative variables:
    - U-dummy columns (one-hot).
    - Optionally O-dummy columns (type ``"J"``) for ordered categoricals.
* Optionally: pairwise interaction columns (outer product of linear terms).
* Assembles the full augmented design matrix ``X̃`` passed to the elastic-net
  backend.

The :func:`new_input` factory mirrors the R ``newInput()`` function and is
the recommended way to construct ``AGLMInput`` objects.

Reference:
  Fujita, Tanaka, Kondo & Iwasawa (2020).
  AGLM: A Hybrid Modeling Method of GLM and Data Science Techniques.
  Actuarial Colloquium Paris 2020.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .dummies import get_u_dummy_mat, get_o_dummy_mat, get_lvar_mat


# ---------------------------------------------------------------------------
# VarInfo — metadata for one original variable
# ---------------------------------------------------------------------------

@dataclass
class VarInfo:
    """Stores the feature-engineering specification for one input column.

    This is the Python equivalent of the per-variable list entries in the R
    ``AGLM_Input`` S4 class.

    Attributes:
        idx:          Position in ``vars_info`` list (0-based).
        name:         Column name (str).
        data_col_idx: Column index in the original ``DataFrame`` (−1 for interactions).
        type:         ``"quan"`` | ``"qual"`` | ``"inter"``.
        use_linear:   Include the raw numeric column as a linear term.
        use_ud:       Include U-dummy (one-hot) columns.
        use_od:       Include O-dummy columns.
        use_lv:       Include L-variable columns.
        od_type:      ``"C"`` (ramp) or ``"J"`` (step) — only relevant when ``use_od``.
        extrapolation: ``"default"`` or ``"flat"`` (clamp predictions to training range).
        ud_info:      Dict returned by :func:`get_u_dummy_mat` with ``only_info=True``.
        od_info:      Dict returned by :func:`get_o_dummy_mat` with ``only_info=True``.
        lv_info:      Dict returned by :func:`get_lvar_mat`  with ``only_info=True``.
        var_idx1:     Index of first variable in an interaction term.
        var_idx2:     Index of second variable in an interaction term.
    """

    idx: int
    name: str
    data_col_idx: int
    type: str                       # "quan" | "qual" | "inter"
    use_linear: bool = False
    use_ud: bool = False
    use_od: bool = False
    use_lv: bool = False
    od_type: str = "C"
    extrapolation: str = "default"
    ud_info: Optional[dict] = None
    od_info: Optional[dict] = None
    lv_info: Optional[dict] = None
    var_idx1: Optional[int] = None  # for interactions
    var_idx2: Optional[int] = None  # for interactions


# ---------------------------------------------------------------------------
# AGLMInput
# ---------------------------------------------------------------------------

class AGLMInput:
    """Stores training data + variable metadata; builds the augmented design matrix.

    This is the Python equivalent of the ``AGLM_Input`` S4 class in the R package.

    Attributes:
        data:      Original training ``DataFrame`` (column types preserved).
        vars_info: List of :class:`VarInfo` objects, one per original variable
                   plus one per interaction pair (if enabled).
    """

    def __init__(self, data: pd.DataFrame, vars_info: List[VarInfo]) -> None:
        self.data: pd.DataFrame = data
        self.vars_info: List[VarInfo] = vars_info
        self._col_name_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(data.columns)
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_extrapolation(
        self, x_vec: np.ndarray, vi: VarInfo
    ) -> np.ndarray:
        """Clamp values to the training range when ``extrapolation='flat'``."""
        if vi.extrapolation != "flat":
            return x_vec
        info = vi.lv_info if vi.use_lv else vi.od_info
        if info is None:
            return x_vec
        b = info["breaks"]
        return np.clip(x_vec, b[0], b[-1])

    def _column_block(
        self,
        x_vec: np.ndarray,
        vi: VarInfo,
        drop_od: bool = False,
    ) -> Optional[np.ndarray]:
        """Build the matrix block for a *single* non-interaction variable.

        Args:
            x_vec:   Raw data column (1-D array).
            vi:      VarInfo for this column.
            drop_od: When ``True`` skip O-dummy and L-variable columns
                     (used for computing interaction inputs).

        Returns:
            2-D ``np.ndarray`` or ``None`` if the variable produces no columns.
        """
        x_vec = self._apply_extrapolation(x_vec, vi)
        parts: List[np.ndarray] = []

        if vi.use_linear:
            parts.append(x_vec.astype(float).reshape(-1, 1))

        if vi.use_lv and not drop_od:
            res = get_lvar_mat(x_vec.astype(float), breaks=vi.lv_info["breaks"])
            mat = res["dummy_mat"]
            if mat is not None:
                parts.append(mat)

        if vi.use_od and not drop_od:
            if vi.type == "qual":
                levels = vi.ud_info["levels"] if vi.ud_info else sorted(set(str(v) for v in x_vec))
                lmap = {lv: float(k) for k, lv in enumerate(levels)}
                x_od = np.array([lmap.get(str(v), 0.0) for v in x_vec], dtype=float)
            else:
                x_od = x_vec.astype(float)
            res = get_o_dummy_mat(
                x_od,
                breaks=vi.od_info["breaks"],
                dummy_type=vi.od_type,
            )
            mat = res["dummy_mat"]
            if mat is not None:
                parts.append(mat)

        if vi.use_ud:
            res = get_u_dummy_mat(
                x_vec,
                levels=vi.ud_info["levels"],
                drop_last=vi.ud_info["drop_last"],
            )
            mat = res["dummy_mat"]
            if mat is not None:
                parts.append(mat)

        if not parts:
            return None
        return np.hstack(parts)

    def _interaction_block(
        self,
        vi: VarInfo,
        raw_data: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """Build the column block for a pairwise interaction term."""
        vi1 = self.vars_info[vi.var_idx1]
        vi2 = self.vars_info[vi.var_idx2]

        col1 = raw_data.iloc[:, vi1.data_col_idx].values
        col2 = raw_data.iloc[:, vi2.data_col_idx].values

        z1 = self._column_block(col1, vi1, drop_od=True)
        z2 = self._column_block(col2, vi2, drop_od=True)

        if z1 is None or z2 is None:
            return None

        self_inter = vi.var_idx1 == vi.var_idx2
        c1, c2 = z1.shape[1], z2.shape[1]
        cols: List[np.ndarray] = []

        for i in range(c1):
            j_range = range(i + 1, c2) if self_inter else range(c2)
            for j in j_range:
                cols.append((z1[:, i] * z2[:, j]).reshape(-1, 1))

        if not cols:
            return None
        return np.hstack(cols)

    def _var_block_from_df(
        self, vi: VarInfo, raw_data: pd.DataFrame, drop_od: bool = False
    ) -> Optional[np.ndarray]:
        """Dispatch to the correct builder for a variable given a DataFrame."""
        if vi.type == "inter":
            return self._interaction_block(vi, raw_data)
        col = raw_data.iloc[:, vi.data_col_idx].values
        return self._column_block(col, vi, drop_od=drop_od)

    # ------------------------------------------------------------------
    # Public design-matrix builders
    # ------------------------------------------------------------------

    def get_design_matrix(self) -> np.ndarray:
        """Assemble the full augmented design matrix from training data.

        Returns:
            2-D ``np.ndarray`` of shape ``(n_obs, n_expanded_features)``.
        """
        blocks = [
            b for vi in self.vars_info
            if (b := self._var_block_from_df(vi, self.data)) is not None
        ]
        if not blocks:
            raise ValueError("Design matrix is empty — no feature columns were generated.")
        return np.hstack(blocks)

    def transform(self, x_new: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Apply the *training-time* encoding to new data.

        Column order / types in ``x_new`` must match the training ``DataFrame``.

        Args:
            x_new: New feature matrix as a ``DataFrame`` or ``ndarray``.

        Returns:
            2-D ``np.ndarray`` of shape ``(n_new, n_expanded_features)``.
        """
        if not isinstance(x_new, pd.DataFrame):
            x_new = pd.DataFrame(x_new, columns=self.data.columns)

        # Re-index columns to match training layout (in case order differs)
        x_new = x_new.reindex(columns=self.data.columns)

        blocks = [
            b for vi in self.vars_info
            if (b := self._var_block_from_df(vi, x_new)) is not None
        ]
        if not blocks:
            raise ValueError("Transformed design matrix is empty.")
        return np.hstack(blocks)

    # ------------------------------------------------------------------
    # Column-name tracking
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return the names of all expanded design matrix columns.

        Useful for inspecting coefficients.
        """
        names: List[str] = []
        for vi in self.vars_info:
            if vi.type == "inter":
                vi1 = self.vars_info[vi.var_idx1]
                vi2 = self.vars_info[vi.var_idx2]
                z1 = self._column_block(
                    self.data.iloc[:, vi1.data_col_idx].values, vi1, drop_od=True
                )
                z2 = self._column_block(
                    self.data.iloc[:, vi2.data_col_idx].values, vi2, drop_od=True
                )
                if z1 is None or z2 is None:
                    continue
                self_inter = vi.var_idx1 == vi.var_idx2
                for i in range(z1.shape[1]):
                    j_range = (
                        range(i + 1, z2.shape[1]) if self_inter else range(z2.shape[1])
                    )
                    for j in j_range:
                        names.append(f"{vi.name}_{i+1}_{j+1}")
                continue

            col = self.data.iloc[:, vi.data_col_idx].values

            if vi.use_linear:
                names.append(vi.name)

            if vi.use_lv:
                n_lv = len(vi.lv_info["breaks"]) - 2
                names.extend(f"{vi.name}_LV_{k+1}" for k in range(max(n_lv, 0)))

            if vi.use_od:
                n_od = len(vi.od_info["breaks"]) - 1
                names.extend(f"{vi.name}_OD_{k+1}" for k in range(n_od))

            if vi.use_ud:
                lv = vi.ud_info["levels"]
                n_ud = len(lv) - int(vi.ud_info["drop_last"])
                names.extend(f"{vi.name}_UD_{k+1}" for k in range(n_ud))

        return names

    def __repr__(self) -> str:
        n_orig = sum(v.type != "inter" for v in self.vars_info)
        n_inter = sum(v.type == "inter" for v in self.vars_info)
        return (
            f"AGLMInput(n_obs={len(self.data)}, "
            f"n_original_vars={n_orig}, "
            f"n_interaction_terms={n_inter})"
        )


# ---------------------------------------------------------------------------
# Factory: new_input
# ---------------------------------------------------------------------------

def new_input(
    x: Union[pd.DataFrame, np.ndarray],
    qualitative_vars_ud_only: Optional[Union[List[int], List[str]]] = None,
    qualitative_vars_both: Optional[Union[List[int], List[str]]] = None,
    qualitative_vars_od_only: Optional[Union[List[int], List[str]]] = None,
    quantitative_vars: Optional[Union[List[int], List[str]]] = None,
    use_lvar: bool = False,
    extrapolation: str = "default",
    add_linear_columns: bool = True,
    add_od_columns_of_qualitatives: bool = True,
    add_interaction_columns: bool = False,
    od_type_of_quantitatives: str = "C",
    nbin_max: Optional[int] = None,
    bins_list: Optional[List[np.ndarray]] = None,
    bins_names: Optional[Union[List[str], List[int]]] = None,
) -> AGLMInput:
    """Create an :class:`AGLMInput` from a DataFrame or array.

    This is the Python equivalent of ``newInput()`` in R/aglm-input.R.

    Column type auto-detection:
      * **Numeric** dtypes  → quantitative.
      * **Object** / **bool** / **category** (unordered) → qualitative (U-dummies).
      * **Ordered CategoricalDtype** → qualitative with both U- and O-dummies.

    Args:
        x:                             Feature matrix (``DataFrame`` or ``ndarray``).
        qualitative_vars_ud_only:      Columns to treat as *unordered* categorical
                                       (U-dummies only).  List of int indices or str names.
        qualitative_vars_both:         Columns treated as ordered categorical (U + O dummies).
        qualitative_vars_od_only:      Columns for O-dummies only.
        quantitative_vars:             Columns to treat as numeric (overrides auto-detection).
        use_lvar:                      Use L-variables instead of O-dummies for numeric cols.
        extrapolation:                 ``"default"`` or ``"flat"`` (clamp to training range).
        add_linear_columns:            Include the raw linear term for numeric columns.
        add_od_columns_of_qualitatives: Include O-dummies (type ``"J"``) for ordered cols.
        add_interaction_columns:       Add all pairwise interaction columns.
        od_type_of_quantitatives:      ``"C"`` (ramp), ``"J"`` (step), or ``"N"`` (none).
        nbin_max:                      Max bins for numeric discretisation.
        bins_list:                     Pre-computed breakpoints, one array per OD/LV variable.
        bins_names:                    Names / indices matching ``bins_list``.

    Returns:
        Constructed :class:`AGLMInput` ready for ``get_design_matrix()`` or ``transform()``.
    """
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    else:
        x = x.copy()

    nvar = x.shape[1]
    col_names = list(x.columns)
    _nbin_max = nbin_max if nbin_max is not None else 100

    # ------------------------------------------------------------------
    # Helper: resolve column specs → set of 0-based indices
    # ------------------------------------------------------------------
    def resolve(spec) -> set:
        if spec is None:
            return set()
        if all(isinstance(s, (int, np.integer)) for s in spec):
            return set(int(s) for s in spec)
        if all(isinstance(s, str) for s in spec):
            return {col_names.index(s) for s in spec}
        raise ValueError("Column specs must be all-int or all-str.")

    set_ud_only = resolve(qualitative_vars_ud_only)
    set_od_only = resolve(qualitative_vars_od_only)
    set_both    = resolve(qualitative_vars_both)
    set_quan    = resolve(quantitative_vars)

    # ------------------------------------------------------------------
    # Build initial VarInfo list from dtype inspection
    # ------------------------------------------------------------------
    vars_info: List[VarInfo] = []

    for i in range(nvar):
        col = x.iloc[:, i]
        dtype = col.dtype

        is_ordered_cat = (
            isinstance(dtype, pd.CategoricalDtype) and dtype.ordered
        )
        # pandas 4+ uses StringDtype for string columns (not object dtype).
        # Treat anything that is NOT purely numeric as qualitative.
        is_qualitative = not pd.api.types.is_numeric_dtype(dtype)

        vtype = "qual" if is_qualitative else "quan"

        vi = VarInfo(
            idx=i,
            name=col_names[i],
            data_col_idx=i,
            type=vtype,
            extrapolation=extrapolation,
        )

        if vtype == "quan":
            vi.use_linear = add_linear_columns
            vi.use_ud = False
            vi.use_od = (od_type_of_quantitatives != "N") and not use_lvar
            vi.use_lv = use_lvar
            vi.od_type = od_type_of_quantitatives

        else:  # qualitative
            vi.use_linear = False
            vi.use_ud = True
            vi.use_od = is_ordered_cat and add_od_columns_of_qualitatives
            vi.od_type = "J"
            vi.use_lv = False

        vars_info.append(vi)

    # ------------------------------------------------------------------
    # Override from explicit column spec arguments
    # ------------------------------------------------------------------
    for i in set_ud_only:
        vi = vars_info[i]
        vi.type = "qual"; vi.use_linear = False
        vi.use_ud = True; vi.use_od = False; vi.use_lv = False

    for i in set_od_only:
        vi = vars_info[i]
        vi.type = "qual"; vi.use_linear = False
        vi.use_ud = False; vi.use_od = True; vi.use_lv = False; vi.od_type = "J"

    for i in set_both:
        vi = vars_info[i]
        vi.type = "qual"; vi.use_linear = False
        vi.use_ud = True; vi.use_od = True; vi.use_lv = False; vi.od_type = "J"

    for i in set_quan:
        vi = vars_info[i]
        vi.type = "quan"
        vi.use_linear = add_linear_columns
        vi.use_ud = False
        vi.use_od = (od_type_of_quantitatives != "N") and not use_lvar
        vi.use_lv = use_lvar
        vi.od_type = od_type_of_quantitatives

    # ------------------------------------------------------------------
    # Apply custom breakpoints from bins_list
    # ------------------------------------------------------------------
    if bins_list is not None:
        od_lv_indices = [vi.idx for vi in vars_info if vi.use_od or vi.use_lv]

        if bins_names is None:
            for k, vi_idx in enumerate(od_lv_indices[: len(bins_list)]):
                brks = np.sort(np.unique(np.asarray(bins_list[k], dtype=float)))
                vi = vars_info[vi_idx]
                if vi.use_od:
                    vi.od_info = {"breaks": brks}
                if vi.use_lv:
                    vi.lv_info = {"breaks": brks}
        else:
            name_map = {col_names[vi.idx]: vi.idx for vi in vars_info}
            idx_map  = {vi.idx: vi.idx for vi in vars_info}
            for k, bn in enumerate(bins_names):
                vi_idx = name_map.get(str(bn), idx_map.get(int(bn) if str(bn).isdigit() else -1))
                if vi_idx is None:
                    continue
                brks = np.sort(np.unique(np.asarray(bins_list[k], dtype=float)))
                vi = vars_info[vi_idx]
                if vi.use_od:
                    vi.od_info = {"breaks": brks}
                if vi.use_lv:
                    vi.lv_info = {"breaks": brks}

    # ------------------------------------------------------------------
    # Infer remaining info from training data
    # ------------------------------------------------------------------
    for vi in vars_info:
        col_vals = x.iloc[:, vi.data_col_idx].values

        if vi.use_ud and vi.ud_info is None:
            vi.ud_info = get_u_dummy_mat(col_vals, only_info=True, drop_last=False)

        if vi.use_od and vi.od_info is None:
            if vi.type == "qual":
                levels = vi.ud_info["levels"] if vi.ud_info else sorted(
                    set(str(v) for v in col_vals)
                )
                lmap = {lv: float(k) for k, lv in enumerate(levels)}
                col_encoded = np.array([lmap.get(str(v), 0.0) for v in col_vals], dtype=float)
            else:
                col_encoded = col_vals.astype(float)
            vi.od_info = get_o_dummy_mat(
                col_encoded, nbin_max=_nbin_max, dummy_type=vi.od_type, only_info=True,
            )
        elif vi.use_od and vi.od_info is not None:
            vi.od_info.setdefault("dummy_type", vi.od_type)

        if vi.use_lv and vi.lv_info is None:
            vi.lv_info = get_lvar_mat(
                col_vals.astype(float),
                nbin_max=_nbin_max,
                only_info=True,
            )

    # ------------------------------------------------------------------
    # Add pairwise interaction terms
    # ------------------------------------------------------------------
    if add_interaction_columns and nvar > 1:
        idx_counter = len(vars_info)
        for i in range(nvar):
            for j in range(i, nvar):
                vi_inter = VarInfo(
                    idx=idx_counter,
                    name=f"{col_names[i]}_x_{col_names[j]}",
                    data_col_idx=-1,
                    type="inter",
                    var_idx1=i,
                    var_idx2=j,
                )
                vars_info.append(vi_inter)
                idx_counter += 1

    return AGLMInput(data=x, vars_info=vars_info)
