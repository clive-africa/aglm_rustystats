# aglm — Accurate Generalized Linear Model (Python)

Python port of the R package [`aglm`](https://github.com/kkondo1981/aglm) by
Kenji Kondo, Kazuhisa Takahashi, and Hikari Banno.

> **Original paper (Hachemeister Prize 2021):**  
> Fujita, Tanaka, Kondo & Iwasawa (2020).  
> *AGLM: A Hybrid Modeling Method of GLM and Data Science Techniques.*  
> Actuarial Colloquium Paris 2020.  
> https://www.institutdesactuaires.com/global/gene/link.php?doc_id=16273&fg=1

---

## What is AGLM?

AGLM is a **regularised GLM** (elastic-net) that automatically enriches the
feature space before fitting, enabling non-linear and interaction effects while
remaining fully interpretable.  It is particularly popular in actuarial modelling
where explainability is a regulatory requirement.

The feature-engineering pipeline adds three types of auxiliary columns:

| Type | Variable kind | Description |
|------|--------------|-------------|
| **U-dummy** | Unordered categorical | Standard one-hot encoding (reference-cell) |
| **O-dummy** | Numeric / ordered categorical | Piecewise-linear ramp (`"C"`) or step (`"J"`) basis |
| **L-variable** | Numeric | Absolute-value spline: `\|x − knot_k\|` — better extrapolation |

These are assembled into an augmented design matrix `X̃` which is passed to
scikit-learn's elastic-net backend (Gaussian / Binomial / Poisson families).

---

## Installation

```bash
pip install -e .        # editable install from source
# or
pip install aglm        # once published to PyPI
```

**Dependencies:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`

---

## Quick-start

```python
import pandas as pd
import numpy as np
from aglm import cv_aglm, plot_aglm

# --- 1. Prepare data -------------------------------------------------------
df = pd.read_csv("motor_claims.csv")
X = df[["age", "vehicle_age", "region", "bonus_malus"]]
y = df["claim_frequency"].values

# --- 2. Fit — lambda chosen by 10-fold CV (LASSO default) ------------------
model = cv_aglm(X, y, alpha=1.0, nfolds=10, family="poisson")
print(model)
# AccurateGLM(family='poisson', alpha=1.0, lambda=0.00312, n_coef=47)

# --- 3. Predict ------------------------------------------------------------
preds = model.predict(X_new)

# --- 4. Interpret via contribution plot ------------------------------------
fig = plot_aglm(model)
fig.savefig("contributions.png", dpi=150, bbox_inches="tight")
```

---

## API

### Fitting functions

| Python | R equivalent | Description |
|--------|-------------|-------------|
| `aglm(x, y, alpha, lambda_)` | `aglm(x, y, alpha, ...)` | Fit at given α and λ |
| `cv_aglm(x, y, alpha, nfolds)` | `cv.aglm(x, y, alpha, ...)` | Fix α; CV for λ |
| `cva_aglm(x, y, alpha_grid, nfolds)` | `cva.aglm(x, y, ...)` | CV for both α and λ |

### Model methods

```python
model.predict(x_new)              # predictions (response scale)
model.predict(x_new, predict_type="link")  # linear predictor
model.coef()                      # coefficient vector (np.ndarray)
model.coef(with_names=True)       # coefficient pd.Series
model.intercept()                 # intercept scalar
model.deviance()                  # model deviance
model.residuals("response")       # response / pearson residuals
```

### Plotting

```python
from aglm import plot_aglm, plot_cva_alpha

plot_aglm(model)                  # per-variable contribution plot
plot_cva_alpha(cva_model)         # CV score vs alpha bar chart
```

### Utility functions

```python
from aglm import (
    create_equal_width_bins,
    create_equal_freq_bins,
    execute_binning,
    get_u_dummy_mat,
    get_o_dummy_mat,
    get_lvar_mat,
)
```

---

## Key parameters

### `aglm` / `cv_aglm` / `cva_aglm`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `1.0` | Elastic-net mix: `1.0` = LASSO, `0.0` = Ridge |
| `lambda_` | `1e-3` | Regularisation strength (`aglm` only) |
| `family` | `"gaussian"` | `"gaussian"` / `"binomial"` / `"poisson"` |
| `nfolds` | `10` | CV folds (`cv_aglm`, `cva_aglm`) |
| `alpha_grid` | `[0, 0.1, …, 1.0]` | α values to search (`cva_aglm`) |
| `use_lvar` | `False` | L-variables instead of O-dummies for numeric cols |
| `extrapolation` | `"default"` | `"flat"` clamps predictions to training range |
| `add_linear_columns` | `True` | Include raw linear term for numeric variables |
| `add_interaction_columns` | `False` | Add all pairwise interaction terms |
| `nbin_max` | `100` | Max bins for numeric discretisation |
| `qualitative_vars_ud_only` | `None` | Override cols → unordered categorical |
| `qualitative_vars_both` | `None` | Override cols → ordered categorical |
| `quantitative_vars` | `None` | Override cols → numeric |
| `bins_list` | `None` | Pre-computed breakpoints per numeric variable |

---

## Column type auto-detection

| pandas dtype | Treatment |
|-------------|-----------|
| `float64`, `int64`, numeric | Quantitative: linear + O-dummies |
| `object`, `bool` | Qualitative (unordered): U-dummies only |
| `CategoricalDtype(ordered=False)` | Qualitative (unordered): U-dummies |
| `CategoricalDtype(ordered=True)` | Qualitative (ordered): U- + O-dummies (type J) |

Use `qualitative_vars_ud_only`, `qualitative_vars_both`, `quantitative_vars`
to override auto-detection.

---

## Regularisation mapping

The R package uses glmnet's convention:

```
λ · [ (1−α)/2 · ‖β‖₂²  +  α · ‖β‖₁ ]
```

This is mapped to rustystats.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Running examples

```bash
python examples/aglm_examples.py
```

This runs six examples covering Gaussian, Binomial, Poisson regression,
CV model selection, L-variables, and categorical variables.

---

## Differences from the R package

#TODO
---

## License

GPL-2.0 — matching the original R package.

---

## Citation

If you use this package, please cite the original paper:

```bibtex
@inproceedings{fujita2020aglm,
  title={AGLM: A Hybrid Modeling Method of GLM and Data Science Techniques},
  author={Fujita, Suguru and Tanaka, Toyoto and Kondo, Kenji and Iwasawa, Hirokazu},
  booktitle={Actuarial Colloquium Paris 2020},
  year={2020}
}
```
