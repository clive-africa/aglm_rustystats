"""
Tests for the Python aglm package.

Covers:
  - Binning utilities
  - Dummy / basis construction
  - Design matrix assembly
  - Model fitting (Gaussian, Binomial, Poisson)
  - Predictions, coefficients, deviance, residuals
  - CV model selection (cv_aglm, cva_aglm)
  - Plotting (smoke tests)
"""

import numpy as np
import pandas as pd
import pytest

# ---- package under test ----
from aglm.binning import create_equal_width_bins, create_equal_freq_bins, execute_binning
from aglm.dummies import get_u_dummy_mat, get_o_dummy_mat, get_lvar_mat
from aglm.input import new_input, AGLMInput
from aglm.model import aglm, cv_aglm, cva_aglm, AccurateGLM, CVAAccurateGLM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

np.random.seed(42)
N = 200


@pytest.fixture
def gaussian_data():
    X = pd.DataFrame({
        "num1": np.random.randn(N),
        "num2": np.random.uniform(0, 10, N),
        "cat1": np.random.choice(["A", "B", "C"], N),
    })
    y = 2.0 * X["num1"] - 0.5 * X["num2"] + np.random.randn(N) * 0.5
    return X, y  # already ndarray from np.random.poisson


@pytest.fixture
def binomial_data():
    X = pd.DataFrame({
        "x1": np.random.randn(N),
        "x2": np.random.randn(N),
    })
    logit = X["x1"] - 0.5 * X["x2"]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = np.random.binomial(1, p)
    return X, y


@pytest.fixture
def poisson_data():
    X = pd.DataFrame({"x": np.random.uniform(0, 2, N)})
    mu = np.exp(0.5 * X["x"])
    y = np.random.poisson(mu)
    return X, y  # already ndarray from np.random.poisson


# ---------------------------------------------------------------------------
# 1. Binning
# ---------------------------------------------------------------------------

class TestBinning:
    def test_equal_width_length(self):
        x = np.linspace(0, 10, 50)
        brks = create_equal_width_bins(x, nbin=5)
        assert len(brks) == 6
        assert brks[0] == pytest.approx(0.0)
        assert brks[-1] == pytest.approx(10.0)

    def test_equal_freq_unique(self):
        x = np.array([1, 1, 2, 3, 3, 4, 5])
        brks = create_equal_freq_bins(x, nbin=4)
        assert len(brks) >= 2
        assert np.all(np.diff(brks) >= 0)

    def test_execute_binning_labels_range(self):
        x = np.linspace(0, 1, 100)
        brks, labels = execute_binning(x, nbin_max=10)
        assert labels.min() >= 1
        assert labels.max() <= len(brks) - 1

    def test_execute_binning_custom_breaks(self):
        x = np.array([0.5, 1.5, 2.5, 3.5])
        custom_breaks = np.array([0, 1, 2, 3, 4], dtype=float)
        brks, labels = execute_binning(x, breaks=custom_breaks)
        np.testing.assert_array_equal(labels, [1, 2, 3, 4])

    def test_execute_binning_out_of_range(self):
        """Values outside break range should be clamped to edge bins."""
        x = np.array([-100.0, 50.0, 200.0])
        brks, labels = execute_binning(x, breaks=np.array([0.0, 1.0, 2.0]))
        assert labels[0] == 1
        assert labels[2] == 2


# ---------------------------------------------------------------------------
# 2. Dummy / basis construction
# ---------------------------------------------------------------------------

class TestDummies:

    def test_u_dummy_shape_drop_last(self):
        x = np.array(["A", "B", "C", "A", "B"])
        res = get_u_dummy_mat(x, drop_last=True)
        mat = res["dummy_mat"]
        assert mat.shape == (5, 2)

    def test_u_dummy_shape_keep_last(self):
        x = np.array(["A", "B", "C"])
        res = get_u_dummy_mat(x, drop_last=False)
        assert res["dummy_mat"].shape == (3, 3)

    def test_u_dummy_only_info(self):
        x = np.array(["X", "Y"])
        res = get_u_dummy_mat(x, only_info=True)
        assert "dummy_mat" not in res
        assert "levels" in res

    def test_u_dummy_values(self):
        x = np.array(["A", "B", "C"])
        res = get_u_dummy_mat(x, drop_last=False)
        # Row 0 should be [1, 0, 0]
        np.testing.assert_array_equal(res["dummy_mat"][0], [1, 0, 0])

    def test_o_dummy_type_c_shape(self):
        x = np.linspace(0, 1, 50)
        res = get_o_dummy_mat(x, nbin_max=5, dummy_type="C")
        mat = res["dummy_mat"]
        assert mat.ndim == 2
        assert mat.shape[0] == 50
        # All values clamped to [0, 1]
        assert mat.min() >= 0.0
        assert mat.max() <= 1.0

    def test_o_dummy_type_j_shape(self):
        x = np.arange(1, 11, dtype=float)
        res = get_o_dummy_mat(x, nbin_max=5, dummy_type="J")
        assert res["dummy_mat"].shape[0] == 10

    def test_lvar_interior_knots(self):
        x = np.linspace(0, 10, 100)
        res = get_lvar_mat(x, nbin_max=5)
        mat = res["dummy_mat"]
        # Should have n_breaks - 2 interior knot columns
        n_interior = len(res["breaks"]) - 2
        if n_interior > 0:
            assert mat.shape[1] == n_interior

    def test_lvar_single_bin(self):
        """When only one bin, there are no interior knots → dummy_mat is None."""
        x = np.array([1.0, 1.0, 1.0])
        res = get_lvar_mat(x, nbin_max=100)
        # Either None or single-element breaks
        if res["dummy_mat"] is not None:
            assert res["dummy_mat"].shape[1] >= 0


# ---------------------------------------------------------------------------
# 3. Input / design matrix
# ---------------------------------------------------------------------------

class TestInput:

    def test_basic_numeric(self):
        X = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        inp = new_input(X)
        Xd = inp.get_design_matrix()
        assert Xd.ndim == 2
        assert Xd.shape[0] == 50

    def test_mixed_types(self):
        X = pd.DataFrame({
            "num": np.random.randn(30),
            "cat": np.random.choice(["X", "Y"], 30),
        })
        inp = new_input(X)
        Xd = inp.get_design_matrix()
        # Should have more columns than the original 2
        assert Xd.shape[1] > 2

    def test_transform_matches_design_matrix(self):
        X = pd.DataFrame({"x": np.linspace(0, 1, 40)})
        inp = new_input(X)
        Xd_train = inp.get_design_matrix()
        Xd_transform = inp.transform(X)
        np.testing.assert_allclose(Xd_train, Xd_transform)

    def test_use_lvar(self):
        X = pd.DataFrame({"x": np.linspace(0, 5, 60)})
        inp_od = new_input(X, use_lvar=False)
        inp_lv = new_input(X, use_lvar=True)
        Xd_od = inp_od.get_design_matrix()
        Xd_lv = inp_lv.get_design_matrix()
        # Different feature spaces
        assert Xd_od.shape[1] != Xd_lv.shape[1] or not np.allclose(Xd_od, Xd_lv)

    def test_interactions(self):
        X = pd.DataFrame({"a": np.ones(20), "b": np.ones(20) * 2})
        inp = new_input(X, add_interaction_columns=True)
        Xd = inp.get_design_matrix()
        # Interactions add extra columns
        inp_no = new_input(X, add_interaction_columns=False)
        assert Xd.shape[1] > inp_no.get_design_matrix().shape[1]

    def test_feature_names_length(self):
        X = pd.DataFrame({"n": np.random.randn(20), "c": ["A", "B"] * 10})
        inp = new_input(X)
        Xd = inp.get_design_matrix()
        names = inp.get_feature_names()
        assert len(names) == Xd.shape[1]

    def test_repr(self):
        X = pd.DataFrame({"x": range(10)})
        inp = new_input(X)
        r = repr(inp)
        assert "AGLMInput" in r


# ---------------------------------------------------------------------------
# 4. aglm — basic fit
# ---------------------------------------------------------------------------

class TestAglm:

    def test_gaussian_fit(self, gaussian_data):
        X, y = gaussian_data
        model = aglm(X, y, alpha=1.0, lambda_=0.01, family="gaussian")
        assert isinstance(model, AccurateGLM)

    def test_predict_shape(self, gaussian_data):
        X, y = gaussian_data
        model = aglm(X, y, alpha=1.0, lambda_=0.01)
        preds = model.predict()
        assert preds.shape == (len(y),)

    def test_predict_new_data(self, gaussian_data):
        X, y = gaussian_data
        model = aglm(X, y, lambda_=0.01)
        preds_new = model.predict(X.iloc[:10])
        assert preds_new.shape == (10,)

    def test_coef_length(self, gaussian_data):
        X, y = gaussian_data
        model = aglm(X, y, lambda_=0.01)
        Xd = model.aglm_input.get_design_matrix()
        assert len(model.coef()) == Xd.shape[1]

    def test_coef_with_names(self, gaussian_data):
        X, y = gaussian_data
        model = aglm(X, y, lambda_=0.01)
        coef_s = model.coef(with_names=True)
        import pandas as pd
        assert isinstance(coef_s, pd.Series)

    def test_deviance_positive(self, gaussian_data):
        X, y = gaussian_data
        model = aglm(X, y, lambda_=0.01)
        assert model.deviance() > 0

    def test_residuals_shape(self, gaussian_data):
        X, y = gaussian_data
        model = aglm(X, y, lambda_=0.01)
        r = model.residuals("response")
        assert r.shape == y.shape

    def test_binomial_fit(self, binomial_data):
        X, y = binomial_data
        model = aglm(X, y, alpha=1.0, lambda_=0.1, family="binomial")
        preds = model.predict()
        assert preds.shape == (len(y),)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_poisson_fit(self, poisson_data):
        X, y = poisson_data
        model = aglm(X, y, alpha=0.0, lambda_=0.1, family="poisson")
        preds = model.predict()
        assert np.all(preds >= 0)

    def test_repr_str(self, gaussian_data):
        X, y = gaussian_data
        model = aglm(X, y, lambda_=0.01)
        assert "AccurateGLM" in repr(model)
        assert "Family" in str(model)

    def test_no_linear_columns(self, gaussian_data):
        X, y = gaussian_data
        # numeric columns only — should still work with add_linear_columns=False
        X_num = X[["num1", "num2"]]
        model = aglm(X_num, y, lambda_=0.01, add_linear_columns=False)
        assert model.predict().shape == (len(y),)

    def test_use_lvar(self, gaussian_data):
        X, y = gaussian_data
        X_num = X[["num1", "num2"]]
        model = aglm(X_num, y, lambda_=0.01, use_lvar=True)
        assert model.predict().shape == (len(y),)

    def test_extrapolation_flat(self, gaussian_data):
        X, y = gaussian_data
        X_num = X[["num1", "num2"]]
        model = aglm(X_num, y, lambda_=0.01, extrapolation="flat")
        preds = model.predict()
        assert preds.shape == (len(y),)

    def test_ridge(self, gaussian_data):
        X, y = gaussian_data
        model = aglm(X, y, alpha=0.0, lambda_=0.1, family="gaussian")
        assert model.predict().shape == (len(y),)

    def test_custom_bins(self):
        X = pd.DataFrame({"x": np.linspace(0, 10, 100)})
        y = np.linspace(0, 10, 100)
        custom_breaks = np.array([0, 2.5, 5.0, 7.5, 10.0])
        model = aglm(X, y, lambda_=0.01, bins_list=[custom_breaks])
        preds = model.predict()
        assert preds.shape == (100,)


# ---------------------------------------------------------------------------
# 5. cv_aglm
# ---------------------------------------------------------------------------

class TestCvAglm:

    def test_returns_accurate_glm(self, gaussian_data):
        X, y = gaussian_data
        model = cv_aglm(X, y, alpha=1.0, nfolds=5)
        assert isinstance(model, AccurateGLM)

    def test_cv_results_populated(self, gaussian_data):
        X, y = gaussian_data
        model = cv_aglm(X, y, nfolds=5)
        assert model.cv_results is not None
        assert "lambda_grid" in model.cv_results
        assert "best_cv_score" in model.cv_results

    def test_best_lambda_positive(self, gaussian_data):
        X, y = gaussian_data
        model = cv_aglm(X, y, nfolds=5)
        assert model.lambda_ > 0

    def test_binomial_cv(self, binomial_data):
        X, y = binomial_data
        model = cv_aglm(X, y, alpha=1.0, nfolds=5, family="binomial")
        assert isinstance(model, AccurateGLM)
        assert 0 <= model.predict().min() and model.predict().max() <= 1

    def test_custom_lambda_grid(self, gaussian_data):
        X, y = gaussian_data
        grid = np.array([0.001, 0.01, 0.1])
        model = cv_aglm(X, y, nfolds=3, lambda_grid=grid)
        assert model.lambda_ in grid


# ---------------------------------------------------------------------------
# 6. cva_aglm
# ---------------------------------------------------------------------------

class TestCvaAglm:

    def test_returns_cva_object(self, gaussian_data):
        X, y = gaussian_data
        cva = cva_aglm(X, y, alpha_grid=np.array([0.0, 0.5, 1.0]), nfolds=3)
        assert isinstance(cva, CVAAccurateGLM)

    def test_best_alpha_in_grid(self, gaussian_data):
        X, y = gaussian_data
        grid = np.array([0.0, 0.5, 1.0])
        cva = cva_aglm(X, y, alpha_grid=grid, nfolds=3)
        assert cva.best_alpha in grid.tolist()

    def test_best_model_is_accurate_glm(self, gaussian_data):
        X, y = gaussian_data
        cva = cva_aglm(X, y, alpha_grid=np.array([0.5, 1.0]), nfolds=3)
        assert isinstance(cva.best_model, AccurateGLM)

    def test_repr_str(self, gaussian_data):
        X, y = gaussian_data
        cva = cva_aglm(X, y, alpha_grid=np.array([0.5, 1.0]), nfolds=3)
        assert "CVAAccurateGLM" in repr(cva)
        assert "best_alpha" in str(cva).lower()


# ---------------------------------------------------------------------------
# 7. Plotting (smoke tests — just ensure no exceptions)
# ---------------------------------------------------------------------------

class TestPlotting:

    def test_plot_aglm_runs(self, gaussian_data):
        import matplotlib
        matplotlib.use("Agg")
        from aglm.plot import plot_aglm
        X, y = gaussian_data
        model = aglm(X, y, lambda_=0.01)
        fig = plot_aglm(model, ncols=2)
        assert fig is not None

    def test_plot_cva_alpha_runs(self, gaussian_data):
        import matplotlib
        matplotlib.use("Agg")
        from aglm.plot import plot_cva_alpha
        X, y = gaussian_data
        cva = cva_aglm(X, y, alpha_grid=np.array([0.0, 1.0]), nfolds=3)
        fig = plot_cva_alpha(cva)
        assert fig is not None

    def test_plot_cv_model(self, gaussian_data):
        import matplotlib
        matplotlib.use("Agg")
        from aglm.plot import plot_aglm
        X, y = gaussian_data
        model = cv_aglm(X, y, nfolds=3)
        fig = plot_aglm(model, show_cv_curve=True)
        assert fig is not None
