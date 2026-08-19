"""
benchmark_report.py
===================
Generates a self-contained, interactive 7-tab HTML report from the results
returned by model_benchmark.main().

Tabs
----
1. Data Overview      – summary cards, feature distributions with frequency overlay
2. Model Leaderboard  – sortable metric table, calibration, training times
3. One-Way Analysis   – actual vs predicted by feature bin (feature selector)
4. Lift & Gini        – Lorenz curves, Gini coefficients, double-lift chart
5. SHAP               – feature importance + beeswarm (tree models, 20 % of data)
6. Underperformance   – A/P by decile, feature residuals, 2-D A/P heatmap
7. Profit Matrix      – N×N competition grid; train/test toggle; tournament bar

Usage
-----
    from benchmark_report import BenchmarkReport

    report = BenchmarkReport(
        results=results,          # dict from model_benchmark.main()
        train=train,
        test=test,
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        response_col=RESPONSE_COL,
        exposure_col=EXPOSURE_COL,
        task=TASK,
        family=TASK_FAMILY[TASK],
        title="FreMTPL2 Frequency Benchmark",
    )
    report.generate(
        output_path="benchmark_report.html",
        model_objects=model_objects,   # dict name -> fit result dict (for SHAP)
        train_preds=train_predictions, # dict name -> count array on train
    )
"""

from __future__ import annotations

import json
import pathlib
import datetime
import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  BenchmarkReport
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkReport:
    # ── tuneable constants ───────────────────────────────────────────────────
    SHAP_SAMPLE_FRAC = 0.20   # fraction of test set used for SHAP
    ONEWAY_BINS      = 10     # quantile bins for numeric one-way
    HEATMAP_BINS     = 8      # max bins per axis for the 2-D A/P heatmap
    MIN_EXP_FRAC     = 0.01   # cells with < 1 % exposure → grey in heatmap

    TREE_MODELS = {"GBM", "CatBoost", "XGBoost", "XGBoost-W"}

    COLORS = {
        "GLM":        "#6C8EAD",
        "RegGLM":     "#3A7CA5",
        "AGLM-Lin":   "#A8C0D6",
        "AGLM-Lvar":  "#1A3C6E",
        "GAM":        "#5C8A5C",
        "GBM":        "#C0392B",
        "CatBoost":   "#8E44AD",
        "XGBoost":    "#E74C3C",
        "XGBoost-W":  "#FF6B35",
        "DerivLasso": "#E67E22",
    }
    DEFAULT_COLOR = "#888888"

    # ── constructor ──────────────────────────────────────────────────────────

    def __init__(
        self,
        results:          dict,
        train:            pd.DataFrame,
        test:             pd.DataFrame,
        numeric_cols:     list[str],
        categorical_cols: list[str],
        response_col:     str,
        exposure_col:     str,
        task:             str,
        family:           str,
        title:            str = "Model Benchmark",
    ):
        self.results          = results
        self.train            = train.reset_index(drop=True)
        self.test             = test.reset_index(drop=True)
        self.numeric_cols     = numeric_cols
        self.categorical_cols = categorical_cols
        self.response_col     = response_col
        self.exposure_col     = exposure_col
        self.task             = task
        self.family           = family
        self.title            = title

        self.models       = results.get("models", [])
        self.predictions  = results.get("predictions", {})     # name → count array (test)
        self.metrics_table = results.get("metrics_table", pd.DataFrame())
        self.train_times  = results.get("train_times", {})

    # ── helpers ───────────────────────────────────────────────────────────────

    def _act_freq(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.response_col].values / np.maximum(df[self.exposure_col].values, 1e-9)

    def _color(self, name: str) -> str:
        return self.COLORS.get(name, self.DEFAULT_COLOR)

    def _bin_numeric(self, series: pd.Series, n: int | None = None) -> pd.Categorical:
        n = n or self.ONEWAY_BINS
        try:
            return pd.qcut(series, q=n, duplicates="drop")
        except ValueError:
            return pd.cut(series, bins=max(2, n), duplicates="drop")

    # ── Tab 1: Overview ───────────────────────────────────────────────────────

    def _compute_overview(self) -> dict:
        test = self.test
        exp  = test[self.exposure_col].values
        af   = self._act_freq(test)

        best = str(self.metrics_table.iloc[0]["Model"]) if len(self.metrics_table) else "—"
        summary = {
            "n_train":    int(len(self.train)),
            "n_test":     int(len(test)),
            "mean_actual": float(np.average(af, weights=exp)),
            "n_models":   int(len(self.models)),
            "best_model": best,
            "task":       self.task,
            "family":     self.family,
        }

        def _num_feature(col: str) -> dict:
            bins = self._bin_numeric(test[col], n=20)
            gb   = test.groupby(bins, observed=True)
            keys = list(gb.groups.keys())
            labels, exposures, freqs = [], [], []
            for b in keys:
                g = gb.get_group(b)
                e = float(g[self.exposure_col].sum())
                labels.append(str(b))
                exposures.append(e)
                freqs.append(float(g[self.response_col].sum() / e) if e > 0 else None)
            return {"labels": labels, "exposure": exposures, "freq": freqs}

        def _cat_feature(col: str) -> dict:
            s  = test[col].astype(str)
            gb = test.groupby(s, observed=True)
            exp_by = gb[self.exposure_col].sum().sort_values(ascending=False)
            levels = list(exp_by.index[:20])
            exposures, freqs = [], []
            for lv in levels:
                g = gb.get_group(lv)
                e = float(g[self.exposure_col].sum())
                exposures.append(e)
                freqs.append(float(g[self.response_col].sum() / e) if e > 0 else None)
            return {"levels": levels, "exposure": exposures, "freq": freqs}

        return {
            "summary":     summary,
            "numeric":     {c: _num_feature(c) for c in self.numeric_cols},
            "categorical": {c: _cat_feature(c) for c in self.categorical_cols},
        }

    # ── Tab 2: Leaderboard ────────────────────────────────────────────────────

    def _compute_leaderboard(self) -> dict:
        mt = self.metrics_table
        dev_label = "Poisson Dev." if self.task == "frequency" else "Gamma Dev."
        models, metrics_js = [], {}
        for _, row in mt.iterrows():
            m = str(row["Model"])
            models.append(m)
            metrics_js[m] = {
                col: (None if (v := row.get(col, float("nan"))) is None
                      or (isinstance(v, float) and np.isnan(v)) else float(v))
                for col in ["Deviance", "MSE", "MAE", "RMSE", "AUC", "Gini",
                            "Avg Pred", "Avg Actual"]
            }
        avg_actual = float(mt["Avg Actual"].iloc[0]) if len(mt) else 0.0
        return {
            "models":      models,
            "metrics":     metrics_js,
            "times":       {m: float(self.train_times.get(m, 0)) for m in models},
            "avg_actual":  avg_actual,
            "dev_label":   dev_label,
            "lower_better": {
                "Deviance": True, "MSE": True, "MAE": True, "RMSE": True,
                "AUC": False, "Gini": False,
            },
        }

    # ── Tab 3: One-Way ────────────────────────────────────────────────────────

    def _compute_oneway(self) -> dict:
        test = self.test
        exp  = test[self.exposure_col].values
        act  = test[self.response_col].values.astype(float)
        result = {}

        for col in self.numeric_cols + self.categorical_cols:
            is_num = col in self.numeric_cols
            bin_col = (self._bin_numeric(test[col]).astype(str) if is_num
                       else test[col].astype(str))

            groups: dict[str, dict] = {}
            for i, b in enumerate(bin_col):
                g = groups.setdefault(b, {"exp": 0.0, "claims": 0.0,
                                           "preds": {m: 0.0 for m in self.models}})
                g["exp"]    += exp[i]
                g["claims"] += act[i]
                for m in self.models:
                    if m in self.predictions:
                        g["preds"][m] += self.predictions[m][i]

            if is_num:
                skeys = sorted(groups)
            else:
                skeys = sorted(groups, key=lambda k: -groups[k]["exp"])

            result[col] = {
                "bins":      skeys,
                "exposure":  [groups[k]["exp"] for k in skeys],
                "actual":    [groups[k]["claims"] / max(groups[k]["exp"], 1e-9) for k in skeys],
                "models":    {
                    m: [groups[k]["preds"][m] / max(groups[k]["exp"], 1e-9) for k in skeys]
                    for m in self.models if m in self.predictions
                },
                "is_numeric": is_num,
            }
        return result

    # ── Tab 4: Lift & Gini ────────────────────────────────────────────────────

    def _compute_lift_gini(self) -> dict:
        test = self.test
        act  = test[self.response_col].values.astype(float)
        exp  = test[self.exposure_col].values

        lorenz, gini, lift_decile = {}, {}, {}

        for m in self.models:
            if m not in self.predictions:
                continue
            pred      = np.asarray(self.predictions[m], dtype=float)
            pred_freq = pred / np.maximum(exp, 1e-9)

            # Lorenz curve (sorted by predicted frequency ascending)
            order     = np.argsort(pred_freq)
            cum_exp   = np.concatenate([[0.0], np.cumsum(exp[order]) / max(exp.sum(), 1e-9)])
            cum_claims= np.concatenate([[0.0], np.cumsum(act[order]) / max(act.sum(), 1e-9)])
            idx       = np.linspace(0, len(cum_exp) - 1, min(200, len(cum_exp)), dtype=int)
            lorenz[m] = {"cum_exp": cum_exp[idx].tolist(), "cum_claims": cum_claims[idx].tolist()}
            gini[m]   = float(1.0 - 2.0 * np.trapezoid(cum_claims, cum_exp))

            # Lift by decile
            cumexp = np.cumsum(exp[order]); total = cumexp[-1]
            da, dp = [], []
            for d in range(10):
                lo, hi = total * d / 10, total * (d + 1) / 10
                mask   = (cumexp > lo) & (cumexp <= hi)
                if mask.any():
                    ed = exp[order][mask].sum()
                    da.append(float(act[order][mask].sum() / max(ed, 1e-9)))
                    dp.append(float(pred[order][mask].sum() / max(ed, 1e-9)))
                else:
                    da.append(None); dp.append(None)
            lift_decile[m] = {"actual": da, "predicted": dp}

        # Double-lift: for all pairs, sort by ratio pred_A / pred_B
        double_lift: dict[str, dict] = {}
        mnames = [m for m in self.models if m in self.predictions]
        for i, ma in enumerate(mnames):
            for j, mb in enumerate(mnames):
                if j <= i:
                    continue
                pa    = np.asarray(self.predictions[ma], float) / np.maximum(exp, 1e-9)
                pb    = np.asarray(self.predictions[mb], float) / np.maximum(exp, 1e-9)
                ratio = pa / np.maximum(pb, 1e-9)
                order = np.argsort(ratio)
                cumexp = np.cumsum(exp[order]); total = cumexp[-1]
                d_act, d_pa, d_pb = [], [], []
                for d in range(10):
                    lo, hi = total * d / 10, total * (d + 1) / 10
                    mask   = (cumexp > lo) & (cumexp <= hi)
                    if mask.any():
                        ed = exp[order][mask].sum()
                        d_act.append(float(act[order][mask].sum() / max(ed, 1e-9)))
                        d_pa.append(float(self.predictions[ma][order][mask].sum() / max(ed, 1e-9)))
                        d_pb.append(float(self.predictions[mb][order][mask].sum() / max(ed, 1e-9)))
                    else:
                        d_act.append(None); d_pa.append(None); d_pb.append(None)
                double_lift[f"{ma}||{mb}"] = {
                    "ma": ma, "mb": mb,
                    "actual": d_act, "pred_a": d_pa, "pred_b": d_pb,
                }

        return {"lorenz": lorenz, "gini": gini,
                "lift_decile": lift_decile, "double_lift": double_lift}

    # ── Tab 5: SHAP ───────────────────────────────────────────────────────────

    def _compute_shap(self, model_objects: dict) -> dict:
        try:
            import shap  # noqa: F401
        except ImportError:
            return {"_error": "shap not installed — run: pip install shap"}

        import shap

        test   = self.test
        n_samp = max(1, int(len(test) * self.SHAP_SAMPLE_FRAC))
        rng    = np.random.RandomState(42)
        idx    = rng.choice(len(test), n_samp, replace=False)
        sample = test.iloc[idx].reset_index(drop=True)

        feature_cols = self.numeric_cols + self.categorical_cols
        result: dict[str, Any] = {}

        for m_name, m_res in model_objects.items():
            if m_name not in self.TREE_MODELS:
                continue
            try:
                model_obj = m_res.get("model")
                if model_obj is None:
                    continue

                # ── build feature matrix for this model ──────────────────────
                if m_name in ("XGBoost", "XGBoost-W"):
                    import xgboost as xgb
                    x = sample[feature_cols].copy()
                    for col in self.categorical_cols:
                        x[col] = x[col].astype("category")
                    dmat        = xgb.DMatrix(x, enable_categorical=True)
                    explainer   = shap.TreeExplainer(model_obj)
                    shap_values = explainer.shap_values(dmat)

                elif m_name == "GBM":
                    x = sample[feature_cols].copy()
                    for col in self.categorical_cols:
                        x[col] = x[col].astype("category")
                    explainer   = shap.TreeExplainer(model_obj)
                    shap_values = explainer.shap_values(x)

                elif m_name == "CatBoost":
                    cat_cols = m_res.get("categorical_cols", self.categorical_cols)
                    num_cols = m_res.get("numeric_cols", self.numeric_cols)
                    fc       = num_cols + cat_cols
                    x        = sample[fc].copy()
                    for col in cat_cols:
                        x[col] = x[col].astype(str)
                    explainer   = shap.TreeExplainer(model_obj)
                    shap_values = explainer.shap_values(x)
                    feature_cols = fc   # local override for this model

                else:
                    continue

                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                n_feat = len(feature_cols)
                if shap_values.shape[1] > n_feat:
                    shap_values = shap_values[:, :n_feat]

                mean_abs = [float(np.abs(shap_values[:, fi]).mean())
                            for fi in range(shap_values.shape[1])]

                # Beeswarm: normalise feature values for colouring
                bee_n  = min(300, n_samp)
                feat_norm: dict[str, list] = {}
                for fi, col in enumerate(feature_cols):
                    v = sample[col].iloc[:bee_n]
                    if col in self.numeric_cols:
                        vf = v.astype(float).values
                        mn, mx = vf.min(), vf.max()
                        feat_norm[col] = ((vf - mn) / max(mx - mn, 1e-9)).tolist()
                    else:
                        cats = v.astype(str)
                        rank = cats.value_counts().rank(ascending=False, method="first")
                        feat_norm[col] = (cats.map(rank) / max(rank.max(), 1)).tolist()

                result[m_name] = {
                    "features":    feature_cols,
                    "importance":  dict(zip(feature_cols, mean_abs)),
                    "shap_sample": shap_values[:bee_n].tolist(),
                    "feat_norm":   feat_norm,
                }

            except Exception as exc:
                result[m_name] = {"_error": str(exc)}

        return result

    # ── Tab 6: Underperformance ───────────────────────────────────────────────

    def _compute_underperformance(self) -> dict:
        test      = self.test
        act       = test[self.response_col].values.astype(float)
        exp       = test[self.exposure_col].values
        act_freq  = act / np.maximum(exp, 1e-9)
        total_exp = float(exp.sum())
        min_exp   = total_exp * self.MIN_EXP_FRAC

        # ── A/P by decile (sorted by predicted frequency) ────────────────────
        ap_by_decile: dict[str, dict] = {}
        for m in self.models:
            if m not in self.predictions:
                continue
            pred      = np.asarray(self.predictions[m], float)
            pred_freq = pred / np.maximum(exp, 1e-9)
            order     = np.argsort(pred_freq)
            cumexp    = np.cumsum(exp[order]); total = cumexp[-1]
            d_act, d_pred, d_ap = [], [], []
            for d in range(10):
                lo, hi = total * d / 10, total * (d + 1) / 10
                mask   = (cumexp > lo) & (cumexp <= hi)
                if mask.any():
                    ed   = exp[order][mask].sum()
                    a_d  = act[order][mask].sum() / max(ed, 1e-9)
                    p_d  = pred[order][mask].sum() / max(ed, 1e-9)
                    d_act.append(float(a_d))
                    d_pred.append(float(p_d))
                    d_ap.append(float(a_d / max(p_d, 1e-9)))
                else:
                    d_act.append(None); d_pred.append(None); d_ap.append(None)
            ap_by_decile[m] = {"actual": d_act, "predicted": d_pred, "ap": d_ap}

        # ── Feature residuals ────────────────────────────────────────────────
        feat_residuals: dict[str, dict] = {}
        for col in self.numeric_cols + self.categorical_cols:
            is_num  = col in self.numeric_cols
            bin_col = (self._bin_numeric(test[col]).astype(str) if is_num
                       else test[col].astype(str))
            groups: dict[str, dict] = {}
            for i, b in enumerate(bin_col):
                g = groups.setdefault(b, {"exp": 0.0, "act_wt": 0.0,
                                           "preds": {m: 0.0 for m in self.models}})
                g["exp"]    += exp[i]
                g["act_wt"] += act_freq[i] * exp[i]
                for m in self.models:
                    if m in self.predictions:
                        pf = self.predictions[m][i] / max(exp[i], 1e-9)
                        g["preds"][m] += pf * exp[i]

            skeys = sorted(groups) if is_num else sorted(groups, key=lambda k: -groups[k]["exp"])
            feat_residuals[col] = {
                "bins":      skeys,
                "exposure":  [groups[k]["exp"] for k in skeys],
                "models":    {
                    m: [float((groups[k]["act_wt"] - groups[k]["preds"][m])
                              / max(groups[k]["exp"], 1e-9))
                        for k in skeys]
                    for m in self.models if m in self.predictions
                },
                "is_numeric": is_num,
            }

        # ── Prediction spread (histogram of rates) ───────────────────────────
        pred_spread: dict[str, dict] = {}
        for m in self.models:
            if m not in self.predictions:
                continue
            pf = np.asarray(self.predictions[m], float) / np.maximum(exp, 1e-9)
            counts, edges = np.histogram(pf, bins=50)
            pred_spread[m] = {"edges": edges.tolist(), "counts": counts.tolist()}

        # ── 2-D A/P heatmap (all feature pairs) ─────────────────────────────
        def _bin_series(col: str) -> pd.Series:
            if col in self.numeric_cols:
                try:
                    return self._bin_numeric(test[col], n=self.HEATMAP_BINS).astype(str)
                except Exception:
                    return test[col].astype(str)
            else:
                s   = test[col].astype(str)
                top = s.value_counts().head(self.HEATMAP_BINS).index
                return s.where(s.isin(top), "Other")

        all_cols = self.numeric_cols + self.categorical_cols
        heatmaps: dict[str, dict] = {}
        for i, col1 in enumerate(all_cols):
            for j, col2 in enumerate(all_cols):
                if j <= i:
                    continue
                b1 = _bin_series(col1)
                b2 = _bin_series(col2)
                cells: dict[tuple, dict] = {}
                for idx in range(len(test)):
                    k = (str(b1.iloc[idx]), str(b2.iloc[idx]))
                    c = cells.setdefault(k, {"exp": 0.0, "claims": 0.0,
                                              "pred": {m: 0.0 for m in self.models}})
                    c["exp"]    += exp[idx]
                    c["claims"] += act[idx]
                    for m in self.models:
                        if m in self.predictions:
                            c["pred"][m] += self.predictions[m][idx]

                keys1 = sorted(set(k[0] for k in cells))
                keys2 = sorted(set(k[1] for k in cells))
                ap_matrices: dict[str, list] = {}
                for m in self.models:
                    if m not in self.predictions:
                        continue
                    mat = []
                    for k1 in keys1:
                        row = []
                        for k2 in keys2:
                            c = cells.get((k1, k2))
                            if c is None or c["exp"] < min_exp:
                                row.append(None)
                            else:
                                p = c["pred"][m]
                                a = c["claims"]
                                row.append(float(a / max(p, 1e-9)) if p > 0 else None)
                        mat.append(row)
                    ap_matrices[m] = mat

                key = f"{col1}||{col2}"
                heatmaps[key] = {
                    "col1": col1, "col2": col2,
                    "labels1": keys1, "labels2": keys2,
                    "ap_matrices": ap_matrices,
                }

        return {
            "ap_by_decile":   ap_by_decile,
            "feat_residuals": feat_residuals,
            "pred_spread":    pred_spread,
            "heatmaps":       heatmaps,
        }

    # ── Tab 7: Profit Matrix ─────────────────────────────────────────────────

    def _compute_profit_matrix(self, train_preds: dict | None = None) -> dict:
        """
        For each ordered pair (A, B):
          Win set A = policies where pred_A < pred_B (ties → equal split).
          Profit A  = sum(predicted_A × win_weight) - sum(actual × win_weight).
          Market share = sum(exposure × win_weight) / total_exposure.

        Also computes a many-model tournament where all models compete
        simultaneously and ties are split equally.
        """
        def _one_set(df: pd.DataFrame, preds: dict) -> dict:
            act      = df[self.response_col].values.astype(float)
            exp      = df[self.exposure_col].values
            total_e  = float(exp.sum())
            mlist    = [m for m in self.models if m in preds]
            n        = len(mlist)

            profit_mat = [[None] * n for _ in range(n)]
            share_mat  = [[None] * n for _ in range(n)]
            ap_mat     = [[None] * n for _ in range(n)]

            pred_arrays = {m: np.asarray(preds[m], float) for m in mlist}

            for i, ma in enumerate(mlist):
                for j, mb in enumerate(mlist):
                    if i == j:
                        continue
                    pa = pred_arrays[ma]
                    pb = pred_arrays[mb]
                    # Weight: 1 for strict win, 0.5 for tie, 0 for loss
                    w = (pa < pb).astype(float) + (pa == pb).astype(float) * 0.5
                    won_exp  = float((exp * w).sum())
                    won_pred = float((pa  * w).sum())
                    won_act  = float((act * w).sum())
                    profit_mat[i][j] = float(won_pred - won_act)
                    share_mat[i][j]  = float(won_exp / max(total_e, 1e-9))
                    ap_mat[i][j]     = float(won_act / max(won_pred, 1e-9)) if won_pred > 0 else None

            # Tournament: all models compete; ties split equally
            stack    = np.stack([pred_arrays[m] for m in mlist], axis=1)
            min_pred = stack.min(axis=1, keepdims=True)
            is_min   = (stack == min_pred).astype(float)
            alloc    = is_min / np.maximum(is_min.sum(axis=1, keepdims=True), 1)
            tournament = {
                m: float((exp * alloc[:, i]).sum() / max(total_e, 1e-9))
                for i, m in enumerate(mlist)
            }
            return {
                "models":       mlist,
                "profit":       profit_mat,
                "market_share": share_mat,
                "ap_ratio":     ap_mat,
                "tournament":   tournament,
            }

        result = {"test": _one_set(self.test, self.predictions)}
        if train_preds:
            result["train"] = _one_set(self.train, train_preds)
        return result

    # ── Public generate() ─────────────────────────────────────────────────────

    def generate(
        self,
        output_path:   str | pathlib.Path,
        model_objects: dict | None = None,
        train_preds:   dict | None = None,
    ) -> pathlib.Path:
        """
        Compute all analyses and write a self-contained HTML report.

        Parameters
        ----------
        output_path   : path to write the HTML file
        model_objects : dict of model_name → fit-result-dict (enables SHAP)
        train_preds   : dict of model_name → count array on training data
                        (enables the Train view of the Profit Matrix)
        """
        print("  Building benchmark report ...")
        overview    = self._compute_overview()
        leaderboard = self._compute_leaderboard()
        oneway      = self._compute_oneway()
        lift_gini   = self._compute_lift_gini()
        underperf   = self._compute_underperformance()
        profit      = self._compute_profit_matrix(train_preds)
        shap_data: dict = {}
        if model_objects:
            print("  Computing SHAP (20 % of test set) ...")
            shap_data = self._compute_shap(model_objects)

        payload = {
            "overview":    overview,
            "leaderboard": leaderboard,
            "oneway":      oneway,
            "lift_gini":   lift_gini,
            "underperf":   underperf,
            "profit":      profit,
            "shap":        shap_data,
            "colors":      self.COLORS,
            "task":        self.task,
            "family":      self.family,
            "title":       self.title,
        }

        data_json = json.dumps(payload, allow_nan=False, default=_json_default)
        ts        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        html      = (_HTML_TEMPLATE
                     .replace("__DATA_JSON__",  data_json)
                     .replace("__TITLE__",       self.title)
                     .replace("__TIMESTAMP__",   ts)
                     .replace("__TASK__",        self.task)
                     .replace("__FAMILY__",      self.family))

        out = pathlib.Path(output_path)
        out.write_text(html, encoding="utf-8")
        print(f"  Report saved → {out}")
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  JSON serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):                      return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.ndarray):                         return obj.tolist()
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  HTML Template  (placeholder tokens: __DATA_JSON__, __TITLE__,
#                  __TIMESTAMP__, __TASK__, __FAMILY__)
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f0f2f5;
         color: #222; font-size: 14px; }
  header { background: #1a3c6e; color: #fff; padding: 14px 24px;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.25rem; font-weight: 600; }
  header span { font-size: 0.8rem; opacity: 0.75; }
  .tab-bar { display: flex; background: #fff; border-bottom: 2px solid #dde;
             padding: 0 12px; gap: 2px; flex-wrap: wrap; }
  .tab-btn { padding: 10px 18px; border: none; background: none; cursor: pointer;
             font-size: 13px; font-weight: 500; color: #555; border-bottom: 3px solid transparent;
             transition: color .15s, border-color .15s; white-space: nowrap; }
  .tab-btn:hover { color: #1a3c6e; }
  .tab-btn.active { color: #1a3c6e; border-bottom-color: #1a3c6e; }
  .tab-pane { display: none; padding: 20px; }
  .tab-pane.active { display: block; }
  .cards { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
  .card { background: #fff; border-radius: 8px; padding: 16px 20px;
          box-shadow: 0 1px 3px #0002; min-width: 140px; }
  .card .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
                 color: #888; margin-bottom: 4px; }
  .card .value { font-size: 1.5rem; font-weight: 700; color: #1a3c6e; }
  .card .sub   { font-size: 11px; color: #aaa; }
  .section { background: #fff; border-radius: 8px; padding: 18px;
             box-shadow: 0 1px 3px #0002; margin-bottom: 18px; }
  .section h3 { font-size: 13px; font-weight: 600; color: #444; margin-bottom: 12px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  canvas { max-width: 100%; }
  select, input[type=checkbox] { accent-color: #1a3c6e; }
  select { padding: 5px 8px; border: 1px solid #ccc; border-radius: 4px;
           background: #fff; font-size: 13px; }
  label { font-size: 13px; }
  .ctrl { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
          margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th { background: #f5f7fa; font-size: 11px; text-transform: uppercase;
       letter-spacing: .05em; padding: 7px 10px; text-align: right;
       cursor: pointer; user-select: none; }
  th:first-child { text-align: left; }
  th:hover { background: #e8edf5; }
  td { padding: 6px 10px; border-top: 1px solid #eee; text-align: right; }
  td:first-child { text-align: left; font-weight: 500; }
  tr:hover td { background: #f9fbff; }
  .pill { display: inline-block; border-radius: 3px; padding: 1px 6px;
          font-size: 11px; font-weight: 600; }
  .heatmap-wrap { overflow: auto; }
  .heatmap-table { border-collapse: collapse; font-size: 11px; white-space: nowrap; }
  .heatmap-table td { padding: 4px 8px; min-width: 52px; text-align: center; }
  .heatmap-table th { background: #f0f2f5; font-weight: 600; padding: 4px 8px;
                      font-size: 10px; }
  .profit-pos { background: rgba(40,167,69,.18); }
  .profit-neg { background: rgba(220,53,69,.18); }
  .tag-best { background: #d4edda; color: #155724; }
  .tag-worst { background: #f8d7da; color: #721c24; }
  .chkbar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
  .chkbar label { display: flex; align-items: center; gap: 4px; cursor: pointer; }
  .toggle-pair { display: flex; gap: 0; border: 1px solid #ccc; border-radius: 4px;
                 overflow: hidden; }
  .toggle-pair button { padding: 5px 14px; border: none; background: #fff; cursor: pointer;
                        font-size: 12px; font-weight: 500; }
  .toggle-pair button.active { background: #1a3c6e; color: #fff; }
  .beeswarm-wrap { overflow: auto; }
  @media (max-width: 700px) { .grid2,.grid3 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span>__TASK__ · __FAMILY__ · generated __TIMESTAMP__</span>
</header>

<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('overview')">1 · Overview</button>
  <button class="tab-btn" onclick="showTab('leaderboard')">2 · Leaderboard</button>
  <button class="tab-btn" onclick="showTab('oneway')">3 · One-Way</button>
  <button class="tab-btn" onclick="showTab('liftgini')">4 · Lift &amp; Gini</button>
  <button class="tab-btn" onclick="showTab('shap')">5 · SHAP</button>
  <button class="tab-btn" onclick="showTab('underperf')">6 · Underperformance</button>
  <button class="tab-btn" onclick="showTab('profit')">7 · Profit Matrix</button>
</div>

<!-- ── Tab 1: Overview ───────────────────────────────────────────────────── -->
<div id="tab-overview" class="tab-pane active">
  <div class="cards" id="ov-cards"></div>
  <div class="grid2">
    <div class="section">
      <h3>Numeric Features</h3>
      <div class="ctrl">
        <label>Feature <select id="ov-num-sel"></select></label>
      </div>
      <canvas id="ov-num-chart" height="220"></canvas>
    </div>
    <div class="section">
      <h3>Categorical Features</h3>
      <div class="ctrl">
        <label>Feature <select id="ov-cat-sel"></select></label>
      </div>
      <canvas id="ov-cat-chart" height="220"></canvas>
    </div>
  </div>
</div>

<!-- ── Tab 2: Leaderboard ───────────────────────────────────────────────── -->
<div id="tab-leaderboard" class="tab-pane">
  <div class="section">
    <h3>Metric Comparison</h3>
    <div class="ctrl">
      <label>Metric
        <select id="lb-metric">
          <option>Deviance</option><option>MSE</option><option>MAE</option>
          <option>RMSE</option><option>AUC</option><option>Gini</option>
        </select>
      </label>
    </div>
    <canvas id="lb-bar" height="160"></canvas>
  </div>
  <div class="grid2">
    <div class="section">
      <h3>All Metrics Table</h3>
      <div style="overflow:auto"><table id="lb-table"></table></div>
    </div>
    <div class="section">
      <h3>Training Time (seconds)</h3>
      <canvas id="lb-time" height="180"></canvas>
    </div>
  </div>
  <div class="section">
    <h3>Predicted vs Actual Frequency — By Decile</h3>
    <div class="ctrl"><label>Model <select id="lb-decile-model"></select></label></div>
    <canvas id="lb-decile" height="180"></canvas>
  </div>
</div>

<!-- ── Tab 3: One-Way ───────────────────────────────────────────────────── -->
<div id="tab-oneway" class="tab-pane">
  <div class="section">
    <div class="ctrl">
      <label>Feature <select id="ow-feat"></select></label>
    </div>
    <div class="chkbar" id="ow-models"></div>
    <canvas id="ow-chart" height="260"></canvas>
  </div>
</div>

<!-- ── Tab 4: Lift & Gini ───────────────────────────────────────────────── -->
<div id="tab-liftgini" class="tab-pane">
  <div class="grid2">
    <div class="section">
      <h3>Lorenz Curves</h3>
      <canvas id="lg-lorenz" height="280"></canvas>
    </div>
    <div class="section">
      <h3>Gini Coefficients</h3>
      <canvas id="lg-gini" height="280"></canvas>
    </div>
  </div>
  <div class="section">
    <h3>Lift by Decile (sorted by predicted frequency)</h3>
    <div class="ctrl"><label>Model <select id="lg-lift-model"></select></label></div>
    <canvas id="lg-lift" height="200"></canvas>
  </div>
  <div class="section">
    <h3>Double Lift (sorted by pred_A / pred_B ratio)</h3>
    <div class="ctrl">
      <label>Model A <select id="dl-a"></select></label>
      <label>vs Model B <select id="dl-b"></select></label>
    </div>
    <canvas id="lg-double" height="200"></canvas>
  </div>
</div>

<!-- ── Tab 5: SHAP ─────────────────────────────────────────────────────── -->
<div id="tab-shap" class="tab-pane">
  <div class="section">
    <div class="ctrl"><label>Model <select id="shap-model"></select></label></div>
    <div id="shap-err" style="display:none;color:#c00;padding:8px"></div>
    <div class="grid2">
      <div>
        <h3 style="margin-bottom:8px">Feature Importance (mean |SHAP|)</h3>
        <canvas id="shap-imp" height="260"></canvas>
      </div>
      <div>
        <h3 style="margin-bottom:8px">Beeswarm</h3>
        <canvas id="shap-bee" height="260"></canvas>
      </div>
    </div>
  </div>
</div>

<!-- ── Tab 6: Underperformance ─────────────────────────────────────────── -->
<div id="tab-underperf" class="tab-pane">
  <div class="section">
    <h3>A/P Ratio by Predicted-Frequency Decile</h3>
    <canvas id="up-ap" height="200"></canvas>
  </div>
  <div class="section">
    <h3>Feature Residuals (Actual − Predicted frequency)</h3>
    <div class="ctrl"><label>Feature <select id="up-feat"></select></label></div>
    <canvas id="up-resid" height="200"></canvas>
  </div>
  <div class="section">
    <h3>Predicted Frequency Distribution</h3>
    <div class="ctrl"><label>Model <select id="up-spread-model"></select></label></div>
    <canvas id="up-spread" height="180"></canvas>
  </div>
  <div class="section">
    <h3>A/P Heatmap — 2-D Feature Cross</h3>
    <div class="ctrl">
      <label>Row feature <select id="up-hm-r"></select></label>
      <label>Col feature <select id="up-hm-c"></select></label>
      <label>Model <select id="up-hm-model"></select></label>
    </div>
    <div class="heatmap-wrap" id="up-heatmap"></div>
  </div>
</div>

<!-- ── Tab 7: Profit Matrix ─────────────────────────────────────────────── -->
<div id="tab-profit" class="tab-pane">
  <div class="section">
    <div class="ctrl">
      <div class="toggle-pair">
        <button id="pm-test-btn" class="active" onclick="pmSetSplit('test')">Test</button>
        <button id="pm-train-btn" onclick="pmSetSplit('train')">Train</button>
      </div>
      <span style="font-size:12px;color:#888">
        Cell (A → B): profit when Model A wins against B (predicted − actual counts on won portfolio).
        Green = profitable, Red = loss.
      </span>
    </div>
    <div class="heatmap-wrap" id="pm-matrix"></div>
  </div>
  <div class="grid2">
    <div class="section">
      <h3>Tournament Market Share (all models compete simultaneously)</h3>
      <canvas id="pm-tournament" height="200"></canvas>
    </div>
    <div class="section">
      <h3>Market Share Matrix (% exposure won when A vs B)</h3>
      <div class="heatmap-wrap" id="pm-share"></div>
    </div>
  </div>
</div>

<script>
// ─── data ─────────────────────────────────────────────────────────────────
const DATA = __DATA_JSON__;

// ─── helpers ──────────────────────────────────────────────────────────────
const COLORS = DATA.colors;
function mcolor(name) { return COLORS[name] || '#888'; }
function fmt(v, d=3) { return v == null ? '—' : Number(v).toFixed(d); }
function fmtPct(v) { return v == null ? '—' : (v*100).toFixed(1)+'%'; }
function fmtK(v)   { return v == null ? '—' : (v>=1000 ? (v/1000).toFixed(1)+'k' : v.toFixed(1)); }
const CHARTS = {};
function destroyChart(id) { if (CHARTS[id]) { CHARTS[id].destroy(); delete CHARTS[id]; } }
function mk(id, type, data, opts={}) {
  destroyChart(id);
  CHARTS[id] = new Chart(document.getElementById(id), { type, data, options: {
    responsive: true, maintainAspectRatio: true,
    plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } },
    ...opts
  }});
  return CHARTS[id];
}

// ─── tab switching ────────────────────────────────────────────────────────
const INIT = {};
function showTab(id) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  const btn = [...document.querySelectorAll('.tab-btn')]
    .find(b => b.onclick && b.onclick.toString().includes("'"+id+"'"));
  if (btn) btn.classList.add('active');
  if (!INIT[id]) { INIT[id]=true; TAB_INIT[id] && TAB_INIT[id](); }
}

// ─── Tab 1: Overview ─────────────────────────────────────────────────────
function initOverview() {
  const s = DATA.overview.summary;
  const cards = [
    { label:'Training rows',  value: s.n_train.toLocaleString() },
    { label:'Test rows',      value: s.n_test.toLocaleString() },
    { label:'Models fitted',  value: s.n_models },
    { label:'Mean actual freq', value: s.mean_actual.toFixed(4) },
    { label:'Best model',     value: s.best_model },
  ];
  document.getElementById('ov-cards').innerHTML = cards.map(c =>
    `<div class="card"><div class="label">${c.label}</div><div class="value">${c.value}</div></div>`
  ).join('');

  // Numeric selector
  const numSel = document.getElementById('ov-num-sel');
  const numKeys = Object.keys(DATA.overview.numeric);
  numKeys.forEach(k => { const o=document.createElement('option'); o.text=k; numSel.add(o); });
  numSel.onchange = () => drawNumFeat(numSel.value);
  if (numKeys.length) drawNumFeat(numKeys[0]);

  // Categorical selector
  const catSel = document.getElementById('ov-cat-sel');
  const catKeys = Object.keys(DATA.overview.categorical);
  catKeys.forEach(k => { const o=document.createElement('option'); o.text=k; catSel.add(o); });
  catSel.onchange = () => drawCatFeat(catSel.value);
  if (catKeys.length) drawCatFeat(catKeys[0]);
}

function drawNumFeat(col) {
  const d = DATA.overview.numeric[col];
  mk('ov-num-chart','bar',{
    labels: d.labels,
    datasets: [
      { label:'Exposure', data:d.exposure, backgroundColor:'#A8C0D620',
        borderColor:'#A8C0D6', borderWidth:1, yAxisID:'y', order:2 },
      { label:'Actual freq', data:d.freq, type:'line', borderColor:'#C0392B',
        backgroundColor:'#C0392B20', pointRadius:3, yAxisID:'y2', tension:.3, order:1 },
    ]
  }, {
    scales: {
      x:  { ticks:{ maxRotation:45, font:{size:9} } },
      y:  { type:'linear', position:'left',  title:{display:true,text:'Exposure'} },
      y2: { type:'linear', position:'right', title:{display:true,text:'Freq'}, grid:{drawOnChartArea:false} },
    }, plugins:{ legend:{position:'top'} }
  });
}

function drawCatFeat(col) {
  const d = DATA.overview.categorical[col];
  mk('ov-cat-chart','bar',{
    labels: d.levels,
    datasets: [
      { label:'Exposure', data:d.exposure, backgroundColor:'#6C8EAD30',
        borderColor:'#6C8EAD', borderWidth:1, yAxisID:'y', order:2 },
      { label:'Actual freq', data:d.freq, type:'line', borderColor:'#C0392B',
        pointRadius:3, yAxisID:'y2', tension:.2, order:1 },
    ]
  }, {
    scales: {
      x:  { ticks:{ font:{size:9} } },
      y:  { type:'linear', position:'left',  title:{display:true,text:'Exposure'} },
      y2: { type:'linear', position:'right', title:{display:true,text:'Freq'}, grid:{drawOnChartArea:false} },
    }, plugins:{ legend:{position:'top'} }
  });
}

// ─── Tab 2: Leaderboard ──────────────────────────────────────────────────
function initLeaderboard() {
  const lb = DATA.leaderboard;
  const models = lb.models;
  let sortCol = null, sortAsc = true;

  // Metric bar
  const metSel = document.getElementById('lb-metric');
  const drawMetBar = () => {
    const metric = metSel.value;
    const vals = models.map(m => lb.metrics[m][metric]);
    mk('lb-bar','bar',{
      labels: models,
      datasets:[{ label:metric, data:vals,
        backgroundColor: models.map(m=>mcolor(m)+'cc'), borderWidth:0 }]
    },{ plugins:{legend:{display:false}},
        scales:{ x:{ticks:{font:{size:10}}}, y:{title:{display:true,text:metric}} } });
  };
  metSel.onchange = drawMetBar;
  drawMetBar();

  // Table
  const cols = ['Deviance','MSE','MAE','RMSE','AUC','Gini','Avg Pred','Avg Actual'];
  function renderTable() {
    let rows = models.map(m => ({m, ...lb.metrics[m]}));
    if (sortCol) {
      rows.sort((a,b)=> {
        const av=a[sortCol]??Infinity, bv=b[sortCol]??Infinity;
        return sortAsc ? av-bv : bv-av;
      });
    }
    const tbl = document.getElementById('lb-table');
    tbl.innerHTML = `<thead><tr><th onclick="lbSort('Model')">Model</th>${
      cols.map(c=>`<th onclick="lbSort('${c}')">${c}</th>`).join('')
    }</tr></thead><tbody>${
      rows.map(r=>{
        const best = cols.map(c=>({c,v:r[c]}));
        return `<tr><td><span class="pill" style="background:${mcolor(r.m)}22;color:${mcolor(r.m)}">${r.m}</span></td>${
          cols.map(c=>`<td>${fmt(r[c],4)}</td>`).join('')
        }</tr>`;
      }).join('')
    }</tbody>`;
  }
  window.lbSort = (col)=>{
    if(sortCol===col) sortAsc=!sortAsc; else { sortCol=col; sortAsc=true; }
    renderTable();
  };
  renderTable();

  // Training time
  mk('lb-time','bar',{
    labels: models,
    datasets:[{ label:'Seconds', data:models.map(m=>lb.times[m]),
      backgroundColor: models.map(m=>mcolor(m)+'bb'), borderWidth:0 }]
  },{ plugins:{legend:{display:false}},
      scales:{ x:{ticks:{font:{size:10}}}, y:{title:{display:true,text:'Seconds'}} } });

  // Decile chart
  const decSel = document.getElementById('lb-decile-model');
  models.forEach(m=>{ const o=document.createElement('option'); o.text=m; decSel.add(o); });
  const drawDecile = () => {
    const m = decSel.value;
    const d = DATA.lift_gini.lift_decile[m];
    if (!d) return;
    const lbls = Array.from({length:10},(_,i)=>'D'+(i+1));
    mk('lb-decile','bar',{
      labels:lbls,
      datasets:[
        { label:'Actual',    data:d.actual,    backgroundColor:'#C0392B44',
          borderColor:'#C0392B', borderWidth:1.5 },
        { label:'Predicted', data:d.predicted, backgroundColor:mcolor(m)+'44',
          borderColor:mcolor(m), borderWidth:1.5 },
      ]
    },{ scales:{x:{title:{display:true,text:'Decile (low→high predicted freq)'}},
                y:{title:{display:true,text:'Mean frequency'}}} });
  };
  decSel.onchange = drawDecile;
  if (models.length) drawDecile();
}

// ─── Tab 3: One-Way ──────────────────────────────────────────────────────
function initOneway() {
  const ow = DATA.oneway;
  const features = Object.keys(ow);
  const allModels = DATA.leaderboard.models;

  const featSel = document.getElementById('ow-feat');
  features.forEach(f=>{ const o=document.createElement('option'); o.text=f; featSel.add(o); });

  // Model checkboxes
  const chkbar = document.getElementById('ow-models');
  allModels.forEach((m,i)=>{
    const id='ow-chk-'+i;
    chkbar.innerHTML += `<label><input type="checkbox" id="${id}" value="${m}" ${i<6?'checked':''}
      onchange="drawOneway()">&nbsp;<span style="color:${mcolor(m)};font-weight:600">${m}</span></label>`;
  });

  window.drawOneway = () => {
    const feat = featSel.value;
    const d    = ow[feat];
    const chk  = [...document.querySelectorAll('#ow-models input:checked')].map(x=>x.value);
    mk('ow-chart','bar',{
      labels: d.bins,
      datasets: [
        { label:'Exposure', data:d.exposure, backgroundColor:'#6C8EAD18',
          borderColor:'#6C8EAD', borderWidth:1, yAxisID:'yexp', order:99 },
        { label:'Actual', data:d.actual, type:'line', borderColor:'#222',
          borderWidth:2.5, pointRadius:3, fill:false, yAxisID:'y', order:0 },
        ...chk.filter(m=>d.models[m]).map(m=>({
          label:m, data:d.models[m], type:'line',
          borderColor:mcolor(m), pointRadius:2, borderWidth:1.5, fill:false,
          yAxisID:'y', tension:.2, borderDash:[]
        })),
      ]
    },{
      scales:{
        x:    { ticks:{maxRotation:50,font:{size:9}} },
        y:    { type:'linear', position:'left',  title:{display:true,text:'Frequency'} },
        yexp: { type:'linear', position:'right', title:{display:true,text:'Exposure'},
                grid:{drawOnChartArea:false} },
      }
    });
  };
  featSel.onchange = drawOneway;
  if (features.length) drawOneway();
}

// ─── Tab 4: Lift & Gini ──────────────────────────────────────────────────
function initLiftGini() {
  const lg = DATA.lift_gini;
  const models = Object.keys(lg.lorenz);

  // Lorenz
  mk('lg-lorenz','line',{
    datasets: [
      { label:'Random', data:[{x:0,y:0},{x:1,y:1}], borderColor:'#aaa',
        borderDash:[4,4], borderWidth:1, pointRadius:0 },
      ...models.map(m=>({
        label:`${m} (G=${fmt(lg.gini[m],3)})`,
        data: lg.lorenz[m].cum_exp.map((x,i)=>({x, y:lg.lorenz[m].cum_claims[i]})),
        borderColor:mcolor(m), pointRadius:0, borderWidth:1.8, fill:false,
      }))
    ]
  },{
    scales:{ x:{type:'linear',title:{display:true,text:'Cumulative Exposure'}},
             y:{title:{display:true,text:'Cumulative Claims'}} },
    plugins:{ legend:{position:'right',labels:{font:{size:10}}} },
  });

  // Gini bar
  const giniModels = Object.keys(lg.gini).sort((a,b)=>lg.gini[b]-lg.gini[a]);
  mk('lg-gini','bar',{
    labels: giniModels,
    datasets:[{ label:'Gini', data:giniModels.map(m=>lg.gini[m]),
      backgroundColor:giniModels.map(m=>mcolor(m)+'cc'), borderWidth:0 }]
  },{
    indexAxis:'y',
    plugins:{legend:{display:false}},
    scales:{ x:{title:{display:true,text:'Gini coefficient'}, max:1},
             y:{ticks:{font:{size:11}}} },
  });

  // Lift by decile
  const liftSel = document.getElementById('lg-lift-model');
  models.forEach(m=>{ const o=document.createElement('option'); o.text=m; liftSel.add(o); });
  const drawLift = () => {
    const m=liftSel.value; const d=lg.lift_decile[m]; if(!d) return;
    const lbls=Array.from({length:10},(_,i)=>'D'+(i+1));
    mk('lg-lift','line',{ labels:lbls, datasets:[
      { label:'Actual',    data:d.actual,    borderColor:'#222', borderWidth:2,
        pointRadius:4, fill:false },
      { label:'Predicted', data:d.predicted, borderColor:mcolor(m), borderWidth:1.8,
        borderDash:[4,3], pointRadius:3, fill:false },
    ]},{
      scales:{ x:{title:{display:true,text:'Decile'}},
               y:{title:{display:true,text:'Mean frequency'}} },
    });
  };
  liftSel.onchange = drawLift;
  if (models.length) drawLift();

  // Double lift
  const dlA=document.getElementById('dl-a'), dlB=document.getElementById('dl-b');
  models.forEach(m=>{ [dlA,dlB].forEach(s=>{ const o=document.createElement('option'); o.text=m; s.add(o); }); });
  if (models.length>1) dlB.selectedIndex=1;
  const drawDouble = () => {
    const ma=dlA.value, mb=dlB.value;
    const key=`${ma}||${mb}` in lg.double_lift ? `${ma}||${mb}` : `${mb}||${ma}`;
    const d=lg.double_lift[key]; if(!d) return;
    const lbls=Array.from({length:10},(_,i)=>'D'+(i+1));
    mk('lg-double','line',{ labels:lbls, datasets:[
      { label:'Actual',   data:d.actual, borderColor:'#222', borderWidth:2.2, pointRadius:4, fill:false },
      { label:d.ma,       data:d.pred_a, borderColor:mcolor(d.ma), borderWidth:1.8, borderDash:[5,3], fill:false },
      { label:d.mb,       data:d.pred_b, borderColor:mcolor(d.mb), borderWidth:1.8, borderDash:[2,2], fill:false },
    ]},{
      scales:{ x:{title:{display:true,text:`Decile of pred_${ma}/pred_${mb} ratio (low→high)`}},
               y:{title:{display:true,text:'Mean frequency'}} }
    });
  };
  dlA.onchange=drawDouble; dlB.onchange=drawDouble;
  drawDouble();
}

// ─── Tab 5: SHAP ─────────────────────────────────────────────────────────
function initShap() {
  const shap = DATA.shap;
  const shapModels = Object.keys(shap).filter(m=>!shap[m]._error);
  const sel = document.getElementById('shap-model');
  if (!shapModels.length) {
    document.getElementById('shap-err').style.display='';
    document.getElementById('shap-err').textContent='No SHAP data available.';
    return;
  }
  shapModels.forEach(m=>{ const o=document.createElement('option'); o.text=m; sel.add(o); });
  const drawShap = () => {
    const m=sel.value; const d=shap[m];
    if (d._error) {
      document.getElementById('shap-err').style.display='';
      document.getElementById('shap-err').textContent=d._error;
      destroyChart('shap-imp'); destroyChart('shap-bee'); return;
    }
    document.getElementById('shap-err').style.display='none';
    const feats=d.features;
    const imp=d.importance;
    // Sort by importance
    const order=[...feats.keys()].sort((a,b)=>imp[feats[b]]-imp[feats[a]]);
    const sortedF=order.map(i=>feats[i]);
    mk('shap-imp','bar',{
      labels:sortedF,
      datasets:[{ label:'mean |SHAP|', data:sortedF.map(f=>imp[f]),
        backgroundColor:mcolor(m)+'bb', borderWidth:0 }]
    },{
      indexAxis:'y',
      plugins:{legend:{display:false}},
      scales:{ x:{title:{display:true,text:'mean |SHAP value|'}},
               y:{ticks:{font:{size:10}}} }
    });

    // Beeswarm as scatter per feature
    const sv=d.shap_values; const fn=d.feat_norm; const n=sv.length;
    const datasets=feats.map((f,fi)=>({
      label:f,
      data: sv.map((row,ri)=>({ x:row[fi],
        y: fi + (((ri*2654435769)&0x7fffffff)/0x7fffffff - 0.5)*0.6 })),
      pointRadius:2, pointHoverRadius:3,
      backgroundColor: (fn[f]||[]).slice(0,n).map(v=>{
        const r=Math.round(220*v+10*(1-v)), b=Math.round(10*v+220*(1-v));
        return `rgba(${r},40,${b},0.6)`;
      }),
      borderWidth:0, showLine:false,
    }));
    mk('shap-bee','scatter',{ datasets },{
      plugins:{ legend:{position:'right',labels:{font:{size:9},boxWidth:8}} },
      scales:{
        x:{title:{display:true,text:'SHAP value'}},
        y:{title:{display:true,text:'Feature'},
           ticks:{ callback:(_,i)=>feats[Math.round(i)], stepSize:1 },
           min:-0.5, max:feats.length-0.5 }
      }
    });
  };
  sel.onchange=drawShap;
  drawShap();
}

// ─── Tab 6: Underperformance ─────────────────────────────────────────────
function initUnderperf() {
  const up=DATA.underperf;
  const models=DATA.leaderboard.models;

  // A/P by decile — all models
  mk('up-ap','line',{
    labels:Array.from({length:10},(_,i)=>'D'+(i+1)),
    datasets: models.filter(m=>up.ap_by_decile[m]).map(m=>({
      label:m, data:up.ap_by_decile[m].ap,
      borderColor:mcolor(m), pointRadius:3, borderWidth:1.8, fill:false,
    })).concat([{
      label:'Ideal (A/P=1)', data:Array(10).fill(1),
      borderColor:'#aaa', borderDash:[5,3], borderWidth:1.5, pointRadius:0, fill:false,
    }])
  },{ scales:{ x:{title:{display:true,text:'Decile (low→high predicted)'}},
               y:{title:{display:true,text:'A/P ratio'}} } });

  // Feature residuals
  const featKeys=Object.keys(up.feat_residuals);
  const featSel=document.getElementById('up-feat');
  featKeys.forEach(f=>{ const o=document.createElement('option'); o.text=f; featSel.add(o); });
  const drawResid=()=>{
    const f=featSel.value; const d=up.feat_residuals[f];
    mk('up-resid','bar',{
      labels:d.bins,
      datasets: models.filter(m=>d.models[m]).map(m=>({
        label:m, data:d.models[m], backgroundColor:mcolor(m)+'99', borderWidth:0,
      }))
    },{ scales:{ x:{ticks:{font:{size:9},maxRotation:50}},
                 y:{title:{display:true,text:'Actual − Predicted freq'}} } });
  };
  featSel.onchange=drawResid;
  if(featKeys.length) drawResid();

  // Prediction spread
  const spreadSel=document.getElementById('up-spread-model');
  models.filter(m=>up.pred_spread[m]).forEach(m=>{ const o=document.createElement('option'); o.text=m; spreadSel.add(o); });
  const drawSpread=()=>{
    const m=spreadSel.value; const d=up.pred_spread[m]; if(!d) return;
    const lbls=d.edges.slice(0,-1).map((e,i)=>((e+d.edges[i+1])/2).toFixed(4));
    mk('up-spread','bar',{ labels:lbls,
      datasets:[{ label:'Count', data:d.counts, backgroundColor:mcolor(m)+'88', borderWidth:0 }]
    },{ plugins:{legend:{display:false}},
        scales:{ x:{ticks:{maxTicksLimit:12,font:{size:9}}},
                 y:{title:{display:true,text:'# policies'}} } });
  };
  spreadSel.onchange=drawSpread;
  spreadSel.dispatchEvent(new Event('change'));

  // 2-D heatmap selectors
  const allCols=[...new Set(Object.keys(up.feat_residuals))];
  const hmR=document.getElementById('up-hm-r'), hmC=document.getElementById('up-hm-c');
  const hmM=document.getElementById('up-hm-model');
  allCols.forEach(c=>{ [hmR,hmC].forEach(s=>{ const o=document.createElement('option'); o.text=c; s.add(o); }); });
  if(allCols.length>1) hmC.selectedIndex=1;
  models.forEach(m=>{ const o=document.createElement('option'); o.text=m; hmM.add(o); });
  const drawHeatmap=()=>{
    const r=hmR.value, c=hmC.value, m=hmM.value;
    const key=r+'||'+c in up.heatmaps ? r+'||'+c : c+'||'+r;
    const h=up.heatmaps[key]; if(!h) { document.getElementById('up-heatmap').innerHTML='No data'; return; }
    const mat=(h.col1===r) ? h.ap_matrices[m] : h.ap_matrices[m].map((_,ri)=>h.ap_matrices[m].map(row=>row[ri]));
    const l1=h.col1===r?h.labels1:h.labels2, l2=h.col1===r?h.labels2:h.labels1;
    let html=`<table class="heatmap-table"><thead><tr><th>${r}↓ / ${c}→</th>`;
    l2.forEach(v=>html+=`<th>${v}</th>`); html+='</tr></thead><tbody>';
    mat.forEach((row,ri)=>{
      html+=`<tr><th>${l1[ri]}</th>`;
      row.forEach(v=>{
        if(v==null){html+='<td style="background:#eee;color:#aaa">—</td>';return;}
        const d=v-1; const intens=Math.min(Math.abs(d)/0.4,1);
        const bg=d<0?`rgba(40,167,69,${intens*0.5})`:`rgba(220,53,69,${intens*0.5})`;
        const txt=d<0?'#155724':'#721c24';
        html+=`<td style="background:${bg};color:${txt};font-weight:600">${v.toFixed(2)}</td>`;
      });
      html+='</tr>';
    });
    html+='</tbody></table>';
    document.getElementById('up-heatmap').innerHTML=html;
  };
  hmR.onchange=hmC.onchange=hmM.onchange=drawHeatmap;
  drawHeatmap();
}

// ─── Tab 7: Profit Matrix ─────────────────────────────────────────────────
let _pmSplit = 'test';
function pmSetSplit(s) {
  _pmSplit=s;
  document.getElementById('pm-test-btn').classList.toggle('active', s==='test');
  document.getElementById('pm-train-btn').classList.toggle('active', s==='train');
  renderProfit();
}
function renderProfit() {
  const pm=DATA.profit[_pmSplit];
  if(!pm){document.getElementById('pm-matrix').innerHTML='No '+_pmSplit+' data'; return;}
  const models=pm.models; const n=models.length;
  const profit=pm.profit, share=pm.market_share;

  // Find max absolute profit for colour scaling
  const allP=profit.flat().filter(x=>x!=null);
  const maxAbs=Math.max(...allP.map(Math.abs),1);

  // Profit matrix
  let html=`<table class="heatmap-table"><thead><tr>
    <th>A → B</th>${models.map(m=>`<th>${m}</th>`).join('')}</tr></thead><tbody>`;
  models.forEach((ma,i)=>{
    html+=`<tr><th>${ma}</th>`;
    models.forEach((mb,j)=>{
      if(i===j){html+='<td style="background:#f0f2f5;color:#bbb">—</td>';return;}
      const v=profit[i][j]; if(v==null){html+='<td>—</td>';return;}
      const intens=Math.min(Math.abs(v)/maxAbs,1)*0.6+0.1;
      const bg=v>=0?`rgba(40,167,69,${intens})`:`rgba(220,53,69,${intens})`;
      const fg=v>=0?'#0a3a18':'#4a0010';
      const s=pm.market_share[i][j]; const sh=s!=null?`<br><small>${(s*100).toFixed(1)}% exp</small>`:'';
      html+=`<td style="background:${bg};color:${fg};font-weight:600">${v>=0?'+':''}${v.toFixed(1)}${sh}</td>`;
    });
    html+='</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('pm-matrix').innerHTML=html;

  // Tournament
  const t=pm.tournament;
  const tModels=Object.keys(t).sort((a,b)=>t[b]-t[a]);
  destroyChart('pm-tournament');
  CHARTS['pm-tournament']=new Chart(document.getElementById('pm-tournament'),{
    type:'bar', data:{
      labels:tModels,
      datasets:[{ label:'Market share', data:tModels.map(m=>t[m]*100),
        backgroundColor:tModels.map(m=>mcolor(m)+'cc'), borderWidth:0 }]
    }, options:{
      responsive:true, maintainAspectRatio:true,
      plugins:{legend:{display:false}},
      scales:{ x:{ticks:{font:{size:10}}}, y:{title:{display:true,text:'% exposure won'},max:100} }
    }
  });

  // Market share matrix
  let shtml=`<table class="heatmap-table"><thead><tr>
    <th>A → B</th>${models.map(m=>`<th>${m}</th>`).join('')}</tr></thead><tbody>`;
  models.forEach((ma,i)=>{
    shtml+=`<tr><th>${ma}</th>`;
    models.forEach((mb,j)=>{
      if(i===j){shtml+='<td style="background:#f0f2f5;color:#bbb">—</td>';return;}
      const v=share[i][j];
      if(v==null){shtml+='<td>—</td>';return;}
      const intens=Math.min(v/0.6,1)*0.5;
      shtml+=`<td style="background:rgba(100,150,200,${intens})">${(v*100).toFixed(1)}%</td>`;
    });
    shtml+='</tr>';
  });
  shtml+='</tbody></table>';
  document.getElementById('pm-share').innerHTML=shtml;
}

function initProfit() {
  const pm=DATA.profit;
  if(!pm.train){
    document.getElementById('pm-train-btn').disabled=true;
    document.getElementById('pm-train-btn').title='Train predictions not available';
  }
  renderProfit();
}

// ─── Tab dispatch ─────────────────────────────────────────────────────────
const TAB_INIT = {
  overview:    initOverview,
  leaderboard: initLeaderboard,
  oneway:      initOneway,
  liftgini:    initLiftGini,
  shap:        initShap,
  underperf:   initUnderperf,
  profit:      initProfit,
};

// Initialise active tab
initOverview();
INIT['overview']=true;
</script>
</body>
</html>
"""
