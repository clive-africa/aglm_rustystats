"""
pricing_studio.py — Insurance GLM Pricing Studio
=================================================
A Flet desktop application for end-to-end insurance GLM pricing analysis.

Workflow
--------
  1. Data Setup     — Load policy exposure + claims files, date range, join field
  2. Variable Types — Classify columns; configure percentile binning (default 40 bins)
  3. EDA            — One-way frequency / severity / burning cost charts per predictor
  4. Exercises      — Multi-peril, multi-distribution modelling exercises with cap/collar
  5. Results        — Full model comparison dashboard with diagnostics

GLM Families
------------
  Poisson | Negative Binomial | Tweedie (p = 1.2, 1.5, 1.8) |
  Gamma   | Inverse Gaussian  | Gaussian (OLS baseline)

Install
-------
  pip install flet pandas numpy statsmodels matplotlib openpyxl pyarrow
"""
from __future__ import annotations

import base64, io, threading, warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import flet as ft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import rustystats
except ImportError:
    rustystats = None

warnings.filterwarnings("ignore")

# ─── Palette ──────────────────────────────────────────────────────────────────
BG, CARD          = "#F0F4F8", "#FFFFFF"
PRIMARY, SEC      = "#1A3C6E", "#3A7CA5"
ACCENT            = "#E67E22"
SUCCESS, DANGER   = "#27AE60", "#E74C3C"
MUTED, BORDER     = "#95A5A6", "#D5D8DC"
TXT               = "#2C3E50"
DEFAULT_BINS      = 40

MODEL_COLS = ["#1A3C6E","#3A7CA5","#E67E22","#27AE60",
              "#8E44AD","#E74C3C","#16A085","#D4AC0D"]

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.family": "sans-serif",
})

# ─── Distribution registry ────────────────────────────────────────────────────
# name -> (description, factory)
DIST_REG: Dict[str, tuple] = {
    "Poisson":         ("Frequency",   ''),
    "Neg. Binomial":   ("Freq (OD)",   ''),
    "Tweedie (p=1.2)": ("Freq / BC",   ''),
    "Tweedie (p=1.5)": ("BC",          ''),
    "Tweedie (p=1.8)": ("BC / Sev",    ''),
    "Gamma":           ("Severity",    ''),
    "Inv. Gaussian":   ("Sev (HT)",    ''),
    "Gaussian":        ("Baseline",    ''),
}
FREQ_DISTS = ["Poisson", "Neg. Binomial", "Tweedie (p=1.2)"]
SEV_DISTS  = ["Gamma", "Tweedie (p=1.5)", "Tweedie (p=1.8)", "Inv. Gaussian"]
BC_DISTS   = ["Tweedie (p=1.2)", "Tweedie (p=1.5)", "Tweedie (p=1.8)", "Gamma", "Poisson"]

# Distributions that require strictly positive y
POSITIVE_Y_DISTS = {"Gamma", "Inv. Gaussian", "Tweedie (p=1.5)", "Tweedie (p=1.8)"}

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ColInfo:
    name: str
    detected_type: str       # "Categorical" | "Continuous"
    col_type: str            # user-specified
    n_bins: int = DEFAULT_BINS
    unique_count: int = 0


@dataclass
class ExerciseConfig:
    id: int
    name: str
    perils: List[str]        # empty = all claims
    model_type: str          # "Freq+Sev" | "Burning Cost"
    features: List[str]
    cap: Optional[float] = None
    collar: Optional[float] = None


@dataclass
class ModelResult:
    dist_name: str
    model_subtype: str       # "Frequency" | "Severity" | "Burning Cost"
    deviance: float = float("nan")
    aic: float = float("nan")
    rmse: float = float("nan")
    mae: float = float("nan")
    gini: float = float("nan")
    avg_pred: float = float("nan")
    avg_actual: float = float("nan")
    n_obs: int = 0
    chart_b64: str = ""
    is_best: bool = False
    error: str = ""


@dataclass
class ExerciseResult:
    exercise_id: int
    exercise_name: str
    freq_results: List[ModelResult] = field(default_factory=list)
    sev_results:  List[ModelResult] = field(default_factory=list)
    bc_results:   List[ModelResult] = field(default_factory=list)
    comparison_chart_b64: str = ""
    error: str = ""

    @property
    def all_results(self) -> List[ModelResult]:
        return self.freq_results + self.sev_results + self.bc_results

    @property
    def best_result(self) -> Optional[ModelResult]:
        valid = [r for r in self.all_results
                 if not r.error and not np.isnan(r.deviance)]
        return min(valid, key=lambda r: r.deviance) if valid else None


class AppState:
    def __init__(self):
        self.policy_df:  Optional[pd.DataFrame] = None
        self.claims_df:  Optional[pd.DataFrame] = None
        self.merged_df:  Optional[pd.DataFrame] = None
        self.start_date = self.end_date = ""
        self.policy_date_col = self.incident_date_col = ""
        self.link_col = self.peril_col = ""
        self.col_info: Dict[str, ColInfo] = {}
        self.perils: List[str] = []
        self.exercises: List[ExerciseConfig] = []
        self.exercise_results: List[ExerciseResult] = []
        self._ctr = 0

    def next_id(self) -> int:
        self._ctr += 1
        return self._ctr

    @property
    def policy_cols(self) -> List[str]:
        return list(self.policy_df.columns) if self.policy_df is not None else []

    @property
    def claims_cols(self) -> List[str]:
        return list(self.claims_df.columns) if self.claims_df is not None else []

    @property
    def feature_cols(self) -> List[str]:
        skip = {self.policy_date_col, self.link_col}
        return [c for c, info in self.col_info.items()
                if info.col_type != "Exclude" and c not in skip]


# ─── Backend ──────────────────────────────────────────────────────────────────

def load_file(path: str) -> pd.DataFrame:
    ext = path.lower()
    if ext.endswith(".parquet"):   return pd.read_parquet(path)
    if ext.endswith((".xlsx",".xls")): return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def detect_col_type(s: pd.Series) -> str:
    return "Categorical" if (s.dtype == object or s.nunique() < 20) else "Continuous"


def _ts(s: str) -> Optional[pd.Timestamp]:
    try: return pd.Timestamp(s) if s else None
    except Exception: return None


def filter_by_date(df: pd.DataFrame, col: str, start: str, end: str) -> pd.DataFrame:
    if not col or col not in df.columns:
        return df
    dt   = pd.to_datetime(df[col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    s, e = _ts(start), _ts(end)
    if s: mask &= dt >= s
    if e: mask &= dt <= e
    return df[mask]


def prepare_merged(state: AppState) -> Optional[pd.DataFrame]:
    """Merge policy + claims, apply date filters, produce _claim_count / _claim_amount."""
    if state.policy_df is None:
        return None
    pol = filter_by_date(state.policy_df.copy(),
                          state.policy_date_col, state.start_date, state.end_date)

    if state.claims_df is None or not state.link_col or state.link_col not in pol.columns:
        pol["_claim_count"] = 0
        pol["_claim_amount"] = 0.0
        return pol

    clm = filter_by_date(state.claims_df.copy(),
                          state.incident_date_col, state.start_date, state.end_date)
    if state.link_col not in clm.columns:
        pol["_claim_count"] = 0
        pol["_claim_amount"] = 0.0
        return pol

    # Detect claim amount column
    amt_col = None
    for c in clm.columns:
        if c.lower() in ("claimamount","lossamount","amount","loss","incurredloss",
                          "paidloss","totalloss","claimamt","paid"):
            amt_col = c; break
    if amt_col is None:
        clm["_zero"] = 0.0
        amt_col = "_zero"

    grp_cols = [state.link_col]
    if state.peril_col and state.peril_col in clm.columns:
        grp_cols.append(state.peril_col)

    agg = clm.groupby(grp_cols, as_index=False).agg(
        _claim_count=(state.link_col, "count"),
        _claim_amount=(amt_col, "sum"),
    )

    if state.peril_col and state.peril_col in clm.columns:
        p_cnt = agg.pivot_table(index=state.link_col, columns=state.peril_col,
                                 values="_claim_count",  aggfunc="sum", fill_value=0)
        p_amt = agg.pivot_table(index=state.link_col, columns=state.peril_col,
                                 values="_claim_amount", aggfunc="sum", fill_value=0)
        p_cnt.columns = [f"_claim_count_{c}"  for c in p_cnt.columns]
        p_amt.columns = [f"_claim_amount_{c}" for c in p_amt.columns]
        wide = p_cnt.join(p_amt).reset_index()
        wide["_claim_count"]  = p_cnt.sum(axis=1).values
        wide["_claim_amount"] = p_amt.sum(axis=1).values
        merged = pol.merge(wide, on=state.link_col, how="left")
    else:
        merged = pol.merge(agg, on=state.link_col, how="left")

    for c in merged.columns:
        if c.startswith("_claim"):
            merged[c] = merged[c].fillna(0.0)
    return merged


def get_exposure(df: pd.DataFrame) -> np.ndarray:
    for c in ["Exposure","exposure","PolicyYears","Duration","PolicyCount",
               "policy_years","vehicle_years","Years"]:
        if c in df.columns:
            return np.maximum(pd.to_numeric(df[c], errors="coerce").fillna(1.0).values, 1e-9)
    return np.ones(len(df))


def _qcut(series: pd.Series, n: int) -> pd.Series:
    try:
        return pd.qcut(series, q=n, duplicates="drop").astype(str)
    except Exception:
        return series.astype(str)


def build_X(df: pd.DataFrame, features: List[str],
             col_info: Dict[str, ColInfo]) -> np.ndarray:
    """Design matrix: continuous→qcut→dummies, categorical→dummies."""
    parts = []
    for col in features:
        if col not in df.columns:
            continue
        info = col_info.get(col)
        if info is None or info.col_type == "Exclude":
            continue
        if info.col_type == "Continuous":
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            d = pd.get_dummies(_qcut(s, info.n_bins), prefix=col, drop_first=True, dtype=float)
        else:
            d = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float)
        parts.append(d)
    if not parts:
        return np.ones((len(df), 1))
    X_df = pd.concat(parts, axis=1).fillna(0)
    return sm.add_constant(X_df.values, has_constant="add")


def _gini(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    ws = w.sum()
    ls = (y * w).sum()
    if ws < 1e-9 or ls < 1e-9:
        return float("nan")
    o    = np.argsort(mu)
    cum_e = np.cumsum(w[o]) / ws
    cum_l = np.cumsum(y[o] * w[o]) / ls
    return float(1.0 - 2.0 * np.trapz(cum_l, cum_e))


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def fit_one_glm(
    y: np.ndarray, X: np.ndarray, exp: np.ndarray,
    dist_name: str, subtype: str,
    fw: Optional[np.ndarray] = None,
    cap: Optional[float] = None,
    collar: Optional[float] = None,
) -> ModelResult:
    res = ModelResult(dist_name=dist_name, model_subtype=subtype, n_obs=len(y))
    try:
        family = DIST_REG[dist_name][1]()

        offset = np.log(np.maximum(exp, 1e-9)) if subtype == "Frequency" else None
        fw_arg = fw if subtype == "Severity" else (exp if subtype == "Burning Cost" else None)

        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (y >= 0)
        if dist_name in POSITIVE_Y_DISTS or subtype in ("Severity", "Burning Cost"):
            valid &= (y > 0)
        if fw_arg is not None:
            valid &= np.isfinite(fw_arg) & (fw_arg > 0)
        if valid.sum() < 10:
            res.error = f"Only {valid.sum()} valid rows"
            return res

        yv = y[valid]; Xv = X[valid]; ev = exp[valid]
        offv = offset[valid] if offset is not None else None
        fwv  = fw_arg[valid]  if fw_arg is not None else None

        fitted = sm.GLM(yv, Xv, family=family, offset=offv,
                        freq_weights=fwv).fit(maxiter=100, disp=False)
        mu_raw = np.asarray(fitted.predict(Xv, offset=offv), float)

        if collar is not None: mu_raw = np.maximum(mu_raw, collar)
        if cap    is not None: mu_raw = np.minimum(mu_raw, cap)
        mu = np.maximum(mu_raw, 1e-12)

        res.n_obs      = int(valid.sum())
        res.deviance   = float(fitted.deviance / max(len(yv), 1))
        res.aic        = float(fitted.aic)
        wt             = ev / ev.sum() if ev.sum() > 0 else np.ones(len(yv)) / len(yv)
        res.rmse       = float(np.sqrt(np.mean((yv - mu) ** 2)))
        res.mae        = float(np.mean(np.abs(yv - mu)))
        res.avg_pred   = float(np.dot(mu, wt))
        res.avg_actual = float(np.dot(yv, wt))
        res.gini       = _gini(yv, mu, ev)

        # Diagnostic chart (3 panels)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(
            f"{dist_name} — {subtype}  "
            f"(Dev={res.deviance:.5f}, AIC={res.aic:.1f}, Gini={res.gini:.4f})",
            fontsize=10, fontweight="bold", color=PRIMARY)

        # Panel 1: Predicted vs Actual
        ax = axes[0]
        idx = np.random.choice(len(yv), min(3000, len(yv)), replace=False)
        ax.scatter(mu[idx], yv[idx], alpha=0.25, s=7, color=PRIMARY)
        lim = max(mu.max(), yv.max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set(xlabel="Predicted", ylabel="Actual", title="Predicted vs Actual")

        # Panel 2: Residual distribution
        ax = axes[1]
        ax.hist(yv - mu, bins=40, color=SEC, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="red", lw=1, ls="--")
        ax.set(xlabel="Actual − Predicted", title="Residual Distribution")

        # Panel 3: Lorenz / Gini
        ax = axes[2]
        o = np.argsort(mu)
        ce = np.cumsum(ev[o]) / ev.sum()
        cl = np.cumsum(yv[o] * ev[o]) / max((yv * ev).sum(), 1e-9)
        ax.plot(ce, cl, color=PRIMARY, lw=2, label=f"Gini = {res.gini:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.fill_between(ce, cl, ce, alpha=0.15, color=ACCENT)
        ax.set(xlabel="Cumul. Exposure", ylabel="Cumul. Loss", title="Lorenz Curve")
        ax.legend(fontsize=9)

        fig.tight_layout()
        res.chart_b64 = _fig_to_b64(fig)

    except Exception as ex:
        res.error = str(ex)[:160]

    return res


def make_comparison_chart(results: List[ModelResult]) -> str:
    valid = sorted([r for r in results if not r.error and not np.isnan(r.deviance)],
                   key=lambda r: r.deviance)
    if not valid:
        return ""
    labels = [f"{r.dist_name}\n({r.model_subtype[:4]})" for r in valid]
    cols   = [MODEL_COLS[i % len(MODEL_COLS)] for i in range(len(valid))]
    fig, axes = plt.subplots(1, 3, figsize=(14, max(3.5, 0.6 * len(valid) + 1.5)))
    fig.suptitle("Model Comparison", fontsize=11, fontweight="bold")
    for ax, (title, vals, lo) in zip(axes, [
        ("Deviance ↓", [r.deviance for r in valid], True),
        ("RMSE ↓",     [r.rmse     for r in valid], True),
        ("Gini",       [r.gini     for r in valid], False),
    ]):
        bars = ax.barh(labels, vals, color=cols, alpha=0.85, edgecolor="white")
        bi = int(np.nanargmin(vals) if lo else np.nanargmax(vals))
        bars[bi].set_edgecolor("gold"); bars[bi].set_linewidth(2.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() * 1.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}" if not np.isnan(v) else "–", va="center", fontsize=7.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_b64(fig)


def compute_one_way(
    df: pd.DataFrame, predictor: str, metric: str,
    col_info: Dict[str, ColInfo],
) -> Optional[plt.Figure]:
    if df is None or predictor not in df.columns:
        return None
    exp = get_exposure(df)
    cc  = df["_claim_count"].fillna(0).values  if "_claim_count"  in df.columns else np.zeros(len(df))
    ca  = df["_claim_amount"].fillna(0).values if "_claim_amount" in df.columns else np.zeros(len(df))

    info = col_info.get(predictor)
    if info and info.col_type == "Continuous":
        x_col = _qcut(pd.to_numeric(df[predictor], errors="coerce").fillna(0), info.n_bins)
    else:
        x_col = df[predictor].astype(str).fillna("(missing)")

    tmp = pd.DataFrame({"x": x_col, "exp": exp, "cc": cc, "ca": ca})
    grp = tmp.groupby("x", as_index=False).agg(
        exp=("exp","sum"), cc=("cc","sum"), ca=("ca","sum"), n=("exp","count"))

    if metric == "Frequency":
        grp["y"] = grp["cc"] / grp["exp"].clip(lower=1e-9)
        ylabel = "Claim Frequency"
    elif metric == "Severity":
        grp["y"] = grp["ca"] / grp["cc"].clip(lower=1e-9)
        ylabel = "Average Severity"
    else:
        grp["y"] = grp["ca"] / grp["exp"].clip(lower=1e-9)
        ylabel = "Burning Cost"

    grp = grp.nlargest(min(len(grp), 50), "exp").sort_values("y").reset_index(drop=True)
    overall = float((grp["y"] * grp["exp"]).sum() / max(grp["exp"].sum(), 1e-9))

    fig, ax1 = plt.subplots(figsize=(max(9, len(grp) * 0.42 + 2), 5))
    ax2 = ax1.twinx()
    ax1.bar(range(len(grp)), grp["y"], color=PRIMARY, alpha=0.75)
    ax2.plot(range(len(grp)), grp["exp"], "o-", color=ACCENT, lw=1.5, ms=5, label="Exposure")
    ax1.axhline(overall, color="red", lw=1.2, ls="--", alpha=0.85,
                label=f"Wtd avg = {overall:.4f}")
    ax1.set_xticks(range(len(grp)))
    ax1.set_xticklabels(grp["x"], rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel(ylabel, color=PRIMARY, fontsize=10)
    ax2.set_ylabel("Total Exposure", color=ACCENT, fontsize=10)
    ax1.set_title(f"One-Way: {predictor}  →  {metric}",
                  fontsize=11, fontweight="bold", color=TXT)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def fit_exercise(
    cfg: ExerciseConfig, state: AppState,
    merged: pd.DataFrame, on_progress=None,
) -> ExerciseResult:
    er = ExerciseResult(exercise_id=cfg.id, exercise_name=cfg.name)
    try:
        df = merged.copy()

        # Peril filter: recompute totals for selected perils
        if cfg.perils:
            cc_cols = [f"_claim_count_{p}"  for p in cfg.perils if f"_claim_count_{p}"  in df.columns]
            ca_cols = [f"_claim_amount_{p}" for p in cfg.perils if f"_claim_amount_{p}" in df.columns]
            if cc_cols: df["_claim_count"]  = df[cc_cols].sum(axis=1).fillna(0)
            if ca_cols: df["_claim_amount"] = df[ca_cols].sum(axis=1).fillna(0)

        feats = [f for f in cfg.features if f in df.columns
                  and state.col_info.get(f, ColInfo(f,"Exclude","Exclude")).col_type != "Exclude"]
        X   = build_X(df, feats, state.col_info)
        exp = get_exposure(df)
        cc  = df["_claim_count"].fillna(0).values  if "_claim_count"  in df.columns else np.zeros(len(df))
        ca  = df["_claim_amount"].fillna(0).values if "_claim_amount" in df.columns else np.zeros(len(df))

        if cfg.model_type == "Freq+Sev":
            for d in FREQ_DISTS:
                if on_progress: on_progress(f"[{cfg.name}] Frequency → {d}")
                er.freq_results.append(
                    fit_one_glm(cc.astype(float), X, exp, d, "Frequency",
                                cap=cfg.cap, collar=cfg.collar))
            y_sev = np.where(cc > 0, ca / np.maximum(cc, 1), 0.0)
            for d in SEV_DISTS:
                if on_progress: on_progress(f"[{cfg.name}] Severity  → {d}")
                er.sev_results.append(
                    fit_one_glm(y_sev, X, exp, d, "Severity",
                                fw=cc.astype(float), cap=cfg.cap, collar=cfg.collar))
        else:
            for d in BC_DISTS:
                if on_progress: on_progress(f"[{cfg.name}] Burn. Cost → {d}")
                er.bc_results.append(
                    fit_one_glm(ca.astype(float), X, exp, d, "Burning Cost",
                                cap=cfg.cap, collar=cfg.collar))

        best = er.best_result
        if best: best.is_best = True
        er.comparison_chart_b64 = make_comparison_chart(er.all_results)

    except Exception as ex:
        er.error = str(ex)[:200]
    return er


# ─── UI ───────────────────────────────────────────────────────────────────────

def main(page: ft.Page):
    page.title          = "Insurance GLM Pricing Studio"
    page.window_width   = 1460
    page.window_height  = 940
    page.bgcolor        = BG
    page.padding        = 0

    state   = AppState()
    panels: List[ft.Container] = []

    # ── Helpers ──────────────────────────────────────────────────────────────

    def snack(msg: str, err: bool = False):
        page.snack_bar = ft.SnackBar(
            ft.Text(msg, color="white"),
            bgcolor=DANGER if err else SUCCESS, open=True)
        page.update()

    def _show(idx: int):
        for i, p in enumerate(panels): p.visible = (i == idx)
        page.update()

    def _card(title: str, body: ft.Control, icon=None) -> ft.Container:
        hdr = ([ft.Icon(icon, color=PRIMARY, size=17), ft.Container(width=5)] if icon else [])
        hdr.append(ft.Text(title, weight=ft.FontWeight.BOLD, color=PRIMARY, size=13))
        return ft.Container(
            ft.Column([ft.Row(hdr), ft.Divider(height=1, color=BORDER), body], spacing=8),
            bgcolor=CARD, padding=16, border_radius=10,
            border=ft.border.all(1, BORDER),
            shadow=ft.BoxShadow(blur_radius=6, color="#18000000", spread_radius=0),
            margin=ft.margin.only(bottom=12),
        )

    def _h(txt: str) -> ft.Text:
        return ft.Text(txt, size=19, weight=ft.FontWeight.BOLD, color=PRIMARY)

    def _opts(items) -> List[ft.dropdown.Option]:
        return [ft.dropdown.Option(str(i)) for i in items]

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 0 — Data Setup
    # ══════════════════════════════════════════════════════════════════════════
    pol_status = ft.Text("No file loaded", color=MUTED, size=12)
    clm_status = ft.Text("No file loaded", color=MUTED, size=12)
    pol_fp = ft.FilePicker(); clm_fp = ft.FilePicker()
    page.overlay.extend([pol_fp, clm_fp])

    dd_pol_date = ft.Dropdown(label="Policy Date Column",   options=[], width=270, dense=True)
    dd_inc_date = ft.Dropdown(label="Incident Date Column", options=[], width=270, dense=True)
    dd_link     = ft.Dropdown(label="Join Field (Policy ↔ Claims)", options=[], width=300, dense=True)
    dd_peril    = ft.Dropdown(label="Peril / Cause Field (optional)", options=[], width=360, dense=True)
    tf_start    = ft.TextField(label="Start Date", hint_text="YYYY-MM-DD", width=180, dense=True)
    tf_end      = ft.TextField(label="End Date",   hint_text="YYYY-MM-DD", width=180, dense=True)

    def _refresh_dds():
        p = _opts(state.policy_cols); c = _opts(state.claims_cols)
        for dd in [dd_pol_date, dd_link]: dd.options = p
        dd_inc_date.options = c; dd_peril.options = c
        page.update()

    def _on_pol(e: ft.FilePickerResultEvent):
        if not e.files: return
        try:
            state.policy_df = load_file(e.files[0].path)
            pol_status.value = (f"✓ {e.files[0].name}  "
                                f"({len(state.policy_df):,} rows, {len(state.policy_df.columns)} cols)")
            pol_status.color = SUCCESS
            state.col_info = {}
            for col in state.policy_df.columns:
                dt = detect_col_type(state.policy_df[col])
                state.col_info[col] = ColInfo(col, dt, dt, DEFAULT_BINS,
                                               int(state.policy_df[col].nunique()))
            _refresh_dds()
        except Exception as ex:
            pol_status.value = f"Error: {ex}"; pol_status.color = DANGER; page.update()

    def _on_clm(e: ft.FilePickerResultEvent):
        if not e.files: return
        try:
            state.claims_df = load_file(e.files[0].path)
            clm_status.value = (f"✓ {e.files[0].name}  "
                                f"({len(state.claims_df):,} rows, {len(state.claims_df.columns)} cols)")
            clm_status.color = SUCCESS; _refresh_dds()
        except Exception as ex:
            clm_status.value = f"Error: {ex}"; clm_status.color = DANGER; page.update()

    pol_fp.on_result = _on_pol; clm_fp.on_result = _on_clm

    def _on_save(e):
        state.start_date = tf_start.value or ""; state.end_date = tf_end.value or ""
        state.policy_date_col   = dd_pol_date.value or ""
        state.incident_date_col = dd_inc_date.value or ""
        state.link_col  = dd_link.value  or ""; state.peril_col = dd_peril.value or ""
        if (state.claims_df is not None and state.peril_col
                and state.peril_col in state.claims_df.columns):
            state.perils = sorted([str(v) for v in
                                    state.claims_df[state.peril_col].dropna().unique()])
        else:
            state.perils = []
        state.merged_df = prepare_merged(state)
        n = len(state.merged_df) if state.merged_df is not None else 0
        snack(f"Saved. Merged data: {n:,} rows | Perils found: {len(state.perils)}")

    p0 = ft.Column([
        _h("1 — Data Setup"),
        ft.Text("Load policy and claims files, set date range, and configure the join field.", color=MUTED),
        ft.Divider(height=8),
        _card("Policy Exposure File", ft.Column([
            ft.ElevatedButton("Browse…", icon=ft.icons.FOLDER_OPEN,
                               on_click=lambda _: pol_fp.pick_files(
                                   allowed_extensions=["csv","xlsx","xls","parquet"])),
            pol_status,
        ]), icon=ft.icons.POLICY),
        _card("Claims File", ft.Column([
            ft.ElevatedButton("Browse…", icon=ft.icons.FOLDER_OPEN,
                               on_click=lambda _: clm_fp.pick_files(
                                   allowed_extensions=["csv","xlsx","xls","parquet"])),
            clm_status,
        ]), icon=ft.icons.RECEIPT_LONG),
        _card("Date Range", ft.Row([tf_start, ft.Text("to", color=MUTED), tf_end], spacing=12),
               icon=ft.icons.DATE_RANGE),
        _card("Field Mapping", ft.Column([
            ft.Row([dd_pol_date, dd_inc_date], spacing=12),
            ft.Row([dd_link, dd_peril], spacing=12),
        ]), icon=ft.icons.LINK),
        ft.ElevatedButton("Save Configuration & Prepare Data", icon=ft.icons.SAVE,
                           bgcolor=PRIMARY, color="white", on_click=_on_save, height=44),
    ], scroll=ft.ScrollMode.AUTO, spacing=8, expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Variable Types
    # ══════════════════════════════════════════════════════════════════════════
    types_col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=4, expand=True)

    def rebuild_types(_=None):
        types_col.controls.clear()
        if not state.col_info:
            types_col.controls.append(ft.Text("Load a policy file first.", color=MUTED))
            page.update(); return
        types_col.controls += [
            _h("2 — Variable Types"),
            ft.Text(f"Classify each of the {len(state.col_info)} policy columns. "
                    "Continuous columns are split into equal-percentile bins for GLM modelling.",
                    color=MUTED),
            ft.Divider(height=8),
            ft.Row([
                ft.Container(ft.Text("Column",   weight=ft.FontWeight.BOLD, size=12), width=185),
                ft.Container(ft.Text("Unique",   weight=ft.FontWeight.BOLD, size=12), width=60),
                ft.Container(ft.Text("Detected", weight=ft.FontWeight.BOLD, size=12), width=115),
                ft.Container(ft.Text("Type",     weight=ft.FontWeight.BOLD, size=12), width=185),
                ft.Container(ft.Text("Bins",     weight=ft.FontWeight.BOLD, size=12), width=75),
                ft.Text("Sample Values",          weight=ft.FontWeight.BOLD, size=12),
            ]),
            ft.Divider(height=1, color=BORDER),
        ]
        for col, info in state.col_info.items():
            sample = ""
            if state.policy_df is not None and col in state.policy_df.columns:
                sample = " | ".join(str(v) for v in state.policy_df[col].dropna().unique()[:4])
            type_dd = ft.Dropdown(
                options=_opts(["Categorical","Ordinal","Continuous","Exclude"]),
                value=info.col_type, dense=True, width=180)
            bins_tf = ft.TextField(value=str(info.n_bins), width=70, dense=True,
                                    disabled=(info.col_type != "Continuous"))

            def _td(c, dd, btf):
                def h(_):
                    state.col_info[c].col_type = dd.value
                    btf.disabled = (dd.value != "Continuous"); page.update()
                return h
            def _bd(c):
                def h(ev):
                    try: state.col_info[c].n_bins = max(2, int(ev.control.value))
                    except ValueError: pass
                return h

            type_dd.on_change = _td(col, type_dd, bins_tf)
            bins_tf.on_blur   = _bd(col)
            types_col.controls.append(ft.Row([
                ft.Container(ft.Text(col, size=12), width=185),
                ft.Container(ft.Text(str(info.unique_count), size=12, color=MUTED), width=60),
                ft.Container(ft.Text(info.detected_type, size=12, color=SEC), width=115),
                type_dd, bins_tf,
                ft.Text(sample, size=11, color=MUTED),
            ], spacing=6))
        page.update()

    p1 = ft.Column([
        ft.ElevatedButton("Refresh Column List", icon=ft.icons.REFRESH,
                           bgcolor=SEC, color="white", on_click=rebuild_types),
        ft.Divider(height=6),
        types_col,
    ], expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 2 — EDA
    # ══════════════════════════════════════════════════════════════════════════
    eda_dd   = ft.Dropdown(label="Predictor", options=[], width=290, dense=True)
    eda_rg   = ft.RadioGroup(
        ft.Row([ft.Radio(value="Frequency",    label="Frequency"),
                ft.Radio(value="Severity",     label="Severity"),
                ft.Radio(value="Burning Cost", label="Burning Cost")]),
        value="Frequency")
    eda_img  = ft.Image(src_base64="", width=1140, height=490, fit=ft.ImageFit.CONTAIN)
    eda_stat = ft.Text("Select a predictor and click Update Chart.", color=MUTED, size=12)

    def _eda_update(_=None):
        pred = eda_dd.value
        if not pred or state.merged_df is None:
            eda_stat.value = "Load and save data first."; page.update(); return
        eda_stat.value = "Computing…"; page.update()
        try:
            fig = compute_one_way(state.merged_df, pred, eda_rg.value, state.col_info)
            if fig:
                eda_img.src_base64 = _fig_to_b64(fig)
                eda_stat.value = f"{pred} → {eda_rg.value}"
            else:
                eda_stat.value = "Could not compute chart."
        except Exception as ex:
            eda_stat.value = f"Error: {ex}"
        page.update()

    def _eda_refresh(_=None):
        eda_dd.options = _opts(state.feature_cols); page.update()

    p2 = ft.Column([
        _h("3 — Exploratory Data Analysis"),
        ft.Text("One-way analysis of each predictor against frequency, severity, or burning cost.", color=MUTED),
        ft.Divider(height=8),
        ft.Row([
            eda_dd, eda_rg,
            ft.ElevatedButton("Update Chart", icon=ft.icons.BAR_CHART,
                               bgcolor=PRIMARY, color="white", on_click=_eda_update),
            ft.ElevatedButton("Refresh Predictors", icon=ft.icons.REFRESH,
                               on_click=_eda_refresh),
        ], spacing=12),
        eda_stat,
        ft.Divider(height=4),
        ft.Container(eda_img, bgcolor=CARD, padding=8, border_radius=8,
                     border=ft.border.all(1, BORDER)),
    ], scroll=ft.ScrollMode.AUTO, spacing=8, expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 3 — Model Exercises
    # ══════════════════════════════════════════════════════════════════════════
    ex_col = ft.Column(spacing=6)

    def rebuild_ex_col():
        ex_col.controls.clear()
        if not state.exercises:
            ex_col.controls.append(ft.Text("No exercises yet — click '+ Add Exercise'.", color=MUTED))
        for ex in state.exercises:
            ex_col.controls.append(_ex_card(ex))
        page.update()

    def _ex_card(ex: ExerciseConfig) -> ft.Container:
        ci  = ex.id % len(MODEL_COLS)
        p_s = (", ".join(ex.perils[:3]) + ("…" if len(ex.perils) > 3 else "")
               if ex.perils else "All perils")
        c_s = f"Cap={ex.cap}" if ex.cap is not None else "No cap"
        o_s = f"Collar={ex.collar}" if ex.collar is not None else "No collar"

        def _del(_, eid=ex.id):
            state.exercises = [x for x in state.exercises if x.id != eid]
            rebuild_ex_col()

        return ft.Container(
            ft.Row([
                ft.Container(width=7, height=68, bgcolor=MODEL_COLS[ci],
                             border_radius=ft.border_radius.only(top_left=8, bottom_left=8)),
                ft.Column([
                    ft.Text(ex.name, weight=ft.FontWeight.BOLD, size=13, color=TXT),
                    ft.Text(f"{ex.model_type}  |  Perils: {p_s}  |  "
                            f"{len(ex.features)} features  |  {c_s}, {o_s}",
                            size=11, color=MUTED),
                ], expand=True, spacing=2),
                ft.IconButton(ft.icons.DELETE_OUTLINE, icon_color=DANGER, on_click=_del),
            ], spacing=0),
            bgcolor=CARD, border_radius=8, border=ft.border.all(1, BORDER),
            padding=ft.padding.only(right=8, top=6, bottom=6),
            margin=ft.margin.only(bottom=4),
        )

    # ── Exercise dialog ───────────────────────────────────────────────────────
    ex_dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("New Modelling Exercise", weight=ft.FontWeight.BOLD, color=PRIMARY))
    page.overlay.append(ex_dlg)

    def _open_add(_=None):
        pid = state._ctr + 1
        name_tf   = ft.TextField(label="Exercise Name", value=f"Exercise {pid}",
                                  width=310, dense=True)
        type_dd   = ft.Dropdown(label="Model Type",
                                 options=_opts(["Freq+Sev","Burning Cost"]),
                                 value="Freq+Sev", width=220, dense=True)
        cap_tf    = ft.TextField(label="Cap (max prediction)",
                                  hint_text="e.g. 0.5  —  blank = none", width=220, dense=True)
        collar_tf = ft.TextField(label="Collar (min prediction)",
                                  hint_text="e.g. 0.0  —  blank = none", width=220, dense=True)

        p_cbs: Dict[str, ft.Checkbox] = {}
        p_list = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO, height=145)
        for p in state.perils:
            cb = ft.Checkbox(label=str(p), value=False)
            p_cbs[p] = cb; p_list.controls.append(cb)
        if not state.perils:
            p_list.controls.append(
                ft.Text("No peril field configured — all claims will be used.", color=MUTED, size=12))

        f_cbs: Dict[str, ft.Checkbox] = {}
        f_list = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO, height=190)
        for col in state.feature_cols:
            cb = ft.Checkbox(label=col, value=True)
            f_cbs[col] = cb; f_list.controls.append(cb)
        if not state.feature_cols:
            f_list.controls.append(
                ft.Text("No features found — save data configuration first.", color=MUTED, size=12))

        def _all_feats(_):
            for cb in f_cbs.values(): cb.value = True
            page.update()
        def _none_feats(_):
            for cb in f_cbs.values(): cb.value = False
            page.update()

        def _add(_=None):
            state.next_id()
            perils = [p for p, cb in p_cbs.items() if cb.value]
            feats  = [c for c, cb in f_cbs.items()  if cb.value]
            cap_v = collar_v = None
            try: cap_v    = float(cap_tf.value)    if cap_tf.value    else None
            except ValueError: pass
            try: collar_v = float(collar_tf.value) if collar_tf.value else None
            except ValueError: pass
            state.exercises.append(ExerciseConfig(
                id=pid, name=name_tf.value or f"Exercise {pid}",
                perils=perils, model_type=type_dd.value or "Freq+Sev",
                features=feats, cap=cap_v, collar=collar_v))
            ex_dlg.open = False; rebuild_ex_col()

        def _cancel(_=None):
            ex_dlg.open = False; page.update()

        ex_dlg.content = ft.Container(
            ft.Column([
                ft.Row([name_tf, type_dd], spacing=12),
                ft.Row([cap_tf, collar_tf], spacing=12),
                ft.Divider(),
                ft.Text("Peril Filter  (leave all unchecked → use all claims)",
                        weight=ft.FontWeight.BOLD, color=PRIMARY, size=12),
                p_list,
                ft.Divider(),
                ft.Row([
                    ft.Text("Features", weight=ft.FontWeight.BOLD, color=PRIMARY, size=12),
                    ft.TextButton("All", on_click=_all_feats),
                    ft.TextButton("None", on_click=_none_feats),
                ], spacing=8),
                f_list,
            ], spacing=10, scroll=ft.ScrollMode.AUTO),
            width=720, height=600,
        )
        ex_dlg.actions = [
            ft.TextButton("Cancel", on_click=_cancel),
            ft.ElevatedButton("Add Exercise", bgcolor=PRIMARY, color="white", on_click=_add),
        ]
        ex_dlg.open = True; page.update()

    p3 = ft.Column([
        _h("4 — Modelling Exercises"),
        ft.Text("Each exercise fits all relevant GLM distributions for the chosen model type.",
                color=MUTED),
        ft.Divider(height=8),
        ft.ElevatedButton("+ Add Exercise", icon=ft.icons.ADD_CIRCLE_OUTLINE,
                           bgcolor=PRIMARY, color="white", height=44, on_click=_open_add),
        ft.Divider(height=4),
        ex_col,
    ], scroll=ft.ScrollMode.AUTO, spacing=8, expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 4 — Results
    # ══════════════════════════════════════════════════════════════════════════
    run_btn  = ft.ElevatedButton("▶  Run All Exercises", bgcolor=ACCENT, color="white",
                                  icon=ft.icons.PLAY_CIRCLE, height=46)
    prog_bar = ft.ProgressBar(width=500, visible=False, color=ACCENT, bgcolor=BORDER)
    prog_txt = ft.Text("", color=MUTED, size=12)
    res_col  = ft.Column(spacing=12, scroll=ft.ScrollMode.AUTO, expand=True)

    def _render():
        res_col.controls.clear()
        if not state.exercise_results:
            res_col.controls.append(ft.Text("No results yet.", color=MUTED))
            page.update(); return

        for er in state.exercise_results:
            if er.error:
                res_col.controls.append(
                    _card(er.exercise_name, ft.Text(f"Error: {er.error}", color=DANGER)))
                continue

            items: List[ft.Control] = []

            # Best model banner
            best = er.best_result
            if best:
                items.append(ft.Container(
                    ft.Row([
                        ft.Icon(ft.icons.STAR, color="white", size=18),
                        ft.Text(f"Recommended: {best.dist_name} ({best.model_subtype})  "
                                f"Dev={best.deviance:.5f}  AIC={best.aic:.1f}  "
                                f"Gini={best.gini:.4f}  N={best.n_obs:,}",
                                color="white", weight=ft.FontWeight.BOLD, size=12),
                    ], spacing=8),
                    bgcolor=PRIMARY, padding=10, border_radius=8,
                ))

            # Metrics table
            def _tbl(results: List[ModelResult], title: str) -> ft.Control:
                if not results: return ft.Container()
                rows = []
                for r in results:
                    fmt = lambda v, p=5: (f"{v:.{p}f}" if not np.isnan(v) else "–")
                    rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text(r.dist_name, size=12,
                                            weight=ft.FontWeight.BOLD if r.is_best else None,
                                            color=ACCENT if r.is_best else TXT)),
                        ft.DataCell(ft.Text(fmt(r.deviance), size=12)),
                        ft.DataCell(ft.Text(fmt(r.aic, 1),   size=12)),
                        ft.DataCell(ft.Text(fmt(r.rmse),      size=12)),
                        ft.DataCell(ft.Text(fmt(r.mae),       size=12)),
                        ft.DataCell(ft.Text(fmt(r.gini, 4),  size=12)),
                        ft.DataCell(ft.Text(fmt(r.avg_pred),  size=12)),
                        ft.DataCell(ft.Text(fmt(r.avg_actual),size=12)),
                        ft.DataCell(ft.Text(str(r.n_obs), size=12)),
                        ft.DataCell(ft.Text(
                            "★ Best" if r.is_best else (r.error[:40] if r.error else "OK"),
                            color=(ACCENT if r.is_best else (DANGER if r.error else SUCCESS)),
                            size=12)),
                    ]))
                tbl = ft.DataTable(
                    columns=[ft.DataColumn(ft.Text(h, weight=ft.FontWeight.BOLD, size=11))
                              for h in ["Distribution","Deviance","AIC","RMSE","MAE",
                                        "Gini","Avg Pred","Avg Actual","N Obs","Status"]],
                    rows=rows,
                    heading_row_color="#EBF5FB",
                    border=ft.border.all(1, BORDER), border_radius=8, column_spacing=14,
                )
                return ft.Column([
                    ft.Text(title, weight=ft.FontWeight.BOLD, color=PRIMARY, size=12),
                    ft.Row([tbl], scroll=ft.ScrollMode.AUTO),
                ], spacing=6)

            if er.freq_results: items.append(_tbl(er.freq_results, "Frequency Models"))
            if er.sev_results:  items.append(_tbl(er.sev_results,  "Severity Models"))
            if er.bc_results:   items.append(_tbl(er.bc_results,   "Burning Cost Models"))

            # Comparison chart
            if er.comparison_chart_b64:
                items += [
                    ft.Divider(),
                    ft.Text("Model Comparison", weight=ft.FontWeight.BOLD, color=PRIMARY, size=12),
                    ft.Container(
                        ft.Image(src_base64=er.comparison_chart_b64, width=980, height=370,
                                 fit=ft.ImageFit.CONTAIN),
                        bgcolor=CARD, padding=8, border_radius=8, border=ft.border.all(1, BORDER)),
                ]

            # Diagnostic charts
            diags = [r for r in er.all_results if r.chart_b64]
            if diags:
                items += [ft.Divider(),
                          ft.Text("Diagnostic Charts", weight=ft.FontWeight.BOLD, color=PRIMARY, size=12)]
                for r in diags:
                    items.append(ft.Container(
                        ft.Column([
                            ft.Text(f"{r.dist_name} — {r.model_subtype}",
                                    weight=ft.FontWeight.BOLD,
                                    color="white" if r.is_best else PRIMARY, size=11),
                            ft.Image(src_base64=r.chart_b64, width=980, height=290,
                                     fit=ft.ImageFit.CONTAIN),
                        ], spacing=4),
                        bgcolor=PRIMARY if r.is_best else CARD,
                        padding=10, border_radius=8,
                        border=ft.border.all(2 if r.is_best else 1,
                                             "gold" if r.is_best else BORDER),
                        margin=ft.margin.only(bottom=6),
                    ))

            res_col.controls.append(
                _card(er.exercise_name, ft.Column(items, spacing=10), icon=ft.icons.ANALYTICS))

        page.update()

    def _on_run(e):
        if state.merged_df is None:
            snack("Configure and save data first (Step 1).", err=True); return
        if not state.exercises:
            snack("Add at least one exercise (Step 4).", err=True); return
        run_btn.disabled = True
        prog_bar.visible = True; prog_bar.value = 0
        res_col.controls.clear(); page.update()

        def _bg():
            state.exercise_results = []
            n = len(state.exercises)
            for i, ex in enumerate(state.exercises):
                def _upd(msg):
                    prog_txt.value = msg
                    prog_bar.value = (i + 0.5) / n
                    page.update()
                er = fit_exercise(ex, state, state.merged_df, on_progress=_upd)
                state.exercise_results.append(er)
            run_btn.disabled = False
            prog_bar.visible = False
            prog_txt.value   = f"✓ {n} exercise(s) complete."
            _render()

        threading.Thread(target=_bg, daemon=True).start()

    run_btn.on_click = _on_run

    p4 = ft.Column([
        _h("5 — Results & Model Comparison"),
        ft.Text("Run all exercises. Each exercise fits all applicable distributions "
                "and recommends the best by deviance.", color=MUTED),
        ft.Divider(height=8),
        ft.Row([run_btn, ft.Container(width=12), prog_bar], spacing=0),
        prog_txt,
        ft.Divider(height=4),
        res_col,
    ], spacing=8, expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Navigation & layout
    # ══════════════════════════════════════════════════════════════════════════
    nav_dests = [
        ("Data Setup",  ft.icons.STORAGE),
        ("Var. Types",  ft.icons.TABLE_CHART),
        ("EDA",         ft.icons.BAR_CHART),
        ("Exercises",   ft.icons.TUNE),
        ("Results",     ft.icons.ANALYTICS),
    ]

    for body in [p0, p1, p2, p3, p4]:
        panels.append(ft.Container(
            ft.Container(body,
                          padding=ft.padding.only(left=24, right=24, top=20, bottom=20),
                          expand=True),
            visible=False, expand=True,
        ))
    panels[0].visible = True

    def _on_nav(e):
        idx = e.control.selected_index
        _show(idx)
        if idx == 1: rebuild_types()
        if idx == 2: _eda_refresh()

    nav = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        bgcolor=PRIMARY,
        indicator_color=ACCENT,
        on_change=_on_nav,
        min_width=108,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icon(ic, color="white54"),
                selected_icon=ft.Icon(ic, color="white"),
                label_content=ft.Text(lbl, color="white70", size=10),
            )
            for lbl, ic in nav_dests
        ],
    )

    page.appbar = ft.AppBar(
        leading=ft.Icon(ft.icons.SHOW_CHART, color="white"),
        title=ft.Text("Insurance GLM Pricing Studio", color="white",
                      weight=ft.FontWeight.W_500),
        bgcolor=PRIMARY,
        actions=[ft.IconButton(ft.icons.INFO_OUTLINE, icon_color="white",
                               tooltip="v1.0 — Poisson | NegBin | Tweedie | Gamma | InvGaussian | Gaussian")],
    )

    page.add(ft.Row([
        nav,
        ft.VerticalDivider(width=1, color=BORDER),
        ft.Stack(panels, expand=True),
    ], expand=True, spacing=0))


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ft.run(target=main)