"""RQ2 step 5 (ML round 2, Tier 1 of ML_Options.md): strong baselines + probabilistic forecasts.

Same series, target, folds and ex-ante feature set as rq2_2_lightgbm.py (rolling origin, expanding
window, refit every 3 months / weekly 12, test from 1 Jan 2021), so every number is directly
comparable with round 1. All models are fitted on asinh(price); errors are reported in EUR/MWh.

Models (--models, default all):
  lear     LEAR-style benchmark (Lago et al. 2021): LASSO on standardised features, target asinh(price)
           as change vs same slot yesterday (--lear-target level: asinh(price) in levels as in the original;
           level LEAR was far worse than naive in a check on aFRR 4h and mFRR weekly), lagged prices also asinh-transformed, slot / weekday dummies, alpha chosen by
           time-series CV (LassoCV, TimeSeriesSplit) inside the training window. Point forecast.
           --lear-per-slot fits one model per 4h slot (the original LEAR has one model per hour).
  lgbm_q   LightGBM with quantile objective, one model per quantile level (delta target, like
           round 1). Median = point forecast.
  cat_q    CatBoost MultiQuantile (one model for all levels, native categorical handling).
  qrf      Quantile Regression Forest (quantile-forest package), median-imputed features.
  svr      RBF Support Vector Regression (Bezold et al. 2025), delta target, point forecast; trained
           on the last --svr-max training blocks to keep the O(n^2) cost bounded.
Quantile levels QS below. Crossing quantiles are sorted. The round-1 LightGBM point forecast
(Output/rq2_2_delta/preds_exante.parquet) is merged as `lgbm_l1` when the file exists.

Calibrated intervals: sequential conformalised quantile regression (CQR, Romano et al. 2019). For
every fold the 80 % / 90 % intervals are widened (or narrowed) by the conformity-score quantile of
ALL EARLIER test folds of the same series - those predictions are genuinely out of sample, so no
calibration data is taken from the training window. The first fold(s) with < 30 past scores are
not adjusted and are excluded from the CQR coverage figures (raw coverage is shown on the same rows).

Evaluation (common rows: every point model + all baselines non-missing):
  point   MAE, RMSE, sMAPE, MAE_asinh; skill vs b_prev_slot and vs LEAR; Diebold-Mariano (|error|)
          of every model vs LEAR, vs b_prev_slot and vs round-1 LightGBM
  prob    mean pinball loss over QS (EUR/MWh), CRPS approximation (2 x mean pinball), coverage and
          mean width of the 80 / 90 % intervals, raw and CQR; DM on pinball loss vs lgbm_q
  by year MAE per model and year (regime reporting)

    python Modelling/rq2_5_tier1.py --only "mFRR|up|week"            # quick check (~1 min)
    python Modelling/rq2_5_tier1.py                                   # all series, all models
    python Modelling/rq2_5_tier1.py --models lear lgbm_q --products fcr
Output: Modelling/Output/rq2_5/{preds.parquet, metrics_point.csv, metrics_prob.csv, metrics_by_year.csv,
        dm.csv, fold_log.csv, lear_nonzero.csv, lear_coefficients.csv, report_rq2_5.txt, run.log,
        run_meta.json, diagnostics/ (see diagnostics/README.txt)}
A macOS notification is sent when the run ends (run_utils.py).
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import run_utils as RU  # run.log, run_meta.json, run history, macOS notification

import rq2_common as C
from rq2_2_lightgbm import feature_list, make_X

warnings.filterwarnings("ignore")

QS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
QTAG = [f"{int(round(q * 100)):02d}" for q in QS]
MED = int(np.flatnonzero(np.isclose(QS, 0.5))[0])
INTERVALS = {80: ("10", "90"), 90: ("05", "95")}
MODELS = ["lear", "lgbm_q", "cat_q", "qrf", "svr"]
PROB_MODELS = ["lgbm_q", "cat_q", "qrf"]
LABEL = {"lear": "LEAR (LASSO)", "lgbm_l1": "LightGBM L1 (round 1)", "lgbm_q": "LightGBM quantile",
         "cat_q": "CatBoost MultiQuantile", "qrf": "Quantile Regression Forest", "svr": "SVR (RBF)"}


# ---------------------------------------------------------------- feature matrices
def tree_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """categories -> integer codes (NaN for missing), everything float."""
    Z = X.copy()
    for c in Z.columns:
        if isinstance(Z[c].dtype, pd.CategoricalDtype):
            Z[c] = Z[c].cat.codes.replace(-1, np.nan)
    return Z.astype("float64")


def linear_frame(X: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for LEAR / SVR: numeric columns, asinh of lagged prices, one-hot categories,
    slot and weekday dummies."""
    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num = X.drop(columns=cat).astype("float64")
    pc = [c for c in num.columns if c.startswith("tgt_prev") and "price" in c]
    num[pc] = np.arcsinh(num[pc])
    parts = [num]
    if cat:
        parts.append(pd.get_dummies(X[cat], dtype="float64"))
    loc = g["block_start_utc"].dt.tz_convert(C.TZ)
    parts.append(pd.get_dummies(g["slot"].astype(str), prefix="slot", dtype="float64").set_index(X.index))
    parts.append(pd.get_dummies(loc.dt.dayofweek.astype(str), prefix="dow", dtype="float64").set_index(X.index))
    return pd.concat(parts, axis=1)


def prep(Ztr: pd.DataFrame, Zte: pd.DataFrame, scale: bool):
    """Median-impute with training medians, drop all-NaN / constant training columns, optionally
    standardise with training mean / std."""
    med = Ztr.median()
    keep = med.notna() & (Ztr.std() > 1e-12)
    A, B = Ztr.loc[:, keep].fillna(med[keep]), Zte.loc[:, keep].fillna(med[keep])
    if scale:                                           # linear models: clip test features to the
        B = B.clip(A.min(), A.max(), axis=1)            # training range (no extrapolation after breaks)
        mu, sd = A.mean(), A.std().replace(0, 1.0)
        A, B = (A - mu) / sd, (B - mu) / sd
    return A.to_numpy(np.float64), B.to_numpy(np.float64), list(A.columns)


# ---------------------------------------------------------------- models
def _share(v, names) -> pd.Series:
    v = pd.Series(np.abs(np.asarray(v, float)), index=list(names))
    return v / v.sum() if v.sum() > 0 else v


def fit_lear(Atr, ytr, Ate, a, names):
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import TimeSeriesSplit
    kw = dict(cv=TimeSeriesSplit(n_splits=4), max_iter=5000, tol=1e-3, n_jobs=a.threads or None)
    try:
        m = LassoCV(alphas=a.lear_alphas, **kw)
    except TypeError:                                   # older scikit-learn: n_alphas
        m = LassoCV(n_alphas=a.lear_alphas, **kw)
    m.fit(Atr, ytr)
    coef = pd.Series(m.coef_, index=list(names))
    return np.clip(m.predict(Ate), ytr.min(), ytr.max()), {"nonzero": int((m.coef_ != 0).sum()),
                                                           "alpha": float(m.alpha_), "coef": coef}


def _lgb_params(n, a, seed, q):
    return dict(objective="quantile", alpha=float(q), learning_rate=a.lr, num_leaves=a.leaves,
                min_data_in_leaf=max(5, min(20, n // 30)), feature_fraction=a.colsample,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed,
                num_threads=a.threads)


def fit_lgbm_q(Xtr, ytr, Xte, a, seed):
    n = len(Xtr); nv = max(10, int(0.15 * n))
    out, iters, imp = np.empty((len(Xte), len(QS))), {}, None
    for k, q in enumerate(QS):
        p = _lgb_params(n, a, seed, q)
        m0 = lgb.train(p, lgb.Dataset(Xtr.iloc[:-nv], ytr[:-nv]), num_boost_round=a.rounds,
                       valid_sets=[lgb.Dataset(Xtr.iloc[-nv:], ytr[-nv:])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        best = max(20, int((m0.best_iteration or 20) * 1.15))
        mk = lgb.train(p, lgb.Dataset(Xtr, ytr), num_boost_round=best)
        out[:, k] = mk.predict(Xte); iters[QTAG[k]] = best
        if k == MED:
            imp = _share(mk.feature_importance("gain"), Xtr.columns)
    return out, {"best_iter_q50": iters["50"], "best_iter_q05": iters["05"], "best_iter_q95": iters["95"], "imp": imp}


def _cat_frame(X: pd.DataFrame):
    Z = X.copy()
    ci = [i for i, c in enumerate(Z.columns) if isinstance(Z[c].dtype, pd.CategoricalDtype)]
    for i in ci:
        c = Z.columns[i]
        Z[c] = np.where(Z[c].isna(), "NA", Z[c].astype(object).astype(str)).astype(object)
    return Z, ci


def fit_cat_q(Xtr, ytr, Xte, a, seed):
    from catboost import CatBoostRegressor, Pool
    Ztr, ci = _cat_frame(Xtr); Zte, _ = _cat_frame(Xte)
    n = len(Ztr); nv = max(10, int(0.15 * n))
    kw = dict(loss_function="MultiQuantile:alpha=" + ",".join(f"{q:g}" for q in QS),
              learning_rate=a.cat_lr, depth=a.cat_depth, rsm=a.colsample if a.colsample < 1 else None,
              random_seed=seed, verbose=0, allow_writing_files=False,
              thread_count=a.threads if a.threads > 0 else -1)
    m0 = CatBoostRegressor(iterations=a.cat_iters, od_type="Iter", od_wait=50, **kw)
    m0.fit(Pool(Ztr.iloc[:-nv], ytr[:-nv], cat_features=ci),
           eval_set=Pool(Ztr.iloc[-nv:], ytr[-nv:], cat_features=ci), use_best_model=True)
    best = max(20, int((m0.get_best_iteration() or 20) * 1.15))
    m = CatBoostRegressor(iterations=best, **kw)
    m.fit(Pool(Ztr, ytr, cat_features=ci))
    return np.asarray(m.predict(Pool(Zte, cat_features=ci))).reshape(len(Zte), len(QS)), \
        {"best_iter": best, "imp": _share(m.get_feature_importance(), Ztr.columns)}


def fit_qrf(Atr, ytr, Ate, a, seed, names):
    from quantile_forest import RandomForestQuantileRegressor
    m = RandomForestQuantileRegressor(n_estimators=a.qrf_trees, min_samples_leaf=a.qrf_leaf,
                                      max_features=a.colsample, n_jobs=-1 if a.threads <= 0 else a.threads,
                                      random_state=seed)
    m.fit(Atr, ytr)
    return np.asarray(m.predict(Ate, quantiles=list(QS))).reshape(len(Ate), len(QS)), \
        {"imp": _share(m.feature_importances_, names)}


def fit_svr(Atr, ytr, Ate, a):
    from sklearn.svm import SVR
    if len(Atr) > a.svr_max:
        Atr, ytr = Atr[-a.svr_max:], ytr[-a.svr_max:]
    m = SVR(kernel="rbf", C=a.svr_c, epsilon=a.svr_eps, gamma="scale", cache_size=1000)
    m.fit(Atr, ytr)
    return np.clip(m.predict(Ate), ytr.min(), ytr.max()), {"n_support": int(len(m.support_)), "n_train_used": len(Atr)}


# ---------------------------------------------------------------- evaluation helpers
def pinball(y, q, lvl):
    d = y - q
    return np.maximum(lvl * d, (lvl - 1) * d)


def dm_loss(d, lags):
    """Diebold-Mariano on a loss differential d (model 1 loss - model 2 loss), Newey-West."""
    d = np.asarray(d, float); n = len(d)
    if n < 30:
        return float("nan"), float("nan")
    d0 = d - d.mean(); v = d0 @ d0 / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (d0[l:] @ d0[:-l]) / n
    if v <= 0:
        return float("nan"), float("nan")
    st = d.mean() / math.sqrt(v / n)
    return st, math.erfc(abs(st) / math.sqrt(2))


def cqr(P: pd.DataFrame, m: str, lo_tag: str, hi_tag: str, level: int, min_scores: int = 30):
    """Sequential CQR: adjust each fold with the conformity scores of all earlier folds (asinh space).
    Returns (lo, hi) in EUR/MWh, NaN where the fold had too few past scores."""
    lo = np.arcsinh(P[f"q_{m}_{lo_tag}"].to_numpy(float)); hi = np.arcsinh(P[f"q_{m}_{hi_tag}"].to_numpy(float))
    y = np.arcsinh(P["y"].to_numpy(float)); fold = P["fold"].to_numpy()
    out_lo, out_hi, past = np.full(len(P), np.nan), np.full(len(P), np.nan), []
    for f in np.unique(fold):
        idx = fold == f
        s_all = np.concatenate(past) if past else np.empty(0)
        if len(s_all) >= min_scores:
            n = len(s_all)
            qhat = np.quantile(s_all, min(1.0, math.ceil((n + 1) * level / 100) / n), method="higher")
            out_lo[idx], out_hi[idx] = lo[idx] - qhat, hi[idx] + qhat
        s = np.maximum(lo[idx] - y[idx], y[idx] - hi[idx])
        past.append(s[~np.isnan(s)])
    return np.sinh(out_lo), np.sinh(out_hi)


# ---------------------------------------------------------------- main loop
def point_order_all(models, P) -> list[str]:
    order = [m for m in ["lear", "lgbm_l1", "lgbm_q", "cat_q", "qrf", "svr"] if m in models or m == "lgbm_l1"]
    return [m for m in order if f"p_{m}" in P and P[f"p_{m}"].notna().any()]


def run_series(g: pd.DataFrame, Xg: pd.DataFrame, s: str, a, log) -> pd.DataFrame | None:
    ya = np.arcsinh(g["y"].to_numpy(float))
    off = np.arcsinh(g["b_prev_slot"].to_numpy(float))
    yd = ya - off                                       # delta target (tree models, SVR)
    ok_d = ~np.isnan(yd)
    offl = off if a.lear_target == "delta" else np.zeros(len(ya))
    yl = ya - offl
    ok_l = ~np.isnan(yl)
    Zt = tree_numeric(Xg) if "qrf" in a.models else None
    Zl = linear_frame(Xg, g) if {"lear", "svr"} & set(a.models) else None
    rm = a.refit_months_week if s.endswith("|week") else a.refit_months
    mt = C.MIN_TRAIN[s.split("|")[2]]
    keys = ["series", "product", "direction", "procurement", "auction_id", "block_start_utc", "slot", "y", *C.BASELINES,
            *[c for c in g.columns if c.startswith("regime_")]]
    rows, nz, timers = [], [], {m: 0.0 for m in a.models}
    imps: dict[str, list] = {m: [] for m in a.models}       # per model: list of per-fold importance shares
    flog: list[dict] = []                                    # one row per fold: sizes, timings, tuning results
    for f, start, tr_all, te in C.fold_plan(g, rm, mt):
        trd, trl = tr_all[ok_d[tr_all]], tr_all[ok_l[tr_all]]
        if len(trd) < mt:
            continue
        r = g.iloc[te][keys].copy(); r["fold"] = f; r["n_train"] = len(trd)
        fl = {"series": s, "fold": f, "fold_start": start.date(), "train_from": g["block_start_utc"].iloc[trd].min().date(),
              "test_from": g["block_start_utc"].iloc[te].min().date(), "test_to": g["block_start_utc"].iloc[te].max().date(),
              "n_train": len(trd), "n_test": len(te)}
        for m in a.models:
            t0 = time.time()
            if m == "lear":
                if a.lear_per_slot and s.split("|")[2] == "4h":
                    p = np.full(len(te), np.nan); k_nz = []
                    for sl in np.unique(g["slot"].to_numpy()[te]):
                        tr_s = trl[g["slot"].to_numpy()[trl] == sl]; te_s = te[g["slot"].to_numpy()[te] == sl]
                        A, B, cols = prep(Zl.iloc[tr_s], Zl.iloc[te_s], scale=True)
                        ps, inf = fit_lear(A, yl[tr_s], B, a, cols)
                        p[np.isin(te, te_s)] = ps; k_nz.append(inf["nonzero"])
                        nz.append({"series": s, "fold": f, "slot": sl, "n_train": len(tr_s), "nonzero": inf["nonzero"],
                                   "alpha": inf["alpha"]})
                    fl["lear_nonzero"] = float(np.mean(k_nz))
                else:
                    A, B, cols = prep(Zl.iloc[trl], Zl.iloc[te], scale=True)
                    p, inf = fit_lear(A, yl[trl], B, a, cols)
                    imps["lear"].append(_share(inf["coef"], inf["coef"].index))
                    nz.append({"series": s, "fold": f, "slot": "all", "n_train": len(trl), "nonzero": inf["nonzero"],
                               "alpha": inf["alpha"], "n_candidates": len(cols)})
                    nz[-1].update({f"coef__{c}": v for c, v in inf["coef"][inf["coef"] != 0].items()})
                    fl.update(lear_nonzero=inf["nonzero"], lear_alpha=inf["alpha"])
                r["p_lear"] = np.sinh(p + offl[te])
            elif m == "svr":
                A, B, _ = prep(Zl.iloc[trd], Zl.iloc[te], scale=True)
                ps, inf = fit_svr(A, yd[trd], B, a)
                r["p_svr"] = np.sinh(ps + off[te]); fl.update({f"svr_{k}": v for k, v in inf.items()})
            else:
                Q = np.zeros((len(te), len(QS)))
                for sd in range(a.seeds):
                    if m == "lgbm_q":
                        q_, inf = fit_lgbm_q(Xg.iloc[trd], yd[trd], Xg.iloc[te], a, seed=1 + sd)
                    elif m == "cat_q":
                        q_, inf = fit_cat_q(Xg.iloc[trd], yd[trd], Xg.iloc[te], a, seed=1 + sd)
                    else:
                        A, B, cols = prep(Zt.iloc[trd], Zt.iloc[te], scale=False)
                        q_, inf = fit_qrf(A, yd[trd], B, a, seed=1 + sd, names=cols)
                    Q += q_
                    if sd == 0:
                        if inf.get("imp") is not None:
                            imps[m].append(inf["imp"])
                        fl.update({f"{m}_{k}": v for k, v in inf.items() if k != "imp"})
                raw = Q / a.seeds
                fl[f"{m}_crossing_share"] = float((np.diff(raw, axis=1) < 0).any(axis=1).mean())
                Q = np.sort(raw, axis=1) + off[te][:, None]
                for k, t in enumerate(QTAG):
                    r[f"q_{m}_{t}"] = np.sinh(Q[:, k])
                r[f"p_{m}"] = r[f"q_{m}_50"]
            timers[m] += time.time() - t0
            fl[f"{m}_seconds"] = round(time.time() - t0, 1)
        rows.append(r); flog.append(fl)
    log(f"   {s:16s} folds={len(rows):3d}  " + "  ".join(f"{m} {t:5.0f}s" for m, t in timers.items()))
    if not rows:
        return None, nz, imps, flog
    return pd.concat(rows), nz, imps, flog


def evaluate(P: pd.DataFrame, s: str, point_models: list[str], prob_models: list[str]):
    L = 4 if s.endswith("|week") else 12
    cols = [f"p_{m}" for m in point_models]
    E = P.dropna(subset=["y", *cols, *C.BASELINES]).sort_values("block_start_utc").copy()
    E["year"] = E["block_start_utc"].dt.tz_convert(C.TZ).dt.year
    mp, my, dm, mq = [], [], [], []
    names = {**{f"p_{m}": m for m in point_models}, **{b: b for b in C.BASELINES}}
    for col, nm in names.items():
        mp.append({"series": s, "model": nm, **C.metrics(E["y"], E[col])})
        for yr, gy in E.groupby("year"):
            my.append({"series": s, "model": nm, "year": yr, **C.metrics(gy["y"], gy[col])})
    y = E["y"].to_numpy(float)
    refs = [r for r in ["lear", "b_prev_slot", "lgbm_l1"] if r in point_models or r in C.BASELINES]
    for m in point_models:
        e1 = np.abs(E[f"p_{m}"].to_numpy(float) - y)
        for ref in refs:
            if ref == m:
                continue
            rc = ref if ref in C.BASELINES else f"p_{ref}"
            st, p = dm_loss(e1 - np.abs(E[rc].to_numpy(float) - y), L)
            dm.append({"series": s, "loss": "abs", "model": m, "vs": ref, "stat": st, "p": p})
    for m in prob_models:
        pl = np.mean([pinball(y, E[f"q_{m}_{t}"].to_numpy(float), q) for q, t in zip(QS, QTAG)], axis=0)
        row = {"series": s, "model": m, "n": len(E), "pinball": pl.mean(), "CRPS_approx": 2 * pl.mean()}
        for lev, (lo_t, hi_t) in INTERVALS.items():
            lo, hi = E[f"q_{m}_{lo_t}"].to_numpy(float), E[f"q_{m}_{hi_t}"].to_numpy(float)
            clo, chi = E[f"cqr{lev}_lo_{m}"].to_numpy(float), E[f"cqr{lev}_hi_{m}"].to_numpy(float)
            h = ~np.isnan(clo)
            row[f"cov{lev}_raw"] = np.mean((y[h] >= lo[h]) & (y[h] <= hi[h])) * 100 if h.any() else np.nan
            row[f"cov{lev}_cqr"] = np.mean((y[h] >= clo[h]) & (y[h] <= chi[h])) * 100 if h.any() else np.nan
            row[f"width{lev}_raw"] = np.mean(hi[h] - lo[h]) if h.any() else np.nan
            row[f"width{lev}_cqr"] = np.mean(chi[h] - clo[h]) if h.any() else np.nan
            row[f"n_cqr{lev}"] = int(h.sum())
        mq.append(row)
        E[f"_pl_{m}"] = pl
    if "lgbm_q" in prob_models:
        for m in prob_models:
            if m != "lgbm_q":
                st, p = dm_loss(E[f"_pl_{m}"] - E["_pl_lgbm_q"], L)
                dm.append({"series": s, "loss": "pinball", "model": m, "vs": "lgbm_q", "stat": st, "p": p})
    return mp, my, dm, mq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=C.VIEWS)
    ap.add_argument("--only", default=None, help='series filter, e.g. "mFRR|up|4h"')
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--refit-months-week", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1, help="average tree models over n seeds")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--colsample", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--cat-iters", type=int, default=1500)
    ap.add_argument("--cat-lr", type=float, default=0.05)
    ap.add_argument("--cat-depth", type=int, default=6)
    ap.add_argument("--qrf-trees", type=int, default=300)
    ap.add_argument("--qrf-leaf", type=int, default=10)
    ap.add_argument("--lear-alphas", type=int, default=30)
    ap.add_argument("--lear-target", default="delta", choices=["level", "delta"],
                    help="delta: change vs same slot yesterday (default, like the tree models); level: asinh(price) as in the original LEAR")
    ap.add_argument("--lear-per-slot", action="store_true", help="one LEAR model per 4h slot")
    ap.add_argument("--svr-max", type=int, default=8000)
    ap.add_argument("--svr-c", type=float, default=1.0)
    ap.add_argument("--svr-eps", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else C.ROOT / "Modelling" / "Output" / "rq2_5"
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    lines: list[str] = []
    log = lambda x: print(x, flush=True)

    r1_path = C.ROOT / "Modelling" / "Output" / "rq2_2_delta" / "preds_exante.parquet"
    r1 = pd.read_parquet(r1_path, columns=["series", "block_start_utc", "pred"]).drop_duplicates(
        ["series", "block_start_utc"]).rename(columns={"pred": "p_lgbm_l1"}) if r1_path.exists() else None

    t_all = time.time()
    allP, MP, MY, DM, MQ, NZ, FLOG, IMP = [], [], [], [], [], [], [], {}
    for view in a.products:
        df = C.load_view(view)
        feats, _ = feature_list(df, view, "exante")
        X = make_X(df, feats)
        log(f"{view}: {len(df):,} blocks, {len(feats)} features")
        for s in df["series"].unique():
            if a.only and a.only != s:
                continue
            gi = np.flatnonzero((df["series"] == s).to_numpy())
            g, Xg = df.iloc[gi].reset_index(drop=True), X.iloc[gi].reset_index(drop=True)
            P, nz, imps, flog = run_series(g, Xg, s, a, log)
            NZ += nz; FLOG += flog
            for m, lst in imps.items():
                IMP.setdefault(m, {})[s] = lst
            if P is None:
                lines.append(f"== {s}: no fold with >= {C.MIN_TRAIN[s.split('|')[2]]} training blocks"); continue
            point = [m for m in a.models]
            if r1 is not None:
                P = P.merge(r1, on=["series", "block_start_utc"], how="left")
                if P["p_lgbm_l1"].notna().any():
                    point.append("lgbm_l1")
            prob = [m for m in PROB_MODELS if m in a.models]
            P = P.sort_values("block_start_utc").reset_index(drop=True)
            for m in prob:
                for lev, (lo_t, hi_t) in INTERVALS.items():
                    P[f"cqr{lev}_lo_{m}"], P[f"cqr{lev}_hi_{m}"] = cqr(P, m, lo_t, hi_t, lev)
            allP.append(P)
            mp, my, dm, mq = evaluate(P, s, point, prob)
            MP += mp; MY += my; DM += dm; MQ += mq

    if not MP:
        print("nothing evaluated"); return
    MP, MY, DM, MQ = map(pd.DataFrame, (MP, MY, DM, MQ))
    pd.concat(allP).to_parquet(out / "preds.parquet")
    MP.to_csv(out / "metrics_point.csv", index=False); MY.to_csv(out / "metrics_by_year.csv", index=False)
    DM.to_csv(out / "dm.csv", index=False); MQ.to_csv(out / "metrics_prob.csv", index=False)
    if NZ:
        NZd = pd.DataFrame(NZ)
        cc = [c for c in NZd.columns if c.startswith("coef__")]
        NZd.drop(columns=cc).to_csv(out / "lear_nonzero.csv", index=False)            # alpha and #non-zero per fold
        if cc:
            LC = NZd[["series", "fold", *cc]].melt(["series", "fold"], var_name="feature", value_name="coef")
            LC["feature"] = LC["feature"].str.removeprefix("coef__"); LC["coef"] = LC["coef"].fillna(0.0)
            nf = NZd.groupby("series")["fold"].nunique()
            LS = LC.groupby(["series", "feature"])["coef"].agg(coef_mean="mean", coef_std="std",
                                                               selected_folds=lambda x: (x != 0).sum()).reset_index()
            LS["selected_share"] = LS["selected_folds"] / LS["series"].map(nf)
            LS["abs_mean"] = LS["coef_mean"].abs()
            LS.sort_values(["series", "abs_mean"], ascending=[True, False]).to_csv(out / "lear_coefficients.csv", index=False)
    pd.DataFrame(FLOG).to_csv(out / "fold_log.csv", index=False)
    import ml_diagnostics as D
    imp_tables = {m: D.importance_table(d) for m, d in IMP.items() if any(d.values())}
    Pall = pd.concat(allP)
    models = {m: f"p_{m}" for m in point_order_all(a.models, Pall)} | {b: b for b in C.BASELINES}
    quant = {m: {q: f"q_{m}_{t}" for q, t in zip(QS, QTAG)} for m in PROB_MODELS if m in a.models}
    D.forecast_diagnostics(Pall, models, out, baseline="b_prev_slot", quantiles=quant,
                           importance={f"{m}": t for m, t in imp_tables.items()},
                           group_cols=[c for c in Pall.columns if c.startswith("regime_")], title="RQ2 Tier 1")

    # ---------------------------------------------------------------- report
    hdr = ["RQ2 9.2 round 2 (Tier 1, ML_Options.md) - strong baselines and probabilistic forecasts.",
           f"Rolling origin, expanding window, refit every {a.refit_months} months (weekly {a.refit_months_week}); "
           "test from 2021-01-01; errors in EUR/MWh on the common rows of all point models and baselines.",
           f"Models: {', '.join(LABEL.get(m, m) for m in a.models)}; quantiles {', '.join(QTAG)}; seeds {a.seeds}; "
           f"LEAR target {a.lear_target}{', per slot' if a.lear_per_slot else ''}.",
           "lgbm_l1 = round-1 LightGBM (rq2_2_delta), merged for comparison.", ""]
    point_order = [m for m in ["lear", "lgbm_l1", "lgbm_q", "cat_q", "qrf", "svr"] if m in set(MP.model)]
    for s, g in MP.groupby("series", sort=False):
        g = g.set_index("model")
        order = [m for m in point_order if m in g.index] + C.BASELINES
        hdr.append(f"== {s}   n = {int(g['n'].iloc[0]):,}")
        T = g.loc[order, ["MAE", "RMSE", "sMAPE_%", "MAE_asinh"]].copy()
        T["skill_vs_prev_slot_%"] = (1 - T["MAE"] / g.at["b_prev_slot", "MAE"]) * 100
        if "lear" in g.index:
            T["skill_vs_LEAR_%"] = (1 - T["MAE"] / g.at["lear", "MAE"]) * 100
        hdr.append(C.fmt(T))
        D = DM[(DM.series == s) & (DM.loss == "abs")]
        if len(D):
            hdr.append("   Diebold-Mariano |error| (stat < 0: row model better), p-values:")
            hdr.append("   " + C.fmt(D.pivot(index="model", columns="vs", values="stat").round(2)).replace("\n", "\n   "))
            hdr.append("   " + C.fmt(D.pivot(index="model", columns="vs", values="p")).replace("\n", "\n   "))
        Q = MQ[MQ.series == s].set_index("model") if len(MQ) else pd.DataFrame()
        if len(Q):
            hdr.append("   Probabilistic (pinball / CRPS in EUR/MWh; coverage %, width EUR/MWh on CQR-eligible rows):")
            hdr.append("   " + C.fmt(Q.drop(columns=["series", "n"])).replace("\n", "\n   "))
            Dp = DM[(DM.series == s) & (DM.loss == "pinball")]
            if len(Dp):
                hdr.append("   DM pinball vs lgbm_q: " + "; ".join(
                    f"{r.model} stat {r.stat:.2f} p {r.p:.4f}" for r in Dp.itertuples()))
        Y = MY[MY.series == s].pivot(index="year", columns="model", values="MAE")
        hdr.append("   MAE by year:")
        hdr.append("   " + C.fmt(Y[[c for c in order if c in Y.columns]]).replace("\n", "\n   "))
        hdr.append("")
    S = MP.pivot(index="series", columns="model", values="MAE")
    sk = pd.DataFrame({m: (1 - S[m] / S["b_prev_slot"]) * 100 for m in point_order}, index=S.index)
    hdr.append("== Summary: MAE skill vs same-slot-yesterday (%)")
    hdr.append(C.fmt(sk))
    hdr.append("   best point model per series: " + "; ".join(f"{i}: {S.loc[i, point_order].idxmin()}" for i in S.index))
    if len(MQ):
        hdr.append("== Summary: mean pinball loss (EUR/MWh), lower is better")
        hdr.append(C.fmt(MQ.pivot(index="series", columns="model", values="pinball")))
        hdr.append("== Summary: 90 % interval coverage raw -> CQR (%)")
        cv = MQ.assign(c=lambda d: d["cov90_raw"].round(1).astype(str) + " -> " + d["cov90_cqr"].round(1).astype(str))
        hdr.append(cv.pivot(index="series", columns="model", values="c").to_string())
    for m, t in imp_tables.items():
        if len(t) == 0: continue
        hdr.append(f"== Top features: {LABEL.get(m, m)} (mean importance share over folds; % of folds in top 20)")
        for s_, g_ in t.groupby("series", sort=False):
            g_ = g_[g_["mean"] > 0]
            hdr.append(f"   {s_}: " + ("; ".join(f"{r.feature} {r.mean*100:.1f}% ({r.top20_share_of_folds:.0%})"
                                                for r in g_.head(8).itertuples())
                                      or "no feature selected in any fold (model = same slot yesterday + constant)"))
    hdr.append("\nMore detail: fold_log.csv (per fold sizes / timings / tuning), lear_coefficients.csv (signed LEAR "
               "coefficients), diagnostics/ (breakdowns, worst errors, calibration, importance, plots; see its README.txt)")
    for i_ in S.index:
        b_ = S.loc[i_, point_order].idxmin()
        RU.summary(f"{i_.replace('|', ' ')}: {b_} {(1 - S.loc[i_, b_] / S.loc[i_, 'b_prev_slot']) * 100:+.1f}%")
    hdr += lines
    hdr.append(f"\ntotal runtime {time.time() - t_all:.0f}s")
    (out / "report_rq2_5.txt").write_text("\n".join(hdr))
    print("\n".join(hdr))


if __name__ == "__main__":
    RU.run(main, "RQ2 Tier 1 models")
