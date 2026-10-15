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

Hyperparameter search (--tune N, done_todo 2.2 item 5; default 0 = fixed defaults, identical to round 2):
  Equal budget: every tuned model evaluates the SAME number N of configurations at every tuning point -
  config 0 = the round-2 defaults, configs 1..N-1 random draws from the model's search space (SPACE below;
  the draw is fixed per series and model, --tune-seed). Tuning points: the first fold and then every
  --tune-every-months (default 12: yearly for the 4h series, every fold for the weekly series); the chosen
  configuration is kept until the next tuning point. Validation is time-aware and INSIDE the training
  window: the last --tune-val (20 %) of the training rows, optionally cut into --tune-splits consecutive
  chunks (expanding). Score: mean pinball loss at 10 / 50 / 90 % in asinh-delta space (SVR: MAE). LEAR:
  LassoCV over N alphas on the same split(s). Boosting iterations are not searched (early stopping as before);
  QRF is tuned with --tune-qrf-trees trees. Output: tuning_log.csv (every configuration, score, chosen,
  gain vs default), hp columns in fold_log.csv, tuned_vs_default.csv (DM tuned vs the round-2 predictions
  in Output/rq2_5_<product>, --compare-default).
Validation scheme (--tune-cv, done_todo 2.2 item 5b): train / validation / test.
  Test   = the next fold (3 months, weekly 12), never used for any choice.
  Train / validation = split of the fold's training window only, rows in time order (never shuffled):
    holdout  train = first (1 - tune_val) of the window, validation = last tune_val (default 20 %)
             (= the 14 Oct run; --tune-splits > 1 cuts the validation part into chunks).
    cv       full time-series cross-validation over the WHOLE training window: the window is cut into
             --cv-folds + 1 consecutive chunks; split k trains on chunks 1..k and validates on chunk k+1
             (expanding, like sklearn TimeSeriesSplit). Score = mean over the --cv-folds splits.
    auto     cv for small series (<= --cv-max-rows blocks in total: all weekly series, aFRR 4h),
             holdout for the large ones (FCR 4h, mFRR 4h), where the data allows a plain split and
             full CV would cost --cv-folds times the runtime.
  After the search the chosen configuration is refitted on the full training window and predicts the test fold.
  LEAR uses the same splits for its alpha (LassoCV). The scheme per fold is logged (fold_log.csv: val_scheme,
  n_val_splits; tuning_log.csv: val_scheme).
Seeds (--seeds n, item 6): tree models averaged over n seeds; with n > 1 the per-seed medians are kept as
  p_<model>_seed<k> and seed_spread.csv / seed_mae.csv give the MAE of every seed, the spread and the gain
  of the seed average (DM seed average vs seed 1). Tuning always uses seed 1.

Training window (--window, done_todo 2.2 item 7 / Tier 3 regime handling; default expanding = identical to round 2):
  expanding  all blocks delivered before the fold start (round 2).
  sliding    only blocks that start in the last --window-months (4h, default 24) / --window-months-week (weekly,
             default 48) before the fold start; never fewer than MIN_TRAIN valid blocks (then the last MIN_TRAIN).
  regime     only blocks since the latest structural break on or before the fold start, once that sub-window holds
             >= --regime-min valid blocks (0 = MIN_TRAIN); before that the expanding window is kept (the regime
             flags stay in the features either way). Breaks = first block where one of --break-flags takes a new
             value (default: market-design flags of reserve procurement, announced in advance -> ex ante) plus
             --break-extra dates (e.g. the crisis 2021-09-01 / 2023-07-01, chosen ex post -> robustness only).
  The rule used per fold is logged in fold_log.csv (window_rule, train_from); window_vs_default.csv compares the
  variant with the expanding-window predictions in Output/rq2_5_<product> (DM, same rows). rq2_9_regime_compare.py
  collects all variants.

Evaluation (common rows: every point model + all baselines non-missing):
  point   MAE, RMSE, sMAPE, MAE_asinh; skill vs b_prev_slot and vs LEAR; Diebold-Mariano (|error|)
          of every model vs LEAR, vs b_prev_slot and vs round-1 LightGBM
  prob    mean pinball loss over QS (EUR/MWh), CRPS approximation (2 x mean pinball), coverage and
          mean width of the 80 / 90 % intervals, raw and CQR; DM on pinball loss vs lgbm_q
  by year MAE per model and year (regime reporting)

    python Modelling/rq2_5_tier1.py --only "mFRR|up|week"            # quick check (~1 min)
    python Modelling/rq2_5_tier1.py                                   # all series, all models
    python Modelling/rq2_5_tier1.py --models lear lgbm_q --products fcr
    python Modelling/rq2_5_tier1.py --products afrr --tune 20 --seeds 3 --out Modelling/Output/rq2_5t_afrr
    python Modelling/rq2_5_tier1.py --products afrr --tune 20 --tune-cv auto --out Modelling/Output/rq2_5cv_afrr
    python Modelling/rq2_5_tier1.py --products fcr --window sliding --out Modelling/Output/rq2_9s_fcr
    python Modelling/rq2_5_tier1.py --products mfrr --window regime --out Modelling/Output/rq2_9r_mfrr
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


def fit_lear(Atr, ytr, Ate, a, names, cv=None):
    """cv=None: TimeSeriesSplit(4) as in round 2; cv=list of (train, val) index pairs: the tuning splits."""
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import TimeSeriesSplit
    kw = dict(cv=cv if cv is not None else TimeSeriesSplit(n_splits=4), max_iter=5000, tol=1e-3, n_jobs=a.threads or None)
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
                min_data_in_leaf=a.min_leaf if a.min_leaf else max(5, min(20, n // 30)), feature_fraction=a.colsample,
                bagging_fraction=a.bagging, bagging_freq=1, lambda_l2=a.l2, verbose=-1, seed=seed,
                num_threads=a.threads)


def fit_lgbm_q(Xtr, ytr, Xte, a, seed, levels=None):
    """levels: indices into QS to fit (tuning fits only TUNE_Q); default all 11."""
    n = len(Xtr); nv = max(10, int(0.15 * n))
    out, iters, imp = np.full((len(Xte), len(QS)), np.nan), {}, None
    for k, q in enumerate(QS):
        if levels is not None and k not in levels:
            continue
        p = _lgb_params(n, a, seed, q)
        m0 = lgb.train(p, lgb.Dataset(Xtr.iloc[:-nv], ytr[:-nv]), num_boost_round=a.rounds,
                       valid_sets=[lgb.Dataset(Xtr.iloc[-nv:], ytr[-nv:])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        best = max(20, int((m0.best_iteration or 20) * 1.15))
        mk = lgb.train(p, lgb.Dataset(Xtr, ytr), num_boost_round=best)
        out[:, k] = mk.predict(Xte); iters[QTAG[k]] = best
        if k == MED:
            imp = _share(mk.feature_importance("gain"), Xtr.columns)
    return out, {"best_iter_q50": iters.get("50"), "best_iter_q05": iters.get("05"), "best_iter_q95": iters.get("95"), "imp": imp}


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
              **({"l2_leaf_reg": a.cat_l2} if a.cat_l2 is not None else {}),
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
                                      max_features=a.colsample, max_depth=a.qrf_depth or None,
                                      n_jobs=-1 if a.threads <= 0 else a.threads,
                                      random_state=seed)
    m.fit(Atr, ytr)
    return np.asarray(m.predict(Ate, quantiles=list(QS))).reshape(len(Ate), len(QS)), \
        {"imp": _share(m.feature_importances_, names)}


def fit_svr(Atr, ytr, Ate, a):
    from sklearn.svm import SVR
    if len(Atr) > a.svr_max:
        Atr, ytr = Atr[-a.svr_max:], ytr[-a.svr_max:]
    gamma = "scale" if a.svr_gamma is None else a.svr_gamma / (Atr.shape[1] * Atr.var())   # multiple of 'scale'
    m = SVR(kernel="rbf", C=a.svr_c, epsilon=a.svr_eps, gamma=gamma, cache_size=1000)
    m.fit(Atr, ytr)
    return np.clip(m.predict(Ate), ytr.min(), ytr.max()), {"n_support": int(len(m.support_)), "n_train_used": len(Atr)}


# ---------------------------------------------------------------- hyperparameter search (done_todo 2.2 item 5)
TUNE_Q = [int(np.flatnonzero(np.isclose(QS, q))[0]) for q in (0.1, 0.5, 0.9)]   # levels scored in tuning


def _logu(r, lo, hi):
    return float(np.exp(r.uniform(np.log(lo), np.log(hi))))


# Search spaces. Config 0 of every search = the round-2 defaults (so tuning can only replace them when a
# candidate is better on the validation window). Iterations are not searched: early stopping as before.
SPACE = {
    "lgbm_q": lambda r: dict(lr=_logu(r, 0.01, 0.1), leaves=int(r.choice([7, 15, 31, 63])),
                             min_leaf=int(r.choice([5, 10, 20, 50, 100])), colsample=float(r.uniform(0.1, 0.7)),
                             bagging=float(r.uniform(0.6, 1.0)), l2=_logu(r, 1e-3, 10.0)),
    "cat_q": lambda r: dict(cat_lr=_logu(r, 0.01, 0.1), cat_depth=int(r.choice([4, 5, 6, 7, 8])),
                            cat_l2=_logu(r, 1.0, 30.0), colsample=float(r.uniform(0.1, 0.7))),
    "qrf": lambda r: dict(qrf_leaf=int(r.choice([1, 3, 5, 10, 20, 50])), colsample=float(r.uniform(0.1, 0.7)),
                          qrf_depth=int(r.choice([0, 8, 12, 20]))),
    "svr": lambda r: dict(svr_c=_logu(r, 0.1, 100.0), svr_eps=_logu(r, 0.005, 0.3), svr_gamma=_logu(r, 0.1, 10.0)),
}
DEFAULT_KEYS = {"lgbm_q": ["lr", "leaves", "min_leaf", "colsample", "bagging", "l2"],
                "cat_q": ["cat_lr", "cat_depth", "cat_l2", "colsample"],
                "qrf": ["qrf_leaf", "colsample", "qrf_depth"], "svr": ["svr_c", "svr_eps", "svr_gamma"]}


def candidates(m: str, a, series: str) -> list[dict]:
    """a.tune configurations for model m: the defaults + (a.tune - 1) random draws. The draw depends only on
    (tune_seed, series, model), so the SAME candidates are re-evaluated at every tuning point of a series."""
    base = {k: getattr(a, k) for k in DEFAULT_KEYS[m]}
    seed = (a.tune_seed * 1_000_003 + sum(ord(c) * (i + 1) for i, c in enumerate(series + m))) % (2 ** 32)
    r = np.random.default_rng(seed)
    return [base] + [SPACE[m](r) for _ in range(a.tune - 1)]


def tune_splits(n: int, a) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-ordered validation splits INSIDE the training window (rows sorted by time): the last tune_val share
    of the rows is cut into tune_splits consecutive chunks; each chunk is predicted by a model fitted on all
    rows before it (expanding)."""
    nv = max(20, int(a.tune_val * n))
    edges = np.linspace(n - nv, n, a.tune_splits + 1).astype(int)
    return [(np.arange(0, edges[j]), np.arange(edges[j], edges[j + 1])) for j in range(a.tune_splits)]


def cv_splits(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Full time-series cross-validation over all n training rows (sorted by time): n is cut into k + 1
    consecutive chunks; split j trains on chunks 0..j and validates on chunk j + 1 (expanding window).
    Splits whose training part has < 20 rows are dropped."""
    edges = np.linspace(0, n, k + 2).astype(int)
    out = [(np.arange(0, edges[j + 1]), np.arange(edges[j + 1], edges[j + 2])) for j in range(k)]
    return [(t, v) for t, v in out if len(t) >= 20 and len(v) > 0]


def use_cv(a, n_series: int) -> bool:
    """True if this series is validated with full CV (--tune-cv cv, or auto and a small series)."""
    return a.tune_cv == "cv" or (a.tune_cv == "auto" and n_series <= a.cv_max_rows)


def val_splits(n: int, a, cv: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    return cv_splits(n, a.cv_folds) if cv else tune_splits(n, a)


def _val_score(y, Q, point: bool) -> float:
    """mean pinball loss over TUNE_Q (probabilistic models) or pinball at 0.5 (= MAE / 2, point models),
    in the model's own target space (asinh delta)."""
    if point:
        return float(np.mean(pinball(y, Q, 0.5)))
    return float(np.mean([pinball(y, Q[:, k], QS[k]).mean() for k in TUNE_Q]))


def tune_model(m, a, s, f, Xd, Zt_, Zl_, yd_, trd, cv: bool = False) -> tuple[argparse.Namespace, list[dict]]:
    """Random search with a.tune configurations for model m on the training window trd. Returns the namespace
    with the best configuration and one log row per configuration."""
    cands, splits, rows = candidates(m, a, s), val_splits(len(trd), a, cv), []
    for ci, cfg in enumerate(cands):
        ac = argparse.Namespace(**{**vars(a), **cfg})
        if m == "qrf":
            ac.qrf_trees = a.tune_qrf_trees
        t0, sc = time.time(), []
        try:
            for itr, iva in splits:
                tr, va = trd[itr], trd[iva]
                if m == "lgbm_q":
                    Q, _ = fit_lgbm_q(Xd.iloc[tr], yd_[tr], Xd.iloc[va], ac, seed=1, levels=TUNE_Q)
                elif m == "cat_q":
                    Q, _ = fit_cat_q(Xd.iloc[tr], yd_[tr], Xd.iloc[va], ac, seed=1)
                elif m == "qrf":
                    A, B, cols = prep(Zt_.iloc[tr], Zt_.iloc[va], scale=False)
                    Q, _ = fit_qrf(A, yd_[tr], B, ac, seed=1, names=cols)
                else:
                    A, B, _ = prep(Zl_.iloc[tr], Zl_.iloc[va], scale=True)
                    Q, _ = fit_svr(A, yd_[tr], B, ac)
                sc.append(_val_score(yd_[va], Q, point=(m == "svr")))
            score = float(np.mean(sc))
        except Exception as e:                          # a failing candidate is skipped, never stops the run
            score = float("inf"); print(f"      tuning {s} {m} config {ci} failed: {e}")
        rows.append({"series": s, "fold": f, "model": m, "val_scheme": "cv" if cv else "holdout",
                     "n_val_splits": len(splits), "config": ci, "is_default": ci == 0, "score": score,
                     "seconds": round(time.time() - t0, 1), **{f"hp_{k}": v for k, v in cfg.items()}})
    best = int(np.argmin([r["score"] for r in rows]))
    for r in rows:
        r["chosen"] = r["config"] == best
        r["gain_vs_default_%"] = (1 - rows[best]["score"] / rows[0]["score"]) * 100 if np.isfinite(rows[0]["score"]) else np.nan
    return argparse.Namespace(**{**vars(a), **cands[best]}), rows


# ---------------------------------------------------------------- training window (done_todo 2.2 item 7, Tier 3)
# Market-design breaks of reserve procurement (dates published before they take effect -> usable ex ante).
# Not used: regime_imb_resolution / regime_imb_pricing (balancing-energy settlement, not capacity procurement) and
# regime_ch_gen_reporting (a data-reporting change).
DESIGN_BREAK_FLAGS = ["regime_afrr_dir", "regime_fcr", "regime_de_zone", "regime_mfrr_merged", "regime_afrr_daily"]


def break_dates(g: pd.DataFrame, flags: list[str], extra: list[str]) -> list[pd.Timestamp]:
    """Structural breaks of one series (UTC): the first block at which a regime flag takes a new value (missing
    values ignored), plus the --break-extra dates (Europe/Zurich midnight)."""
    out, t = [], g["block_start_utc"]
    for c in [c for c in flags if c in g.columns]:
        idx = np.flatnonzero(g[c].notna().to_numpy())
        if len(idx) < 2:
            continue
        v = g[c].astype(str).to_numpy()
        out += list(t.iloc[idx[1:][v[idx[1:]] != v[idx[:-1]]]])
    out += [pd.Timestamp(d, tz=C.TZ).tz_convert("UTC") for d in extra]
    return sorted(set(out))


def train_window(g: pd.DataFrame, tr_all: np.ndarray, start: pd.Timestamp, s: str, a, mt: int,
                 breaks: list[pd.Timestamp], ok: np.ndarray) -> tuple[np.ndarray, str]:
    """Restrict the expanding training window tr_all (positions in time order) of the fold starting at `start`.
    Returns (positions, rule). ok = rows with a valid target (the MIN_TRAIN checks count those only)."""
    if a.window == "expanding" or len(tr_all) == 0:
        return tr_all, "expanding"
    t = g["block_start_utc"].iloc[tr_all]
    if a.window == "sliding":
        months = a.window_months_week if s.endswith("|week") else a.window_months
        cut = (start.tz_convert(C.TZ) - pd.DateOffset(months=months)).tz_convert("UTC")
        keep = tr_all[(t >= cut).to_numpy()]
        if ok[keep].sum() >= mt:
            return keep, "sliding"
        valid = tr_all[ok[tr_all]]                     # too short: the last mt valid blocks
        return (tr_all[tr_all >= valid[-mt]], "sliding_min") if len(valid) >= mt else (tr_all, "sliding_min")
    past = [b for b in breaks if b <= start]
    if not past:
        return tr_all, "regime_none"
    keep = tr_all[(t >= past[-1]).to_numpy()]
    if ok[keep].sum() >= (a.regime_min or mt):
        return keep, f"regime_since_{past[-1].tz_convert(C.TZ).date()}"
    return tr_all, f"regime_fallback_{past[-1].tz_convert(C.TZ).date()}"


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


def run_series(g: pd.DataFrame, Xg: pd.DataFrame, s: str, a, log, tlog: list | None = None) -> pd.DataFrame | None:
    ya = np.arcsinh(g["y"].to_numpy(float))
    off = np.arcsinh(g["b_prev_slot"].to_numpy(float))
    yd = ya - off                                       # delta target (tree models, SVR)
    ok_d = ~np.isnan(yd)
    offl = off if a.lear_target == "delta" else np.zeros(len(ya))
    yl = ya - offl
    ok_l = ~np.isnan(yl)
    Zt = tree_numeric(Xg) if "qrf" in a.models else None
    tuned: dict[str, argparse.Namespace] = {}           # model -> namespace with the chosen hyperparameters
    last_tune = None                                    # month index of the last tuning point
    al = argparse.Namespace(**{**vars(a), "lear_alphas": a.tune}) if a.tune else a   # LEAR: same budget
    cv = bool(a.tune) and use_cv(a, len(g))             # full CV for small series (--tune-cv)
    lear_cv = (lambda n: val_splits(n, a, cv)) if a.tune else (lambda n: None)
    Zl = linear_frame(Xg, g) if {"lear", "svr"} & set(a.models) else None
    rm = a.refit_months_week if s.endswith("|week") else a.refit_months
    mt = C.MIN_TRAIN[s.split("|")[2]]
    keys = ["series", "product", "direction", "procurement", "auction_id", "block_start_utc", "slot", "y", *C.BASELINES,
            *[c for c in g.columns if c.startswith("regime_")]]
    rows, nz, timers = [], [], {m: 0.0 for m in a.models}
    imps: dict[str, list] = {m: [] for m in a.models}       # per model: list of per-fold importance shares
    flog: list[dict] = []                                    # one row per fold: sizes, timings, tuning results
    breaks = break_dates(g, a.break_flags, a.break_extra) if a.window == "regime" else []
    if a.window == "regime":
        log(f"   {s}: breaks " + (", ".join(str(b.tz_convert(C.TZ).date()) for b in breaks) or "none"))
    for f, start, tr_all, te in C.fold_plan(g, rm, mt):
        tr_all, wrule = train_window(g, tr_all, start, s, a, mt, breaks, ok_d)   # --window (item 7)
        trd, trl = tr_all[ok_d[tr_all]], tr_all[ok_l[tr_all]]
        if len(trd) < mt:
            continue
        r = g.iloc[te][keys].copy(); r["fold"] = f; r["n_train"] = len(trd)
        fl = {"series": s, "fold": f, "fold_start": start.date(), "train_from": g["block_start_utc"].iloc[trd].min().date(),
              "test_from": g["block_start_utc"].iloc[te].min().date(), "test_to": g["block_start_utc"].iloc[te].max().date(),
              "n_train": len(trd), "n_test": len(te), "window_rule": wrule}
        mo = start.tz_convert(C.TZ).year * 12 + start.tz_convert(C.TZ).month
        if a.tune and (last_tune is None or mo - last_tune >= a.tune_every_months):
            for m in [m for m in a.models if m in SPACE]:
                t0 = time.time()
                tuned[m], rows_t = tune_model(m, a, s, f, Xg, Zt, Zl, yd, trd, cv=cv)
                if tlog is not None:
                    tlog += rows_t
                ch = next(r for r in rows_t if r["chosen"])
                fl[f"{m}_tune_seconds"] = round(time.time() - t0, 1)
                log(f"      {s} fold {f:2d}: tuned {m} on {len(trd):,} rows ({'cv' if cv else 'holdout'}) - config {ch['config']} "
                    f"(val gain vs default {ch['gain_vs_default_%']:+.1f} %, {time.time() - t0:.0f}s)")
            last_tune = mo
            fl["tuning_point"] = True
        if a.tune:
            fl["val_scheme"] = "cv" if cv else "holdout"
            fl["n_val_splits"] = len(val_splits(len(trd), a, cv))
        for m in [m for m in a.models if m in tuned]:
            fl.update({f"{m}_hp_{k}": getattr(tuned[m], k) for k in DEFAULT_KEYS[m]})
        for m in a.models:
            am = tuned.get(m, a)
            t0 = time.time()
            if m == "lear":
                if a.lear_per_slot and s.split("|")[2] == "4h":
                    p = np.full(len(te), np.nan); k_nz = []
                    for sl in np.unique(g["slot"].to_numpy()[te]):
                        tr_s = trl[g["slot"].to_numpy()[trl] == sl]; te_s = te[g["slot"].to_numpy()[te] == sl]
                        A, B, cols = prep(Zl.iloc[tr_s], Zl.iloc[te_s], scale=True)
                        ps, inf = fit_lear(A, yl[tr_s], B, al, cols, cv=lear_cv(len(tr_s)))
                        p[np.isin(te, te_s)] = ps; k_nz.append(inf["nonzero"])
                        nz.append({"series": s, "fold": f, "slot": sl, "n_train": len(tr_s), "nonzero": inf["nonzero"],
                                   "alpha": inf["alpha"]})
                    fl["lear_nonzero"] = float(np.mean(k_nz))
                else:
                    A, B, cols = prep(Zl.iloc[trl], Zl.iloc[te], scale=True)
                    p, inf = fit_lear(A, yl[trl], B, al, cols, cv=lear_cv(len(trl)))
                    imps["lear"].append(_share(inf["coef"], inf["coef"].index))
                    nz.append({"series": s, "fold": f, "slot": "all", "n_train": len(trl), "nonzero": inf["nonzero"],
                               "alpha": inf["alpha"], "n_candidates": len(cols)})
                    nz[-1].update({f"coef__{c}": v for c, v in inf["coef"][inf["coef"] != 0].items()})
                    fl.update(lear_nonzero=inf["nonzero"], lear_alpha=inf["alpha"])
                r["p_lear"] = np.sinh(p + offl[te])
            elif m == "svr":
                A, B, _ = prep(Zl.iloc[trd], Zl.iloc[te], scale=True)
                ps, inf = fit_svr(A, yd[trd], B, am)
                r["p_svr"] = np.sinh(ps + off[te]); fl.update({f"svr_{k}": v for k, v in inf.items()})
            else:
                Q = np.zeros((len(te), len(QS)))
                for sd in range(a.seeds):
                    if m == "lgbm_q":
                        q_, inf = fit_lgbm_q(Xg.iloc[trd], yd[trd], Xg.iloc[te], am, seed=1 + sd)
                    elif m == "cat_q":
                        q_, inf = fit_cat_q(Xg.iloc[trd], yd[trd], Xg.iloc[te], am, seed=1 + sd)
                    else:
                        A, B, cols = prep(Zt.iloc[trd], Zt.iloc[te], scale=False)
                        q_, inf = fit_qrf(A, yd[trd], B, am, seed=1 + sd, names=cols)
                    Q += q_
                    if a.seeds > 1:                     # per-seed median (seed robustness, item 6)
                        r[f"p_{m}_seed{sd + 1}"] = np.sinh(np.sort(q_, axis=1)[:, MED] + off[te])
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


def seed_spread(P: pd.DataFrame, s: str, models: list[str], L: int):
    """Seed robustness (done_todo 2.2 item 6): MAE of every single seed, spread over seeds, gain of the seed
    average over a single seed (DM seed average vs seed 1). Uses rows where all seeds of all models exist."""
    sc = {m: sorted([c for c in P.columns if c.startswith(f"p_{m}_seed")], key=lambda c: int(c.rsplit("seed", 1)[1]))
          for m in models}
    sc = {m: c for m, c in sc.items() if len(c) > 1 and f"p_{m}" in P}
    if not sc:
        return [], []
    E = P.dropna(subset=["y", *[f"p_{m}" for m in sc], *[c for v in sc.values() for c in v]]).sort_values("block_start_utc")
    y = E["y"].to_numpy(float)
    rows, per = [], []
    for m, cols in sc.items():
        maes = np.array([np.abs(E[c].to_numpy(float) - y).mean() for c in cols])
        e_avg = np.abs(E[f"p_{m}"].to_numpy(float) - y)
        st, p = dm_loss(e_avg - np.abs(E[cols[0]].to_numpy(float) - y), L)
        nb = np.abs(E["b_prev_slot"].to_numpy(float) - y).mean()
        rows.append({"series": s, "model": m, "n": len(E), "n_seeds": len(cols), "mae_seed_mean": maes.mean(),
                     "mae_seed_std": maes.std(ddof=1), "mae_seed_min": maes.min(), "mae_seed_max": maes.max(),
                     "seed_range_%_of_mae": (maes.max() - maes.min()) / maes.mean() * 100,
                     "skill_seed_min_%": (1 - maes.max() / nb) * 100, "skill_seed_max_%": (1 - maes.min() / nb) * 100,
                     "mae_seed_avg": e_avg.mean(), "gain_avg_vs_mean_seed_%": (1 - e_avg.mean() / maes.mean()) * 100,
                     "dm_avg_vs_seed1": st, "p_avg_vs_seed1": p})
        per += [{"series": s, "model": m, "seed": int(c.rsplit("seed", 1)[1]), "MAE": v} for c, v in zip(cols, maes)]
    return rows, per


def default_preds(view: str, spec: str):
    """Round-2 (untuned, 1 seed) predictions to compare a tuned run with: --compare-default auto | none | path."""
    if spec == "none":
        return None
    base = C.ROOT / "Modelling" / "Output"
    cands = [Path(spec)] if spec != "auto" else [base / f"rq2_5_{view}" / "preds.parquet", base / "rq2_5" / "preds.parquet"]
    for p in cands:
        p = p if p.suffix == ".parquet" else p / "preds.parquet"
        if p.exists():
            return pd.read_parquet(p)
    return None


def compare_default(P: pd.DataFrame, D0: pd.DataFrame, s: str, models: list[str], prob: list[str], L: int):
    """Tuned (this run) vs default (round 2) on the rows both have: MAE / pinball and DM per model."""
    d = D0[D0["series"] == s]
    cols = [c for c in d.columns if (c.startswith("p_") and c[2:] in models)
            or (c.startswith("q_") and c[2:].rsplit("_", 1)[0] in prob)]
    if d.empty or not cols:
        return []
    M = P.merge(d[["series", "block_start_utc", *cols]], on=["series", "block_start_utc"], how="inner", suffixes=("", "__def"))
    out = []
    for m in models:
        if f"p_{m}__def" not in M:
            continue
        E = M.dropna(subset=["y", f"p_{m}", f"p_{m}__def"]).sort_values("block_start_utc")
        y = E["y"].to_numpy(float)
        e1, e0 = np.abs(E[f"p_{m}"].to_numpy(float) - y), np.abs(E[f"p_{m}__def"].to_numpy(float) - y)
        nb = np.abs(E["b_prev_slot"].to_numpy(float) - y)
        ok = ~np.isnan(nb)
        st, p = dm_loss(e1 - e0, L)
        row = {"series": s, "model": m, "n": len(E), "mae_default": e0.mean(), "mae_tuned": e1.mean(),
               "change_%": (e1.mean() / e0.mean() - 1) * 100,
               "skill_default_%": (1 - e0[ok].mean() / nb[ok].mean()) * 100, "skill_tuned_%": (1 - e1[ok].mean() / nb[ok].mean()) * 100,
               "dm_tuned_vs_default": st, "p": p}
        if m in prob and all(f"q_{m}_{t}__def" in E for t in QTAG):
            pl1 = np.mean([pinball(y, E[f"q_{m}_{t}"].to_numpy(float), q) for q, t in zip(QS, QTAG)], axis=0)
            pl0 = np.mean([pinball(y, E[f"q_{m}_{t}__def"].to_numpy(float), q) for q, t in zip(QS, QTAG)], axis=0)
            stq, pq = dm_loss(pl1 - pl0, L)
            row.update(pinball_default=pl0.mean(), pinball_tuned=pl1.mean(), dm_pinball=stq, p_pinball=pq)
        out.append(row)
    return out


def WINDOW_TXT(a) -> str:
    if a.window == "sliding":
        return f"sliding window ({a.window_months} months; weekly {a.window_months_week})"
    if a.window == "regime":
        return (f"regime window (since the latest break of {', '.join(a.break_flags) or '-'}"
                f"{'; extra breaks ' + ', '.join(a.break_extra) if a.break_extra else ''})")
    return "expanding window"


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
    ap.add_argument("--min-leaf", type=int, default=0, help="LightGBM min_data_in_leaf (0 = round-2 rule max(5, min(20, n/30)))")
    ap.add_argument("--bagging", type=float, default=0.8, help="LightGBM bagging_fraction")
    ap.add_argument("--l2", type=float, default=1.0, help="LightGBM lambda_l2")
    ap.add_argument("--cat-l2", type=float, default=None, help="CatBoost l2_leaf_reg (default: CatBoost default)")
    ap.add_argument("--qrf-depth", type=int, default=0, help="QRF max_depth (0 = unlimited)")
    ap.add_argument("--svr-gamma", type=float, default=None, help="SVR gamma as a multiple of 'scale' (default: 'scale')")
    ap.add_argument("--tune", type=int, default=0,
                    help="hyperparameter search: n configurations per model and tuning point (0 = fixed defaults as round 2)")
    ap.add_argument("--tune-every-months", type=int, default=12, help="re-tune when the fold start is >= n months after the last tuning point")
    ap.add_argument("--tune-val", type=float, default=0.2, help="share of the training window used for validation")
    ap.add_argument("--tune-splits", type=int, default=1, help="consecutive validation chunks (expanding); 1 = one hold-out")
    ap.add_argument("--tune-cv", default="holdout", choices=["holdout", "cv", "auto"],
                    help="validation scheme: holdout (last --tune-val of the window), cv (full time-series CV), "
                         "auto (cv for series with <= --cv-max-rows blocks, holdout otherwise)")
    ap.add_argument("--cv-folds", type=int, default=5, help="number of time-series CV splits")
    ap.add_argument("--cv-max-rows", type=int, default=5000, help="auto: series with at most this many blocks use full CV")
    ap.add_argument("--tune-qrf-trees", type=int, default=100, help="QRF trees while tuning (final fit uses --qrf-trees)")
    ap.add_argument("--tune-seed", type=int, default=0, help="seed of the random candidate draw")
    ap.add_argument("--compare-default", default="auto",
                    help="untuned predictions for the tuned-vs-default table: auto (Output/rq2_5_<product>) | none | path")
    ap.add_argument("--window", default="expanding", choices=["expanding", "sliding", "regime"],
                    help="training window (done_todo 2.2 item 7): expanding (round 2), sliding, regime (since last break)")
    ap.add_argument("--window-months", type=int, default=24, help="sliding window length, 4h / day series")
    ap.add_argument("--window-months-week", type=int, default=48, help="sliding window length, weekly series")
    ap.add_argument("--break-flags", nargs="*", default=DESIGN_BREAK_FLAGS,
                    help="regime flags whose changes count as structural breaks (--window regime)")
    ap.add_argument("--break-extra", nargs="*", default=[],
                    help="extra break dates, e.g. 2021-09-01 2023-07-01 (crisis start / end; chosen ex post = robustness)")
    ap.add_argument("--regime-min", type=int, default=0,
                    help="valid blocks needed after the latest break before training on it alone (0 = MIN_TRAIN)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else C.ROOT / "Modelling" / "Output" / (
        "rq2_5_tuned" if a.tune else (f"rq2_9_{a.window}" if a.window != "expanding" else "rq2_5"))
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    lines: list[str] = []
    log = lambda x: print(x, flush=True)

    r1_path = C.ROOT / "Modelling" / "Output" / "rq2_2_delta" / "preds_exante.parquet"
    r1 = pd.read_parquet(r1_path, columns=["series", "block_start_utc", "pred"]).drop_duplicates(
        ["series", "block_start_utc"]).rename(columns={"pred": "p_lgbm_l1"}) if r1_path.exists() else None

    t_all = time.time()
    allP, MP, MY, DM, MQ, NZ, FLOG, IMP = [], [], [], [], [], [], [], {}
    TLOG, SS, SM, CD = [], [], [], []
    for view in a.products:
        df = C.load_view(view)
        feats, _ = feature_list(df, view, "exante")
        X = make_X(df, feats)
        log(f"{view}: {len(df):,} blocks, {len(feats)} features")
        D0 = default_preds(view, a.compare_default) if a.tune or a.seeds > 1 or a.window != "expanding" else None
        for s in df["series"].unique():
            if a.only and a.only != s:
                continue
            gi = np.flatnonzero((df["series"] == s).to_numpy())
            g, Xg = df.iloc[gi].reset_index(drop=True), X.iloc[gi].reset_index(drop=True)
            P, nz, imps, flog = run_series(g, Xg, s, a, log, TLOG)
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
            L_ = 4 if s.endswith("|week") else 12
            ss, sm = seed_spread(P, s, [m for m in prob], L_)
            SS += ss; SM += sm
            if D0 is not None:
                CD += compare_default(P, D0, s, list(a.models), prob, L_)

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
    TL, SS, SM, CD = map(pd.DataFrame, (TLOG, SS, SM, CD))
    cmp_name = "window_vs_default" if a.window != "expanding" and not a.tune else "tuned_vs_default"
    for nm, d_ in [("tuning_log", TL), ("seed_spread", SS), ("seed_mae", SM), (cmp_name, CD)]:
        if len(d_):
            d_.to_csv(out / f"{nm}.csv", index=False)
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
           f"Rolling origin, {WINDOW_TXT(a)}, refit every {a.refit_months} months (weekly {a.refit_months_week}); "
           "test from 2021-01-01; errors in EUR/MWh on the common rows of all point models and baselines.",
           f"Models: {', '.join(LABEL.get(m, m) for m in a.models)}; quantiles {', '.join(QTAG)}; seeds {a.seeds}; "
           f"LEAR target {a.lear_target}{', per slot' if a.lear_per_slot else ''}.",
           (f"Hyperparameter search: {a.tune} configurations per model (config 0 = round-2 defaults, rest random draws), "
            f"re-tuned every {a.tune_every_months} months; validation {a.tune_cv}: "
            + {"holdout": f"last {a.tune_val:.0%} of the training window ({a.tune_splits} chunk(s))",
               "cv": f"full time-series CV, {a.cv_folds} expanding splits over the whole training window",
               "auto": f"full time-series CV ({a.cv_folds} splits) for series with <= {a.cv_max_rows:,} blocks, "
                       f"else last {a.tune_val:.0%} of the window"}[a.tune_cv]
            + f"; score = mean pinball at 10/50/90 % (SVR: MAE); LEAR: LassoCV over {a.tune} alphas "
            f"on the same split(s); QRF tuned with {a.tune_qrf_trees} trees." if a.tune else
            "No hyperparameter search (fixed defaults; LEAR alpha by LassoCV, TimeSeriesSplit(4))."),
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
    if len(TL):
        hdr.append("== Hyperparameter search (tuning_log.csv): per series / model - tuning points, how often the default "
                   "config won, median validation gain of the chosen config vs default, tuning time")
        ch = TL[TL["chosen"]]
        S_ = ch.groupby(["series", "model"]).agg(points=("fold", "nunique"), default_chosen=("is_default", "sum"),
                                                 median_val_gain_pct=("gain_vs_default_%", "median"))
        S_["tune_minutes"] = TL.groupby(["series", "model"])["seconds"].sum() / 60
        hdr.append(C.fmt(S_))
        for (s_, m_), g_ in ch.groupby(["series", "model"], sort=False):
            hp = [c for c in g_.columns if c.startswith("hp_") and g_[c].notna().any()]
            hdr.append(f"   {s_} {m_}: " + " | ".join(
                f"fold {r['fold']}: " + ", ".join(f"{c[3:]}={r[c]:.3g}" if isinstance(r[c], float) else f"{c[3:]}={r[c]}"
                                                  for c in hp) for _, r in g_.iterrows()))
    if len(CD):
        what = "tuned" if a.tune else (f"{a.window} window" if a.window != "expanding" else "this run")
        hdr.append(f"== {what} vs default (round 2, expanding window) on the same rows ({cmp_name}.csv; columns *_tuned = "
                   f"this run); DM stat < 0 = this run better")
        hdr.append(C.fmt(CD.set_index(["series", "model"])))
        for _, r in CD.iterrows():
            if r["p"] < 0.05:
                RU.summary(f"{r['series'].replace('|', ' ')} {r['model']}: {what} {r['change_%']:+.1f}% MAE (p={r['p']:.3f})")
    if a.window != "expanding":
        Fw = pd.DataFrame(FLOG)
        if "window_rule" in Fw:
            hdr.append("== Training window rule per series (folds)")
            Fw["rule"] = Fw["window_rule"].str.replace(r"_\d{4}-\d{2}-\d{2}$", "", regex=True)
            hdr.append(C.fmt(Fw.groupby(["series", "rule"]).agg(folds=("fold", "count"), median_n_train=("n_train", "median"))))
    if len(SS):
        hdr.append(f"== Seed robustness ({a.seeds} seeds): MAE per seed, spread, gain of the seed average (seed_spread.csv, seed_mae.csv)")
        hdr.append(C.fmt(SS.set_index(["series", "model"])))
        if len(SM):
            for s_, g_ in SM.groupby("series", sort=False):
                pv = g_.pivot(index="seed", columns="model", values="MAE")
                hdr.append(f"   {s_}: best model per seed: " + ", ".join(f"seed {i} {pv.loc[i].idxmin()}" for i in pv.index))
    hdr += lines
    hdr.append(f"\ntotal runtime {time.time() - t_all:.0f}s")
    (out / "report_rq2_5.txt").write_text("\n".join(hdr))
    print("\n".join(hdr))


if __name__ == "__main__":
    RU.run(main, "RQ2 Tier 1 models")
