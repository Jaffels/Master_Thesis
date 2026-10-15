"""RQ3b (todo 9.5): can a probabilistic model that uses only information available before the
forecast origin identify high-risk hours better than calendar / persistence baselines?

Design (decisions 3 Oct / 8 Oct 2026, D2):
  * Unit = HOUR (UTC hour, 4 complete quarter-hours): target = any spike QH in the hour
    (short = spike_short_q990, long = spike_long_q010; rolling 365-day thresholds, past only).
  * Origins: d1 = D-1 18:00 (day-ahead), h1 = 1 h before the first quarter-hour of the hour (nowcast).
    Features are taken as published at the origin (view columns __d1 / __h1, rq3_feature_columns);
    day-ahead forecasts (__fc_d1) are averaged over the 4 QH, everything else is read at the first QH.
  * Validation: ROLLING ORIGIN, expanding window, refit every 6 months, test 1 Jan 2021 -> 31 Aug 2026,
    2-day embargo between train and fold. Pooled out-of-fold predictions.
  * Baselines (past only): clim_exp = spike frequency of (hour x day type) over all earlier hours;
    clim_roll = same over the last 150 observations of the cell; persist = 1-feature (d1) or 2-feature (h1)
    logistic on the share of spike QH in the last 24 h (and 4 h); cal_hist = logistic on calendar + persistence.
  * Models: lr = logistic on cal + fund (+ hist, + nowcast for h1); gbm = LightGBM on all strict ex-ante
    features of the origin; gbm_pf = gbm + observed weather (perfect-forecast upper bound, not ex ante).
  * Metrics (pooled test, and pre-/post-cap split at 3 Mar 2025): Brier, Brier skill vs the best baseline,
    log loss, PR-AUC, ROC-AUC; reliability table (deciles); day-block bootstrap CI of the Brier gain over the best baseline.

Variants (decision round 10 Oct 2026, after the first full run):
    --target fixed          target = hour with any QH beyond FIXED absolute thresholds (default: RQ3c prior12m values,
                            short q99 = 361.0, long q01 = -228.5 EUR/MWh) instead of the rolling 365-day quantile;
                            persistence features are rebuilt on the same fixed flag (fixhist_*).
    --train-from DATE       train only on hours from DATE (e.g. 2022-06-01 = QH-price regime)
    --test-start DATE       first test day (default 2021-01-01)
    --tag NAME              output folder suffix: Modelling/Output/rq3b_1<NAME>/

Calibration variants (done_todo 2.3 item 10, 15 Oct 2026; defaults = bit-identical to the 10 Oct runs):
    --calib platt           (default) model fitted on hours before (fold start - 6 months - embargo), logistic
                            recalibration (Platt) on the last 6 months before the embargo.
    --calib isotonic        same split, isotonic regression instead of Platt.
    --calib platt_oof       CROSS-FITTED: final model on the FULL training window (no 6 months lost); the calibrator is
    --calib isotonic_oof    fitted on out-of-fold predictions of the last --calib-blocks (default 3) 6-month blocks
                            before the embargo, each predicted by a model trained only on hours before that block
                            (minus embargo). Isotonic needs the larger OOF sample (rare positives).
    --calib none            raw model on the full training window, no recalibration (shows the size of the problem).
    --gbm-init lr           LightGBM boosted FROM the logistic model (init_score = logit of lr_d1 / lr_h1 fitted on the
                            same training rows): the trees learn corrections to the linear model, so the forecast can
                            still extrapolate when the spike rate leaves the training range (2021-H2 crisis onset).
    The calibration flag applies to lr_* and gbm_* (baselines never recalibrated). LightGBM uses the plain binary
    objective without scale_pos_weight / is_unbalance (class weights would distort the probability level).
    New metrics: mean_pred, O/E (observed / expected), ECE (10 equal-count bins), cal_slope (logistic slope of y on
    logit p; < 1 = over-confident). Compare variants with rq3b_2_calib_compare.py.

Run from Master_Thesis with .venv active (needs rq3a_1_catalogue.py and rq3a_2_logit.py in Modelling/):
    python Modelling/rq3b_1_riskmodel.py [--fast]       # -> Modelling/Output/rq3b_1/
--fast: refit every 12 months, fewer trees (smoke test, ~1/3 of the time).
Read-only on Clean/Data; writes only Modelling/Output/rq3b_1/.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import run_utils as RU  # run.log, run_meta.json, run history, macOS notification

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Clean"))
sys.path.insert(0, str(HERE))
import views as V  # noqa: E402
from rq3a_2_logit import fit_logit, predict, auc, avg_precision  # noqa: E402

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except Exception:                               # pragma: no cover
    lgb = None

OUT = HERE / "Output" / "rq3b_1"
TARGETS = {"short": "spike_short_q990", "long": "spike_long_q010"}
TEST_START = pd.Timestamp("2021-01-01")
CAP = pd.Timestamp("2025-03-03")
EMBARGO = pd.Timedelta(days=2)
TRAIN_FROM = pd.Timestamp("1900-01-01")
FIXED = None            # (thr_short, thr_long) when --target fixed
CALIB = "platt"         # platt | isotonic | platt_oof | isotonic_oof | none
CALIB_BLOCKS = 3        # 6-month OOF blocks for the *_oof calibrators
GBM_INIT = "none"       # none | lr
PF = ["ch_temp_lw_degc__pf", "ch_ghi_lw_wm2__pf", "ch_ghi_ramp_lw_wm2__pf", "ch_wind_lw_ms__pf",
      "ch_hdh_lw_kh__pf", "ch_cdh_lw_kh__pf", "ch_precip_hydro_mm__pf", "ch_snow_hydro_cm__pf",
      "ch_melt_dh_hydro_kh__pf"]
REGIME_MAP = {"mixed": 0, "hourly_dominant": 1, "qh": 2, "dual": 0, "single": 1}
lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


# ------------------------------------------------------------------ data
def add_fixhist(h: pd.DataFrame) -> None:
    """Persistence features for the fixed-threshold target, from the hourly flag itself: share of spike hours in the
    24 h before the d1 origin (D-1 18:00 local) and in the 4 h before the h1 origin (hour start - 1 h)."""
    idx = pd.DatetimeIndex(h["hk"])
    full = pd.date_range(idx.min(), idx.max(), freq="h", tz=idx.tz)
    pos = full.get_indexer(idx)
    loc = idx.tz_convert(V.TZ).tz_localize(None)
    o_utc = (loc.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize(V.TZ).tz_convert(idx.tz)
    o_pos = full.get_indexer(o_utc.floor("h"), method="pad")
    for side in ("short", "long"):
        sr = pd.Series(h[f"spike_{side}_fix"].to_numpy(float), index=idx).reindex(full).fillna(0.0)
        cs = np.r_[0.0, sr.cumsum().to_numpy()]
        a = np.clip(o_pos - 24, 0, None)
        b = np.clip(o_pos, 0, None)
        h[f"fixhist_{side}__lb24h_d1"] = (cs[b] - cs[a]) / np.maximum(b - a, 1)
        a4 = np.clip(pos - 5, 0, None)
        b4 = np.clip(pos - 1, 0, None)
        h[f"fixhist_{side}__lb4h_h1"] = (cs[b4] - cs[a4]) / np.maximum(b4 - a4, 1)


def build_hourly() -> tuple[pd.DataFrame, dict]:
    df = V.load_rq3(school_holidays=True)
    feats = {o: V.rq3_feature_columns(df, origin=o) for o in ("d1", "h1")}
    pf = [c for c in PF if c in df.columns]
    # derived QH-level columns (past-only): DA price level vs trailing 30 d, neighbour spreads
    ch = df["ch_price_da_eur_mwh__fc_d1"]
    df["da_ch_rel30"] = ch - ch.rolling(96 * 30, min_periods=96 * 7).mean()
    for z in ("de_lu", "fr", "it_nord", "at"):
        c = f"{z}_price_da_eur_mwh__fc_d1"
        if c in df:
            df[f"da_{z}_minus_ch"] = df[c] - ch
    t = pd.to_datetime(df["ts_utc"])
    df["hk"] = t.dt.floor("h")
    for c in ("regime_imb_resolution", "regime_imb_pricing"):
        df[c] = df[c].astype(str).map(REGIME_MAP).astype("float32")
    spikes = ["spike_short_q990", "spike_long_q010"]      # rolling-threshold flags (always loaded)
    for s in spikes:
        df[s] = df[s].fillna(0).astype("int8")
    num = [c for c in df.columns if c not in ("ts_utc", "ts_local", "hk") and pd.api.types.is_numeric_dtype(df[c])]
    fc = [c for c in num if c.endswith("__fc_d1") or c.startswith("da_")]
    agg = {c: "first" for c in num}
    agg.update({c: "mean" for c in fc})
    agg.update({s: "max" for s in spikes})
    agg["tgt_price_short_eur_mwh"] = "max"
    agg["tgt_price_long_eur_mwh"] = "min"
    g = df.groupby("hk")
    h = g.agg(agg)
    h["nqh"] = g.size()
    h = h[h["nqh"] == 4].drop(columns="nqh").reset_index()
    loc = h["hk"].dt.tz_localize("UTC").dt.tz_convert(V.TZ) if h["hk"].dt.tz is None else h["hk"].dt.tz_convert(V.TZ)
    h["date"] = loc.dt.tz_localize(None).dt.normalize()
    hr = h["cal_hour"].astype(float)
    for k in (1, 2):
        h[f"hr_sin{k}"] = np.sin(2 * np.pi * k * hr / 24)
        h[f"hr_cos{k}"] = np.cos(2 * np.pi * k * hr / 24)
    for d in range(1, 7):
        h[f"dow{d}"] = (h["cal_dow"] == d).astype(float)
    # day type for the climatology cell: weekday / saturday / sunday-or-holiday
    h["daytype"] = np.where((h["cal_holiday"] == 1) | (h["cal_dow"] == 6), 2, np.where(h["cal_dow"] == 5, 1, 0))
    h["cell"] = h["cal_hour"].astype(int) * 3 + h["daytype"]
    h = h.sort_values("hk").reset_index(drop=True)
    if FIXED is not None:
        h["spike_short_fix"] = (h["tgt_price_short_eur_mwh"] > FIXED[0]).astype("int8")
        h["spike_long_fix"] = (h["tgt_price_long_eur_mwh"] < FIXED[1]).astype("int8")
        add_fixhist(h)
    for k in list(feats):
        feats[k] = [c for c in feats[k] if c in h.columns and c != "cal_year"]
    if FIXED is not None:
        feats["d1"] += [f"fixhist_{sd}__lb24h_d1" for sd in ("short", "long")]
        feats["h1"] += [f"fixhist_{sd}__lb24h_d1" for sd in ("short", "long")] + [f"fixhist_{sd}__lb4h_h1" for sd in ("short", "long")]
    feats["pf"] = [c for c in pf if c in h.columns]
    return h, feats


def blocks(h: pd.DataFrame) -> dict[str, list[str]]:
    b = {
        "cal": ["hr_sin1", "hr_cos1", "hr_sin2", "hr_cos2", *[f"dow{d}" for d in range(1, 7)], "cal_holiday",
                "cal_day_before_holiday", "cal_day_after_holiday", "cal_bridge_day", "cal_xmas_period",
                "cal_dst_switch_day", "cal_school_holiday_share", "cal_doy_sin", "cal_doy_cos"],
        "fund": ["da_ch_rel30", "da_de_lu_minus_ch", "da_fr_minus_ch", "da_it_nord_minus_ch", "da_at_minus_ch",
                 "ch_resid_load_da_fc_mw__fc_d1", "ch_resid_load_da_fc_mw_ramp__fc_d1",
                 "de_lu_resid_load_da_fc_mw__fc_d1", "de_lu_resid_load_da_fc_mw_ramp__fc_d1",
                 "ch_gen_solar_da_fc_mw__fc_d1", "ch_gen_solar_da_fc_ramp_mw__fc_d1",
                 "de_lu_gen_solar_da_fc_mw__fc_d1", "ch_load_da_fc_mw__fc_d1"],
        "hist": ["spikehist_short_q990__lb24h_d1", "spikehist_long_q010__lb24h_d1",
                 "err_ch_load_mw__lb24h_d1", "err_ch_solar_mw__lb24h_d1", "err_de_lu_solar_mw__lb24h_d1",
                 "err_de_lu_wind_on_mw__lb24h_d1", "err_de_lu_load_mw__lb24h_d1"],
        "now": ["spikehist_short_q990__lb4h_h1", "spikehist_long_q010__lb4h_h1",
                "ch_imb_price_short_eur_mwh__last_h1", "ch_imb_price_long_eur_mwh__last_h1",
                "ch_imb_vol_mwh__lb1h_h1", "ch_imb_vol_mwh__lb4h_h1", "err_ch_load_mw__lb1h_h1",
                "err_ch_solar_mw__lb1h_h1", "err_de_lu_solar_mw__lb1h_h1", "err_de_lu_wind_on_mw__lb1h_h1"],
    }
    if FIXED is not None:
        ren = lambda c: (c.replace("spikehist_short_q990", "fixhist_short").replace("spikehist_long_q010", "fixhist_long"))
        b = {k: [ren(c) for c in v] for k, v in b.items()}
    return {k: [c for c in v if c in h.columns] for k, v in b.items()}


# ------------------------------------------------------------------ baselines (past only)
def climatology(h: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    y = h[target].astype(float)
    g = y.groupby(h["cell"])
    prior = y.expanding().mean().shift(1).fillna(y.mean())
    cnt = g.cumcount()
    exp_mean = (g.cumsum() - y) / cnt.replace(0, np.nan)
    exp_ = exp_mean.fillna(prior).to_numpy()
    roll = g.transform(lambda s: s.shift(1).rolling(150, min_periods=20).mean()).fillna(pd.Series(exp_)).to_numpy()
    return np.clip(exp_, 1e-4, 1 - 1e-4), np.clip(roll, 1e-4, 1 - 1e-4)


# ------------------------------------------------------------------ helpers
def lr_fit_predict(tr, te, cols, target, lam=5.0):
    cols = [c for c in cols if tr[c].isna().mean() <= 0.30 and tr[c].nunique() > 1]
    med = tr[cols].median()
    mu, sd = tr[cols].fillna(med).mean(), tr[cols].fillna(med).std().replace(0, 1)
    Xtr = ((tr[cols].fillna(med) - mu) / sd).clip(-6, 6).to_numpy()
    Xte = ((te[cols].fillna(med) - mu) / sd).clip(-6, 6).to_numpy()
    beta, _ = fit_logit(Xtr, tr[target].to_numpy(float), lam)
    return np.clip(predict(beta, Xte), 1e-5, 1 - 1e-5)


def lr_logits(tr, te, cols, target, lam=30.0):
    """Same logistic model as lr_fit_predict, returning the linear predictor (logit) for train and test rows;
    used as LightGBM init_score for --gbm-init lr."""
    cols = [c for c in cols if tr[c].isna().mean() <= 0.30 and tr[c].nunique() > 1]
    med = tr[cols].median()
    mu, sd = tr[cols].fillna(med).mean(), tr[cols].fillna(med).std().replace(0, 1)
    Xtr = ((tr[cols].fillna(med) - mu) / sd).clip(-6, 6).to_numpy()
    Xte = ((te[cols].fillna(med) - mu) / sd).clip(-6, 6).to_numpy()
    beta, _ = fit_logit(Xtr, tr[target].to_numpy(float), lam)
    lin = lambda X: np.clip(np.column_stack([np.ones(len(X)), X]) @ beta, -11.5, 11.5)
    return lin(Xtr), lin(Xte)


IMP_LOG: list = []      # gain importance of every LightGBM fit (side, fold, model set in CTX by main)
CTX: dict = {}


def gbm_fit_predict(tr, te, cols, target, fast, init_cols=None):
    """Plain binary objective (no scale_pos_weight / is_unbalance). init_cols: boost from the logistic model on
    these columns (--gbm-init lr)."""
    cols = [c for c in cols if tr[c].notna().any() and tr[c].nunique() > 1]
    params = dict(objective="binary", learning_rate=0.04, num_leaves=15, min_data_in_leaf=40, bagging_fraction=0.8,
                  bagging_freq=1, feature_fraction=0.5, lambda_l2=5.0, num_threads=4, verbose=-1, seed=7)
    z_tr = z_te = None
    if init_cols:
        z_tr, z_te = lr_logits(tr, te, init_cols, target, 30.0)
    ds = lgb.Dataset(tr[cols].astype("float32"), label=tr[target].to_numpy(), init_score=z_tr, free_raw_data=True)
    m = lgb.train(params, ds, num_boost_round=150 if fast else 350)
    if not CTX.get("aux"):
        g = pd.Series(m.feature_importance("gain"), index=cols)
        IMP_LOG.append({k: v for k, v in CTX.items() if k != "aux"} | {"imp": g / g.sum() if g.sum() > 0 else g})
    if z_te is None:
        return np.clip(m.predict(te[cols].astype("float32")), 1e-5, 1 - 1e-5)
    raw = m.predict(te[cols].astype("float32"), raw_score=True) + z_te
    return np.clip(1 / (1 + np.exp(-raw)), 1e-5, 1 - 1e-5)


def logloss(y, p):
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(y, p, bins=10):
    """Expected calibration error over equal-count bins of the predicted probability."""
    if len(y) < bins:
        return float("nan")
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False).to_numpy()
    n = np.bincount(q, minlength=bins)
    mp = np.bincount(q, weights=p, minlength=bins) / np.maximum(n, 1)
    my = np.bincount(q, weights=y, minlength=bins) / np.maximum(n, 1)
    return float((n * np.abs(mp - my)).sum() / n.sum())


def cal_slope(y, p):
    """Slope of a logistic regression of y on logit(p): 1 = well calibrated, < 1 = over-confident."""
    if y.sum() < 5 or y.sum() > len(y) - 5:
        return float("nan")
    z = np.log(p / (1 - p))
    beta, _ = fit_logit(((z - z.mean()) / (z.std() or 1.0))[:, None], y, 0.0)
    return float(beta[1] / (z.std() or 1.0))


def metrics(y, p) -> dict:
    return {"n": len(y), "pos": int(y.sum()), "Brier": float(((p - y) ** 2).mean()), "logloss": logloss(y, p),
            "PR_AUC": avg_precision(y, p), "ROC_AUC": auc(y, p), "base_rate": float(y.mean()),
            "mean_pred": float(p.mean()), "O_E": float(y.mean() / max(p.mean(), 1e-12)), "ECE": ece(y, p),
            "cal_slope": cal_slope(y, p)}


def platt(fn, tr_fit, cal, te, *args):
    """Fit fn on tr_fit, then recalibrate (slope + intercept on logit) on the later calibration window so
    that the probability LEVEL follows the recent spike rate (the base rate drifts 0.2 - 10 % across years)."""
    pr = fn(tr_fit, pd.concat([cal, te]), *args)
    pc, pt = pr[: len(cal)], pr[len(cal):]
    target = args[1]
    y = cal[target].to_numpy(float)
    if y.sum() < 10 or len(cal) < 500:
        return pt
    return (_iso_map if CALIB == "isotonic" else _platt_map)(pc, y, pt)


def _platt_map(pc, y, pt):
    z = lambda p: np.log(p / (1 - p))
    zc = z(pc)
    mu, sd = zc.mean(), zc.std() or 1.0
    beta, _ = fit_logit(((zc - mu) / sd)[:, None], y, 0.1)
    return np.clip(predict(beta, ((z(pt) - mu) / sd)[:, None]), 1e-5, 1 - 1e-5)


def _iso_map(pc, y, pt):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=1e-5, y_max=1 - 1e-5, out_of_bounds="clip", increasing=True)
    ir.fit(pc, y)
    return np.clip(ir.predict(pt), 1e-5, 1 - 1e-5)


CAL_LOG: list = []      # per fit: calibration rows / positives used (OOF variants)


def recal_oof(fn, tr, te, hk_tr, f0, *args):
    """Cross-fitted recalibration: final model on the full training window tr; calibrator fitted on out-of-fold
    predictions of the last CALIB_BLOCKS 6-month blocks before f0 - EMBARGO (each block predicted by a model trained
    on hours before block start - EMBARGO). CALIB == 'none': raw model, no recalibration."""
    target = args[1]
    pt = fn(tr, te, *args)
    if CALIB == "none":
        return pt
    end = f0 - EMBARGO
    ytr = tr[target].to_numpy(float)
    pcs, ys = [], []
    CTX["aux"] = True
    try:
        for k in range(CALIB_BLOCKS, 0, -1):
            b0, b1 = end - pd.DateOffset(months=6 * k), end - pd.DateOffset(months=6 * (k - 1))
            fm = hk_tr < (b0 - EMBARGO)
            bm = (hk_tr >= b0) & (hk_tr < b1)
            if bm.sum() == 0 or ytr[fm].sum() < 30:
                continue
            pcs.append(fn(tr[fm], tr[bm], *args))
            ys.append(ytr[bm])
    finally:
        CTX["aux"] = False
    n_c = sum(len(v) for v in ys)
    pos_c = float(sum(v.sum() for v in ys))
    CAL_LOG.append({k: v for k, v in CTX.items() if k != "aux"} | {"fold": str(f0.date()), "n_cal": n_c,
                                                                 "pos_cal": pos_c, "blocks": len(ys)})
    if pos_c < 10 or n_c < 500:
        return pt
    return (_iso_map if CALIB == "isotonic_oof" else _platt_map)(np.concatenate(pcs), np.concatenate(ys), pt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--target", choices=["rolling", "fixed"], default="rolling")
    ap.add_argument("--thr-short", type=float, default=361.0)
    ap.add_argument("--thr-long", type=float, default=-228.5)
    ap.add_argument("--train-from", default="1900-01-01")
    ap.add_argument("--test-start", default="2021-01-01")
    ap.add_argument("--tag", default="")
    ap.add_argument("--calib", choices=["platt", "isotonic", "platt_oof", "isotonic_oof", "none"], default="platt")
    ap.add_argument("--calib-blocks", type=int, default=3)
    ap.add_argument("--gbm-init", choices=["none", "lr"], default="none")
    a = ap.parse_args()
    global OUT, TARGETS, TEST_START, TRAIN_FROM, FIXED, CALIB, CALIB_BLOCKS, GBM_INIT
    CALIB, CALIB_BLOCKS, GBM_INIT = a.calib, a.calib_blocks, a.gbm_init
    OUT = HERE / "Output" / ("rq3b_1" + a.tag)
    TEST_START = pd.Timestamp(a.test_start)
    TRAIN_FROM = pd.Timestamp(a.train_from)
    if a.target == "fixed":
        FIXED = (a.thr_short, a.thr_long)
        TARGETS = {"short": "spike_short_fix", "long": "spike_long_fix"}
    OUT.mkdir(parents=True, exist_ok=True)
    RU.set_output(OUT)
    t0 = time.time()
    h, feats = build_hourly()
    B = blocks(h)
    log(f"RQ3b: {len(h):,} complete hours {h['hk'].min()} -> {h['hk'].max()}; features d1 {len(feats['d1'])}, "
        f"h1 {len(feats['h1'])}, pf {len(feats['pf'])}; LightGBM {'yes' if lgb else 'NO (gbm skipped)'}")
    log(f"  calibration {CALIB}" + (f" ({CALIB_BLOCKS} OOF blocks of 6 months)" if CALIB.endswith("_oof") else "")
        + f"; LightGBM init {GBM_INIT}")
    log(f"  spike hour share: short {h[TARGETS['short']].mean():.4f}, long {h[TARGETS['long']].mean():.4f}")
    step = 12 if a.fast else 6
    folds = []
    s = TEST_START
    end = h["hk"].max().tz_localize(None) if h["hk"].dt.tz is not None else h["hk"].max()
    while s <= end:
        folds.append((s, s + pd.DateOffset(months=step)))
        s = s + pd.DateOffset(months=step)
    hk = h["hk"].dt.tz_localize(None) if h["hk"].dt.tz is not None else h["hk"]
    preds = []
    for side, target in TARGETS.items():
        exp_, roll_ = climatology(h, target)
        base = pd.DataFrame({"hk": hk, "side": side, "y": h[target].astype(int), "date": h["date"],
                             "clim_exp": exp_, "clim_roll": roll_})
        out = {k: np.full(len(h), np.nan) for k in
               ["persist_d1", "cal_hist_d1", "lr_d1", "gbm_d1", "gbm_pf_d1",
                "persist_h1", "lr_h1", "gbm_h1", "gbm_pf_h1"]}
        other = "long" if side == "short" else "short"
        if FIXED is not None:
            p24, p4 = f"fixhist_{side}__lb24h_d1", f"fixhist_{side}__lb4h_h1"
        else:
            p24 = f"spikehist_{side}_q{'990' if side == 'short' else '010'}__lb24h_d1"
            p4 = f"spikehist_{side}_q{'990' if side == 'short' else '010'}__lb4h_h1"
        for (f0, f1) in folds:
            tr_m = ((hk < f0 - EMBARGO) & (hk >= TRAIN_FROM)).to_numpy()
            te_m = ((hk >= f0) & (hk < f1)).to_numpy()
            if te_m.sum() == 0:
                continue
            tr, te = h[tr_m], h[te_m]
            if tr[target].sum() < 30:
                continue
            out["persist_d1"][te_m] = lr_fit_predict(tr, te, [p24], target)
            out["persist_h1"][te_m] = lr_fit_predict(tr, te, [p24, p4], target)
            out["cal_hist_d1"][te_m] = lr_fit_predict(tr, te, B["cal"] + [p24], target)
            cs = f0 - EMBARGO - pd.DateOffset(months=6)
            fit_m = ((hk < cs - EMBARGO) & (hk >= TRAIN_FROM)).to_numpy()
            cal_m = ((hk >= cs) & (hk < f0 - EMBARGO) & (hk >= TRAIN_FROM)).to_numpy()
            trf, cal = h[fit_m], h[cal_m]
            if trf[target].sum() < 30:
                trf, cal = tr, tr.iloc[:0]
            f_lr = lambda t, x, c, tg: lr_fit_predict(t, x, c, tg, 30.0)
            cl = B["cal"] + B["fund"] + B["hist"]
            hk_tr = hk[tr_m].to_numpy()

            def fitcal(fn, cols_):
                if CALIB in ("platt", "isotonic"):
                    return platt(fn, trf, cal, te, cols_, target)
                return recal_oof(fn, tr, te, hk_tr, f0, cols_, target)

            CTX.update(side=side, fold=str(f0.date()), model="lr_d1")
            out["lr_d1"][te_m] = fitcal(f_lr, cl)
            CTX.update(model="lr_h1")
            out["lr_h1"][te_m] = fitcal(f_lr, cl + B["now"])
            if lgb is not None:
                for mk, cols_, ic in (("gbm_d1", feats["d1"], cl), ("gbm_pf_d1", feats["d1"] + feats["pf"], cl),
                                      ("gbm_h1", feats["h1"], cl + B["now"]),
                                      ("gbm_pf_h1", feats["h1"] + feats["pf"], cl + B["now"])):
                    CTX.update(side=side, fold=str(f0.date()), model=mk)
                    f_gb = (lambda t, x, c, tg, ic=ic: gbm_fit_predict(t, x, c, tg, a.fast,
                                                                       ic if GBM_INIT == "lr" else None))
                    out[mk][te_m] = fitcal(f_gb, cols_)
            log(f"  {side} fold {f0.date()} -> {f1.date()}: train {tr_m.sum():,} h ({int(tr[target].sum())} pos), "
                f"test {te_m.sum():,} h ({int(te[target].sum())} pos)  [{time.time() - t0:.0f}s]")
        for k, v in out.items():
            base[k] = v
        preds.append(base[base["hk"] >= TEST_START])
    P = pd.concat(preds).reset_index(drop=True)
    P.to_csv(OUT / "predictions.csv", index=False)
    models = [c for c in P.columns if c not in ("hk", "side", "y", "date")]
    P["post_cap"] = P["date"] >= CAP

    rows = []
    for side in TARGETS:
        d = P[P["side"] == side]
        for per, dd in (("all", d), ("from_2023", d[d["date"] >= pd.Timestamp("2023-01-01")]),
                        ("pre_cap", d[~d["post_cap"]]), ("post_cap", d[d["post_cap"]])):
            ok = dd[models].notna().all(axis=1) if lgb is not None else dd[[m for m in models if not m.startswith("gbm")]].notna().all(axis=1)
            dd = dd[ok]
            for m in models:
                if dd[m].isna().all():
                    continue
                rows.append({"side": side, "period": per, "model": m, **metrics(dd["y"].to_numpy(float), dd[m].to_numpy(float))})
    R = pd.DataFrame(rows)
    # Brier skill vs the best baseline (by Brier on the pooled test)
    baselines = ["clim_exp", "clim_roll", "persist_d1", "persist_h1", "cal_hist_d1"]
    best = {}
    for side in TARGETS:
        t = R[(R["side"] == side) & (R["period"] == "all") & R["model"].isin(baselines)]
        best[side] = t.sort_values("Brier").iloc[0]["model"]
    R["best_baseline"] = R["side"].map(best)
    ref = R.merge(R[["side", "period", "model", "Brier"]].rename(columns={"model": "best_baseline", "Brier": "Brier_ref"}),
                  on=["side", "period", "best_baseline"], how="left")
    R["Brier_skill_vs_best_baseline"] = 1 - ref["Brier"].to_numpy() / ref["Brier_ref"].to_numpy()
    R.to_csv(OUT / "metrics.csv", index=False)
    if CAL_LOG:
        pd.DataFrame(CAL_LOG).to_csv(OUT / "calibration_windows.csv", index=False)
    for side in TARGETS:
        for per in ("all", "from_2023", "pre_cap", "post_cap"):
            t = R[(R["side"] == side) & (R["period"] == per)].set_index("model")[
                ["n", "pos", "Brier", "Brier_skill_vs_best_baseline", "logloss", "PR_AUC", "ROC_AUC", "base_rate",
                 "mean_pred", "ECE", "cal_slope"]]
            log(f"\n-- {side}, test period {per}; best baseline (pooled) = {best[side]}")
            log(t.to_string(float_format=lambda x: f"{x:,.4f}"))

    # day-block bootstrap of the Brier gain over the best baseline
    rng = np.random.default_rng(20261010)
    bt = []
    for side in TARGETS:
        d = P[(P["side"] == side)].dropna(subset=[m for m in models if lgb is not None or not m.startswith("gbm")])
        ref_loss = (d[best[side]] - d["y"]) ** 2
        days = d["date"].to_numpy()
        ud, inv = np.unique(days, return_inverse=True)
        for m in ["lr_d1", "gbm_d1", "gbm_pf_d1", "lr_h1", "gbm_h1", "gbm_pf_h1"]:
            if m not in d or d[m].isna().all():
                continue
            diff = (ref_loss - (d[m] - d["y"]) ** 2).to_numpy()
            s_day = np.bincount(inv, weights=diff, minlength=len(ud))
            n_day = np.bincount(inv, minlength=len(ud))
            est = s_day.sum() / n_day.sum()
            draws = []
            for _ in range(1000):
                i = rng.integers(0, len(ud), len(ud))
                draws.append(s_day[i].sum() / n_day[i].sum())
            bt.append({"side": side, "model": m, "best_baseline": best[side], "dBrier": est,
                       "ci_lo": np.percentile(draws, 2.5), "ci_hi": np.percentile(draws, 97.5)})
    BT = pd.DataFrame(bt)
    BT.to_csv(OUT / "brier_gain_bootstrap.csv", index=False)
    log("\n-- Brier gain over the best baseline (positive = model better), day-block bootstrap 95 % CI")
    log(BT.to_string(float_format=lambda x: f"{x:,.5f}"))

    # reliability (deciles of predicted probability) for key models
    rel = []
    for side in TARGETS:
        d = P[P["side"] == side]
        for m in (best[side], "lr_d1", "gbm_d1", "gbm_h1"):
            if m not in d or d[m].isna().all():
                continue
            dd = d.dropna(subset=[m])
            q = pd.qcut(dd[m].rank(method="first"), 10, labels=False)
            t = dd.groupby(q).agg(mean_pred=(m, "mean"), obs_rate=("y", "mean"), n=("y", "size"))
            t.insert(0, "model", m)
            t.insert(0, "side", side)
            rel.append(t.reset_index(names="decile"))
    Rel = pd.concat(rel)
    Rel.to_csv(OUT / "reliability.csv", index=False)
    log("\n-- reliability (decile of predicted probability: mean prediction vs observed rate)")
    log(Rel.to_string(float_format=lambda x: f"{x:,.4f}", index=False))
    # ---- diagnostics: capture rate of the riskiest hours, metrics per year, calibration, plots, importance
    import ml_diagnostics as D
    D.classifier_diagnostics(P, models, OUT, y_col="y", time_col="hk", group_col="side", reference=best,
                             title=f"RQ3b{a.tag}")
    if IMP_LOG:
        per = {}
        for e in IMP_LOG:
            per.setdefault(f"{e['side']}|{e['model']}", []).append(e["imp"])
        IT = D.importance_table(per)
        IT[["side", "model"]] = IT["series"].str.split("|", expand=True); IT = IT.drop(columns="series")
        IT.to_csv(OUT / "diagnostics" / "importance_gbm.csv", index=False)
        log("\n-- LightGBM gain importance (mean share over fits; share of fits in the top 20)")
        for (sd_, mk), g in IT.groupby(["side", "model"], sort=False):
            log(f"   {sd_} {mk}: " + "; ".join(f"{r.feature} {r.mean*100:.1f}% ({r.top20_share_of_folds:.0%})" for r in g.head(10).itertuples()))
            D._try_plot(D._plot_importance, g, OUT / "diagnostics" / "plots" / sd_ / f"importance_{mk}.png", f"RQ3b {sd_} {mk}")
    cap = pd.read_csv(OUT / "diagnostics" / "capture_rate.csv")
    log("\n-- share of all spike hours inside the top 5 % riskiest hours (capture rate)")
    log(cap[cap.top_share == 0.05].pivot(index="model", columns="group", values="capture_rate").to_string(float_format=lambda x: f"{x:.3f}"))
    for side in TARGETS:
        t = R[(R["side"] == side) & (R["period"] == "all")].set_index("model")
        if len(t):
            bm = t["Brier"].idxmin()
            RU.summary(f"{side}: best {bm}, Brier skill {t.at[bm, 'Brier_skill_vs_best_baseline']:+.3f} vs {best[side]}")
    (OUT / "report_rq3b_1.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    RU.run(main, "RQ3b risk model")
