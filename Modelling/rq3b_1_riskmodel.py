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
PF = ["ch_temp_lw_degc__pf", "ch_ghi_lw_wm2__pf", "ch_ghi_ramp_lw_wm2__pf", "ch_wind_lw_ms__pf",
      "ch_hdh_lw_kh__pf", "ch_cdh_lw_kh__pf", "ch_precip_hydro_mm__pf", "ch_snow_hydro_cm__pf",
      "ch_melt_dh_hydro_kh__pf"]
REGIME_MAP = {"mixed": 0, "hourly_dominant": 1, "qh": 2, "dual": 0, "single": 1}
lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


# ------------------------------------------------------------------ data
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
    spikes = list(TARGETS.values())
    for s in spikes:
        df[s] = df[s].fillna(0).astype("int8")
    num = [c for c in df.columns if c not in ("ts_utc", "ts_local", "hk") and pd.api.types.is_numeric_dtype(df[c])]
    fc = [c for c in num if c.endswith("__fc_d1") or c.startswith("da_")]
    agg = {c: "first" for c in num}
    agg.update({c: "mean" for c in fc})
    agg.update({s: "max" for s in spikes})
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
    for k in list(feats):
        feats[k] = [c for c in feats[k] if c in h.columns and c != "cal_year"]
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


def gbm_fit_predict(tr, te, cols, target, fast):
    cols = [c for c in cols if tr[c].notna().any() and tr[c].nunique() > 1]
    params = dict(objective="binary", learning_rate=0.04, num_leaves=15, min_data_in_leaf=40, bagging_fraction=0.8,
                  bagging_freq=1, feature_fraction=0.5, lambda_l2=5.0, num_threads=4, verbose=-1, seed=7)
    ds = lgb.Dataset(tr[cols].astype("float32"), label=tr[target].to_numpy(), free_raw_data=True)
    m = lgb.train(params, ds, num_boost_round=150 if fast else 350)
    return np.clip(m.predict(te[cols].astype("float32")), 1e-5, 1 - 1e-5)


def logloss(y, p):
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def metrics(y, p) -> dict:
    return {"n": len(y), "pos": int(y.sum()), "Brier": float(((p - y) ** 2).mean()), "logloss": logloss(y, p),
            "PR_AUC": avg_precision(y, p), "ROC_AUC": auc(y, p), "base_rate": float(y.mean())}


def platt(fn, tr_fit, cal, te, *args):
    """Fit fn on tr_fit, then recalibrate (slope + intercept on logit) on the later calibration window so
    that the probability LEVEL follows the recent spike rate (the base rate drifts 0.2 - 10 % across years)."""
    pr = fn(tr_fit, pd.concat([cal, te]), *args)
    pc, pt = pr[: len(cal)], pr[len(cal):]
    target = args[1]
    y = cal[target].to_numpy(float)
    if y.sum() < 10 or len(cal) < 500:
        return pt
    z = lambda p: np.log(p / (1 - p))
    zc = z(pc)
    mu, sd = zc.mean(), zc.std() or 1.0
    beta, _ = fit_logit(((zc - mu) / sd)[:, None], y, 0.1)
    return np.clip(predict(beta, ((z(pt) - mu) / sd)[:, None]), 1e-5, 1 - 1e-5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h, feats = build_hourly()
    B = blocks(h)
    log(f"RQ3b: {len(h):,} complete hours {h['hk'].min()} -> {h['hk'].max()}; features d1 {len(feats['d1'])}, "
        f"h1 {len(feats['h1'])}, pf {len(feats['pf'])}; LightGBM {'yes' if lgb else 'NO (gbm skipped)'}")
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
        p24 = f"spikehist_{side}_q{'990' if side == 'short' else '010'}__lb24h_d1"
        p4 = f"spikehist_{side}_q{'990' if side == 'short' else '010'}__lb4h_h1"
        for (f0, f1) in folds:
            tr_m = (hk < f0 - EMBARGO).to_numpy()
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
            fit_m = (hk < cs - EMBARGO).to_numpy()
            cal_m = ((hk >= cs) & (hk < f0 - EMBARGO)).to_numpy()
            trf, cal = h[fit_m], h[cal_m]
            if trf[target].sum() < 30:
                trf, cal = tr, tr.iloc[:0]
            f_lr = lambda t, x, c, tg: lr_fit_predict(t, x, c, tg, 30.0)
            f_gb = lambda t, x, c, tg: gbm_fit_predict(t, x, c, tg, a.fast)
            cl = B["cal"] + B["fund"] + B["hist"]
            out["lr_d1"][te_m] = platt(f_lr, trf, cal, te, cl, target)
            out["lr_h1"][te_m] = platt(f_lr, trf, cal, te, cl + B["now"], target)
            if lgb is not None:
                out["gbm_d1"][te_m] = platt(f_gb, trf, cal, te, feats["d1"], target)
                out["gbm_pf_d1"][te_m] = platt(f_gb, trf, cal, te, feats["d1"] + feats["pf"], target)
                out["gbm_h1"][te_m] = platt(f_gb, trf, cal, te, feats["h1"], target)
                out["gbm_pf_h1"][te_m] = platt(f_gb, trf, cal, te, feats["h1"] + feats["pf"], target)
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
    for side in TARGETS:
        for per in ("all", "from_2023", "pre_cap", "post_cap"):
            t = R[(R["side"] == side) & (R["period"] == per)].set_index("model")[
                ["n", "pos", "Brier", "Brier_skill_vs_best_baseline", "logloss", "PR_AUC", "ROC_AUC", "base_rate"]]
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
    (OUT / "report_rq3b_1.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
