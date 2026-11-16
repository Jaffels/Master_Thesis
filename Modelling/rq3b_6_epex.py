"""RQ3b EPEX sensitivity (appendix): do EPEX SPOT CH market data improve the spike-risk forecast?

A/B design (16 Nov 2026): the SAME models, folds, calibration and rows as rq3b_1_riskmodel.py, once with
the frozen baseline feature set and once with baseline + EPEX features (Clean/build_epex_features.py).
Only the feature set differs, so any difference in skill is the contribution of the EPEX data.

  * Unit, targets, origins: as rq3b_1 (hour; any spike QH; spike_short_q990 / spike_long_q010; d1 = D-1 18:00,
    h1 = 1 h before the hour).
  * Window: EPEX window only - training from 18 Apr 2019 (first CH intraday auction, trades from 2019),
    test 1 Jan 2021 -> 31 Dec 2025 (licence: EPEX historical package up to 2025; 2026 not licensed).
    Rolling origin, expanding window, refit every 6 months, 2-day embargo; calibration as rq3b_1 (--calib).
    Baseline AND EPEX models are re-estimated on this window, so the comparison is like for like; the
    baseline numbers are therefore NOT the main rq3b_1 results (shorter training history).
  * Pairs compared: lr_d1 vs lr_d1_epex, gbm_d1 vs gbm_d1_epex, lr_h1 vs lr_h1_epex, gbm_h1 vs gbm_h1_epex.
      gbm_*_epex  = all EPEX features of the origin (d1: d1 set; h1: d1 + h1 set), minus the four D-1 pre-trade
                    columns (< 3 % non-NaN, see build_epex_features_report.txt).
      lr_*_epex   = logistic baseline blocks + a small curated EPEX block (LR_EPEX below).
  * Output: metrics per period (all / 2021-2022 / 2023-2025 / pre-cap / post-cap), paired day-block bootstrap of
    the Brier and PR-AUC differences (EPEX - baseline), share of LightGBM gain from EPEX features, reliability,
    standard classifier diagnostics (ml_diagnostics), run.log / run_meta.json, macOS notification at the end.

Run from Master_Thesis with .venv active (after Clean/build_epex_features.py --write):
    python Modelling/rq3b_6_epex.py --fast     # smoke test (12-month refits, fewer trees), ~ a few minutes
    python Modelling/rq3b_6_epex.py            # full run -> Modelling/Output/rq3b_6_epex/
Options: --calib (same choices as rq3b_1, default platt), --test-start, --tag.
Read-only on Clean/Data; writes only Modelling/Output/rq3b_6_epex<tag>/.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import run_utils as RU

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Clean"))
sys.path.insert(0, str(HERE))
import views as V  # noqa: E402
import rq3b_1_riskmodel as M  # noqa: E402  (models, calibration, metrics re-used unchanged)

warnings.filterwarnings("ignore")

EPEX_FILE = V.VIEWS_DIR / "rq3_epex.parquet"
EPEX_DICT = V.VIEWS_DIR / "_rq3_epex_dictionary.csv"
WIN_START = pd.Timestamp("2019-04-18")          # local
WIN_END = pd.Timestamp("2026-01-01")            # local, exclusive (licence)
SPARSE_D1 = ["ch_idc_vwap_pre_eur_mwh__fc_d1", "ch_idc_spread_pre_da_eur_mwh__fc_d1",
             "ch_idc_vol_pre_mwh__fc_d1", "ch_idc_ntrades_pre_n__fc_d1"]
LR_EPEX = {
    "d1": ["ch_idc_da_absspread_eur_mwh__lb24h_d1", "ch_idc_da_absspread_eur_mwh__lb7d_d1",
           "ch_idc_da_spread_eur_mwh__lb24h_d1", "ch_idc_mkt_spread_eur_mwh__lb3h_d1", "ch_idc_vol_mwh__lb24h_d1"],
    "h1": ["ch_idc_da_absspread_eur_mwh__lb4h_h1", "ch_idc_da_spread_eur_mwh__lb4h_h1",
           "ch_idc_spread_pre_da_eur_mwh__nc_h1", "ch_idc_range_pre_eur_mwh__nc_h1",
           "ch_idc_mkt_spread_eur_mwh__lb1h_h1", "ch_idc_vol_pre_mwh__nc_h1"],
}
PAIRS = [("lr_d1", "lr_d1_epex"), ("gbm_d1", "gbm_d1_epex"), ("lr_h1", "lr_h1_epex"), ("gbm_h1", "gbm_h1_epex")]
PERIODS = {"all": (None, None), "2021_2022": ("2021-01-01", "2023-01-01"), "2023_2025": ("2023-01-01", "2026-01-01"),
           "pre_cap": (None, str(M.CAP.date())), "post_cap": (str(M.CAP.date()), None)}
OUT = HERE / "Output" / "rq3b_6_epex"
log = M.log


def epex_columns(origin: str) -> list[str]:
    d = pd.read_csv(EPEX_DICT)
    ok = d["role"].isin(["forecast", "lookback", "lag", "nowcast"])
    orig = {"d1": {"d1"}, "h1": {"d1", "h1"}}[origin]
    cols = d.loc[ok & d["origin"].isin(orig), "column"].tolist()
    return [c for c in cols if c not in SPARSE_D1]


def build() -> tuple[pd.DataFrame, dict, dict]:
    if not EPEX_FILE.exists():
        raise SystemExit(f"{EPEX_FILE} missing: run python Clean/build_epex_features.py --write first")
    orig_load = V.load_rq3

    def load_with_epex(columns=None, weather_fc=False, school_holidays=False):
        df = orig_load(columns, weather_fc=weather_fc, school_holidays=school_holidays)
        ep = pd.read_parquet(EPEX_FILE)
        ep["ts_utc"] = pd.to_datetime(ep["ts_utc"], utc=True)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        n = len(df)
        df = df.merge(ep, on="ts_utc", how="left", validate="one_to_one")
        assert len(df) == n
        return df

    V.load_rq3 = load_with_epex
    try:
        h, feats = M.build_hourly()
    finally:
        V.load_rq3 = orig_load
    loc = h["date"]
    h = h[(loc >= WIN_START) & (loc < WIN_END)].reset_index(drop=True)
    ep = {o: [c for c in epex_columns(o) if c in h.columns] for o in ("d1", "h1")}
    lr_ep = {o: [c for c in LR_EPEX[o] if c in h.columns] for o in ("d1", "h1")}
    # baseline sets must not contain EPEX columns (they are not in _rq3_dictionary.csv, so they never are)
    allep = set(ep["h1"])
    assert not (set(feats["d1"]) | set(feats["h1"])) & allep
    return h, feats, {"gbm": ep, "lr": lr_ep}


def period_mask(d: pd.DataFrame, per: str) -> pd.Series:
    a, b = PERIODS[per]
    m = pd.Series(True, index=d.index)
    if a:
        m &= d["date"] >= pd.Timestamp(a)
    if b:
        m &= d["date"] < pd.Timestamp(b)
    return m


def paired_bootstrap(d: pd.DataFrame, base: str, alt: str, rng, n_draw=1000, n_draw_ap=300) -> dict:
    y = d["y"].to_numpy(float)
    pb, pa = d[base].to_numpy(float), d[alt].to_numpy(float)
    diff = (pb - y) ** 2 - (pa - y) ** 2                       # > 0 = EPEX better
    ud, inv = np.unique(d["date"].to_numpy(), return_inverse=True)
    s_day = np.bincount(inv, weights=diff, minlength=len(ud))
    n_day = np.bincount(inv, minlength=len(ud))
    est = s_day.sum() / n_day.sum()
    draws = [s_day[i].sum() / n_day[i].sum() for i in (rng.integers(0, len(ud), len(ud)) for _ in range(n_draw))]
    ap_est = M.avg_precision(y, pa) - M.avg_precision(y, pb)
    rows_by_day = np.split(np.argsort(inv, kind="stable"), np.cumsum(n_day)[:-1])
    ap_draws = []
    for _ in range(n_draw_ap):
        i = rng.integers(0, len(ud), len(ud))
        idx = np.concatenate([rows_by_day[k] for k in i])
        if y[idx].sum() < 5:
            continue
        ap_draws.append(M.avg_precision(y[idx], pa[idx]) - M.avg_precision(y[idx], pb[idx]))
    ref_b = float(((pb - y) ** 2).mean())
    return {"base": base, "epex": alt, "Brier_base": ref_b, "dBrier": est,
            "dBrier_rel": est / ref_b if ref_b > 0 else np.nan,
            "dBrier_lo": np.percentile(draws, 2.5), "dBrier_hi": np.percentile(draws, 97.5),
            "P(dBrier>0)": float(np.mean(np.array(draws) > 0)),
            "dPR_AUC": ap_est, "dPR_AUC_lo": np.percentile(ap_draws, 2.5) if ap_draws else np.nan,
            "dPR_AUC_hi": np.percentile(ap_draws, 97.5) if ap_draws else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--calib", choices=["platt", "isotonic", "platt_oof", "isotonic_oof", "none"], default="platt")
    ap.add_argument("--calib-blocks", type=int, default=3)
    ap.add_argument("--test-start", default="2021-01-01")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global OUT
    M.CALIB, M.CALIB_BLOCKS, M.GBM_INIT = a.calib, a.calib_blocks, "none"
    test_start = pd.Timestamp(a.test_start)
    OUT = HERE / "Output" / ("rq3b_6_epex" + a.tag)
    OUT.mkdir(parents=True, exist_ok=True)
    RU.set_output(OUT)
    t0 = time.time()

    h, feats, EP = build()
    B = M.blocks(h)
    log(f"RQ3b EPEX A/B: {len(h):,} complete hours {h['hk'].min()} -> {h['hk'].max()} (EPEX window)")
    log(f"  baseline features: d1 {len(feats['d1'])}, h1 {len(feats['h1'])}; EPEX add-on: gbm d1 +{len(EP['gbm']['d1'])}, "
        f"gbm h1 +{len(EP['gbm']['h1'])}, lr d1 +{len(EP['lr']['d1'])}, lr h1 +{len(EP['lr']['h1'])}")
    log(f"  calibration {M.CALIB}; LightGBM {'yes' if M.lgb else 'NO (gbm skipped)'}; test from {test_start.date()}")
    log(f"  EPEX add-on (gbm h1): {', '.join(EP['gbm']['h1'])}")
    log(f"  spike hour share: short {h['spike_short_q990'].mean():.4f}, long {h['spike_long_q010'].mean():.4f}")
    cov = h[EP["gbm"]["h1"]].notna().groupby(h["date"].dt.year).mean().T.round(2)
    log("  EPEX feature coverage per year (hourly rows):\n" + cov.to_string())

    step = 12 if a.fast else 6
    hk = h["hk"].dt.tz_localize(None) if h["hk"].dt.tz is not None else h["hk"]
    end = hk.max()
    folds, s = [], test_start
    while s <= end:
        folds.append((s, s + pd.DateOffset(months=step)))
        s = s + pd.DateOffset(months=step)
    models = ["persist_d1", "persist_h1", "lr_d1", "lr_d1_epex", "lr_h1", "lr_h1_epex",
              "gbm_d1", "gbm_d1_epex", "gbm_h1", "gbm_h1_epex"]
    preds = []
    for side, target in M.TARGETS.items():
        exp_, roll_ = M.climatology(h, target)
        base = pd.DataFrame({"hk": hk, "side": side, "y": h[target].astype(int), "date": h["date"],
                             "clim_exp": exp_, "clim_roll": roll_})
        out = {k: np.full(len(h), np.nan) for k in models}
        q = "990" if side == "short" else "010"
        p24, p4 = f"spikehist_{side}_q{q}__lb24h_d1", f"spikehist_{side}_q{q}__lb4h_h1"
        for f0, f1 in folds:
            tr_m = (hk < f0 - M.EMBARGO).to_numpy()
            te_m = ((hk >= f0) & (hk < f1)).to_numpy()
            if te_m.sum() == 0:
                continue
            tr, te = h[tr_m], h[te_m]
            if tr[target].sum() < 30:
                log(f"  {side} fold {f0.date()}: only {int(tr[target].sum())} training spikes - skipped")
                continue
            out["persist_d1"][te_m] = M.lr_fit_predict(tr, te, [p24], target)
            out["persist_h1"][te_m] = M.lr_fit_predict(tr, te, [p24, p4], target)
            cs = f0 - M.EMBARGO - pd.DateOffset(months=6)
            fit_m = (hk < cs - M.EMBARGO).to_numpy()
            cal_m = ((hk >= cs) & (hk < f0 - M.EMBARGO)).to_numpy()
            trf, cal = h[fit_m], h[cal_m]
            if trf[target].sum() < 30:
                trf, cal = tr, tr.iloc[:0]
            hk_tr = hk[tr_m].to_numpy()

            def fitcal(fn, cols_):
                if M.CALIB in ("platt", "isotonic"):
                    return M.platt(fn, trf, cal, te, cols_, target)
                return M.recal_oof(fn, tr, te, hk_tr, f0, cols_, target)

            f_lr = lambda t, x, c, tg: M.lr_fit_predict(t, x, c, tg, 30.0)  # noqa: E731
            cl = B["cal"] + B["fund"] + B["hist"]
            specs = [("lr_d1", f_lr, cl), ("lr_d1_epex", f_lr, cl + EP["lr"]["d1"]),
                     ("lr_h1", f_lr, cl + B["now"]), ("lr_h1_epex", f_lr, cl + B["now"] + EP["lr"]["d1"] + EP["lr"]["h1"])]
            if M.lgb is not None:
                f_gb = lambda t, x, c, tg: M.gbm_fit_predict(t, x, c, tg, a.fast, None)  # noqa: E731
                specs += [("gbm_d1", f_gb, feats["d1"]), ("gbm_d1_epex", f_gb, feats["d1"] + EP["gbm"]["d1"]),
                          ("gbm_h1", f_gb, feats["h1"]), ("gbm_h1_epex", f_gb, feats["h1"] + EP["gbm"]["h1"])]
            for mk, fn, cols_ in specs:
                M.CTX.update(side=side, fold=str(f0.date()), model=mk)
                out[mk][te_m] = fitcal(fn, cols_)
            log(f"  {side} fold {f0.date()} -> {f1.date()}: train {tr_m.sum():,} h ({int(tr[target].sum())} pos), "
                f"test {te_m.sum():,} h ({int(te[target].sum())} pos)  [{time.time() - t0:.0f}s]")
        for k, v in out.items():
            base[k] = v
        preds.append(base[base["hk"] >= test_start])
    P = pd.concat(preds).reset_index(drop=True)
    P.to_csv(OUT / "predictions.csv", index=False)
    mcols = [c for c in P.columns if c not in ("hk", "side", "y", "date") and P[c].notna().any()]

    # ---- metrics per period (rows where every model has a prediction)
    rows = []
    for side in M.TARGETS:
        d0 = P[P["side"] == side].dropna(subset=mcols)
        for per in PERIODS:
            dd = d0[period_mask(d0, per)]
            if dd["y"].sum() < 5:
                continue
            for m in mcols:
                rows.append({"side": side, "period": per, "model": m,
                             **M.metrics(dd["y"].to_numpy(float), dd[m].to_numpy(float))})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "metrics.csv", index=False)
    show = ["n", "pos", "Brier", "logloss", "PR_AUC", "ROC_AUC", "base_rate", "mean_pred", "ECE", "cal_slope"]
    for side in M.TARGETS:
        for per in PERIODS:
            t = R[(R["side"] == side) & (R["period"] == per)].set_index("model")
            if len(t):
                log(f"\n-- {side}, period {per}")
                log(t[show].to_string(float_format=lambda x: f"{x:,.4f}"))

    # ---- paired comparison: EPEX vs baseline, day-block bootstrap
    rng = np.random.default_rng(20261116)
    bt = []
    for side in M.TARGETS:
        d0 = P[P["side"] == side].dropna(subset=mcols)
        for per in PERIODS:
            dd = d0[period_mask(d0, per)]
            if dd["y"].sum() < 10:
                continue
            for b, e in PAIRS:
                if b in dd and e in dd:
                    bt.append({"side": side, "period": per, **paired_bootstrap(dd, b, e, rng)})
    BT = pd.DataFrame(bt)
    BT.to_csv(OUT / "epex_vs_baseline_bootstrap.csv", index=False)
    log("\n== EPEX vs baseline (paired, day-block bootstrap 95 % CI; dBrier > 0 and dPR_AUC > 0 = EPEX better)")
    for side in M.TARGETS:
        t = BT[BT["side"] == side].drop(columns="side")
        log(f"\n-- {side}")
        log(t.to_string(index=False, float_format=lambda x: f"{x:,.5f}"))

    # ---- LightGBM: share of gain from EPEX features
    if M.IMP_LOG:
        allep = set(EP["gbm"]["h1"])
        sh = []
        for e in M.IMP_LOG:
            if not e["model"].endswith("_epex"):
                continue
            imp = e["imp"]
            sh.append({"side": e["side"], "model": e["model"], "fold": e["fold"],
                       "epex_gain_share": float(imp[[c for c in imp.index if c in allep]].sum())})
        SH = pd.DataFrame(sh)
        import ml_diagnostics as D
        per = {}
        for e in M.IMP_LOG:
            per.setdefault(f"{e['side']}|{e['model']}", []).append(e["imp"])
        IT = D.importance_table(per)
        IT[["side", "model"]] = IT["series"].str.split("|", expand=True)
        IT = IT.drop(columns="series")
        IT["is_epex"] = IT["feature"].isin(allep)
        (OUT / "diagnostics").mkdir(exist_ok=True)
        IT.to_csv(OUT / "diagnostics" / "importance_gbm.csv", index=False)
        if len(SH):
            SH.to_csv(OUT / "epex_gain_share.csv", index=False)
            log("\n-- LightGBM: share of total gain from EPEX features (mean over folds)")
            log(SH.groupby(["side", "model"])["epex_gain_share"].agg(["mean", "min", "max"]).round(3).to_string())
        log("\n-- top EPEX features by mean gain share (EPEX models)")
        for (sd_, mk), g in IT[IT["model"].str.endswith("_epex")].groupby(["side", "model"], sort=False):
            ge = g[g["is_epex"]].head(8)
            log(f"   {sd_} {mk}: " + "; ".join(f"{r.feature} {r.mean * 100:.1f}%" for r in ge.itertuples()))

    # ---- reliability + standard classifier diagnostics
    rel = []
    for side in M.TARGETS:
        d = P[P["side"] == side]
        for m in ("gbm_d1", "gbm_d1_epex", "gbm_h1", "gbm_h1_epex"):
            if m not in d or d[m].isna().all():
                continue
            dd = d.dropna(subset=[m])
            qd = pd.qcut(dd[m].rank(method="first"), 10, labels=False)
            t = dd.groupby(qd).agg(mean_pred=(m, "mean"), obs_rate=("y", "mean"), n=("y", "size"))
            t.insert(0, "model", m)
            t.insert(0, "side", side)
            rel.append(t.reset_index(names="decile"))
    if rel:
        pd.concat(rel).to_csv(OUT / "reliability.csv", index=False)
    import ml_diagnostics as D
    ref = {s: "persist_h1" for s in M.TARGETS}
    D.classifier_diagnostics(P, mcols, OUT, y_col="y", time_col="hk", group_col="side", reference=ref,
                             title=f"RQ3b EPEX{a.tag}")
    cap_f = OUT / "diagnostics" / "capture_rate.csv"
    if cap_f.exists():
        cap = pd.read_csv(cap_f)
        log("\n-- share of all spike hours inside the top 5 % riskiest hours (capture rate)")
        log(cap[cap.top_share == 0.05].pivot(index="model", columns="group", values="capture_rate")
            .to_string(float_format=lambda x: f"{x:.3f}"))

    for side in M.TARGETS:
        t = BT[(BT["side"] == side) & (BT["period"] == "all")]
        for r in t.itertuples():
            if r.base.startswith("gbm"):
                RU.summary(f"{side} {r.base}: dBrier {r.dBrier_rel:+.1%} "
                           f"[{r.dBrier_lo:+.5f}, {r.dBrier_hi:+.5f}], dPR-AUC {r.dPR_AUC:+.3f}")
    (OUT / "report_rq3b_6_epex.txt").write_text("\n".join(M.lines), encoding="utf-8")
    print(f"\nwritten: {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    RU.run(main, "RQ3b EPEX A/B")
