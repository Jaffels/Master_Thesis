"""RQ3 (done_todo 2.3 item 12): quantile LightGBM on the imbalance price itself (direct tail quantiles) as an
alternative to the spike classifier of rq3b_1.

Target per UTC hour (same hours, origins, features and folds as rq3b_1):
    z_short = max over the 4 QH of (short price - thr_short_q990)      spike hour  <=>  z_short > 0
    z_long  = max over the 4 QH of (thr_long_q010 - long price)        spike hour  <=>  z_long  > 0
    modelled on the scale asinh(z / 50) (quantiles are invariant to this monotone transform; it only tames the
    gradient steps for EUR 10,000 outliers). Extra feature: the threshold two days earlier (known at both origins).
Models: LightGBM objective=quantile for tau in --taus (default 0.5 0.75 0.9 0.95 0.975 0.99 0.995), d1 and h1
feature sets (rq3_feature_columns, no perfect-forecast weather), refit every 6 months from 1 Jan 2021, 2-day embargo.
Quantile crossing fixed by sorting (rearrangement).
Benchmark quantiles: clim365 = empirical tau-quantiles of z in the same (hour x day type) cell over the 365 days
before the fold.

Implied spike probability P(z > 0): tau at which the predicted quantile curve crosses 0, interpolated linearly in
log(1 - tau); above the highest tau an exponential tail from the two highest quantiles; below tau = 0.5 linear in
log(tau). -> qgbm_d1, qgbm_h1, qclim (from clim365).

Evaluation
  quantiles: pinball loss per tau (EUR and asinh scale), coverage (share z <= q_tau), tail score = mean pinball over
             tau >= 0.95 (EUR); periods all / 2021_22 / from_2023 / post_cap; day-block bootstrap (1,000) of the tail
             score gain vs clim365.
  probability: merged with rq3b_1 (baselines + logistic Platt, Output/rq3b_1), LightGBM classifier main spec
             (Output/rq3b_1_raw) and, if present, Hawkes (Output/rq3b_4): Brier, skill vs best baseline, log loss,
             PR / ROC-AUC, ECE, top-5 % capture; bootstrap of the Brier gain vs best baseline and vs gbm_h1 / gbm_d1.

Outputs (Modelling/Output/rq3b_5<tag>/): quantile_predictions.parquet, quantile_scores.csv, quantile_bootstrap.csv,
predictions.csv, metrics.csv, brier_gain_bootstrap.csv, importance_q.csv, report_rq3b_5.txt, diagnostics/.

Run from Master_Thesis with .venv active, after rq3b_1 and rq3b_1_raw (rq3b_4 optional) (~30-60 min):
    python Modelling/rq3b_5_quantile.py                  # -> Modelling/Output/rq3b_5/
    options: --fast (12-month refits, 150 trees), --taus ..., --scale 50, --tag NAME
Read-only on Clean/Data and Output/rq3b_1*, rq3b_4; writes only Modelling/Output/rq3b_5<tag>/.
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
import rq3b_1_riskmodel as R1  # noqa: E402  (build_hourly: same hours / features as rq3b_1)

warnings.filterwarnings("ignore")
import lightgbm as lgb  # noqa: E402

OUT = HERE / "Output" / "rq3b_5"
SIDES = {"short": ("tgt_price_short_eur_mwh", "thr_short_q990", 1.0, "spike_short_q990"),
         "long": ("tgt_price_long_eur_mwh", "thr_long_q010", -1.0, "spike_long_q010")}
TEST_START = pd.Timestamp("2021-01-01")
CAP = pd.Timestamp("2025-03-03")
EMBARGO = pd.Timedelta(days=2)
BASELINES = ["clim_exp", "clim_roll", "persist_d1", "persist_h1", "cal_hist_d1"]
lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


# ------------------------------------------------------------------ data
def hourly_targets() -> pd.DataFrame:
    """z per UTC hour (max over the QH of the signed excess over the rolling threshold) + threshold 2 days earlier."""
    cols = ["ts_utc"] + [c for v in SIDES.values() for c in v[:2]]
    df = V.load_rq3(columns=cols)
    t = pd.to_datetime(df["ts_utc"])
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t
    df["hk"] = t.dt.floor("h")
    out = {}
    for side, (price, thr, sign, _) in SIDES.items():
        df[f"z_{side}"] = sign * (df[price] - df[thr])
        out[f"z_{side}"] = "max"
        out[thr] = "first"
    H = df.groupby("hk").agg(out)
    for side, (_, thr, _, _) in SIDES.items():
        H[f"thrlag2d_{side}"] = H[thr].shift(48)          # hourly grid assumed; checked below
    H = H.drop(columns=[v[1] for v in SIDES.values()])
    gap = H.index.to_series().diff().dt.total_seconds().ne(3600).sum()
    log(f"  hourly targets: {len(H):,} hours, {gap} grid breaks (thrlag2d = value 48 rows earlier)")
    return H.reset_index()


def tf(z, sc):
    return np.arcsinh(z / sc)


def itf(v, sc):
    return sc * np.sinh(v)


def prob_above_zero(Q: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """P(z > 0) from sorted quantiles Q (n, K) on any monotone scale where 0 maps to 0."""
    n, K = Q.shape
    p = np.empty(n)
    lt = np.log(1 - taus)
    for i in range(n):
        q = Q[i]
        if q[-1] <= 0:                                   # above the top quantile: exponential tail
            dq = q[-1] - q[-2]
            if dq <= 1e-9:
                p[i] = (1 - taus[-1]) * 1e-3
            else:
                slope = (lt[-1] - lt[-2]) / dq           # d log(1-tau) / dq  (< 0)
                p[i] = np.exp(lt[-1] + slope * (0 - q[-1]))
        elif q[0] > 0:                                   # below the lowest quantile
            dq = q[1] - q[0]
            lo = np.log(taus)
            slope = (lo[1] - lo[0]) / dq if dq > 1e-9 else 50.0
            p[i] = 1 - np.exp(lo[0] + slope * (0 - q[0]))
        else:
            j = np.searchsorted(q, 0.0)                  # q[j-1] < 0 <= q[j]
            a, b = q[j - 1], q[j]
            w = 0.0 if b - a < 1e-12 else (0 - a) / (b - a)
            p[i] = np.exp(lt[j - 1] + w * (lt[j] - lt[j - 1]))
    return np.clip(p, 1e-5, 1 - 1e-5)


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def boot(diff, days, rng, n=1000):
    ud, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(ud))
    c = np.bincount(inv, minlength=len(ud))
    draws = np.array([s[i].sum() / c[i].sum() for i in (rng.integers(0, len(ud), len(ud)) for _ in range(n))])
    return s.sum() / c.sum(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def scores(y, p) -> dict:
    from rq3a_2_logit import auc, avg_precision
    q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).to_numpy()
    n = np.bincount(q, minlength=10).astype(float)
    mp = np.bincount(q, weights=p, minlength=10) / np.maximum(n, 1)
    my = np.bincount(q, weights=y, minlength=10) / np.maximum(n, 1)
    k = max(1, int(round(0.05 * len(y))))
    top = np.argsort(-p)[:k]
    return {"n": len(y), "pos": int(y.sum()), "base_rate": y.mean(), "mean_pred": p.mean(),
            "Brier": float(((p - y) ** 2).mean()), "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "PR_AUC": avg_precision(y, p), "ROC_AUC": auc(y, p), "ECE": float((n * np.abs(mp - my)).sum() / n.sum()),
            "capture_top5": y[top].sum() / max(y.sum(), 1)}


# ------------------------------------------------------------------ main
def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995])
    ap.add_argument("--scale", type=float, default=50.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    taus = np.array(sorted(a.taus))
    OUT = HERE / "Output" / ("rq3b_5" + a.tag)
    (OUT / "diagnostics").mkdir(parents=True, exist_ok=True)
    RU.set_output(OUT)
    t0 = time.time()
    h, feats = R1.build_hourly()
    Z = hourly_targets()
    hk_naive = h["hk"].dt.tz_localize(None) if h["hk"].dt.tz is not None else h["hk"]
    Zk = Z["hk"].dt.tz_localize(None) if Z["hk"].dt.tz is not None else Z["hk"]
    Z = Z.assign(hk=Zk).set_index("hk")
    h = h.assign(hk=hk_naive)
    for c in Z.columns:
        h[c] = Z[c].reindex(h["hk"]).to_numpy()
    sc = a.scale
    for side, (_, _, _, flag) in SIDES.items():
        y, z = h[flag].to_numpy(float), h[f"z_{side}"].to_numpy(float)
        ok = np.isfinite(z)
        log(f"  {side}: z finite {ok.mean():.2%}; (z > 0) == spike flag in {((z[ok] > 0) == (y[ok] > 0)).mean():.4%} of hours")
    log(f"RQ3 quantile LightGBM: {len(h):,} hours; features d1 {len(feats['d1'])}, h1 {len(feats['h1'])}; "
        f"taus {taus.tolist()}; scale asinh(z/{sc:g})")
    step = 12 if a.fast else 6
    rounds = 150 if a.fast else 350
    folds, s = [], TEST_START
    while s <= h["hk"].max():
        folds.append((s, s + pd.DateOffset(months=step)))
        s = s + pd.DateOffset(months=step)
    hk = h["hk"]
    daytype_cell = h["cell"].to_numpy()
    params = dict(objective="quantile", learning_rate=0.04, num_leaves=15, min_data_in_leaf=40, bagging_fraction=0.8,
                  bagging_freq=1, feature_fraction=0.5, lambda_l2=5.0, num_threads=4, verbose=-1, seed=7)
    K = len(taus)
    QP = {}            # (side, model) -> (n, K) asinh-scale quantiles
    imp = []
    for side in SIDES:
        zt = tf(h[f"z_{side}"].to_numpy(float), sc)
        for mk in ("qgbm_d1", "qgbm_h1", "clim365"):
            QP[(side, mk)] = np.full((len(h), K), np.nan)
        for f0, f1 in folds:
            tr_m = ((hk < f0 - EMBARGO) & np.isfinite(zt)).to_numpy()
            te_m = ((hk >= f0) & (hk < f1)).to_numpy()
            if te_m.sum() == 0 or tr_m.sum() < 1000:
                continue
            # climatology quantiles: same cell, last 365 days before the fold
            rec = tr_m & (hk >= f0 - EMBARGO - pd.Timedelta(days=365)).to_numpy()
            cq = pd.DataFrame(zt[rec]).groupby(daytype_cell[rec])[0].quantile(taus.tolist()).unstack()
            allq = np.quantile(zt[rec], taus)
            QP[(side, "clim365")][te_m] = cq.reindex(daytype_cell[te_m]).fillna(pd.Series(allq, index=cq.columns)).to_numpy()
            for org in ("d1", "h1"):
                cols = [c for c in feats[org] if h.loc[tr_m, c].notna().any() and h.loc[tr_m, c].nunique() > 1]
                cols = cols + [f"thrlag2d_{side}"]
                Xtr = h.loc[tr_m, cols].astype("float32")
                Xte = h.loc[te_m, cols].astype("float32")
                Q = np.empty((te_m.sum(), K))
                for k, tau in enumerate(taus):
                    m = lgb.train(params | {"alpha": float(tau)}, lgb.Dataset(Xtr, label=zt[tr_m]), num_boost_round=rounds)
                    Q[:, k] = m.predict(Xte)
                    if tau in (0.95, 0.99):
                        g = pd.Series(m.feature_importance("gain"), index=cols)
                        imp.append({"side": side, "model": f"qgbm_{org}", "tau": tau, "fold": str(f0.date()),
                                    "imp": g / max(g.sum(), 1e-12)})
                QP[(side, f"qgbm_{org}")][te_m] = np.sort(Q, axis=1)
            log(f"  {side} fold {f0.date()} -> {f1.date()}: train {tr_m.sum():,} h, test {te_m.sum():,} h  "
                f"[{time.time() - t0:.0f}s]")

    # ---------- quantile evaluation
    test = (hk >= TEST_START).to_numpy()
    rng = np.random.default_rng(20261016)
    qrows, qb, qsave, prob = [], [], [], {}
    for side, (_, _, _, flag) in SIDES.items():
        z = h[f"z_{side}"].to_numpy(float)
        for mk in ("qgbm_d1", "qgbm_h1", "clim365"):
            Q = QP[(side, mk)]
            ok = test & np.isfinite(z) & np.isfinite(Q).all(axis=1)
            d = pd.DataFrame({"hk": hk[ok].to_numpy(), "side": side, "model": mk, "z": z[ok]})
            Qe = itf(Q[ok], sc)
            for k, tau in enumerate(taus):
                d[f"q{tau:g}"] = Qe[:, k]
            qsave.append(d)
            okp = test & np.isfinite(Q).all(axis=1)
            prob[(side, mk)] = pd.Series(prob_above_zero(Q[okp], taus), index=hk[okp].to_numpy())
        D = pd.concat([x for x in qsave if x["side"].iat[0] == side])
        D["date"] = pd.to_datetime(D["hk"]).dt.tz_localize("UTC").dt.tz_convert(V.TZ).dt.tz_localize(None).dt.normalize()
        yr = D["date"].dt.year
        for pname, msk in (("all", yr > 0), ("2021_22", yr <= 2022), ("from_2023", yr >= 2023), ("post_cap", D["date"] >= CAP)):
            DD = D[msk]
            tails = {}
            for mk, g in DD.groupby("model"):
                row = {"side": side, "period": pname, "model": mk, "n": len(g)}
                tl = np.zeros(len(g))
                nt = 0
                for tau in taus:
                    pl = pinball(g["z"].to_numpy(), g[f"q{tau:g}"].to_numpy(), tau)
                    row[f"pinball_{tau:g}"] = pl.mean()
                    row[f"cov_{tau:g}"] = (g["z"] <= g[f"q{tau:g}"]).mean()
                    if tau >= 0.95:
                        tl += pl
                        nt += 1
                row["tail_score"] = (tl / max(nt, 1)).mean()
                tails[mk] = pd.Series(tl / max(nt, 1), index=g["hk"].to_numpy())
                qrows.append(row)
            if "clim365" in tails:
                for mk in ("qgbm_d1", "qgbm_h1"):
                    j = tails["clim365"].index.intersection(tails[mk].index)
                    days = pd.to_datetime(pd.Series(j)).dt.normalize().to_numpy()
                    e, lo, hi = boot((tails["clim365"][j] - tails[mk][j]).to_numpy(), days, rng)
                    qb.append({"side": side, "period": pname, "model": mk, "dTail_vs_clim365": e, "ci_lo": lo, "ci_hi": hi})
    QS, QB = pd.DataFrame(qrows), pd.DataFrame(qb)
    QS.to_csv(OUT / "quantile_scores.csv", index=False)
    QB.to_csv(OUT / "quantile_bootstrap.csv", index=False)
    pd.concat(qsave).to_parquet(OUT / "quantile_predictions.parquet", index=False)
    log("\n== quantile forecasts (EUR/MWh): pinball per tau, coverage, tail score (mean pinball tau >= 0.95)")
    show = ["side", "period", "model", "n", "tail_score"] + [f"pinball_{t:g}" for t in taus if t >= 0.9] + \
           [f"cov_{t:g}" for t in taus]
    log(QS[show].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    log("\ntail score gain vs clim365 (positive = better), day-block bootstrap 95 % CI:")
    log(QB.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    # ---------- spike probability vs rq3b_1 / rq3b_4
    base = pd.read_csv(HERE / "Output" / "rq3b_1" / "predictions.csv", parse_dates=["hk", "date"])
    raw = pd.read_csv(HERE / "Output" / "rq3b_1_raw" / "predictions.csv", parse_dates=["hk"])
    gb = [c for c in raw.columns if c.startswith("gbm")]
    P = base[["hk", "side", "y", "date", *BASELINES, "lr_d1", "lr_h1"]].merge(raw[["hk", "side", *gb]], on=["hk", "side"])
    extra = []
    hw = HERE / "Output" / "rq3b_4" / "predictions.csv"
    if hw.exists():
        Hh = pd.read_csv(hw, parse_dates=["hk"])[["hk", "side", "hawkes_d1", "hawkes_h1"]]
        P = P.merge(Hh, on=["hk", "side"], how="left")
        extra = ["hawkes_d1", "hawkes_h1"]
    for mk, name in (("qgbm_d1", "qgbm_d1"), ("qgbm_h1", "qgbm_h1"), ("clim365", "qclim")):
        P[name] = np.nan
        for side in SIDES:
            m = P["side"] == side
            P.loc[m, name] = prob[(side, mk)].reindex(P.loc[m, "hk"].to_numpy()).to_numpy()
    models = [*BASELINES, "lr_d1", "lr_h1", *gb, *extra, "qclim", "qgbm_d1", "qgbm_h1"]
    P = P.dropna(subset=["qgbm_d1", "qgbm_h1", "qclim"])
    P.to_csv(OUT / "predictions.csv", index=False)
    log(f"\n== implied spike probability: {len(P):,} test hours merged with rq3b_1 ({len(base):,})")
    out, bt, best = [], [], {}
    rngb = np.random.default_rng(20261010)
    for side in SIDES:
        d = P[P.side == side]
        bb = {m: ((d[m] - d["y"]) ** 2).mean() for m in BASELINES}
        best[side] = min(bb, key=bb.get)
        yr = d["date"].dt.year
        for pname, dd in (("all", d), ("2021_22", d[yr <= 2022]), ("from_2023", d[yr >= 2023]), ("post_cap", d[d["date"] >= CAP])):
            ref = ((dd[best[side]] - dd["y"]) ** 2).mean()
            ms = [m for m in models if dd[m].notna().all()]
            for m in ms:
                s_ = scores(dd["y"].to_numpy(float), dd[m].to_numpy(float))
                out.append({"side": side, "period": pname, "model": m, **s_, "best_baseline": best[side],
                            "Brier_skill_vs_best_baseline": 1 - s_["Brier"] / ref})
            for m, r_ in (("qgbm_d1", best[side]), ("qgbm_h1", best[side]), ("qgbm_d1", "gbm_d1"), ("qgbm_h1", "gbm_h1"),
                          ("qgbm_h1", "lr_h1")):
                diff = ((dd[r_] - dd["y"]) ** 2 - (dd[m] - dd["y"]) ** 2).to_numpy()
                e, lo, hi = boot(diff, dd["date"].to_numpy(), rngb)
                bt.append({"side": side, "period": pname, "model": m, "vs": r_, "dBrier_x1e4": e * 1e4,
                           "ci_lo": lo * 1e4, "ci_hi": hi * 1e4})
    R, BT = pd.DataFrame(out), pd.DataFrame(bt)
    R.to_csv(OUT / "metrics.csv", index=False)
    BT.to_csv(OUT / "brier_gain_bootstrap.csv", index=False)
    for side in SIDES:
        for pname in ("all", "2021_22", "from_2023", "post_cap"):
            t = R[(R.side == side) & (R.period == pname)].set_index("model")[
                ["n", "pos", "Brier", "Brier_skill_vs_best_baseline", "logloss", "PR_AUC", "ROC_AUC", "mean_pred",
                 "ECE", "capture_top5"]]
            log(f"\n-- {side}, {pname}; best baseline {best[side]}")
            log(t.to_string(float_format=lambda v: f"{v:,.4f}"))
    log("\n-- Brier gain (positive = first model better), day-block bootstrap 95 % CI, x1e-4")
    log(BT.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    if imp:
        rows = []
        for (sd_, mk, tau), grp in pd.DataFrame(imp).groupby(["side", "model", "tau"]):
            m = pd.concat(grp["imp"].tolist(), axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
            rows += [{"side": sd_, "model": mk, "tau": tau, "feature": f, "gain_share": v} for f, v in m.head(25).items()]
        IM = pd.DataFrame(rows)
        IM.to_csv(OUT / "importance_q.csv", index=False)
        log("\n-- gain importance, top 8 (tau 0.99)")
        for (sd_, mk), g in IM[IM.tau == 0.99].groupby(["side", "model"]):
            log(f"   {sd_} {mk}: " + "; ".join(f"{r.feature} {r.gain_share:.1%}" for r in g.head(8).itertuples()))
    try:
        import ml_diagnostics as D
        D.classifier_diagnostics(P, models, OUT, y_col="y", time_col="hk", group_col="side", reference=best,
                                 title=f"RQ3 quantile LightGBM{a.tag}")
    except Exception as e:  # noqa: BLE001
        log(f"  (diagnostics skipped: {e})")
    for side in SIDES:
        t = R[(R.side == side) & (R.period == "from_2023")].set_index("model")["Brier_skill_vs_best_baseline"]
        RU.summary(f"{side} from 2023: qgbm_h1 {t.get('qgbm_h1', np.nan):+.3f}, gbm_h1 {t.get('gbm_h1', np.nan):+.3f}")
    (OUT / "report_rq3b_5.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    RU.run(main, "RQ3 quantile LightGBM")
