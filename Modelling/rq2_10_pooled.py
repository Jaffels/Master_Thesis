"""RQ2 step 10 (done_todo 2.2 item 8, Tier 2 of ML_Options.md "multi-output / multitask"): ONE pooled (global)
tree model for all capacity-price series instead of one model per series.

Why: the series are small (weekly 300-600 blocks, aFRR 4h ~2,000) and products are procured in the same market,
so a model fitted on all products together can borrow strength across them ("global" forecasting model,
Montero-Manso & Hyndman 2021). The feature sets of the three views are (almost) identical (776 of 779 columns
shared; the lagged-target features tgt_prev_* are each series' OWN lags under the same names), so the rows of all
series can be stacked into one training table.

Design (everything else identical to rq2_5_tier1.py round 2, so the comparison is clean):
  * Rows: all series of the FCR, aFRR and mFRR ex-ante views (--pool all; FCR day / week and aFRR sym week only
    contribute pre-2021 training rows) or only the evaluated series (--pool evaluated).
  * Features: union of the per-view ex-ante features + series identity: pool_product, pool_direction,
    pool_procurement, pool_series (categorical), pool_slot, pool_level = asinh(same-slot-yesterday price).
  * Target: delta asinh(y) - asinh(same slot yesterday) as in Tier 1, divided by the series' standard deviation of
    that delta in the current training window (--scale sd; --scale none keeps it raw) so that volatile series do
    not dominate the pinball loss; predictions are rescaled per series (scaling is monotone -> quantiles map back).
  * Folds: exactly the folds of rq2_5 for every series (C.fold_plan, refit every 3 months / weekly 12, skip folds
    with < MIN_TRAIN own blocks). One pooled model is fitted per distinct fold start, on ALL rows of all series
    delivered before that start, and predicts the test blocks of every series whose fold starts there -> every
    series sees the same information cut-off as in its single-series run (weekly series are still refit yearly).
  * Models: lgbm_q (LightGBM quantile, one model per level) and cat_q (CatBoost MultiQuantile), round-2 default
    hyperparameters (no tuning, 1 seed by default), early stopping on the last 15 % IN TIME of the pooled window.
  * Calibration: sequential CQR per series on earlier test folds (as rq2_5).
Evaluation per series on common rows with the single-series Tier 1 run (Output/rq2_5_<product>, --single):
  MAE skill vs same-slot-yesterday; DM |error| pooled vs single (same model), pooled vs LEAR, pooled vs naive;
  pinball DM pooled vs single; 80 / 90 % coverage raw and CQR; MAE by year; gain share of the series-identity
  features (how much the pooled model separates the series).

    python Modelling/rq2_10_pooled.py --quick                      # smoke test: 2 fold starts, few rounds (~5-10 min)
    python Modelling/rq2_10_pooled.py                              # full run, lgbm_q + cat_q
    python Modelling/rq2_10_pooled.py --models lgbm_q --out Modelling/Output/rq2_10_lgbm
Output: Modelling/Output/rq2_10/{preds.parquet, metrics_point.csv, metrics_prob.csv, metrics_by_year.csv, dm.csv,
        fold_log.csv, id_importance.csv, importance_pool_<model>.csv, report_rq2_10.txt, run.log, run_meta.json, diagnostics/}
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_utils as RU  # run.log, run_meta.json, run history, macOS notification

import rq2_common as C
import rq2_5_tier1 as T
from rq2_2_lightgbm import feature_list, make_X

MODELS = ["lgbm_q", "cat_q"]
ID_FEATS = ["pool_product", "pool_direction", "pool_procurement", "pool_series", "pool_slot", "pool_level"]
SINGLE = ["lear", "lgbm_q", "cat_q", "qrf", "svr", "lgbm_l1"]


def build_pool(a, log):
    """Stack the three ex-ante views. Returns df (sorted by series, time) and the pooled feature matrix X."""
    dfs, feats = [], []
    for v in C.VIEWS:
        d = C.load_view(v)
        f, _ = feature_list(d, v, "exante")
        feats += [c for c in f if c not in feats]
        dfs.append(d)
        log(f"{v}: {len(d):,} blocks, {len(f)} features, series {', '.join(d['series'].unique())}")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df = df.sort_values(["series", "block_start_utc"]).reset_index(drop=True)
    df["pool_product"] = df["product"].astype(str)
    df["pool_direction"] = df["direction"].astype(str)
    df["pool_procurement"] = df["procurement"].astype(str)
    df["pool_series"] = df["series"].astype(str)
    df["pool_slot"] = df["slot"].astype(float)
    df["pool_level"] = np.arcsinh(df["b_prev_slot"].astype(float))
    for c in feats:                                   # bool columns missing in one view become object -> numeric
        if df[c].dtype == object and df[c].dropna().map(type).isin([bool, np.bool_]).all():
            df[c] = df[c].astype(float)
    X = make_X(df, feats + ID_FEATS)
    log(f"pooled: {len(df):,} blocks, {X.shape[1]} features ({len(ID_FEATS)} series-identity), "
        f"{df['series'].nunique()} series")
    return df, X


def plan(df: pd.DataFrame, ok: np.ndarray, a):
    """Fold plan of every series as in rq2_5: {fold start: [(series, fold, global test positions, n own train)]}."""
    by_start: dict[pd.Timestamp, list] = {}
    for s in df["series"].unique():
        if a.only and a.only != s:
            continue
        gi = np.flatnonzero((df["series"] == s).to_numpy())
        g = df.iloc[gi].reset_index(drop=True)
        rm = a.refit_months_week if s.endswith("|week") else a.refit_months
        mt = C.MIN_TRAIN[s.split("|")[2]]
        for f, start, tr, te in C.fold_plan(g, rm, mt):
            n_own = int(ok[gi[tr]].sum())
            if n_own < mt:                             # same skip rule as rq2_5 run_series
                continue
            by_start.setdefault(start, []).append((s, f, gi[te], n_own))
    return dict(sorted(by_start.items()))


def series_scale(df: pd.DataFrame, yd: np.ndarray, tr: np.ndarray, a) -> pd.Series:
    """Per-series scale of the delta target in the training window (1.0 with --scale none or too few rows)."""
    ser = df["series"].to_numpy()
    if a.scale == "none":
        return pd.Series(1.0, index=pd.unique(ser))
    sd = pd.Series(yd[tr]).groupby(ser[tr]).std(ddof=1)
    sd = sd.where(sd > 1e-6)
    return sd.reindex(pd.unique(ser)).fillna(sd.median() if sd.notna().any() else 1.0)


def run_pool(df, X, a, log):
    ya = np.arcsinh(df["y"].to_numpy(float))
    off = np.arcsinh(df["b_prev_slot"].to_numpy(float))
    yd = ya - off
    ok = ~np.isnan(yd)
    end = df["block_end_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    tstart = df["block_start_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    in_pool = np.ones(len(df), bool)
    by_start = plan(df, ok, a)
    if a.pool == "evaluated":
        ev = {s for jobs in by_start.values() for s, *_ in jobs}
        in_pool = df["series"].isin(ev).to_numpy()
    if a.max_folds:
        by_start = dict(list(by_start.items())[:a.max_folds])
    log(f"{len(by_start)} pooled fits (distinct fold starts), "
        f"{sum(len(j) for j in by_start.values())} series-folds to predict")
    keys = ["series", "product", "direction", "procurement", "auction_id", "block_start_utc", "slot", "y", *C.BASELINES,
            *[c for c in df.columns if c.startswith("regime_")]]
    rows, flog, imps = [], [], {m: [] for m in a.models}
    ser = df["series"].to_numpy()
    for k, (start, jobs) in enumerate(by_start.items()):
        t_fold = time.time()
        tr = np.flatnonzero((end <= np.datetime64(start.tz_convert("UTC").tz_localize(None))) & ok & in_pool)
        tr = tr[np.argsort(tstart[tr], kind="stable")]          # time order across series (early-stopping split)
        te = np.concatenate([j[2] for j in jobs])
        sc = series_scale(df, yd, tr, a)
        s_tr, s_te = sc.reindex(ser[tr]).to_numpy(), sc.reindex(ser[te]).to_numpy()
        z = yd[tr] / s_tr
        part = df.iloc[te][keys].copy()
        part["fold"] = np.concatenate([[f] * len(idx) for _, f, idx, _ in jobs])
        part["n_train_own"] = np.concatenate([[n] * len(idx) for _, _, idx, n in jobs])
        part["n_train_pool"] = len(tr)
        fl = {"fit": k, "fold_start": start.tz_convert(C.TZ).date(), "n_train_pool": len(tr),
              "n_series_train": int(pd.Series(ser[tr]).nunique()), "n_test": len(te),
              "series_predicted": " ".join(sorted({s for s, *_ in jobs}))}
        fl.update({f"scale__{s}": float(v) for s, v in sc.items() if s in set(ser[te])})
        for m in a.models:
            t0 = time.time()
            Q = np.zeros((len(te), len(T.QS)))
            for sd in range(a.seeds):
                if m == "lgbm_q":
                    q_, inf = T.fit_lgbm_q(X.iloc[tr], z, X.iloc[te], a, seed=1 + sd)
                else:
                    q_, inf = T.fit_cat_q(X.iloc[tr], z, X.iloc[te], a, seed=1 + sd)
                Q += q_
                if a.seeds > 1:
                    part[f"p_pool_{m}_seed{sd + 1}"] = np.sinh(np.sort(q_, axis=1)[:, T.MED] * s_te + off[te])
                if sd == 0:
                    if inf.get("imp") is not None:
                        imps[m].append(inf["imp"])
                    fl.update({f"{m}_{kk}": v for kk, v in inf.items() if kk != "imp"})
            raw = Q / a.seeds
            fl[f"{m}_crossing_share"] = float((np.diff(raw, axis=1) < 0).any(axis=1).mean())
            Q = np.sort(raw, axis=1) * s_te[:, None] + off[te][:, None]
            for kk, t in enumerate(T.QTAG):
                part[f"q_pool_{m}_{t}"] = np.sinh(Q[:, kk])
            part[f"p_pool_{m}"] = part[f"q_pool_{m}_50"]
            fl[f"{m}_seconds"] = round(time.time() - t0, 1)
        rows.append(part); flog.append(fl)
        log(f"   fit {k + 1:2d}/{len(by_start)} start {start.tz_convert(C.TZ).date()}: train {len(tr):,} rows "
            f"({fl['n_series_train']} series), test {len(te):,} ({len(jobs)} series)  "
            + "  ".join(f"{m} {fl[f'{m}_seconds']:.0f}s" for m in a.models) + f"  [{time.time() - t_fold:.0f}s]")
        if a.notify_every and (k + 1) % a.notify_every == 0:
            RU.notify_mac("RQ2 pooled model", f"fit {k + 1} / {len(by_start)} done", sound="Glass")
    return pd.concat(rows, ignore_index=True), flog, imps


def load_single(spec: str) -> pd.DataFrame | None:
    base = C.ROOT / "Modelling" / "Output"
    files = [base / f"{spec}_{v}" / "preds.parquet" for v in C.VIEWS]
    files = [f for f in files if f.exists()]
    if not files:
        return None
    S = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cols = ["series", "block_start_utc"] + [c for c in S.columns if (c.startswith("p_") and c[2:] in SINGLE)
                                            or (c.startswith("q_") and c[2:].rsplit("_", 1)[0] in T.PROB_MODELS)]
    return S[cols].drop_duplicates(["series", "block_start_utc"])


def evaluate(P: pd.DataFrame, s: str, models: list[str]):
    L = 4 if s.endswith("|week") else 12
    pool = [f"pool_{m}" for m in models]
    single = [m for m in SINGLE if f"p_{m}" in P and P[f"p_{m}"].notna().any()]
    E = P.dropna(subset=["y", *C.BASELINES, *[f"p_{m}" for m in pool + single]]).sort_values("block_start_utc")
    if len(E) < 30:
        return [], [], [], []
    y = E["y"].to_numpy(float)
    E = E.assign(year=E["block_start_utc"].dt.tz_convert(C.TZ).dt.year)
    mp, my, dm, mq = [], [], [], []
    names = {**{m: f"p_{m}" for m in pool + single}, **{b: b for b in C.BASELINES}}
    for nm, col in names.items():
        mp.append({"series": s, "model": nm, **C.metrics(y, E[col])})
        for yr, gy in E.groupby("year"):
            my.append({"series": s, "model": nm, "year": yr, **C.metrics(gy["y"], gy[col])})
    err = {nm: np.abs(E[col].to_numpy(float) - y) for nm, col in names.items()}
    for m in models:
        pm = f"pool_{m}"
        for ref in [m, "lear", "b_prev_slot"]:
            if ref in err and ref != pm:
                st, p = T.dm_loss(err[pm] - err[ref], L)
                dm.append({"series": s, "loss": "abs", "model": pm, "vs": ref if ref != m else f"single_{m}",
                           "stat": st, "p": p})
    prob = [m for m in pool + [m for m in T.PROB_MODELS if m in single]
            if all(f"q_{m}_{t}" in E for t in T.QTAG) and E[[f"q_{m}_{t}" for t in T.QTAG]].notna().all().all()]
    pl = {}
    for m in prob:
        pl[m] = np.mean([T.pinball(y, E[f"q_{m}_{t}"].to_numpy(float), q) for q, t in zip(T.QS, T.QTAG)], axis=0)
        row = {"series": s, "model": m, "n": len(E), "pinball": pl[m].mean(), "CRPS_approx": 2 * pl[m].mean()}
        for lev, (lo_t, hi_t) in T.INTERVALS.items():
            lo, hi = E[f"q_{m}_{lo_t}"].to_numpy(float), E[f"q_{m}_{hi_t}"].to_numpy(float)
            if f"cqr{lev}_lo_{m}" in E:
                clo, chi = E[f"cqr{lev}_lo_{m}"].to_numpy(float), E[f"cqr{lev}_hi_{m}"].to_numpy(float)
                h = ~np.isnan(clo)
                row[f"cov{lev}_raw"] = np.mean((y[h] >= lo[h]) & (y[h] <= hi[h])) * 100 if h.any() else np.nan
                row[f"cov{lev}_cqr"] = np.mean((y[h] >= clo[h]) & (y[h] <= chi[h])) * 100 if h.any() else np.nan
                row[f"width{lev}_cqr"] = np.mean(chi[h] - clo[h]) if h.any() else np.nan
            else:
                row[f"cov{lev}_raw"] = np.mean((y >= lo) & (y <= hi)) * 100
            row[f"width{lev}_raw"] = np.mean(hi - lo)
        mq.append(row)
    for m in models:
        if f"pool_{m}" in pl and m in pl:
            st, p = T.dm_loss(pl[f"pool_{m}"] - pl[m], L)
            dm.append({"series": s, "loss": "pinball", "model": f"pool_{m}", "vs": f"single_{m}", "stat": st, "p": p})
    return mp, my, dm, mq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--only", default=None, help='predict only this series (the pool still uses all), e.g. "aFRR|up|week"')
    ap.add_argument("--pool", default="all", choices=["all", "evaluated"],
                    help="training rows: all series of the three views, or only the series that are evaluated")
    ap.add_argument("--scale", default="sd", choices=["sd", "none"], help="per-series scaling of the delta target")
    ap.add_argument("--single", default="rq2_5", help="folder prefix of the single-series Tier 1 run (<prefix>_<product>)")
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--refit-months-week", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max-folds", type=int, default=0, help="only the first n pooled fits (0 = all)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 fits, 100 rounds / 200 CatBoost iterations")
    ap.add_argument("--notify-every", type=int, default=5, help="macOS notification every n pooled fits (0 = off)")
    # round-2 defaults of rq2_5_tier1.py (names must match T.fit_lgbm_q / T.fit_cat_q)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--min-leaf", type=int, default=0)
    ap.add_argument("--colsample", type=float, default=0.3)
    ap.add_argument("--bagging", type=float, default=0.8)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--cat-iters", type=int, default=1500)
    ap.add_argument("--cat-lr", type=float, default=0.05)
    ap.add_argument("--cat-depth", type=int, default=6)
    ap.add_argument("--cat-l2", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.quick:
        a.max_folds, a.rounds, a.cat_iters = a.max_folds or 2, 100, 200
    out = Path(a.out) if a.out else C.ROOT / "Modelling" / "Output" / ("rq2_10_quick" if a.quick else "rq2_10")
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    log = lambda x: print(x, flush=True)
    t_all = time.time()

    df, X = build_pool(a, log)
    P, flog, imps = run_pool(df, X, a, log)
    S = load_single(a.single)
    if S is not None:
        P = P.merge(S, on=["series", "block_start_utc"], how="left")
        log(f"merged single-series predictions from {a.single}_<product>")
    else:
        log(f"no single-series run {a.single}_<product> found - pooled results only")
    parts = []
    for s, g in P.groupby("series", sort=False):
        g = g.sort_values("block_start_utc").reset_index(drop=True)
        for m in [f"pool_{m}" for m in a.models] + [m for m in T.PROB_MODELS if f"q_{m}_50" in g and g[f"q_{m}_50"].notna().any()]:
            for lev, (lo_t, hi_t) in T.INTERVALS.items():
                g[f"cqr{lev}_lo_{m}"], g[f"cqr{lev}_hi_{m}"] = T.cqr(g, m, lo_t, hi_t, lev)
        parts.append(g)
    P = pd.concat(parts, ignore_index=True)
    P.to_parquet(out / "preds.parquet")
    pd.DataFrame(flog).to_csv(out / "fold_log.csv", index=False)

    MP, MY, DM, MQ = [], [], [], []
    for s, g in P.groupby("series", sort=False):
        mp, my, dm, mq = evaluate(g, s, a.models)
        MP += mp; MY += my; DM += dm; MQ += mq
    MP, MY, DM, MQ = map(pd.DataFrame, (MP, MY, DM, MQ))
    for nm, d_ in [("metrics_point", MP), ("metrics_by_year", MY), ("dm", DM), ("metrics_prob", MQ)]:
        d_.to_csv(out / f"{nm}.csv", index=False)

    import ml_diagnostics as D
    imp_tab = {m: D.importance_table({"ALL (pooled)": lst}) for m, lst in imps.items() if lst}
    IDI = []
    for m, lst in imps.items():
        for k, sh in enumerate(lst):
            IDI.append({"model": m, "fit": k, **{f: float(sh.get(f, 0.0)) for f in ID_FEATS},
                        "id_total": float(sh.reindex(ID_FEATS).fillna(0).sum())})
    IDI = pd.DataFrame(IDI)
    IDI.to_csv(out / "id_importance.csv", index=False)
    for m, t in imp_tab.items():
        t.to_csv(out / f"importance_pool_{m}.csv", index=False)
    try:
        models = {f"pool_{m}": f"p_pool_{m}" for m in a.models}
        models |= {m: f"p_{m}" for m in SINGLE if f"p_{m}" in P and P[f"p_{m}"].notna().any()}
        models |= {b: b for b in C.BASELINES}
        quant = {f"pool_{m}": {q: f"q_pool_{m}_{t}" for q, t in zip(T.QS, T.QTAG)} for m in a.models}
        D.forecast_diagnostics(P, models, out, baseline="b_prev_slot", quantiles=quant,   # importance: one pooled
                               group_cols=[c for c in P.columns if c.startswith("regime_")], title="RQ2 pooled model")
    except Exception as e:                              # diagnostics must never lose the run
        log(f"diagnostics failed: {e}")

    # ---------------------------------------------------------------- report
    rep = ["RQ2 step 10 - pooled (global) tree model across all capacity-price series (done_todo 2.2 item 8).",
           f"Pool: {a.pool} series; target delta asinh, scale {a.scale}; folds as rq2_5 (refit {a.refit_months} months, "
           f"weekly {a.refit_months_week}); models {', '.join(a.models)}; seeds {a.seeds}; round-2 default hyperparameters.",
           f"Single-series comparison: Output/{a.single}_<product>. Errors in EUR/MWh on the common rows of all models.",
           f"{len(flog)} pooled fits{' (QUICK - not for the thesis)' if a.quick else ''}.", ""]
    if len(MP):
        S_ = MP.pivot(index="series", columns="model", values="MAE")
        cols = [c for c in [f"pool_{m}" for m in a.models] + SINGLE if c in S_.columns]
        sk = pd.DataFrame({c: (1 - S_[c] / S_["b_prev_slot"]) * 100 for c in cols}, index=S_.index)
        rep += ["== MAE skill vs same-slot-yesterday (%)", C.fmt(sk), ""]
        A = DM[DM["loss"] == "abs"]
        if len(A):
            rep.append("== Diebold-Mariano |error|, pooled vs reference (stat < 0 = pooled better)")
            rep.append(C.fmt(A.pivot_table(index="series", columns=["model", "vs"], values="p")))
            for _, r in A[A["vs"].str.startswith("single_")].iterrows():
                tag = ("better" if r["stat"] < 0 else "worse") if r["p"] < 0.05 else "n.s."
                RU.summary(f"{r['series'].replace('|', ' ')} {r['model']} vs {r['vs']}: {tag} (p={r['p']:.3f})")
            rep.append("")
        if len(MQ):
            rep += ["== Mean pinball loss (EUR/MWh)", C.fmt(MQ.pivot(index="series", columns="model", values="pinball"))]
            Pq = DM[DM["loss"] == "pinball"]
            if len(Pq):
                rep.append("   DM pinball pooled vs single: " + "; ".join(
                    f"{r.series} {r.model} stat {r.stat:.2f} p {r.p:.4f}" for r in Pq.itertuples()))
            if "cov90_cqr" in MQ:
                rep += ["== 90 % coverage raw -> CQR (%)", MQ.assign(
                    c=lambda d: d["cov90_raw"].round(1).astype(str) + " -> " + d["cov90_cqr"].round(1).astype(str)
                ).pivot(index="series", columns="model", values="c").to_string(), ""]
        Y = MY[MY["model"].isin(cols + ["b_prev_slot"])]
        rep.append("== MAE by year")
        for s, g in Y.groupby("series", sort=False):
            rep.append(f"   {s}:")
            rep.append("   " + C.fmt(g.pivot(index="year", columns="model", values="MAE")).replace("\n", "\n   "))
    if len(IDI):
        rep += ["", "== Gain share of the series-identity features (mean over fits; id_importance.csv)",
                C.fmt(IDI.groupby("model")[ID_FEATS + ["id_total"]].mean())]
    for m, t in imp_tab.items():
        rep.append(f"== Top features pooled {m}: " + "; ".join(
            f"{r.feature} {r.mean * 100:.1f}%" for r in t[t["mean"] > 0].head(12).itertuples()))
    rep.append(f"\ntotal runtime {time.time() - t_all:.0f}s")
    (out / "report_rq2_10.txt").write_text("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    RU.run(main, "RQ2 pooled global model")
