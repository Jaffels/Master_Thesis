"""RQ2 step 2 (todo 9.2b/c): LightGBM on the strict ex-ante features, rolling-origin, vs baselines.

Per series (product x direction x procurement) and feature set:
  exante             strict ex-ante features (views.feature_columns, perfect_forecast=False)
  perfect            + __pf upper bound (D-1 forecasts / observed weather that were not yet public
                     at gate closure; replaces the matching __dw columns)
  exante_noweather   exante without the MeteoSwiss block   (weather ablation)
  perfect_noweather  perfect without the MeteoSwiss block
Model: LightGBM, target asinh(price), L1 objective (= conditional median; sinh() is monotone so the
median maps back exactly, which is the MAE-optimal point forecast in EUR/MWh; --objective l2 for
the mean). Early stopping on the last 15 % (in time) of the training window, then refit on the
full window with the best iteration count. Expanding window, refit every --refit-months (4h / day
blocks) or --refit-months-week (weekly blocks); the first test fold per series is the first month
>= 1 Jan 2021 with enough training blocks (rq2_common.MIN_TRAIN).
Dropped features: cal_year (a trend counter, no meaning out of sample).

    python Modelling/rq2_2_lightgbm.py [--products mfrr] [--sets exante perfect] [--only "mFRR|up|4h"]
Output: Modelling/Output/rq2_2/{preds_<set>.parquet, metrics.csv, metrics_by_year.csv,
        importance_<set>.csv, report_rq2_2.txt}
Check first: python Modelling/rq2_2_lightgbm.py --only "mFRR|up|week" --sets exante   (seconds)
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
import rq2_common as C

DROP = {"cal_year"}
SETS = ["exante", "perfect", "exante_noweather", "perfect_noweather"]


def family(c: str, weather: set[str]) -> str:
    if c in weather: return "weather (MeteoSwiss)"
    if c.startswith("tgt_prev"): return "lagged target"
    if c.startswith("xm_"): return "cross-market (DE prices)"
    if c.startswith("cal_"): return "calendar"
    if c.startswith(("regime_", "src_")): return "regime / source"
    if c.endswith("__exante"): return "outages (ex ante)"
    if c.endswith("__dw"): return "published forecasts"
    if c.endswith("__pf"): return "perfect-forecast block"
    if "__lb" in c: return "lookback (observed)"
    return "other"


def feature_list(df: pd.DataFrame, view: str, fset: str):
    d = C.V.dictionary(view, "exante")
    perfect = fset.startswith("perfect")
    feats = [c for c in C.V.feature_columns(df, view, "exante", perfect_forecast=perfect) if c not in DROP]
    weather = {c for c in feats if "MeteoSwiss" in str(d["source"].get(c, ""))}
    if fset.endswith("noweather"):
        feats = [c for c in feats if c not in weather]
    return feats, weather


def make_X(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = df[feats].copy()
    for c in X.columns:
        dt = X[c].dtype
        if pd.api.types.is_bool_dtype(dt):
            X[c] = X[c].astype("int8")
        elif not pd.api.types.is_numeric_dtype(dt):          # object / str / category -> category codes
            X[c] = X[c].astype("category")
    return X


def fit_predict(Xtr, ytr, Xte, a, seed=1):
    n = len(Xtr)
    params = dict(objective="regression_l1" if a.objective == "l1" else "regression",
                  learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=max(5, min(20, n // 30)),
                  feature_fraction=a.colsample, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, seed=seed, num_threads=a.threads)
    nv = max(10, int(0.15 * n))
    dtr, dva = lgb.Dataset(Xtr.iloc[:-nv], ytr[:-nv]), lgb.Dataset(Xtr.iloc[-nv:], ytr[-nv:])
    m0 = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    best = max(20, int((m0.best_iteration or 20) * 1.15))
    m = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=best)
    return m.predict(Xte), m, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=C.VIEWS)
    ap.add_argument("--sets", nargs="+", default=["exante"], choices=SETS)
    ap.add_argument("--only", default=None, help='series filter, e.g. "mFRR|up|4h"')
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--refit-months-week", type=int, default=12)
    ap.add_argument("--mode", default="delta", choices=["delta", "level"],
                    help="delta: learn asinh(y) - asinh(same-slot-yesterday) and add it back (default); level: learn asinh(y)")
    ap.add_argument("--objective", default="l1", choices=["l1", "l2"])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--colsample", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else C.ROOT / "Modelling" / "Output" / f"rq2_2_{a.mode}"; out.mkdir(parents=True, exist_ok=True)
    lines = ["RQ2 9.2b/c - LightGBM, rolling origin, expanding window; errors in EUR/MWh on the common rows "
             "of the model and all three baselines.",
             f"mode={a.mode}, objective={a.objective} on asinh(price), lr={a.lr}, leaves={a.leaves}, colsample={a.colsample}, "
             f"refit every {a.refit_months} months (weekly: {a.refit_months_week}), test from blocks >= 2021-01-01", ""]
    allm, ally, pred_store = [], [], {s: [] for s in a.sets}
    imp_store = {s: {} for s in a.sets}
    fam_store = {s: {} for s in a.sets}
    t_all = time.time()
    for view in a.products:
        df = C.load_view(view)
        base_keys = ["series", "product", "direction", "procurement", "auction_id", "block_start_utc", "y", *C.BASELINES]
        for fset in a.sets:
            feats, weather = feature_list(df, view, fset)
            X = make_X(df, feats)
            for s in df["series"].unique():
                if a.only and a.only != s:
                    continue
                gi = np.flatnonzero((df["series"] == s).to_numpy())
                g = df.iloc[gi]
                Xg = X.iloc[gi]
                ya = np.arcsinh(g["y"].to_numpy(dtype=float))
                ba = np.arcsinh(g["b_prev_slot"].to_numpy(dtype=float))
                off = ba if a.mode == "delta" else np.zeros(len(ya))     # offset added back to the prediction
                yt = ya - off
                ok = ~np.isnan(yt)
                rm = a.refit_months_week if s.endswith("|week") else a.refit_months
                mt = C.MIN_TRAIN[s.split("|")[2]]
                t0, preds, nf, skipped = time.time(), [], 0, 0
                for f, start, tr, te in C.fold_plan(g.reset_index(drop=True), rm, mt):
                    tr = tr[ok[tr]]
                    if len(tr) < mt:
                        skipped += 1; continue
                    p, model, best = fit_predict(Xg.iloc[tr], yt[tr], Xg.iloc[te], a)
                    gain = pd.Series(model.feature_importance("gain"), index=feats)
                    share = gain / max(gain.sum(), 1e-12)
                    imp_store[fset].setdefault(s, []).append(share)
                    fam_store[fset].setdefault(s, []).append(share.groupby([family(c, weather) for c in feats]).sum())
                    r = g.iloc[te][base_keys].copy()
                    r["pred"] = np.sinh(p + off[te]); r["fold"] = f; r["n_train"] = len(tr); r["best_iter"] = best
                    r["set"] = fset
                    preds.append(r); nf += 1
                print(f"{view} {fset:18s} {s:16s} folds={nf:3d} skipped={skipped} {time.time()-t0:6.1f}s", flush=True)
                if not preds:
                    lines.append(f"== {s} [{fset}]: no fold with >= {mt} training blocks"); continue
                P = pd.concat(preds)
                pred_store[fset].append(P)
                P = P.dropna(subset=["y", "pred", *C.BASELINES]).copy()
                P["year"] = P["block_start_utc"].dt.tz_convert(C.TZ).dt.year
                row = {"series": s, "set": fset, "n_folds": nf, "first_test": P["block_start_utc"].min().date()}
                for name in ["pred", *C.BASELINES]:
                    mm = C.metrics(P["y"], P[name])
                    allm.append({**row, "model": "lgbm" if name == "pred" else name, **mm})
                    for y, gy in P.groupby("year"):
                        ally.append({"series": s, "set": fset, "model": "lgbm" if name == "pred" else name,
                                     "year": y, **C.metrics(gy["y"], gy[name])})
    M, MY = pd.DataFrame(allm), pd.DataFrame(ally)
    if len(M) == 0:
        print("nothing evaluated"); return
    M.to_csv(out / "metrics.csv", index=False); MY.to_csv(out / "metrics_by_year.csv", index=False)
    for fset, ps in pred_store.items():
        if ps: pd.concat(ps).to_parquet(out / f"preds_{fset}.parquet")
    # ---- report
    for (s, fset), g in M.groupby(["series", "set"], sort=False):
        g = g.set_index("model")
        best_b = g.loc[C.BASELINES, "MAE"].idxmin()
        lines.append(f"== {s} [{fset}]   n = {int(g['n'].iloc[0]):,}, folds = {int(g['n_folds'].iloc[0])}, "
                     f"first test block {g['first_test'].iloc[0]}")
        lines.append(C.fmt(g[["MAE", "RMSE", "sMAPE_%", "MAE_asinh"]].loc[["lgbm", *C.BASELINES]]))
        sk = 1 - g.at["lgbm", "MAE"] / g.at[best_b, "MAE"]
        sk2 = 1 - g.at["lgbm", "MAE"] / g.at["b_prev_slot", "MAE"]
        P = pd.concat(pred_store[fset]); P = P[P.series == s].dropna(subset=["y", "pred", *C.BASELINES])
        P = P.sort_values("block_start_utc")
        L = 4 if s.endswith("|week") else 12
        st1, p1 = C.dm_test(P["y"], P["pred"], P["b_prev_slot"], L)
        st2, p2 = C.dm_test(P["y"], P["pred"], P[best_b], L)
        lines.append(f"   MAE skill vs best baseline ({best_b}): {sk*100:+.1f} %   vs b_prev_slot: {sk2*100:+.1f} %")
        lines.append(f"   Diebold-Mariano (|error|, NW lags {L}): vs b_prev_slot stat {st1:.2f} p {p1:.4f};"
                     f" vs {best_b} stat {st2:.2f} p {p2:.4f}   (stat < 0: LightGBM better)")
        yy = MY[(MY.series == s) & (MY.set == fset)].pivot(index="year", columns="model", values="MAE")
        yy["skill_vs_prev_slot_%"] = (1 - yy["lgbm"] / yy["b_prev_slot"]) * 100
        lines.append("   MAE by year:"); lines.append(C.fmt(yy[["lgbm", *C.BASELINES, "skill_vs_prev_slot_%"]]))
        lines.append("")
    # ---- weather ablation / perfect-forecast comparison
    if M["set"].nunique() > 1:
        lines.append("== Feature-set comparison: LightGBM MAE per series (same rows only if the sets share folds)")
        W = M[M.model == "lgbm"].pivot(index="series", columns="set", values="MAE")
        Wn = M[M.model == "lgbm"].pivot(index="series", columns="set", values="n")
        lines.append(C.fmt(W)); lines.append("   n per set:"); lines.append(C.fmt(Wn)); lines.append("")
    # ---- importance
    for fset, dd in imp_store.items():
        if not dd: continue
        I = pd.concat([pd.concat(l, axis=1).mean(axis=1).rename(s) for s, l in dd.items()], axis=1).fillna(0)
        I.to_csv(out / f"importance_{fset}.csv")
        F = pd.concat([pd.concat(l, axis=1).mean(axis=1).rename(s) for s, l in fam_store[fset].items()], axis=1).fillna(0)
        F.to_csv(out / f"importance_family_{fset}.csv")
        lines.append(f"== Gain importance [{fset}] (share of total gain, mean over folds)")
        lines.append("   family shares (%):"); lines.append(C.fmt(F * 100))
        for s in I.columns:
            top = I[s].sort_values(ascending=False).head(10)
            lines.append(f"   {s} top 10: " + "; ".join(f"{c} {v*100:.1f}%" for c, v in top.items()))
        lines.append("")
    (out / "report_rq2_2.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"total {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
