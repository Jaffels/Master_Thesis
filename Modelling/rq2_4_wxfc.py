"""RQ2 step 4 (todo 9.2e): weather-forecast robustness run, 4h series, test period from 2022.

Question: do archived weather FORECASTS (Open-Meteo previous-runs, freshest lead published by gate
closure, build_weather_fc_features.py) add anything to the strict ex-ante model? Main models use only
lagged observed weather (lookbacks); observed weather at delivery (__pf) is the perfect-forecast
upper bound. Weekly blocks have no forecast weather -> 4h series only (FCR; aFRR from Sep 2025; mFRR).
Coverage: temperature 2022-, ghi / humidity / wind / precipitation 2024-; earlier blocks are NaN
in training and LightGBM treats them as missing.

Feature sets (all LightGBM, delta target, same folds as rq2_2, refit every --refit-months):
  noweather   strict ex-ante without any MeteoSwiss column
  lookback    strict ex-ante (incl. lagged observed weather)          = 'exante' of rq2_2
  wxfc        noweather + weather forecasts (__wxfc)
  lookback_wxfc  lookback + weather forecasts
  pf_weather  noweather + observed weather at delivery (__pf, MeteoSwiss only)  = weather upper bound
Evaluation on common rows: window A = test blocks from 2022-01-01 with a forecast temperature,
window B = test blocks from 2024-01-01 (all forecast variables). DM test: stat < 0 = first model better.

    python Modelling/rq2_4_wxfc.py [--only "mFRR|up|4h"] [--start 2022-01-01]
Output: Modelling/Output/rq2_4/{preds.parquet, metrics.csv, report_rq2_4.txt}
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
import rq2_common as C
import rq2_2_lightgbm as L

SETS = ["noweather", "lookback", "wxfc", "lookback_wxfc", "pf_weather"]
PAIRS = [("wxfc", "noweather"), ("wxfc", "lookback"), ("lookback_wxfc", "lookback"),
         ("pf_weather", "noweather"), ("pf_weather", "wxfc")]


def features(df, view, fset):
    d = C.V.dictionary(view, "exante")
    nw, _ = L.feature_list(df, view, "exante_noweather")
    lb, _ = L.feature_list(df, view, "exante")
    wx = C.V.weather_fc_columns(df)
    pf_all, _ = L.feature_list(df, view, "perfect")
    pf_w = [c for c in pf_all if c.endswith("__pf") and "MeteoSwiss" in str(d["source"].get(c, ""))]
    return {"noweather": nw, "lookback": lb, "wxfc": nw + wx, "lookback_wxfc": lb + wx, "pf_weather": nw + pf_w}[fset]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=C.VIEWS)
    ap.add_argument("--only", default=None)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    ap.add_argument("--no-report", action="store_true", help="only save this chunk's predictions")
    ap.add_argument("--out", default=str(C.ROOT / "Modelling" / "Output" / "rq2_4"))
    a = ap.parse_args()
    lg = SimpleNamespace(mode="delta", objective="l1", lr=0.05, leaves=15, colsample=0.3, rounds=800, threads=0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(a.start, tz=C.TZ).tz_convert("UTC")
    store = []
    for view in a.products:
        df = C.load_view(view, weather_fc=True)
        for fset in a.sets:
            feats = features(df, view, fset)
            X = L.make_X(df, feats)
            for s in df["series"].unique():
                if not s.endswith("|4h") or (a.only and a.only != s):
                    continue
                gi = np.flatnonzero((df["series"] == s).to_numpy()); g = df.iloc[gi]; Xg = X.iloc[gi]
                ya = np.arcsinh(g["y"].to_numpy(float)); off = np.arcsinh(g["b_prev_slot"].to_numpy(float))
                yt = ya - off; ok = ~np.isnan(yt)
                mt = C.MIN_TRAIN["4h"]; t0 = time.time(); n = 0
                for f, fs, tr, te in C.fold_plan(g.reset_index(drop=True), a.refit_months, mt):
                    if fs < start: continue
                    tr = tr[ok[tr]]
                    if len(tr) < mt: continue
                    p, _, _ = L.fit_predict(Xg.iloc[tr], yt[tr], Xg.iloc[te], lg)
                    r = g.iloc[te][["series", "auction_id", "block_start_utc", "y", *C.BASELINES]].copy()
                    r["pred"] = np.sinh(p + off[te]); r["set"] = fset
                    r["has_temp_fc"] = g.iloc[te]["ch_temp_lw_degc__wxfc"].notna().to_numpy()
                    r["has_all_fc"] = g.iloc[te]["ch_ghi_lw_wm2__wxfc"].notna().to_numpy()
                    store.append(r); n += 1
                print(f"{view} {fset:14s} {s:14s} folds={n:3d} {time.time()-t0:6.1f}s", flush=True)
    if store:                                   # chunked runs: each call saves its own file, the report merges all
        tag = "_".join(a.products) + "_" + "_".join(a.sets) + ("_" + a.only.replace("|", "-") if a.only else "")
        pd.concat(store).to_parquet(out / f"preds_{tag}.parquet")
    files = sorted(out.glob("preds_*.parquet"))
    if not files or a.no_report:
        print("saved chunk" if store else "nothing evaluated"); return
    P = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates(["series", "auction_id", "block_start_utc", "set"], keep="last")
    wide = P.pivot_table(index=["series", "auction_id", "block_start_utc"], columns="set", values="pred")
    meta = P.drop_duplicates(["series", "auction_id", "block_start_utc"]).set_index(["series", "auction_id", "block_start_utc"])
    wide = wide.join(meta[["y", "b_prev_slot", "has_temp_fc", "has_all_fc"]]).dropna()
    rows, lines = [], ["RQ2 9.2e - weather-forecast robustness, 4h series, LightGBM delta; MAE in EUR/MWh on common rows", ""]
    wins = {"A (2022-, forecast temperature available)": wide["has_temp_fc"].astype(bool),
            "B (2024-, all forecast variables)": wide["has_all_fc"].astype(bool)}
    sets = [x for x in SETS if x in wide.columns]
    for wname, mask in wins.items():
        w = wide[mask].reset_index().sort_values("block_start_utc")
        for s, g in w.groupby("series", sort=False):
            r = {"window": wname[0], "series": s, "n": len(g), "MAE_naive": C.metrics(g.y, g.b_prev_slot)["MAE"]}
            for x in sets: r[f"MAE_{x}"] = C.metrics(g.y, g[x])["MAE"]
            for x, y_ in PAIRS:
                if x in g and y_ in g:
                    r[f"DM_{x}_vs_{y_}_p"] = C.dm_test(g.y, g[x], g[y_], 12)[1]
                    r[f"gain_{x}_vs_{y_}_%"] = (1 - r[f"MAE_{x}"] / r[f"MAE_{y_}"]) * 100
            rows.append(r)
        lines.append(f"== Window {wname}")
        sub = pd.DataFrame([r for r in rows if r["window"] == wname[0]]).set_index("series")
        lines.append(C.fmt(sub[["n", "MAE_naive", *[f"MAE_{x}" for x in sets]]]))
        gcols = [c for c in sub.columns if c.startswith("gain_")]
        lines.append("   MAE gain of the first set over the second (%), and DM p-value:")
        lines.append(C.fmt(sub[gcols])); lines.append(C.fmt(sub[[c for c in sub.columns if c.startswith("DM_")]])); lines.append("")
    M = pd.DataFrame(rows); M.to_csv(out / "metrics.csv", index=False)
    (out / "report_rq2_4.txt").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
