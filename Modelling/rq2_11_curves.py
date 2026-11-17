"""RQ2 sensitivity (appendix): do the EPEX SPOT day-ahead aggregated curves improve capacity-price forecasts?

A/B design (16 Nov 2026), same as Appendix D: the reference model of Chapter 4 (LightGBM, strict ex-ante
feature set, delta target, rolling origin, rq2_2_lightgbm.py settings) is estimated twice per series and
random seed - once with the thesis feature set and once with the curve lookbacks added - and the MAE
skill against the same-slot-previous-day forecast is compared on the same test blocks.

  * Thesis feature set: views.feature_columns(..., "exante") WITHOUT the FCR Cooperation members added on
    27 Oct 2026 (columns of be, nl, dk1, si, cz), i.e. the feature set of Section 3.84 (Path A).
  * Curve features (Clean/clean_epex_curves.py -> Clean/Data/epex/da_curves.parquet): the order book of the
    Swiss day-ahead auction is published with the result, D-1 after the auction ("D-1 after day-ahead
    coupling" = D-1 13:00 in the ex-ante plan). Every Swiss capacity auction closes before the day-ahead
    auction of its delivery day, so the curves enter only as LOOKBACKS at gate closure: mean of the 96 / 672
    latest published quarter-hours (24 h / 7 d), as for the other ex-ante lookbacks, plus the 24-h maximum
    price rise for +500 MW and minimum price fall for -500 MW. Coverage >= 50 % of the window, else NaN.
    Curves exist 2016-2025; curve features of blocks delivered from 1 Jan 2026 are set to NaN (licence).
  * Evaluation window: test blocks from 1 Jan 2021 to 31 Dec 2025 (both arms; 2026 blocks are predicted by
    the folds but not evaluated, because the curve features are missing there). The seed-1 baseline over
    the full test period is also reported, to compare with Table 5.
  * Output: per series and seed the MAE skill of both arms, the mean over seeds and the difference,
    Diebold-Mariano test of the seed-averaged forecasts (curves vs base, |error|, Newey-West lags 12 / 4),
    gain share of the curve features, run.log / run_meta.json, macOS notification.

Run from Master_Thesis with .venv active (after Clean/clean_epex_curves.py --write):
    python Modelling/rq2_11_curves.py --only "mFRR|up|week" --seeds 1      # quick check (seconds)
    python Modelling/rq2_11_curves.py                                       # full run, 3 seeds (~15-20 min)
Read-only on Clean/Data; writes only Modelling/Output/rq2_11_curves<tag>/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import run_utils as RU

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rq2_common as C  # noqa: E402
import rq2_2_lightgbm as L  # noqa: E402

CURVES = C.ROOT / "Clean" / "Data" / "epex" / "da_curves.parquet"
NEW_MEMBERS = ("be", "nl", "dk1", "si", "cz")           # added 27 Oct 2026, not in the thesis feature set
EVAL_END = pd.Timestamp("2026-01-01", tz=C.TZ)          # exclusive
LICENCE_END = pd.Timestamp("2026-01-01", tz=C.TZ)
BASE_CURVE = ["dp_up100_eur_mwh", "dp_up500_eur_mwh", "dp_down100_eur_mwh", "dp_down500_eur_mwh",
              "head_sell_mw", "head_buy_mw", "inel_sell_mw", "inel_buy_mw"]
QH = pd.Timedelta("15min").value


def curve_pub_and_values() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not CURVES.exists():
        raise SystemExit(f"{CURVES} missing: run python Clean/clean_epex_curves.py --write first")
    cv = pd.read_parquet(CURVES)
    ts = pd.DatetimeIndex(pd.to_datetime(cv.pop("ts_utc"), utc=True))
    cv.index = ts
    cv = cv[[f"ch_da_curve_{c}" for c in BASE_CURVE]].astype("float64")
    loc = ts.tz_convert(C.TZ).tz_localize(None)
    pub = (loc.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=13)).tz_localize(C.TZ).tz_convert("UTC")
    return ts.asi8, np.asarray(pub.asi8), cv


def window(x: np.ndarray, e: np.ndarray, n: int, how: str) -> np.ndarray:
    ok = ~np.isnan(x)
    out = np.full(len(e), np.nan)
    if how == "mean":
        cs = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
        cn = np.concatenate([[0], np.cumsum(ok)])
        s = np.maximum(e - n, 0)
        cnt = cn[e] - cn[s]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (cs[e] - cs[s]) / cnt
        out[cnt < 0.5 * n] = np.nan
        return out
    for i, ei in enumerate(e):                      # max / min (only 24 h windows, small)
        w = x[max(ei - n, 0):ei]
        w = w[~np.isnan(w)]
        if len(w) >= 0.5 * n:
            out[i] = w.max() if how == "max" else w.min()
    return out


def add_curve_lookbacks(df: pd.DataFrame, ts_ns, pub_ns, cv) -> list[str]:
    gc = pd.to_datetime(df["gate_closure_utc"], utc=True).to_numpy().astype("datetime64[ns]").astype("int64")
    e = np.searchsorted(pub_ns, gc, side="right")            # QHs published by gate closure (pub non-decreasing)
    cols = []
    late = (pd.to_datetime(df["block_start_utc"], utc=True) >= LICENCE_END.tz_convert("UTC")).to_numpy()
    for c in BASE_CURVE:
        x = cv[f"ch_da_curve_{c}"].to_numpy()
        for tag, n in (("lb24h", 96), ("lb7d", 672)):
            name = f"ch_da_curve_{c}__{tag}"
            v = window(x, e, n, "mean")
            v[late] = np.nan
            df[name] = v
            cols.append(name)
    for c, how in (("dp_up500_eur_mwh", "max"), ("dp_down500_eur_mwh", "min")):
        name = f"ch_da_curve_{c}__{how}24h"
        v = window(cv[f"ch_da_curve_{c}"].to_numpy(), e, 96, how)
        v[late] = np.nan
        df[name] = v
        cols.append(name)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=C.VIEWS)
    ap.add_argument("--only", default=None, help='series filter, e.g. "mFRR|up|4h"')
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    hp = SimpleNamespace(objective="l1", lr=0.05, leaves=15, colsample=0.3, rounds=800, threads=0)
    out = HERE / "Output" / f"rq2_11_curves{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    t_all = time.time()
    ts_ns, pub_ns, cv = curve_pub_and_values()
    lines = ["RQ2 sensitivity: EPEX day-ahead aggregated curves as lookback features (A/B, reference LightGBM)",
             f"seeds {a.seeds}; thesis feature set without {', '.join(NEW_MEMBERS)}; evaluation on test blocks "
             f"2021-01-01 to 2025-12-31 (seed-1 base also over the full test period)", ""]
    rows, imp_rows, preds_all = [], [], []
    for view in a.products:
        df = C.load_view(view)
        curve_cols = add_curve_lookbacks(df, ts_ns, pub_ns, cv)
        base, _ = L.feature_list(df, view, "exante")
        base = [c for c in base if c.split("_")[0] not in NEW_MEMBERS]
        cov = df[curve_cols].notna().groupby(pd.to_datetime(df["block_start_utc"], utc=True).dt.tz_convert(C.TZ).dt.year).mean()
        lines.append(f"[{view}] base features {len(base)}, curve features {len(curve_cols)}; curve coverage per delivery year "
                     f"(mean over columns): " + ", ".join(f"{y} {v:.2f}" for y, v in cov.mean(axis=1).items()))
        arms = {"base": base, "curves": base + curve_cols}
        X = {k: L.make_X(df, v) for k, v in arms.items()}
        for s in df["series"].unique():
            if a.only and a.only != s:
                continue
            gi = np.flatnonzero((df["series"] == s).to_numpy())
            g = df.iloc[gi]
            ya = np.arcsinh(g["y"].to_numpy(dtype=float))
            off = np.arcsinh(g["b_prev_slot"].to_numpy(dtype=float))
            yt = ya - off
            ok = ~np.isnan(yt)
            rm = 12 if s.endswith("|week") else 3
            mt = C.MIN_TRAIN[s.split("|")[2]]
            t0 = time.time()
            for arm, feats in arms.items():
                Xg = X[arm].iloc[gi]
                for seed in a.seeds:
                    for f, start, tr, te in C.fold_plan(g.reset_index(drop=True), rm, mt):
                        tr = tr[ok[tr]]
                        if len(tr) < mt:
                            continue
                        p, model, best = L.fit_predict(Xg.iloc[tr], yt[tr], Xg.iloc[te], hp, seed=seed)
                        r = g.iloc[te][["series", "block_start_utc", "y", *C.BASELINES]].copy()
                        r["pred"] = np.sinh(p + off[te])
                        r["arm"], r["seed"], r["fold"] = arm, seed, f
                        preds_all.append(r)
                        if arm == "curves":
                            gain = pd.Series(model.feature_importance("gain"), index=feats)
                            imp_rows.append({"series": s, "seed": seed, "fold": f,
                                             "curve_gain_share": float(gain[curve_cols].sum() / max(gain.sum(), 1e-12)),
                                             **{c: float(gain[c] / max(gain.sum(), 1e-12)) for c in curve_cols}})
            print(f"{view} {s:16s} done {time.time() - t0:6.1f}s", flush=True)
    P = pd.concat(preds_all, ignore_index=True)
    P.to_parquet(out / "preds.parquet")
    P = P.dropna(subset=["y", "pred", *C.BASELINES])
    P["eval"] = pd.to_datetime(P["block_start_utc"], utc=True) < EVAL_END.tz_convert("UTC")

    # ---- skill per series / arm / seed
    for (s, arm, seed), g in P.groupby(["series", "arm", "seed"], sort=False):
        for per, gg in (("2021-2025", g[g["eval"]]), ("full", g)):
            if len(gg) == 0:
                continue
            mae, mae_b = (gg["pred"] - gg["y"]).abs().mean(), (gg["b_prev_slot"] - gg["y"]).abs().mean()
            rows.append({"series": s, "arm": arm, "seed": seed, "period": per, "n": len(gg),
                         "MAE": mae, "MAE_prev_slot": mae_b, "skill_%": (1 - mae / mae_b) * 100})
    S = pd.DataFrame(rows)
    S.to_csv(out / "skill.csv", index=False)
    full1 = S[(S.period == "full") & (S.arm == "base") & (S.seed == min(a.seeds))].set_index("series")["skill_%"]
    E = S[S.period == "2021-2025"]
    T = E.pivot_table(index="series", columns=["arm", "seed"], values="skill_%")
    summ = pd.DataFrame({"base_full_seed1": full1,
                         "base_mean": E[E.arm == "base"].groupby("series")["skill_%"].mean(),
                         "curves_mean": E[E.arm == "curves"].groupby("series")["skill_%"].mean()})
    summ["diff_pp"] = summ["curves_mean"] - summ["base_mean"]
    summ["seed_spread_base_pp"] = E[E.arm == "base"].groupby("series")["skill_%"].agg(lambda v: v.max() - v.min())
    # ---- DM test on seed-averaged forecasts
    dm = []
    for s, g in P[P["eval"]].groupby("series", sort=False):
        w = g.pivot_table(index="block_start_utc", columns="arm", values="pred", aggfunc="mean").dropna()
        yy = g.drop_duplicates("block_start_utc").set_index("block_start_utc").loc[w.index, "y"]
        L_ = 4 if s.endswith("|week") else 12
        st, pv = C.dm_test(yy.to_numpy(), w["curves"].to_numpy(), w["base"].to_numpy(), L_)
        mae_c, mae_bse = (w["curves"] - yy).abs().mean(), (w["base"] - yy).abs().mean()
        dm.append({"series": s, "n": len(w), "MAE_base_avg": mae_bse, "MAE_curves_avg": mae_c,
                   "rel_change_%": (mae_c / mae_bse - 1) * 100, "DM_stat": st, "p": pv})
    DM = pd.DataFrame(dm).set_index("series")
    DM.to_csv(out / "dm_curves_vs_base.csv")
    summ = summ.join(DM[["DM_stat", "p"]])
    summ.to_csv(out / "summary.csv")
    lines.append("\n== MAE skill vs same slot previous day (%), test blocks 2021-2025, per seed")
    lines.append(C.fmt(T))
    lines.append("\n== Summary: base_full_seed1 = base over the full test period (compare Table 5); means over seeds on 2021-2025;"
                 " diff = curves - base (pp); DM on seed-averaged forecasts (stat < 0: curves more accurate)")
    lines.append(C.fmt(summ))
    if imp_rows:
        I = pd.DataFrame(imp_rows)
        I.to_csv(out / "importance_curves.csv", index=False)
        lines.append("\n== Gain share of the curve features in the curves arm (mean over folds and seeds)")
        lines.append(C.fmt(I.groupby("series")["curve_gain_share"].agg(["mean", "min", "max"])))
        top = I.drop(columns=["seed", "fold", "curve_gain_share"]).groupby("series").mean()
        for s, r in top.iterrows():
            t = r.sort_values(ascending=False).head(5)
            lines.append(f"   {s}: " + "; ".join(f"{c.replace('ch_da_curve_', '')} {v * 100:.1f}%" for c, v in t.items()))
    (out / "report_rq2_11.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    RU.summary("curves - base (pp): " + ", ".join(f"{k.replace('|', ' ')} {v:+.1f}" for k, v in summ["diff_pp"].items()))
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    RU.run(main, "RQ2 curves A/B")
