"""RQ2 step 9 (done_todo 2.2 item 7, Tier 3 of ML_Options.md): does the training window matter?

Collects the Tier 1 runs with a different training window (rq2_5_tier1.py --window ...) and compares each with
the round-2 expanding-window run (Output/rq2_5_<product>) on the SAME test rows. Nothing is refitted.

Variants (--variants name=folder-prefix; a variant is skipped when none of its folders exists):
  sliding        rq2_9s_<product>   last 24 months (weekly 48) before every fold start
  regime         rq2_9r_<product>   since the latest market-design break (ex ante), else expanding
  regime_crisis  rq2_9rc_<product>  as regime + the crisis start / end as breaks (chosen ex post -> robustness)
A per-product folder may be split (e.g. rq2_9s_mfrr_upw); --extra name=folder adds any folder to a variant.

Per series x model (lear, lgbm_q, cat_q, qrf, svr):
  overall    MAE skill vs same-slot-yesterday of expanding and every variant; Diebold-Mariano |error| variant vs
             expanding (stat < 0 = variant better; Newey-West lags 12 for 4h, 4 weekly); pinball DM for
             lgbm_q / cat_q / qrf; change of the best model per series.
  by period  the same split into periods declared before looking at the results (rq2_8 + the Sep 2025 merger):
             1_pre_crisis (< 2021-09-01), 2_crisis (.. 2023-06-30), 3_post_crisis (.. 2025-09-27),
             4_post_merger (>= 2025-09-28). p_holm = Holm over the periods of one (series, model, variant).

    python Modelling/rq2_9_regime_compare.py
    python Modelling/rq2_9_regime_compare.py --variants sliding=rq2_9s regime=rq2_9r --models lgbm_q cat_q
Output: Modelling/Output/rq2_9_compare/{compare_overall.csv, compare_by_period.csv, best_model.csv,
        report_rq2_9.txt, run.log, run_meta.json}
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_utils as RU  # run.log, run_meta.json, run history, macOS notification

import rq2_common as C
from rq2_5_tier1 import QS, QTAG, dm_loss, pinball
from rq2_8_subperiods import holm

OUT = C.ROOT / "Modelling" / "Output"
MODELS = ["lear", "lgbm_q", "cat_q", "qrf", "svr"]
PROB = ["lgbm_q", "cat_q", "qrf"]
PERIODS = [("1_pre_crisis", None, "2021-09-01"), ("2_crisis", "2021-09-01", "2023-07-01"),
           ("3_post_crisis", "2023-07-01", "2025-09-28"), ("4_post_merger", "2025-09-28", None)]


def load_run(prefix: str, extra: list[str]) -> pd.DataFrame | None:
    """All preds.parquet of folders Output/<prefix>_* (plus extra folders); one row per (series, block)."""
    files = sorted(OUT.glob(f"{prefix}_*/preds.parquet")) + [OUT / e / "preds.parquet" for e in extra]
    files = [f for f in files if f.exists()]
    if not files:
        return None
    P = pd.concat([pd.read_parquet(f).assign(_src=f.parent.name) for f in files], ignore_index=True)
    return P.drop_duplicates(["series", "block_start_utc"], keep="last")


def period_of(t: pd.Series) -> pd.Series:
    loc = t.dt.tz_convert(C.TZ)
    lab = pd.Series("", index=t.index)
    for name, lo, hi in PERIODS:
        m = pd.Series(True, index=t.index)
        if lo:
            m &= loc >= pd.Timestamp(lo, tz=C.TZ)
        if hi:
            m &= loc < pd.Timestamp(hi, tz=C.TZ)
        lab[m] = name
    return lab


def pl_mean(y, E, m):
    return np.mean([pinball(y, E[f"q_{m}_{t}"].to_numpy(float), q) for q, t in zip(QS, QTAG)], axis=0)


def compare(D: pd.DataFrame, V: pd.DataFrame, s: str, var: str, models: list[str]):
    """D = expanding run, V = variant, both one series. Returns overall rows, period rows."""
    L = 4 if s.endswith("|week") else 12
    cols = ["series", "block_start_utc"]
    M = D.merge(V.drop(columns=["y", *C.BASELINES], errors="ignore"), on=cols, suffixes=("", "__v"))
    ov, per = [], []
    for m in models:
        need = [f"p_{m}", f"p_{m}__v"]
        if not all(c in M for c in need):
            continue
        E = M.dropna(subset=["y", "b_prev_slot", *need]).sort_values("block_start_utc").reset_index(drop=True)
        if len(E) < 30:
            continue
        y = E["y"].to_numpy(float)
        e0 = np.abs(E[f"p_{m}"].to_numpy(float) - y)
        e1 = np.abs(E[f"p_{m}__v"].to_numpy(float) - y)
        nb = np.abs(E["b_prev_slot"].to_numpy(float) - y)
        st, p = dm_loss(e1 - e0, L)
        row = {"series": s, "variant": var, "model": m, "n": len(E), "mae_expanding": e0.mean(), "mae_variant": e1.mean(),
               "change_%": (e1.mean() / e0.mean() - 1) * 100, "skill_expanding_%": (1 - e0.mean() / nb.mean()) * 100,
               "skill_variant_%": (1 - e1.mean() / nb.mean()) * 100, "dm": st, "p": p}
        prob_ok = m in PROB and all(f"q_{m}_{t}" in E and f"q_{m}_{t}__v" in E for t in QTAG)
        if prob_ok:
            Ev = E.rename(columns={f"q_{m}_{t}__v": f"q_v{m}_{t}" for t in QTAG})
            p0, p1 = pl_mean(y, E, m), pl_mean(y, Ev, f"v{m}")
            stq, pq = dm_loss(p1 - p0, L)
            row.update(pinball_expanding=p0.mean(), pinball_variant=p1.mean(), dm_pinball=stq, p_pinball=pq)
        ov.append(row)
        lab = period_of(E["block_start_utc"])
        for g, idx in E.groupby(lab).groups.items():
            ii = np.asarray(idx)
            st_g, p_g = dm_loss(e1[ii] - e0[ii], L) if len(ii) >= 30 else (np.nan, np.nan)
            per.append({"series": s, "variant": var, "model": m, "period": g, "n": len(ii),
                        "skill_expanding_%": (1 - e0[ii].mean() / nb[ii].mean()) * 100,
                        "skill_variant_%": (1 - e1[ii].mean() / nb[ii].mean()) * 100,
                        "change_%": (e1[ii].mean() / e0[ii].mean() - 1) * 100, "dm": st_g, "p": p_g})
    return ov, per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--default", default="rq2_5", help="folder prefix of the expanding-window run (rq2_5_<product>)")
    ap.add_argument("--variants", nargs="+", default=["sliding=rq2_9s", "regime=rq2_9r", "regime_crisis=rq2_9rc"])
    ap.add_argument("--extra", nargs="*", default=[], help="name=folder: add a folder to a variant")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT / "rq2_9_compare"
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    t0 = time.time()
    D = pd.concat([pd.read_parquet(OUT / f"{a.default}_{v}" / "preds.parquet") for v in C.VIEWS
                   if (OUT / f"{a.default}_{v}" / "preds.parquet").exists()], ignore_index=True)
    extra = {}
    for e in a.extra:
        k, v = e.split("=", 1); extra.setdefault(k, []).append(v)
    runs = {}
    for spec in a.variants:
        name, prefix = spec.split("=", 1)
        R = load_run(prefix, extra.get(name, []))
        if R is None:
            print(f"variant {name}: no folder {prefix}_* - skipped"); continue
        runs[name] = R
        print(f"variant {name}: {', '.join(sorted(R['_src'].unique()))}")
    if not runs:
        print("no variant run found - run rq2_5_tier1.py --window ... first"); return
    OV, PER = [], []
    for s in D["series"].unique():
        Ds = D[D["series"] == s]
        for name, R in runs.items():
            Rs = R[R["series"] == s]
            if len(Rs):
                o, p = compare(Ds, Rs, s, name, a.models)
                OV += o; PER += p
    OV, PER = pd.DataFrame(OV), pd.DataFrame(PER)
    if len(PER):
        PER["p_holm"] = PER.groupby(["series", "variant", "model"], group_keys=False)["p"].apply(holm)
    OV.to_csv(out / "compare_overall.csv", index=False); PER.to_csv(out / "compare_by_period.csv", index=False)

    # best model per series and variant (same rows as the comparison)
    B = []
    for (s, v), g in OV.groupby(["series", "variant"], sort=False):
        g = g.set_index("model")
        B.append({"series": s, "variant": v, "best_expanding": g["mae_expanding"].idxmin(),
                  "skill_best_expanding_%": g["skill_expanding_%"].max(), "best_variant": g["mae_variant"].idxmin(),
                  "skill_best_variant_%": g["skill_variant_%"].max()})
    B = pd.DataFrame(B); B.to_csv(out / "best_model.csv", index=False)

    rep = ["RQ2 step 9 - training window (done_todo 2.2 item 7): variant vs expanding window (round 2), same rows.",
           "change_% = MAE change of the variant (negative = better); dm < 0 = variant better; Newey-West lags 12 (4h) / 4 (week).",
           "Variants: " + ", ".join(f"{k} ({', '.join(sorted(r['_src'].unique()))})" for k, r in runs.items()), ""]
    for v in runs:
        T = OV[OV["variant"] == v]
        if T.empty:
            continue
        rep.append(f"== {v}: MAE skill vs naive (%) expanding -> variant, p of DM")
        T = T.assign(cell=[f"{r['skill_expanding_%']:.1f} -> {r['skill_variant_%']:.1f} (p={r['p']:.3f})"
                           + ((" better" if r["dm"] < 0 else " worse") if r["p"] < 0.05 else "") for _, r in T.iterrows()])
        rep.append(T.pivot(index="series", columns="model", values="cell").to_string())
        nb_, nw_ = ((T["p"] < 0.05) & (T["dm"] < 0)).sum(), ((T["p"] < 0.05) & (T["dm"] > 0)).sum()
        rep.append(f"   significantly better: {nb_} / worse: {nw_} of {len(T)} series x model pairs")
        RU.summary(f"{v}: better {nb_}, worse {nw_} of {len(T)}")
        if "p_pinball" in T:
            Tp = T.dropna(subset=["p_pinball"])
            rep.append(f"   pinball: significantly better {((Tp.p_pinball < 0.05) & (Tp.dm_pinball < 0)).sum()} / worse "
                       f"{((Tp.p_pinball < 0.05) & (Tp.dm_pinball > 0)).sum()} of {len(Tp)}")
        Bv = B[B["variant"] == v]
        rep.append("   best model per series (expanding -> variant): " + "; ".join(
            f"{r['series']}: {r['best_expanding']} {r['skill_best_expanding_%']:+.1f}% -> {r['best_variant']} "
            f"{r['skill_best_variant_%']:+.1f}%" for _, r in Bv.iterrows()))
        Pv = PER[PER["variant"] == v]
        if len(Pv):
            rep.append("   by period - MAE change % of the variant ((+) / (-) = Holm-significantly better / worse):")
            Pv = Pv.assign(cell=lambda d: d["change_%"].round(1).astype(str)
                           + np.where(d["p_holm"] < 0.05, np.where(d["dm"] < 0, " (+)", " (-)"), ""))
            for m in a.models:
                X = Pv[Pv["model"] == m]
                if len(X):
                    rep.append(f"   {m}:")
                    rep.append("   " + X.pivot(index="series", columns="period", values="cell").to_string().replace("\n", "\n   "))
        rep.append("")
    rep.append(f"runtime {time.time() - t0:.0f}s")
    (out / "report_rq2_9.txt").write_text("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    RU.run(main, "RQ2 training-window comparison")
