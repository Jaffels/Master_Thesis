"""RQ2 step 7 (Tier 2, item 4 of done_todo 2.1): forecast combination / stacking of Tier 1 (+ Tier 2).

Reads the test-block predictions already produced (nothing is refitted):
  Tier 1  Output/rq2_5_<product>/preds.parquet (or Output/rq2_5/preds.parquet): lear, lgbm_l1, lgbm_q,
          cat_q, qrf, svr (+ quantiles of lgbm_q, cat_q, qrf)
  Tier 2  every Output/rq2_6*/preds.parquet whose run was not a --quick run: tft, nhits, tabpfn, chronos.
          If a model / series appears in several folders, the newest file wins.
Per series, the components are all point models that cover >= 95 % of the Tier 1 test rows
(--exclude drops models, --components fixes the list).

Combinations (point, EUR/MWh). Every weight is estimated ONLY from earlier test folds of the same
series (those forecasts were genuinely out of sample), so the combination is itself ex ante. Folds
with fewer than --min-hist earlier rows use the equal-weight average (flag `warmup` in weights).
  avg_all        equal-weight mean of all components
  median_all     median of all components (robust to one bad model)
  avg_trees      mean of lgbm_l1, lgbm_q, cat_q, qrf
  avg_lear_trees mean of LEAR + the trees
  best_past      the component with the lowest MAE on all earlier folds (model selection benchmark)
  avg_top3       mean of the 3 components with the lowest MAE on earlier folds
  invmae         weights proportional to 1 / MAE on earlier folds
  ridge          ridge stacker on asinh(component forecasts), non-negative weights + intercept,
                 fitted on all earlier folds (--ridge-alpha)
Probabilistic: qavg = quantile averaging (Vincentisation) of all probabilistic components, qavg_t1
= Tier 1 only; then sequential CQR exactly as in rq2_5.

Evaluation on the common rows: MAE / RMSE / sMAPE / MAE_asinh, skill vs naive / LEAR / best single
component (chosen ex post, strict), Diebold-Mariano |error| vs LEAR, lgbm_l1, best single component,
naive; pinball, coverage raw / CQR, DM pinball vs lgbm_q and the best probabilistic component.

    python Modelling/rq2_7_stack.py                          # all series with Tier 1 predictions
    python Modelling/rq2_7_stack.py --exclude svr chronos    # without the weakest components
Output: Modelling/Output/rq2_7/{preds.parquet, metrics_point.csv, metrics_prob.csv, metrics_by_year.csv,
        dm.csv, weights.csv, components.csv, report_rq2_7.txt, run.log, run_meta.json, diagnostics/}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_utils as RU  # run.log, run_meta.json, run history, macOS notification

import rq2_common as C
from rq2_5_tier1 import INTERVALS, QS, QTAG, cqr, dm_loss, pinball

POINT = ["lear", "lgbm_l1", "lgbm_q", "cat_q", "qrf", "svr", "tft", "nhits", "tabpfn", "chronos"]
PROB = ["lgbm_q", "cat_q", "qrf", "tft", "nhits", "tabpfn", "chronos"]
TIER1 = {"lear", "lgbm_l1", "lgbm_q", "cat_q", "qrf", "svr"}
TREES = ["lgbm_l1", "lgbm_q", "cat_q", "qrf"]
METHODS = ["avg_all", "median_all", "avg_trees", "avg_lear_trees", "best_past", "avg_top3", "invmae", "ridge"]
KEYS = ["series", "product", "direction", "procurement", "auction_id", "block_start_utc", "slot", "y", "fold",
        "n_train", *C.BASELINES]


def _is_quick(p: Path) -> bool:
    try:
        meta = json.loads((p.parent / "run_meta.json").read_text())
        return "--quick" in meta.get("argv", []) or any(str(x).startswith("--max-folds") for x in meta.get("argv", []))
    except Exception:
        return False


def load_all(a):
    base = C.ROOT / "Modelling" / "Output"
    t1 = [Path(p) for p in a.tier1] if a.tier1 else \
        [base / f"rq2_5_{v}" / "preds.parquet" for v in C.VIEWS] + [base / "rq2_5" / "preds.parquet"]
    parts = []
    for p in t1:
        p = p if p.suffix == ".parquet" else p / "preds.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            parts.append(d)
            print(f"   tier 1: {_rel(p)}  ({len(d):,} rows)")
    if not parts:
        raise SystemExit("no Tier 1 predictions found (run rq2_5_tier1.py first)")
    T1 = pd.concat(parts, ignore_index=True).drop_duplicates(["series", "block_start_utc"])
    reg = [c for c in T1.columns if c.startswith("regime_")]
    keep = [c for c in KEYS + reg if c in T1.columns] + [c for c in T1.columns if c.startswith(("p_", "q_"))
                                                       and _model_of(c) in TIER1]
    P = T1[keep].copy()
    t2 = [Path(p) for p in a.tier2] if a.tier2 is not None else sorted(base.glob("rq2_6*/preds.parquet"),
                                                                      key=lambda p: p.stat().st_mtime)
    for p in t2:
        p = p if p.suffix == ".parquet" else p / "preds.parquet"
        if not p.exists():
            continue
        if _is_quick(p) and not a.allow_quick:
            print(f"   tier 2: {_rel(p)} skipped (quick / max-folds run)")
            continue
        d = pd.read_parquet(p)
        cols = [c for c in d.columns if c.startswith(("p_", "q_")) and _model_of(c) not in TIER1]
        if not cols:
            continue
        print(f"   tier 2: {_rel(p)}  models {sorted({_model_of(c) for c in cols})}")
        d = d[["series", "block_start_utc", *cols]]
        P = P.merge(d, on=["series", "block_start_utc"], how="left", suffixes=("", "__new"))
        for c in cols:                                  # newest file wins where it has values
            if f"{c}__new" in P:
                P[c] = P[f"{c}__new"].combine_first(P[c])
                P = P.drop(columns=f"{c}__new")
    return P


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(C.ROOT))
    except ValueError:
        return str(p)


def _model_of(c: str) -> str:
    if c.startswith("p_"):
        return c[2:]
    return c[2:].rsplit("_", 1)[0]                      # q_<model>_<tag>


def combine(E: pd.DataFrame, comps: list[str], a):
    """Sequential combinations; returns dict method -> forecast array and a weights log."""
    y = E["y"].to_numpy(float)
    F = np.column_stack([E[f"p_{m}"].to_numpy(float) for m in comps])
    folds = E["fold"].to_numpy()
    out = {m: np.full(len(E), np.nan) for m in METHODS}
    wlog = []
    ti = [i for i, m in enumerate(comps) if m in TREES]
    li = [i for i, m in enumerate(comps) if m in TREES or m == "lear"]
    for f in np.unique(folds):
        idx, past = folds == f, folds < f
        Ff = F[idx]
        out["avg_all"][idx] = Ff.mean(1)
        out["median_all"][idx] = np.median(Ff, 1)
        out["avg_trees"][idx] = Ff[:, ti].mean(1) if ti else np.nan
        out["avg_lear_trees"][idx] = Ff[:, li].mean(1) if li else np.nan
        warm = past.sum() < a.min_hist
        if warm:
            w_eq = np.full(len(comps), 1 / len(comps))
            for m in ["best_past", "avg_top3", "invmae", "ridge"]:
                out[m][idx] = Ff.mean(1)
            wlog += [{"fold": f, "method": m, "component": c, "weight": w, "warmup": True}
                     for m in ["invmae"] for c, w in zip(comps, w_eq)]
            continue
        mae = np.abs(F[past] - y[past][:, None]).mean(0)
        b = int(np.argmin(mae))
        out["best_past"][idx] = Ff[:, b]
        top = np.argsort(mae)[:3]
        out["avg_top3"][idx] = Ff[:, top].mean(1)
        w = (1 / mae) / (1 / mae).sum()
        out["invmae"][idx] = Ff @ w
        from sklearn.linear_model import Ridge
        rg = Ridge(alpha=a.ridge_alpha, positive=True).fit(np.arcsinh(F[past]), np.arcsinh(y[past]))
        out["ridge"][idx] = np.sinh(rg.predict(np.arcsinh(Ff)))
        wlog += [{"fold": f, "method": "best_past", "component": comps[b], "weight": 1.0, "warmup": False}]
        wlog += [{"fold": f, "method": "avg_top3", "component": comps[i], "weight": 1 / 3, "warmup": False} for i in top]
        wlog += [{"fold": f, "method": "invmae", "component": c, "weight": wi, "warmup": False} for c, wi in zip(comps, w)]
        wlog += [{"fold": f, "method": "ridge", "component": c, "weight": wi, "warmup": False}
                 for c, wi in zip(comps, rg.coef_)] + [{"fold": f, "method": "ridge", "component": "(intercept)",
                                                         "weight": float(rg.intercept_), "warmup": False}]
    return out, wlog


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, help="series filter(s)")
    ap.add_argument("--tier1", nargs="*", default=None)
    ap.add_argument("--tier2", nargs="*", default=None, help="Tier 2 preds files / folders (default: all Output/rq2_6*)")
    ap.add_argument("--allow-quick", action="store_true", help="also use Tier 2 outputs from --quick / --max-folds runs")
    ap.add_argument("--exclude", nargs="*", default=[], help="components to leave out, e.g. svr chronos")
    ap.add_argument("--components", nargs="*", default=None, help="fixed component list")
    ap.add_argument("--min-hist", type=int, default=100, help="earlier rows needed before weights are estimated (weekly: 50)")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--coverage", type=float, default=0.95, help="a component must cover this share of the rows")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else C.ROOT / "Modelling" / "Output" / "rq2_7"
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    t0 = time.time()
    P = load_all(a)
    allE, MP, MY, DM, MQ, W, CP, BEST = [], [], [], [], [], [], [], {}
    for s, g in P.groupby("series", sort=False):
        if a.only and s not in a.only:
            continue
        cand = a.components or [m for m in POINT if m not in a.exclude]
        comps = [m for m in cand if f"p_{m}" in g and g[f"p_{m}"].notna().mean() >= a.coverage]
        missing = [m for m in cand if m not in comps and f"p_{m}" in g and g[f"p_{m}"].notna().any()]
        if len(comps) < 2:
            print(f"== {s}: fewer than 2 components, skipped"); continue
        E = g.dropna(subset=["y", *[f"p_{m}" for m in comps], *C.BASELINES]).sort_values("block_start_utc").reset_index(drop=True)
        mh = a.min_hist if not s.endswith("|week") else min(a.min_hist, 50)
        a_s = argparse.Namespace(**{**vars(a), "min_hist": mh})
        comb, wlog = combine(E, comps, a_s)
        for m, v in comb.items():
            E[f"p_{m}"] = v
        W += [{"series": s, **w} for w in wlog]
        # quantile averaging
        pc = [m for m in PROB if m in comps and all(f"q_{m}_{t}" in E and E[f"q_{m}_{t}"].notna().all() for t in QTAG)]
        qsets = {"qavg": pc, "qavg_t1": [m for m in pc if m in TIER1]}
        prob = []
        for name, ms in qsets.items():
            if len(ms) < 2 or (name == "qavg_t1" and ms == pc):
                continue
            Q = np.sort(np.mean([np.column_stack([E[f"q_{m}_{t}"] for t in QTAG]) for m in ms], axis=0), axis=1)
            for k, t in enumerate(QTAG):
                E[f"q_{name}_{t}"] = Q[:, k]
            prob.append(name)
        prob_all = pc + prob
        for m in prob_all:
            for lev, (lo_t, hi_t) in INTERVALS.items():
                E[f"cqr{lev}_lo_{m}"], E[f"cqr{lev}_hi_{m}"] = cqr(E, m, lo_t, hi_t, lev)
        CP.append({"series": s, "components": " ".join(comps), "prob_components": " ".join(pc),
                   "left_out_low_coverage": " ".join(missing), "n_rows": len(E)})
        print(f"== {s}: {len(E):,} rows, components {comps}" + (f" (low coverage, left out: {missing})" if missing else ""))
        # ---- evaluation
        L = 4 if s.endswith("|week") else 12
        E["year"] = E["block_start_utc"].dt.tz_convert(C.TZ).dt.year
        y = E["y"].to_numpy(float)
        names = {**{f"p_{m}": m for m in comps + METHODS}, **{b: b for b in C.BASELINES}}
        for col, nm in names.items():
            if E[col].isna().all():
                continue
            MP.append({"series": s, "model": nm, "kind": "combination" if nm in METHODS else ("baseline" if nm in C.BASELINES else "component"),
                       **C.metrics(E["y"], E[col])})
            for yr, gy in E.groupby("year"):
                MY.append({"series": s, "model": nm, "year": yr, **C.metrics(gy["y"], gy[col])})
        best = min(comps, key=lambda m: np.abs(E[f"p_{m}"] - E["y"]).mean())
        BEST[s] = best
        for m in METHODS:
            if E[f"p_{m}"].isna().all():
                continue
            e1 = np.abs(E[f"p_{m}"].to_numpy(float) - y)
            for ref in dict.fromkeys([r for r in ["lear", "lgbm_l1"] if r in comps] + [best, "b_prev_slot"]):
                rc = ref if ref in C.BASELINES else f"p_{ref}"
                st, p = dm_loss(e1 - np.abs(E[rc].to_numpy(float) - y), L)
                DM.append({"series": s, "loss": "abs", "model": m, "vs": ref, "vs_is_best_component": ref == best, "stat": st, "p": p})
        pls = {}
        for m in prob_all:
            pl = np.mean([pinball(y, E[f"q_{m}_{t}"].to_numpy(float), q) for q, t in zip(QS, QTAG)], axis=0)
            pls[m] = pl
            row = {"series": s, "model": m, "kind": "combination" if m in qsets else "component", "n": len(E),
                   "pinball": pl.mean(), "CRPS_approx": 2 * pl.mean()}
            for lev, (lo_t, hi_t) in INTERVALS.items():
                lo, hi = E[f"q_{m}_{lo_t}"].to_numpy(float), E[f"q_{m}_{hi_t}"].to_numpy(float)
                clo, chi = E[f"cqr{lev}_lo_{m}"].to_numpy(float), E[f"cqr{lev}_hi_{m}"].to_numpy(float)
                h = ~np.isnan(clo)
                row[f"cov{lev}_raw"] = np.mean((y[h] >= lo[h]) & (y[h] <= hi[h])) * 100 if h.any() else np.nan
                row[f"cov{lev}_cqr"] = np.mean((y[h] >= clo[h]) & (y[h] <= chi[h])) * 100 if h.any() else np.nan
                row[f"width{lev}_cqr"] = np.mean(chi[h] - clo[h]) if h.any() else np.nan
            MQ.append(row)
        if pc:
            bestq = min(pc, key=lambda m: pls[m].mean())
            for m in prob:
                for ref in dict.fromkeys([r for r in ["lgbm_q"] if r in pc] + [bestq]):
                    st, p = dm_loss(pls[m] - pls[ref], L)
                    DM.append({"series": s, "loss": "pinball", "model": m, "vs": ref, "vs_is_best_component": ref == bestq,
                               "stat": st, "p": p})
        allE.append(E)

    if not MP:
        print("nothing evaluated"); return
    MP, MY, DM, MQ, W, CP = map(pd.DataFrame, (MP, MY, DM, MQ, W, CP))
    Eall = pd.concat(allE)
    Eall.to_parquet(out / "preds.parquet")
    for n, d in [("metrics_point", MP), ("metrics_by_year", MY), ("dm", DM), ("metrics_prob", MQ), ("weights", W),
                 ("components", CP)]:
        d.to_csv(out / f"{n}.csv", index=False)
    import ml_diagnostics as D
    show = [m for m in METHODS if f"p_{m}" in Eall]
    models = {m: f"p_{m}" for m in show + ["lear", "lgbm_l1"] if f"p_{m}" in Eall} | {b: b for b in C.BASELINES}
    quant = {m: {q: f"q_{m}_{t}" for q, t in zip(QS, QTAG)} for m in ["qavg", "qavg_t1", "lgbm_q"]
             if all(f"q_{m}_{t}" in Eall for t in QTAG)}
    D.forecast_diagnostics(Eall, models, out, baseline="b_prev_slot", quantiles=quant,
                           group_cols=[c for c in Eall.columns if c.startswith("regime_")], title="RQ2 stacking")

    # ---- report
    hdr = ["RQ2 forecast combination / stacking (Tier 1 + Tier 2). Weights only from earlier test folds (ex ante).",
           f"min-hist {a.min_hist} rows (weekly <= 50), ridge alpha {a.ridge_alpha}, component coverage >= {a.coverage:.0%}; "
           "best = best single component on these rows (ex post).", ""]
    for s, g in MP.groupby("series", sort=False):
        g = g.set_index("model")
        cp = CP.set_index("series").loc[s]
        hdr.append(f"== {s}   n = {int(g['n'].iloc[0]):,}   components: {cp['components']}   best single: {BEST[s]}")
        order = [m for m in METHODS if m in g.index] + [m for m in POINT if m in g.index] + C.BASELINES
        T = g.loc[order, ["MAE", "RMSE", "sMAPE_%", "MAE_asinh"]].copy()
        T["skill_vs_prev_slot_%"] = (1 - T["MAE"] / g.at["b_prev_slot", "MAE"]) * 100
        if "lear" in g.index:
            T["skill_vs_LEAR_%"] = (1 - T["MAE"] / g.at["lear", "MAE"]) * 100
        T["skill_vs_best_single_%"] = (1 - T["MAE"] / g.at[BEST[s], "MAE"]) * 100
        hdr.append(C.fmt(T))
        Dd = DM[(DM.series == s) & (DM.loss == "abs")]
        if len(Dd):
            hdr.append("   Diebold-Mariano |error| (stat < 0: combination better) - stat / p:")
            st = Dd.pivot(index="model", columns="vs", values="stat").round(2).astype(str)
            pv = Dd.pivot(index="model", columns="vs", values="p").map(lambda x: f"{x:.3f}")
            hdr.append("   " + (st + " / " + pv).to_string().replace("\n", "\n   "))
        Q = MQ[MQ.series == s].set_index("model")
        if len(Q):
            hdr.append("   Probabilistic (pinball / CRPS EUR/MWh; coverage %):")
            hdr.append("   " + C.fmt(Q.drop(columns=["series", "n", "kind"])).replace("\n", "\n   "))
            Dp = DM[(DM.series == s) & (DM.loss == "pinball")]
            if len(Dp):
                hdr.append("   DM pinball: " + "; ".join(f"{r.model} vs {r.vs} stat {r.stat:.2f} p {r.p:.4f}" for r in Dp.itertuples()))
        Wr = W[(W.series == s) & (W.method == "ridge") & (~W.warmup)]
        if len(Wr):
            hdr.append("   ridge weights (mean over folds): " + "; ".join(
                f"{c} {v:.2f}" for c, v in Wr.groupby("component")["weight"].mean().sort_values(ascending=False).items()))
        Wb = W[(W.series == s) & (W.method == "best_past")]
        if len(Wb):
            hdr.append("   best_past picked: " + ", ".join(f"{c} x{n}" for c, n in Wb["component"].value_counts().items()))
        hdr.append("")
    S = MP.pivot(index="series", columns="model", values="MAE")
    cols = [m for m in METHODS + POINT if m in S.columns]
    hdr.append("== Summary: MAE skill vs same-slot-yesterday (%)")
    hdr.append(C.fmt(pd.DataFrame({m: (1 - S[m] / S["b_prev_slot"]) * 100 for m in cols}, index=S.index)))
    for i_ in S.index:
        mc = [m for m in METHODS if m in S.columns and not np.isnan(S.loc[i_, m])]
        b = S.loc[i_, mc].idxmin()
        RU.summary(f"{i_.replace('|', ' ')}: {b} {(1 - S.loc[i_, b] / S.loc[i_, 'b_prev_slot']) * 100:+.1f}% "
                   f"(best single {BEST[i_]} {(1 - S.loc[i_, BEST[i_]] / S.loc[i_, 'b_prev_slot']) * 100:+.1f}%)")
    hdr.append(f"\ntotal runtime {RU.fmt_duration(time.time() - t0)}")
    (out / "report_rq2_7.txt").write_text("\n".join(hdr))
    print("\n".join(hdr))


if __name__ == "__main__":
    RU.run(main, "RQ2 stacking")
