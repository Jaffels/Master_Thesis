"""RQ2 step 8 (done_todo 2.2, item 9): accuracy and Diebold-Mariano tests per regime / sub-period.

The pooled DM tests in rq2_5 / rq2_6 / rq2_7 average over 2021-2026. A model can win on the pooled
sample because of one period (e.g. the 2022 price crisis) and lose elsewhere. This script repeats the
evaluation inside sub-samples. Nothing is refitted: it reads the common-row predictions of rq2_7
(all Tier 1 + Tier 2 point models, the combinations and the quantile average) and splits them.

Groupings (each one is a separate partition of the test rows of a series):
  year      calendar year of delivery (Europe/Zurich)
  period    declared before looking at sub-period results: pre_crisis (to --crisis-start), crisis
            (--crisis-start .. --crisis-end), post_crisis (after) - default Sep 2021 / Jun 2023
  regime_*  every regime flag of the ex-ante view that takes > 1 value inside the series' test rows
            (e.g. regime_mfrr_merged, regime_afrr_daily, regime_imb_pricing, regime_ch_gen_reporting)
  slot      4h series only: delivery slot (00, 04, ..., 20 local)

Tests (absolute-error loss; stat < 0 = row model better). Per group and model:
  vs lear, vs b_prev_slot (naive), vs best_single (the single component with the lowest POOLED MAE,
  as in rq2_7 - fixed per series, not re-chosen per group), and vs invmae (declared main combination).
  Pinball loss (mean over the 11 quantiles): qavg vs lgbm_q and vs the pooled best probabilistic component.
  Newey-West lags as pooled (4h: 12, weekly: 4); slot groups (one row per day) use 7. Groups with < 30
  rows get no test; groups with < 100 rows are flagged `small`.
  Multiple testing: p_holm = Holm adjustment over the levels of one grouping for the same (series,
  model, vs) - i.e. "is the pooled verdict significant in at least one / all sub-periods" is judged
  with family-wise error control per grouping.

Stability (stability.csv, report): for the key pairs
  best_single vs lear | best_single vs naive | invmae vs best_single | qavg vs lgbm_q (pinball)
  every group is classified as  same_sig (same sign as pooled, p < 0.05) / same_ns / opposite_ns /
  REVERSAL (opposite sign, p < 0.05). Plus: winner (lowest MAE single component) per group and the
  Spearman correlation between the group ranking and the pooled ranking of the single components.

    python Modelling/rq2_8_subperiods.py
    python Modelling/rq2_8_subperiods.py --only "mFRR|up|4h" --groupings year period
Output: Modelling/Output/rq2_8/{metrics_by_group.csv, dm_by_group.csv, pinball_by_group.csv,
        stability.csv, winners.csv, table_year.csv, report_rq2_8.txt, plots/, run.log, run_meta.json}
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

SINGLE = ["lear", "lgbm_l1", "lgbm_q", "cat_q", "qrf", "svr", "tft", "nhits", "tabpfn", "chronos"]
COMBOS = ["avg_all", "median_all", "avg_trees", "avg_lear_trees", "best_past", "avg_top3", "invmae", "ridge"]
PROB = ["lgbm_q", "cat_q", "qrf", "tft", "nhits", "tabpfn", "chronos"]
KEY_PAIRS = [("best_single", "lear"), ("best_single", "b_prev_slot"), ("invmae", "best_single")]
ALPHA = 0.05


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values (NaN kept)."""
    p = p.astype(float)
    ok = p.dropna().sort_values()
    m = len(ok)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * ok.to_numpy()))
    out = pd.Series(np.nan, index=p.index)
    out.loc[ok.index] = adj
    return out


def groupings(E: pd.DataFrame, s: str, a) -> dict[str, pd.Series]:
    loc = E["block_start_utc"].dt.tz_convert(C.TZ)
    g = {}
    if "year" in a.groupings:
        g["year"] = loc.dt.year.astype(str)
    if "period" in a.groupings:
        cs, ce = pd.Timestamp(a.crisis_start, tz=C.TZ), pd.Timestamp(a.crisis_end, tz=C.TZ)
        g["period"] = pd.Series(np.where(loc < cs, "1_pre_crisis", np.where(loc <= ce, "2_crisis", "3_post_crisis")),
                                index=E.index)
    if "regime" in a.groupings:
        for c in [c for c in E.columns if c.startswith("regime_")]:
            if E[c].nunique(dropna=True) < 2:
                continue
            lab = E[c].astype(str)
            dup = next((k for k, v in g.items() if k.startswith("regime_") and _same_partition(v, lab)), None)
            if dup:                                     # e.g. aFRR daily auction and mFRR merger (1 day apart)
                g[f"{dup}={c}"] = g.pop(dup)
            else:
                g[c] = lab
    if "slot" in a.groupings and s.endswith("|4h"):
        g["slot"] = E["slot"].astype(int).map(lambda h: f"{h:02d}")
    return {k: v for k, v in g.items() if v.nunique() > 1}


def _same_partition(a_: pd.Series, b_: pd.Series, tol: float = 0.99) -> bool:
    """True if two labelings split the rows (almost) identically (>= tol of rows agree under the best label map)."""
    if a_.nunique() != b_.nunique():
        return False
    ct = pd.crosstab(a_, b_)
    return ct.max(axis=1).sum() >= tol * len(a_) and ct.max(axis=0).sum() >= tol * len(a_)


def classify(stat_pool: float, stat: float, p: float) -> str:
    if np.isnan(stat) or np.isnan(p) or np.isnan(stat_pool):
        return "no_test"
    same = np.sign(stat) == np.sign(stat_pool)
    if p < ALPHA:
        return "same_sig" if same else "REVERSAL"
    return "same_ns" if same else "opposite_ns"


def spearman(a_: pd.Series, b_: pd.Series) -> float:
    d = pd.concat([a_, b_], axis=1).dropna()
    if len(d) < 3:
        return np.nan
    r = d.rank()
    return float(np.corrcoef(r.iloc[:, 0], r.iloc[:, 1])[0, 1])


def run_series(E: pd.DataFrame, s: str, a):
    E = E.sort_values("block_start_utc").reset_index(drop=True)
    y = E["y"].to_numpy(float)
    single = [m for m in SINGLE if f"p_{m}" in E and E[f"p_{m}"].notna().all()]
    combos = [m for m in COMBOS if f"p_{m}" in E and E[f"p_{m}"].notna().all()]
    col = {**{m: f"p_{m}" for m in single + combos}, **{b: b for b in C.BASELINES}}
    err = {m: np.abs(E[c].to_numpy(float) - y) for m, c in col.items()}
    pooled_mae = pd.Series({m: e.mean() for m, e in err.items()})
    best = pooled_mae[single].idxmin()
    err["best_single"] = err[best]
    prob = [m for m in PROB if all(f"q_{m}_{t}" in E and E[f"q_{m}_{t}"].notna().all() for t in QTAG)]
    prob += [m for m in ["qavg"] if all(f"q_{m}_{t}" in E and E[f"q_{m}_{t}"].notna().all() for t in QTAG)]
    pl = {m: np.mean([pinball(y, E[f"q_{m}_{t}"].to_numpy(float), q) for q, t in zip(QS, QTAG)], axis=0) for m in prob}
    best_q = min([m for m in prob if m != "qavg"], key=lambda m: pl[m].mean()) if len(prob) > 1 else None
    L = 4 if s.endswith("|week") else 12
    refs = [r for r in ["lear", "b_prev_slot", "best_single", "invmae"] if r in err]
    models = single + combos + C.BASELINES

    def dm_rows(idx, lag, gname, level):
        rows = []
        for m in models:
            for r in refs:
                if r == m or (r == "best_single" and m == best):
                    continue
                st, p = dm_loss(err[m][idx] - err[r][idx], lag)
                rows.append({"series": s, "grouping": gname, "group": level, "n": int(idx.sum()), "loss": "abs",
                             "model": m, "vs": r, "mae_model": err[m][idx].mean(), "mae_vs": err[r][idx].mean(),
                             "stat": st, "p": p})
        if "qavg" in pl:
            for r in dict.fromkeys([x for x in ["lgbm_q", best_q] if x and x in pl]):
                st, p = dm_loss(pl["qavg"][idx] - pl[r][idx], lag)
                rows.append({"series": s, "grouping": gname, "group": level, "n": int(idx.sum()), "loss": "pinball",
                             "model": "qavg", "vs": r, "mae_model": pl["qavg"][idx].mean(), "mae_vs": pl[r][idx].mean(),
                             "stat": st, "p": p})
        return rows

    MG, DMr, PG = [], [], []
    allidx = np.ones(len(E), bool)
    DMr += dm_rows(allidx, L, "pooled", "all")
    for gname, lab in {"pooled": pd.Series("all", index=E.index), **groupings(E, s, a)}.items():
        lag = 7 if gname == "slot" else L
        for level in sorted(lab.dropna().unique()):
            idx = (lab == level).to_numpy()
            n = int(idx.sum())
            nb = err["b_prev_slot"][idx].mean()
            for m in models + ["best_single"]:
                if m == "best_single":
                    continue
                MG.append({"series": s, "grouping": gname, "group": level, "n": n, "model": m,
                           "kind": "single" if m in single else ("combination" if m in combos else "baseline"),
                           **C.metrics(y[idx], E[col[m]].to_numpy(float)[idx]),
                           "skill_vs_naive_%": (1 - err[m][idx].mean() / nb) * 100 if nb > 0 else np.nan,
                           "first_ts": E["block_start_utc"][idx].min(), "last_ts": E["block_start_utc"][idx].max()})
            for m in pl:
                PG.append({"series": s, "grouping": gname, "group": level, "n": n, "model": m, "pinball": pl[m][idx].mean()})
            if gname != "pooled":
                DMr += dm_rows(idx, lag, gname, level)
    DMd = pd.DataFrame(DMr)
    DMd["small"] = DMd["n"] < 100
    DMd["p_holm"] = np.nan
    sub = DMd["grouping"] != "pooled"
    DMd.loc[sub, "p_holm"] = DMd[sub].groupby(["grouping", "loss", "model", "vs"], group_keys=False)["p"].apply(holm)
    DMd.loc[~sub, "p_holm"] = DMd.loc[~sub, "p"]
    meta = {"series": s, "best_single": best, "best_prob": best_q, "n": len(E), "single": " ".join(single),
            "combos": " ".join(combos)}
    return pd.DataFrame(MG), DMd, pd.DataFrame(PG), meta


def stability(DM: pd.DataFrame, MG: pd.DataFrame, META: pd.DataFrame):
    rows, wins = [], []
    pairs = [("abs", m, v) for m, v in KEY_PAIRS] + [("pinball", "qavg", "lgbm_q")]
    bmap = META.set_index("series")["best_single"]
    for s, D in DM.groupby("series", sort=False):
        for loss, m0, v in pairs:
            m = bmap[s] if m0 == "best_single" else m0
            if m == v:
                continue
            d = D[(D.loss == loss) & (D.model == m) & (D.vs == v)]
            pool = d[d.grouping == "pooled"]
            if pool.empty:
                continue
            sp, pp = float(pool["stat"].iloc[0]), float(pool["p"].iloc[0])
            for gname, dg in d[d.grouping != "pooled"].groupby("grouping", sort=False):
                cls = [classify(sp, r.stat, r.p) for r in dg.itertuples()]
                cls_h = [classify(sp, r.stat, r.p_holm) for r in dg.itertuples()]
                rows.append({"series": s, "loss": loss, "pair": f"{m0} vs {v}", "model": m, "grouping": gname,
                             "pooled_stat": sp, "pooled_p": pp, "n_groups": len(dg),
                             **{k: cls.count(k) for k in ["same_sig", "same_ns", "opposite_ns", "REVERSAL", "no_test"]},
                             "REVERSAL_holm": cls_h.count("REVERSAL"), "same_sig_holm": cls_h.count("same_sig"),
                             "groups_reversal": ", ".join(str(r.group) for r, c in zip(dg.itertuples(), cls) if c == "REVERSAL"),
                             "groups_same_sig": ", ".join(str(r.group) for r, c in zip(dg.itertuples(), cls) if c == "same_sig")})
    for (s, gname, level), g in MG[MG.kind == "single"].groupby(["series", "grouping", "group"], sort=False):
        pooled = MG[(MG.series == s) & (MG.grouping == "pooled") & (MG.kind == "single")].set_index("model")["MAE"]
        gm = g.set_index("model")["MAE"]
        best = META.set_index("series").at[s, "best_single"]
        allg = MG[(MG.series == s) & (MG.grouping == gname) & (MG.group == level)].set_index("model")
        wins.append({"series": s, "grouping": gname, "group": level, "n": int(g["n"].iloc[0]),
                     "winner_single": gm.idxmin(), "pooled_best": best, "pooled_best_rank": int(gm.rank().get(best, np.nan)),
                     "n_single": len(gm), "spearman_vs_pooled": spearman(gm, pooled),
                     "winner_all": allg["MAE"].idxmin(),
                     "skill_best_single_%": allg.at[best, "skill_vs_naive_%"],
                     "skill_invmae_%": allg["skill_vs_naive_%"].get("invmae", np.nan),
                     "skill_lear_%": allg["skill_vs_naive_%"].get("lear", np.nan)})
    return pd.DataFrame(rows), pd.DataFrame(wins)


def plot_heat(MG: pd.DataFrame, DM: pd.DataFrame, s: str, gname: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = MG[(MG.series == s) & (MG.grouping.isin(["pooled", gname])) & (MG.model != "b_prev_slot")]
    g = g[~g.model.isin(["b_prev_auction", "b_roll_med"])]
    T = g.pivot_table(index="model", columns="group", values="skill_vs_naive_%")
    cols = [c for c in T.columns if c != "all"] + (["all"] if "all" in T.columns else [])
    order = [m for m in SINGLE + COMBOS if m in T.index]
    T = T.loc[order, cols]
    d = DM[(DM.series == s) & (DM.vs == "b_prev_slot") & (DM.loss == "abs") & (DM.grouping.isin(["pooled", gname]))]
    Pv = d.pivot_table(index="model", columns="group", values="p").reindex(index=order, columns=cols)
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(cols), 0.9 + 0.32 * len(order)))
    lim = max(5.0, np.nanpercentile(np.abs(T.to_numpy()), 95))
    im = ax.imshow(T.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            v, p = T.iat[i, j], Pv.iat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0f}{'*' if (not np.isnan(p) and p < ALPHA) else ''}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(cols)), [str(c).replace("_", " ") for c in cols], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.grid(False)
    ax.set_title(f"{s}: MAE skill vs same slot yesterday (%) by {gname}\n* = DM vs naive p < 0.05", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    (out / "plots").mkdir(exist_ok=True)
    fig.savefig(out / "plots" / f"skill_{s.replace('|', '_')}_{gname}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default=None, help="default Output/rq2_7/preds.parquet")
    ap.add_argument("--only", nargs="+", default=None, help="series filter(s)")
    ap.add_argument("--groupings", nargs="+", default=["year", "period", "regime", "slot"],
                    choices=["year", "period", "regime", "slot"])
    ap.add_argument("--crisis-start", default="2021-09-01")
    ap.add_argument("--crisis-end", default="2023-06-30")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else C.ROOT / "Modelling" / "Output" / "rq2_8"
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    t0 = time.time()
    src = Path(a.preds) if a.preds else C.ROOT / "Modelling" / "Output" / "rq2_7" / "preds.parquet"
    if not src.exists():
        raise SystemExit(f"{src} not found - run rq2_7_stack.py first")
    P = pd.read_parquet(src)
    print(f"read {src} ({len(P):,} rows, {P['series'].nunique()} series)")
    MG, DM, PG, META = [], [], [], []
    for s, E in P.groupby("series", sort=False):
        if a.only and s not in a.only:
            continue
        mg, dm, pg, meta = run_series(E, s, a)
        MG.append(mg); DM.append(dm); PG.append(pg); META.append(meta)
        print(f"   {s:16s} n={meta['n']:6,}  best single {meta['best_single']}  groupings "
              f"{sorted(set(mg.grouping) - {'pooled'})}")
    MG, DM, PG, META = pd.concat(MG), pd.concat(DM), pd.concat(PG), pd.DataFrame(META)
    ST, WN = stability(DM, MG, META)
    MG.to_csv(out / "metrics_by_group.csv", index=False)
    DM.to_csv(out / "dm_by_group.csv", index=False)
    PG.to_csv(out / "pinball_by_group.csv", index=False)
    ST.to_csv(out / "stability.csv", index=False)
    WN.to_csv(out / "winners.csv", index=False)
    META.to_csv(out / "series_meta.csv", index=False)

    # thesis table: per series x year - skill of LEAR / best single / invmae vs naive, DM p best single vs LEAR
    ty = WN[WN.grouping == "year"][["series", "group", "n", "skill_lear_%", "skill_best_single_%", "skill_invmae_%",
                                     "winner_single"]].rename(columns={"group": "year"})
    d = DM[(DM.grouping == "year") & (DM.loss == "abs") & (DM.vs == "lear")]
    bs = META.set_index("series")["best_single"]
    d = d[d.apply(lambda r: r.model == bs.get(r.series), axis=1)][["series", "group", "stat", "p", "p_holm"]]
    ty = ty.merge(d.rename(columns={"group": "year", "stat": "dm_best_vs_lear", "p": "p_best_vs_lear",
                                    "p_holm": "p_holm_best_vs_lear"}), on=["series", "year"], how="left")
    ty.to_csv(out / "table_year.csv", index=False)

    for s in META["series"]:
        for gname in sorted(set(MG[MG.series == s].grouping) - {"pooled"}):
            try:
                plot_heat(MG, DM, s, gname, out)
            except Exception as e:  # plotting never stops the run
                print(f"   plot {s} {gname} failed: {e}")

    # ---------------------------------------------------------------- report
    R = ["RQ2 accuracy and Diebold-Mariano tests per regime / sub-period (no refitting; rows = rq2_7 common rows).",
         f"crisis period {a.crisis_start} .. {a.crisis_end}; DM |error| (stat < 0: model better); "
         "Holm over the groups of one grouping; best_single = lowest pooled MAE among single models.", ""]
    for s in META["series"]:
        m = META.set_index("series").loc[s]
        R.append(f"== {s}   n = {int(m['n']):,}   best single (pooled): {m['best_single']}   best prob.: {m['best_prob']}")
        for gname in ["year", "period"] + sorted(g for g in set(MG[MG.series == s].grouping) if g.startswith("regime_")) + ["slot"]:
            g = MG[(MG.series == s) & (MG.grouping == gname)]
            if g.empty:
                continue
            show = [x for x in ["lear", m["best_single"], "invmae", "avg_all", "qrf", "cat_q", "lgbm_q", "tabpfn", "chronos"]
                    if x in set(g.model)]
            T = g.pivot_table(index="group", columns="model", values="skill_vs_naive_%")[list(dict.fromkeys(show))]
            T.insert(0, "n", g.groupby("group")["n"].first())
            d = DM[(DM.series == s) & (DM.grouping == gname) & (DM.loss == "abs")]
            for (mm, vv, nm) in [(m["best_single"], "lear", "p_best_vs_LEAR"), ("invmae", "best_single", "p_invmae_vs_best"),
                                 (m["best_single"], "b_prev_slot", "p_best_vs_naive")]:
                x = d[(d.model == mm) & (d.vs == vv)].set_index("group")
                if len(x):
                    T[nm] = (x["stat"].map(lambda v: "+" if v > 0 else "-") + x["p"].map(lambda v: f"{v:.3f}"))
            dq = DM[(DM.series == s) & (DM.grouping == gname) & (DM.loss == "pinball") & (DM.vs == "lgbm_q")].set_index("group")
            if len(dq):
                T["p_qavg_vs_lgbmq_pinball"] = dq["stat"].map(lambda v: "+" if v > 0 else "-") + dq["p"].map(lambda v: f"{v:.3f}")
            R.append(f"   by {gname}: MAE skill vs naive (%); p-values with sign of the DM stat (- = first model better)")
            R.append("   " + T.to_string(float_format=lambda v: f"{v:,.1f}").replace("\n", "\n   "))
        w = WN[(WN.series == s) & (WN.grouping == "year")]
        if len(w):
            R.append("   winner (single) by year: " + "; ".join(f"{r.group} {r.winner_single} (pooled best rank {r.pooled_best_rank}/"
                                                             f"{r.n_single}, rho {r.spearman_vs_pooled:.2f})" for r in w.itertuples()))
        R.append("")
    R.append("== Stability of the pooled verdicts (counts of groups; REVERSAL = opposite sign and p < 0.05)")
    if len(ST):
        R.append(ST[["series", "pair", "grouping", "pooled_stat", "pooled_p", "n_groups", "same_sig", "same_ns", "opposite_ns",
                     "REVERSAL", "REVERSAL_holm", "groups_reversal"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        rv = ST[ST["REVERSAL"] > 0]
        R.append(f"\n   reversals (unadjusted): {len(rv)} of {len(ST)} series x pair x grouping; "
                 f"after Holm: {(ST['REVERSAL_holm'] > 0).sum()}")
        for r in rv.itertuples():
            RU.summary(f"reversal {r.series} {r.pair} ({r.grouping}: {r.groups_reversal})")
    if not len(ST) or (ST["REVERSAL"] == 0).all():
        RU.summary("no significant reversal of any pooled verdict")
    R.append("\nFiles: metrics_by_group.csv (all metrics per group), dm_by_group.csv (all DM tests, p and p_holm), "
             "pinball_by_group.csv, stability.csv, winners.csv, table_year.csv (thesis table), plots/skill_<series>_<grouping>.png")
    R.append(f"\ntotal runtime {RU.fmt_duration(time.time() - t0)}")
    (out / "report_rq2_8.txt").write_text("\n".join(R))
    print("\n".join(R))


if __name__ == "__main__":
    RU.run(main, "RQ2 sub-period DM")
