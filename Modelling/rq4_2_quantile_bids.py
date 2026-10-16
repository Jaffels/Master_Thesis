"""RQ4 extension (done_todo 2.4 items 13-14): bid with QUANTILE forecasts and with the BEST RQ2 forecast per series.

Same market model, rows, revenue rule and offsets as rq4_1_revenue.py (imported, not copied): price-taker with 1 MW,
pay-as-bid, a bid b is accepted if b <= c (c = highest accepted bid of the block), revenue = b when accepted.
Forecasts come from the RQ2 combination run (default Modelling/Output/rq2_7/preds.parquet; same 4h blocks as rq4_1,
checked at start-up). Everything that is chosen (offset k, quantile level tau) is chosen at the start of every local
month from the 90 days before, using only blocks whose result was known (24 h lag) - exactly the rq4_1 rule.

Strategies (rq4_1 strategies are recomputed on the same rows and must reproduce rq4_1 exactly):
  oracle, naive, naive_off, lgbm, lgbm_off       as rq4_1 (lgbm = round-1 LightGBM median; lgbm_off = rq4_1 main)
  item 14 - best point forecast per series:
    invmae, invmae_off                           inverse-MAE combination (RQ2 main combination rule)
    tabpfn, tabpfn_off                           TabPFN v3.5 (aFRR 4h only)
    best_past_off                                model with the lowest past MAE (rq2_7 best_past, chosen ex ante)
    best, best_off                               declared best per series: invmae for mFRR 4h, TabPFN for aFRR 4h
  item 13 - quantile bids (bid = tau-quantile of the price forecast; tau in 5 ... 95 %):
    <m>_qsel   tau re-chosen every month (11 levels, fallback tau = 0.5 while < 150 known blocks)
    <m>_qk     joint choice of tau and k (11 x 12 grid; bid = k x tau-quantile) - robustness (more choices)
    <m>_off    median of the same model x k (isolates "quantile vs offset" on the same forecast)
    for m in: qavg (quantile average, RQ2 main probabilistic combination), lgbmq (LightGBM quantile),
              tabpfnq (TabPFN quantiles, aFRR only), bestq (declared best: qavg for mFRR, TabPFN for aFRR)
  Ex-post diagnostic only (not a strategy): revenue of every FIXED tau over the whole test period
  (quantile_curve.csv) - shows the shape of revenue vs tau; choosing the best tau from it would be look-ahead.

Decision rule (declared before the run): quantile / best-forecast bids replace lgbm_off as the main RQ4 strategy if
they are significantly better than lgbm_off (moving-block bootstrap, 95 % CI) in at least two of the four series and
not significantly worse in any; main candidates = bestq_qsel (item 13) and best_off (item 14). Everything else =
robustness. Caveat: "best per series" is taken from the RQ2 test results (TabPFN / invmae) -> best_past_off is the
fully ex-ante version.

Run from Master_Thesis with .venv active (about 1 minute):
    python Modelling/rq4_2_quantile_bids.py                       # -> Modelling/Output/rq4_2/
    python Modelling/rq4_2_quantile_bids.py --stack rq2_7t --out Modelling/Output/rq4_2t    # tuned + seeded models
Read-only on Clean/Data and Modelling/Output/rq2_* / rq4_1; writes only into its output folder.
Outputs: report_rq4_2.txt, revenue_summary.csv, revenue_by_year.csv, revenue_diff_bootstrap.csv, choices.csv
(chosen k / tau per series x strategy x month), quantile_curve.csv (+ plot), check_vs_rq4_1.csv, blocks.parquet,
diagnostics/ (as rq4_1: cumulative revenue, win rate, bid / clearing ratio, offsets) + plots of the chosen tau.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import run_utils as RU  # run.log, run_meta.json, run history, macOS notification
import rq4_1_revenue as R4

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = R4.ROOT
SERIES = R4.SERIES
TZ = R4.TZ
QTAG = ["05", "10", "20", "30", "40", "50", "60", "70", "80", "90", "95"]
QLEV = np.array([int(t) / 100 for t in QTAG])
BEST_POINT = {"aFRR": "p_tabpfn", "mFRR": "p_invmae"}       # declared best point forecast per product (RQ2)
BEST_Q = {"aFRR": "tabpfn", "mFRR": "qavg"}                  # declared best quantile forecast per product
QMODELS = {"qavg": "qavg", "lgbmq": "lgbm_q", "tabpfnq": "tabpfn", "bestq": None}
BASE_STRATS = ["oracle", "naive", "naive_off", "lgbm", "lgbm_off"]
POINT_STRATS = ["invmae", "invmae_off", "tabpfn", "tabpfn_off", "best_past_off", "best", "best_off"]
Q_STRATS = [f"{m}_{v}" for m in QMODELS for v in ("qsel", "qk", "off")]
STRATS = BASE_STRATS + POINT_STRATS + Q_STRATS
TESTS = [("bestq_qsel", "lgbm_off"), ("best_off", "lgbm_off"), ("qavg_qsel", "lgbm_off"), ("invmae_off", "lgbm_off"),
         ("tabpfn_off", "lgbm_off"), ("best_past_off", "lgbm_off"), ("bestq_qk", "lgbm_off"), ("lgbmq_qsel", "lgbm_off"),
         ("qavg_qsel", "qavg_off"), ("lgbmq_qsel", "lgbmq_off"), ("tabpfnq_qsel", "tabpfnq_off"), ("bestq_qsel", "bestq_off"),
         ("bestq_qsel", "best_off"), ("bestq_qsel", "naive_off"), ("best_off", "naive_off"), ("best", "lgbm")]


def choose_monthly(g: pd.DataFrame, cands: np.ndarray, labels: list, fallback: int, name: str, log: list) -> np.ndarray:
    """Ex-ante choice among candidate bid columns (n x J) for one series g: at the start of each local month pick the
    candidate with the highest mean revenue over the blocks of the 90 days before that were already known (t+24 h <=
    month start). Fewer than R4.MIN_PAST known blocks -> candidate `fallback` (k = 1 / tau = 0.5). Same rule as
    rq4_1.add_offset_bids, vectorised over candidates."""
    R = R4.revenue(np.nan_to_num(cands, nan=-1.0), g["c"].to_numpy()[:, None])      # NaN bid -> never accepted
    okc = ~np.isnan(cands).any(axis=1)
    out = np.full(len(g), np.nan)
    loc = g["start"].dt.tz_convert(TZ)
    months = loc.dt.strftime("%Y-%m").to_numpy()
    t_av = g["start"] + pd.Timedelta("24h")                    # block t is known from t+24h on
    for mth in pd.unique(months):
        ms = pd.Timestamp(mth + "-01", tz=TZ).tz_convert("UTC")
        past = ((t_av <= ms) & (g["start"] >= ms - pd.Timedelta(days=R4.LOOKBACK_D))).to_numpy() & okc
        j = fallback
        if past.sum() >= R4.MIN_PAST:
            j = int(np.argmax(R[past].mean(axis=0)))
        inm = months == mth
        out[inm] = cands[inm, j]
        log.append((g["series"].iloc[0], name, mth, labels[j], int(past.sum())))
    return out


def qcols(d: pd.DataFrame, m: str) -> np.ndarray | None:
    cols = [f"q_{m}_{t}" for t in QTAG]
    return d[cols].to_numpy(float) if all(c in d for c in cols) else None


def add_strategies(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log: list = []
    for s in STRATS:
        if s not in d:
            d[s] = np.nan
    for s, g in d.groupby("series", sort=False):
        ix = g.index
        prod = s.split("|")[0]
        g = g.copy()
        g["best"] = g[BEST_POINT[prod]]
        d.loc[ix, "best"] = g["best"]
        d.loc[ix, "invmae"] = g["p_invmae"]
        d.loc[ix, "tabpfn"] = g["p_tabpfn"]
        k1 = int(np.flatnonzero(np.isclose(R4.K_GRID, 1.0))[0])
        for name, base in [("naive_off", "naive"), ("lgbm_off", "lgbm"), ("invmae_off", "p_invmae"),
                           ("tabpfn_off", "p_tabpfn"), ("best_past_off", "p_best_past"), ("best_off", "best")]:
            b = g[base].to_numpy(float)
            if np.isnan(b).all():
                continue
            d.loc[ix, name] = choose_monthly(g, b[:, None] * R4.K_GRID[None, :], list(R4.K_GRID), k1, name, log)
        for m, src in QMODELS.items():
            Q = qcols(g, src if src else BEST_Q[prod])
            if Q is None or np.isnan(Q).all():
                continue
            Q = np.sort(Q, axis=1)                          # no crossing (already sorted in rq2_5 / rq2_7; safety)
            med = int(np.flatnonzero(np.isclose(QLEV, 0.5))[0])
            d.loc[ix, f"{m}_qsel"] = choose_monthly(g, Q, list(QLEV), med, f"{m}_qsel", log)
            d.loc[ix, f"{m}_off"] = choose_monthly(g, Q[:, [med]] * R4.K_GRID[None, :], list(R4.K_GRID), k1, f"{m}_off", log)
            QK = (Q[:, :, None] * R4.K_GRID[None, None, :]).reshape(len(g), -1)
            lab = [f"{q:.2f}x{k:g}" for q in QLEV for k in R4.K_GRID]
            d.loc[ix, f"{m}_qk"] = choose_monthly(g, QK, lab, med * len(R4.K_GRID) + k1, f"{m}_qk", log)
    ch = pd.DataFrame(log, columns=["series", "strategy", "month", "choice", "n_past"])
    return d, ch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="rq2_7", help="RQ2 combination output folder with preds.parquet (rq2_7 = main, rq2_7t = tuned)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "Modelling" / "Output" / "rq4_2"
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)

    # ---- rows exactly as rq4_1 (same load, same dropna), then merge the RQ2 combination forecasts
    d = R4.load("exante")
    d = d.dropna(subset=["c", "naive", "roll_med", "lgbm"]).reset_index(drop=True)
    d = d.dropna(subset=["fixed_30d"]).reset_index(drop=True)
    P = pd.read_parquet(ROOT / "Modelling" / "Output" / a.stack / "preds.parquet")
    keep = ["series", "block_start_utc", "p_invmae", "p_tabpfn", "p_best_past", "p_avg_all"] + \
           [c for c in P.columns if any(c.startswith(f"q_{m}_") for m in ("qavg", "lgbm_q", "tabpfn")) and not c.startswith("q_qavg_t1")]
    P = P[P["series"].isin(SERIES)][[c for c in keep if c in P.columns]]
    if P.duplicated(["series", "block_start_utc"]).any():
        raise ValueError("duplicate blocks in the combination file")
    n0 = len(d)
    d = d.merge(P, on=["series", "block_start_utc"], how="left", validate="one_to_one")
    miss = d.groupby("series")["p_invmae"].apply(lambda x: int(x.isna().sum()))
    print(f"rows: {n0:,}; blocks without a combination forecast per series: {miss.to_dict()}")
    d = d.dropna(subset=["p_invmae", "q_qavg_50"]).reset_index(drop=True)      # common rows for every strategy
    d, ch = add_strategies(d)
    strats = [s for s in STRATS if d[s].notna().any()]
    for s in strats:
        bid = d[s].to_numpy(float)
        d[f"rev_{s}"] = np.where(np.isnan(bid), np.nan, R4.revenue(np.nan_to_num(bid), d["c"].to_numpy()))
        d[f"win_{s}"] = np.where(np.isnan(bid), np.nan, (np.maximum(bid, 0) <= d["c"] + 1e-9).astype(float))
    d["year"] = d["start"].dt.tz_convert(TZ).dt.year

    lines = [f"RQ4 extension - quantile bids and best forecast per series  (forecasts: {a.stack}; price-taker 1 MW; "
             "pay-as-bid; c = highest accepted bid)",
             f"common rows per series: {d.groupby('series').size().to_dict()}", ""]

    # ---- check: rq4_1 strategies reproduce rq4_1 exactly (same rows -> same revenue)
    chk = []
    r41 = ROOT / "Modelling" / "Output" / "rq4_1" / "revenue_summary.csv"
    if r41.exists():
        S1 = pd.read_csv(r41).set_index(["series", "strategy"])
        for s, g in d.groupby("series"):
            for st in ["oracle", "naive", "naive_off", "lgbm", "lgbm_off"]:
                if (s, st) in S1.index:
                    v1 = S1.loc[(s, st), "rev_per_block"]
                    chk.append(dict(series=s, strategy=st, rq4_1=v1, rq4_2=g[f"rev_{st}"].mean(), n_rq4_1=S1.loc[(s, st), "n"],
                                    n_rq4_2=len(g), abs_diff=abs(v1 - g[f"rev_{st}"].mean())))
        chk = pd.DataFrame(chk)
        chk.to_csv(out / "check_vs_rq4_1.csv", index=False)
        lines += [f"check vs rq4_1 (same rows and revenue for the rq4_1 strategies): max |diff| per block = "
                  f"{chk['abs_diff'].max():.2e}, rows equal: {bool((chk.n_rq4_1 == chk.n_rq4_2).all())}", ""]
        print(lines[-2])

    summ, by_year, diffs, curve = [], [], [], []
    rng = np.random.default_rng(42)
    for s, g in d.groupby("series", sort=False):
        orc = g["rev_oracle"].sum()
        for st in strats:
            if g[st].isna().all():
                continue
            r = g[f"rev_{st}"]
            summ.append(dict(series=s, strategy=st, n=int(r.notna().sum()), rev_per_block=r.mean(),
                             rev_per_mw_year=r.mean() * R4.BLOCKS_PER_YEAR * 4, win_rate=g[f"win_{st}"].mean(),
                             share_of_oracle=r.sum() / orc, value_loss_per_block=g["rev_oracle"].mean() - r.mean(),
                             mae_vs_y=(g[st] - g["y"]).abs().mean(), mae_vs_c=(g[st] - g["c"]).abs().mean(),
                             mean_bid=g[st].mean()))
            for yr, gy in g.groupby("year"):
                by_year.append(dict(series=s, year=yr, strategy=st, n=len(gy), rev_per_block=gy[f"rev_{st}"].mean(),
                                    share_of_oracle=gy[f"rev_{st}"].sum() / gy["rev_oracle"].sum(), win_rate=gy[f"win_{st}"].mean()))
        for st, ref in TESTS:
            if g[st].isna().all() or g[ref].isna().all():
                continue
            x = (g[f"rev_{st}"] - g[f"rev_{ref}"]).to_numpy()
            lo, hi = R4.moving_block_boot(x, rng)
            diffs.append(dict(series=s, strategy=st, vs=ref, mean_diff_per_block=x.mean(),
                              diff_per_mw_year=x.mean() * R4.BLOCKS_PER_YEAR * 4, ci_lo=lo, ci_hi=hi,
                              rel_diff_pct=100 * x.mean() / g[f"rev_{ref}"].mean(), significant=(lo > 0) or (hi < 0)))
        prod = s.split("|")[0]
        for m in ("qavg", "lgbm_q", "tabpfn"):              # ex-post curve: every fixed tau (diagnostic only)
            Q = qcols(g, m)
            if Q is None or np.isnan(Q).all():
                continue
            Q = np.sort(Q, axis=1)
            for k, q in enumerate(QLEV):
                rv = R4.revenue(Q[:, k], g["c"].to_numpy())
                curve.append(dict(series=s, model=m, tau=q, rev_per_block=rv.mean(), share_of_oracle=rv.sum() / orc,
                                  win_rate=(np.maximum(Q[:, k], 0) <= g["c"].to_numpy() + 1e-9).mean(),
                                  declared_best=(m == BEST_Q[prod])))
    summ, by_year, diffs, curve = map(pd.DataFrame, (summ, by_year, diffs, curve))
    summ.to_csv(out / "revenue_summary.csv", index=False)
    by_year.to_csv(out / "revenue_by_year.csv", index=False)
    diffs.to_csv(out / "revenue_diff_bootstrap.csv", index=False)
    curve.to_csv(out / "quantile_curve.csv", index=False)
    ch.to_csv(out / "choices.csv", index=False)
    d[["series", "start", "c", "y"] + strats + [f"rev_{x}" for x in strats]].to_parquet(out / "blocks.parquet")

    # ---- diagnostics (rq4_1 style) + chosen tau over time + revenue vs fixed tau
    import ml_diagnostics as D
    offs = ch[ch.strategy.str.endswith("_off")].rename(columns={"strategy": "base", "choice": "k"})
    offs["k"] = offs["k"].astype(float)
    show = ["oracle", "naive_off", "lgbm_off", "best_off", "qavg_qsel", "bestq_qsel", "bestq_qk"]
    D.revenue_diagnostics(d, strats, out, offsets=offs[offs.base.isin(["lgbm_off", "best_off", "qavg_off", "bestq_off"])],
                          show=[x for x in show if x in strats])
    plt = D._plt()
    for s, gc in ch[ch.strategy.str.endswith("_qsel")].groupby("series"):
        fig, ax = plt.subplots(figsize=(12, 3.5))
        for i, (st, gg) in enumerate(gc.groupby("strategy")):
            ax.step(pd.to_datetime(gg["month"]), gg["choice"].astype(float), where="post", color=D.COLORS[i % 9], label=st)
        ax.set_ylabel("chosen quantile level tau"); ax.set_title(f"{s} - monthly re-chosen bid quantile"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "diagnostics" / "plots" / D._safe(s) / "chosen_tau.png"); plt.close(fig)
    if len(curve):
        fig, axs = plt.subplots(1, len(SERIES), figsize=(16, 3.8))
        for ax, s in zip(axs, SERIES):
            for i, (m, gg) in enumerate(curve[curve.series == s].groupby("model")):
                ax.plot(gg["tau"], gg["share_of_oracle"] * 100, marker="o", ms=3, color=D.COLORS[i], label=m)
            sel = summ[(summ.series == s) & (summ.strategy.isin(["lgbm_off", "bestq_qsel"]))]
            for j, r in enumerate(sel.itertuples()):
                ax.axhline(r.share_of_oracle * 100, ls="--", lw=0.8, color=D.COLORS[5 + j], label=r.strategy)
            ax.set_title(s, fontsize=9); ax.set_xlabel("fixed tau (ex post)")
        axs[0].set_ylabel("% of oracle revenue"); axs[-1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "diagnostics" / "plots" / "quantile_curve.png"); plt.close(fig)

    # ---- report
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", lambda v: f"{v:,.3f}")
    for s in SERIES:
        t = summ[summ.series == s].set_index("strategy")[["rev_per_block", "rev_per_mw_year", "win_rate", "share_of_oracle",
                                                         "mae_vs_y", "mae_vs_c", "mean_bid"]]
        lines += [f"== {s}  (n = {int(summ[summ.series == s].n.max()):,})", t.to_string(), ""]
        dd = diffs[diffs.series == s][["strategy", "vs", "mean_diff_per_block", "diff_per_mw_year", "ci_lo", "ci_hi",
                                       "rel_diff_pct", "significant"]]
        lines += ["paired revenue difference per block (moving-block bootstrap 7 days, 95 % CI):", dd.to_string(index=False), ""]
        cs = ch[(ch.series == s) & ch.strategy.isin(["bestq_qsel", "qavg_qsel", "best_off", "lgbm_off"])]
        lines += ["chosen k / tau (share of months): " + "; ".join(
            f"{st}: " + ", ".join(f"{k} {v:.0%}" for k, v in gg["choice"].value_counts(normalize=True).head(4).items())
            for st, gg in cs.groupby("strategy")), ""]
        cv = curve[curve.series == s].pivot(index="tau", columns="model", values="share_of_oracle")
        lines += ["ex-post revenue share of oracle for every FIXED tau (diagnostic only, not a strategy):", (cv * 100).round(1).to_string(), ""]
    # decision rule
    lines.append("== Decision rule (declared before the run): candidate replaces lgbm_off if significantly better in >= 2 of 4 "
                 "series and not significantly worse in any")
    for cand in ["bestq_qsel", "best_off", "qavg_qsel", "bestq_qk", "best_past_off"]:
        dd = diffs[(diffs.strategy == cand) & (diffs.vs == "lgbm_off")]
        nb = int(((dd.ci_lo > 0)).sum()); nw = int(((dd.ci_hi < 0)).sum())
        verdict = "MEETS the rule" if nb >= 2 and nw == 0 else "does not meet the rule"
        lines.append(f"   {cand:14s} better in {nb}, worse in {nw} of {len(dd)} series -> {verdict}   "
                     + "  ".join(f"{r.series.replace('|', ' ')} {r.rel_diff_pct:+.1f}%{'*' if r.significant else ''}" for r in dd.itertuples()))
    lines.append("")
    lines += ["== share of oracle revenue by year (selected strategies)",
              by_year[by_year.strategy.isin(["naive_off", "lgbm_off", "best_off", "qavg_qsel", "bestq_qsel"])].pivot_table(
                  index=["series", "year"], columns="strategy", values="share_of_oracle").to_string(), ""]
    (out / "report_rq4_2.txt").write_text("\n".join(lines))
    for s in SERIES:
        z = summ[summ.series == s].set_index("strategy")["share_of_oracle"]
        RU.summary(f"{s.replace('|', ' ')}: lgbm_off {z.get('lgbm_off', np.nan):.0%} -> bestq_qsel {z.get('bestq_qsel', np.nan):.0%}, "
                   f"best_off {z.get('best_off', np.nan):.0%}")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    RU.run(main, "RQ4 quantile bids")
