"""RQ3b step 2 (done_todo 2.3 item 10): does a different recalibration fix the LightGBM risk model?

Collects the rq3b_1_riskmodel.py runs with different --calib / --gbm-init settings and compares them with the
default run (Output/rq3b_1, Platt on the last 6 months) on the SAME test hours. Nothing is refitted.

Runs (--runs name=folder; a run is skipped when its predictions.csv does not exist):
  default         rq3b_1                 Platt on the last 6 months (10 Oct run)
  isotonic        rq3b_1_iso             isotonic on the last 6 months
  platt_oof       rq3b_1_plattoof        model on full window, Platt on out-of-fold predictions (3 x 6 months)
  isotonic_oof    rq3b_1_isooof          model on full window, isotonic on out-of-fold predictions
  raw             rq3b_1_raw             model on full window, no recalibration
  lrinit          rq3b_1_lrinit          LightGBM boosted from the logistic model, Platt on the last 6 months
  lrinit_oof      rq3b_1_lrinit_isooof   LightGBM boosted from the logistic model, isotonic OOF

Per run x side x model (lr_d1, lr_h1, gbm_d1, gbm_pf_d1, gbm_h1, gbm_pf_h1) x period
(all, 2021-22 = crisis years, from_2023, post_cap >= 3 Mar 2025 -- same splits as rq3b_1):
  Brier, Brier skill vs the best baseline of the default run, Murphy decomposition of the Brier score
  (REL = calibration error, RES = resolution, UNC; 10 equal-count bins -> approximate), ECE, O/E, calibration slope,
  log loss, PR-AUC, ROC-AUC, capture rate of the top 5 % riskiest hours.
  Day-block bootstrap (1,000 draws) 95 % CI of the Brier gain of every run over the default run (same model) and
  over the best baseline; 'sig' = CI excludes 0.
  Reliability plots: per side, gbm_d1 / gbm_h1 / lr_h1, all runs on one chart (log axes).

    python Modelling/rq3b_2_calib_compare.py
Output: Modelling/Output/rq3b_2_calib/{compare_metrics.csv, brier_gain_vs_default.csv, summary_skill.csv,
        reliability_bins.csv, plots/*.png, report_rq3b_2.txt, run.log, run_meta.json}
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_utils as RU  # run.log, run_meta.json, run history, macOS notification

HERE = Path(__file__).resolve().parent
OUTROOT = HERE / "Output"
RUNS = {"default": "rq3b_1", "isotonic": "rq3b_1_iso", "platt_oof": "rq3b_1_plattoof",
        "isotonic_oof": "rq3b_1_isooof", "raw": "rq3b_1_raw", "lrinit": "rq3b_1_lrinit",
        "lrinit_oof": "rq3b_1_lrinit_isooof"}
MODELS = ["lr_d1", "lr_h1", "gbm_d1", "gbm_pf_d1", "gbm_h1", "gbm_pf_h1"]
BASELINES = ["clim_exp", "clim_roll", "persist_d1", "persist_h1", "cal_hist_d1"]
CAP = pd.Timestamp("2025-03-03")
lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


def bins_of(p: np.ndarray, k: int = 10) -> np.ndarray:
    return pd.qcut(pd.Series(p).rank(method="first"), k, labels=False).to_numpy()


def murphy(y, p, k=10):
    """Brier = REL - RES + UNC (approximate with equal-count bins of p)."""
    q = bins_of(p, k)
    n = np.bincount(q, minlength=k).astype(float)
    mp = np.bincount(q, weights=p, minlength=k) / np.maximum(n, 1)
    my = np.bincount(q, weights=y, minlength=k) / np.maximum(n, 1)
    yb = y.mean()
    rel = float((n * (mp - my) ** 2).sum() / n.sum())
    res = float((n * (my - yb) ** 2).sum() / n.sum())
    return rel, res, float(yb * (1 - yb)), float((n * np.abs(mp - my)).sum() / n.sum())


def cal_slope(y, p):
    if y.sum() < 5:
        return float("nan")
    from rq3a_2_logit import fit_logit
    z = np.log(p / (1 - p))
    s = z.std() or 1.0
    beta, _ = fit_logit(((z - z.mean()) / s)[:, None], y, 0.0)
    return float(beta[1] / s)


def scores(y, p) -> dict:
    from rq3a_2_logit import auc, avg_precision
    rel, res, unc, ece = murphy(y, p)
    k = max(1, int(round(0.05 * len(y))))
    top = np.argsort(-p)[:k]
    return {"n": len(y), "pos": int(y.sum()), "base_rate": y.mean(), "mean_pred": p.mean(),
            "O_E": y.mean() / max(p.mean(), 1e-12), "Brier": float(((p - y) ** 2).mean()), "REL": rel, "RES": res,
            "UNC": unc, "ECE": ece, "cal_slope": cal_slope(y, p),
            "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "PR_AUC": avg_precision(y, p), "ROC_AUC": auc(y, p), "capture_top5": y[top].sum() / max(y.sum(), 1)}


def periods(d: pd.DataFrame):
    yr = d["date"].dt.year
    return [("all", d), ("2021_22", d[yr <= 2022]), ("from_2023", d[yr >= 2023]), ("post_cap", d[d["date"] >= CAP])]


def boot(diff: np.ndarray, days: np.ndarray, rng, n=1000):
    ud, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(ud))
    c = np.bincount(inv, minlength=len(ud))
    draws = np.empty(n)
    for b in range(n):
        i = rng.integers(0, len(ud), len(ud))
        draws[b] = s[i].sum() / c[i].sum()
    return s.sum() / c.sum(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="name=folder pairs (default: the list above)")
    ap.add_argument("--out", default=str(OUTROOT / "rq3b_2_calib"))
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs) if a.runs else dict(RUNS)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    t0 = time.time()

    P = {}
    for name, folder in runs.items():
        f = OUTROOT / folder / "predictions.csv"
        if not f.exists():
            log(f"  skip {name}: {f} not found")
            continue
        d = pd.read_csv(f, parse_dates=["hk", "date"])
        P[name] = d.set_index(["side", "hk"]).sort_index()
        log(f"  {name:<13} {folder:<24} {len(d):,} rows")
    if "default" not in P:
        raise SystemExit("default run (Output/rq3b_1/predictions.csv) missing")
    D = P["default"]
    key = D.index
    for name, d in P.items():
        key = key.intersection(d.index)
    log(f"common test hours (both sides): {len(key):,}")

    # best baseline per side (default run, pooled, by Brier) -- as in rq3b_1
    best = {}
    for side in D.index.get_level_values(0).unique():
        dd = D.loc[side]
        bl = {b: ((dd[b] - dd["y"]) ** 2).mean() for b in BASELINES if b in dd}
        best[side] = min(bl, key=bl.get)
    log("best baseline: " + ", ".join(f"{s} = {b}" for s, b in best.items()))

    rows, gains, relbins = [], [], []
    rng = np.random.default_rng(20261015)
    for side in best:
        base = D.loc[side].loc[key[key.get_level_values(0) == side].get_level_values(1)].reset_index()
        for name, d in P.items():
            dd = d.loc[side].loc[base["hk"]].reset_index()
            for m in MODELS:
                if m not in dd or dd[m].isna().all():
                    continue
                df = pd.DataFrame({"hk": base["hk"], "date": base["date"], "y": base["y"].astype(float),
                                   "p": dd[m].to_numpy(float), "p_def": base[m].to_numpy(float),
                                   "p_bl": base[best[side]].to_numpy(float)}).dropna()
                for per, x in periods(df):
                    if len(x) == 0 or x["y"].sum() == 0:
                        continue
                    y, p = x["y"].to_numpy(), x["p"].to_numpy()
                    sc = scores(y, p)
                    bl = float(((x["p_bl"] - x["y"]) ** 2).mean())
                    rows.append({"side": side, "run": name, "model": m, "period": per, **sc,
                                 "Brier_skill_vs_best_baseline": 1 - sc["Brier"] / bl})
                    lp = (x["p"] - x["y"]) ** 2
                    for ref, col in (("default", "p_def"), ("best_baseline", "p_bl")):
                        if ref == "default" and name == "default":
                            continue
                        diff = ((x[col] - x["y"]) ** 2 - lp).to_numpy()
                        est, lo, hi = boot(diff, x["date"].to_numpy(), rng)
                        same = np.abs(diff).max() < 1e-12          # identical forecasts (e.g. lr_* under --gbm-init)
                        gains.append({"side": side, "run": name, "model": m, "period": per, "vs": ref,
                                      "dBrier": est, "ci_lo": lo, "ci_hi": hi,
                                      "sig": "" if same else ("better" if lo > 0 else ("worse" if hi < 0 else ""))})
                if m in ("gbm_d1", "gbm_h1", "lr_h1"):
                    q = bins_of(df["p"].to_numpy())
                    t = df.groupby(q).agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
                    relbins.append(t.reset_index(names="decile").assign(side=side, run=name, model=m))
    M = pd.DataFrame(rows)
    G = pd.DataFrame(gains)
    RB = pd.concat(relbins, ignore_index=True)
    M.to_csv(out / "compare_metrics.csv", index=False)
    G.to_csv(out / "brier_gain_vs_default.csv", index=False)
    RB.to_csv(out / "reliability_bins.csv", index=False)

    S = M.pivot_table(index=["side", "model"], columns=["period", "run"], values="Brier_skill_vs_best_baseline")
    S.to_csv(out / "summary_skill.csv")
    order = [r for r in runs if r in P]
    for per in ("all", "2021_22", "from_2023", "post_cap"):
        for val, fmt in (("Brier_skill_vs_best_baseline", "{:+.3f}"), ("ECE", "{:.4f}"), ("cal_slope", "{:.2f}"),
                         ("REL", "{:.5f}"), ("RES", "{:.5f}"), ("PR_AUC", "{:.3f}")):
            t = M[M["period"] == per].pivot_table(index=["side", "model"], columns="run", values=val)
            t = t[[r for r in order if r in t.columns]]
            log(f"\n-- {val}, period {per}")
            log(t.to_string(float_format=lambda v, f=fmt: f.format(v)))
    log("\n-- Brier gain over the default run (same model), day-block bootstrap 95 % CI; period all / from_2023")
    for per in ("all", "from_2023"):
        t = G[(G["vs"] == "default") & (G["period"] == per)].copy()
        t["cell"] = t.apply(lambda r: f"{r.dBrier * 1e4:+.1f}{'*' if r.sig == 'better' else ('-' if r.sig == 'worse' else '')}", axis=1)
        log(f"   period {per} (x 1e-4; * = sig. better, - = sig. worse)")
        log(t.pivot_table(index=["side", "model"], columns="run", values="cell", aggfunc="first").to_string())
    log("\n-- Brier gain over the best baseline, period all (x 1e-4; * = sig. better, - = sig. worse)")
    t = G[(G["vs"] == "best_baseline") & (G["period"] == "all")].copy()
    t["cell"] = t.apply(lambda r: f"{r.dBrier * 1e4:+.1f}{'*' if r.sig == 'better' else ('-' if r.sig == 'worse' else '')}", axis=1)
    log(t.pivot_table(index=["side", "model"], columns="run", values="cell", aggfunc="first").to_string())

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pdir = out / "plots"
        pdir.mkdir(exist_ok=True)
        for (side, m), g in RB.groupby(["side", "model"]):
            fig, ax = plt.subplots(figsize=(5.5, 5))
            lo = max(1e-4, min(g["mean_pred"].min(), g["obs_rate"].replace(0, np.nan).min()) * 0.7)
            ax.plot([lo, 1], [lo, 1], color="0.6", lw=1, ls="--")
            for r in order:
                x = g[g["run"] == r]
                if len(x):
                    ax.plot(x["mean_pred"], x["obs_rate"].clip(lower=lo), marker="o", ms=3, lw=1.2, label=r)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("mean predicted probability (decile)"); ax.set_ylabel("observed spike rate")
            ax.set_title(f"RQ3b reliability: {side} {m}"); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(pdir / f"reliability_{side}_{m}.png", dpi=130); plt.close(fig)
    except Exception as e:                                  # pragma: no cover
        log(f"  plots skipped: {e}")

    for side in best:
        t = M[(M["side"] == side) & (M["period"] == "all") & M["model"].str.startswith("gbm")]
        if len(t):
            r = t.sort_values("Brier").iloc[0]
            d0 = M[(M["side"] == side) & (M["period"] == "all") & (M["model"] == r["model"]) & (M["run"] == "default")]
            RU.summary(f"{side}: best gbm {r['run']}/{r['model']} skill {r['Brier_skill_vs_best_baseline']:+.3f} "
                       f"(default {d0['Brier_skill_vs_best_baseline'].iloc[0]:+.3f})")
    (out / "report_rq3b_2.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    RU.run(main, "RQ3b calibration comparison")
