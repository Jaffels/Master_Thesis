"""Shared analysis outputs for the modelling scripts: tables + figures that show WHAT the models did,
not only how accurate they were on average. Plotting never stops a run (errors are printed and skipped).

forecast_diagnostics(...)    point / quantile price forecasts (RQ2 scripts, RQ4 inputs)
classifier_diagnostics(...)  probability forecasts of rare events (RQ3a / RQ3b)
revenue_diagnostics(...)     bidding simulation (RQ4)
importance_table(...)        mean / spread of per-fold feature importances

Every function writes into <out>/diagnostics/ (tables) and <out>/diagnostics/plots/ (PNG), plus a
README.txt that explains each file.
"""
from __future__ import annotations

import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"
COLORS = ["#2f6db3", "#d1495b", "#3a9b5d", "#e09f3e", "#7b5ea7", "#3aa6b9", "#8c6d46", "#c2549d", "#555555"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "font.size": 9})
    return plt


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def _try_plot(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except Exception as e:                           # noqa: BLE001
        print(f"   [plot skipped: {fn.__name__}: {type(e).__name__}: {e}]")
        traceback.print_exc(limit=1)


def _local(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    return ts.dt.tz_convert(TZ) if ts.dt.tz is not None else ts


def _write_readme(d: Path, text: str) -> None:
    f = d / "README.txt"
    old = f.read_text(encoding="utf-8") if f.exists() else ""
    if text.strip() not in old:
        f.write_text((old + "\n" + text).strip() + "\n", encoding="utf-8")


def importance_table(per_fold: dict[str, list[pd.Series]]) -> pd.DataFrame:
    """per_fold[series] = list of Series (feature -> importance share) -> long table with mean, std,
    min, max over folds and the share of folds in which the feature is in the top 20."""
    rows = []
    for s, lst in per_fold.items():
        if not lst:
            continue
        M = pd.concat(lst, axis=1).fillna(0.0)
        top = pd.concat([(x.rank(ascending=False) <= 20) for x in lst], axis=1).reindex(M.index).fillna(False)
        t = pd.DataFrame({"series": s, "feature": M.index, "mean": M.mean(axis=1), "std": M.std(axis=1),
                          "min": M.min(axis=1), "max": M.max(axis=1), "n_folds": M.shape[1],
                          "top20_share_of_folds": top.mean(axis=1)})
        rows.append(t.sort_values("mean", ascending=False))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ================================================================ forecasts (regression)
FORECAST_README = """FORECAST DIAGNOSTICS (ml_diagnostics.forecast_diagnostics)
Errors are prediction - actual (positive = over-forecast), EUR/MWh, test period only.
  summary_by_model.csv     per series x model: n, MAE, RMSE, bias, median error, error quantiles, share over-forecast,
                           R2 and correlation with the actual, share of total |error| caused by the worst 1 % / 5 % of blocks
  fold_metrics.csv         per series x refit fold x model: test window, n_train, MAE, bias, skill vs the baseline
                           -> does accuracy drift with time since the last refit or across regimes?
  error_breakdown.csv      MAE / bias per year, quarter, delivery slot, weekday and every regime_* column present
  worst_errors.csv         the 25 largest errors per series (best model), with every model's forecast next to them
  quantile_calibration.csv observed share of actuals below each predicted quantile (should equal the level), per model
                           and year; plus pinball loss per level
  importance_*.csv         feature importance per model (mean / std over folds, how often in the top 20)
  plots/<series>/          timeseries.png (daily mean actual vs forecasts), zoom_last_8w.png (block level, with the
                           80 % / 90 % interval of the first quantile model), rolling_mae.png (90-day rolling MAE),
                           mae_by_year.png, mae_by_slot.png (4h only), scatter.png (forecast vs actual, asinh axes),
                           error_hist.png, quantile_calibration.png, importance_<model>.png
"""


def forecast_diagnostics(P: pd.DataFrame, models: dict[str, str], out: Path, baseline: str | None = "b_prev_slot",
                         quantiles: dict[str, dict[float, str]] | None = None,
                         importance: dict[str, pd.DataFrame] | None = None,
                         group_cols: list[str] | None = None, title: str = "", subdir: str = "diagnostics") -> None:
    """P: one row per test block with columns series, block_start_utc, y and the forecast columns.
    models: display name -> column in P (baselines included). baseline: display name used for skill.
    quantiles: model -> {level: column}. importance: model -> long table from importance_table().
    group_cols: extra columns in P for the error breakdown (e.g. regime_*)."""
    d = Path(out) / subdir; pdir = d / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    _write_readme(d, FORECAST_README)
    P = P.copy()
    P["_t"] = _local(P["block_start_utc"])
    P["year"] = P["_t"].dt.year; P["quarter"] = P["_t"].dt.year.astype(str) + "Q" + P["_t"].dt.quarter.astype(str)
    P["weekday"] = P["_t"].dt.dayofweek
    if "slot" not in P:
        P["slot"] = np.where(P["series"].str.endswith("|4h"), P["_t"].dt.hour, 0)
    groups = ["year", "quarter", "slot", "weekday"] + [c for c in (group_cols or []) if c in P.columns]
    cols = list(models.values())
    summ, folds, brk, worst, qcal = [], [], [], [], []
    for s, g in P.groupby("series", sort=False):
        g = g.dropna(subset=["y", *cols]).sort_values("block_start_utc")
        if len(g) == 0:
            continue
        y = g["y"].to_numpy(float)
        for name, c in models.items():
            p = g[c].to_numpy(float); e = p - y; ae = np.abs(e)
            srt = np.sort(ae)[::-1]
            summ.append({"series": s, "model": name, "n": len(g), "MAE": ae.mean(), "RMSE": np.sqrt((e ** 2).mean()),
                         "bias": e.mean(), "median_error": np.median(e), "err_p05": np.quantile(e, .05),
                         "err_p25": np.quantile(e, .25), "err_p75": np.quantile(e, .75), "err_p95": np.quantile(e, .95),
                         "share_over_forecast": (e > 0).mean(), "max_abs_error": ae.max(),
                         "R2": 1 - (e ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12),
                         "corr_pred_actual": np.corrcoef(p, y)[0, 1] if np.std(p) > 0 else np.nan,
                         "abs_error_share_worst1pct": srt[:max(1, len(srt) // 100)].sum() / max(srt.sum(), 1e-12),
                         "abs_error_share_worst5pct": srt[:max(1, len(srt) // 20)].sum() / max(srt.sum(), 1e-12)})
        base_c = models.get(baseline) if baseline else None
        if "fold" in g:
            for f, gf in g.groupby("fold"):
                yb = gf["y"].to_numpy(float)
                mb = np.abs(gf[base_c].to_numpy(float) - yb).mean() if base_c else np.nan
                for name, c in models.items():
                    e = gf[c].to_numpy(float) - yb
                    folds.append({"series": s, "fold": f, "test_from": gf["_t"].min().date(), "test_to": gf["_t"].max().date(),
                                  "n_test": len(gf), "n_train": gf["n_train"].iloc[0] if "n_train" in gf else np.nan,
                                  "model": name, "MAE": np.abs(e).mean(), "bias": e.mean(),
                                  "skill_vs_baseline_%": (1 - np.abs(e).mean() / mb) * 100 if base_c and mb > 0 else np.nan})
        for gc in groups:
            for v, gv in g.groupby(gc):
                yv = gv["y"].to_numpy(float)
                for name, c in models.items():
                    e = gv[c].to_numpy(float) - yv
                    brk.append({"series": s, "dimension": gc, "value": v, "model": name, "n": len(gv),
                                "MAE": np.abs(e).mean(), "bias": e.mean(), "MedAE": np.median(np.abs(e))})
        best = min(models, key=lambda m: np.abs(g[models[m]] - g["y"]).mean() if m != baseline else np.inf)
        w = g.assign(abs_error_best=(g[models[best]] - g["y"]).abs()).nlargest(25, "abs_error_best")
        w = w[["series", "block_start_utc", "y", *cols, "abs_error_best"]].rename(columns={v: k for k, v in models.items()})
        w.insert(2, "best_model", best); worst.append(w)
        for m, qd in (quantiles or {}).items():
            for lvl, qc in sorted(qd.items()):
                if qc not in g:
                    continue
                q = g[qc].to_numpy(float)
                pin = np.maximum(lvl * (y - q), (lvl - 1) * (y - q)).mean()
                qcal.append({"series": s, "model": m, "year": "all", "level": lvl, "observed_below": (y <= q).mean(),
                             "pinball": pin, "n": len(g)})
                for yr, gy in g.groupby("year"):
                    qcal.append({"series": s, "model": m, "year": yr, "level": lvl,
                                 "observed_below": (gy["y"].to_numpy(float) <= gy[qc].to_numpy(float)).mean(),
                                 "pinball": np.nan, "n": len(gy)})
        sp = pdir / _safe(s); sp.mkdir(parents=True, exist_ok=True)
        _try_plot(_plot_timeseries, g, models, sp / "timeseries.png", f"{title} {s}")
        _try_plot(_plot_zoom, g, models, quantiles or {}, sp / "zoom_last_8w.png", f"{title} {s}")
        _try_plot(_plot_rolling_mae, g, models, sp / "rolling_mae.png", f"{title} {s}")
        _try_plot(_plot_bar_group, g, models, "year", sp / "mae_by_year.png", f"{title} {s}: MAE by year")
        if s.endswith("|4h"):
            _try_plot(_plot_bar_group, g, models, "slot", sp / "mae_by_slot.png", f"{title} {s}: MAE by delivery slot (local start hour)")
        _try_plot(_plot_scatter, g, models, best, sp / "scatter.png", f"{title} {s}")
        _try_plot(_plot_error_hist, g, models, sp / "error_hist.png", f"{title} {s}")
    pd.DataFrame(summ).to_csv(d / "summary_by_model.csv", index=False)
    if folds:
        pd.DataFrame(folds).to_csv(d / "fold_metrics.csv", index=False)
    pd.DataFrame(brk).to_csv(d / "error_breakdown.csv", index=False)
    if worst:
        pd.concat(worst).to_csv(d / "worst_errors.csv", index=False)
    if qcal:
        Q = pd.DataFrame(qcal); Q.to_csv(d / "quantile_calibration.csv", index=False)
        for s, gq in Q[Q.year == "all"].groupby("series"):
            _try_plot(_plot_qcal, gq, pdir / _safe(s) / "quantile_calibration.png", f"{title} {s}")
    for m, I in (importance or {}).items():
        if I is None or len(I) == 0:
            continue
        I.to_csv(d / f"importance_{_safe(m)}.csv", index=False)
        for s, gi in I.groupby("series"):
            _try_plot(_plot_importance, gi, pdir / _safe(s) / f"importance_{_safe(m)}.png", f"{title} {s}: {m}")
    print(f"   diagnostics written: {d}")


def _sel_models(models: dict, k: int = 5) -> dict:
    return dict(list(models.items())[:k])


def _plot_timeseries(g, models, f, title):
    plt = _plt()
    daily = g.set_index("_t")[["y", *models.values()]].resample("W" if "|week" in title else "D").mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["y"], color="black", lw=1.2, label="actual")
    for i, (n, c) in enumerate(_sel_models(models).items()):
        ax.plot(daily.index, daily[c], lw=0.8, alpha=0.85, color=COLORS[i % len(COLORS)], label=n)
    ax.set_title(title + " - actual vs forecasts (daily mean; weekly series: weekly)")
    ax.set_ylabel("EUR/MWh"); ax.legend(ncol=6, fontsize=8)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_zoom(g, models, quantiles, f, title):
    plt = _plt()
    z = g[g["_t"] >= g["_t"].max() - pd.Timedelta(weeks=8 if "|4h" in title else 52)]
    fig, ax = plt.subplots(figsize=(12, 4))
    for m, qd in quantiles.items():
        lv = sorted(qd)
        for lo, hi, a in ((lv[0], lv[-1], 0.15), (min(lv, key=lambda x: abs(x - .1)), min(lv, key=lambda x: abs(x - .9)), 0.25)):
            if qd[lo] in z and qd[hi] in z:
                ax.fill_between(z["_t"], z[qd[lo]], z[qd[hi]], color=COLORS[0], alpha=a, lw=0,
                                label=f"{m} {int(lo*100)}-{int(hi*100)} %")
        break
    ax.plot(z["_t"], z["y"], color="black", lw=1.1, marker=".", ms=2, label="actual")
    for i, (n, c) in enumerate(_sel_models(models).items()):
        ax.plot(z["_t"], z[c], lw=0.8, color=COLORS[(i + 1) % len(COLORS)], label=n)
    ax.set_title(title + " - last 8 weeks (weekly series: last year), block level")
    ax.set_ylabel("EUR/MWh"); ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_rolling_mae(g, models, f, title):
    plt = _plt()
    s = g.set_index("_t")
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (n, c) in enumerate(models.items()):
        r = (s[c] - s["y"]).abs().rolling("90D", min_periods=10).mean()
        ax.plot(r.index, r, lw=1, color=COLORS[i % len(COLORS)], label=n)
    ax.set_title(title + " - rolling 90-day MAE"); ax.set_ylabel("EUR/MWh"); ax.legend(ncol=6, fontsize=8)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_bar_group(g, models, col, f, title):
    plt = _plt()
    t = pd.DataFrame({n: (g[c] - g["y"]).abs().groupby(g[col]).mean() for n, c in models.items()})
    ax = t.plot(kind="bar", figsize=(max(6, 1.2 * len(t)), 4), color=COLORS[:t.shape[1]], width=0.8)
    ax.set_title(title); ax.set_ylabel("MAE, EUR/MWh"); ax.set_xlabel(col); ax.legend(ncol=4, fontsize=8)
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_scatter(g, models, best, f, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    y, p = np.arcsinh(g["y"]), np.arcsinh(g[models[best]])
    ax.scatter(y, p, s=4, alpha=0.3, color=COLORS[0])
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
    ax.set_xlabel("actual, asinh(EUR/MWh)"); ax.set_ylabel(f"{best}, asinh(EUR/MWh)")
    ax.set_title(title + f"\nforecast vs actual ({best})")
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_error_hist(g, models, f, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    E = {n: (g[c] - g["y"]).to_numpy(float) for n, c in _sel_models(models).items()}
    lim = np.nanquantile(np.abs(np.concatenate(list(E.values()))), 0.98)
    for i, (n, e) in enumerate(E.items()):
        ax.hist(np.clip(e, -lim, lim), bins=60, histtype="step", lw=1.1, color=COLORS[i % len(COLORS)], label=n)
    ax.set_title(title + " - error distribution (forecast - actual, clipped at 98th pct of |error|)")
    ax.set_xlabel("EUR/MWh"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_qcal(gq, f, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, (m, gm) in enumerate(gq.groupby("model")):
        ax.plot(gm["level"], gm["observed_below"], marker="o", color=COLORS[i % len(COLORS)], label=m)
    ax.plot([0, 1], [0, 1], color="black", lw=0.8)
    ax.set_xlabel("nominal quantile level"); ax.set_ylabel("observed share of actuals below")
    ax.set_title(title + "\nquantile calibration (on the diagonal = calibrated)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_importance(gi, f, title, k=20):
    plt = _plt()
    t = gi.nlargest(k, "mean").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(t) + 1.2))
    ax.barh(t["feature"], t["mean"], xerr=t["std"], color=COLORS[0], alpha=0.85)
    ax.set_title(title + f" - top {k} features (mean share over folds, bar = std)"); ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


# ================================================================ classifiers (rare events)
CLASSIFIER_README = """CLASSIFIER DIAGNOSTICS (ml_diagnostics.classifier_diagnostics)
  capture_rate.csv        share of all event hours that fall into the top 1 / 2 / 5 / 10 / 20 % of hours ranked by
                          predicted risk ("if Swissgrid / a trader watched only the riskiest x % of hours, how many
                          spikes would they see?"), plus precision and lift in that top group
  metrics_by_year.csv     Brier, Brier skill vs the reference model, PR-AUC, ROC-AUC, event rate per model and year
  calibration_bins.csv    20 quantile bins of predicted probability: mean prediction vs observed rate
  monthly_rates.csv       observed event rate vs mean predicted probability per month
  plots/<group>/          reliability.png, pr_curve.png, capture_rate.png, monthly_rates.png, brier_by_year.png
"""


def _auc(y, p):
    r = pd.Series(p).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def _ap(y, p):
    o = np.argsort(-p); ys = y[o]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(prec[ys == 1].mean()) if ys.sum() else np.nan


def classifier_diagnostics(P: pd.DataFrame, models: list[str], out: Path, y_col: str = "y", time_col: str = "hk",
                           group_col: str | None = "side", reference: dict | str | None = None, title: str = "",
                           subdir: str = "diagnostics") -> None:
    """P: one row per test unit with y (0/1), a time column and one probability column per model.
    reference: model used for the Brier skill (str, or dict group -> model)."""
    d = Path(out) / subdir; pdir = d / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    _write_readme(d, CLASSIFIER_README)
    P = P.copy(); P["_t"] = pd.to_datetime(P[time_col]); P["year"] = P["_t"].dt.year
    P["month"] = P["_t"].dt.to_period("M").dt.to_timestamp()
    if group_col is None:
        P["_g"] = "all"; group_col = "_g"
    cap, yr, cal, mon = [], [], [], []
    for gname, g in P.groupby(group_col, sort=False):
        ms = [m for m in models if m in g and g[m].notna().any()]
        g = g.dropna(subset=ms)
        if len(g) == 0 or not ms:
            continue
        y = g[y_col].to_numpy(float)
        ref = reference.get(gname) if isinstance(reference, dict) else reference
        for m in ms:
            p = g[m].to_numpy(float); o = np.argsort(-p)
            for share in (0.01, 0.02, 0.05, 0.10, 0.20):
                k = max(1, int(round(share * len(g))))
                hit = y[o[:k]].sum()
                cap.append({"group": gname, "model": m, "top_share": share, "n_flagged": k, "events_flagged": int(hit),
                            "events_total": int(y.sum()), "capture_rate": hit / max(y.sum(), 1),
                            "precision": hit / k, "lift": (hit / k) / max(y.mean(), 1e-12)})
            for yv, gy in g.groupby("year"):
                yy, pp = gy[y_col].to_numpy(float), gy[m].to_numpy(float)
                b = ((pp - yy) ** 2).mean()
                bref = ((gy[ref].to_numpy(float) - yy) ** 2).mean() if ref in gy else np.nan
                yr.append({"group": gname, "model": m, "year": yv, "n": len(gy), "events": int(yy.sum()),
                           "event_rate": yy.mean(), "mean_pred": pp.mean(), "Brier": b,
                           "Brier_skill_vs_ref": 1 - b / bref if bref and bref > 0 else np.nan, "reference": ref,
                           "PR_AUC": _ap(yy, pp), "ROC_AUC": _auc(yy, pp)})
            q = pd.qcut(pd.Series(p).rank(method="first"), 20, labels=False)
            t = pd.DataFrame({"p": p, "y": y}).groupby(q.to_numpy()).agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
            cal.append(t.reset_index(names="bin").assign(group=gname, model=m))
            mm = g.groupby("month").agg(obs_rate=(y_col, "mean"), mean_pred=(m, "mean"), n=(y_col, "size")).reset_index()
            mon.append(mm.assign(group=gname, model=m))
        sp = pdir / _safe(gname); sp.mkdir(parents=True, exist_ok=True)
        _try_plot(_plot_reliability, pd.concat([c for c in cal if c["group"].iloc[0] == gname]), sp / "reliability.png", f"{title} {gname}")
        _try_plot(_plot_pr, g, ms, y_col, sp / "pr_curve.png", f"{title} {gname}")
        _try_plot(_plot_capture, pd.DataFrame([c for c in cap if c["group"] == gname]), sp / "capture_rate.png", f"{title} {gname}")
        _try_plot(_plot_monthly, pd.concat([x for x in mon if x["group"].iloc[0] == gname]), sp / "monthly_rates.png", f"{title} {gname}")
        _try_plot(_plot_brier_year, pd.DataFrame([r for r in yr if r["group"] == gname]), sp / "brier_by_year.png", f"{title} {gname}")
    pd.DataFrame(cap).to_csv(d / "capture_rate.csv", index=False)
    pd.DataFrame(yr).to_csv(d / "metrics_by_year.csv", index=False)
    if cal:
        pd.concat(cal).to_csv(d / "calibration_bins.csv", index=False)
    if mon:
        pd.concat(mon).to_csv(d / "monthly_rates.csv", index=False)
    print(f"   diagnostics written: {d}")


def _plot_reliability(c, f, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (m, gm) in enumerate(c.groupby("model", sort=False)):
        ax.plot(gm["mean_pred"], gm["obs_rate"], marker="o", ms=3, lw=1, color=COLORS[i % len(COLORS)], label=m)
    mx = float(max(c["mean_pred"].max(), c["obs_rate"].max()))
    ax.plot([0, mx], [0, mx], color="black", lw=0.8)
    ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("mean predicted probability (20 bins)"); ax.set_ylabel("observed event rate")
    ax.set_title(title + "\nreliability (diagonal = calibrated)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_pr(g, ms, y_col, f, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    y = g[y_col].to_numpy(float)
    for i, m in enumerate(ms):
        o = np.argsort(-g[m].to_numpy(float)); ys = y[o]; tp = np.cumsum(ys)
        prec, rec = tp / np.arange(1, len(ys) + 1), tp / max(ys.sum(), 1)
        ax.plot(rec, prec, lw=1, color=COLORS[i % len(COLORS)], label=f"{m} (AP {_ap(y, g[m].to_numpy(float)):.3f})")
    ax.axhline(y.mean(), color="grey", ls="--", lw=0.8, label=f"base rate {y.mean():.4f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(title + "\nprecision-recall"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_capture(c, f, title):
    plt = _plt()
    t = c.pivot(index="top_share", columns="model", values="capture_rate")
    ax = (t * 100).plot(kind="bar", figsize=(8, 4), color=COLORS[:t.shape[1]], width=0.85)
    ax.set_xticklabels([f"top {x*100:g} %" for x in t.index], rotation=0)
    ax.set_ylabel("% of all events captured"); ax.set_title(title + " - events captured in the riskiest hours")
    ax.legend(fontsize=7, ncol=3)
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_monthly(m, f, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 4))
    first = m[m["model"] == m["model"].iloc[0]]
    ax.plot(first["month"], first["obs_rate"], color="black", lw=1.4, label="observed")
    for i, (mm, gm) in enumerate(m.groupby("model", sort=False)):
        ax.plot(gm["month"], gm["mean_pred"], lw=0.9, color=COLORS[i % len(COLORS)], label=mm)
    ax.set_ylabel("event rate"); ax.set_title(title + " - monthly event rate vs mean predicted probability")
    ax.legend(fontsize=7, ncol=6); fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_brier_year(t, f, title):
    plt = _plt()
    p = t.pivot(index="year", columns="model", values="Brier_skill_vs_ref")
    ax = p.plot(kind="bar", figsize=(10, 4), color=COLORS[:p.shape[1]], width=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Brier skill vs reference"); ax.set_title(title + " - Brier skill by year"); ax.legend(fontsize=7, ncol=4)
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(f); plt.close(fig)


# ================================================================ revenue simulation
REVENUE_README = """REVENUE DIAGNOSTICS (ml_diagnostics.revenue_diagnostics)
  revenue_monthly.csv     revenue per block, win rate and mean bid per series x strategy x month
  bid_vs_clearing.csv     per series x strategy: how far bids are from the highest accepted bid c (median ratio,
                          share of bids lost because too high, revenue left on the table when accepted)
  plots/<series>/         cumulative_revenue.png, share_of_oracle_by_year.png, win_rate_monthly.png,
                          bid_ratio_hist.png (bid / c), offset_k.png (chosen offset over time)
"""


def revenue_diagnostics(d: pd.DataFrame, strategies: list[str], out: Path, offsets: pd.DataFrame | None = None,
                        time_col: str = "start", show: list[str] | None = None) -> None:
    dd = Path(out) / "diagnostics"; pdir = dd / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    _write_readme(dd, REVENUE_README)
    d = d.copy(); d["_t"] = _local(d[time_col]); d["month"] = d["_t"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    show = show or strategies
    mon, bvc = [], []
    for s, g in d.groupby("series", sort=False):
        for st in strategies:
            b, c = np.maximum(g[st].to_numpy(float), 0), g["c"].to_numpy(float)
            win = b <= c + 1e-9
            ratio = np.where(c > 0, b / c, np.nan)
            bvc.append({"series": s, "strategy": st, "n": len(g), "win_rate": win.mean(),
                        "median_bid_over_c": np.nanmedian(ratio), "share_lost_too_high": (~win).mean(),
                        "mean_left_on_table_when_won": np.mean((c - b)[win]) if win.any() else np.nan,
                        "mean_c_when_lost": np.mean(c[~win]) if (~win).any() else np.nan})
            m = g.assign(_w=win).groupby("month").agg(rev_per_block=(f"rev_{st}", "mean"), win_rate=("_w", "mean"),
                                                      mean_bid=(st, "mean"), mean_c=("c", "mean"), n=("c", "size"))
            mon.append(m.reset_index().assign(series=s, strategy=st))
        sp = pdir / _safe(s); sp.mkdir(parents=True, exist_ok=True)
        _try_plot(_plot_cumrev, g, show, sp / "cumulative_revenue.png", s)
        _try_plot(_plot_share_year, g, show, sp / "share_of_oracle_by_year.png", s)
        _try_plot(_plot_bid_ratio, g, [x for x in show if x != "oracle"], sp / "bid_ratio_hist.png", s)
        if offsets is not None and len(offsets):
            _try_plot(_plot_offsets, offsets[offsets["series"] == s], sp / "offset_k.png", s)
    M = pd.concat(mon); M.to_csv(dd / "revenue_monthly.csv", index=False)
    pd.DataFrame(bvc).to_csv(dd / "bid_vs_clearing.csv", index=False)
    for s, gm in M.groupby("series"):
        _try_plot(_plot_winrate, gm[gm.strategy.isin(show)], pdir / _safe(s) / "win_rate_monthly.png", s)
    print(f"   diagnostics written: {dd}")


def _plot_cumrev(g, show, f, s):
    plt = _plt(); fig, ax = plt.subplots(figsize=(12, 4))
    g = g.sort_values("_t")
    for i, st in enumerate(show):
        ax.plot(g["_t"], g[f"rev_{st}"].cumsum() * 4, lw=1, color=COLORS[i % len(COLORS)], label=st)
    ax.set_ylabel("cumulative revenue per MW"); ax.set_title(f"{s} - cumulative revenue by strategy"); ax.legend(ncol=6, fontsize=8)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_share_year(g, show, f, s):
    plt = _plt()
    y = g["_t"].dt.year
    t = pd.DataFrame({st: g[f"rev_{st}"].groupby(y).sum() / g["rev_oracle"].groupby(y).sum() for st in show if st != "oracle"})
    ax = (t * 100).plot(kind="bar", figsize=(9, 4), color=COLORS[:t.shape[1]], width=0.85)
    ax.set_ylabel("% of oracle revenue"); ax.set_title(f"{s} - share of oracle revenue by year"); ax.legend(ncol=4, fontsize=7)
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_bid_ratio(g, show, f, s):
    plt = _plt(); fig, ax = plt.subplots(figsize=(8, 4))
    c = g["c"].to_numpy(float)
    for i, st in enumerate(show):
        r = np.where(c > 0, np.maximum(g[st].to_numpy(float), 0) / c, np.nan)
        ax.hist(np.clip(r[~np.isnan(r)], 0, 3), bins=60, histtype="step", lw=1, color=COLORS[i % len(COLORS)], label=st)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_xlabel("bid / highest accepted bid (right of 1 = rejected)"); ax.set_title(f"{s} - bid relative to clearing")
    ax.legend(fontsize=7, ncol=3); fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_winrate(gm, f, s):
    plt = _plt(); fig, ax = plt.subplots(figsize=(12, 4))
    for i, (st, g) in enumerate(gm.groupby("strategy", sort=False)):
        ax.plot(g["month"], g["win_rate"] * 100, lw=1, color=COLORS[i % len(COLORS)], label=st)
    ax.set_ylabel("win rate, %"); ax.set_title(f"{s} - monthly win rate"); ax.legend(ncol=6, fontsize=7)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def _plot_offsets(o, f, s):
    plt = _plt(); fig, ax = plt.subplots(figsize=(12, 3.5))
    for i, (b, g) in enumerate(o.groupby("base")):
        ax.step(pd.to_datetime(g["month"].astype(str)), g["k"], where="post", color=COLORS[i % len(COLORS)], label=b)
    ax.set_ylabel("offset k (bid = k x forecast)"); ax.set_title(f"{s} - monthly re-chosen offset"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)
