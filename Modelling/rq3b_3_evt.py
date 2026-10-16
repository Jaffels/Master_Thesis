"""RQ3 (done_todo 2.3 item 11a): extreme-value theory for the SIZE of imbalance-price spikes.

Peaks-over-threshold (POT) with a generalised Pareto distribution (GPD) for the excess of the price over a
threshold. Spikes come in episodes (consecutive quarter-hours), so exceedances are DECLUSTERED first (runs
declustering: a cluster ends after --run consecutive QH below the threshold, default 4 = 1 h) and only the cluster
peak is used (approximately independent).  Long side = lower tail -> the analysis uses x = -price.

Parts
  A  Descriptive, per year and per period: clusters, cluster length, mean / max peak excess over the ROLLING
     threshold (thr_short_q990 / thr_long_q010 = the RQ3b spike definition).
  B  Tail shape per period with a FIXED threshold (q99 of x inside the period; stability check at q97.5 / q99.5):
     GPD shape xi and scale sigma (MLE, cluster bootstrap 95 % CI), Anderson-Darling / KS on the PIT (approximate,
     parameters estimated), 1-year return level (price reached by one cluster peak per year on average), and a
     likelihood-ratio test 'same xi in all periods' (common xi, period-specific sigma vs fully separate).
     Periods (declared up front, as in rq2_8 / rq3b_2): pre_crisis 31 Mar 2016 - 31 Aug 2021, crisis Sep 2021 -
     Jun 2023, calm Jul 2023 - 2 Mar 2025, post_cap from 3 Mar 2025 (aFRR energy price cap).
  C  OUT-OF-SAMPLE size forecast: given that a spike cluster starts, how large will its peak excess over the rolling
     threshold be?  Rolling origin as rq3b_1 (refit every 6 months from 1 Jan 2021, 2-day embargo; only clusters that
     ENDED before fold start - embargo are used). Predictive distributions:
        gpd_exp     GPD on all earlier cluster excesses (expanding)
        gpd_365     GPD on the clusters of the last 365 days (expanding if < --min-fit clusters)
        gpd_rel     GPD on the RELATIVE excess (excess / max(|threshold|, 100)), rescaled with the threshold of the test cluster
                    (threshold is known ex ante) -> follows the price level
        expo_exp    exponential (xi = 0) on all earlier excesses
        emp_exp     empirical distribution of all earlier excesses (benchmark)
        emp_365     empirical, last 365 days
     Scores on the test cluster peaks: CRPS (quantile-score approximation on 99 levels), tail quantile score (mean over
     tau >= 0.9), coverage of the 50 / 90 / 99 % quantiles, log score (parametric only), PIT. Day-block bootstrap
     (1,000) of the CRPS difference vs emp_exp.

Outputs (Modelling/Output/rq3b_3<tag>/): clusters.csv, desc_year.csv, desc_period.csv, gpd_period.csv,
gpd_stability.csv, xi_equality_test.csv, oos_predictions.csv, oos_scores.csv, oos_bootstrap.csv, report_rq3b_3.txt,
plots/ (mean residual life, xi vs threshold, QQ, PIT, return levels).

Run from Master_Thesis with .venv active (~2-4 min):
    python Modelling/rq3b_3_evt.py                       # -> Modelling/Output/rq3b_3/
    options: --run 4 (declustering gap in QH), --boot 500, --min-fit 30, --tag NAME, --quick (boot 50)
Read-only on Clean/Data; writes only Modelling/Output/rq3b_3<tag>/.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import run_utils as RU  # run.log, run_meta.json, run history, macOS notification

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Clean"))
import views as V  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "Output" / "rq3b_3"
SIDES = {"short": ("tgt_price_short_eur_mwh", "thr_short_q990", 1.0),
         "long": ("tgt_price_long_eur_mwh", "thr_long_q010", -1.0)}
PERIODS = [("pre_crisis", "2016-01-01", "2021-09-01"), ("crisis", "2021-09-01", "2023-07-01"),
           ("calm", "2023-07-01", "2025-03-03"), ("post_cap", "2025-03-03", "2100-01-01")]
TEST_START = pd.Timestamp("2021-01-01")
EMBARGO = pd.Timedelta(days=2)
TAUS = np.arange(1, 100) / 100.0
QCOV = (0.5, 0.9, 0.99)
XI_MAX = 2.0          # upper bound of the GPD shape (xi >= 1: infinite mean; reported as at_bound)
REL_MIN = 100.0       # gpd_rel: excess / max(|threshold|, REL_MIN) (long threshold crosses 0 in some years)
lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


# ------------------------------------------------------------------ GPD
def gpd_nll(par, y):
    ls, xi = par
    s = np.exp(ls)
    if abs(xi) < 1e-6:
        return len(y) * ls + y.sum() / s
    z = 1 + xi * y / s
    if (z <= 0).any():
        return 1e12
    return len(y) * ls + (1 + 1 / xi) * np.log(z).sum()


def gpd_fit(y: np.ndarray, xi_fixed: float | None = None) -> tuple[float, float, float]:
    """MLE of (sigma, xi); returns (sigma, xi, loglik). xi bounded to [-0.5, XI_MAX] (multistart)."""
    from scipy.optimize import minimize
    y = np.asarray(y, float)
    y = y[y > 0]
    if len(y) < 5:
        return np.nan, np.nan, np.nan
    m = y.mean()
    if xi_fixed is not None:
        if xi_fixed == 0:
            return m, 0.0, -gpd_nll((np.log(m), 0.0), y)
        r = minimize(lambda p: gpd_nll((p[0], xi_fixed), y), [np.log(m)], method="Nelder-Mead")
        return float(np.exp(r.x[0])), xi_fixed, -float(r.fun)
    best = None
    for xi0 in (-0.2, 0.1, 0.4, 0.8):
        s0 = max(m * (1 - xi0), 1e-3)
        r = minimize(gpd_nll, [np.log(s0), xi0], args=(y,), method="L-BFGS-B",
                     bounds=[(np.log(m) - 12, np.log(m) + 12), (-0.5, XI_MAX)])
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[0])), float(best.x[1]), -float(best.fun)


def gpd_cdf(y, s, xi):
    y = np.maximum(np.asarray(y, float), 0)
    if abs(xi) < 1e-6:
        return 1 - np.exp(-y / s)
    z = np.maximum(1 + xi * y / s, 0)
    return 1 - np.where(z > 0, z ** (-1 / xi), 0.0)


def gpd_ppf(p, s, xi):
    p = np.asarray(p, float)
    if abs(xi) < 1e-6:
        return -s * np.log(1 - p)
    return s / xi * ((1 - p) ** (-xi) - 1)


def gpd_logpdf(y, s, xi):
    y = np.asarray(y, float)
    if abs(xi) < 1e-6:
        return -np.log(s) - y / s
    z = 1 + xi * y / s
    return np.where(z > 0, -np.log(s) - (1 + 1 / xi) * np.log(np.maximum(z, 1e-300)), -50.0)


def ad_ks(u: np.ndarray) -> tuple[float, float, float]:
    """Anderson-Darling A2, KS D and KS p (approximate: parameters estimated on the same data) for PIT values u."""
    from scipy import stats
    u = np.sort(np.clip(u, 1e-10, 1 - 1e-10))
    n = len(u)
    i = np.arange(1, n + 1)
    a2 = -n - ((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))).sum() / n
    ks = stats.kstest(u, "uniform")
    return float(a2), float(ks.statistic), float(ks.pvalue)


# ------------------------------------------------------------------ data
def load() -> pd.DataFrame:
    cols = ["ts_utc", "tgt_price_short_eur_mwh", "tgt_price_long_eur_mwh", "thr_short_q990", "thr_long_q010"]
    df = V.load_rq3(columns=cols)
    t = pd.to_datetime(df["ts_utc"])
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t
    df["t"] = t.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.set_index("t").drop(columns="ts_utc")
    full = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    df = df.reindex(full)                          # regular grid: missing QH = gap (breaks clusters only via run)
    df["date"] = df.index.tz_localize("UTC").tz_convert(V.TZ).tz_localize(None).normalize()
    return df


def period_of(d: pd.Series) -> pd.Series:
    out = pd.Series(index=d.index, dtype=object)
    for name, a, b in PERIODS:
        out[(d >= pd.Timestamp(a)) & (d < pd.Timestamp(b))] = name
    return out


def clusters(x: np.ndarray, u: np.ndarray, t: pd.DatetimeIndex, run: int) -> pd.DataFrame:
    """Runs declustering of exceedances x > u (NaN = no exceedance). Returns one row per cluster: start, end, n_qh
    (exceedance QH), peak excess, threshold at the peak, peak value."""
    ex = np.where(np.isfinite(x) & np.isfinite(u), x - u, np.nan)
    pos = np.flatnonzero(ex > 0)
    if len(pos) == 0:
        return pd.DataFrame(columns=["start", "end", "n_qh", "excess", "thr", "peak"])
    brk = np.flatnonzero(np.diff(pos) > run) + 1
    starts = np.r_[0, brk]
    ends = np.r_[brk, len(pos)]
    rows = []
    for a, b in zip(starts, ends):
        p = pos[a:b]
        k = p[np.argmax(ex[p])]
        rows.append((t[p[0]], t[p[-1]], len(p), ex[k], u[k], x[k]))
    return pd.DataFrame(rows, columns=["start", "end", "n_qh", "excess", "thr", "peak"])


# ------------------------------------------------------------------ part B
def fixed_threshold_fits(df, side, sign, price, per, q, run, boot, rng):
    x = sign * df[price].to_numpy(float)
    res = []
    for name, a, b in PERIODS:
        m = ((per == name).to_numpy())
        if m.sum() == 0:
            continue
        xm = x[m]
        u = np.nanquantile(xm, q)
        tt = df.index[m]
        c = clusters(xm, np.full(m.sum(), u), tt, run)
        years = m.sum() / (96 * 365.25)
        y = c["excess"].to_numpy(float)
        s, xi, ll = gpd_fit(y)
        se, xe, _ = gpd_fit(y, 0.0)
        lr0 = 2 * (ll - gpd_fit(y, 0.0)[2]) if np.isfinite(ll) else np.nan
        bx, bs = [], []
        for _ in range(boot):
            yb = rng.choice(y, len(y), replace=True)
            sb, xb, _ = gpd_fit(yb)
            bx.append(xb)
            bs.append(sb)
        lam = len(y) / years
        rl = u + gpd_ppf(1 - 1 / lam, s, xi) if lam > 1 else np.nan
        rlb = [u + gpd_ppf(1 - 1 / lam, sb, xb) for sb, xb in zip(bs, bx) if lam > 1 and np.isfinite(xb)]
        a2, ksd, ksp = ad_ks(gpd_cdf(y, s, xi)) if len(y) >= 10 else (np.nan,) * 3
        res.append({"side": side, "period": name, "q": q, "threshold": sign * u, "years": years,
                    "clusters": len(y), "clusters_per_year": lam, "mean_excess": y.mean() if len(y) else np.nan,
                    "max_excess": y.max() if len(y) else np.nan, "sigma": s, "xi": xi,
                    "xi_at_bound": bool(np.isfinite(xi) and xi > XI_MAX - 1e-3),
                    "xi_lo": np.nanpercentile(bx, 2.5) if bx else np.nan,
                    "xi_hi": np.nanpercentile(bx, 97.5) if bx else np.nan, "loglik": ll,
                    "LR_xi0": lr0, "p_xi0": _chi2p(lr0, 1), "AD": a2, "KS_D": ksd, "KS_p": ksp,
                    "return_level_1y": sign * rl if np.isfinite(rl) else np.nan,
                    "rl_lo": sign * np.nanpercentile(rlb, 2.5 if sign > 0 else 97.5) if rlb else np.nan,
                    "rl_hi": sign * np.nanpercentile(rlb, 97.5 if sign > 0 else 2.5) if rlb else np.nan,
                    "_y": y})
    return res


def _chi2p(stat, dfree):
    from scipy import stats
    return float(stats.chi2.sf(stat, dfree)) if np.isfinite(stat) else np.nan


def xi_equality(fits) -> dict:
    """LR test: common xi (period-specific sigma) vs separate (sigma, xi) per period."""
    from scipy.optimize import minimize
    ys = [f["_y"] for f in fits if len(f["_y"]) >= 10]
    if len(ys) < 2:
        return {}
    ll_sep = sum(gpd_fit(y)[2] for y in ys)

    def nll(xi):
        return -sum(gpd_fit(y, float(xi))[2] for y in ys)
    best = min((minimize(lambda v: nll(v[0]), [x0], method="Nelder-Mead", bounds=None) for x0 in (0.0, 0.3, 0.7)),
               key=lambda r: r.fun)
    xi_c = float(np.clip(best.x[0], -0.5, XI_MAX))
    ll_c = -nll(xi_c)
    lr = 2 * (ll_sep - ll_c)
    return {"periods": len(ys), "xi_common": xi_c, "loglik_common": ll_c, "loglik_separate": ll_sep,
            "LR": lr, "df": len(ys) - 1, "p": _chi2p(lr, len(ys) - 1)}


# ------------------------------------------------------------------ part C
def qscore(y, q, taus):
    """Quantile (pinball) score, mean over taus; y (n,), q (n, k)."""
    d = y[:, None] - q
    return (np.maximum(taus * d, (taus - 1) * d)).mean(axis=1)


def predictive(kind, past: pd.DataFrame, t0, thr_test):
    """Return (quantiles (n_test, 99), cdf fn, logpdf fn or None) for the test thresholds thr_test."""
    n = len(thr_test)
    if kind.endswith("_365"):
        rec = past[past["end"] >= t0 - pd.Timedelta(days=365)]
        if len(rec) >= MIN_FIT:
            past = rec
    y = past["excess"].to_numpy(float)
    if kind.startswith("emp"):
        qs = np.quantile(y, TAUS)
        ys = np.sort(y)
        cdf = lambda v: np.searchsorted(ys, v, side="right") / len(ys)
        return np.tile(qs, (n, 1)), cdf, None, (np.nan, np.nan)
    if kind == "gpd_rel":
        r = y / np.abs(past["thr"].to_numpy(float)).clip(min=REL_MIN)
        s, xi, _ = gpd_fit(r)
        sc = np.abs(thr_test).clip(min=REL_MIN)
        q = gpd_ppf(TAUS, s, xi)[None, :] * sc[:, None]
        cdf = lambda v: gpd_cdf(v / sc, s, xi)
        lp = lambda v: gpd_logpdf(v / sc, s, xi) - np.log(sc)
        return q, cdf, lp, (s, xi)
    s, xi, _ = gpd_fit(y, 0.0 if kind.startswith("expo") else None)
    q = np.tile(gpd_ppf(TAUS, s, xi), (n, 1))
    return q, (lambda v: gpd_cdf(v, s, xi)), (lambda v: gpd_logpdf(v, s, xi)), (s, xi)


MIN_FIT = 30
KINDS = ["gpd_exp", "gpd_365", "gpd_rel", "expo_exp", "emp_exp", "emp_365"]


def oos(cl: pd.DataFrame, side: str, step_months: int = 6):
    end = cl["start"].max()
    rows, fits = [], []
    f0 = TEST_START
    while f0 <= end:
        f1 = f0 + pd.DateOffset(months=step_months)
        past = cl[cl["end"] < f0 - EMBARGO]
        test = cl[(cl["start"] >= f0) & (cl["start"] < f1)]
        if len(past) >= MIN_FIT and len(test):
            y = test["excess"].to_numpy(float)
            thr = test["thr"].to_numpy(float)
            out = test[["start", "end", "n_qh", "excess", "thr"]].copy()
            out["side"], out["fold"] = side, f0.date()
            for k in KINDS:
                q, cdf, lp, par = predictive(k, past, f0, thr)
                out[f"crps_{k}"] = qscore(y, q, TAUS)
                ti = TAUS >= 0.9
                out[f"tailqs_{k}"] = qscore(y, q[:, ti], TAUS[ti])
                for c in QCOV:
                    out[f"hit{int(c * 100)}_{k}"] = (y <= q[:, int(round(c * 100)) - 1]).astype(float)
                out[f"pit_{k}"] = cdf(y)
                out[f"logs_{k}"] = lp(y) if lp is not None else np.nan
                out[f"q90_{k}"] = q[:, 89]
                fits.append({"side": side, "fold": f0.date(), "model": k, "n_fit": len(past), "sigma": par[0],
                             "xi": par[1]})
            rows.append(out)
        f0 = f1
    return (pd.concat(rows) if rows else pd.DataFrame()), pd.DataFrame(fits)


def boot_diff(diff, days, rng, n=1000):
    ud, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(ud))
    c = np.bincount(inv, minlength=len(ud))
    draws = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(ud), len(ud)) for _ in range(n))]
    return s.sum() / c.sum(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


# ------------------------------------------------------------------ plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_mrl(x, side, f):
    plt = _plt()
    x = x[np.isfinite(x)]
    us = np.nanquantile(x, np.linspace(0.90, 0.999, 60))
    me = [np.mean(x[x > u] - u) for u in us]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(us, me, ".-")
    ax.set_xlabel("threshold (EUR/MWh, sign-adjusted)")
    ax.set_ylabel("mean excess")
    ax.set_title(f"Mean residual life, {side} (all QH, 90-99.9 % quantiles)")
    fig.tight_layout(); fig.savefig(f, dpi=120); plt.close(fig)


def plot_stab(st, side, f):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for per, g in st[st.side == side].groupby("period", sort=False):
        ax.errorbar(g["q"], g["xi"], yerr=[g["xi"] - g["xi_lo"], g["xi_hi"] - g["xi"]],
                    marker="o", capsize=3, label=per)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("threshold quantile"); ax.set_ylabel("GPD shape xi (95 % CI)")
    ax.set_title(f"Threshold stability, {side}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f, dpi=120); plt.close(fig)


def plot_qq(fits, side, f):
    plt = _plt()
    fs = [r for r in fits if r["side"] == side and len(r["_y"]) >= 10]
    fig, axes = plt.subplots(1, len(fs), figsize=(3.2 * len(fs), 3.2), squeeze=False)
    for ax, r in zip(axes[0], fs):
        y = np.sort(r["_y"])
        p = (np.arange(1, len(y) + 1) - 0.5) / len(y)
        th = gpd_ppf(p, r["sigma"], r["xi"])
        ax.loglog(th, y, ".", ms=3)
        lo, hi = max(min(th.min(), y.min()), 1e-2), max(th.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"{r['period']} xi={r['xi']:.2f}", fontsize=9)
        ax.set_xlabel("GPD quantile"); ax.set_ylabel("observed excess")
    fig.suptitle(f"QQ, cluster peak excess over fixed q99, {side}", fontsize=10)
    fig.tight_layout(); fig.savefig(f, dpi=120); plt.close(fig)


def plot_pit(P, side, f):
    plt = _plt()
    g = P[P.side == side]
    fig, axes = plt.subplots(1, len(KINDS), figsize=(2.6 * len(KINDS), 2.6), squeeze=False)
    for ax, k in zip(axes[0], KINDS):
        ax.hist(g[f"pit_{k}"].dropna(), bins=10, range=(0, 1), density=True)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_title(k, fontsize=9)
    fig.suptitle(f"PIT of test cluster excess, {side} (flat = calibrated)", fontsize=10)
    fig.tight_layout(); fig.savefig(f, dpi=120); plt.close(fig)


def _try(fn, *a):
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001
        log(f"  (plot skipped: {e})")


# ------------------------------------------------------------------ main
def main() -> None:
    global OUT, MIN_FIT
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=4, help="declustering: QH below threshold that end a cluster")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--min-fit", type=int, default=30)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    MIN_FIT = a.min_fit
    boot = 50 if a.quick else a.boot
    OUT = HERE / "Output" / ("rq3b_3" + a.tag)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    RU.set_output(OUT)
    t0 = time.time()
    rng = np.random.default_rng(20261016)
    df = load()
    per = period_of(df["date"])
    log(f"RQ3 EVT: {df[SIDES['short'][0]].notna().sum():,} QH with prices, {df.index.min()} -> {df.index.max()} (UTC); "
        f"declustering run = {a.run} QH; bootstrap {boot}")

    # ---------- A: clusters over the rolling threshold
    allc, desc_y, desc_p = [], [], []
    for side, (price, thr, sign) in SIDES.items():
        c = clusters(sign * df[price].to_numpy(float), sign * df[thr].to_numpy(float), df.index, a.run)
        c["side"] = side
        c["peak_price"] = sign * c["peak"]
        c["year"] = c["start"].dt.year
        c["period"] = period_of(c["start"].dt.tz_localize("UTC").dt.tz_convert(V.TZ).dt.tz_localize(None).dt.normalize()).to_numpy()
        allc.append(c)
        for key, grp, store in (("year", "year", desc_y), ("period", "period", desc_p)):
            t = c.groupby(grp, sort=False).agg(clusters=("excess", "size"), qh=("n_qh", "sum"),
                                               mean_len_qh=("n_qh", "mean"), mean_excess=("excess", "mean"),
                                               median_excess=("excess", "median"), max_excess=("excess", "max"),
                                               mean_thr=("thr", "mean"))
            t.insert(0, "side", side)
            store.append(t.reset_index())
    C = pd.concat(allc, ignore_index=True)
    C.drop(columns=["peak"]).to_csv(OUT / "clusters.csv", index=False)
    DY, DP = pd.concat(desc_y), pd.concat(desc_p)
    DY.to_csv(OUT / "desc_year.csv", index=False)
    DP.to_csv(OUT / "desc_period.csv", index=False)
    log("\n== A  clusters over the rolling threshold (RQ3b spike definition); excess in EUR/MWh (long: below threshold)")
    log(DP.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    log("\nper year:")
    log(DY.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    # ---------- B: fixed threshold per period
    log("\n== B  GPD tail per period, FIXED threshold = quantile of x inside the period (cluster peaks)")
    fits_main, stab, eq = [], [], []
    for side, (price, thr, sign) in SIDES.items():
        _try(plot_mrl, sign * df[price].to_numpy(float), side, OUT / "plots" / f"mrl_{side}.png")
        for q in (0.975, 0.99, 0.995):
            f = fixed_threshold_fits(df, side, sign, price, per, q, a.run, boot, rng)
            stab += f
            if q == 0.99:
                fits_main += f
                e = xi_equality(f)
                if e:
                    eq.append({"side": side, "q": q, **e})
                fpc = [r for r in f if r["period"] in ("calm", "post_cap")]
                e2 = xi_equality(fpc)
                if e2:
                    eq.append({"side": side, "q": q, "test": "calm_vs_post_cap", **e2})
    keep = lambda L: pd.DataFrame([{k: v for k, v in r.items() if k != "_y"} for r in L])
    G, S = keep(fits_main), keep(stab)
    E = pd.DataFrame(eq)
    G.to_csv(OUT / "gpd_period.csv", index=False)
    S.to_csv(OUT / "gpd_stability.csv", index=False)
    E.to_csv(OUT / "xi_equality_test.csv", index=False)
    show = ["side", "period", "threshold", "clusters", "clusters_per_year", "mean_excess", "sigma", "xi", "xi_at_bound", "xi_lo",
            "xi_hi", "p_xi0", "KS_p", "return_level_1y", "rl_lo", "rl_hi"]
    log(G[show].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    log("\nthreshold stability (xi at q97.5 / q99 / q99.5):")
    log(S.pivot_table(index=["side", "period"], columns="q", values="xi", sort=False).to_string(float_format=lambda v: f"{v:+.3f}"))
    log("\nLR test 'same xi in all periods' (and calm vs post-cap):")
    log(E.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    for side in SIDES:
        _try(plot_stab, S, side, OUT / "plots" / f"xi_stability_{side}.png")
        _try(plot_qq, fits_main, side, OUT / "plots" / f"qq_{side}.png")

    # ---------- C: out-of-sample size forecast
    log("\n== C  out-of-sample forecast of the cluster peak excess (rolling threshold), refit every 6 months from 2021")
    Ps, Fs = [], []
    for side in SIDES:
        p, fz = oos(C[C.side == side].reset_index(drop=True), side)
        Ps.append(p); Fs.append(fz)
    P = pd.concat(Ps, ignore_index=True)
    F = pd.concat(Fs, ignore_index=True)
    P.to_csv(OUT / "oos_predictions.csv", index=False)
    F.to_csv(OUT / "oos_fits.csv", index=False)
    P["date"] = P["start"].dt.normalize()
    P["from_2023"] = P["start"] >= pd.Timestamp("2023-01-01")
    sc, bt = [], []
    for side in SIDES:
        for pname, g in (("all", P[P.side == side]), ("from_2023", P[(P.side == side) & P.from_2023])):
            if len(g) == 0:
                continue
            for k in KINDS:
                sc.append({"side": side, "period": pname, "model": k, "n": len(g), "CRPS": g[f"crps_{k}"].mean(),
                           "tail_QS": g[f"tailqs_{k}"].mean(),
                           **{f"cov{int(c * 100)}": g[f"hit{int(c * 100)}_{k}"].mean() for c in QCOV},
                           "logscore": g[f"logs_{k}"].mean(),
                           "PIT_KS_p": ad_ks(g[f"pit_{k}"].dropna().to_numpy())[2] if g[f"pit_{k}"].notna().sum() > 5 else np.nan})
                if k != "emp_exp":
                    d, lo, hi = boot_diff((g["crps_emp_exp"] - g[f"crps_{k}"]).to_numpy(), g["date"].to_numpy(), rng)
                    dt, lot, hit = boot_diff((g["tailqs_emp_exp"] - g[f"tailqs_{k}"]).to_numpy(), g["date"].to_numpy(), rng)
                    bt.append({"side": side, "period": pname, "model": k, "dCRPS_vs_emp_exp": d, "ci_lo": lo,
                               "ci_hi": hi, "dTailQS_vs_emp_exp": dt, "tail_ci_lo": lot, "tail_ci_hi": hit})
    SC, BT = pd.DataFrame(sc), pd.DataFrame(bt)
    SC.to_csv(OUT / "oos_scores.csv", index=False)
    BT.to_csv(OUT / "oos_bootstrap.csv", index=False)
    log(SC.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    log("\nCRPS / tail quantile score gain vs empirical-expanding (positive = better), day-block bootstrap 95 % CI:")
    log(BT.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    log("\nGPD shape of the expanding fit by fold:")
    log(F[F.model == "gpd_exp"].pivot(index="fold", columns="side", values="xi").to_string(float_format=lambda v: f"{v:+.3f}"))
    for side in SIDES:
        _try(plot_pit, P, side, OUT / "plots" / f"pit_{side}.png")
    for side in SIDES:
        g = G[G.side == side].set_index("period")
        bits = ", ".join(f"{p} {g.at[p, 'xi']:+.2f}" for p in g.index)
        RU.summary(f"{side}: xi {bits}")
        s = SC[(SC.side == side) & (SC.period == "all")].set_index("model")["CRPS"]
        if len(s):
            RU.summary(f"{side} OOS best CRPS {s.idxmin()}")
    (OUT / "report_rq3b_3.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    RU.run(main, "RQ3 EVT spike size")
