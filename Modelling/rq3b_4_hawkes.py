"""RQ3 (done_todo 2.3 item 11b): self-exciting (Hawkes) model for the CLUSTERING of imbalance-price spikes.

Model (discrete-time bivariate Hawkes on the HOURLY grid, Bernoulli link):
    events N_t^s = 1 if the UTC hour has >= 1 spike QH (spike_short_q990 / spike_long_q010) = exactly the rq3b_1
    target (NaN flags = 0 as in rq3b_1); hours without 4 QH are excluded from the fit (no event for the state). Hourly rather than quarter-hourly
    events, because before Jun 2022 imbalance prices were (mostly) hourly, so a spike hour shows up as 4 identical
    spike QH -> on the QH grid the 'excitation' would mainly measure that mechanical persistence (first test run:
    fast half-life at the 15-min bound, branching ratio ~2).
    lambda_t^s = mu_t^s + a_fast S_fast,t + a_slow S_slow,t + a_cross S_cross,t        (per hour)
    S_k,t      = d_k S_k,t-1 + N_t-1  (own side for fast / slow, OTHER side for cross),  d_k = 2^(-1 / half-life_k)
    P(N_t = 1) = 1 - exp(-lambda_t)
    mu_t       = exp(X_t b): hour-of-day Fourier (3 pairs, local time), Saturday, Sunday / holiday, imbalance-resolution regime (hourly_dominant, qh),
                 log share of spike QH in the previous 30 days (spikehist_*__lb30d_d1, known at D-1 18:00 -> ex ante).
    Half-life ranges: fast 15 min - 6 h (next hours), slow 6 h - 14 days (episodes cluster over days),
    cross 15 min - 14 days. Fitted by maximum likelihood (L-BFGS, analytic gradient), each side separately (the
    likelihood factorises; both use both event histories).
    Branching ratio n = a / (1 - d): expected number of further own spike HOURS triggered by one spike hour.
    'bg' = the same background without excitation (inhomogeneous Bernoulli / Poisson) -> LR test of clustering.

Parts
  A  In-sample fits for the whole sample and per period (pre_crisis / crisis / calm / post_cap, as rq3b_3):
     coefficients with standard errors (inverse numerical Hessian), half-lives, branching ratios, share of spike QH
     attributable to excitation, LR test Hawkes vs bg, AIC.
  B  Out-of-sample risk forecast comparable with rq3b_1: rolling origin (refit every 6 months from 1 Jan 2021,
     2-day embargo), the same hourly target (any spike QH in the UTC hour) and origins:
        d1 = D-1 18:00 local, h1 = 1 h before the hour. Imbalance prices are public 1 h after the quarter-hour
        (build_views_exante rule) -> only hours that ended >= 1 h before the origin are known (h1: the 2 hours
        before the target hour are unknown; d1: from D-1 17:00 local). The unknown gap is SIMULATED (--sims paths,
        both sides jointly); the probability of a spike in the target hour is 1 - mean(exp(-lambda)) over the paths
        (Rao-Blackwellised).
     Models: hawkes_d1, hawkes_h1, bg_d1 (= bg_h1, background only). Scored against rq3b_1 on the same hours:
     baselines + logistic (Platt, Output/rq3b_1) + LightGBM main spec (raw, Output/rq3b_1_raw).
     Brier, skill vs best baseline, log loss, PR / ROC-AUC, ECE, top-5 % capture; periods all / 2021_22 /
     from_2023 / post_cap; day-block bootstrap (1,000) of the Brier gain vs the best baseline.
     Also writes hawkes_features.csv (excitation at the d1 / h1 origin, ex ante) for a possible later use as a
     feature in rq3b_1.

Outputs (Modelling/Output/rq3b_4<tag>/): params_insample.csv, insample_summary.csv, params_folds.csv,
predictions.csv, metrics.csv, brier_gain_bootstrap.csv, hawkes_features.csv, report_rq3b_4.txt, diagnostics/
(classifier diagnostics as rq3b_1), plots/ (intensity examples, kernel).

Run from Master_Thesis with .venv active, after rq3b_1 (default) and rq3b_1_raw exist (~10-25 min):
    python Modelling/rq3b_4_hawkes.py                       # -> Modelling/Output/rq3b_4/
    options: --sims 1000 (d1) --sims-h1 400, --no-insample, --quick (12-month refits, 200 / 100 sims), --tag NAME
Read-only on Clean/Data and Output/rq3b_1*; writes only Modelling/Output/rq3b_4<tag>/.
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
sys.path.insert(0, str(HERE))
import views as V  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "Output" / "rq3b_4"
FLAGS = {"short": "spike_short_q990", "long": "spike_long_q010"}
HIST30 = {"short": "spikehist_short_q990__lb30d_d1", "long": "spikehist_long_q010__lb30d_d1"}
PERIODS = [("pre_crisis", "2016-01-01", "2021-09-01"), ("crisis", "2021-09-01", "2023-07-01"),
           ("calm", "2023-07-01", "2025-03-03"), ("post_cap", "2025-03-03", "2100-01-01")]
TEST_START = pd.Timestamp("2021-01-01")
CAP = pd.Timestamp("2025-03-03")
EMBARGO = pd.Timedelta(days=2)
GAP_H1 = 2      # h1: target hour h, origin h - 1 h, last known hour ends h - 2 h -> hours h-2, h-1 unknown
KERNELS = {"fast": (0.25, 6.0), "slow": (6.0, 336.0), "cross": (0.25, 336.0)}   # half-life range in hours
BASELINES = ["clim_exp", "clim_roll", "persist_d1", "persist_h1", "cal_hist_d1"]
lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


# ------------------------------------------------------------------ data
def load() -> tuple[pd.DataFrame, list[str]]:
    """Hourly grid (UTC). N_side = any spike QH in the hour (complete hours only), calendar of the hour (local)."""
    cols = ["ts_utc", *FLAGS.values(), *HIST30.values(), "cal_holiday", "regime_imb_resolution"]
    df = V.load_rq3(columns=cols)
    t = pd.to_datetime(df["ts_utc"])
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t
    df.index = t.dt.tz_localize(None)
    df = df.drop(columns="ts_utc")
    df["hk"] = df.index.floor("h")
    agg = {f: "max" for f in FLAGS.values()} | {h: "first" for h in HIST30.values()} | \
          {"cal_holiday": "first", "regime_imb_resolution": "first"}
    gq = df.groupby("hk")
    hh = gq.agg(agg)
    complete = gq.size() == 4                    # as rq3b_1: complete hours, NaN flags count as 0
    full = pd.date_range(hh.index.min(), hh.index.max(), freq="h")
    hh = hh.reindex(full)
    complete = complete.reindex(full).fillna(False).to_numpy()
    loc = full.tz_localize("UTC").tz_convert(V.TZ)
    g = pd.DataFrame(index=full)
    g["date"] = loc.tz_localize(None).normalize()
    hr = loc.hour.to_numpy() + 0.5
    X = {"const": np.ones(len(full))}
    for k in (1, 2, 3):
        X[f"hr_sin{k}"] = np.sin(2 * np.pi * k * hr / 24)
        X[f"hr_cos{k}"] = np.cos(2 * np.pi * k * hr / 24)
    hol = hh["cal_holiday"].fillna(0).to_numpy() == 1
    dow = loc.dayofweek.to_numpy()
    X["saturday"] = ((dow == 5) & ~hol).astype(float)
    X["sun_holiday"] = ((dow == 6) | hol).astype(float)
    reg = hh["regime_imb_resolution"].ffill().bfill().astype(str).to_numpy()
    X["reg_hourly_dominant"] = (reg == "hourly_dominant").astype(float)
    X["reg_qh"] = (reg == "qh").astype(float)
    xcols = list(X)
    for c, v in X.items():
        g[c] = v
    for side, f in FLAGS.items():
        v = np.where(complete, np.nan_to_num(hh[f].to_numpy(float), nan=0.0), np.nan)
        g[f"N_{side}"] = np.nan_to_num(v, nan=0.0)
        g[f"ok_{side}"] = np.isfinite(v)
        h = pd.Series(hh[HIST30[side]].to_numpy(float), index=full).ffill()
        h = h.fillna(pd.Series(np.nan_to_num(v), index=full).expanding().mean().shift(1) / 4).fillna(0.01)
        g[f"lh30_{side}"] = np.log(h.clip(lower=0) + 1e-3)
    return g, xcols


# ------------------------------------------------------------------ model
def _sig(x):
    return 1 / (1 + np.exp(-x))


def decay(h, rng_):
    lo, hi = rng_
    s = _sig(h)
    hl = lo * np.exp(np.log(hi / lo) * s)
    d = np.exp(-np.log(2) / hl)
    dd = d * np.log(2) / hl * np.log(hi / lo) * s * (1 - s)
    return d, hl, dd


def filt(N, d):
    """S_t = d S_t-1 + N_t-1 (S_0 = 0) and G = dS/dd."""
    from scipy.signal import lfilter
    S = lfilter([0.0, 1.0], [1.0, -d], N)
    G = lfilter([0.0, 1.0], [1.0, -d], S)
    return S, G


class Side:
    """Hawkes for one side. theta = [b (p), log a_fast, h_fast, log a_slow, h_slow, log a_cross, h_cross]."""

    def __init__(self, X, N_own, N_oth, ok, excite=True):
        self.X, self.N, self.Nc, self.ok, self.excite = X, N_own, N_oth, ok, excite
        self.p = X.shape[1]

    def unpack(self, th):
        b = th[: self.p]
        if not self.excite:
            return b, []
        ks = []
        for i, (name, src) in enumerate((("fast", "own"), ("slow", "own"), ("cross", "oth"))):
            la, h = th[self.p + 2 * i], th[self.p + 2 * i + 1]
            ks.append((name, src, la, h))
        return b, ks

    def lam(self, th, upto=None, need_grad=False):
        b, ks = self.unpack(th)
        sl = slice(0, upto)
        mu = np.exp(np.clip(self.X[sl] @ b, -30, 5))
        lam = mu.copy()
        parts = []
        for name, src, la, h in ks:
            d, hl, dd = decay(h, KERNELS[name])
            S, G = filt((self.N if src == "own" else self.Nc)[sl], d)
            a = np.exp(la)
            lam += a * S
            parts.append((a, S, G, dd))
        return mu, lam, parts

    def nll(self, th, upto, mask):
        mu, lam, parts = self.lam(th, upto)
        lam = np.maximum(lam, 1e-12)
        N = self.N[:upto][mask]
        L = lam[mask]
        ll = (N * np.log(-np.expm1(-L)) - (1 - N) * L).sum()
        w = np.zeros_like(lam)
        w[mask] = np.where(N > 0, 1.0 / np.expm1(L), 0.0) - (1 - N)
        g = [(self.X[:upto] * (w * mu)[:, None]).sum(0)]
        for a, S, G, dd in parts:
            g.append([(w * a * S).sum(), (w * a * G).sum() * dd])
        grad = np.concatenate([g[0], *[np.asarray(x) for x in g[1:]]])
        return -ll, -grad

    def fit(self, upto, mask, th0=None):
        from scipy.optimize import minimize
        rate = max(self.N[:upto][mask].mean(), 1e-5)
        if th0 is None:
            th0 = np.zeros(self.p + (6 if self.excite else 0))
            th0[0] = np.log(rate)
            if self.excite:
                th0[self.p:] = [np.log(0.3), 0.0, np.log(0.01), 0.0, np.log(0.001), 0.0]
        bnds = [(-12, 12)] * self.p + ([(-15, 2), (-6, 6)] * 3 if self.excite else [])
        r = minimize(self.nll, th0, args=(upto, mask), jac=True, method="L-BFGS-B", bounds=bnds,
                     options={"maxiter": 2000})
        return r.x, -r.fun, r


def hessian_se(side: Side, th, upto, mask):
    k = len(th)
    H = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k); e[i] = 1e-4
        H[:, i] = (side.nll(th + e, upto, mask)[1] - side.nll(th - e, upto, mask)[1]) / 2e-4
    H = (H + H.T) / 2
    try:
        cov = np.linalg.pinv(H)
        return np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:  # noqa: BLE001
        return np.full(k, np.nan)


def describe(side_name, sd: Side, th, se, xcols, upto, mask, ll, ll_bg, label):
    b, ks = sd.unpack(th)
    rows = [{"fit": label, "side": side_name, "param": c, "estimate": b[i], "se": se[i]} for i, c in enumerate(xcols)]
    summ = {"fit": label, "side": side_name, "n_hours": int(mask.sum()), "events": int(sd.N[:upto][mask].sum()),
            "loglik": ll, "loglik_bg": ll_bg, "LR_vs_bg": 2 * (ll - ll_bg), "AIC": -2 * ll + 2 * len(th),
            "AIC_bg": -2 * ll_bg + 2 * sd.p}
    nself = 0.0
    for j, (name, src, la, h) in enumerate(ks):
        d, hl, _ = decay(h, KERNELS[name])
        a = np.exp(la)
        n = a / (1 - d)
        i = sd.p + 2 * j
        rows += [{"fit": label, "side": side_name, "param": f"log_a_{name}", "estimate": la, "se": se[i]},
                 {"fit": label, "side": side_name, "param": f"h_{name}", "estimate": h, "se": se[i + 1]}]
        summ[f"a_{name}"] = a
        summ[f"halflife_h_{name}"] = hl
        summ[f"n_{name}"] = n
        if src == "own":
            nself += n
    summ["n_self"] = nself
    mu, lam, _ = sd.lam(th, upto)
    ev = mask & (sd.N[:upto] > 0)
    summ["triggered_share"] = float(((lam[ev] - mu[ev]) / lam[ev]).mean()) if ev.any() else np.nan
    summ["mean_mu"] = float(mu[mask].mean())
    return rows, summ


# ------------------------------------------------------------------ forecasting
def states_at(g, th_all, sides_obj, upto):
    """Filtered kernel states (events up to t-1) for t in [0, upto) for both sides, with the fitted decays."""
    st = {}
    for s, sd in sides_obj.items():
        b, ks = sd.unpack(th_all[s])
        for name, src, la, h in ks:
            d, _, _ = decay(h, KERNELS[name])
            S, _ = filt((sd.N if src == "own" else sd.Nc)[:upto], d)
            st[(s, name)] = (S, d, np.exp(la))
    return st


def mu_all(sd: Side, th):
    return np.exp(np.clip(sd.X @ th[: sd.p], -30, 5))


def simulate(o, hour_offsets, mus, st, n_sims, rng):
    """o: (m,) first unknown QH index per origin; hour_offsets: list of (origin row, offset j, key) target hours.
    Simulates both sides jointly from o with Bernoulli draws; returns dict side -> key -> P(spike in the target hour)
    (Rao-Blackwellised: 1 - mean over paths of exp(-lambda) at the target hour)."""
    m = len(o)
    sides = list(mus)
    S = {k: np.repeat(v[0][o][:, None], n_sims, axis=1) for k, v in st.items()}
    dk = {k: v[1] for k, v in st.items()}
    ak = {k: v[2] for k, v in st.items()}
    other = {"short": "long", "long": "short"}
    jmax = max(j for _, j, _ in hour_offsets)
    by_j: dict[int, list] = {}
    for r, j, key in hour_offsets:
        by_j.setdefault(j, []).append((r, key))
    res = {s: {} for s in sides}
    for j in range(jmax + 1):
        idx = o + j
        if j in by_j:
            rr = np.array([r for r, _ in by_j[j]])
            keys = [k for _, k in by_j[j]]
            for s in sides:
                lam = mus[s][idx[rr]][:, None] + sum(ak[(s, kn)] * S[(s, kn)][rr] for kn in ("fast", "slow", "cross"))
                surv = np.exp(-lam).mean(axis=1)
                for kk, v in zip(keys, 1 - surv):
                    res[s][kk] = v
        if j == jmax:
            break
        draws = {}
        for s in sides:
            lam = mus[s][idx][:, None] + sum(ak[(s, kn)] * S[(s, kn)] for kn in ("fast", "slow", "cross"))
            draws[s] = (rng.random((m, n_sims)) < -np.expm1(-lam)).astype(float)
        for s in sides:
            for kn in ("fast", "slow"):
                S[(s, kn)] = dk[(s, kn)] * S[(s, kn)] + draws[s]
            S[(s, "cross")] = dk[(s, "cross")] * S[(s, "cross")] + draws[other[s]]
    return res


# ------------------------------------------------------------------ scoring (as rq3b_2)
def scores(y, p) -> dict:
    from rq3a_2_logit import auc, avg_precision
    q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).to_numpy()
    n = np.bincount(q, minlength=10).astype(float)
    mp = np.bincount(q, weights=p, minlength=10) / np.maximum(n, 1)
    my = np.bincount(q, weights=y, minlength=10) / np.maximum(n, 1)
    k = max(1, int(round(0.05 * len(y))))
    top = np.argsort(-p)[:k]
    return {"n": len(y), "pos": int(y.sum()), "base_rate": y.mean(), "mean_pred": p.mean(),
            "Brier": float(((p - y) ** 2).mean()), "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "PR_AUC": avg_precision(y, p), "ROC_AUC": auc(y, p), "ECE": float((n * np.abs(mp - my)).sum() / n.sum()),
            "capture_top5": y[top].sum() / max(y.sum(), 1)}


def boot(diff, days, rng, n=1000):
    ud, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(ud))
    c = np.bincount(inv, minlength=len(ud))
    draws = np.array([s[i].sum() / c[i].sum() for i in (rng.integers(0, len(ud), len(ud)) for _ in range(n))])
    return s.sum() / c.sum(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_intensity(g, sides_obj, th, f, start, days=10):
    plt = _plt()
    t = g.index
    i0 = int(np.searchsorted(t, pd.Timestamp(start)))
    i1 = min(i0 + 24 * days, len(t))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, (s, sd) in zip(axes, sides_obj.items()):
        mu, lam, _ = sd.lam(th[s], i1)
        ax.plot(t[i0:i1], -np.expm1(-lam[i0:i1]), lw=0.8, label="P(spike hour), Hawkes")
        ax.plot(t[i0:i1], -np.expm1(-mu[i0:i1]), lw=0.8, label="background only")
        ev = np.flatnonzero(sd.N[i0:i1] > 0)
        ax.plot(t[i0:i1][ev], np.full(len(ev), 1.02), "|", color="r", label="spike hour")
        ax.set_ylabel(s); ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Hawkes intensity (in-sample, full fit) {start} + {days} days", fontsize=10)
    fig.tight_layout(); fig.savefig(f, dpi=120); plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000, help="simulated paths per d1 origin")
    ap.add_argument("--sims-h1", type=int, default=400, help="simulated paths per h1 origin")
    ap.add_argument("--step-months", type=int, default=6)
    ap.add_argument("--no-insample", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.quick:
        a.sims, a.sims_h1, a.step_months = 200, 100, 12
    OUT = HERE / "Output" / ("rq3b_4" + a.tag)
    for sub in ("plots", "diagnostics"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    RU.set_output(OUT)
    t0 = time.time()
    g, xcols = load()
    T = len(g)
    log(f"RQ3 Hawkes: {T:,} hours {g.index.min()} -> {g.index.max()} (UTC grid); background {len(xcols) + 1} terms; "
        f"spike hours short {int(g.N_short.sum()):,}, long {int(g.N_long.sum()):,}")

    def make(side, excite=True):
        oth = "long" if side == "short" else "short"
        X = np.column_stack([g[xcols].to_numpy(float), g[f"lh30_{side}"].to_numpy(float)])
        return Side(X, g[f"N_{side}"].to_numpy(float), g[f"N_{oth}"].to_numpy(float), g[f"ok_{side}"].to_numpy(),
                    excite)
    allx = xcols + ["log_spikeshare_30d"]

    # ---------- A: in-sample
    th_full = {}
    if not a.no_insample:
        log("\n== A  in-sample fits (whole sample and per period)")
        prow, summ = [], []
        fits = [("full", g.index.min(), g.index.max() + pd.Timedelta("15min"))] + \
               [(n, pd.Timestamp(x), pd.Timestamp(y)) for n, x, y in PERIODS]
        for label, s0, s1 in fits:
            for side in FLAGS:
                sd, bg = make(side), make(side, excite=False)
                upto = int(np.searchsorted(g.index, s1))
                m = sd.ok[:upto] & (g.index[:upto] >= s0)
                th, ll, r = sd.fit(upto, m)
                thb, llb, _ = bg.fit(upto, m)
                se = hessian_se(sd, th, upto, m)
                rows, s_ = describe(side, sd, th, se, allx, upto, m, ll, llb, label)
                s_["converged"] = bool(r.success)
                prow += rows
                summ.append(s_)
                if label == "full":
                    th_full[side] = th
                log(f"  {label:10s} {side:5s}: events {s_['events']:,}, n_self {s_['n_self']:.3f} (fast {s_['n_fast']:.3f}, "
                    f"half-life {s_['halflife_h_fast']:.2f} h; slow {s_['n_slow']:.3f}, {s_['halflife_h_slow']:.1f} h), "
                    f"cross {s_['n_cross']:.4f} ({s_['halflife_h_cross']:.1f} h), triggered {s_['triggered_share']:.1%}, "
                    f"LR vs bg {s_['LR_vs_bg']:,.0f}  [{time.time() - t0:.0f}s]")
        PR, SM = pd.DataFrame(prow), pd.DataFrame(summ)
        PR.to_csv(OUT / "params_insample.csv", index=False)
        SM.to_csv(OUT / "insample_summary.csv", index=False)
        log("\n" + SM.drop(columns=["loglik", "loglik_bg"]).to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        log("\nbackground coefficients (full sample), estimate (se):")
        t = PR[PR.fit == "full"].pivot(index="param", columns="side", values=["estimate", "se"])
        log(t.to_string(float_format=lambda v: f"{v:+.3f}"))
        sides_full = {s: make(s) for s in FLAGS}
        for st_ in ("2021-11-15", "2024-06-10", "2025-06-10"):
            try:
                plot_intensity(g, sides_full, th_full, OUT / "plots" / f"intensity_{st_}.png", st_)
            except Exception as e:  # noqa: BLE001
                log(f"  (plot skipped: {e})")

    # ---------- B: rolling-origin forecasts
    log(f"\n== B  rolling-origin forecasts (refit every {a.step_months} months, sims d1 {a.sims}, h1 {a.sims_h1})")
    hours = g.index[g["ok_short"].to_numpy() & g["ok_long"].to_numpy()]
    hour_start_idx = np.searchsorted(g.index, hours)
    hour_date = g["date"].to_numpy()[hour_start_idx]
    yh = {s: g[f"N_{s}"].to_numpy()[hour_start_idx] for s in FLAGS}
    folds, f0 = [], TEST_START
    while f0 <= g.index.max():
        folds.append((f0, f0 + pd.DateOffset(months=a.step_months)))
        f0 = f0 + pd.DateOffset(months=a.step_months)
    rng = np.random.default_rng(20261016)
    pred = {k: np.full((2, len(hours)), np.nan) for k in ("hawkes_d1", "hawkes_h1", "bg_d1", "exc_d1", "exc_h1")}
    sidx = {"short": 0, "long": 1}
    fold_rows = []
    for f0, f1 in folds:
        tf = time.time()
        upto = int(np.searchsorted(g.index, f0 - EMBARGO))
        sides_obj, th, thb = {}, {}, {}
        for side in FLAGS:
            sd, bg = make(side), make(side, excite=False)
            m = sd.ok[:upto]
            th[side], ll, r = sd.fit(upto, m)       # cold start every fold (a warm start kept a_cross at its bound)
            thb[side], llb, _ = bg.fit(upto, m)
            sides_obj[side] = sd
            _, s_ = describe(side, sd, th[side], np.full(len(th[side]), np.nan), allx, upto, m, ll, llb, str(f0.date()))
            s_["converged"] = bool(r.success)
            fold_rows.append(s_)
        # test hours of this fold (by local date of the hour, as rq3b_1 uses hk in UTC)
        hsel = np.flatnonzero((hours >= f0) & (hours < f1))
        if len(hsel) == 0:
            continue
        last = int(hour_start_idx[hsel].max())
        st = states_at(g, th, sides_obj, min(last + 1, T))
        mus = {s: mu_all(sides_obj[s], th[s]) for s in FLAGS}
        mub = {s: mu_all(make(s, excite=False), thb[s]) for s in FLAGS}
        # h1: origin = hour start - 1 h; first unknown hour = target hour - 2
        h0 = hour_start_idx[hsel]
        o_h1 = h0 - GAP_H1
        ok = o_h1 >= 0
        for c0 in range(0, ok.sum(), 3000):
            sel = np.flatnonzero(ok)[c0:c0 + 3000]
            res = simulate(o_h1[sel], [(i, GAP_H1, i) for i in range(len(sel))], mus,
                           {k: v for k, v in st.items()}, a.sims_h1, rng)
            for s in FLAGS:
                pred["hawkes_h1"][sidx[s], hsel[sel]] = [res[s][i] for i in range(len(sel))]
        # d1: origin = D-1 18:00 local; first unknown hour starts 1 h before the origin (17:00 local)
        dates = hour_date[hsel]
        ud = np.unique(dates)
        org = (pd.DatetimeIndex(ud) - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize(V.TZ, ambiguous=True,
                                                                                                  nonexistent="shift_forward")
        org = org.tz_convert("UTC").tz_localize(None).floor("h") - pd.Timedelta(hours=1)
        o_d1 = np.searchsorted(g.index, org)
        day_row = {d: i for i, d in enumerate(ud)}
        offs = [(day_row[d], int(h0[i] - o_d1[day_row[d]]), i) for i, d in enumerate(dates)]
        res = simulate(o_d1, offs, mus, st, a.sims, rng)
        for s in FLAGS:
            pred["hawkes_d1"][sidx[s], hsel] = [res[s][i] for i in range(len(hsel))]
            # background only: deterministic
            pred["bg_d1"][sidx[s], hsel] = -np.expm1(-mub[s][h0])
            ex = sum(st[(s, k)][2] * st[(s, k)][0] for k in ("fast", "slow", "cross"))
            pred["exc_h1"][sidx[s], hsel] = np.where(ok, ex[np.clip(o_h1, 0, None)], np.nan)
            pred["exc_d1"][sidx[s], hsel] = ex[o_d1[[day_row[d] for d in dates]]]
        log(f"  fold {f0.date()} -> {f1.date()}: train {upto:,} h, test {len(hsel):,} h; n_self short "
            f"{fold_rows[-2]['n_self']:.3f}, long {fold_rows[-1]['n_self']:.3f}  [{time.time() - tf:.0f}s fold, "
            f"{time.time() - t0:.0f}s total]")
    FR = pd.DataFrame(fold_rows)
    FR.to_csv(OUT / "params_folds.csv", index=False)

    rows = []
    for s in FLAGS:
        d = pd.DataFrame({"hk": hours, "side": s, "y_own": yh[s], "date": hour_date})
        for k in pred:
            d[k] = pred[k][sidx[s]]
        rows.append(d[d.hk >= TEST_START])
    H = pd.concat(rows, ignore_index=True)
    H[["hk", "side", "exc_d1", "exc_h1"]].to_csv(OUT / "hawkes_features.csv", index=False)
    for c in ("hawkes_d1", "hawkes_h1", "bg_d1"):
        H[c] = H[c].clip(1e-5, 1 - 1e-5)

    # ---------- merge with rq3b_1 (main spec: logistic Platt from rq3b_1, LightGBM raw from rq3b_1_raw)
    base = pd.read_csv(HERE / "Output" / "rq3b_1" / "predictions.csv", parse_dates=["hk", "date"])
    raw = pd.read_csv(HERE / "Output" / "rq3b_1_raw" / "predictions.csv", parse_dates=["hk"])
    gb = [c for c in raw.columns if c.startswith("gbm")]
    keep = ["hk", "side", "y", "date", *BASELINES, "lr_d1", "lr_h1"]
    P = base[keep].merge(raw[["hk", "side", *gb]], on=["hk", "side"], how="left")
    P = P.merge(H.drop(columns=["date", "exc_d1", "exc_h1"]), on=["hk", "side"], how="inner")
    agree = (P["y"] == P["y_own"]).mean()
    log(f"\n  merged with rq3b_1: {len(P):,} rows (rq3b_1 has {len(base):,}); target agreement {agree:.4%}")
    P = P.drop(columns="y_own").dropna(subset=["hawkes_d1", "hawkes_h1", "bg_d1"])
    P.to_csv(OUT / "predictions.csv", index=False)
    models = [*BASELINES, "lr_d1", "lr_h1", *gb, "bg_d1", "hawkes_d1", "hawkes_h1"]
    best = {}
    out, bt = [], []
    rngb = np.random.default_rng(20261010)
    for s in FLAGS:
        d = P[P.side == s]
        bb = {m: ((d[m] - d["y"]) ** 2).mean() for m in BASELINES}
        best[s] = min(bb, key=bb.get)
        yr = d["date"].dt.year
        for pname, dd in (("all", d), ("2021_22", d[yr <= 2022]), ("from_2023", d[yr >= 2023]),
                          ("post_cap", d[d["date"] >= CAP])):
            ref = ((dd[best[s]] - dd["y"]) ** 2).mean()
            for m in models:
                sc = scores(dd["y"].to_numpy(float), dd[m].to_numpy(float))
                out.append({"side": s, "period": pname, "model": m, **sc, "best_baseline": best[s],
                            "Brier_skill_vs_best_baseline": 1 - sc["Brier"] / ref})
            for m in ("bg_d1", "hawkes_d1", "hawkes_h1", "lr_h1", "gbm_h1"):
                diff = ((dd[best[s]] - dd["y"]) ** 2 - (dd[m] - dd["y"]) ** 2).to_numpy()
                e, lo, hi = boot(diff, dd["date"].to_numpy(), rngb)
                bt.append({"side": s, "period": pname, "model": m, "vs": best[s], "dBrier": e, "ci_lo": lo, "ci_hi": hi})
            for m, r_ in (("hawkes_d1", "bg_d1"), ("hawkes_h1", "persist_h1"), ("hawkes_d1", "persist_d1")):
                diff = ((dd[r_] - dd["y"]) ** 2 - (dd[m] - dd["y"]) ** 2).to_numpy()
                e, lo, hi = boot(diff, dd["date"].to_numpy(), rngb)
                bt.append({"side": s, "period": pname, "model": m, "vs": r_, "dBrier": e, "ci_lo": lo, "ci_hi": hi})
    R, BT = pd.DataFrame(out), pd.DataFrame(bt)
    R.to_csv(OUT / "metrics.csv", index=False)
    BT.to_csv(OUT / "brier_gain_bootstrap.csv", index=False)
    for s in FLAGS:
        for pname in ("all", "2021_22", "from_2023", "post_cap"):
            t = R[(R.side == s) & (R.period == pname)].set_index("model")[
                ["n", "pos", "Brier", "Brier_skill_vs_best_baseline", "logloss", "PR_AUC", "ROC_AUC", "mean_pred",
                 "ECE", "capture_top5"]]
            log(f"\n-- {s}, {pname}; best baseline {best[s]}")
            log(t.to_string(float_format=lambda v: f"{v:,.4f}"))
    log("\n-- Brier gain (positive = model better), day-block bootstrap 95 % CI (x1e-4)")
    bt_show = BT.copy()
    for c in ("dBrier", "ci_lo", "ci_hi"):
        bt_show[c] = bt_show[c] * 1e4
    log(bt_show.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    log("\n-- fold parameters (branching ratios, half-lives in hours)")
    log(FR[["fit", "side", "events", "n_self", "n_fast", "halflife_h_fast", "n_slow", "halflife_h_slow", "n_cross",
            "halflife_h_cross", "triggered_share", "LR_vs_bg", "converged"]].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    try:
        import ml_diagnostics as D
        D.classifier_diagnostics(P, models, OUT, y_col="y", time_col="hk", group_col="side", reference=best,
                                 title=f"RQ3 Hawkes{a.tag}")
    except Exception as e:  # noqa: BLE001
        log(f"  (diagnostics skipped: {e})")
    for s in FLAGS:
        t = R[(R.side == s) & (R.period == "from_2023")].set_index("model")["Brier_skill_vs_best_baseline"]
        RU.summary(f"{s} from 2023: hawkes_h1 {t['hawkes_h1']:+.3f}, gbm_h1 {t.get('gbm_h1', np.nan):+.3f}")
    (OUT / "report_rq3b_4.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    RU.run(main, "RQ3 Hawkes clustering")
