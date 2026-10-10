"""RQ1b (todo 9.1, second half): do the price drivers change at the market-design breaks?

For every break and every product it applies to, the sample is cut to +-WINDOW months around
the break (the calendar month containing the break is dropped) and split pre / post.
Per direction (aFRR split: directions pooled, FE include direction):
  1. y = log(price_settle_ch); fixed effects = procurement x month x delivery hour (demeaning).
     => within-month relations only; LEVEL shifts at the break are not identified here
     (see EDA 3 for level breaks vs placebo).
  2. candidates: driver columns (no offered volumes etc.) with >= MIN_VALID valid on BOTH sides;
     top N_CAND by |corr| with y (pooled, within FE), near-duplicates removed (|r| > 0.8);
     forward selection by BIC on the pooled sample, max MAX_VARS regressors.
  3. interaction model y = sum b_k x_k + sum d_k x_k*post, Newey-West SE (6 lags).
     Reported: coef_pre = b, coef_post = b + d, t / p of d, joint Wald test d = 0 (F form).
Regressors are in within-FE SD units (pooled), so coefficients are comparable pre / post.

Added 10 Oct (v2):
  - weekly mode: 168 h blocks tested separately (FE = quarter, <= 4 regressors, min 60 blocks per side,
    NW 2 lags) - needed for the aFRR breaks (aFRR only has weekly blocks before Sep 2025)
  - placebo breaks at 12 and 24 months BEFORE each real break (using only data before the real break
    month): the real joint Wald statistic is compared with the placebo ones ("exceeds placebo")
  - untestable breaks are labelled data-limited (views start ~2021; FCR / aFRR have no sample before)

    python Modelling/rq1b_regimes.py [--products fcr afrr mfrr] [--window 24] [--placebo 12 24]
Output: Modelling/Output/rq1b/regimes.csv, report_rq1b.txt
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Clean")); sys.path.insert(0, str(HERE))
import views as V  # noqa: E402
from rq1a_1_correlations import kind_of  # noqa: E402
from rq1a_3_weekly_granger import f_sf  # noqa: E402

TZ = "Europe/Zurich"
# key, date, products, pool_directions
BREAKS = [
    ("de_zone_split", "2018-10-01", ["fcr", "afrr", "mfrr"], False),
    ("afrr_split",    "2018-06-11", ["afrr"], True),
    ("fcr_daily",     "2019-07-01", ["fcr"], False),
    ("fcr_4h",        "2020-07-01", ["fcr"], False),
    ("afrr_fallback", "2024-02-09", ["afrr"], False),
    ("mfrr_merged",   "2025-09-29", ["mfrr"], True),
    ("afrr_daily",    "2025-09-30", ["afrr"], False),
]
MIN_VALID, N_CAND, MAX_CORR = 0.85, 40, 0.8
MODES = {  # min blocks per side, max regressors, NW lags
    '4h':     dict(min_side=150, max_vars=8, nw=6),
    'weekly': dict(min_side=60,  max_vars=4, nw=2),
}


def nw_cov(X, e, lags=6):
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lags + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (lags + 1)) * (G + G.T)
    return XtXi @ S @ XtXi * n / max(n - k, 1)


def p_norm(t):
    return math.erfc(abs(t) / math.sqrt(2))


def bic(e, k):
    n = len(e)
    return n * np.log((e @ e) / n) + k * math.log(n)


def analyse(s: pd.DataFrame, bdate: pd.Timestamp, pool: bool, feats: list[str], mode: str = '4h'):
    MIN_SIDE, MAX_VARS, NW = (MODES[mode][k] for k in ('min_side', 'max_vars', 'nw'))
    weekly = mode == 'weekly'
    s = s.sort_values("block_start_local")
    post = (s["block_start_local"] >= bdate).astype(float)
    y = np.log(s["price_settle_ch"].where(s["price_settle_ch"] > 0))
    loc = s["block_start_local"]
    per = loc.dt.to_period("Q").astype(str) if weekly else (loc.dt.strftime("%Y-%m") + "_" + loc.dt.hour.astype(str))
    g = s["procurement"].astype(str) + (("_" + s["direction"].astype(str)) if pool else "") + "_" + per
    pre_ok = (post == 0)
    cand = [c for c in feats if c in s and s.loc[pre_ok, c].notna().mean() >= MIN_VALID
            and s.loc[~pre_ok, c].notna().mean() >= MIN_VALID and s[c].nunique() > 3]
    if not cand:
        return None, "no candidate with enough data on both sides"
    ok = y.notna() & s[cand].notna().all(axis=1)
    # drop candidates only if complete cases would collapse
    if ok.sum() < 2 * MIN_SIDE or (ok & ~pre_ok).sum() < MIN_SIDE or (ok & pre_ok).sum() < MIN_SIDE:
        keep = [c for c in cand if s[c].notna().mean() >= 0.95]
        ok = y.notna() & s[keep].notna().all(axis=1) if keep else ok
        cand = keep
        if not cand or (ok & ~pre_ok).sum() < MIN_SIDE or (ok & pre_ok).sum() < MIN_SIDE:
            return None, f"too few complete rows (pre {int((ok & pre_ok).sum())}, post {int((ok & ~pre_ok).sum())})"
    y, X, g, post = y[ok], s.loc[ok, cand], g[ok], post[ok]
    dem = lambda d: d - d.groupby(g, sort=False).transform("mean")
    Xd, yd = dem(X), dem(y.to_frame("y"))["y"]
    sd = Xd.std(); Xd = Xd.loc[:, sd > 1e-9]; Z = Xd / Xd.std()
    corr = Z.apply(lambda c: abs(c.corr(yd))).sort_values(ascending=False).head(N_CAND)
    Z = Z[corr.index]
    C = Z.corr().abs(); keep = []
    for c in Z.columns:
        if all(C.loc[c, k] <= MAX_CORR for k in keep): keep.append(c)
    Z = Z[keep]; yv = yd.values
    sel, best = [], len(yv) * math.log(yv @ yv / len(yv))
    while len(sel) < MAX_VARS:
        t = None
        for c in Z.columns:
            if c in sel: continue
            A = Z[sel + [c]].values; b = np.linalg.lstsq(A, yv, rcond=None)[0]
            bc = bic(yv - A @ b, len(sel) + 1)
            if bc < best - 1e-9 and (t is None or bc < t[1]): t = (c, bc)
        if t is None: break
        sel.append(t[0]); best = t[1]
    if not sel:
        return None, "no regressor selected"
    A0 = Z[sel].values; pv = post.values[:, None]
    A = np.hstack([A0, A0 * pv])
    # post-specific demeaning of the interaction terms is not needed: FE are group means of both
    # sides; the interaction columns are demeaned by group as well to keep FE clean
    Ad = pd.DataFrame(A).groupby(g.values, sort=False).transform(lambda c: c - c.mean()).values
    b = np.linalg.lstsq(Ad, yv, rcond=None)[0]; e = yv - Ad @ b
    cov = nw_cov(Ad, e, NW); k = len(sel)
    se = np.sqrt(np.diag(cov))
    d, Vd = b[k:], cov[k:, k:]
    W = float(d @ np.linalg.pinv(Vd) @ d)
    n = len(yv); joint_p = f_sf(W / k, k, max(n - 2 * k, 1))
    r2 = lambda m: (lambda yy, ee: 1 - (ee @ ee) / (yy @ yy))(yv[m], e[m]) if m.sum() > 5 else np.nan
    mpre, mpost = (post.values == 0), (post.values == 1)
    rows = [dict(feature=f, coef_pre=b[i], coef_post=b[i] + b[k + i], diff=b[k + i], t_diff=b[k + i] / se[k + i],
                 p_diff=p_norm(b[k + i] / se[k + i])) for i, f in enumerate(sel)]
    meta = dict(n_pre=int(mpre.sum()), n_post=int(mpost.sum()), joint_p=joint_p, W=W / k, r2_pre=r2(mpre), r2_post=r2(mpost))
    return (pd.DataFrame(rows), meta), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["fcr", "afrr", "mfrr"])
    ap.add_argument("--window", type=int, default=24, help="months each side")
    ap.add_argument("--placebo", type=int, nargs="*", default=[12, 24], help="placebo offsets (months before the break)")
    ap.add_argument("--out", default=str(ROOT / "Modelling" / "Output" / "rq1b"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep, allrows = [], []
    for prod in a.products:
        df = V.load(prod, kind="expost", rq="RQ1b", impute=True)
        dic = V.dictionary(prod, "expost")
        feats = [c for c in dic.index[dic["role"] == "feature"] if c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c]) and kind_of(c) == "driver"]
        for mode in ("4h", "weekly"):
            dm = df[df["duration_h"] >= 168] if mode == "weekly" else df[df["duration_h"] < 168]
            if dm.empty: continue
            ms = MODES[mode]["min_side"]
            for key, date, prods, pool in BREAKS:
                if prod not in prods: continue
                b = pd.Timestamp(date, tz=TZ)

                def run(bd, data):
                    mstart = bd.replace(day=1); mend = mstart + pd.offsets.MonthBegin(1)
                    w = data[(data.block_start_local >= mstart - pd.DateOffset(months=a.window)) &
                             (data.block_start_local < mend + pd.DateOffset(months=a.window)) &
                             ~((data.block_start_local >= mstart) & (data.block_start_local < mend))]
                    groups = [("all", w)] if pool else list(w.groupby("direction", dropna=False))
                    out_ = []
                    for d, s in groups:
                        npre, npost = int((s.block_start_local < mstart).sum()), int((s.block_start_local >= mend).sum())
                        if min(npre, npost) < ms:
                            out_.append((d, None, f"data-limited (pre {npre}, post {npost} blocks, need {ms})")); continue
                        res, why = analyse(s, bd, pool, feats, mode)
                        out_.append((d, res, why))
                    return out_

                real = run(b, dm)
                realW = {}
                for d, res, why in real:
                    tag = f"{prod}|{mode}|{key}|{d}"
                    if res is None:
                        rep.append(f"\n== {tag}: not testable - {why} =="); continue
                    tab, m = res
                    realW[d] = m["W"]
                    tab.insert(0, "test", tag); tab["joint_p"] = m["joint_p"]
                    tab["n_pre"], tab["n_post"] = m["n_pre"], m["n_post"]; tab["placebo"] = ""
                    allrows.append(tab)
                    rep.append(f"\n== {tag}: pre {m['n_pre']:,} / post {m['n_post']:,} blocks, R2 pre {m['r2_pre']:.2f} post {m['r2_post']:.2f}, "
                               f"joint test p = {m['joint_p']:.4f} (F = {m['W']:.1f}); "
                               f"{int((tab.p_diff < .05).sum())}/{len(tab)} coefficients differ (p<0.05) ==")
                    rep.append(tab[["feature", "coef_pre", "coef_post", "diff", "t_diff", "p_diff"]].round(3).to_string(index=False))
                # placebo breaks: only data before the real break month
                if realW:
                    before = dm[dm.block_start_local < b.replace(day=1)]
                    for off in a.placebo:
                        pb = b - pd.DateOffset(months=off)
                        for d, res, why in run(pb, before):
                            if d not in realW: continue
                            if res is None:
                                rep.append(f"   placebo -{off}m [{d}]: not testable - {why}"); continue
                            tab, m = res
                            tab.insert(0, "test", f"{prod}|{mode}|{key}|{d}"); tab["placebo"] = f"-{off}m"
                            tab["joint_p"] = m["joint_p"]; tab["n_pre"], tab["n_post"] = m["n_pre"], m["n_post"]
                            allrows.append(tab)
                            verdict = "real break EXCEEDS placebo" if realW[d] > m["W"] else "placebo is as large or larger"
                            rep.append(f"   placebo -{off}m [{d}]: pre {m['n_pre']:,} / post {m['n_post']:,}, "
                                       f"joint p = {m['joint_p']:.4f} (F = {m['W']:.1f}) -> {verdict}")
    if allrows:
        pd.concat(allrows).to_csv(out / "regimes.csv", index=False)
    (out / "report_rq1b.txt").write_text("\n".join(rep)); print("\n".join(rep)); print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
