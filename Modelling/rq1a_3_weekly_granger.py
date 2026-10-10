"""RQ1a step 3 (todo 9.1c): weekly products - Granger causality + differenced regression.

Weekly blocks (duration 168 h: aFRR / mFRR weekly, RQ1a window) of the ex-post views.
Series: log(price_settle_ch), first-differenced (weekly changes; avoids the 2021-23 trend,
no unit-root test needed beyond that). Drivers: numeric feature columns >= MIN_VALID valid,
outcome columns (offered volumes, ...) excluded, first-differenced as well.

 A. Granger F-tests, x -> y and y -> x, lags 1, 2, 4 (own F-test, numpy only):
    restricted y_t ~ const + y lags, unrestricted + x lags.  Primary lag = 2 (weeks);
    Benjamini-Hochberg q-values over all drivers of a series (x -> y, lag 2).
 B. Contemporaneous differenced regression: dy ~ dx, forward selection by BIC from the
    top N_REG candidates by |corr|, max 6 regressors, Newey-West SE (4 lags).

    python Modelling/rq1a_3_weekly_granger.py
Output: Modelling/Output/rq1a_3/granger_<product>.csv, reg_weekly.csv, report_rq1a_3.txt
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Clean")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import views as V  # noqa: E402
from rq1a_1_correlations import kind_of  # noqa: E402
from rq1a_2_regression import ols_nw, bic  # noqa: E402

MIN_VALID, MIN_N, LAGS, N_REG = 0.85, 80, (1, 2, 4), 30


# ---- F distribution survival function (regularised incomplete beta, Lentz) ----
def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-300, a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny); h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        de = d * c; h *= de
        if abs(de - 1) < 3e-12:
            break
    return h


def _betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def f_sf(F, d1, d2):
    return _betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * F)) if F > 0 else 1.0


def granger(y: pd.Series, x: pd.Series, L: int):
    d = {"y": y, "x": x}
    for l in range(1, L + 1):
        d[f"y{l}"] = y.shift(l); d[f"x{l}"] = x.shift(l)
    D = pd.DataFrame(d).dropna()
    n = len(D)
    if n < MIN_N or D.x.std() == 0:
        return np.nan, n
    Ar = np.column_stack([np.ones(n)] + [D[f"y{l}"] for l in range(1, L + 1)])
    Au = np.column_stack([Ar] + [D[f"x{l}"] for l in range(1, L + 1)])
    rss = lambda A: ((D.y.values - A @ np.linalg.lstsq(A, D.y.values, rcond=None)[0]) ** 2).sum()
    rr, ru = rss(Ar), rss(Au)
    d2 = n - Au.shape[1]
    if d2 <= 0 or ru <= 0:
        return np.nan, n
    return f_sf(((rr - ru) / L) / (ru / d2), L, d2), n


def bh(p: pd.Series) -> pd.Series:
    p = p.dropna().sort_values(); m = len(p)
    q = (p * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["afrr", "mfrr"])
    ap.add_argument("--out", default=str(ROOT / "Modelling" / "Output" / "rq1a_3"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep, regs = [], []
    for prod in a.products:
        df = V.load(prod, kind="expost", rq="RQ1a", impute=False)
        dic = V.dictionary(prod, "expost")
        df = df[df["duration_h"] >= 168]
        feats = [c for c in dic.index[dic["role"] == "feature"] if c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c]) and kind_of(c) == "driver"]
        res = []
        for (d, p), s in df.groupby(["direction", "procurement"], dropna=False):
            name = f"{prod}|{d}|{p}"
            s = s.sort_values("block_start_local").set_index("block_start_local")
            s = s[~s.index.duplicated()]
            y = np.log(s["price_settle_ch"].where(s["price_settle_ch"] > 0)).diff()
            if y.notna().sum() < MIN_N:
                rep.append(f"\n== {name}: {y.notna().sum()} weekly changes, too few =="); continue
            fe = [c for c in feats if s[c].notna().mean() >= MIN_VALID and s[c].nunique() > 10]
            dx = s[fe].diff()
            rows = []
            for c in fe:
                r = dict(feature=c)
                for L in LAGS:
                    r[f"p_x2y_L{L}"], r["n"] = granger(y, dx[c], L)
                r["p_y2x_L2"], _ = granger(dx[c], y, 2)
                r["corr0"] = y.corr(dx[c])
                rows.append(r)
            g = pd.DataFrame(rows)
            g["q_x2y_L2"] = bh(g["p_x2y_L2"]); g["q_y2x_L2"] = bh(g["p_y2x_L2"])
            g.insert(0, "series", name); res.append(g)
            top = g.sort_values("p_x2y_L2").head(12)
            rep.append(f"\n== {name}: {int(y.notna().sum())} weekly changes, {len(fe)} drivers; "
                       f"x->y significant at q<0.05 (lag 2): {(g.q_x2y_L2 < .05).sum()}; y->x: {(g.q_y2x_L2 < .05).sum()} ==")
            rep.append(top[["feature", "n", "p_x2y_L1", "p_x2y_L2", "p_x2y_L4", "q_x2y_L2", "p_y2x_L2", "corr0"]]
                       .round(4).to_string(index=False))
            # B. differenced regression
            cand = g.reindex(g.corr0.abs().sort_values(ascending=False).index).feature
            cand = [c for c in cand if dx[c].notna().mean() >= 0.95][:N_REG]
            ok = y.notna() & dx[cand].notna().all(axis=1)
            if ok.sum() >= MIN_N and cand:
                yv = y[ok].values - y[ok].mean(); Z = dx.loc[ok, cand]; Z = (Z - Z.mean()) / Z.std()
                Z = Z.loc[:, Z.std() > 0]
                sel, best = [], len(yv) * np.log(yv @ yv / len(yv))
                while len(sel) < 6:
                    t = None
                    for c in Z.columns:
                        if c in sel: continue
                        A = Z[sel + [c]].values; b = np.linalg.lstsq(A, yv, rcond=None)[0]
                        bc = bic(yv - A @ b, len(sel) + 1)
                        if bc < best - 1e-9 and (t is None or bc < t[1]): t = (c, bc)
                    if t is None: break
                    sel.append(t[0]); best = t[1]
                if sel:
                    A = Z[sel].values; b, se, e = ols_nw(A, yv, lags=4)
                    r2 = 1 - (e @ e) / (yv @ yv)
                    t_ = pd.DataFrame(dict(series=name, feature=sel, coef=b, se=se, t=b / se, n=len(yv), r2=r2))
                    regs.append(t_)
                    rep.append(f"-- differenced regression {name}: n={len(yv)} R2={r2:.3f}")
                    rep.append(t_[["feature", "coef", "se", "t"]].round(3).to_string(index=False))
        if res:
            pd.concat(res).to_csv(out / f"granger_{prod}.csv", index=False)
    if regs:
        pd.concat(regs).to_csv(out / "reg_weekly.csv", index=False)
    (out / "report_rq1a_3.txt").write_text("\n".join(rep)); print("\n".join(rep)); print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
