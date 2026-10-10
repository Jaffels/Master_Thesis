"""RQ1a step 2 (todo 9.1b): multivariate fixed-effects regressions on log prices.

Per series (product x direction x procurement, 4h blocks):
  1. candidates = top N drivers of step 1 (corr_<product>.csv, kind == 'driver'), columns with
     > MAX_MISSING NaN dropped, near-duplicates removed (|r| > MAX_CORR on demeaned data)
  2. y = log(price_settle_ch), month x delivery-hour fixed effects (demeaning)
  3. forward selection by BIC (max MAX_VARS regressors) -> final OLS with Newey-West SE
  4. opportunity-cost spec: y on the day-ahead prices only (ch/de_lu/at/fr/it_nord_price_da_eur_mwh)
Coefficients are per 1 SD of the regressor (within-group), y in log points.
View loaded with impute=True (interior gaps <= 24 h, todo 5.5).

    python Modelling/rq1a_2_regression.py [--products fcr afrr mfrr] [--top 60]
Output: Modelling/Output/rq1a_2/reg_<product>.csv, report_rq1a_2.txt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Clean"))
import views as V  # noqa: E402

MAX_MISSING, MAX_CORR, MAX_VARS, MIN_ROWS = 0.10, 0.8, 12, 400
NW_LAGS = 6   # 6 x 4h = 1 day
DA = ["ch_price_da_eur_mwh", "de_lu_price_da_eur_mwh", "at_price_da_eur_mwh",
      "fr_price_da_eur_mwh", "it_nord_price_da_eur_mwh"]


def ols_nw(X: np.ndarray, y: np.ndarray, lags: int = NW_LAGS):
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    V_ = XtXi @ S @ XtXi * n / max(n - k, 1)
    se = np.sqrt(np.diag(V_))
    return b, se, e


def bic(e: np.ndarray, k: int) -> float:
    n = len(e)
    return n * np.log((e @ e) / n) + k * np.log(n)


def demean(df: pd.DataFrame, g: pd.Series) -> pd.DataFrame:
    return df - df.groupby(g, sort=False).transform("mean")


def run_series(s: pd.DataFrame, cands: list[str]):
    s = s.sort_values("block_start_local")
    y = np.log(s["price_settle_ch"].where(s["price_settle_ch"] > 0))
    g = s["block_start_local"].dt.strftime("%Y-%m") + "_" + s["block_start_local"].dt.hour.astype(str)
    cands = [c for c in cands if c in s and s[c].isna().mean() <= MAX_MISSING and s[c].nunique() > 3]
    X = s[cands]
    ok = y.notna() & X.notna().all(axis=1)
    if ok.sum() < MIN_ROWS:
        return None
    y, X, g = y[ok], X[ok], g[ok]
    Xd, yd = demean(X, g), demean(y.to_frame("y"), g)["y"]
    Xd = Xd.loc[:, Xd.std() > 1e-9]
    Z = (Xd - 0) / Xd.std()                       # within-group SD units
    yv = yd.values
    # drop near-duplicates (greedy in candidate order = strength order)
    keep: list[str] = []
    C = Z.corr().abs()
    for c in Z.columns:
        if all(C.loc[c, k] <= MAX_CORR for k in keep):
            keep.append(c)
    Z = Z[keep]
    sel: list[str] = []
    best = bic(yv, 0) if len(yv) else np.inf
    best = len(yv) * np.log((yv @ yv) / len(yv))
    while len(sel) < MAX_VARS:
        trial = None
        for c in Z.columns:
            if c in sel:
                continue
            A = Z[sel + [c]].values
            b = np.linalg.lstsq(A, yv, rcond=None)[0]
            bc = bic(yv - A @ b, len(sel) + 1)
            if bc < best - 1e-9 and (trial is None or bc < trial[1]):
                trial = (c, bc)
        if trial is None:
            break
        sel.append(trial[0]); best = trial[1]
    out = []
    if sel:
        A = Z[sel].values
        b, se, e = ols_nw(A, yv)
        r2 = 1 - (e @ e) / (yv @ yv)
        for c, bb, ss in zip(sel, b, se):
            out.append(dict(spec="selected", feature=c, coef=bb, se=ss, t=bb / ss, n=len(yv), r2_within=r2))
    # opportunity-cost spec
    da = [c for c in DA if c in s and s[c].isna().mean() <= MAX_MISSING]
    if da:
        okd = y.index.intersection(s.index)
        sd = s.loc[okd, da].dropna()
        yy = np.log(s.loc[sd.index, "price_settle_ch"].where(s["price_settle_ch"] > 0)).dropna()
        sd = sd.loc[yy.index]; gg = g.reindex(sd.index) if False else \
            (s.loc[sd.index, "block_start_local"].dt.strftime("%Y-%m") + "_" + s.loc[sd.index, "block_start_local"].dt.hour.astype(str))
        if len(sd) >= MIN_ROWS:
            Dd = demean(sd, gg); Dd = Dd / Dd.std()
            yd2 = demean(yy.to_frame("y"), gg)["y"].values
            b, se, e = ols_nw(Dd.values, yd2)
            r2 = 1 - (e @ e) / (yd2 @ yd2)
            for c, bb, ss in zip(da, b, se):
                out.append(dict(spec="da_only", feature=c, coef=bb, se=ss, t=bb / ss, n=len(yd2), r2_within=r2))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["fcr", "afrr", "mfrr"])
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "Modelling" / "Output" / "rq1a_2"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    corr_dir = ROOT / "Modelling" / "Output" / "rq1a_1"
    rep = []
    for prod in a.products:
        corr = pd.read_csv(corr_dir / f"corr_{prod}.csv")
        df = V.load(prod, kind="expost", rq="RQ1a", impute=True)
        res = []
        for (d, p), s in df.groupby(["direction", "procurement"], dropna=False):
            name = f"{prod}|{d}|{p}"
            c = corr[(corr.series == name) & (corr.kind == "driver")]
            cands = c.sort_values("spearman", key=abs, ascending=False).feature.head(a.top).tolist()
            r = run_series(s, cands) if cands else None
            if r is None or r.empty:
                rep.append(f"\n== {name}: too few complete rows =="); continue
            r.insert(0, "series", name); res.append(r)
            for spec in ("selected", "da_only"):
                rr = r[r.spec == spec]
                if rr.empty: continue
                rep.append(f"\n== {name} [{spec}] n={rr.n.iloc[0]:,} R2_within={rr.r2_within.iloc[0]:.3f} ==")
                rep.append(rr[["feature", "coef", "se", "t"]].round(3).to_string(index=False))
        if res:
            pd.concat(res).to_csv(out / f"reg_{prod}.csv", index=False)
    (out / "report_rq1a_2.txt").write_text("\n".join(rep)); print("\n".join(rep)); print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
