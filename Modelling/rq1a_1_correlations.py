"""RQ1a step 1 (todo 9.1): within-month x delivery-slot driver correlations.

Per series (product x direction x procurement) the target is log(price_settle_ch)
(prices > 0). Every numeric feature of the ex-post view is correlated with it AFTER
removing the month x slot level (EDA 4: level correlations are dominated by the
2021-23 trend). Pooled within-group Spearman (ranks inside each group, centred) and
Pearson on log target / raw feature (centred per group).

Run from Master_Thesis with .venv active:
    python Modelling/rq1a_1_correlations.py             # -> Modelling/Output/rq1a_1/
    python Modelling/rq1a_1_correlations.py --products fcr
Outputs: corr_<product>.csv (long: series, feature, n, spearman, pearson, kind),
         report_rq1a_1.txt (top 15 per series).
Read-only on Clean/Data; writes only Modelling/Output/rq1a_1/.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Clean"))
import views as V  # noqa: E402

MIN_VALID_SHARE = 0.5     # drops the empty / mostly empty RQ1a columns
MIN_GROUP = 5             # blocks per month x slot group
MIN_VARYING = 0.8         # share of groups in which the feature must vary
MIN_N = 300               # pooled pairs per correlation
OUTCOME = ("offered_mw", "awarded_mw", "awarded_ch_mw", "_offered_mw", "n_bids", "n_accepted", "cost")


def kind_of(col: str) -> str:
    """market outcome (offered volumes etc.) vs candidate driver."""
    return "outcome" if col.endswith(OUTCOME) else "driver"


def within(x: pd.DataFrame, g: pd.Series, rank: bool) -> pd.DataFrame:
    if rank:
        x = x.groupby(g, sort=False).rank()
    return x - x.groupby(g, sort=False).transform("mean")


def one_series(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    y = np.log(df["price_settle_ch"].where(df["price_settle_ch"] > 0))
    weekly = df["duration_h"].ge(168).all()
    if weekly:   # weekly blocks: too few per month -> group by quarter
        g = df["block_start_local"].dt.to_period("Q").astype(str)
    else:
        g = (df["block_start_local"].dt.strftime("%Y-%m") + "_" + df["block_start_local"].dt.hour.astype(str))
    ok = y.notna()
    df, y, g = df[ok], y[ok], g[ok]
    size = g.map(g.value_counts())
    keep = size >= MIN_GROUP
    df, y, g = df[keep], y[keep], g[keep]
    X = df[feats]
    rows = []
    for c in feats:
        m = X[c].notna()
        if m.sum() < MIN_N or X.loc[m, c].nunique() < 3:
            continue
        gg = g[m]
        gs = gg.map(gg.value_counts()); mm = gs >= MIN_GROUP
        xs, ys, gg = X.loc[m, c][mm], y[m][mm], gg[mm]
        if len(xs) < MIN_N:
            continue
        pair = pd.DataFrame({"x": xs, "y": ys})
        sp = within(pair, gg, True); pe = within(pair, gg, False)
        # skip features that barely vary inside a group (e.g. month-ahead forecasts): ranks are ties
        varying = (xs.groupby(gg).nunique() > 1).mean()
        if varying < MIN_VARYING:
            continue
        rows.append((c, len(xs), sp.x.corr(sp.y), pe.x.corr(pe.y), kind_of(c)))
    return pd.DataFrame(rows, columns=["feature", "n", "spearman", "pearson", "kind"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["fcr", "afrr", "mfrr"])
    ap.add_argument("--out", default=str(ROOT / "Modelling" / "Output" / "rq1a_1"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = []
    for prod in a.products:
        df = V.load(prod, kind="expost", rq="RQ1a", impute=False)
        dic = V.dictionary(prod, "expost")
        feats = [c for c in dic.index[dic["role"] == "feature"] if c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().mean() >= MIN_VALID_SHARE]
        print(f"{prod}: {len(df):,} blocks, {len(feats)} features with >= {MIN_VALID_SHARE:.0%} valid")
        res = []
        for (d, p), s in df.groupby(["direction", "procurement"], dropna=False):
            r = one_series(s, feats)
            if r.empty:
                continue
            r.insert(0, "series", f"{prod}|{d}|{p}")
            r["abs_sp"] = r.spearman.abs()
            r = r.sort_values("abs_sp", ascending=False)
            res.append(r)
            rep.append(f"\n== {prod} | {d} | {p}: {len(s):,} blocks, {len(r)} features ==")
            rep.append(r.head(15)[["feature", "n", "spearman", "pearson", "kind"]].round(3).to_string(index=False))
        if res:
            pd.concat(res).drop(columns="abs_sp").to_csv(out / f"corr_{prod}.csv", index=False)
    (out / "report_rq1a_1.txt").write_text("\n".join(rep))
    print("\n".join(rep)); print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
