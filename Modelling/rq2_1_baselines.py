"""RQ2 step 1 (todo 9.2a): naive baselines and their errors.

Baselines (all known at the auction gate closure, ex-ante views):
  b_prev_slot     same delivery slot the day before (weekly products: previous week)  tgt_prev_slot__*
  b_prev_auction  result of the previous auction                                      tgt_prev_auction__*
  b_roll_med      median of the last 7 same-slot results (weekly: last 4 weeks)

Per series (product x direction x procurement): MAE, RMSE, sMAPE (EUR/MWh) and MAE on the asinh
scale for the test period (blocks from 1 Jan 2021) and per calendar year; common rows of all
three baselines.

    python Modelling/rq2_1_baselines.py [--products fcr afrr mfrr] [--out DIR]
Output: Modelling/Output/rq2_1/baseline_errors.csv, baseline_errors_by_year.csv, report_rq2_1.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import rq2_common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=C.VIEWS)
    ap.add_argument("--out", default=str(C.ROOT / "Modelling" / "Output" / "rq2_1"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, rows_y, lines = [], [], []
    for view in a.products:
        df = C.load_view(view)
        df = df[df["block_start_utc"] >= C.TEST_START.tz_convert("UTC")]
        df = df.dropna(subset=["y", *C.BASELINES]).copy()
        df["year"] = df["block_start_utc"].dt.tz_convert(C.TZ).dt.year
        for s, g in df.groupby("series", sort=False):
            for b in C.BASELINES:
                rows.append({"series": s, "baseline": b, **C.metrics(g["y"], g[b]),
                             "first_block": g["block_start_utc"].min().date(),
                             "median_price": g["y"].median()})
                for y, gy in g.groupby("year"):
                    rows_y.append({"series": s, "baseline": b, "year": y, **C.metrics(gy["y"], gy[b])})
    res, ry = pd.DataFrame(rows), pd.DataFrame(rows_y)
    res.to_csv(out / "baseline_errors.csv", index=False)
    ry.to_csv(out / "baseline_errors_by_year.csv", index=False)
    lines.append("RQ2 9.2a - naive baselines (test period: blocks from 1 Jan 2021; common rows of all baselines)")
    lines.append("Baselines: " + "; ".join(f"{k} = {v}" for k, v in C.BASE_LABEL.items()))
    lines.append("")
    for s, g in res.groupby("series", sort=False):
        lines.append(f"== {s}   (n = {int(g['n'].iloc[0]):,}, median price {g['median_price'].iloc[0]:.2f}, "
                     f"first block {g['first_block'].iloc[0]})")
        lines.append(C.fmt(g.set_index("baseline")[["MAE", "RMSE", "sMAPE_%", "MAE_asinh"]]))
        lines.append(f"   best by MAE: {g.loc[g['MAE'].idxmin(), 'baseline']}")
        lines.append("   MAE by year:")
        lines.append(C.fmt(ry[ry.series == s].pivot(index="year", columns="baseline", values="MAE")))
        lines.append("")
    (out / "report_rq2_1.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
