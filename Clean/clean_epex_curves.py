"""Clean layer: EPEX SPOT CH day-ahead aggregated curves -> hourly curve features, written 16 Nov 2026.

The Swiss day-ahead auction is not coupled to its neighbours, so the aggregated sell and purchase
curves of an hour fully determine its price: the price where the curves cross reproduces the
published EPEX price (checked per year in the report). The curves therefore describe the shape of
the Swiss merit order around the clearing point, which no other source in the thesis contains.

Licence: EPEX historical package, CH, up to 2025 (2026 not licensed; nothing after 2025 is kept).

Input  EpexSpot_histo/Day-Ahead Auction/Hourly/Historical/Aggregated curves/
       auction_aggregated_curves_switzerland_YYYY.zip   (2016-2025; one csv per delivery day,
       rows: Date, Week, Week Day, Hour (1-24; '3B' = second 02:00 hour in October), Price,
       Volume (cumulative MW), Sale/Purchase)
Output (Clean/Data/epex/, only with --write)
  da_curves.parquet   15-min grid (hourly values repeated), columns:
    ch_da_curve_dp_up100_eur_mwh    price rise if demand were 100 MW higher (or supply 100 MW lower)
    ch_da_curve_dp_up500_eur_mwh    same for 500 MW
    ch_da_curve_dp_down100_eur_mwh  price fall (negative) if supply were 100 MW higher
    ch_da_curve_dp_down500_eur_mwh  same for 500 MW
    ch_da_curve_head_sell_mw        sell volume offered above the clearing volume up to the price
                                    limit (spare supply in the order book)
    ch_da_curve_head_buy_mw         purchase volume bid beyond the clearing volume down to the
                                    lowest price (spare demand)
    ch_da_curve_inel_sell_mw        sell volume at the lowest price (price-taking supply)
    ch_da_curve_inel_buy_mw         purchase volume at the highest price (price-taking demand)
    ch_da_curve_q_mw                volume where the curves cross
    ch_da_curve_p_eur_mwh           price where the curves cross (check against the EPEX price)
  da_curves_dictionary.csv, clean_epex_curves_report.txt

Method per hour: both curves are interpolated linearly on the union of their price points; excess
supply E(p) = sell(p) - purchase(p) is non-decreasing in p. The clearing price solves E(p) = 0; the
price for +X MW demand solves E(p) = +X, for +X MW supply E(p) = -X (linear interpolation between
price points; at the end of the curve the price limit is returned).
Availability: published with the day-ahead result, D-1 after the auction (rule "D-1 after
day-ahead coupling" = D-1 13:00 in the ex-ante plan): known at the RQ3 d1 origin (D-1 18:00), and
only as lookbacks at the gate closure of the Swiss capacity auctions (RQ2).

Run from Master_Thesis with .venv active (several minutes; years run in parallel):
    python Clean/clean_epex_curves.py            # build + checks, writes nothing
    python Clean/clean_epex_curves.py --write
    options: --workers N (default 4), --years 2016 2017 (subset, for a quick check)
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import config as C  # noqa: E402

DOMAIN, TABLE = "epex", "da_curves"
SRC = "EPEX SPOT market data (university historical package, CH)"
CURVE_DIR = C.ROOT / "EpexSpot_histo" / "Day-Ahead Auction" / "Hourly" / "Historical" / "Aggregated curves"
LICENCE_END = pd.Timestamp("2026-01-01", tz=K.TZ).tz_convert("UTC")
FEATS = ["dp_up100", "dp_up500", "dp_down100", "dp_down500", "head_sell", "head_buy",
         "inel_sell", "inel_buy", "q", "p"]
COLS = {f: f"ch_da_curve_{f}_" + ("eur_mwh" if f.startswith("dp") or f == "p" else "mw") for f in FEATS}

_report: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _report.append(s)


def hour_features(sp, sv, bp, bv) -> list[float]:
    o = np.argsort(sp, kind="stable")
    sp, sv = sp[o], sv[o]
    o = np.argsort(bp, kind="stable")
    bp, bv = bp[o], bv[o]
    grid = np.union1d(sp, bp)
    S = np.interp(grid, sp, sv)
    D = np.interp(grid, bp, bv)
    E = np.maximum.accumulate(S - D)            # guard against tiny non-monotone steps

    def peq(x):
        k = np.searchsorted(E, x)
        if k == 0:
            return grid[0]
        if k >= len(E):
            return grid[-1]
        e0, e1 = E[k - 1], E[k]
        return grid[k] if e1 == e0 else grid[k - 1] + (x - e0) * (grid[k] - grid[k - 1]) / (e1 - e0)

    p = peq(0.0)
    q = float(np.interp(p, grid, S))
    return [peq(100.0) - p, peq(500.0) - p, peq(-100.0) - p, peq(-500.0) - p,
            float(sv.max() - q), float(bv.max() - q), float(sv.min()), float(bv.min()), q, p]


def one_year(path: str) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(path) as Z:
        for n in sorted(Z.namelist()):
            if not n.endswith(".csv"):
                continue
            d = pd.read_csv(Z.open(n), skiprows=1, usecols=["Date", "Hour", "Price", "Volume", "Sale/Purchase"],
                            dtype={"Date": str, "Hour": str, "Price": "float64", "Volume": "float64",
                                   "Sale/Purchase": str})
            if d.empty:
                continue
            sell = d["Sale/Purchase"].str.lower().str.startswith("s").to_numpy()
            for (day, hr), idx in d.groupby(["Date", "Hour"], sort=False).indices.items():
                s_i, b_i = idx[sell[idx]], idx[~sell[idx]]
                if len(s_i) < 2 or len(b_i) < 2:
                    continue
                f = hour_features(d["Price"].to_numpy()[s_i], d["Volume"].to_numpy()[s_i],
                                  d["Price"].to_numpy()[b_i], d["Volume"].to_numpy()[b_i])
                rows.append([day, hr, *f])
    out = pd.DataFrame(rows, columns=["date", "hour", *FEATS])
    out.attrs["file"] = Path(path).name
    return out


def to_utc(f: pd.DataFrame) -> pd.DatetimeIndex:
    h = f["hour"].astype(str).str.strip()
    hr = pd.to_numeric(h.str.replace("B", "", regex=False).str.replace("A", "", regex=False), errors="coerce")
    amb = ~h.str.endswith("B")                           # '3' / '3A' = CEST, '3B' = CET on the October day
    naive = pd.to_datetime(f["date"], format="%d/%m/%Y") + pd.to_timedelta(hr - 1, unit="h")
    return (pd.DatetimeIndex(naive).tz_localize(K.TZ, ambiguous=amb.to_numpy(), nonexistent="NaT")
            .tz_convert("UTC").as_unit("ns"))


def main_() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--years", nargs="*", type=int)
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    files = sorted(CURVE_DIR.glob("auction_aggregated_curves_switzerland_*.zip"))
    if a.years:
        files = [f for f in files if any(str(y) in f.name for y in a.years)]
    if not files:
        raise SystemExit(f"no curve files in {CURVE_DIR}")
    t0 = time.time()
    log(f"{len(files)} yearly files, {a.workers} workers")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        parts = list(ex.map(one_year, [str(f) for f in files]))
    f = pd.concat(parts, ignore_index=True)
    f.index = to_utc(f)
    bad = f.index.isna()
    log(f"hours built: {len(f):,} in {time.time() - t0:.0f}s; hours without a valid local time (spring DST): {int(bad.sum())}")
    f = f[~bad]
    f = f[~f.index.duplicated(keep="last")].sort_index()
    f = f[f.index < LICENCE_END]

    g = pd.DataFrame(index=K.grid().index)
    for k in FEATS:
        g[COLS[k]] = K.hourly_to_qh(f[k].astype("float64"), "repeat")

    # ---- checks
    yr = g.index.tz_convert(K.TZ).year
    log("\n=== coverage (non-NaN share per year)\n" + g[[COLS["p"]]].notna().groupby(yr).mean().round(3).to_string())
    da = C.DATA_DIR / "epex" / "da.parquet"
    if da.exists():
        e = pd.read_parquet(da, columns=["ts_utc", "ch_price_da_epex_eur_mwh"])
        e = e.set_index(K.to_utc(e.pop("ts_utc")))["ch_price_da_epex_eur_mwh"].reindex(g.index)
        d = (g[COLS["p"]] - e).abs()
        ok = d.notna()
        log("\n=== curve crossing vs published EPEX price (|diff| EUR/MWh)\n"
            + pd.DataFrame({"median": d[ok].groupby(yr[ok]).median().round(3),
                            "p99": d[ok].groupby(yr[ok]).quantile(0.99).round(3),
                            "share_gt_1": (d[ok] > 1).groupby(yr[ok]).mean().round(4)}).to_string())
    else:
        log("(Clean/Data/epex/da.parquet missing: price check skipped; run clean_epex.py --write)")
    log("\n=== yearly mean\n" + g.groupby(yr).mean().round(1).T.to_string())
    log("\n=== distribution (hours 2016-2025)\n" + f[FEATS].describe(percentiles=[0.01, 0.5, 0.99]).T.round(2).to_string())
    log("\n=== correlation of the curve features with the clearing price / with each other (2016-2025)\n"
        + f[FEATS].corr(method="spearman").round(2).to_string())

    if a.write:
        dd = []
        desc = {"dp_up100": "price change if demand +100 MW (E(p) = +100)", "dp_up500": "price change if demand +500 MW",
                "dp_down100": "price change if supply +100 MW (E(p) = -100)", "dp_down500": "price change if supply +500 MW",
                "head_sell": "sell volume offered beyond the clearing volume (curve maximum - clearing volume)",
                "head_buy": "purchase volume bid beyond the clearing volume (curve maximum - clearing volume)",
                "inel_sell": "sell volume at the lowest price (price-taking supply)",
                "inel_buy": "purchase volume at the highest price (price-taking demand)",
                "q": "volume where the curves cross", "p": "price where the curves cross (= EPEX price, check column)"}
        for k in FEATS:
            dd.append(dict(column=COLS[k], source=SRC, source_series="Day-Ahead Auction aggregated curves CH",
                           unit="EUR/MWh" if COLS[k].endswith("eur_mwh") else "MW", resolution_native="1h",
                           aggregation_rule="mean", availability_rule="D-1 after day-ahead coupling",
                           notes=desc[k]))
        p = K.write_clean(g, DOMAIN, TABLE, dd)
        (C.DATA_DIR / DOMAIN / "clean_epex_curves_report.txt").write_text("\n".join(_report))
        log(f"\nwritten: {p.relative_to(C.ROOT)} + {TABLE}_dictionary.csv + clean_epex_curves_report.txt")
    else:
        log("\n(dry run: nothing written; add --write)")


if __name__ == "__main__":
    main_()
