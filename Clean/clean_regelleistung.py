"""Clean layer: regelleistung.net capacity auctions (to-do 3.6, 2 Oct 2026).

Run from Master_Thesis with .venv active:
    python Clean/clean_regelleistung.py            # build, check, print summary
    python Clean/clean_regelleistung.py --write    # also write Clean/Data/regelleistung/

Output  Clean/Data/regelleistung/auction_blocks_rl.parquet (+ _dictionary.csv):
        one row per market x product x direction x delivery block, same keys as
        Clean/Data/swissgrid/auction_blocks.parquet (stacked in build_master.py)
  market fcr_coop          FCR cooperation, from 1 Jul 2019 (daily product until
                           30 Jun 2020, 4h blocks after): cross-border price and,
                           per country, settlement price, demand, net export
  market de_regelleistung  German aFRR / mFRR, 4h blocks from 12 Jul 2018:
                           min / average / marginal capacity price (DE, AT for aFRR),
                           offered and allocated volume, net export

Rules (regelleistung.net findings R1-R5, checks 2.5 / 2.6)
- R1 FCR: only tender_number == 1; flag_second_auction marks blocks that also had
  a second auction (its prices are extreme or empty and are not used).
- R2 aFRR / mFRR: capacity prices per block (EUR/MW, until 7 Dec 2021) and per hour
  (EUR/MW/h, from 8 Dec 2021) are merged into EUR/MW per hour: old values / block
  hours. FCR prices are per block throughout -> / block hours (24 h for the daily
  product; 3 h / 5 h on DST days). All prices in this table are EUR per MW per hour.
- R3 FCR: 2-letter country prefixes (until 6 Sep 2022) and full names (from 7 Sep
  2022) are merged; import(-)/export(+) and deficit(-)/surplus(+) have the same
  meaning (check 2.5): net_export_<cc>_mw > 0 = country procured more than its demand.
- flag_fcr_ch_balance_mismatch: CH net export differs from Swissgrid awarded_ch -
  CH demand by >= 0.5 MW (128 blocks 2020-22, check 2.5).
- Energy-price columns (until 2020/21, separate energy market since Nov 2020) are
  not carried.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import config as C  # noqa: E402

RL = C.ROOT / "Regelleistung" / "Data" / "production" / "results_CAPACITY"
CC = {"at": "austria", "be": "belgium", "ch": "switzerland", "de": "germany", "fr": "france",
      "nl": "netherlands", "si": "slovenia", "dk": "denmark", "cz": "czech_republic"}
SHORT = {v: k for k, v in CC.items()}


def load(prod: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sorted((RL / prod).glob("[0-9]*.parquet"))], ignore_index=True)


def fcr():
    d = load("FCR")
    hrs = d.productname.str.extract(r"_(\d\d)_(\d\d)$").astype(int)
    day = pd.to_datetime(d.date_from)
    start_l = day + pd.to_timedelta(hrs[0], unit="h")
    end_l = day + pd.to_timedelta(hrs[1], unit="h")
    d["block_start_utc"] = K.to_utc(pd.DatetimeIndex(start_l).tz_localize(K.TZ, ambiguous=True, nonexistent="shift_forward"))
    d["block_end_utc"] = K.to_utc(pd.DatetimeIndex(end_l).tz_localize(K.TZ, ambiguous=True, nonexistent="shift_forward"))
    d["duration_h"] = (d.block_end_utc - d.block_start_utc).dt.total_seconds() / 3600
    second = set(map(tuple, d.loc[d.tender_number == 2, ["block_start_utc", "block_end_utc"]].astype("int64").to_numpy()))
    d = d[d.tender_number == 1].copy()
    out = pd.DataFrame({"market": "fcr_coop", "product": "FCR", "direction": "sym",
                        "procurement": np.where(d.duration_h > 20, "day", "4h"),
                        "block_start_utc": d.block_start_utc, "block_end_utc": d.block_end_utc,
                        "duration_h": d.duration_h, "product_name": d.productname})
    out["price_settle_coop"] = d["crossborder_settlementcapacity_price_eur_mw"] / d.duration_h
    for cc, full in CC.items():
        for kind, old_suf, new_suf in [("price_settle", "settlementcapacity_price_eur_mw", "settlementcapacity_price_eur_mw"),
                                       ("demand", "demand_mw", "demand_mw"),
                                       ("net_export", "import_minus_export_plus_mw", "deficit_minus_surplus_plus_mw")]:
            a = d.get(f"{cc}_{old_suf}")
            b = d.get(f"{full}_{new_suf}")
            if a is None and b is None:
                continue
            v = (a if a is not None else pd.Series(np.nan, index=d.index)).combine_first(
                b if b is not None else pd.Series(np.nan, index=d.index)).astype("float64")
            if kind == "price_settle":
                out[f"price_settle_{cc}"] = v / d.duration_h
            else:
                out[f"{kind}_{cc}_mw"] = v
    key = list(zip(out.block_start_utc.astype("int64"), out.block_end_utc.astype("int64")))
    out["flag_second_auction"] = [k in second for k in key]
    # CH balance vs Swissgrid (check 2.5)
    sgp = C.DATA_DIR / "swissgrid" / "auction_blocks.parquet"
    if sgp.exists():
        sg = pd.read_parquet(sgp, columns=["product", "block_start_utc", "awarded_ch_mw"])
        sg = sg[sg["product"] == "FCR"].drop(columns="product")
        m = out[["block_start_utc", "net_export_ch_mw", "demand_ch_mw"]].merge(sg, on="block_start_utc", how="left")
        mis = (m.net_export_ch_mw - (m.awarded_ch_mw - m.demand_ch_mw)).abs() >= 0.5
        out["flag_fcr_ch_balance_mismatch"] = mis.fillna(False).to_numpy()
    else:
        print("  NOTE: Clean/Data/swissgrid/auction_blocks.parquet missing -> run clean_swissgrid.py --write first; flag set to False")
        out["flag_fcr_ch_balance_mismatch"] = False
    return out


def de_reserve(prod: str):
    d = load(prod)
    d["block_start_utc"] = K.to_utc(d.delivery_start)
    d["block_end_utc"] = K.to_utc(d.delivery_end)
    d["duration_h"] = (d.block_end_utc - d.block_start_utc).dt.total_seconds() / 3600
    out = pd.DataFrame({"market": "de_regelleistung", "product": prod,
                        "direction": np.where(d["product"].str.startswith("POS"), "up", "down"),
                        "procurement": "4h", "block_start_utc": d.block_start_utc, "block_end_utc": d.block_end_utc,
                        "duration_h": d.duration_h, "product_name": d["product"]})
    for area in ("total", "germany", "austria"):
        for stat in ("min", "average", "marginal"):
            new = d.get(f"{area}_{stat}_capacity_price_eur_mw_h")
            old = d.get(f"{area}_{stat}_capacity_price_eur_mw")
            if new is None and old is None:
                continue
            old_h = (old / d.duration_h) if old is not None else pd.Series(np.nan, index=d.index)
            new = new if new is not None else pd.Series(np.nan, index=d.index)
            tag = {"total": "total", "germany": "de", "austria": "at"}[area]
            out[f"price_{stat}_{tag}"] = new.combine_first(old_h).astype("float64")
    for src, name in [("germany_sum_of_offered_capacity_mw", "offered_de_mw"),
                      ("germany_allocated_volume_mw", "allocated_de_mw"),
                      ("germany_import_minus_export_plus_mw", "net_export_de_mw"),
                      ("austria_import_minus_export_plus_mw", "net_export_at_mw")]:
        if src in d:
            out[name] = d[src].astype("float64")
    return out


def build():
    t = pd.concat([fcr(), de_reserve("aFRR"), de_reserve("mFRR")], ignore_index=True)
    t = t[(t.block_start_utc < C.end_utc()) & (t.block_end_utc > C.start_utc())].copy()
    t["partial_in_sample"] = (t.block_end_utc > C.end_utc())
    t["block_start_local"] = t.block_start_utc.dt.tz_convert(K.TZ)
    t["currency"] = "EUR"
    r = K.regimes(pd.DatetimeIndex(t.block_start_utc))
    for c in r.columns:
        t[c] = r[c].to_numpy()
    for c in ("flag_second_auction", "flag_fcr_ch_balance_mismatch"):
        t[c] = t[c].fillna(False).astype(bool)
    dup = t.duplicated(["market", "product", "direction", "block_start_utc"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate blocks")
    return t.sort_values(["market", "product", "direction", "block_start_utc"]).reset_index(drop=True)


def dictionary(t: pd.DataFrame) -> list[dict]:
    notes = {"price_settle_coop": "FCR cooperation cross-border settlement price, EUR/MW/h",
             "flag_second_auction": "FCR block also had a second auction (tender 2; not used, R1)",
             "flag_fcr_ch_balance_mismatch": "CH net export != Swissgrid awarded_ch - CH demand (check 2.5)",
             "product_name": "regelleistung.net product name (e.g. NEGPOS_04_08, POS_00_04)"}
    out = []
    for c in t.columns:
        n = notes.get(c, "")
        if c.startswith("price_") and not n:
            n = "EUR per MW per hour (R2 / block-hour conversion applied)"
        if c.startswith(("demand_", "net_export_", "offered_", "allocated_")) and not n:
            n = "MW; net_export > 0 = procured more than own demand" if c.startswith("net_export_") else "MW"
        out.append(dict(column=c, source="regelleistung.net" if not c.startswith("regime_") else "derived (common.regimes)",
                        source_series="Regelleistung/Data/production/results_CAPACITY", notes=n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    t = build()
    print(t.groupby(["market", "product", "direction", "procurement"]).agg(
        n=("block_start_utc", "size"), first=("block_start_local", "min"), last=("block_start_local", "max")).to_string())
    print("\nflags:", {c: int(t[c].sum()) for c in t.columns if c.startswith("flag_")})
    # R2 continuity check: DE aFRR marginal price per hour, 4 weeks before / after 8 Dec 2021
    x = t[(t["product"] == "aFRR") & (t.direction == "up")].set_index("block_start_local")["price_marginal_de"]
    print("R2 check, DE aFRR up marginal EUR/MW/h median: before", round(x["2021-11-10":"2021-12-07"].median(), 2),
          "| after", round(x["2021-12-08":"2022-01-05"].median(), 2))
    f = t[t["product"] == "FCR"].set_index("block_start_local")
    print("FCR CH price EUR/MW/h median by year:", f["price_settle_ch"].groupby(f.index.year).median().round(2).to_dict())
    nan = t.filter(like="price_").isna().mean().round(3)
    print("price NaN shares:", nan[nan > 0].to_dict())
    if a.write:
        print("written:", K.write_long(t, "regelleistung", "auction_blocks_rl", dictionary(t)))


if __name__ == "__main__":
    main()
