"""Clean layer: ENTSO-E Balancing (to-do 3.3 Balancing, 2 Oct 2026).

Run from Master_Thesis with .venv active:
    python Clean/clean_balancing.py            # build, check, print summary
    python Clean/clean_balancing.py --write    # also write Clean/Data/balancing/

Output (Clean/Data/balancing/), each with <name>_dictionary.csv, 15-min grid
  balancing.parquet
    ch_imb_price_{long,short}_eur_mwh   CH imbalance prices (17.1.G), from 31 Mar 2016;
                                        Swissgrid files take priority from 2023 (build_master)
    {de_lu,fr,it_nord}_imb_price_{long,short}_eur_mwh   neighbour imbalance prices
    ch_imb_vol_mwh, de_amprion_imb_vol_mwh, fr_imb_vol_mwh, it_imb_vol_mwh   (17.1.H)
    ch_afrr_{up,down}_act_price_eur_mwh, ch_rr_act_price_eur_mwh   CH activated
                                        balancing energy prices (17.1.F)
    ch_{afrr,mfrr}_bids_{up,down}_{offered,act}_mw   CH aggregated bids (12.3.E), 2021 ->
  contracted_reserves_ch.parquet        CH contracted reserves (17.1.B/C), cross-check only
    ch_{fcr,afrr,mfrr}_{weekly,daily}_{sym,up,down}_{price,amount}  (block values on the grid)
  contracted_reserves_fcr.parquet       (26 Oct 2026) FCR contracted reserves (17.1.B/C) of the
                                        other FCR Cooperation members de, fr, at, be, nl, dk1, si, cz
    <zone>_fcr_{weekly,daily,monthly,hourly}_<direction>_{price_eur_mw,amount_mw}
                                        whatever is on disk; a member without files is skipped
                                        with a notice. Not in the master (block products, like
                                        contracted_reserves_ch); regelleistung.net auction_blocks_rl
                                        stays the primary per-country FCR source from Jul 2019.

Rules
- Areas are named after what the series covers: DE prices = DE_LU, DE volumes =
  Amprion control area only (~1/4 of Germany, finding D2), IT prices = IT_NORD,
  IT volumes = Italy. Neighbour activation prices (sparse, single German TSO) not carried.
- CH imbalance volume = MWh per quarter-hour; same sign as Swissgrid
  ch_system_imbalance_mw (< 0 = short), about 1/4 of it (2026 corr 0.95).
- flag_imb_price_entsoe_error_2025_01: 2-5 Jan 2025 Long = Short on ENTSO-E (publication
  error; Swissgrid correct). From 2026 Long = Short = single price (regime).
- CH RR price (Swiss tertiary reserve = mFRR): up and down are one price; one column.
- Contracted reserves: ENTSO-E CH FCR weekly price is stuck at 6.26 from 2020 -> NaN
  (2015-2019 values are real). Swissgrid auction_blocks are the primary reserve
  source; these columns are only a cross-check (block values repeated on the grid, D5).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import config as C  # noqa: E402

SRC = "ENTSO-E Transparency Platform"


def entry(col, series, unit, res, agg, avail, note=""):
    return dict(column=col, source=SRC, source_series=series, unit=unit, resolution_native=res,
                aggregation_rule=agg, availability_rule=avail, notes=note)


def read(ds, var, area, **kw):
    try:
        return K.read_entsoe("Balancing", ds, var, area, **kw)
    except FileNotFoundError:
        return None


def to_grid(s: pd.Series) -> pd.Series:
    return K.hourly_to_qh(s.astype("float64"), "repeat")


def balancing():
    g = K.grid().index
    out, dic = pd.DataFrame(index=g), []
    AVP = "near real time on ENTSO-E (Swissgrid files: monthly, after delivery)"
    for area, name, note in (("CH", "ch", "from 31 Mar 2016; Jul-Sep 2016 missing at source; Swissgrid files take priority from 2023"),
                             ("DE", "de_lu", "DE_LU bidding zone"), ("FR", "fr", "30-min in early years"),
                             ("IT", "it_nord", "IT_NORD zone")):
        d = read("imbalance_prices", "all", area)
        if d is None:
            continue
        for src, side in (("Long", "long"), ("Short", "short")):
            col = f"{name}_imb_price_{side}_eur_mwh"
            out[col] = to_grid(d[src])
            dic.append(entry(col, f"17.1.G imbalance prices, {area}, {src}", "EUR/MWh", "15min (30min early FR)", "mean", AVP, note))
    e = (out.index >= pd.Timestamp("2025-01-02", tz=K.TZ).tz_convert("UTC")) & (out.index < pd.Timestamp("2025-01-06", tz=K.TZ).tz_convert("UTC"))
    out["flag_imb_price_entsoe_error_2025_01"] = e
    dic.append(entry("flag_imb_price_entsoe_error_2025_01", "derived", "bool", "15min", "any", "-",
                     "2-5 Jan 2025: ENTSO-E CH Long = Short (publication error); Swissgrid values are correct"))
    for area, name, note in (("CH", "ch", "MWh per quarter-hour; same sign as Swissgrid system imbalance (< 0 = short)"),
                             ("DE", "de_amprion", "Amprion control area only (~1/4 of Germany)"),
                             ("FR", "fr", ""), ("IT", "it", "Italy (national)")):
        d = read("imbalance_volumes", "all", area)
        if d is None:
            continue
        col = f"{name}_imb_vol_mwh"
        out[col] = to_grid(d.iloc[:, 0])
        dic.append(entry(col, f"17.1.H total imbalance volumes, {area}", "MWh", "15min", "sum", "ex post", note))
    # CH activated balancing energy prices (long: Direction, ReserveType)
    a = read("activated_balancing_energy_prices", "A16_realised", "CH", dedup=False)
    a = a.reset_index().drop_duplicates(["ts_utc", "ReserveType", "Direction"], keep="last")
    p = a.pivot(index="ts_utc", columns=["ReserveType", "Direction"], values="Price")
    for d_src, d in (("Up", "up"), ("Down", "down")):
        if ("aFRR", d_src) in p:
            col = f"ch_afrr_{d}_act_price_eur_mwh"
            out[col] = to_grid(p[("aFRR", d_src)])
            dic.append(entry(col, "17.1.F activated balancing energy prices, CH, aFRR " + d_src, "EUR/MWh", "15min", "mean", "ex post",
                             "cross-check of Swissgrid Energy Overview / system balance aFRR prices"))
    if ("RR", "Up") in p:
        up, dn = p[("RR", "Up")], p.get(("RR", "Down"))
        same = float(((up - dn).abs() < 0.01).mean()) if dn is not None else 1.0
        col = "ch_rr_act_price_eur_mwh"
        out[col] = to_grid(up if dn is None else up.combine_first(dn))
        dic.append(entry(col, "17.1.F activated balancing energy prices, CH, RR (Swiss tertiary = mFRR)", "EUR/MWh", "15min", "mean",
                         "ex post", f"up and down are one price ({same:.1%} identical)"))
    # CH aggregated bids (2021 ->)
    for var, prod in (("A51_aFRR", "afrr"), ("A47_mFRR", "mfrr")):
        b = read("aggregated_bids", var, "CH", dedup=False)
        if b is None:
            continue
        b = b.reset_index().groupby(["ts_utc", "direction"])[["Offered", "Activated"]].sum(min_count=1).unstack("direction")
        for d_src, d in (("Up", "up"), ("Down", "down")):
            for src, suf in (("Offered", "offered"), ("Activated", "act")):
                if (src, d_src) in b:
                    col = f"ch_{prod}_bids_{d}_{suf}_mw"
                    out[col] = to_grid(b[(src, d_src)])
                    dic.append(entry(col, f"12.3.E aggregated balancing energy bids, CH, {var}, {d_src}, {src}", "MW", "15min", "mean",
                                     "ex post", "2021 ->; mFRR sparse in 2021 (105 days) and 2022 (197 days)" if prod == "mfrr" else "2021 ->"))
    return out, dic


def contracted():
    g = K.grid().index
    out, dic = pd.DataFrame(index=g), []
    stale = pd.Timestamp("2020-01-01", tz=K.TZ).tz_convert("UTC")
    for kind, unit in (("prices", "EUR/MW (as published)"), ("amount", "MW")):
        for var in ("FCR_A02_weekly", "FCR_A01_daily", "aFRR_A02_weekly", "aFRR_A01_daily", "mFRR_A02_weekly", "mFRR_A01_daily"):
            d = read(f"contracted_reserve_{kind}", var, "CH")
            if d is None:
                continue
            prod, per = var.split("_")[0].lower(), var.split("_")[2]
            for src in d.columns:
                col = f"ch_{prod}_{per}_{src.lower()}_{'price_eur_mw' if kind == 'prices' else 'amount_mw'}"
                s = d[src].astype("float64")
                note = "cross-check only; block value repeated on the grid (D5); Swissgrid auction_blocks are primary"
                if kind == "prices" and var == "FCR_A02_weekly":
                    s = s.where(s.index < stale)
                    note += "; stale constant 6.26 from 2020 -> NaN (2015-2019 real)"
                out[col] = to_grid(s)
                dic.append(entry(col, f"17.1.B/C contracted reserve {kind}, CH, {var}, {src}", unit, "block (week/day/4h)", "mean", "after the auction", note))
    return out, dic


# 26 Oct 2026: FCR Cooperation members other than CH (ENTSO-E Balancing folder label -> token).
# DE amount = sum of the four TSO control areas (probe SUM_AREAS), DE price = TenneT (identical
# across TSOs, see probe). DK_1 = West Denmark.
FCR_COOP = {"DE": "de", "FR": "fr", "AT": "at", **K.FCR_MEMBERS}
FCR_VARIANTS = ("FCR_A02_weekly", "FCR_A01_daily", "FCR_A03_monthly", "FCR_A13_hourly")


def contracted_fcr():
    g = K.grid().index
    out, dic = pd.DataFrame(index=g), []
    for area, tok in FCR_COOP.items():
        found = False
        for kind, unit in (("prices", "EUR/MW (as published)"), ("amount", "MW")):
            for var in FCR_VARIANTS:
                d = read(f"contracted_reserve_{kind}", var, area)
                if d is None:
                    continue
                found = True
                per = var.split("_")[2]
                for src in d.columns:
                    col = f"{tok}_fcr_{per}_{src.lower()}_{'price_eur_mw' if kind == 'prices' else 'amount_mw'}"
                    out[col] = to_grid(d[src])
                    note = "block value repeated on the grid (D5); regelleistung.net is primary from Jul 2019"
                    if area == "DE":
                        note += ("; sum of the four German TSO control areas" if kind == "amount"
                                 else "; TenneT control area (price identical across the German TSOs)")
                    if area == "DK_1":
                        note += "; West Denmark (check the code used in the probe coverage: DK_1 or DK_CA)"
                    dic.append(entry(col, f"17.1.B/C contracted reserve {kind}, {area}, {var}, {src}", unit,
                                     "block (week/day/4h)", "mean", "after the auction", note))
        if not found:
            print(f"notice: no ENTSO-E FCR contracted reserves on disk for {area} - skipped")
    # NL weekly price (found 27 Oct 2026): published as the TOTAL cost of the week, not per MW
    # (median 133,762). Total / contracted MW / 168 h gives 9-32 EUR/MW/h (median 17.6), the same
    # level as the CH weekly price in the same common auction. Raw kept as *_cost_eur.
    raw_c, amt_c = "nl_fcr_weekly_symmetric_price_eur_mw", "nl_fcr_weekly_symmetric_amount_mw"
    if raw_c in out and amt_c in out:
        cost_c, new_c = "nl_fcr_weekly_symmetric_cost_eur", "nl_fcr_weekly_symmetric_price_eur_mw_h"
        out[new_c] = out[raw_c] / out[amt_c] / 168.0
        out = out.rename(columns={raw_c: cost_c})
        for d in dic:
            if d["column"] == raw_c:
                d["column"], d["unit"] = cost_c, "EUR per week (total, as published)"
                d["notes"] += "; published as the total cost of the week, not per MW; see " + new_c
        dic.append(entry(new_c, "derived: nl_fcr_weekly_symmetric_cost_eur / nl_fcr_weekly_symmetric_amount_mw / 168 h",
                         "EUR/MW/h", "block (week)", "mean", "after the auction",
                         "converted 27 Oct 2026; DST weeks have 167 / 169 h (< 0.6 % error); "
                         "level matches the CH weekly FCR price"))
    # AT weekly price: unit as published, NOT verified. Its level (median 3.6) is about 1/4 - 1/5 of
    # the CH and NL weekly prices of the same common auction (2016-2019); do not compare across zones.
    for d in dic:
        if d["column"] == "at_fcr_weekly_symmetric_price_eur_mw":
            d["notes"] += ("; unit unverified: level about 1/4-1/5 of the CH / NL weekly price in the same "
                           "common auction, do not compare across zones")
    return out, dic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    res = {"balancing": balancing(), "contracted_reserves_ch": contracted(),
           "contracted_reserves_fcr": contracted_fcr()}
    for name, (df, dic) in res.items():
        bad = K.check_names(df.columns)
        assert not bad, (name, bad)
        assert len(df) == 409_052
        num = df.select_dtypes(include="number")
        if num.empty:
            print(f"\n=== {name} === no columns (nothing on disk yet)")
            continue
        fv, lv = num.apply(lambda c: c.first_valid_index()), num.apply(lambda c: c.last_valid_index())
        s = pd.DataFrame({"first": fv.dt.tz_convert(K.TZ).dt.strftime("%Y-%m-%d"), "last": lv.dt.tz_convert(K.TZ).dt.strftime("%Y-%m-%d"),
                          "median": num.median().round(2), "min": num.min().round(1), "max": num.max().round(1),
                          "nan_in_range": [round(num.loc[fv[c]:lv[c], c].isna().mean(), 4) if fv[c] is not None else np.nan for c in num.columns]})
        print(f"\n=== {name} ===\n{s.to_string()}")
    # cross-checks against Swissgrid (if written)
    b = res["balancing"][0]
    sgi = C.DATA_DIR / "swissgrid" / "imbalance.parquet"
    if sgi.exists():
        x = pd.read_parquet(sgi).set_index("ts_utc")
        j = pd.concat([b["ch_imb_price_short_eur_mwh"], x["ch_imb_price_short_eur_mwh"]], axis=1, keys=["e", "s"]).dropna()
        print(f"\ncheck CH short price ENTSO-E vs Swissgrid 2023 ->: identical {((j.e - j.s).abs() < 0.05).mean():.3f} (n={len(j)})")
    sgs = C.DATA_DIR / "swissgrid" / "system_balance.parquet"
    if sgs.exists():
        x = pd.read_parquet(sgs, columns=["ts_utc", "ch_system_imbalance_mw"]).set_index("ts_utc")["ch_system_imbalance_mw"]
        j = pd.concat([b["ch_imb_vol_mwh"], x], axis=1).dropna()
        print(f"check CH imbalance volume vs Swissgrid system imbalance 2026: corr {j.corr().iloc[0, 1]:.3f}, "
              f"median ratio {(j.iloc[:, 0] / j.iloc[:, 1]).replace([np.inf, -np.inf], np.nan).median():.3f}")
    if a.write:
        for name, (df, dic) in res.items():
            if df.empty or len(df.columns) == 0:
                print(f"not written (no columns): {name}")
                continue
            print("written:", K.write_clean(df, "balancing", name, dic))


if __name__ == "__main__":
    main()
