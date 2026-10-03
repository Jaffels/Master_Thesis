"""Clean layer: ENTSO-E day-ahead prices (12.1.D), written 3 Oct 2026 (to-do 7.1).

Input: Entsoe/Prices/Data/production/day_ahead_price/<AREA>/<YEAR>.parquet
       (Entsoe/Prices/entsoe_prices_pull.py; index 'timestamp' UTC, price_eur_mwh, area_code)

Run from Master_Thesis with .venv active (after the pull):
    python Clean/clean_prices.py            # build, check, print summary (no files)
    python Clean/clean_prices.py --write    # also write Clean/Data/prices/
Then build_master.py picks the table up automatically (OPTIONAL_TABLES), and
build_views_exante.py / build_view_rq3.py use it through the data dictionary.

Output (Clean/Data/prices/)
  day_ahead.parquet            <zone>_price_da_eur_mwh for ch, de_lu, at, fr, it_nord
  day_ahead_dictionary.csv

Rules
- Hourly prices are repeated over their own four quarter-hours (common.hourly_to_qh);
  15-min prices (SDAC 15-min MTU from 1 Oct 2025) are used as they are. Gaps stay NaN.
- de_lu: DE_AT_LU price until 30 Sep 2018, DE_LU from 1 Oct 2018 (one column, see
  regime_de_zone), as for the ENTSO-E load.
- at: before 1 Oct 2018 Austria was part of DE_AT_LU, so the AT price IS the DE_AT_LU
  price; filled from it and flagged (flag_at_price_da_from_de_at_lu).
- availability_rule "D-1 after day-ahead coupling" -> D-1 13:00 local in the ex-ante
  plan: after every CH capacity gate closure (so only lookbacks / __pf in the block
  views), before the RQ3 D-1 18:00 origin (so __fc_d1 in the RQ3 view).
- Negative prices are valid values. Suspect: outside the SDAC harmonised limits
  [-500, +4000] EUR/MWh (only possible by error) -> flag_<zone>_price_da_suspect, kept.
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

DOMAIN, TABLE = "prices", "day_ahead"
SRC = "ENTSO-E Transparency Platform"
IN_DIR = C.ROOT / "Entsoe" / "Prices" / "Data" / "production" / "day_ahead_price"
AREAS = {"CH": "ch", "DE": "de_lu", "AT": "at", "FR": "fr", "IT_NORD": "it_nord"}
LIMIT_LO, LIMIT_HI = -500.0, 4000.0          # SDAC harmonised min / max clearing price (since 2022)
SPLIT = pd.Timestamp("2018-10-01", tz=K.TZ).tz_convert("UTC")


def read_area(area: str) -> pd.DataFrame:
    files = sorted((IN_DIR / area).glob("[0-9]*.parquet"))
    if not files:
        raise FileNotFoundError(f"no files under {IN_DIR / area}: run Entsoe/Prices/entsoe_prices_pull.py first")
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index(kind="stable")
    df.index = K.to_utc(df.index)
    df = df[~df.index.duplicated(keep="last")]
    if "price_eur_mwh" not in df.columns:
        raise ValueError(f"{area}: unexpected columns {list(df.columns)}")
    return df


def build():
    g = K.grid()
    out = pd.DataFrame(index=g.index)
    dd, info = [], {}
    for area, a in AREAS.items():
        raw = read_area(area)
        s = raw["price_eur_mwh"].astype("float64")
        col = f"{a}_price_da_eur_mwh"
        out[col] = K.hourly_to_qh(s, "repeat")
        codes = "/".join(dict.fromkeys(raw["area_code"].astype(str))) if "area_code" in raw else area
        note = ""
        if a == "de_lu":
            note = "until 30 Sep 2018 DE_AT_LU, from 1 Oct 2018 DE_LU (see regime_de_zone)"
        dd.append(dict(column=col, source=SRC, source_series=f"12.1.D day-ahead prices, {codes}", unit="EUR/MWh",
                       resolution_native=str(K.native_step(K.trim(s.to_frame()).index)),
                       aggregation_rule="mean", availability_rule="D-1 after day-ahead coupling", notes=note))
        info[a] = raw
    # AT before the split = DE_AT_LU price
    pre = out.index < SPLIT
    fill = pre & out["at_price_da_eur_mwh"].isna() & out["de_lu_price_da_eur_mwh"].notna()
    out.loc[fill, "at_price_da_eur_mwh"] = out.loc[fill, "de_lu_price_da_eur_mwh"]
    out["flag_at_price_da_from_de_at_lu"] = fill
    dd[[r["column"] for r in dd].index("at_price_da_eur_mwh")]["notes"] = \
        "before 1 Oct 2018 = DE_AT_LU price (AT part of that zone), see flag_at_price_da_from_de_at_lu"
    dd.append(dict(column="flag_at_price_da_from_de_at_lu", source="derived", source_series="de_lu_price_da_eur_mwh",
                   unit="bool", resolution_native="15min", aggregation_rule="any", availability_rule="-",
                   notes="True where the AT price is the DE_AT_LU zone price (before 1 Oct 2018)"))
    for a in AREAS.values():
        col, fl = f"{a}_price_da_eur_mwh", f"flag_{a}_price_da_suspect"
        out[fl] = ((out[col] < LIMIT_LO) | (out[col] > LIMIT_HI)).fillna(False).astype(bool)
        dd.append(dict(column=fl, source="derived", source_series=col, unit="bool", resolution_native="15min",
                       aggregation_rule="any", availability_rule="-",
                       notes=f"price outside the SDAC limits [{LIMIT_LO:.0f}, {LIMIT_HI:.0f}] EUR/MWh; value kept"))
    return out, dd, int(fill.sum())


def main_() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    df, dd, n_fill = build()
    bad = K.check_names(list(df.columns))
    assert not bad, bad
    assert len(df) == len(K.grid()) and not df.index.duplicated().any()
    yr = df.index.tz_convert(K.TZ).year
    prices = [c for c in df.columns if c.endswith("_eur_mwh")]
    print("=== NaN share per year ===")
    print(df[prices].isna().groupby(yr).mean().round(4).to_string())
    print("\n=== yearly mean [EUR/MWh] ===")
    print(df[prices].groupby(yr).mean().round(1).to_string())
    print("\n=== checks ===")
    print(f"AT filled from DE_AT_LU before the split: {n_fill:,} QH")
    for c in prices:
        fl = "flag_" + c.split("_price")[0] + "_price_da_suspect"
        print(f"{c}: min {df[c].min():.1f}, max {df[c].max():.1f}, suspect {int(df[fl].sum())}")
    if "ch_price_da_eur_mwh" in df and "de_lu_price_da_eur_mwh" in df:
        both = df[["ch_price_da_eur_mwh", "de_lu_price_da_eur_mwh"]].dropna()
        print(f"CH vs DE corr by year:\n{both.groupby(both.index.tz_convert(K.TZ).year).corr().xs('ch_price_da_eur_mwh', level=1)['de_lu_price_da_eur_mwh'].round(3).to_string()}")
    if a.write:
        p = K.write_clean(df, DOMAIN, TABLE, dd)
        print(f"\nwritten: {p} + {TABLE}_dictionary.csv")


if __name__ == "__main__":
    main_()
