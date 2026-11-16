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
                               + FCR Cooperation members be, nl, dk1, si, cz (26 Oct 2026;
                               optional: skipped with a notice until their pull has run)
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
- CH: EPEX SPOT CH day-ahead prices (Clean/Data/epex/da.parquet, clean_epex.py) replace the
  ENTSO-E values up to 30 Sep 2025 (decided 16 Nov 2026). ENTSO-E 12.1.D CH is shifted by one
  hour against EPEX on 28 Apr - 29 Oct 2017 (94 % of days; EPEX lines up better with FR / DE)
  and differs on ~11 days in 2015 and 1 day in 2018; identical otherwise. From 1 Oct 2025
  (15-min SDAC MTU; EPEX file is hourly) and in 2026 (EPEX not licensed) ENTSO-E stays.
  No new column (feature sets unchanged); the replacement is printed and noted in the dictionary.
  Without Clean/Data/epex/da.parquet the script stops (run clean_epex.py --write first).
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
AREAS = {"CH": "ch", "DE": "de_lu", "AT": "at", "FR": "fr", "IT_NORD": "it_nord", **K.FCR_MEMBERS}
OPTIONAL = set(K.FCR_MEMBERS)                # missing on disk -> skipped, not an error
LIMIT_LO, LIMIT_HI = -500.0, 4000.0          # SDAC harmonised min / max clearing price (since 2022)
SPLIT = pd.Timestamp("2018-10-01", tz=K.TZ).tz_convert("UTC")
EPEX_DA = C.DATA_DIR / "epex" / "da.parquet"
EPEX_UNTIL = pd.Timestamp("2025-10-01", tz=K.TZ).tz_convert("UTC")   # exclusive: 15-min DA from here


def epex_ch(index: pd.DatetimeIndex) -> pd.Series:
    """EPEX CH day-ahead price on the grid (NaN from EPEX_UNTIL on)."""
    if not EPEX_DA.exists():
        raise FileNotFoundError(f"{EPEX_DA} missing: run python Clean/clean_epex.py --write first")
    e = pd.read_parquet(EPEX_DA, columns=["ts_utc", "ch_price_da_epex_eur_mwh"])
    s = e.set_index(K.to_utc(e.pop("ts_utc")))["ch_price_da_epex_eur_mwh"].astype("float64").reindex(index)
    return s.where(s.index < EPEX_UNTIL)


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
        try:
            raw = read_area(area)
        except FileNotFoundError:
            if area not in OPTIONAL:
                raise
            print(f"notice: no day-ahead prices on disk for {area} (FCR member) - skipped; "
                  f"run Entsoe/Prices/entsoe_prices_pull.py --areas {area}")
            continue
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
    # CH: EPEX replaces ENTSO-E where available (see docstring)
    ep = epex_ch(out.index)
    ent = out["ch_price_da_eur_mwh"].copy()
    use = ep.notna()
    changed = use & ~(np.isclose(ep, ent, atol=0.01) & ent.notna())
    out.loc[use, "ch_price_da_eur_mwh"] = ep[use]
    info["ch_epex"] = dict(replaced=int(use.sum()), changed=changed,
                           filled=int((use & ent.isna()).sum()))
    i_ch = [r["column"] for r in dd].index("ch_price_da_eur_mwh")
    dd[i_ch]["source"] = "EPEX SPOT (to 30 Sep 2025) / ENTSO-E Transparency Platform (from 1 Oct 2025)"
    dd[i_ch]["source_series"] = "EPEX CH Day-Ahead Auction hourly prices; ENTSO-E 12.1.D CH from 1 Oct 2025"
    dd[i_ch]["notes"] = ("EPEX replaces ENTSO-E up to 30 Sep 2025 (decided 16 Nov 2026): ENTSO-E CH shifted by "
                         "1 h on 28 Apr - 29 Oct 2017, minor differences 2015 / 2018")
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
        if col not in out:
            continue
        out[fl] = ((out[col] < LIMIT_LO) | (out[col] > LIMIT_HI)).fillna(False).astype(bool)
        dd.append(dict(column=fl, source="derived", source_series=col, unit="bool", resolution_native="15min",
                       aggregation_rule="any", availability_rule="-",
                       notes=f"price outside the SDAC limits [{LIMIT_LO:.0f}, {LIMIT_HI:.0f}] EUR/MWh; value kept"))
    return out, dd, int(fill.sum()), info["ch_epex"]


def main_() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    df, dd, n_fill, ch_ep = build()
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
    ch_changed = ch_ep["changed"]
    print(f"CH: EPEX values used for {ch_ep['replaced']:,} QH (to 30 Sep 2025), of which changed vs ENTSO-E "
          f"{int(ch_changed.sum()):,}, filled ENTSO-E gaps {ch_ep['filled']:,}")
    if ch_changed.any():
        loc = ch_changed[ch_changed].index.tz_convert(K.TZ)
        print("CH changed QH per year-month:\n"
              + pd.Series(1, index=loc).groupby([loc.year, loc.month]).sum().unstack(fill_value=0).to_string())
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
