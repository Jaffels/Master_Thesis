"""Clean layer: ENTSO-E Load (to-do 3.2, 2 Oct 2026).

Run from Master_Thesis with .venv active:
    python Clean/clean_load.py            # build, check, print summary (no files)
    python Clean/clean_load.py --write    # also write Clean/Data/load/

Output (Clean/Data/load/)
  load.parquet               actual load + day-ahead forecast, 5 zones, 15-min grid
  load_fc_long.parquet       week- / month- / year-ahead forecasts (min / max),
                             as daily / weekly step functions; kept for optional use
  <table>_dictionary.csv     one row per column (source, unit, resolution, rules,
                             first/last value, NaN share)

Rules (table design 2-4, decisions of 2 Oct 2026)
- Both ENTSO-E trees (production_pre2021 + production), UTC, trimmed to the sample.
- Hourly values (CH; FR / IT_NORD in early years) are repeated over their own four
  quarter-hours; 15-min values are used as they are; missing stays NaN.
- DE: until 30 Sep 2018 the series is Germany without Luxembourg (code DE), from
  1 Oct 2018 DE_LU (~0.85 % larger); one column, see regime_de_zone in the master.
- flag_de_lu_load_da_fc_q4_2018_patch: the 3,368 quarter-hours of Q4 2018 filled
  from code DE by patch_de_loadforecast_2018q4.py (2018_q4patch_filled.csv).
- flag_<zone>_load_actual_suspect: actual load < 50 % or > 160 % of the centred
  7-day rolling median (CH 3 days 2023-24, AT 3 days 2015, IT_NORD 2015). Values are
  kept; each view decides whether to drop them.
- Week/month/year-ahead stamps are rounded to the nearest local midnight (some sit
  at 23:00 / 01:00 at the source).
- 8.1 forecast margin: not used (decided 2 Oct 2026).
- Forecast column variants (D-L2): the script stops if a series has more than the
  expected columns, instead of silently picking one.
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

AREAS = {"CH": "ch", "DE_LU": "de_lu", "FR": "fr", "IT_NORD": "it_nord", "AT": "at"}
DOMAIN = "load"
SRC = "ENTSO-E Transparency Platform"
SUSPECT_LO, SUSPECT_HI = 0.5, 1.6
DE_NOTE = "until 30 Sep 2018 Germany without Luxembourg (code DE), from 1 Oct 2018 DE_LU (~0.85 % larger)"


def one_col(df: pd.DataFrame, expected: str, sid: str) -> pd.Series:
    if list(df.columns) != [expected]:
        raise ValueError(f"{sid}: unexpected columns {list(df.columns)} (D-L2 check)")
    return df[expected].astype("float64")


def step_blocks(df: pd.DataFrame, days: int, cols: list[str]) -> pd.DataFrame:
    """Values stamped at a local day/week start -> step function over `days` local days."""
    loc = df.index.tz_convert(K.TZ)
    # round to the nearest local midnight: some source stamps sit at 23:00 / 01:00
    # (DST offset quirk at ENTSO-E) instead of 00:00
    start_local = (loc.tz_localize(None) + pd.Timedelta(hours=12)).normalize()
    end_local = start_local + pd.Timedelta(days=days)
    b = df[cols].copy()
    b["block_start_utc"] = start_local.tz_localize(K.TZ, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
    b["block_end_utc"] = pd.DatetimeIndex(end_local).tz_localize(K.TZ, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
    b = b.reset_index(drop=True).sort_values("block_start_utc")
    # a newer block replaces the tail of an older overlapping one (should not happen; checked)
    b = b.drop_duplicates("block_start_utc", keep="last")
    over = (b.block_start_utc.iloc[1:].to_numpy() < b.block_end_utc.iloc[:-1].to_numpy())
    if over.any():
        b.loc[b.index[:-1][over], "block_end_utc"] = b.block_start_utc.iloc[1:].to_numpy()[over]
    return K.blocks_to_qh(b, cols)


def build():
    g = K.grid()
    main = pd.DataFrame(index=g.index)
    long = pd.DataFrame(index=g.index)
    dmain, dlong = [], []
    for code, a in AREAS.items():
        note_de = DE_NOTE if code == "DE_LU" else ""
        # actual load (6.1.A)
        s = one_col(K.read_entsoe("Load", "actual_load", "A16_realised", code), "Actual Load", f"actual {code}")
        main[f"{a}_load_actual_mw"] = K.hourly_to_qh(s, "repeat")
        dmain.append(dict(column=f"{a}_load_actual_mw", source=SRC, source_series=f"6.1.A actual total load, {code}",
                          unit="MW", resolution_native=str(K.native_step(K.trim(s.to_frame()).index)),
                          aggregation_rule="mean", availability_rule="ex post (about 1 h after delivery)",
                          notes=note_de))
        # day-ahead forecast (6.1.B)
        s = one_col(K.read_entsoe("Load", "load_forecast", "day_ahead_A01", code), "Forecasted Load", f"da {code}")
        main[f"{a}_load_da_fc_mw"] = K.hourly_to_qh(s, "repeat")
        dmain.append(dict(column=f"{a}_load_da_fc_mw", source=SRC, source_series=f"6.1.B day-ahead total load forecast, {code}",
                          unit="MW", resolution_native=str(K.native_step(K.trim(s.to_frame()).index)),
                          aggregation_rule="mean", availability_rule="D-1, before day-ahead gate closure (12:00)",
                          notes=note_de + ("; Q4 2018: 57 days filled from code DE, see flag" if code == "DE_LU" else "")))
        # longer horizons (6.1.C / 6.1.D / 6.1.E): min / max
        for var, h, days, avail, art in [("week_ahead_A31", "wa", 1, "week W-1", "6.1.C"),
                                         ("month_ahead_A32", "ma", 7, "month M-1", "6.1.D"),
                                         ("year_ahead_A33", "ya", 7, "year Y-1", "6.1.E")]:
            try:
                df = K.read_entsoe("Load", "load_forecast", var, code)
            except FileNotFoundError:
                continue
            if sorted(df.columns) != ["Max Forecasted Load", "Min Forecasted Load"]:
                raise ValueError(f"{var} {code}: unexpected columns {list(df.columns)}")
            q = step_blocks(df, days, ["Min Forecasted Load", "Max Forecasted Load"])
            for src_c, mm in [("Min Forecasted Load", "min"), ("Max Forecasted Load", "max")]:
                col = f"{a}_load_{h}_fc_{mm}_mw"
                long[col] = q[src_c]
                dlong.append(dict(column=col, source=SRC, source_series=f"{art} {var.split('_A')[0].replace('_', '-')} load forecast ({mm}), {code}",
                                  unit="MW", resolution_native="1 value per day" if days == 1 else "1 value per week",
                                  aggregation_rule="mean (step function, one value per day / week)",
                                  availability_rule=avail, notes=note_de))
    # suspect actual-load values: < 50 % or > 160 % of the centred 7-day rolling median
    for code, a in AREAS.items():
        x = main[f"{a}_load_actual_mw"]
        med = x.rolling(96 * 7, center=True, min_periods=96).median()
        r = x / med
        col = f"flag_{a}_load_actual_suspect"
        main[col] = ((r < SUSPECT_LO) | (r > SUSPECT_HI)).fillna(False).astype(bool)
        dmain.append(dict(column=col, source="derived", source_series=f"{a}_load_actual_mw", unit="bool",
                          resolution_native="15min", aggregation_rule="any", availability_rule="-",
                          notes=f"value < {SUSPECT_LO:.0%} or > {SUSPECT_HI:.0%} of the centred 7-day rolling median; value kept"))
    # Q4 2018 patch flag
    pf = C.ROOT / "Entsoe/Load/Data/production_pre2021/load_forecast/day_ahead_A01/DE_LU/2018_q4patch_filled.csv"
    patched = K.to_utc(pd.read_csv(pf, index_col=0).index)
    main["flag_de_lu_load_da_fc_q4_2018_patch"] = main.index.isin(patched)
    dmain.append(dict(column="flag_de_lu_load_da_fc_q4_2018_patch", source="derived",
                      source_series="Entsoe/Load/.../DE_LU/2018_q4patch_filled.csv", unit="bool",
                      resolution_native="15min", aggregation_rule="any", availability_rule="-",
                      notes="True where de_lu_load_da_fc_mw was filled from code DE (Germany without LU)"))
    return main, long, dmain, dlong, len(patched)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        nn = df[c].dropna() if df[c].dtype != bool else df[c][df[c]]
        rows.append(dict(column=c, first=nn.index.min(), last=nn.index.max(),
                         nan_share=round(float(df[c].isna().mean()), 4) if df[c].dtype != bool else np.nan,
                         n_true=int(df[c].sum()) if df[c].dtype == bool else np.nan))
    return pd.DataFrame(rows)


def main_() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    main, long, dmain, dlong, n_patch = build()
    bad = K.check_names(list(main.columns) + list(long.columns))
    assert not bad, bad
    assert len(main) == 409_052 and not main.index.duplicated().any()
    assert int(main.flag_de_lu_load_da_fc_q4_2018_patch.sum()) == n_patch == 3368, n_patch
    print("=== load (actual + day-ahead) ===")
    print(summary(main).to_string(index=False))
    print("\n=== load_fc_long (week / month / year ahead) ===")
    print(summary(long).to_string(index=False))
    if a.write:
        p1 = K.write_clean(main, DOMAIN, "load", dmain)
        p2 = K.write_clean(long, DOMAIN, "load_fc_long", dlong)
        print(f"\nwritten: {p1}\n         {p2}\n         + *_dictionary.csv")


if __name__ == "__main__":
    main_()
