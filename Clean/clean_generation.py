"""Clean layer: ENTSO-E Generation (to-do 3.2 Generation, 2 Oct 2026).

Run from Master_Thesis with .venv active:
    python Clean/clean_generation.py            # build, check, print summary
    python Clean/clean_generation.py --write    # also write Clean/Data/generation/

Output (Clean/Data/generation/), each with <name>_dictionary.csv
  generation.parquet          15-min grid, per zone (ch, de_lu, fr, it_nord, at):
      actual generation by group (16.1.B/C), pumped-storage consumption,
      day-ahead total generation forecast (14.1.C), wind/solar day-ahead and
      intraday forecasts (14.1.D), weekly hydro reservoir level (16.1.D, step)
  installed_capacity.parquet  long table: zone x year x production type, MW (14.1.A)

Rules
- Production types are grouped: solar, wind_on, wind_off, nuclear, hydro_ror,
  hydro_res, hydro_ps (generation), gas (incl. coal-derived gas), coal (hard coal +
  lignite), oil (incl. oil shale, peat), biomass, other (geothermal, waste, marine,
  other, other renewable, energy storage). gen_total = sum of all generation groups.
  A group that a zone never reports is not created; a group that starts later
  (e.g. FR offshore wind 2023) is NaN before its first value.
- hydro_ps_cons = pumping consumption ("Actual Consumption" of pumped storage);
  reported by the neighbours, not by CH.
- Hourly values (CH; early years of FR / IT_NORD / AT) are repeated over their
  quarter-hours; nothing is filled across gaps.
- DE_LU until 30 Sep 2018 = code DE (Germany without LU). Day-ahead generation
  forecast: "generation_forecast" or "Scheduled Aggregated" depending on year/zone.
- Reservoirs: weekly value as a step function over its local week (MWh).
- CH reporting breaks (found 2 Oct 2026): Jan-Jun 2015 actuals set to NaN
  (flag_ch_gen_partial_2015h1); coverage jumps 1 Jan 2020 (actual solar), 1 Jan 2024
  (solar DA forecast), 1 Jan 2025 (run-of-river) -> regime_ch_gen_reporting; CH wind
  DA forecast = 0 in 2020-23 (not reported) -> NaN. For the CH total use Swissgrid
  ch_prod_mw (ENTSO-E covers ~60-70 % until 2024, ~90 % from 2025).
- AT intraday wind forecast = 10000 MW placeholder -> NaN + flag.
- Pumped storage net reporting (3 Oct 2026): FR (until 19 Dec 2024) and IT_NORD report per
  quarter-hour either PS generation or PS consumption; the other side NaN -> 0 (both NaN
  stays a gap). FR reports gross from 20 Dec 2024.
- Per-unit generation (16.1.A) is not carried into the clean layer (only needed if
  unit-level features are built later; gaps are in the gap list).
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
SRC = "ENTSO-E Transparency Platform"
DE_NOTE = "DE_LU until 30 Sep 2018 = code DE (Germany without LU)"
GROUP = {"Solar": "solar", "Wind Onshore": "wind_on", "Wind Offshore": "wind_off", "Nuclear": "nuclear",
         "Hydro Run-of-river and pondage": "hydro_ror", "Hydro Water Reservoir": "hydro_res",
         "Hydro Pumped Storage": "hydro_ps", "Fossil Gas": "gas", "Fossil Coal-derived gas": "gas",
         "Fossil Hard coal": "coal", "Fossil Brown coal/Lignite": "coal", "Fossil Oil": "oil",
         "Fossil Oil shale": "oil", "Fossil Peat": "oil", "Biomass": "biomass", "Geothermal": "other",
         "Waste": "other", "Marine": "other", "Other": "other", "Other renewable": "other",
         "Energy storage": "other"}
ORDER = ["solar", "wind_on", "wind_off", "nuclear", "hydro_ror", "hydro_res", "hydro_ps", "gas",
         "coal", "oil", "biomass", "other"]


def entry(col, series, unit, res, agg, avail, note=""):
    return dict(column=col, source=SRC, source_series=series, unit=unit, resolution_native=res,
                aggregation_rule=agg, availability_rule=avail, notes=note)


def actual_per_type(code: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """Generation per group (wide) and pumped-storage consumption, native resolution, UTC."""
    raw = K.read_entsoe("Generation", "actual_generation", "per_type", code, dedup=False)
    if "level_0" in raw.columns:                       # long layout (neighbours)
        raw = raw.reset_index()
        raw = raw.drop_duplicates(["ts_utc", "level_0"], keep="last")
        unknown = set(raw.level_0) - set(GROUP)
        if unknown:
            raise ValueError(f"{code}: unknown production types {sorted(unknown)}")
        raw["grp"] = raw.level_0.map(GROUP)
        gen = (raw.groupby(["ts_utc", "grp"])["Actual Aggregated"].sum(min_count=1)
               .unstack("grp"))
        ps = raw[raw.level_0 == "Hydro Pumped Storage"].set_index("ts_utc")["Actual Consumption"]
        ps = ps if ps.notna().any() else None
    else:                                              # wide layout (CH)
        raw = raw[~raw.index.duplicated(keep="last")]
        unknown = set(raw.columns) - set(GROUP)
        if unknown:
            raise ValueError(f"{code}: unknown production types {sorted(unknown)}")
        gen = raw.T.groupby(raw.columns.map(GROUP)).sum(min_count=1).T
        ps = None
    gen = gen[[g for g in ORDER if g in gen.columns]].astype("float64")
    return gen, ps


def build():
    g = K.grid().index
    out, dic = pd.DataFrame(index=g), []
    for code, a in AREAS.items():
        dn = DE_NOTE if code == "DE_LU" else ""
        gen, ps = actual_per_type(code)
        res = str(K.native_step(K.trim(gen).index))
        q = K.hourly_to_qh(gen, "repeat")
        for grp in gen.columns:
            col = f"{a}_gen_{grp}_mw"
            out[col] = q[grp]
            dic.append(entry(col, f"16.1.B/C actual generation per type, {code}, group {grp}", "MW", res, "mean",
                             "ex post (about 1 h after delivery)", dn))
        out[f"{a}_gen_total_mw"] = q.sum(axis=1, min_count=1)
        dic.append(entry(f"{a}_gen_total_mw", f"sum of {a}_gen_*_mw groups", "MW", res, "mean",
                         "ex post", (dn + "; " if dn else "") + "sum of the reported groups"))
        if ps is not None:
            out[f"{a}_gen_hydro_ps_cons_mw"] = K.hourly_to_qh(ps.astype("float64"), "repeat")
            dic.append(entry(f"{a}_gen_hydro_ps_cons_mw", f"16.1.B/C pumped storage, Actual Consumption, {code}", "MW",
                             res, "mean", "ex post", "pumping consumption (positive)"))
        # day-ahead total generation forecast (14.1.C)
        f = K.read_entsoe("Generation", "generation_forecast", "day_ahead", code)
        s = f["generation_forecast"] if "generation_forecast" in f else None
        if "Scheduled Aggregated" in f:
            s = f["Scheduled Aggregated"] if s is None else s.combine_first(f["Scheduled Aggregated"])
        out[f"{a}_gen_total_da_fc_mw"] = K.hourly_to_qh(s.astype("float64"), "repeat")
        dic.append(entry(f"{a}_gen_total_da_fc_mw", f"14.1.C day-ahead aggregated generation forecast, {code}", "MW",
                         str(K.native_step(K.trim(s.to_frame()).index)), "mean", "D-1 (by 18:00)",
                         (dn + "; " if dn else "") + "column 'generation_forecast' or 'Scheduled Aggregated' depending on year"))
        # wind / solar forecasts (14.1.D)
        for var, tag, avail in (("day_ahead", "da_fc", "D-1 (by 18:00)"), ("intraday", "id_fc", "intraday (updated during the day)")):
            try:
                w = K.read_entsoe("Generation", "wind_solar_forecast", var, code)
            except FileNotFoundError:
                continue
            wq = K.hourly_to_qh(w.astype("float64"), "repeat")
            for src, grp in (("Solar", "solar"), ("Wind Onshore", "wind_on"), ("Wind Offshore", "wind_off")):
                if src in w:
                    col = f"{a}_gen_{grp}_{tag}_mw"
                    out[col] = wq[src]
                    dic.append(entry(col, f"14.1.D {var} wind/solar forecast, {code}, {src}", "MW",
                                     str(K.native_step(K.trim(w[[src]].dropna()).index)), "mean", avail, dn))
        # reservoirs (16.1.D), weekly
        try:
            r = K.read_entsoe("Generation", "water_reservoirs", "weekly", code)
        except FileNotFoundError:
            continue
        col = f"{a}_hydro_reservoir_mwh"
        out[col] = K.local_steps(r.astype("float64"), 7, ["water_reservoirs"])["water_reservoirs"]
        dic.append(entry(col, f"16.1.D aggregated filling of water reservoirs, {code}", "MWh", "1 value per week",
                         "mean (step function per week)", "published the following week",
                         "stored energy; weekly value repeated over its local week"))
    # ---- pumped storage reported net (found 3 Oct 2026, EDA) ----
    # FR (until 19 Dec 2024) and IT_NORD (whole sample) publish per quarter-hour either
    # generation or pumping consumption, never both: the missing side means 0, not a gap.
    # Rule: where exactly one side is NaN, set it to 0; both NaN stays a gap. FR reports
    # both sides (gross) from 20 Dec 2024 -> reporting break, noted in the dictionary.
    for a in AREAS.values():
        g, c = f"{a}_gen_hydro_ps_mw", f"{a}_gen_hydro_ps_cons_mw"
        if g in out and c in out:
            one = out[g].isna() ^ out[c].isna()
            out.loc[one & out[g].isna(), g] = 0.0
            out.loc[one & out[c].isna(), c] = 0.0
            for d in dic:
                if d["column"] in (g, c):
                    d["notes"] = (d["notes"] + "; " if d["notes"] else "") + (
                        "NaN set to 0 where only the other pumped-storage side is reported (net reporting)"
                        + ("; FR reports gross (both sides) from 20 Dec 2024" if a == "fr" else ""))
    # (gen_total is unchanged: sum(min_count=1) already treated the NaN side as missing)
    # ---- CH reporting breaks and placeholders (found 2 Oct 2026) ----
    loc_ts = lambda d: pd.Timestamp(d, tz=K.TZ).tz_convert("UTC")
    ch_act = [c for c in out.columns if c.startswith("ch_gen_") and not c.endswith(("_da_fc_mw", "_id_fc_mw"))]
    early = out.index < loc_ts("2015-07-01")
    out["flag_ch_gen_partial_2015h1"] = early & out[ch_act].notna().any(axis=1).to_numpy()
    out.loc[early, ch_act] = np.nan
    wz = (out.index >= loc_ts("2020-01-01")) & (out.index < loc_ts("2024-01-01"))
    out.loc[wz & (out["ch_gen_wind_on_da_fc_mw"] == 0), "ch_gen_wind_on_da_fc_mw"] = np.nan
    out["regime_ch_gen_reporting"] = pd.Categorical(
        np.select([out.index >= loc_ts("2025-01-01"), out.index >= loc_ts("2024-01-01"), out.index >= loc_ts("2020-01-01")],
                  ["r2025", "r2024", "r2020"], "r2015"), categories=["r2015", "r2020", "r2024", "r2025"])
    ph = out["at_gen_wind_on_id_fc_mw"] == 10000
    out["flag_at_gen_wind_on_id_fc_placeholder"] = ph.to_numpy()
    out.loc[ph, "at_gen_wind_on_id_fc_mw"] = np.nan
    dic += [entry("flag_ch_gen_partial_2015h1", "derived", "bool", "15min", "any", "-",
                  "CH actual generation Jan-Jun 2015 mostly unreported (total 11-180 MW); set to NaN"),
            entry("regime_ch_gen_reporting", "derived", "category", "15min", "mode", "-",
                  "CH ENTSO-E reporting coverage: r2015 (from Jul 2015), r2020 (actual solar coverage jumps; "
                  "wind DA fc not reported 2020-23 -> NaN), r2024 (solar DA fc coverage jumps), "
                  "r2025 (run-of-river coverage jumps). Do not compare CH solar / ror / total across regimes"),
            entry("flag_at_gen_wind_on_id_fc_placeholder", "derived", "bool", "15min", "any", "-",
                  "AT intraday wind forecast exactly 10000 MW (placeholder, capacity ~4 GW); set to NaN")]
    for d in dic:
        if d["column"] in ("ch_gen_solar_mw", "ch_gen_hydro_ror_mw", "ch_gen_total_mw", "ch_gen_solar_da_fc_mw",
                           "ch_gen_wind_on_da_fc_mw"):
            d["notes"] = (d["notes"] + "; " if d["notes"] else "") + "reporting coverage changes, see regime_ch_gen_reporting; CH total: prefer Swissgrid ch_prod_mw"
    return out, dic


def installed_capacity():
    rows = []
    for code, a in AREAS.items():
        d = K.read_entsoe("Generation", "installed_capacity", "per_type", code)
        for ts, row in d.iterrows():
            for t, v in row.items():
                if pd.notna(v):
                    rows.append(dict(zone=a, year=int(ts.tz_convert(K.TZ).year), production_type=t,
                                     group=GROUP.get(t, "other"), capacity_mw=float(v)))
    t = pd.DataFrame(rows).drop_duplicates(["zone", "year", "production_type"], keep="last")
    t = t[(t.year >= 2015) & (t.year <= 2026)].sort_values(["zone", "year", "production_type"]).reset_index(drop=True)
    dic = [dict(column=c, source=SRC, source_series="14.1.A installed generation capacity aggregated (per year)",
                notes={"capacity_mw": "MW, as published for the start of the year",
                       "group": "same grouping as generation.parquet"}.get(c, "")) for c in t.columns]
    return t, dic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    out, dic = build()
    cap, cdic = installed_capacity()
    bad = K.check_names(out.columns)
    assert not bad, bad
    assert len(out) == 409_052
    num = out.select_dtypes(include="number")          # summary only for numeric columns
    s = pd.DataFrame({"first": num.apply(lambda c: c.first_valid_index()),
                      "median": num.median().round(0), "max": num.max().round(0),
                      "nan_after_first": [round(num.loc[num[c].first_valid_index():, c].isna().mean(), 4)
                                          if num[c].first_valid_index() is not None else np.nan for c in num.columns]})
    s["first"] = s["first"].dt.tz_convert(K.TZ).dt.strftime("%Y-%m-%d")
    print(s.to_string())
    print("\nflags:", {c: int(out[c].sum()) for c in out.columns if c.startswith("flag_")},
          "| regime_ch_gen_reporting:", out["regime_ch_gen_reporting"].value_counts().to_dict())
    print(f"\ninstalled_capacity: {len(cap)} rows, zones {sorted(cap.zone.unique())}, years {cap.year.min()}-{cap.year.max()}")
    # cross-check CH total vs Swissgrid production (Energy Overview), if written
    eo = C.DATA_DIR / "swissgrid" / "energy_overview.parquet"
    if eo.exists():
        e = pd.read_parquet(eo, columns=["ts_utc", "ch_prod_mw"]).set_index("ts_utc")["ch_prod_mw"]
        j = pd.concat([out["ch_gen_total_mw"], e], axis=1).dropna()
        yr = j.groupby(j.index.year).mean()
        print("\nCH ENTSO-E total / Swissgrid production (annual mean ratio):",
              (yr["ch_gen_total_mw"] / yr["ch_prod_mw"]).round(3).to_dict())
    if a.write:
        print("written:", K.write_clean(out, "generation", "generation", dic))
        print("written:", K.write_long(cap, "generation", "installed_capacity", cdic))


if __name__ == "__main__":
    main()
