"""Clean layer: ENTSO-E Outages and aFRR platform fall-back (to-do 3.6 Outages, 2 Oct 2026).

Run from Master_Thesis with .venv active:
    python Clean/clean_outages.py            # build, check, print summary
    python Clean/clean_outages.py --write    # also write Clean/Data/outages/

Output (Clean/Data/outages/), each with <name>_dictionary.csv
  outages.parquet         15-min grid
    {zone}_outage_gen_{planned,forced}_mw   unavailable generation-unit capacity (15.1.A/B)
    {zone}_outage_prod_{planned,forced}_mw  unavailable production-unit capacity (15.1.C/D)
    {from}_{to}_outage_trans_{planned,forced}_n   concurrent transmission outages per border (10.1.A/B)
    ch_afrr_platform_fallback, ch_afrr_fallback_share, flag_ch_afrr_fallback_published_late
  outage_events.parquet   long table: one row per active availability period after cleaning
                          (zone, kind, unit, start/end, unavailable MW, created) -> ex-ante views

Rules
- Unavailable MW = nominal power - available quantity (>= 0), per availability period.
- Cancelled (docstatus A09) and withdrawn (A13) documents are dropped. Test: Leibstadt
  (1,233 MW) shows up to 6,273 MW unavailable with cancelled documents, 1,233 without.
  Newer files carry no docstatus at all.
- Overlapping active documents of the same unit (6-21 % of unavailable MW-time) are
  resolved per unit: the most recently created document wins for every quarter-hour;
  the planned / forced split follows the winning document. Period boundaries are
  rounded to the nearest quarter-hour.
- Duplicates across year files: (doc_mrid, revision, start, end) de-duplicated; highest
  revision per document kept.
- Pre-split DE files (DE_AT_LU bidding zone, until 30 Sep 2018) contain Austrian units:
  rows with the AT bidding-zone EIC are dropped from the DE_LU zone.
- aFRR fall-back (IF 3.10): documents de-duplicated across the business-type folders;
  ch_afrr_fallback_share = share of the local day in fall-back; ch_afrr_platform_fallback
  = 1 if share >= 0.5. 2022-2024 documents were created on 15 Apr 2025 (median lag
  998 / 618 / 201 days): not known in real time before Apr 2025 ->
  flag_ch_afrr_fallback_published_late (created > 2 days after start).
- A quarter-hour without any active document = 0 MW / 0 outages (not NaN): absence of
  an outage message means no reported outage. Coverage starts with each series' first
  document.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import config as C  # noqa: E402

OD = C.ROOT / "Entsoe" / "Outages" / "Data"
TREES = ("production_pre2021", "production")
SRC = "ENTSO-E Transparency Platform"
AT_EIC = "10YAT-APG------L"
ZONES = {"CH": "ch", "DE_LU": "de_lu", "FR": "fr", "IT_NORD": "it_nord", "AT": "at"}
RES = "ts.production_registeredresource"
COLS = ["doc_mrid", "revision", "docstatus", "created", "businesstype", "start", "end", "quantity",
        "unavail_start", "unavail_end", "ts.biddingzone_domain.mrid", f"{RES}.mrid", f"{RES}.name",
        f"{RES}.psrtype.powersystemresources.nominalp", "ts.asset_registeredresource.mrid",
        "ts.asset_registeredresource.name"]


def read_folder(kind: str, area: str) -> pd.DataFrame:
    parts = []
    for tree in TREES:
        for f in sorted((OD / tree / kind).glob(f"*/{area}/[0-9]*.parquet")):
            have = set(pq.read_schema(f).names)
            d = pd.read_parquet(f, columns=[c for c in COLS if c in have]).reindex(columns=COLS)
            d["folder"] = f.parent.parent.name
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=COLS + ["folder"])
    d = pd.concat(parts, ignore_index=True)
    for c in ("created", "start", "end", "unavail_start", "unavail_end"):
        d[c] = pd.to_datetime(d[c], utc=True).dt.as_unit("ns")   # grid arithmetic is in ns
    d = d[~d.docstatus.isin(["A09", "A13"])]
    d = d.drop_duplicates(["doc_mrid", "revision", "start", "end"])
    maxrev = d.groupby("doc_mrid").revision.transform("max")
    return d[d.revision.fillna(0) == maxrev.fillna(0)]


def qh_index(ts: pd.Series, g0: int) -> np.ndarray:
    ns = pd.to_datetime(ts, utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    return np.rint((ns - g0) / K.QH.value).astype(np.int64)


def unit_profiles(d: pd.DataFrame, n: int, g0: int):
    """Latest-created-wins per unit -> (total MW, forced MW) arrays on the grid."""
    tot = np.zeros(n, dtype=np.float64)
    forced = np.zeros(n, dtype=np.float64)
    d = d.assign(a=qh_index(d.start, g0).clip(0, n), b=qh_index(d.end, g0).clip(0, n))
    d = d[d.b > d.a].sort_values("created", kind="stable")
    for _, u in d.groupby(f"{RES}.mrid", sort=False):
        lo, hi = int(u.a.min()), int(u.b.max())
        val = np.zeros(hi - lo)
        frc = np.zeros(hi - lo, dtype=bool)
        for a, b, v, bt in zip(u.a.to_numpy(), u.b.to_numpy(), u.unav.to_numpy(), u.businesstype.to_numpy()):
            val[a - lo:b - lo] = v                     # later-created overwrites earlier
            frc[a - lo:b - lo] = (bt == "A54")
        tot[lo:hi] += val
        forced[lo:hi] += np.where(frc, val, 0.0)
    return tot, forced


def build():
    g = K.grid().index
    g0, n = g.asi8[0], len(g)
    out, dic, ev = pd.DataFrame(index=g), [], []
    for kind, tag, art in (("gen_unit_outages", "gen", "15.1.A/B generation units"),
                           ("prod_unit_outages", "prod", "15.1.C/D production units")):
        for code, z in ZONES.items():
            d = read_folder(kind, code)
            if d.empty:
                continue
            if code == "DE_LU":
                d = d[d["ts.biddingzone_domain.mrid"] != AT_EIC]
            d["unav"] = (d[f"{RES}.psrtype.powersystemresources.nominalp"] - d["quantity"]).clip(lower=0)
            d = d[d.unav > 0]
            tot, frc = unit_profiles(d, n, g0)
            first = d.start.min()
            mask = g >= first
            for col, arr, note in ((f"{z}_outage_{tag}_planned_mw", tot - frc, "planned (A53)"),
                                   (f"{z}_outage_{tag}_forced_mw", frc, "forced (A54)")):
                out[col] = np.where(mask, arr, np.nan)
                dic.append(dict(column=col, source=SRC, source_series=f"{art}, {code}", unit="MW",
                                resolution_native="event (minute periods)", aggregation_rule="mean",
                                availability_rule="ex post as final; ex-ante views must use outage_events.created",
                                notes=f"{note} unavailable capacity; cancelled docs dropped; latest-created document wins per unit"
                                      + ("; Austrian units removed from pre-split DE_AT_LU files" if code == "DE_LU" else "")))
            ev.append(pd.DataFrame({"zone": z, "kind": tag, "outage_type": np.where(d.businesstype == "A54", "forced", "planned"),
                                    "unit_mrid": d[f"{RES}.mrid"], "unit_name": d[f"{RES}.name"],
                                    "start_utc": d.start, "end_utc": d.end, "unavailable_mw": d.unav,
                                    "created_utc": d.created, "doc_mrid": d.doc_mrid}))
    # transmission outages: concurrent events per border
    for f, t in [("CH", nb) for nb in ("DE_LU", "FR", "IT_NORD", "AT")] + [(nb, "CH") for nb in ("DE_LU", "FR", "IT_NORD", "AT")]:
        d = read_folder("transmission_outages", f"{f}_to_{t}")
        if d.empty:
            continue
        a = qh_index(d.start, g0).clip(0, n)
        b = qh_index(d.end, g0).clip(0, n)
        for bt, tag in (("A53", "planned"), ("A54", "forced")):
            m = (d.businesstype == bt).to_numpy() & (b > a)
            diff = np.zeros(n + 1)
            # count distinct documents: one interval per (doc, asset) -> use unavail_start/end of the doc
            dd = d[m].drop_duplicates(["doc_mrid", "ts.asset_registeredresource.mrid"])
            aa, bb = qh_index(dd.unavail_start, g0).clip(0, n), qh_index(dd.unavail_end, g0).clip(0, n)
            np.add.at(diff, aa, 1)
            np.add.at(diff, bb, -1)
            col = f"{ZONES[f]}_{ZONES[t]}_outage_trans_{tag}_n"
            out[col] = np.cumsum(diff)[:n]
            dic.append(dict(column=col, source=SRC, source_series=f"10.1.A/B transmission outages, {f}->{t}", unit="count",
                            resolution_native="event", aggregation_rule="mean",
                            availability_rule="ex post as final", notes=f"{tag} concurrent outages (documents x assets)"))
    # aFRR platform fall-back (CH)
    fb = read_folder("fallback_afrr", "CH").drop_duplicates(["doc_mrid", "revision"])
    a = qh_index(fb.unavail_start, g0).clip(0, n)
    b = qh_index(fb.unavail_end, g0).clip(0, n)
    cover = np.zeros(n + 1)
    np.add.at(cover, a, 1)
    np.add.at(cover, b, -1)
    infb = np.cumsum(cover)[:n] > 0
    late = np.zeros(n + 1)
    lm = ((fb.created - fb.unavail_start) > pd.Timedelta(days=2)).to_numpy()
    np.add.at(late, a[lm], 1)
    np.add.at(late, b[lm], -1)
    day = g.tz_convert(K.TZ).normalize()
    share = pd.Series(infb.astype(float), index=g).groupby(day).transform("mean")
    first_fb = fb.unavail_start.min()
    out["ch_afrr_fallback_share"] = np.where(g >= first_fb.normalize(), share.to_numpy(), 0.0)
    out["ch_afrr_platform_fallback"] = (out["ch_afrr_fallback_share"] >= 0.5).astype("int8")
    out["flag_ch_afrr_fallback_published_late"] = np.cumsum(late)[:n] > 0
    dic += [dict(column="ch_afrr_fallback_share", source=SRC, source_series="IF aFRR 3.10 fall-back documents, CH", unit="share",
                 resolution_native="daily (documents start at midnight)", aggregation_rule="mean",
                 availability_rule="from Apr 2025 next day; 2022-2024 published 15 Apr 2025 (not known in real time)",
                 notes="share of the local day in aFRR platform fall-back; 0 before the first document (Jun 2022)"),
            dict(column="ch_afrr_platform_fallback", source="derived", source_series="ch_afrr_fallback_share", unit="0/1",
                 resolution_native="daily", aggregation_rule="max", availability_rule="see share",
                 notes="1 if >= 50 % of the local day in fall-back; regime indicator (table design 4.5)"),
            dict(column="flag_ch_afrr_fallback_published_late", source="derived", source_series="fall-back documents", unit="bool",
                 resolution_native="15min", aggregation_rule="any", availability_rule="-",
                 notes="document created > 2 days after the fall-back started (2022-2024: up to ~1,000 days)")]
    events = pd.concat(ev, ignore_index=True) if ev else pd.DataFrame()
    return out, dic, events, fb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    out, dic, events, fb = build()
    bad = K.check_names(out.columns)
    assert not bad, bad
    assert len(out) == 409_052
    num = out.select_dtypes(include="number")
    yr = num.groupby(num.index.tz_convert(K.TZ).year).mean().round(0)
    print(yr.T.to_string())
    # checks
    loc = lambda d: pd.Timestamp(d, tz=K.TZ).tz_convert("UTC")
    print("\nLeibstadt check, CH gen unavailable on 2017-01-15 12:00:", out.loc[loc("2017-01-15 12:00"), ["ch_outage_gen_planned_mw", "ch_outage_gen_forced_mw"]].to_dict())
    fy = out.groupby(out.index.tz_convert(K.TZ).year)["ch_afrr_fallback_share"].mean().round(3)
    print("aFRR fall-back share by year:", fy[fy > 0].to_dict(), "| expected ~0.17 (2022, from Jun), 0.01, 0.42, 0.99, 1.0")
    print(f"outage_events: {len(events)} rows; fall-back documents after dedup: {len(fb)}")
    if a.write:
        print("written:", K.write_clean(out, "outages", "outages", dic))
        edic = [dict(column=c, source=SRC, source_series="ENTSO-E 15.1.A-D outage documents (cleaned)",
                     notes={"created_utc": "document creation time: use for ex-ante features (known at time t)",
                            "unavailable_mw": "nominal power - available quantity"}.get(c, "")) for c in events.columns]
        print("written:", K.write_long(events, "outages", "outage_events", edic))


if __name__ == "__main__":
    main()
