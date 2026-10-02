"""Build the gap list for the combined dataset (to-do 2.8, 2 Oct 2026).

Every time series on disk is checked against its own expected time grid between
its first timestamp (or the sample start) and the cut-off. Missing stretches are
written as runs, one row per run. Read-only towards all source data.

Run from Master_Thesis with .venv active:
    python Entsoe/Generation/check_data_gaps.py all --write   # per-unit generation input
    python Clean/build_gap_list.py                            # print summary only
    python Clean/build_gap_list.py --write                    # also write the files

Output (with --write), in Clean/Data/master/:
    gap_list.parquet / gap_list.csv   one row per missing run
    gap_series_summary.csv            one row per series: first/last ts, grid, share missing

Rules
- A timestamp counts as present when at least one value column is non-NaN.
- Grid per series and calendar year, from the data itself: 15 min (>= 90 % of
  steps are 15 min), hourly (<= 1 h, incl. years that mix hourly and 15 min),
  daily, weekly (ISO week, Monday) or monthly. Missing single quarter-hours inside
  an hourly series are therefore not reported.
- kind = gap   missing run inside the series
         head  sample start -> first timestamp (series starts later; not a gap)
         tail  last timestamp -> cut-off (series ends early / not yet published)
         resolution  >= 24 h of hourly values inside a 15-min year (only the :00
               quarter-hour present); a resolution change, not missing data
- known_kind / known_note: filled when a run overlaps an entry of KNOWN below
  (confirmed reasons). KNOWN entries are also written as their own rows
  (detected = False) so that gaps found by other means are in the list too.
- Not covered here (event or document data, gaps are not defined): outages,
  fall-back documents, countertrading, congestion costs, aggregated bids,
  installed capacity, 12.3.F, 8.1 forecast margin. Per-unit generation comes from
  Entsoe/_checks/generation_per_unit_missing_hours.csv (check_data_gaps.py).
- sparse (summary) = more than 100 gap runs of at most 2 buckets on average:
  event-like series (e.g. activation prices that only exist when energy is
  activated) or patchy publication. Their runs are kept but should be read as
  "irregular series", not as individual outages.
- Column-level gaps (one column NaN while others are present) are left to the
  clean layer, except the regelleistung CH FCR price, which is added here.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

TZ = C.TZ_LABEL
S0, S1 = C.start_utc(), C.end_utc()          # sample [S0, S1)

ENTSOE_DOMAINS = ["Load", "Generation", "Transmission", "Balancing"]
ENTSOE_TREES = ["production_pre2021", "production"]
ENTSOE_SKIP = {"actual_generation_unit", "installed_capacity_unit", "installed_capacity",
               "countertrading", "congestion_costs", "aggregated_bids",
               "procured_balancing_capacity", "forecast_margin"}

# Confirmed reasons (2026-09-24 .. 2026-10-02). Local dates, end inclusive.
# series = regex on the series id.
KNOWN = [
    dict(series=r"entsoe/Balancing/imbalance_prices/all/CH", start="2022-12-31", end="2022-12-31",
         kind="source_gap", note="missing at ENTSO-E source (API + Transparency Platform, 26 Sep 2026); requested from Swissgrid"),
    dict(series=r"entsoe/Balancing/imbalance_prices/all/CH", start="2026-01-01", end="2026-01-31",
         kind="covered_by_swissgrid", note="missing in ENTSO-E pull; Swissgrid imbalance files cover it"),
    dict(series=r"swissgrid/tre$", start="2023-12-28", end="2023-12-31",
         kind="source_gap", note="Swissgrid TRE file 2023-12 ends 27 Dec"),
    dict(series=r"swissgrid/tre$", start="2024-09-28", end="2024-09-30",
         kind="source_gap", note="Swissgrid TRE file 2024-09 ends 27 Sep"),
    dict(series=r"swissgrid/auctions/mFRR/(?:up|down)/4h", start="2024-01-01", end="2024-12-31",
         kind="source_gap", note="31 days without daily mFRR auctions in 2024; same in Swissgrid archive of 30 Dec 2025 (checked 2 Oct 2026)"),
    dict(series=r"entsoe/Balancing/imbalance_prices/all/CH", start="2016-07-01", end="2016-09-29",
         kind="source_gap", note="ENTSO-E API returns the same rows as on disk (8 QH Jul, 0 Aug, 88 Sep; test 2 Oct 2026); Swissgrid files start 2023"),
    dict(series=r"swissgrid/auctions/aFRR/down/4h", start="2026-07-20", end="2026-07-26",
         kind="source_gap", note="20:00-24:00 down block not published; same in Swissgrid 2026 file of 1 Oct 2026"),
    dict(series=r"swissgrid/auctions/aFRR/(?:up|down)/4h", start="2026-08-10", end="2026-08-16",
         kind="source_gap", note="20:00-24:00 block (up and down) not published; same in Swissgrid 2026 file of 1 Oct 2026"),
    dict(series=r"entsoe/Load/load_forecast/day_ahead_A01/DE", start="2018-10-01", end="2018-12-31",
         kind="filled_patch", note="57 days filled from DE (without LU) by patch_de_loadforecast_2018q4.py; rows in 2018_q4patch_filled.csv"),
    dict(series=r"entsoe/Load/actual_load/A16_realised/DE", start="2018-10-01", end="2018-10-31",
         kind="source_gap", note="4 hours missing at source (1, 8, 28, 30 Oct 2018)"),
    dict(series=r"entsoe/Generation/per_unit/IT_NORD", start="2026-07-19", end="2026-08-31",
         kind="not_yet_published", note="IT_NORD per-unit ends 18 Jul 2026; re-check ~20 Oct 2026"),
    dict(series=r"entsoe/Generation/per_unit/FR", start="2026-02-09", end="2026-02-26",
         kind="source_gap", note="FR per-unit 9 Feb and 16-26 Feb 2026 missing at source (re-pull 27 Sep 2026)"),
    dict(series=r"entsoe/Transmission/explicit_offered/daily_A01/(?:CH-DE|DE-CH)", start="2015-01-01", end="2017-12-31",
         kind="not_published", note="CH<->DE offered capacity not published before 2018 on any German code"),
    dict(series=r"entsoe/Transmission/scheduled_exchanges/dayahead_A01/(?:CH-DE|DE-CH)", start="2015-01-01", end="2018-09-30",
         kind="not_published", note="day-ahead schedules on this border only from 1 Oct 2018 (DE_LU)"),
    dict(series=r"entsoe/Generation/per_unit/CH", start="2015-03-30", end="2015-06-25",
         kind="series_start", note="CH per-unit: one day on 29 Mar 2015, continuous from 26 Jun 2015"),
]


# --------------------------------------------------------------------------- helpers
def to_utc(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    return idx.tz_convert("UTC").as_unit("ns")   # one unit everywhere (asi8 = ns)


def present(df: pd.DataFrame, ts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Timestamps with at least one non-NaN numeric value."""
    num = df.select_dtypes(include=[np.number, "bool"])
    if num.shape[1] == 0:
        keep = np.ones(len(df), dtype=bool)
    else:
        keep = num.notna().any(axis=1).to_numpy()
    out = ts[keep]
    return out[~out.isna()].unique().sort_values()


def grid_kind(ts: pd.DatetimeIndex) -> str:
    if len(ts) < 3:
        return "D"
    d = np.diff(ts.asi8) / 1e9 / 60          # minutes
    d = d[d > 0]
    if len(d) == 0:
        return "D"
    med = np.median(d)
    if med <= 15 and (d == 15).mean() >= 0.9:
        return "15min"
    if med <= 60:
        return "h"
    if med <= 1440:
        return "D"
    if med <= 7 * 1440:
        return "W"
    return "MS"


def buckets(ts: pd.DatetimeIndex, kind: str) -> pd.DatetimeIndex:
    """Map timestamps to bucket starts (UTC)."""
    if kind in ("15min", "h"):
        return ts.floor(kind).unique()
    loc = ts.tz_convert(TZ).tz_localize(None).normalize()
    if kind == "W":
        loc = loc - pd.to_timedelta(loc.dayofweek, unit="D")
    elif kind == "MS":
        loc = loc.to_period("M").to_timestamp()
    return pd.DatetimeIndex(loc.unique()).tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")


def expected(a: pd.Timestamp, b: pd.Timestamp, kind: str) -> pd.DatetimeIndex:
    """Expected bucket starts in [a, b) (UTC in, UTC out)."""
    if kind in ("15min", "h"):
        return pd.date_range(a.floor(kind), b, freq=kind, inclusive="left")
    la, lb = a.tz_convert(TZ).tz_localize(None), b.tz_convert(TZ).tz_localize(None)
    freq = {"D": "D", "W": "W-MON", "MS": "MS"}[kind]
    start = la.normalize()
    if kind == "W":
        start = start - pd.Timedelta(days=start.dayofweek)
    elif kind == "MS":
        start = start.replace(day=1)
    rng = pd.date_range(start, lb, freq=freq, inclusive="left")
    return rng.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")


STEP = {"15min": pd.Timedelta("15min"), "h": pd.Timedelta("1h"), "D": pd.Timedelta("1D"),
        "W": pd.Timedelta("7D"), "MS": pd.Timedelta("31D")}


def runs(missing: pd.DatetimeIndex, kind: str) -> list[tuple]:
    """Merge consecutive missing buckets into (start, end_excl, n)."""
    if len(missing) == 0:
        return []
    m = missing.sort_values()
    step = STEP[kind]
    tol = step * 1.5 if kind in ("15min", "h") else step + pd.Timedelta("2h")   # DST days
    out, s, p, n = [], m[0], m[0], 1
    for t in m[1:]:
        if t - p <= tol:
            p, n = t, n + 1
            continue
        out.append((s, p + step, n))
        s, p, n = t, t, 1
    out.append((s, p + step, n))
    return out


def check_series(sid: str, source: str, ts: pd.DatetimeIndex, rows: list, summ: list,
                 kind_override: str | None = None) -> None:
    ts = pd.DatetimeIndex(ts).as_unit("ns")
    ts = ts[(ts >= S0) & (ts < S1)]
    if len(ts) == 0:
        summ.append(dict(series=sid, source=source, first_ts=None, last_ts=None, grid="",
                         expected=0, missing=0, share_missing=np.nan, note="no data in sample"))
        return
    ts = ts.sort_values()
    first, last = ts[0], ts[-1]
    years = sorted(set(ts.tz_convert(TZ).year))
    tot_exp = tot_miss = 0
    grids = set()
    for y in years:
        ya = max(pd.Timestamp(y, 1, 1, tz=TZ).tz_convert("UTC"), first)
        yb = min(pd.Timestamp(y + 1, 1, 1, tz=TZ).tz_convert("UTC"), S1)
        yts = ts[(ts >= ya) & (ts < yb)]
        if len(yts) == 0:
            continue
        k = kind_override or grid_kind(yts)
        grids.add(k)
        yb_eff = min(yb, last + STEP[k]) if y == years[-1] else yb
        exp = expected(ya, yb_eff, k)
        have = buckets(yts, k)
        miss = exp.difference(have)
        if k == "15min" and len(miss):
            # hourly stretches inside a 15-min year (resolution change): the :00
            # quarter-hour is present, the other three are not -> not a gap
            hrs_present = have[have.minute == 0]
            cand = miss[miss.floor("h").isin(hrs_present)]
            # only stretches of >= 24 consecutive hourly-looking hours count as a
            # resolution change; shorter ones stay gaps (real missing quarter-hours)
            hrs = pd.DatetimeIndex(cand.floor("h").unique()).sort_values()
            keep_h = []
            if len(hrs):
                grp = (pd.Series(hrs).diff() != pd.Timedelta("1h")).cumsum().to_numpy()
                for gi in np.unique(grp):
                    block = hrs[grp == gi]
                    if len(block) >= 24:
                        keep_h.extend(block)
            res = cand[cand.floor("h").isin(pd.DatetimeIndex(keep_h))] if keep_h else cand[:0]
            miss = miss.difference(res)
            for a, b, n in runs(res, k):
                rows.append(dict(series=sid, source=source, kind="resolution", grid=k, start_utc=a,
                                 end_utc=b, n_missing=n, detected=True))
        tot_exp += len(exp)
        tot_miss += len(miss)
        for a, b, n in runs(miss, k):
            rows.append(dict(series=sid, source=source, kind="gap", grid=k, start_utc=a,
                             end_utc=b, n_missing=n, detected=True))
    # years with no data at all between first and last
    for y in range(years[0] + 1, years[-1]):
        if y not in years:
            a = pd.Timestamp(y, 1, 1, tz=TZ).tz_convert("UTC")
            b = pd.Timestamp(y + 1, 1, 1, tz=TZ).tz_convert("UTC")
            rows.append(dict(series=sid, source=source, kind="gap", grid="year", start_utc=a,
                             end_utc=b, n_missing=1, detected=True))
    k_last = kind_override or grid_kind(ts[-200:])
    if first > S0 + pd.Timedelta("1h"):
        rows.append(dict(series=sid, source=source, kind="head", grid="", start_utc=S0,
                         end_utc=first, n_missing=np.nan, detected=True))
    if last + 2 * STEP[k_last] < S1:      # at least one full bucket missing after the last one
        rows.append(dict(series=sid, source=source, kind="tail", grid="", start_utc=last + STEP[k_last],
                         end_utc=S1, n_missing=np.nan, detected=True))
    n_runs = sum(1 for r in rows if r["series"] == sid and r["kind"] == "gap")
    sparse = bool(tot_exp and n_runs > 100 and tot_miss / max(n_runs, 1) <= 2)
    summ.append(dict(series=sid, source=source,
                     first_ts=first.tz_convert(TZ), last_ts=last.tz_convert(TZ),
                     grid="/".join(sorted(grids)), expected=tot_exp, missing=tot_miss,
                     share_missing=round(tot_miss / tot_exp, 5) if tot_exp else np.nan,
                     gap_runs=n_runs, sparse=sparse,
                     note="many short gaps: event-like or patchy series" if sparse else ""))


# --------------------------------------------------------------------------- sources
def entsoe(rows, summ):
    for dom in ENTSOE_DOMAINS:
        groups: dict[str, list[Path]] = {}
        for tree in ENTSOE_TREES:
            base = C.ROOT / "Entsoe" / dom / "Data" / tree
            for f in base.glob("*/*/*/[0-9]*.parquet"):
                ds, var, area = f.parts[-4], f.parts[-3], f.parts[-2]
                if ds in ENTSOE_SKIP:
                    continue
                groups.setdefault(f"entsoe/{dom}/{ds}/{var}/{area}", []).append(f)
        for sid in sorted(groups):
            parts = []
            for f in groups[sid]:
                df = pd.read_parquet(f)
                parts.append(present(df, to_utc(df.index)))
            ts = pd.DatetimeIndex(np.concatenate([p.as_unit("ns").asi8 for p in parts])).tz_localize("UTC").unique() if parts else pd.DatetimeIndex([], tz="UTC")
            check_series(sid, "ENTSO-E", ts, rows, summ)
        print(f"  ENTSO-E {dom}: {len(groups)} series")


def per_unit(rows, summ):
    f = C.ROOT / "Entsoe" / "_checks" / "generation_per_unit_missing_hours.csv"
    if not f.exists():
        print("  per-unit generation: SKIPPED - run `python Entsoe/Generation/check_data_gaps.py all --write` first")
        return
    d = pd.read_csv(f)
    for _, r in d.iterrows():
        a = pd.Timestamp(r["date"]).tz_localize(TZ).tz_convert("UTC")
        b = (pd.Timestamp(r["date"]) + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
        if a >= S1 or b <= S0:
            continue
        rows.append(dict(series=f"entsoe/Generation/per_unit/{r['area']}", source="ENTSO-E",
                         kind="gap", grid="h-in-day", start_utc=a, end_utc=b,
                         n_missing=int(r["missing_hours"]), detected=True))
    print(f"  per-unit generation: {len(d)} day rows from {f.name}")


def swissgrid(rows, summ):
    sg = C.ROOT / "SwissGrid"
    def tsfile(pattern, sid, tcol="timestamp"):
        fs = sorted(sg.glob(pattern))
        if not fs:
            print(f"  {sid}: no files ({pattern})")
            return
        parts = []
        for f in fs:
            df = pd.read_parquet(f)
            parts.append(present(df.drop(columns=[tcol]), to_utc(df[tcol])))
        ts = pd.DatetimeIndex(np.concatenate([p.as_unit("ns").asi8 for p in parts])).tz_localize("UTC").unique()
        check_series(sid, "Swissgrid", ts, rows, summ)
    tsfile("EnergyOverview/Data/qh/*.parquet", "swissgrid/energy_overview")
    tsfile("ImbalancePrices/Data/qh/*.parquet", "swissgrid/imbalance_prices")
    for sub in ("control_energy", "cross_border", "control_area_balance"):
        tsfile(f"SystemBalance/Data/{sub}/*.parquet", f"swissgrid/system_balance/{sub}")
    # TRE: a quarter-hour is present when the file has bids for any product.
    # (A single product without bids in a quarter-hour is a market state, not a gap;
    #  products start/stop over time, e.g. PVTRE from Jun 2025, RR until Dec 2025.)
    q = pd.concat([pd.read_parquet(f, columns=["delivery_start", "n_bids"])
                   for f in sorted(sg.glob("TRE/Data/qh/*/*.parquet"))])
    q = q[q.n_bids > 0]
    check_series("swissgrid/tre", "Swissgrid", to_utc(q.delivery_start).unique(),
                 rows, summ, kind_override="15min")
    # Auction blocks: a gap is a jump between consecutive blocks of one product line
    a = pd.read_parquet(sg / "Auctions" / "Data" / "auctions.parquet",
                        columns=["product", "direction", "delivery_start", "delivery_end", "duration_h"])
    a["line"] = np.where(a.duration_h >= 160, "week", np.where(a.duration_h >= 23, "day", "4h"))
    for (prod, dr, line), g in a.groupby(["product", "direction", "line"], observed=True):
        sid = f"swissgrid/auctions/{prod}/{dr}/{line}"
        g = g.sort_values("delivery_start").drop_duplicates("delivery_start")
        st, en = to_utc(g.delivery_start), to_utc(g.delivery_end)
        keep = (st < S1) & (en > S0)
        st, en = st[keep], en[keep]
        nmiss = 0
        for i in range(1, len(st)):
            if st[i] > en[i - 1]:
                hours = (st[i] - en[i - 1]).total_seconds() / 3600
                nmiss += hours
                rows.append(dict(series=sid, source="Swissgrid", kind="gap", grid="block",
                                 start_utc=en[i - 1], end_utc=st[i], n_missing=hours, detected=True))
        summ.append(dict(series=sid, source="Swissgrid",
                         first_ts=st.min().tz_convert(TZ) if len(st) else None,
                         last_ts=en.max().tz_convert(TZ) if len(en) else None, grid="block",
                         expected=np.nan, missing=nmiss, share_missing=np.nan,
                         note="missing = hours between consecutive blocks"))
    print("  Swissgrid: done")


def regelleistung(rows, summ):
    base = C.ROOT / "Regelleistung" / "Data" / "production" / "results_CAPACITY"
    for prod in ("FCR", "aFRR", "mFRR"):
        fs = sorted((base / prod).glob("[0-9]*.parquet"))
        if not fs:
            continue
        d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        if "tender_number" in d:
            d = d[d.tender_number == 1]
        days = pd.DatetimeIndex(pd.to_datetime(d.date_from).dt.normalize().unique()).tz_localize(TZ).tz_convert("UTC")
        check_series(f"regelleistung/{prod}", "regelleistung.net", days, rows, summ, kind_override="D")
        if prod == "FCR":
            pc = d["ch_settlementcapacity_price_eur_mw"].combine_first(d["switzerland_settlementcapacity_price_eur_mw"])
            ok = pd.to_datetime(d.date_from[pc.notna()]).dt.normalize().unique()
            cdays = pd.DatetimeIndex(ok).tz_localize(TZ).tz_convert("UTC")
            check_series("regelleistung/FCR/ch_price", "regelleistung.net", cdays, rows, summ, kind_override="D")
    print("  regelleistung.net: done")


def weather_frequency(rows, summ):
    w = pd.read_parquet(C.ROOT / "MeteoSwiss" / "Data" / "weather_ch_hourly.parquet")
    vals = w.drop(columns=["timestamp"] + [c for c in w.columns if c.startswith("coverage")])
    check_series("meteoswiss/weather_ch_hourly", "MeteoSwiss", present(vals, to_utc(w.timestamp)), rows, summ,
                 kind_override="h")
    fq = pd.read_parquet(C.ROOT / "Frequency" / "Data" / "production" / "frequency_merged_15min.parquet")
    check_series("frequency/merged_15min", "TSO/Energy-Charts/Zenodo",
                 present(fq[["n_seconds"]].where(fq.n_seconds > 0), to_utc(fq.index)), rows, summ,
                 kind_override="15min")
    print("  MeteoSwiss + frequency: done")


# --------------------------------------------------------------------------- known
def apply_known(gl: pd.DataFrame) -> pd.DataFrame:
    gl["known_kind"], gl["known_note"] = "", ""
    extra = []
    for k in KNOWN:
        a = pd.Timestamp(k["start"]).tz_localize(TZ).tz_convert("UTC")
        b = (pd.Timestamp(k["end"]) + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
        hit = gl.series.str.contains(k["series"], regex=True) & (gl.start_utc < b) & (gl.end_utc > a)
        gl.loc[hit, "known_kind"] = k["kind"]
        gl.loc[hit, "known_note"] = k["note"]
        extra.append(dict(series=k["series"], source="manual", kind=k["kind"], grid="",
                          start_utc=a, end_utc=b, n_missing=np.nan, detected=False,
                          known_kind=k["kind"], known_note=k["note"]))
    return pd.concat([gl, pd.DataFrame(extra)], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write gap_list + summary to Clean/Data/master/")
    ap.add_argument("--only", default="", help="comma list: entsoe,per_unit,swissgrid,regelleistung,weather")
    a = ap.parse_args()
    if not (C.ROOT / "Entsoe").is_dir():
        sys.exit("Entsoe/ not found next to Clean/")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    only = set(filter(None, a.only.split(",")))
    rows, summ = [], []
    print(f"sample {S0} -> {S1} (UTC, end exclusive)")
    for name, fn in [("entsoe", entsoe), ("per_unit", per_unit), ("swissgrid", swissgrid),
                     ("regelleistung", regelleistung), ("weather", weather_frequency)]:
        if not only or name in only:
            fn(rows, summ)
    gl = pd.DataFrame(rows)
    if gl.empty:
        gl = pd.DataFrame(columns=["series", "source", "kind", "grid", "start_utc", "end_utc", "n_missing", "detected"])
    gl = apply_known(gl)
    gl["hours"] = ((gl.end_utc - gl.start_utc).dt.total_seconds() / 3600).round(2)
    gl["start_local"] = gl.start_utc.dt.tz_convert(TZ)
    gl["end_local"] = gl.end_utc.dt.tz_convert(TZ)
    gl = gl.sort_values(["series", "start_utc"]).reset_index(drop=True)
    sm = pd.DataFrame(summ).sort_values("series")

    det = gl[gl.detected.astype(bool)]
    print("\n=== Series with gaps (detected, kind = gap) ===")
    g = det[det.kind == "gap"].groupby("series").agg(runs=("start_utc", "size"), hours=("hours", "sum"),
                                                      known=("known_kind", lambda s: ",".join(sorted(set(s) - {""}))))
    print(g.sort_values("hours", ascending=False).round(1).to_string() if len(g) else "none")
    print("\n=== Series starting after the sample start (head) / ending before the cut-off (tail) ===")
    ht = det[det.kind.isin(["head", "tail"])][["series", "kind", "start_local", "end_local", "known_kind"]]
    print(ht.to_string(index=False) if len(ht) else "none")
    print(f"\nseries checked: {len(sm)} | gap runs: {(det.kind == 'gap').sum()} | "
          f"unexplained gap runs: {((det.kind == 'gap') & (det.known_kind == '')).sum()}")
    if a.write:
        C.MASTER_DIR.mkdir(parents=True, exist_ok=True)
        gl.to_parquet(C.MASTER_DIR / "gap_list.parquet", index=False)
        gl.to_csv(C.MASTER_DIR / "gap_list.csv", index=False)
        sm.to_csv(C.MASTER_DIR / "gap_series_summary.csv", index=False)
        print(f"written: {C.MASTER_DIR}/gap_list.parquet, gap_list.csv, gap_series_summary.csv")


if __name__ == "__main__":
    main()
