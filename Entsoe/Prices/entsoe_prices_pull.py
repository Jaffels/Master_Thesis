"""
ENTSO-E day-ahead prices (Transparency Platform 12.1.D, document A44): probe + pull.

To-do 03.10.2026, section 7 item 1. Free substitute for the EPEX data until it arrives:
  * RQ1a / RQ2: spot-price driver, ex ante only as LOOKBACKS (the D-1 auction result is
    published ~12:45-13:00 CET, after every CH balancing gate closure, see table_design.md);
  * RQ3: day-ahead price of the quarter-hour is public by the D-1 18:00 origin -> usable
    as an __fc_d1 feature, and for a spread-based spike definition (imbalance - DA).

Areas (one parquet per area and calendar year, raw, reconciled later in Clean/):
    CH         Swiss day-ahead auction (EPEX, not coupled with SDAC)
    DE         DE_AT_LU before 1 Oct 2018, DE_LU from then (Entsoe/pre_split.py)
    AT         from 1 Oct 2018 only (before: part of DE_AT_LU -> use the DE series)
    FR, IT_NORD
Resolution is kept as published: hourly, and 15 min where the zone switched with the SDAC
15-min MTU (1 Oct 2025). The manifest records the resolution per area-year.

Request machinery (Throttle, call_with_retry, load_api_key, tidy, profile) is reused from
Entsoe/Load/entsoe_load_probe.py so logging, retries and the parquet schema match the other
ENTSO-E pulls. query_day_ahead_prices is @year_limited in entsoe-py -> one call per year.

Usage (macOS), from Master_Thesis with .venv active:
    python Entsoe/Prices/entsoe_prices_pull.py --plan                  # no API calls
    python Entsoe/Prices/entsoe_prices_pull.py --probe                 # one month per area/code
    python Entsoe/Prices/entsoe_prices_pull.py                         # 2015-01-01 -> 1st of this month
    python Entsoe/Prices/entsoe_prices_pull.py --start 2025-01-01 --force
    python Entsoe/Prices/entsoe_prices_pull.py --check                 # coverage from disk, no API calls

Output: Entsoe/Prices/Data/production/day_ahead_price/<AREA>/<YEAR>.parquet
        Entsoe/Prices/Data/production/_pull_manifest.csv
        Entsoe/Prices/Data/_probe_<date>.csv          (--probe)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Load"))
sys.path.insert(0, str(HERE.parent))
import entsoe_load_probe as probe  # noqa: E402  (shared request machinery)
import pre_split  # noqa: E402

log = logging.getLogger("prices-pull")
TZ = probe.TZ
DATASET = "day_ahead_price"
VALUE_COL = "price_eur_mwh"

# area -> (code before the DE split or None, code from the split or None)
AREAS: dict[str, tuple[str | None, str | None]] = {
    "CH": ("CH", "CH"),
    "DE": ("DE_AT_LU", "DE_LU"),
    "AT": (None, "AT"),
    "FR": ("FR", "FR"),
    "IT_NORD": ("IT_NORD", "IT_NORD"),
}
# probe months: one before and one after the split, one after the 15-min MTU switch
PROBE_MONTHS = ["2015-01", "2018-06", "2019-06", "2025-11"]


def year_windows(start: pd.Timestamp, end: pd.Timestamp):
    for year in range(start.year, end.year + 1):
        s = max(start, pd.Timestamp(f"{year}-01-01", tz=TZ))
        e = min(end, pd.Timestamp(f"{year + 1}-01-01", tz=TZ))
        if s < e:
            yield year, s, e


def fetch(client, code: str, s: pd.Timestamp, e: pd.Timestamp, throttle) -> pd.DataFrame | None:
    """One A44 request (entsoe-py splits by year), tidied, end-trimmed to [s, e)."""
    try:
        raw = probe.call_with_retry(client.query_day_ahead_prices, throttle,
                                    country_code=code, start=s, end=e)
    except probe.NO_DATA_ERRORS:
        return None
    if raw is None or len(raw) == 0:
        return None
    df = probe.tidy(raw, VALUE_COL)
    df.index = df.index.tz_convert("UTC")
    df = df[(df.index >= s.tz_convert("UTC")) & (df.index < e.tz_convert("UTC"))]
    df = df[~df.index.duplicated(keep="last")]
    df["area_code"] = code
    return df if not df.empty else None


def resolution(df: pd.DataFrame) -> str:
    """Modal step per calendar month, joined if it changes within the year (e.g. 'PT60M>PT15M')."""
    if df is None or len(df) < 3:
        return ""
    steps = (pd.Series(df.index).diff().dt.total_seconds() / 60).groupby(
        df.index.tz_convert(TZ).strftime("%Y-%m").to_numpy()).agg(lambda x: x.mode().iloc[0] if x.notna().any() else None)
    seq = [f"PT{int(v)}M" for v in steps.dropna()]
    out = [seq[0]] if seq else []
    for v in seq[1:]:
        if v != out[-1]:
            out.append(v)
    return ">".join(out)


def build_plan(start, end, areas):
    split = pre_split.split_ts(pre_split.DE_LU_SPLIT, TZ)
    plan = []
    for area in areas:
        pre, post = AREAS[area]
        for year, s, e in year_windows(start, end):
            segs = pre_split.segments(s, e, split, pre, post)
            if segs:
                plan.append({"area": area, "year": year, "segs": segs,
                             "area_code": pre_split.codes_label(segs)})
    return plan


def check(out_root: Path) -> int:
    """Coverage report from the files on disk (expected vs present hours per area-year)."""
    rows = []
    for f in sorted((out_root / DATASET).glob("*/*.parquet")):
        df = pd.read_parquet(f)
        loc = df.index.tz_convert(TZ)
        hours = loc.floor("h").unique()
        y = int(f.stem)
        exp = pd.date_range(f"{y}-01-01", f"{y + 1}-01-01", freq="h", tz=TZ, inclusive="left")
        exp = exp[exp < pd.Timestamp.now(tz=TZ).normalize().replace(day=1)]
        if f.parent.name == "AT" and y == 2018:
            exp = exp[exp >= pd.Timestamp(pre_split.DE_LU_SPLIT, tz=TZ)]
        missing = exp.difference(hours)
        rows.append({"area": f.parent.name, "year": y, "rows": len(df),
                     "hours_expected": len(exp), "hours_missing": len(missing),
                     "first_missing": str(missing[0]) if len(missing) else "",
                     "resolution": resolution(df), "min": round(df[VALUE_COL].min(), 1),
                     "max": round(df[VALUE_COL].max(), 1)})
    rep = pd.DataFrame(rows)
    if rep.empty:
        print(f"no files under {out_root / DATASET}")
        return 1
    print(rep.to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2015-01-01", help="inclusive, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="exclusive (default: first of this month)")
    ap.add_argument("--areas", default=",".join(AREAS), help="comma list, default all")
    ap.add_argument("--out", default=str(HERE / "Data" / "production"))
    ap.add_argument("--interval", type=float, default=0.3, help="min seconds between calls")
    ap.add_argument("--force", action="store_true", help="overwrite area-years on disk")
    ap.add_argument("--plan", action="store_true", help="print the plan, no API calls")
    ap.add_argument("--probe", action="store_true", help="one month per area/code, report only")
    ap.add_argument("--check", action="store_true", help="coverage of the files on disk, no API calls")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_root = Path(args.out).expanduser().resolve()
    if args.check:
        return check(out_root)
    areas = [a.strip() for a in args.areas.split(",") if a.strip()]
    bad = [a for a in areas if a not in AREAS]
    if bad:
        log.error("unknown area(s) %s; choose from %s", bad, list(AREAS))
        return 1
    start = pd.Timestamp(args.start, tz=TZ)
    end = (pd.Timestamp(args.end, tz=TZ) if args.end
           else pd.Timestamp.now(tz=TZ).normalize().replace(day=1))
    plan = build_plan(start, end, areas)

    if args.plan:
        for p in plan:
            segs = ", ".join(f"{c} {s.date()}->{e.date()}" for s, e, c in p["segs"])
            print(f"  {p['area']:8s} {p['year']}  {segs}")
        print(f"\n{len(plan)} area-years, {start.date()} -> {end.date()}. No API calls made (--plan).")
        return 0

    api_key = probe.load_api_key()
    if not api_key:
        log.error("ENTSOE_API_KEY not found - export it or put it in .env")
        return 1
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=api_key, timeout=probe.REQUEST_TIMEOUT_S)
    throttle = probe.Throttle(args.interval)

    if args.probe:
        rows = []
        for area in areas:
            for code in sorted({c for c in AREAS[area] if c}):
                for m in PROBE_MONTHS:
                    s = pd.Timestamp(f"{m}-01", tz=TZ)
                    e = s + pd.offsets.MonthBegin(1)
                    try:
                        df = fetch(client, code, s, e, throttle)
                        st = "ok" if df is not None else "no_data"
                    except Exception as exc:  # report, keep probing
                        df, st = None, probe.describe_error(exc)
                    prof = probe.profile(df) if df is not None else {}
                    rows.append({"area": area, "code": code, "month": m, "status": st,
                                 "rows": prof.get("rows", 0), "resolution": resolution(df),
                                 "mean": round(df[VALUE_COL].mean(), 2) if df is not None else None})
                    log.info("probe %-8s %-9s %s %s", area, code, m, st)
        rep = pd.DataFrame(rows)
        print(rep.to_string(index=False))
        dst = HERE / "Data" / f"_probe_{pd.Timestamp.now():%Y%m%d}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(dst, index=False)
        print(f"\nprobe report: {dst}")
        return 0

    manifest = []
    for i, p in enumerate(plan, 1):
        f = out_root / DATASET / p["area"] / f"{p['year']}.parquet"
        tag = f"{p['area']}__{p['year']}"
        if f.exists() and not args.force:
            log.info("[%d/%d] skip (on disk) %s", i, len(plan), tag)
            continue
        df = pre_split.join_parts([fetch(client, c, s, e, throttle) for s, e, c in p["segs"]])
        if df is None or df.empty:
            log.info("[%d/%d] empty %s", i, len(plan), tag)
            manifest.append({"dataset": DATASET, "area": p["area"], "area_code": p["area_code"],
                             "year": p["year"], "rows": 0, "file": ""})
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(f, engine="pyarrow", compression="snappy")
        prof = probe.profile(df)
        log.info("[%d/%d] OK %-14s %6d rows %s", i, len(plan), tag, len(df), resolution(df))
        manifest.append({"dataset": DATASET, "area": p["area"], "area_code": p["area_code"],
                         "year": p["year"], "rows": len(df), "first_ts": prof.get("first_ts", ""),
                         "last_ts": prof.get("last_ts", ""), "resolution": resolution(df),
                         "file": str(f.relative_to(out_root))})
    if manifest:
        mp = out_root / "_pull_manifest.csv"
        man = pd.DataFrame(manifest)
        if mp.exists():
            man = pd.concat([pd.read_csv(mp), man], ignore_index=True).drop_duplicates(
                ["dataset", "area", "year"], keep="last")
        man.to_csv(mp, index=False)
        log.info("manifest: %s", mp)
    print(f"\nparquet written under: {out_root / DATASET}\nNext: python Entsoe/Prices/entsoe_prices_pull.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
