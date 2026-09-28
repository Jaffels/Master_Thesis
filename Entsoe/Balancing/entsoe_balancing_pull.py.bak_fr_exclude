"""
ENTSO-E Balancing production pull — multi-year, driven by the probe's coverage.

The probe (entsoe_balancing_probe.py) established, for one month, which
(dataset, variant, area) series actually return data and on which area code.
This script takes that verdict and pulls the SAME series across the full thesis
range, reusing the probe's request machinery unchanged:

  * the entsoe-py parser patches (symmetric FCR direction; contracted-reserve
    volume/price TimeSeries split) — applied on import of the probe module;
  * chunked_call() for the endpoints entsoe-py does not internally paginate,
    so no response is truncated at the 100-TimeSeries cap;
  * PINNED_AREA_CODE / the coverage report's resolved area_code_used, so a
    control area is never silently swapped for a bidding zone;
  * tidy() so the on-disk schema matches the probe's parquet exactly.

Design choices:
  * RAW pull, reconcile later. One parquet per (dataset, variant, area, year).
    Volume and price live in separate agreement-type variants for some Swiss
    products (e.g. mFRR: price is daily, volume is weekly) and FCR appears
    under both daily and weekly during the 2019-2020 reform overlap — none of
    that is merged here; every series is kept whole so the cleaning stage has
    all it needs and nothing is double-counted at pull time.
  * Per-year files make the pull resumable: an already-written (series, year)
    is skipped unless --force, so a long pull can stop and restart freely.
  * Only status=ok series from the coverage CSV are pulled — no time wasted on
    combinations the probe proved empty.

Usage (macOS), run from the thesis root:
    .venv/bin/python Entsoe/Balancing/entsoe_balancing_pull.py
    .venv/bin/python Entsoe/Balancing/entsoe_balancing_pull.py \
        --start 2021-01-01 --end 2026-09-01
    .venv/bin/python Entsoe/Balancing/entsoe_balancing_pull.py --plan   # dry run
    .venv/bin/python Entsoe/Balancing/entsoe_balancing_pull.py --only aggregated_bids

A manifest CSV (_pull_manifest.csv) records every written series-year with its
row count, timestamp span and file, so coverage across the range is auditable
without re-reading the parquet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from requests.exceptions import HTTPError

# import the probe as a library: this also applies its parser patches and
# defines the request machinery we reuse verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import entsoe_balancing_probe as probe  # noqa: E402
from entsoe import EntsoePandasClient  # noqa: E402

# Library-paginated endpoints (contracted reserve, procured capacity) page past
# the 100-TimeSeries cap with an offset parameter, but ENTSO-E rejects offset
# beyond ~100 (offset=200 -> HTTP 400) for these document types. So a whole-year
# request with >200 TimeSeries fails. Pull them in calendar-month windows
# (proven safe by the probe) and, defensively, halve any window that still 400s
# down to a floor. This does NOT apply to the chunked endpoints, which are
# un-paginated and already time-split by chunked_call.
PAGINATED_FLOOR_DAYS = 1

# (dataset, area) combinations never pulled (D6, decided 2026-09-26): DE and IT
# aggregated bids are too dense - a single day exceeds the API's 100-TimeSeries
# cap, so they come back irreducibly incomplete. Only CH (and FR) bids are kept.
# Remove an entry here to pull it again.
EXCLUDE_SERIES: set[tuple[str, str]] = {
    ("aggregated_bids", "DE"),
    ("aggregated_bids", "IT"),
}

log = logging.getLogger("pull")

TZ = probe.TZ


def latest_coverage(data_dir: Path) -> Path | None:
    """Most recently modified _coverage_*.csv the probe wrote, if any."""
    cands = sorted(data_dir.glob("_coverage_*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_ok_series(coverage_csv: Path) -> pd.DataFrame:
    """The status=ok (dataset, variant, area, area_code_used) rows to pull.

    A coverage CSV may accumulate several probe runs; keep the last verdict per
    (dataset, variant, area). The area_code_used column is the code the probe
    actually got data on — the production pull uses exactly that, no re-probing.
    """
    rep = pd.read_csv(coverage_csv)
    ok = rep[rep["status"] == "ok"].copy()
    ok = ok.drop_duplicates(["dataset", "variant", "area"], keep="last")
    ok["area_code_used"] = ok["area_code_used"].fillna("").astype(str)
    return ok.reset_index(drop=True)


def year_windows(start: pd.Timestamp, end: pd.Timestamp
                 ) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    """(year, y_start, y_end) calendar-year slices clipped to [start, end)."""
    out = []
    for year in range(start.year, end.year + 1):
        y_start = max(start, pd.Timestamp(f"{year}-01-01", tz=start.tz))
        y_end = min(end, pd.Timestamp(f"{year + 1}-01-01", tz=start.tz))
        if y_start < y_end:
            out.append((year, y_start, y_end))
    return out


def pull_series_year(client: EntsoePandasClient, task: probe.Task,
                     area_code: str, y_start: pd.Timestamp, y_end: pd.Timestamp,
                     throttle: probe.Throttle, chunk: str, chunk_floor: str
                     ) -> tuple[pd.DataFrame | None, str]:
    """Fetch one (series, calendar-year) slice, tidied and end-trimmed.

    Chunked endpoints go through the probe's adaptive chunked_call. The
    library-paginated endpoints are pulled in month-sized windows, because
    ENTSO-E rejects the offset paging (offset>~100 -> HTTP 400) that the library
    would use on a whole-year request. Returns (df or None, hint).
    """
    method = getattr(client, task.method)
    if task.chunk:
        df, hint, _ = probe.chunked_call(
            method, throttle, y_start, y_end, chunk, task.dataset,
            floor=chunk_floor, country_code=area_code, **task.kwargs)
        return df, hint
    df = _pull_paginated(method, throttle, y_start, y_end,
                         task.dataset, area_code, task.kwargs)
    if df is not None and isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index < y_end]
    return (df if (df is not None and not df.empty) else None), ""


def _pull_paginated(method: Any, throttle: probe.Throttle,
                    start: pd.Timestamp, end: pd.Timestamp, dataset: str,
                    area_code: str, kwargs: dict) -> pd.DataFrame | None:
    """Pull a library-paginated endpoint in month-sized windows and concatenate.

    Keeps each request's TimeSeries count low enough that entsoe-py's internal
    offset paging never exceeds what ENTSO-E accepts. Any window that still 400s
    (an unusually dense month) is halved and retried down to PAGINATED_FLOOR_DAYS
    before giving up. Empty windows (before a series starts, gaps) are skipped.
    """
    parts: list[pd.DataFrame] = []
    for w_start, w_end in _month_windows(start, end):
        parts.extend(_fetch_window(method, throttle, w_start, w_end,
                                   dataset, area_code, kwargs))
    if not parts:
        return None
    return pd.concat(parts).sort_index()


def _month_windows(start: pd.Timestamp, end: pd.Timestamp
                   ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Calendar-month [w_start, w_end) windows over [start, end).

    Calendar months (not fixed 30-day steps) keep every boundary on a local
    midnight, so DST transitions never shift a boundary off-day. Half-open, so
    concatenating never double-counts a boundary timestamp."""
    out = []
    cur = start
    nxt = start.normalize().replace(day=1) + pd.DateOffset(months=1)
    while cur < end:
        w_end = min(nxt, end)
        out.append((cur, w_end))
        cur = w_end
        nxt = nxt + pd.DateOffset(months=1)
    return out


def _fetch_window(method: Any, throttle: probe.Throttle,
                  a: pd.Timestamp, b: pd.Timestamp, dataset: str,
                  area_code: str, kwargs: dict) -> list[pd.DataFrame]:
    """One paginated window, tidied and end-trimmed; halve on a 400 offset error."""
    try:
        raw = probe.call_with_retry(
            method, throttle, country_code=area_code, start=a, end=b, **kwargs)
    except probe.NO_DATA_ERRORS:
        return []
    except HTTPError:
        # too many TimeSeries in this window for the API's offset limit — split
        if (b - a) <= pd.Timedelta(days=PAGINATED_FLOOR_DAYS):
            raise
        mid = a + (b - a) / 2
        return (_fetch_window(method, throttle, a, mid, dataset, area_code, kwargs)
                + _fetch_window(method, throttle, mid, b, dataset, area_code, kwargs))
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return []
    part = probe.tidy(raw, dataset)
    if isinstance(part.index, pd.DatetimeIndex):
        part = part[part.index < b]
    return [part] if not part.empty else []


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--start", default="2021-01-01", help="inclusive, YYYY-MM-DD")
    ap.add_argument("--end", default=None,
                    help="exclusive, YYYY-MM-DD (default: first of this month)")
    ap.add_argument("--coverage", default=None,
                    help="coverage CSV to drive the pull "
                         "(default: newest _coverage_*.csv under --data)")
    ap.add_argument("--data", default=str(here / "Data"),
                    help="folder holding the probe's coverage CSV")
    ap.add_argument("--out", default=str(here / "Data" / "production"),
                    help="output root for the per-year parquet files")
    ap.add_argument("--only", default="",
                    help="substring filter on dataset name")
    ap.add_argument("--chunk", default=probe.UNPAGINATED_CHUNK,
                    help="ceiling window for un-paginated endpoints "
                         f"(default {probe.UNPAGINATED_CHUNK}); adaptive")
    ap.add_argument("--chunk-floor", default=probe.CHUNK_FLOOR,
                    help=f"smallest adaptive window (default {probe.CHUNK_FLOOR})")
    ap.add_argument("--interval", type=float, default=0.3,
                    help="minimum seconds between API calls (default 0.3 = ~200 req/min, half the 400/min limit)")
    ap.add_argument("--force", action="store_true",
                    help="re-pull and overwrite series-years already on disk")
    ap.add_argument("--plan", action="store_true",
                    help="print the pull plan and exit without calling the API")
    args = ap.parse_args()

    data_dir = Path(args.data).expanduser().resolve()
    coverage = (Path(args.coverage).expanduser().resolve()
                if args.coverage else latest_coverage(data_dir))
    if not coverage or not coverage.exists():
        log.error("no coverage CSV found — run the probe first, or pass "
                  "--coverage. Looked in %s", data_dir)
        return 1

    start = pd.Timestamp(args.start, tz=TZ)
    end = (pd.Timestamp(args.end, tz=TZ) if args.end
           else pd.Timestamp.now(tz=TZ).normalize().replace(day=1))
    if end <= start:
        log.error("--end must be after --start")
        return 1

    ok = load_ok_series(coverage)
    excluded = [(d, a) for d, a in zip(ok["dataset"], ok["area"]) if (d, a) in EXCLUDE_SERIES]
    if excluded:
        log.info("excluded by EXCLUDE_SERIES (D6): %s", sorted(set(excluded)))
    ok = ok[[(d, a) not in EXCLUDE_SERIES for d, a in zip(ok["dataset"], ok["area"])]]
    if args.only:
        ok = ok[ok["dataset"].str.contains(args.only, case=False)]
    if ok.empty:
        log.error("no status=ok series to pull (after --only filter)")
        return 1

    tasks = {(t.dataset, t.variant): t for t in probe.build_tasks()}
    years = year_windows(start, end)
    out_root = Path(args.out).expanduser().resolve()

    # build the work list, skipping series the probe knows but that summed
    # across control areas (not handled by the single-code pull path)
    plan: list[dict[str, Any]] = []
    for _, row in ok.iterrows():
        key = (row["dataset"], row["variant"])
        task = tasks.get(key)
        if task is None:
            log.warning("coverage names %s but build_tasks has no such task — "
                        "skipping (probe/pull version mismatch?)", key)
            continue
        code = row["area_code_used"]
        if "+" in code:
            log.warning("skip summed series %s %s (area_code_used=%s) — the "
                        "pull does not yet reassemble multi-TSO sums",
                        key, row["area"], code)
            continue
        for year, y_start, y_end in years:
            plan.append({
                "dataset": row["dataset"], "variant": row["variant"],
                "area": row["area"], "area_code": code, "task": task,
                "year": year, "y_start": y_start, "y_end": y_end,
            })

    log.info("coverage: %s | %d ok series x %d years = %d series-years | "
             "%s -> %s", coverage.name, len(plan) // max(len(years), 1),
             len(years), len(plan), start.date(), end.date())

    if args.plan:
        print("\nPULL PLAN (series-years):")
        seen = set()
        for p in plan:
            k = (p["dataset"], p["variant"], p["area"], p["area_code"])
            if k not in seen:
                seen.add(k)
                print(f"  {p['dataset']:34s} {p['variant']:16s} "
                      f"{p['area']:3s} <- {p['area_code']:10s} "
                      f"chunk={p['task'].chunk or 'lib-paginated'}")
        print(f"\n{len(seen)} distinct series x {len(years)} years "
              f"({years[0][0]}..{years[-1][0]}). No API calls made (--plan).")
        return 0

    api_key = probe.load_api_key()
    if not api_key:
        log.error("ENTSOE_API_KEY not found — export it or put it in .env")
        return 1
    client = EntsoePandasClient(api_key=api_key,
                                timeout=probe.REQUEST_TIMEOUT_S)
    throttle = probe.Throttle(args.interval)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "_pull_manifest.csv"
    manifest: list[dict[str, Any]] = []

    for i, p in enumerate(plan, 1):
        rel = Path(p["dataset"]) / p["variant"] / p["area"]
        fname = out_root / rel / f"{p['year']}.parquet"
        tag = f"{p['dataset']}__{p['variant']}__{p['area']}__{p['year']}"

        if fname.exists() and not args.force:
            log.info("[%d/%d] skip (on disk) %s", i, len(plan), tag)
            continue

        df, hint = pull_series_year(
            client, p["task"], p["area_code"], p["y_start"], p["y_end"],
            throttle, args.chunk, args.chunk_floor)
        if hint:
            log.warning("STILL CAPPED %s: %s", tag, hint)

        if df is None or df.empty:
            log.info("[%d/%d] empty     %s", i, len(plan), tag)
            manifest.append({**{k: p[k] for k in
                                ("dataset", "variant", "area", "area_code",
                                 "year")}, "rows": 0, "file": ""})
            continue

        fname.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(fname, engine="pyarrow", compression="snappy")
        prof = probe.profile(df)
        log.info("[%d/%d] OK  %-52s %6d rows -> %s",
                 i, len(plan), tag, len(df), rel / f"{p['year']}.parquet")
        manifest.append({
            **{k: p[k] for k in
               ("dataset", "variant", "area", "area_code", "year")},
            "rows": len(df),
            "first_ts": prof.get("first_ts", ""),
            "last_ts": prof.get("last_ts", ""),
            "cols": df.shape[1],
            "file": str(rel / f"{p['year']}.parquet"),
        })

    if manifest:
        man = pd.DataFrame(manifest)
        if manifest_path.exists():
            man = pd.concat([pd.read_csv(manifest_path), man], ignore_index=True)
            man = man.drop_duplicates(
                ["dataset", "variant", "area", "year"], keep="last")
        man.to_csv(manifest_path, index=False)
        got = man[man["rows"] > 0]
        log.info("wrote %d series-years (%d non-empty) | manifest: %s",
                 len(manifest), len(got), manifest_path)

    print(f"\nparquet written under: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
