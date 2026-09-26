"""
ENTSO-E Load production pull — multi-year, driven by the probe's coverage.

The probe (entsoe_load_probe.py) established, for one month, which
(dataset, variant, area) Load series return data and on which area code. This
script takes that verdict and pulls the SAME series across the full thesis
range, reusing the probe's request machinery unchanged:

  * the query_load_forecast_margin wrapper (article 8.1) — added on import of
    the probe module;
  * PINNED_AREA_CODE / the coverage report's resolved area_code_used, so a
    bidding zone is never silently swapped for a control area;
  * tidy() so the on-disk schema matches the probe's parquet exactly.

Load endpoints are @month_limited in entsoe-py and cap-free, so one call per
calendar year is enough — the library splits it into monthly sub-requests and
concatenates. There is none of the adaptive-chunking / offset-pagination the
balancing pull needs; a year is fetched in a single call and trimmed.

Design choices (mirroring the balancing pull):
  * RAW pull, reconcile later. One parquet per (dataset, variant, area, year).
    Actual load (6.1.A), each forecast horizon (6.1.B/C/D/E) and the year-ahead
    margin (8.1) are kept as separate series; the "A&B" / "A&C&D&E" overlays are
    downstream joins on the timestamp index, so nothing is double-counted here.
  * Per-year files make the pull resumable: an already-written (series, year) is
    skipped unless --force, so a long pull can stop and restart freely.
  * Only status=ok series from the coverage CSV are pulled.

Usage (macOS), run from the thesis root:
    .venv/bin/python Entsoe/Load/entsoe_load_pull.py
    .venv/bin/python Entsoe/Load/entsoe_load_pull.py \
        --start 2021-01-01 --end 2026-09-01
    .venv/bin/python Entsoe/Load/entsoe_load_pull.py --plan          # dry run
    .venv/bin/python Entsoe/Load/entsoe_load_pull.py --only forecast

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

# import the probe as a library: this also adds the 8.1 margin method and
# defines the request machinery we reuse verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import entsoe_load_probe as probe  # noqa: E402
from entsoe import EntsoePandasClient  # noqa: E402

log = logging.getLogger("load-pull")

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
                     throttle: probe.Throttle) -> pd.DataFrame | None:
    """Fetch one (series, calendar-year) slice, tidied and end-trimmed.

    A single call per year: the Load methods are @month_limited, so entsoe-py
    splits the year into months internally and concatenates. The inclusive API
    end boundary is trimmed so consecutive years never double-count a timestamp.
    """
    method = getattr(client, task.method)
    try:
        raw = probe.call_with_retry(
            method, throttle, country_code=area_code,
            start=y_start, end=y_end, **task.kwargs)
    except probe.NO_DATA_ERRORS:
        return None
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return None
    df = probe.tidy(raw, task.dataset)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index < y_end]
    return df if not df.empty else None


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
    ap.add_argument("--interval", type=float, default=2.0,
                    help="minimum seconds between API calls (default 2.0)")
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
    if args.only:
        ok = ok[ok["dataset"].str.contains(args.only, case=False)]
    if ok.empty:
        log.error("no status=ok series to pull (after --only filter)")
        return 1

    tasks = {(t.dataset, t.variant): t for t in probe.build_tasks()}
    years = year_windows(start, end)
    out_root = Path(args.out).expanduser().resolve()

    plan: list[dict[str, Any]] = []
    for _, row in ok.iterrows():
        key = (row["dataset"], row["variant"])
        task = tasks.get(key)
        if task is None:
            log.warning("coverage names %s but build_tasks has no such task — "
                        "skipping (probe/pull version mismatch?)", key)
            continue
        code = row["area_code_used"]
        for year, y_start, y_end in years:
            plan.append({
                "dataset": row["dataset"], "variant": row["variant"],
                "area": row["area"], "area_code": code, "task": task,
                "year": year, "y_start": y_start, "y_end": y_end,
            })

    log.info("coverage: %s | %d ok series x %d years = %d series-years | %s -> %s",
             coverage.name, len(plan) // max(len(years), 1), len(years),
             len(plan), start.date(), end.date())

    if args.plan:
        print("\nPULL PLAN (series-years):")
        seen = set()
        for p in plan:
            k = (p["dataset"], p["variant"], p["area"], p["area_code"])
            if k not in seen:
                seen.add(k)
                print(f"  {p['dataset']:16s} {p['variant']:16s} "
                      f"{p['area']:7s} <- {p['area_code']:10s}")
        print(f"\n{len(seen)} distinct series x {len(years)} years "
              f"({years[0][0]}..{years[-1][0]}). No API calls made (--plan).")
        return 0

    api_key = probe.load_api_key()
    if not api_key:
        log.error("ENTSOE_API_KEY not found — export it or put it in .env")
        return 1
    client = EntsoePandasClient(api_key=api_key)
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

        df = pull_series_year(client, p["task"], p["area_code"],
                              p["y_start"], p["y_end"], throttle)

        if df is None or df.empty:
            log.info("[%d/%d] empty     %s", i, len(plan), tag)
            manifest.append({**{k: p[k] for k in
                                ("dataset", "variant", "area", "area_code",
                                 "year")}, "rows": 0, "file": ""})
            continue

        fname.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(fname, engine="pyarrow", compression="snappy")
        prof = probe.profile(df)
        log.info("[%d/%d] OK  %-48s %6d rows -> %s",
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
