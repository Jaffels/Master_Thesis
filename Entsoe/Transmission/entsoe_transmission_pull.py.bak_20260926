"""
ENTSO-E Transmission production pull — multi-year, empty-as-empty, error-safe.

Pulls the requested Transmission articles across the full thesis range for every
Swiss border, reusing the probe's request machinery unchanged:

  * the query_countertrading (13.1.B / A91) and query_congestion_costs
    (13.1.C / A92) wrappers — added on import of the probe module;
  * PINNED_BORDER_CODE / the coverage report's resolved from/to codes, so a
    bidding zone is never silently swapped for a control area;
  * tidy() so the on-disk schema matches the probe's parquet exactly.

Difference from the load pull, and the reason for it
----------------------------------------------------
The load pull only fetched the series the probe had already marked `ok`. Two of
the articles here — 13.1.B countertrading and 13.1.C congestion costs — are
SPARSE and event-driven: a border with no countertrade in the probe's sample
month may still have some in another year, so restricting to probe-`ok` would
silently drop real data. This pull therefore attempts the FULL grid
(every article x every Swiss border x every year) and classifies each series:

  * data   -> one parquet per (article, variant, border, year) with the values.
  * empty  -> the API definitively said "No matching data found"
              (NoMatchingDataError) for EVERY year, with no error in between.
              A single explicit marker file `_empty.parquet` (0 rows) is written
              for the series so an empty table is saved AS empty — and, because
              NoMatchingDataError is never produced by a transient glitch (those
              are retried; only a definitive answer counts), it is certain to be
              genuinely empty and not a swallowed error.
  * error  -> an HTTP 400/5xx or parse failure occurred for at least one year.
              NOTHING is written for that series (no fake-empty marker), the
              reason is recorded in the manifest, and it is logged. This is the
              guard the brief asks for: an error is never saved as if it were an
              empty table.

The probe's latest coverage CSV, if present, is used only to SKIP borders it
already flagged as hard `error` (so the pull doesn't re-hammer a malformed
request); pass --include-errors to attempt them anyway. Everything the probe
found `ok` or `empty` is attempted across the full range regardless.

Transmission endpoints are @month_limited / @year_limited in entsoe-py and
cap-free, so one call per calendar year is enough — the library splits it and
concatenates. No adaptive chunking / offset-pagination (unlike the balancing
pull); a year is fetched in a single call and trimmed.

Layout (per series):
    Data/production/<dataset>/<variant>/<border>/<year>.parquet   # data years
    Data/production/<dataset>/<variant>/<border>/_empty.parquet    # all-empty
Per-year files make the pull resumable: an already-written (series, year) is
skipped unless --force.

Usage (macOS), run from the thesis root:
    .venv/bin/python Entsoe/Transmission/entsoe_transmission_pull.py
    .venv/bin/python Entsoe/Transmission/entsoe_transmission_pull.py \
        --start 2021-01-01 --end 2026-09-01
    .venv/bin/python Entsoe/Transmission/entsoe_transmission_pull.py --plan
    .venv/bin/python Entsoe/Transmission/entsoe_transmission_pull.py --only ntc

A manifest CSV (_pull_manifest.csv) records every series with its final status
(ok / empty / error), row counts, timestamp span and files, so coverage across
the range — including which tables are genuinely empty — is auditable without
re-reading the parquet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# import the probe as a library: this also adds the A91/A92 wrappers and defines
# the request machinery, borders and tasks we reuse verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import entsoe_transmission_probe as probe  # noqa: E402
from entsoe import EntsoePandasClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transmission-pull")

TZ = probe.TZ


def latest_coverage(data_dir: Path) -> Path | None:
    """Most recently modified _coverage_*.csv the probe wrote, if any."""
    cands = sorted(data_dir.glob("_coverage_*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def coverage_lookup(coverage_csv: Path | None) -> dict[tuple[str, str, str], dict]:
    """Map (dataset, variant, border) -> its last probe row, for code hints and
    error-skipping. Empty dict when no coverage exists."""
    if not coverage_csv or not coverage_csv.exists():
        return {}
    rep = pd.read_csv(coverage_csv)
    rep = rep.drop_duplicates(["dataset", "variant", "border"], keep="last")
    out: dict[tuple[str, str, str], dict] = {}
    for _, r in rep.iterrows():
        out[(r["dataset"], r["variant"], r["border"])] = r.to_dict()
    return out


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


def resolve_codes(task: probe.Task, border: probe.Border,
                  cov: dict) -> tuple[str, str]:
    """Which (from_code, to_code) to use for a border. Priority:
    PINNED_BORDER_CODE > the probe's resolved ok codes > the border's primary
    candidates. A single pair — no per-year fallback scanning in the pull."""
    pinned = probe.PINNED_BORDER_CODE.get((task.dataset, border.label))
    if pinned:
        return pinned
    row = cov.get((task.dataset, task.variant, border.label))
    if row and row.get("status") == "ok":
        fc, tc = str(row.get("from_code_used", "")), str(row.get("to_code_used", ""))
        if fc and tc and fc != "nan" and tc != "nan":
            return fc, tc
    return border.from_codes[0], border.to_codes[0]


def pull_series_year(client: EntsoePandasClient, task: probe.Task,
                     from_code: str, to_code: str,
                     y_start: pd.Timestamp, y_end: pd.Timestamp,
                     throttle: probe.Throttle) -> tuple[str, pd.DataFrame | None, str]:
    """Fetch one (series, calendar-year) slice. Returns (status, df, err) where
    status is 'ok' | 'empty' | 'error'. 'empty' only ever comes from a definitive
    NoMatchingDataError (transient failures are retried inside call_with_retry),
    so it is a real no-data answer, never a swallowed glitch."""
    method = getattr(client, task.method)
    try:
        raw = probe.call_with_retry(
            method, throttle, country_code_from=from_code,
            country_code_to=to_code, start=y_start, end=y_end, **task.kwargs)
    except probe.NO_DATA_ERRORS:
        return "empty", None, ""
    except Exception as exc:  # HTTP 400/5xx / parse -> a real error, not empty
        return "error", None, probe.describe_error(exc)

    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return "empty", None, ""
    df = probe.tidy(raw, task.dataset)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index < y_end]
    return ("ok", df, "") if not df.empty else ("empty", None, "")


def write_empty_marker(path: Path, dataset: str) -> None:
    """Write an explicit 0-row parquet so a genuinely empty table is saved AS an
    empty table: an empty DatetimeIndex named 'timestamp' and one float column
    named after the dataset. Readable and unambiguous — not a missing file."""
    empty = pd.DataFrame(
        {dataset: pd.Series(dtype="float64")},
        index=pd.DatetimeIndex([], name="timestamp", tz=TZ))
    path.parent.mkdir(parents=True, exist_ok=True)
    empty.to_parquet(path, engine="pyarrow", compression="snappy")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--start", default="2021-01-01", help="inclusive, YYYY-MM-DD")
    ap.add_argument("--end", default=None,
                    help="exclusive, YYYY-MM-DD (default: first of this month)")
    ap.add_argument("--coverage", default=None,
                    help="probe coverage CSV for code hints / error-skipping "
                         "(default: newest _coverage_*.csv under --data)")
    ap.add_argument("--data", default=str(here / "Data"),
                    help="folder holding the probe's coverage CSV")
    ap.add_argument("--out", default=str(here / "Data" / "production"),
                    help="output root for the per-year parquet files")
    ap.add_argument("--only", default="",
                    help="substring filter on dataset name")
    ap.add_argument("--include-errors", action="store_true",
                    help="also attempt borders the probe flagged as hard error")
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
    cov = coverage_lookup(coverage)

    start = pd.Timestamp(args.start, tz=TZ)
    end = (pd.Timestamp(args.end, tz=TZ) if args.end
           else pd.Timestamp.now(tz=TZ).normalize().replace(day=1))
    if end <= start:
        log.error("--end must be after --start")
        return 1

    tasks = [t for t in probe.build_tasks()
             if t.enabled and args.only.lower() in t.dataset.lower()]
    years = year_windows(start, end)
    out_root = Path(args.out).expanduser().resolve()

    # full grid (per-task targets: directed borders, or single areas for
    # scope='area' products), minus any target the probe flagged as hard error
    grid: list[tuple[probe.Task, probe.Border]] = []
    skipped_error: list[tuple[str, str, str]] = []
    for task in tasks:
        for border in probe.targets_for(task):
            row = cov.get((task.dataset, task.variant, border.label))
            if (row and row.get("status") == "error" and not args.include_errors):
                skipped_error.append((task.dataset, task.variant, border.label))
                continue
            grid.append((task, border))

    log.info("%s | %d articles = %d series x %d years | %s -> %s",
             ("coverage: " + coverage.name) if coverage else "no coverage CSV",
             len(tasks), len(grid), len(years),
             start.date(), end.date())
    if skipped_error:
        log.info("skipping %d series the probe flagged as error "
                 "(use --include-errors to attempt): %s",
                 len(skipped_error),
                 ", ".join(f"{d}/{v}/{b}" for d, v, b in skipped_error))

    if args.plan:
        print("\nPULL PLAN (series):")
        for task, border in grid:
            fc, tc = resolve_codes(task, border, cov)
            print(f"  {task.dataset:20s} {task.variant:14s} "
                  f"{border.label:9s} <- {fc}>{tc}")
        print(f"\n{len(grid)} series x {len(years)} years "
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

    for i, (task, border) in enumerate(grid, 1):
        fc, tc = resolve_codes(task, border, cov)
        series_dir = out_root / task.dataset / task.variant / border.label
        tag = f"{task.dataset}/{task.variant}/{border.label}"

        data_years = 0
        total_rows = 0
        first_ts = last_ts = ""
        errored = False
        err_msg = ""

        for year, y_start, y_end in years:
            fname = series_dir / f"{year}.parquet"
            if fname.exists() and not args.force:
                log.info("[%d/%d] skip (on disk) %s %d", i, len(grid), tag, year)
                data_years += 1  # treat an existing file as a data year
                continue

            status, df, err = pull_series_year(
                client, task, fc, tc, y_start, y_end, throttle)

            if status == "ok" and df is not None:
                fname.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(fname, engine="pyarrow", compression="snappy")
                prof = probe.profile(df)
                data_years += 1
                total_rows += len(df)
                first_ts = first_ts or prof.get("first_ts", "")
                last_ts = prof.get("last_ts", last_ts)
                log.info("[%d/%d] OK    %-42s %d -> %6d rows",
                         i, len(grid), tag, year, len(df))
            elif status == "error":
                errored = True
                err_msg = err
                log.warning("[%d/%d] ERROR %-42s %d %s",
                            i, len(grid), tag, year, err)
            # status == "empty": nothing for this year

        # decide the series-level outcome
        empty_marker = series_dir / "_empty.parquet"
        if data_years > 0:
            # some data present; clear any stale empty marker
            if empty_marker.exists():
                empty_marker.unlink()
            final = "ok"
        elif errored:
            # cannot certify empty — do NOT write an empty marker
            final = "error"
        else:
            # every year definitively empty -> save the empty table as empty
            write_empty_marker(empty_marker, task.dataset)
            final = "empty"
            log.info("[%d/%d] EMPTY %-42s all years no-data -> _empty.parquet",
                     i, len(grid), tag)

        manifest.append({
            "dataset": task.dataset, "variant": task.variant,
            "border": border.label, "from_code": fc, "to_code": tc,
            "status": final, "data_years": data_years, "rows": total_rows,
            "first_ts": first_ts, "last_ts": last_ts,
            "error": err_msg,
            "path": str(series_dir.relative_to(out_root)),
        })

    if manifest:
        man = pd.DataFrame(manifest)
        if manifest_path.exists():
            man = pd.concat([pd.read_csv(manifest_path), man], ignore_index=True)
            man = man.drop_duplicates(
                ["dataset", "variant", "border"], keep="last")
        man.to_csv(manifest_path, index=False)
        n_ok = (man["status"] == "ok").sum()
        n_empty = (man["status"] == "empty").sum()
        n_err = (man["status"] == "error").sum()
        log.info("done: %d ok, %d empty (saved as _empty.parquet), %d error "
                 "| manifest: %s", n_ok, n_empty, n_err, manifest_path)
        if n_err:
            log.warning("%d series errored and were NOT saved as empty — "
                        "inspect the manifest 'error' column", n_err)

    print(f"\nparquet written under: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
