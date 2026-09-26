"""
ENTSO-E Outages production pull — multi-year, driven by the probe's coverage.

The probe (entsoe_outages_probe.py) established, for one month, which
(dataset, variant, area) Outages series return data and on which area code.
This script pulls the same series across the full thesis range, reusing the
probe's machinery unchanged:

  * the query_outages_* client methods (15.1.A-D, 10.1.A&B, 10.1.C and the
    IF aFRR 3.10 / IF mFRR 3.11 / IFs IN 7.2 fall-backs) — added on import of
    the probe module; they paginate, halve dense windows and parse internally;
  * PINNED_AREA_CODE / the coverage report's area_code_used, so a bidding zone
    is never silently swapped for a control area;
  * tidy() / profile() so the on-disk schema matches the probe's parquet.

One method call per (series, calendar year); the method splits the year into
monthly windows internally. Every HTTP request is throttled (--interval).

Which series are pulled:
  * dense series (15.1.x generation / production units): status=ok rows of the
    coverage CSV only — same rule as the Load pull;
  * sparse, event-driven series (10.1.A&B transmission, 10.1.C offshore,
    fall-backs): the FULL grid of the task, because one quiet probe month is
    not a verdict. The code used is the probe's area_code_used if it answered,
    else the pinned code, else the grid's first candidate.

Resumability and empties (same conventions as the Transmission pull):
  * <year>.parquet is skipped if on disk (unless --force);
  * a definitive "no matching data" year writes a <year>.empty marker so reruns
    skip it too; an HTTP / parse error writes NOTHING and is recorded as
    status=error in the manifest, so it is retried on the next run;
  * a series whose every year is empty gets an explicit _empty.parquet.

Outage documents are selected by overlap with the query window, so an outage
spanning New Year appears in both year files — dedupe on (doc_mrid, revision)
downstream. Nothing is trimmed: the index is a RangeIndex, not a time grid.

Output layout (mirrors Entsoe/Load):
    Entsoe/Outages/Data/production/<dataset>/<variant>/<area>/<year>.parquet
    Entsoe/Outages/Data/production/_pull_manifest.csv

Usage (macOS), run from the thesis root:
    .venv/bin/python Entsoe/Outages/entsoe_outages_pull.py
    .venv/bin/python Entsoe/Outages/entsoe_outages_pull.py \\
        --start 2021-01-01 --end 2026-09-01
    .venv/bin/python Entsoe/Outages/entsoe_outages_pull.py --plan      # dry run
    .venv/bin/python Entsoe/Outages/entsoe_outages_pull.py --only fallback
    .venv/bin/python Entsoe/Outages/entsoe_outages_pull.py --ok-only   # skip
                                                   # unprobed sparse combos
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entsoe_outages_probe as probe  # noqa: E402
from entsoe import EntsoePandasClient  # noqa: E402

log = logging.getLogger("outages-pull")

TZ = probe.TZ


def latest_coverage(data_dir: Path) -> Path | None:
    cands = sorted(data_dir.glob("_coverage_*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_coverage(coverage_csv: Path) -> pd.DataFrame:
    """Last verdict per (dataset, variant, area)."""
    rep = pd.read_csv(coverage_csv)
    rep = rep.drop_duplicates(["dataset", "variant", "area"], keep="last")
    rep["area_code_used"] = rep["area_code_used"].fillna("").astype(str)
    return rep.reset_index(drop=True)


def year_windows(start: pd.Timestamp, end: pd.Timestamp
                 ) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    out = []
    for year in range(start.year, end.year + 1):
        y_start = max(start, pd.Timestamp(f"{year}-01-01", tz=start.tz))
        y_end = min(end, pd.Timestamp(f"{year + 1}-01-01", tz=start.tz))
        if y_start < y_end:
            out.append((year, y_start, y_end))
    return out


def build_series(cov: pd.DataFrame, only: str, ok_only: bool
                 ) -> list[dict[str, Any]]:
    """(task, area, code) triples to pull, per the rules in the docstring."""
    verdict = {(r.dataset, r.variant, r.area): (r.status, r.area_code_used)
               for r in cov.itertuples()}
    series = []
    for task in probe.build_tasks():
        if not task.enabled or only.lower() not in task.dataset.lower():
            continue
        for area, codes in task.grid.items():
            status, used = verdict.get((task.dataset, task.variant, area),
                                       ("unprobed", ""))
            if status == "ok":
                code = used
            elif task.sparse and not ok_only:
                code = (probe.PINNED_AREA_CODE.get((task.dataset, area))
                        or codes[0])
            else:
                continue
            series.append({"task": task, "dataset": task.dataset,
                           "variant": task.variant, "area": area,
                           "area_code": code, "probe_status": status})
    return series


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--start", default="2021-01-01", help="inclusive, YYYY-MM-DD")
    ap.add_argument("--end", default=None,
                    help="exclusive, YYYY-MM-DD (default: first of this month)")
    ap.add_argument("--coverage", default=None,
                    help="coverage CSV (default: newest _coverage_*.csv under --data)")
    ap.add_argument("--data", default=str(here / "Data"),
                    help="folder holding the probe's coverage CSV")
    ap.add_argument("--out", default=str(here / "Data" / "production"),
                    help="output root for the per-year parquet files")
    ap.add_argument("--only", default="", help="substring filter on dataset name")
    ap.add_argument("--ok-only", action="store_true",
                    help="pull only probe status=ok series, also for sparse tasks")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="minimum seconds between API requests (default 2.0)")
    ap.add_argument("--force", action="store_true",
                    help="re-pull series-years already on disk (incl. .empty)")
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

    series = build_series(load_coverage(coverage), args.only, args.ok_only)
    if not series:
        log.error("nothing to pull (after --only / --ok-only)")
        return 1
    years = year_windows(start, end)
    out_root = Path(args.out).expanduser().resolve()

    log.info("coverage: %s | %d series x %d years = %d series-years | %s -> %s",
             coverage.name, len(series), len(years), len(series) * len(years),
             start.date(), end.date())

    if args.plan:
        print("\nPULL PLAN (series):")
        for s in series:
            print(f"  {s['dataset']:22s} {s['variant']:20s} {s['area']:14s} "
                  f"<- {s['area_code']:16s} [probe: {s['probe_status']}]")
        print(f"\n{len(series)} series x {len(years)} years "
              f"({years[0][0]}..{years[-1][0]}). No API calls made (--plan).")
        return 0

    api_key = probe.load_api_key()
    if not api_key:
        log.error("ENTSOE_API_KEY not found — export it or put it in .env")
        return 1
    client = EntsoePandasClient(api_key=api_key)
    probe.set_request_throttle(probe.Throttle(args.interval))
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "_pull_manifest.csv"
    manifest: list[dict[str, Any]] = []

    total = len(series) * len(years)
    i = 0
    for s in series:
        rel = Path(s["dataset"]) / s["variant"] / s["area"]
        sdir = out_root / rel
        statuses: list[str] = []
        for year, y_start, y_end in years:
            i += 1
            tag = f"{s['dataset']}__{s['variant']}__{s['area']}__{year}"
            fparq = sdir / f"{year}.parquet"
            fempty = sdir / f"{year}.empty"
            ident = {k: s[k] for k in ("dataset", "variant", "area", "area_code")}

            if not args.force and fparq.exists():
                log.info("[%d/%d] skip (on disk) %s", i, total, tag)
                statuses.append("ok")
                continue
            if not args.force and fempty.exists():
                log.info("[%d/%d] skip (empty)   %s", i, total, tag)
                statuses.append("empty")
                continue

            try:
                df = probe.invoke(client, s["task"], s["area_code"], y_start, y_end)
            except probe.NO_DATA_ERRORS:
                sdir.mkdir(parents=True, exist_ok=True)
                fempty.touch()
                fparq.unlink(missing_ok=True)
                log.info("[%d/%d] empty     %s", i, total, tag)
                manifest.append({**ident, "year": year, "status": "empty",
                                 "rows": 0, "file": ""})
                statuses.append("empty")
                continue
            except Exception as exc:  # HTTP / parse: never recorded as empty
                err = probe.describe_error(exc)
                log.warning("[%d/%d] ERROR     %s %s", i, total, tag, err)
                manifest.append({**ident, "year": year, "status": "error",
                                 "rows": 0, "file": "", "error": err})
                statuses.append("error")
                continue

            sdir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(fparq, engine="pyarrow", compression="snappy")
            fempty.unlink(missing_ok=True)
            prof = probe.profile(df)
            log.info("[%d/%d] OK  %-58s %7d rows %5s docs",
                     i, total, tag, len(df), prof.get("docs", ""))
            manifest.append({**ident, "year": year, "status": "ok",
                             "rows": len(df), "docs": prof.get("docs", ""),
                             "first_ts": prof.get("first_ts", ""),
                             "last_ts": prof.get("last_ts", ""),
                             "cols": df.shape[1],
                             "file": str(rel / f"{year}.parquet")})
            statuses.append("ok")

        # explicit marker only for a definitive all-years-empty series
        if statuses and all(st == "empty" for st in statuses):
            sdir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_parquet(sdir / "_empty.parquet", engine="pyarrow")
        else:
            (sdir / "_empty.parquet").unlink(missing_ok=True)

    if manifest:
        man = pd.DataFrame(manifest)
        if manifest_path.exists():
            man = pd.concat([pd.read_csv(manifest_path), man], ignore_index=True)
            man = man.drop_duplicates(
                ["dataset", "variant", "area", "year"], keep="last")
        man.to_csv(manifest_path, index=False)
        log.info("manifest: %d ok, %d empty, %d error | %s",
                 (man["status"] == "ok").sum(), (man["status"] == "empty").sum(),
                 (man["status"] == "error").sum(), manifest_path)

    print(f"\nparquet written under: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
