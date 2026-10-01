"""
frequency_tso_pull.py — production pull of the TSO 1-second grid-frequency archive
(Continental Europe, TransnetBW measurement, archive hosted on netztransparenz.de).
Download and parsing logic lives in frequency_tso_probe.py.

Stage 1  download   one zip per year -> Frequency/Data/raw/<YYYY>_Frequenz.zip
                    (resumable; a zip placed there by hand is used as is)
Stage 2  build      per year: parse all inner files, local time -> UTC, 1-s grid,
                    then per local month:
                      production/raw_1s/<YYYY>/freq_tso_<YYYY-MM>.parquet  (1-s, float32)
                      production/agg/freq_tso_agg_<YYYY-MM>.parquet        (15-min + 4-h)
                    Features use frequency_features() from EnergyCharts/energycharts_pull.py,
                    so both sources have identical columns.
Stage 3  combine    production/frequency_tso_15min.parquet, frequency_tso_4h.parquet

Default years: 2021 2022 (fills 2021-01 -> 2022-04 before Energy-Charts starts and
overlaps it in May-Jun 2022). Add 2015-2020 with --years if the sample is extended.

Run from the thesis root (venv active):
    python Frequency/frequency_tso_pull.py --plan
    python Frequency/frequency_tso_pull.py                         # 2021 + 2022
    python Frequency/frequency_tso_pull.py --years 2015 2016 2017 2018 2019 2020
    python Frequency/frequency_tso_pull.py --build-only            # rebuild from raw zips
    python Frequency/frequency_tso_pull.py --years 2021 --force    # rebuild one year

Writes only below Frequency/Data/ (atomic writes, never deletes):
    raw/<YYYY>_Frequenz.zip, production/..., _pull_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EnergyCharts"))
import frequency_tso_probe as P                                    # noqa: E402
from energycharts_pull import frequency_features                   # noqa: E402

DATA = P.DATA
PROD = DATA / "production"
MANIFEST = DATA / "_pull_manifest.csv"
LOCAL_TZ = P.LOCAL_TZ
DEFAULT_YEARS = [2021, 2022]
MANIFEST_COLS = ["stage", "key", "status", "rows", "seconds", "coverage", "first_local",
                 "last_local", "file", "message", "run_at"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return {(r["stage"], r["key"]): r for r in csv.DictReader(f)}


def save_manifest(man: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        for k in sorted(man):
            w.writerow({c: man[k].get(c, "") for c in MANIFEST_COLS})
    tmp.replace(MANIFEST)


def record(man: dict, stage: str, key: str, **kw) -> None:
    row = {"stage": stage, "key": key, "run_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    row.update({k: ("" if v is None else v) for k, v in kw.items()})
    man[(stage, key)] = row
    save_manifest(man)


def year_built(man: dict, year: int) -> bool:
    r = man.get(("build", str(year)))
    return bool(r) and r["status"] == "ok"


# ------------------------------------------------------------------------ stages

def stage_download(man: dict, years: list[int], force: bool) -> None:
    for y in years:
        path, msg = P.download_year(y, force=force)
        if path:
            record(man, "download", str(y), status="ok", file=str(path), message=msg,
                   rows=path.stat().st_size)
            log(f"DOWNLOAD {y}: {msg}")
        else:
            record(man, "download", str(y), status="error", message=msg.splitlines()[0][:300])
            log(f"DOWNLOAD {y}: ERROR\n  {msg}")


def stage_build(man: dict, years: list[int], tz_mode: str, force: bool) -> None:
    for y in years:
        if year_built(man, y) and not force:
            log(f"BUILD {y}: already built (use --force to rebuild)")
            continue
        path = P.find_local_zip(y)
        if not path:
            log(f"BUILD {y}: no zip in {P.RAW} — skipped")
            continue
        log(f"BUILD {y}: parsing {path.name} ...")
        t0 = time.time()
        try:
            s, info = P.parse_zip(path, tz_mode=tz_mode, verbose=lambda m: log(m))
        except Exception as e:   # noqa: BLE001
            record(man, "build", str(y), status="error", message=f"{type(e).__name__}: {e}"[:300])
            log(f"BUILD {y}: ERROR {type(e).__name__}: {e}")
            continue
        if s.empty:
            record(man, "build", str(y), status="empty", message="no valid values")
            log(f"BUILD {y}: no valid values")
            continue
        log(f"  parse info: {info}")
        loc = s.index.tz_convert(LOCAL_TZ)
        months = loc.strftime("%Y-%m")
        for m in sorted(set(months)):
            part = s[months == m]
            if not m.startswith(str(y)):
                # e.g. the single 00:00:00 second of 1 Jan of the next year: belongs to
                # the next year's zip — writing it would overwrite that month
                log(f"  {m}: {len(part)} s outside {y} — skipped")
                continue
            a = pd.Timestamp(m + "-01").tz_localize(LOCAL_TZ)
            b = a + pd.offsets.MonthBegin(1)
            cov = len(part) / (b - a).total_seconds()
            raw_path = PROD / "raw_1s" / m[:4] / f"freq_tso_{m}.parquet"
            atomic_parquet(part.astype("float32").to_frame(), raw_path)
            q = frequency_features(part, "15min").assign(resolution="15min")
            h = frequency_features(part, "4h").assign(resolution="4h")
            agg_path = PROD / "agg" / f"freq_tso_agg_{m}.parquet"
            atomic_parquet(pd.concat([q, h]), agg_path)
            pl = part.index.tz_convert(LOCAL_TZ)
            days = pd.date_range(a.tz_localize(None), b.tz_localize(None), freq="D", inclusive="left")
            per_day = pd.Series(1, index=pl).groupby(pl.strftime("%Y-%m-%d")).size()
            per_day = per_day.reindex(days.strftime("%Y-%m-%d"), fill_value=0)
            # files run 00:00:01 -> 24:00:00, so a missing day still gets the 00:00:00
            # second of the day before -> < 60 s counts as missing
            missing = list(per_day[per_day < 60].index)
            partial = [f"{d} ({n / 864:.0f}%)" for d, n in per_day.items() if 60 <= n < 82800]
            note = f"source {path.name}"
            if missing:
                note += f"; missing days: {', '.join(missing)}"
            if partial:
                note += f"; partial days (<23 h): {', '.join(partial)}"
            record(man, "month", m, status="ok", rows=len(part), coverage=f"{cov:.4f}",
                   first_local=pl.min().isoformat(), last_local=pl.max().isoformat(),
                   file=str(agg_path), message=note)
            extra = (f"  missing: {', '.join(missing)}" if missing else "") + \
                    (f"  partial: {', '.join(partial)}" if partial else "")
            log(f"  {m}: {len(part):>10,} s  coverage {cov:.2%}{extra}")
        record(man, "build", str(y), status="ok", rows=info["seconds_after_fill"],
               seconds=round(time.time() - t0), first_local=loc.min().isoformat(),
               last_local=loc.max().isoformat(), file=str(path),
               message=(f"unit {info['unit']}; tz {info['tz_used']} (detected {info['tz_detected']}: "
                        f"{info['tz_note']}); raw rows {info['rows_raw']:,}; seconds with value "
                        f"{info['seconds_with_value']:,}; dropped nonexistent "
                        f"{info['rows_nonexistent_dropped']:,}, out-of-range "
                        f"{info['rows_out_of_range_dropped']:,}")[:600])
        log(f"BUILD {y}: done in {time.time() - t0:.0f} s, {loc.min()} -> {loc.max()}")


def combine() -> None:
    files = sorted((PROD / "agg").glob("freq_tso_agg_*.parquet"))
    if not files:
        log("COMBINE: nothing to combine")
        return
    allagg = pd.concat([pd.read_parquet(p) for p in files])
    for res in ("15min", "4h"):
        part = allagg[allagg["resolution"] == res].drop(columns="resolution")
        part = part[~part.index.duplicated(keep="first")].sort_index()
        out = PROD / f"frequency_tso_{res}.parquet"
        atomic_parquet(part, out)
        log(f"COMBINE: {out.name}: {len(part):,} rows, {part.index.min()} -> {part.index.max()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    ap.add_argument("--plan", action="store_true", help="show what would be done, no requests")
    ap.add_argument("--build-only", action="store_true", help="no downloads; build from zips in raw/")
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download and rebuild the given years")
    ap.add_argument("--tz", choices=["auto", "local", "fixed", "utc"], default="auto",
                    help="timestamp convention of the archive (auto = detect on DST days, default local)")
    args = ap.parse_args()
    man = load_manifest()

    if args.plan:
        for y in args.years:
            z = P.find_local_zip(y)
            log(f"{y}: zip {'present (' + z.name + ')' if z else 'to download'}; "
                f"build {'done' if year_built(man, y) else 'to do'}")
        return
    if args.combine_only:
        combine()
        return
    if not args.build_only:
        stage_download(man, args.years, force=args.force)
    stage_build(man, args.years, tz_mode=args.tz, force=args.force)
    combine()
    errs = [r for r in man.values() if r["status"] == "error"]
    log(f"DONE. Errors: {len(errs)}" + (" — see _pull_manifest.csv" if errs else ""))


if __name__ == "__main__":
    main()
