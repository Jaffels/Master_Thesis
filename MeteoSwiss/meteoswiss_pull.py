"""
MeteoSwiss pull — hourly data for every station in Data/station_selection.csv.

For each station it downloads the hourly `historical` decade files that overlap
the period plus the `recent` file (1 Jan this year -> yesterday), keeps the
thesis parameters, converts timestamps from "UTC, interval end" to
"Europe/Zurich, interval start", puts the data on a complete hourly grid
(missing hours = NaN) and writes one parquet per station.

Outputs (MeteoSwiss/Data/):
  raw/<ABBR>.parquet     timestamp (interval start, Europe/Zurich) + parameters
  raw/_pull_manifest.csv per station: rows, span, missing share per parameter

Existing station files are skipped unless --force (use --force after a new
month to refresh the `recent` part). Never deletes anything.

Usage (from the thesis root, venv active):
    python MeteoSwiss/meteoswiss_pull.py
    python MeteoSwiss/meteoswiss_pull.py --plan              # dry run
    python MeteoSwiss/meteoswiss_pull.py --force             # refresh all
    python MeteoSwiss/meteoswiss_pull.py --stations SMA KLO  # subset
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meteoswiss_common as C  # noqa: E402


def wanted_params() -> list[str]:
    params = sorted(set(sum(C.REQUIRED.values(), [])) | set(C.OPTIONAL))
    snow_file = C.META_DIR / "snow_parameter.txt"
    if snow_file.exists() and snow_file.read_text().strip():
        params.append(snow_file.read_text().strip())
    return params


def hourly_asset_urls(abbr: str, start_year: int, end_year: int) -> list[tuple[str, str]]:
    """(asset name, href) for hourly historical decades overlapping the years + recent."""
    item = C.http_get(f"{C.BASE}/collections/{C.COLLECTION}/items/{abbr.lower()}").json()
    out = []
    for name, a in item["assets"].items():
        m = re.search(r"_h_historical_(\d{4})-(\d{4})\.csv$", name)
        if m and int(m.group(2)) >= start_year and int(m.group(1)) <= end_year:
            out.append((name, a["href"]))
        elif name.endswith("_h_recent.csv"):
            out.append((name, a["href"]))
    # historical first, recent last -> recent wins on any overlap
    return sorted(out, key=lambda x: (x[0].endswith("_recent.csv"), x[0]))


def to_hourly_frame(parts: list[pd.DataFrame], params: list[str]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    ts_end = C.parse_ms_time(df["reference_timestamp"])
    keep = [p for p in params if p in df.columns]
    out = df[keep].apply(pd.to_numeric, errors="coerce")
    out.index = ts_end
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # complete hourly grid in UTC (interval-end labels), then shift to interval start
    start_end_utc = (pd.Timestamp(C.PULL_START_LOCAL, tz=C.TZ) + pd.Timedelta(hours=1)).tz_convert("UTC")
    grid = pd.date_range(start_end_utc, out.index.max(), freq="h", tz="UTC")
    out = out.reindex(grid)
    for p in params:                          # stable schema across stations
        if p not in out.columns:
            out[p] = float("nan")
    out = out[params]
    out.index = C.end_utc_to_start_local(out.index)
    out.index.name = "timestamp"
    return out.reset_index()


def pull_station(abbr: str, params: list[str]) -> pd.DataFrame:
    start_year = pd.Timestamp(C.PULL_START_LOCAL).year
    assets = hourly_asset_urls(abbr, start_year, pd.Timestamp.now().year)
    if not assets:
        raise RuntimeError("no hourly assets found")
    parts = []
    for name, href in assets:
        df = C.read_semicolon_csv(C.http_get(href).content)
        print(f"    {name}: {len(df):,} rows")
        parts.append(df)
        time.sleep(0.3)
    return to_hourly_frame(parts, params)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--stations", nargs="*")
    args = ap.parse_args()

    if not C.SELECTION_CSV.exists():
        sys.exit("Run meteoswiss_probe.py first (Data/station_selection.csv missing).")
    stations = sorted(pd.read_csv(C.SELECTION_CSV)["station_abbr"].astype(str).unique())
    if args.stations:
        stations = [s for s in stations if s in set(args.stations)]
    params = wanted_params()
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)

    todo = [s for s in stations if args.force or not (C.RAW_DIR / f"{s}.parquet").exists()]
    print(f"{len(stations)} stations selected, {len(todo)} to pull; parameters: {params}")
    if args.plan:
        print("Plan:", ", ".join(todo) or "nothing to do")
        return

    errors = []
    for i, abbr in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {abbr}")
        try:
            df = pull_station(abbr, params)
            path = C.RAW_DIR / f"{abbr}.parquet"
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(path)
            print(f"    -> {len(df):,} hours, {df.timestamp.min()} .. {df.timestamp.max()}")
        except Exception as e:  # keep going; report at the end
            print(f"    ERROR: {e}")
            errors.append((abbr, str(e)))

    # manifest rebuilt from disk
    rows = []
    for abbr in stations:
        path = C.RAW_DIR / f"{abbr}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        row = {"station_abbr": abbr, "rows": len(df),
               "first": df.timestamp.min(), "last": df.timestamp.max()}
        for p in params:
            row[f"missing_{p}"] = round(float(df[p].isna().mean()), 4) if p in df else 1.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(C.RAW_DIR / "_pull_manifest.csv", index=False)
    print(f"\nManifest: {len(rows)} stations. Errors: {len(errors)}")
    for abbr, e in errors:
        print(f"  {abbr}: {e}")


if __name__ == "__main__":
    main()
