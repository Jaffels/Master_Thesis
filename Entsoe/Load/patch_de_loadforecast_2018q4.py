"""
One-off patch: fill the Q4 2018 gaps in the DE_LU day-ahead load forecast
(6.1.B) of the 2015-2020 pull (added 2026-09-28).

production_pre2021/load_forecast/day_ahead_A01/DE_LU/2018.parquet is missing
3,368 quarter-hours on 57 days between 1 Oct and 31 Dec 2018 - the first months
of the new DE_LU bidding zone. Before the split the same series came from the
Germany-wide code DE, which may still hold those days.

What it does (nothing else is touched):
  1. queries the day-ahead load forecast for 2018-10-01 .. 2019-01-01, first
     again on DE_LU (in case it was published later), then on DE;
  2. keeps ONLY timestamps missing from the file (existing rows untouched);
  3. level check: on timestamps both the file and a candidate code have, the
     median ratio must be within 3 % (DE excludes Luxembourg, ~1 %) - otherwise
     that code is rejected;
  4. backs up 2018.parquet -> 2018.parquet.bak_q4patch (never overwrites a
     backup), writes the result atomically, and writes a provenance list
     2018_q4patch_filled.csv (timestamp, source code) next to it, so the filled
     rows can be flagged in the clean layer.
If no code has data for the missing timestamps, it says so and changes nothing.

Run from the thesis root:
    python Entsoe/Load/patch_de_loadforecast_2018q4.py            # dry run
    python Entsoe/Load/patch_de_loadforecast_2018q4.py --write    # apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entsoe_load_probe as probe  # noqa: E402  (applies parser patches)
from entsoe import EntsoePandasClient  # noqa: E402

TZ = probe.TZ
TARGET = (Path(__file__).resolve().parent /
          "Data/production_pre2021/load_forecast/day_ahead_A01/DE_LU/2018.parquet")
START = pd.Timestamp("2018-10-01", tz=TZ)
END = pd.Timestamp("2019-01-01", tz=TZ)
CODES = ["DE_LU", "DE"]
MAX_LEVEL_DIFF = 0.03


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="apply the patch (default: dry run)")
    args = ap.parse_args()

    old = pd.read_parquet(TARGET)
    col = old.columns[0]
    grid = pd.date_range(START, END, freq="15min", inclusive="left").tz_convert(old.index.tz)
    missing = grid.difference(old.index)
    print(f"on disk: {len(old)} rows; missing in Q4 2018: {len(missing)} quarter-hours "
          f"on {missing.normalize().nunique()} days")
    if missing.empty:
        print("Q4 2018 already complete - nothing to do.")
        return 0

    key = probe.load_api_key()
    if not key:
        print("ENTSOE_API_KEY not found"); return 1
    client = EntsoePandasClient(api_key=key, timeout=probe.REQUEST_TIMEOUT_S)
    throttle = probe.Throttle(1.0)

    fills: list[pd.DataFrame] = []
    still = missing
    for code in CODES:
        try:
            raw = probe.call_with_retry(client.query_load_forecast, throttle,
                                        country_code=code, start=START, end=END,
                                        process_type="A01")
        except probe.NO_DATA_ERRORS:
            print(f"{code:6s}: no data for Q4 2018")
            continue
        new = probe.tidy(raw, "load_forecast")
        new.index = new.index.tz_convert(old.index.tz)
        new = new[(new.index >= grid[0]) & (new.index <= grid[-1])]
        new = new[~new.index.duplicated(keep="first")]
        if col not in new.columns:
            print(f"{code:6s}: unexpected columns {list(new.columns)} - skipped")
            continue
        both = old.index.intersection(new.index)
        ratio = (new.loc[both, col] / old.loc[both, col]).median() if len(both) else float("nan")
        avail = still.intersection(new.index[new[col].notna()])
        print(f"{code:6s}: {len(new)} rows | overlap with file {len(both)} "
              f"(median ratio {ratio:.4f}) | fills {len(avail)} of {len(still)} missing")
        if len(both) and abs(ratio - 1) > MAX_LEVEL_DIFF:
            print(f"        level differs by more than {MAX_LEVEL_DIFF:.0%} - rejected")
            continue
        if len(avail):
            f = new.loc[avail, [col]].copy()
            f["source_code"] = code
            fills.append(f)
            still = still.difference(avail)
        if still.empty:
            break

    if not fills:
        print("No code has data for the missing timestamps - gap is genuine at source. Nothing changed.")
        return 0

    add = pd.concat(fills).sort_index()
    merged = pd.concat([old, add[[col]].astype(old.dtypes.to_dict())]).sort_index()
    merged.index.name = old.index.name
    assert not merged.index.duplicated().any()
    left = pd.Series(still.normalize()).value_counts().sort_index()
    print(f"would add {len(add)} rows -> {len(merged)} total "
          f"(full year = 35040); still missing: {len(still)} quarter-hours"
          + (f" on {len(left)} days: " + ", ".join(str(d)[:10] for d in left.index[:10]) if len(still) else ""))
    print("filled per source:", add["source_code"].value_counts().to_dict())
    if not args.write:
        print("dry run - rerun with --write to apply.")
        return 0

    bak = TARGET.with_name(TARGET.name + ".bak_q4patch")
    if not bak.exists():
        shutil.copy2(TARGET, bak)
        print(f"backup: {bak.name}")
    tmp = TARGET.with_name(TARGET.name + ".tmp")
    merged.to_parquet(tmp, engine="pyarrow", compression="snappy")
    tmp.replace(TARGET)
    prov = TARGET.with_name("2018_q4patch_filled.csv")
    add[["source_code"]].to_csv(prov)
    print(f"written: {TARGET.name} ({len(merged)} rows) + provenance {prov.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
