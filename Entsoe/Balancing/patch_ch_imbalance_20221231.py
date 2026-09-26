"""
One-off patch: fill the missing 31 Dec 2022 in the ENTSO-E CH imbalance-price
file (17.1.G). The production pull wrote 2022.parquet with 34,944 rows; the
last day (96 quarter-hours) is absent, which leaves a one-day gap in
SwissGrid/ImbalancePrices/Data/combined_ch.parquet (ENTSO-E 2021-22 + Swissgrid 2023+).

What it does (nothing else is touched):
  1. queries CH imbalance prices for 2022-12-30 .. 2023-01-02 (small window,
     so a boundary effect at the year end cannot hide the day again);
  2. keeps only 2022-12-31 rows, checks them (96 rows, 15-min grid, same columns);
  3. backs up 2022.parquet -> 2022.parquet.bak_20260926 (never overwrites a backup);
  4. appends ONLY timestamps not already in the file (existing rows untouched)
     and writes the result atomically.
If ENTSO-E has no data for that day, it says so and changes nothing.

Run from the thesis root:
    python Entsoe/Balancing/patch_ch_imbalance_20221231.py            # dry run
    python Entsoe/Balancing/patch_ch_imbalance_20221231.py --write    # apply
Then rebuild the combined file:
    python SwissGrid/ImbalancePrices/swissgrid_imbalance_prices_parse.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entsoe_balancing_probe as probe  # noqa: E402  (applies parser patches)
from entsoe import EntsoePandasClient  # noqa: E402

TZ = probe.TZ
TARGET = Path(__file__).resolve().parent / "Data/production/imbalance_prices/all/CH/2022.parquet"
DAY_START = pd.Timestamp("2022-12-31", tz=TZ)
DAY_END = pd.Timestamp("2023-01-01", tz=TZ)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="apply the patch (default: dry run)")
    args = ap.parse_args()

    old = pd.read_parquet(TARGET)
    have = old[(old.index >= DAY_START) & (old.index < DAY_END)]
    print(f"on disk: {len(old)} rows, last {old.index.max()}; rows on 2022-12-31: {len(have)}")
    if len(have) == 96:
        print("2022-12-31 already complete - nothing to do.")
        return 0

    key = probe.load_api_key()
    if not key:
        print("ENTSOE_API_KEY not found"); return 1
    client = EntsoePandasClient(api_key=key, timeout=probe.REQUEST_TIMEOUT_S)
    throttle = probe.Throttle(1.0)
    try:
        raw = probe.call_with_retry(client.query_imbalance_prices, throttle, country_code="CH",
                                    start=pd.Timestamp("2022-12-30", tz=TZ),
                                    end=pd.Timestamp("2023-01-02", tz=TZ))
    except probe.NO_DATA_ERRORS:
        print("ENTSO-E returned NO DATA for this window - gap is genuine at source. Nothing changed.")
        return 0

    new = probe.tidy(raw, "imbalance_prices")
    new = new[(new.index >= DAY_START) & (new.index < DAY_END)]
    print(f"fetched for 2022-12-31: {len(new)} rows, columns {list(new.columns)}")
    if new.empty:
        print("ENTSO-E has no rows for 2022-12-31 - gap is genuine at source. Nothing changed.")
        return 0

    # checks
    problems = []
    if list(new.columns) != list(old.columns):
        problems.append(f"columns differ: {list(new.columns)} vs {list(old.columns)}")
    grid = pd.date_range(DAY_START, DAY_END, freq="15min", inclusive="left")
    if not new.index.equals(grid):
        problems.append(f"not a clean 96-row 15-min grid ({len(new)} rows)")
    if new.isna().any().any():
        problems.append(f"NaN present: {new.isna().sum().to_dict()}")
    print(new.describe().loc[["min", "50%", "max"]].to_string())
    if problems:
        print("CHECKS FAILED - nothing written:\n  " + "\n  ".join(problems))
        return 1

    add = new[~new.index.isin(old.index)]
    merged = pd.concat([old, add.astype(old.dtypes.to_dict())]).sort_index()
    merged.index = merged.index.astype(old.index.dtype)
    merged.index.name = old.index.name
    assert not merged.index.duplicated().any()
    print(f"would add {len(add)} rows -> {len(merged)} total (expected 35040)")
    if not args.write:
        print("dry run - rerun with --write to apply.")
        return 0

    bak = TARGET.with_name(TARGET.name + ".bak_20260926")
    if not bak.exists():
        shutil.copy2(TARGET, bak)
        print(f"backup: {bak.name}")
    tmp = TARGET.with_name(TARGET.name + ".tmp")
    merged.to_parquet(tmp, engine="pyarrow", compression="snappy")
    tmp.replace(TARGET)
    print(f"written: {TARGET} ({len(merged)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
