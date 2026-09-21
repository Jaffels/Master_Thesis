"""Show that the 'Actual Min/Max' you see in TP's Total Load Month-Ahead view
(TR 6.1.A&C&D&E) is already in your pull as the hourly actual_load (6.1.A).

TP's month view buckets the realised load into weeks and shows the min and max
per bucket. This reads the hourly series the pull already wrote and computes the
same weekly min/max, so you can eyeball them against the screenshot
(e.g. CH Jan 2021 Week 1: Actual Min 7211.59, Actual Max 9408.40).

Run from Entsoe/Load/:
    ../../.venv/bin/python verify_ch_load.py CH 2021 1
    ../../.venv/bin/python verify_ch_load.py AT 2023 6
"""

import sys
from pathlib import Path

import pandas as pd

zone = sys.argv[1] if len(sys.argv) > 1 else "CH"
year = int(sys.argv[2]) if len(sys.argv) > 2 else 2021
month = int(sys.argv[3]) if len(sys.argv) > 3 else 1

here = Path(__file__).resolve().parent
pq = here / "Data" / "production" / "actual_load" / "A16_realised" / zone / f"{year}.parquet"
if not pq.exists():
    print(f"not found: {pq}\n(run the pull first, or check the zone/year)")
    sys.exit(1)

df = pd.read_parquet(pq)
col = "Actual Load" if "Actual Load" in df.columns else df.columns[0]
s = df[col].dropna()

# clip to the requested month (index is tz-aware Europe/Zurich)
m = s[(s.index.year == year) & (s.index.month == month)]
if m.empty:
    print(f"{zone} {year}-{month:02d}: no rows in the actual_load parquet")
    sys.exit(0)

print(f"{zone} actual load {year}-{month:02d}: {len(m)} points, "
      f"{s.index.freqstr or 'native'} resolution\n")

# whole-month extremes (must bracket every weekly value)
print(f"month  min={m.min():8.2f}  max={m.max():8.2f}")

# weekly buckets — TP's 'Week N' rows. Week edges in the TP UI may differ by a
# day from this calendar-week split, but the values land in the same range and
# the monthly extremes match exactly.
wk = m.resample("W").agg(["min", "max"])
print("\nper calendar week (Sun-anchored):")
for i, (ts, row) in enumerate(wk.iterrows(), 1):
    print(f"  wk~{i}  min={row['min']:8.2f}  max={row['max']:8.2f}")
