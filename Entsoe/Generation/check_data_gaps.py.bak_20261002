"""Repeatable gap checks for two ENTSO-E domains (read-only by default).

  generation  16.1.A actual generation per unit: missing hours per area-year,
              whole missing days, and hours lost on DST-switch days.
  fallback    IF aFRR 3.10 fall-backs (CH): documents deduplicated across the
              business-type folders, fall-back hours per year, publication lag.

Run from Master_Thesis with .venv active:
  python Entsoe/check_data_gaps.py generation
  python Entsoe/check_data_gaps.py fallback
  python Entsoe/check_data_gaps.py all --write     # also save CSVs to Entsoe/_checks/

Notes
- Generation is checked at hourly resolution (UTC hour buckets): an hour counts
  as present if any unit has a row in it. Missing single quarter-hours inside an
  hour are not reported.
- The expected grid runs from 1 Jan 00:00 local to the last timestamp in the
  file, so a file that ends early (e.g. IT_NORD 2026) shows up in `last_ts`,
  not as missing hours.
- Nothing is modified. --write only creates/overwrites CSVs in Entsoe/_checks/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

GEN_DIR = Path("Entsoe/Generation/Data/production/actual_generation_unit/per_unit")
FB_DIR = Path("Entsoe/Outages/Data/production/fallback_afrr")
OUT_DIR = Path("Entsoe/_checks")
AREAS = ["CH", "AT", "DE_LU", "FR", "IT_NORD"]
TZ = "Europe/Zurich"


def last_sunday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year, month, 31)
    return d - pd.Timedelta(days=(d.weekday() + 1) % 7)


def check_generation(write: bool) -> None:
    summary, detail = [], []
    for area in AREAS:
        for f in sorted((GEN_DIR / area).glob("*.parquet")):
            year = int(f.stem)
            df = pd.read_parquet(f)
            ts = pd.DatetimeIndex(df.index).tz_convert(TZ)
            hours = pd.DatetimeIndex(ts.tz_convert("UTC").floor("h").unique())
            start = pd.Timestamp(year, 1, 1, tz=TZ).tz_convert("UTC")
            grid = pd.date_range(start, hours.max(), freq="h")
            miss = grid.difference(hours)
            miss_local = pd.Series(miss.tz_convert(TZ).date, dtype="object")
            per_day = miss_local.value_counts().sort_index()

            dst_days = {last_sunday(year, 3).date(), last_sunday(year, 10).date()}
            dst_miss = int(per_day[per_day.index.isin(dst_days)].sum())
            # full days: every hour of that local day missing (23/24/25 h)
            full_days = []
            for day, n in per_day.items():
                d0 = pd.Timestamp(day).tz_localize(TZ)
                day_len = int((d0 + pd.Timedelta(days=1)).tz_localize(None).tz_localize(TZ)
                              .tz_convert("UTC").value
                              - d0.tz_convert("UTC").value) // 3_600_000_000_000
                if n >= day_len:
                    full_days.append(str(day))

            summary.append(dict(
                area=area, year=year,
                last_ts=ts.max().strftime("%Y-%m-%d %H:%M"),
                missing_hours=len(miss),
                missing_full_days=len(full_days),
                dst_missing_hours=dst_miss,
                full_days=", ".join(full_days) if len(full_days) <= 6
                else f"{', '.join(full_days[:6])}, ... (+{len(full_days) - 6})",
            ))
            for day, n in per_day.items():
                detail.append(dict(area=area, year=year, date=str(day),
                                   missing_hours=int(n),
                                   dst_day=day in dst_days,
                                   full_day=str(day) in full_days))

    s = pd.DataFrame(summary)
    print("\n=== Generation 16.1.A per unit: hourly gaps ===")
    print(s.to_string(index=False))
    print("\nHours missing on DST days, by area:")
    print(s.groupby("area", sort=False).dst_missing_hours.sum().to_string())
    if write:
        OUT_DIR.mkdir(exist_ok=True)
        s.to_csv(OUT_DIR / "generation_per_unit_summary.csv", index=False)
        pd.DataFrame(detail).to_csv(OUT_DIR / "generation_per_unit_missing_hours.csv",
                                    index=False)
        print(f"\nwritten: {OUT_DIR}/generation_per_unit_summary.csv, "
              f"{OUT_DIR}/generation_per_unit_missing_hours.csv")


def check_fallback(write: bool) -> None:
    files = sorted(FB_DIR.glob("*/CH/*.parquet"))
    if not files:
        print(f"no files under {FB_DIR}/*/CH/")
        return
    raw = pd.concat([pd.read_parquet(f).assign(folder=f.parent.parent.name)
                     for f in files], ignore_index=True)
    df = raw.drop_duplicates(["doc_mrid", "revision"]).copy()
    df["hours"] = (df.unavail_end - df.unavail_start).dt.total_seconds() / 3600
    df["year"] = df.unavail_start.dt.year
    df["lag_days"] = (df.created - df.unavail_start).dt.total_seconds() / 86400

    yearly = df.groupby("year").agg(
        documents=("doc_mrid", "size"),
        fallback_hours=("hours", "sum"),
        full_days=("hours", lambda h: int((h >= 23).sum())),
        first=("unavail_start", "min"),
        last_end=("unavail_end", "max"),
        median_publication_lag_days=("lag_days", "median"),
    )
    for y in yearly.index:
        y0 = pd.Timestamp(y, 1, 1, tz=TZ)
        y1 = min(pd.Timestamp(y + 1, 1, 1, tz=TZ), yearly.loc[y, "last_end"])
        yearly.loc[y, "share_of_period"] = round(
            yearly.loc[y, "fallback_hours"] / ((y1 - y0).total_seconds() / 3600), 3)
    yearly["fallback_hours"] = yearly.fallback_hours.round(1)
    yearly["median_publication_lag_days"] = yearly.median_publication_lag_days.round(1)

    print("\n=== CH aFRR fall-backs (IF 3.10) ===")
    print(f"rows on disk: {len(raw)}  unique documents: {len(df)}  "
          f"folders: {raw.folder.nunique()}")
    print(yearly.drop(columns=["first", "last_end"]).to_string())
    print(f"latest fall-back end: {df.unavail_end.max()}")

    # Sanity checks: flag anything that would change the conclusions
    checks = {
        "reason codes": sorted(df.get("ts.reason.code", pd.Series()).dropna().unique()),
        "start hours (local)": sorted(int(h) for h in df.unavail_start.dt.hour.unique()),
        "revisions": sorted(df.revision.dropna().unique().tolist()),
        "has quantity column": "quantity" in df.columns,
    }
    print("\nchecks (baseline 2026-09-28: reason ['B13'], start hours [0], "
          "revisions [1], quantity False):")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    if write:
        OUT_DIR.mkdir(exist_ok=True)
        yearly.to_csv(OUT_DIR / "fallback_afrr_ch_yearly.csv")
        df.drop(columns=["folder"]).sort_values("unavail_start").to_csv(
            OUT_DIR / "fallback_afrr_ch_documents.csv", index=False)
        print(f"\nwritten: {OUT_DIR}/fallback_afrr_ch_yearly.csv, "
              f"{OUT_DIR}/fallback_afrr_ch_documents.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["generation", "fallback", "all"])
    ap.add_argument("--write", action="store_true",
                    help="save CSVs to Entsoe/_checks/")
    a = ap.parse_args()
    if not Path("Entsoe").is_dir():
        sys.exit("run from the Master_Thesis folder")
    pd.set_option("display.width", 200)
    if a.check in ("generation", "all"):
        check_generation(a.write)
    if a.check in ("fallback", "all"):
        check_fallback(a.write)


if __name__ == "__main__":
    main()
