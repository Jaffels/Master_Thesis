#!/usr/bin/env python3
"""
Swissgrid TRE parser (mFRR / RR energy bids, monthly "TRE-Ergebnis" CSVs).

Run from the thesis root:
    python Swissgrid/TRE/swissgrid_tre_parse.py                     # new months only
    python Swissgrid/TRE/swissgrid_tre_parse.py --months 2026-09 --force
    python Swissgrid/TRE/swissgrid_tre_parse.py --years 2026 --force

Needs: pandas, pyarrow

SAFETY: input files are opened read-only. The script writes ONLY under the
output directory (default Swissgrid/TRE/Data/) and never deletes anything.
Existing months are skipped unless --force is given; with --force they are
replaced atomically (write to temp file, then rename).

Source (verified on all 45 files 2023-01 .. 2026-09, downloaded up to 2026-09-24):
  Swissgrid/Manual_Download/Tenders/Tertiary_control_energy/ (recursive).
  File names vary: "2023-01-TRE-Ergebnis.csv", "2026-02-TRE-Ergebnis(1).csv",
  "2026_08_TRE_Ergebnis_2026_09_01_00_01.csv" (= download time stamp).
  latin-1, ';' separated, CRLF, ~1-2 M rows / 70-160 MB per month.
  Columns: Ausschreibung;Von;Bis;Produkt;Angebotene Menge;Einheit;
           Abgerufene Menge;Einheit;Preis;Einheit;Status
  - Ausschreibung "TRE_YY_MM_DD" = delivery day; Von/Bis = LOCAL wall clock
    (Europe/Zurich). Mostly 15-min products; RR_TREnergie-_l uses 1-h blocks.
  - One row = one energy bid for one delivery period.
  - Status: verfügbar (available) / aktiviert (activated, fully or partly) /
    nicht mehr verfügbar (no longer available).
  - Products: TRE_mFRR_{sa,da}{+,-}, RR_TRE_mFRR_sa{+,-}, RR_TREnergie-_l,
    PVTRE_sa- (family / activation type / direction are parsed from the name;
    the raw name is kept in `product`).
  - Autumn DST day: labels 02:00..02:45 occur twice (the first pass ends
    with "02:45-02:00"). Rows are in time order per product, so the first run
    of a label is CEST and the second CET. Spring DST day has no 02:xx rows.
  - Several files for one month (partial download, republished month): the
    file with the most delivery days wins, ties -> newest file. Ignored copies
    are listed in the manifest.

Checks per month (a failing month is reported and skipped, others continue):
  - header exactly as above, units MW / MW / EUR/MWh in every row
  - status and product names known; numbers parse; activated <= offered
  - every Ausschreibung date lies in the file's month
  - delivery days present vs. days in the month (missing days are reported,
    e.g. 2023-12 and 2024-09 have only 27 days at source)

Output (timestamps = delivery START / END, datetime64[us, Europe/Zurich]):
  Data/bids/<YYYY>/<YYYY-MM>.parquet  one row per bid
      delivery_start, delivery_end, duration_min, product, family,
      activation (sa/da/NA), direction (up/down), offered_mw, activated_mw,
      price_eur_mwh, status (available/activated/no_longer_available)
  Data/qh/<YYYY>/<YYYY-MM>.parquet    one row per delivery period x product
      n_bids, n_activated, offered_mw, activated_mw, price_min, price_max,
      act_price_vwap, act_price_marginal (max for up, min for down)
  Data/_parse_manifest.csv            one row per month
  Load: pd.read_parquet("Swissgrid/TRE/Data/qh/")  (all months)
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"
HEADER = ["Ausschreibung", "Von", "Bis", "Produkt", "Angebotene Menge", "Einheit",
          "Abgerufene Menge", "Einheit", "Preis", "Einheit", "Status"]
NAMES = ["auction", "von", "bis", "product", "offered_mw", "u1", "activated_mw", "u2",
         "price_eur_mwh", "u3", "status"]
STATUS = {"verfügbar": "available", "aktiviert": "activated",
          "nicht mehr verfügbar": "no_longer_available"}
PRODUCT_RE = re.compile(r"^(?P<family>RR_TREnergie|RR_TRE_mFRR|TRE_mFRR|PVTRE)"
                        r"(?:_(?P<act>sa|da))?(?P<dir>[+-])(?:_l)?$")
FILE_RES = [re.compile(r"^(\d{4})-(\d{2})-TRE-Ergebnis(?:\(\d+\))?\.csv$", re.I),
            re.compile(r"^(\d{4})_(\d{2})_TRE_Ergebnis(?:_[\d_]+)?\.csv$", re.I)]
AUCTION_RE = r"^TRE_(\d{2})_(\d{2})_(\d{2})$"


def month_of(name: str) -> str | None:
    for rx in FILE_RES:
        m = rx.match(name)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return None


def read_month_file(path: Path) -> pd.DataFrame:
    with open(path, encoding="latin-1") as fh:
        header = fh.readline().rstrip("\r\n").split(";")
    if header != HEADER:
        raise ValueError(f"unexpected header {header}")
    return pd.read_csv(path, sep=";", encoding="latin-1", header=0, names=NAMES,
                       dtype=str, keep_default_na=False, engine="c")


def minutes(s: pd.Series) -> pd.Series:
    hh, mm = s.str.slice(0, 2), s.str.slice(3, 5)
    ok = s.str.match(r"^\d{2}:\d{2}$")
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} bad time labels, e.g. {s[~ok].iloc[0]!r}")
    return hh.astype(int) * 60 + mm.astype(int)


def dst_fold(df: pd.DataFrame, day_mask: pd.Series) -> tuple[np.ndarray, int]:
    """ambiguous flags for tz_localize: True = CEST (first pass) on autumn DST days."""
    amb = np.zeros(len(df), dtype=bool)
    sub = df[day_mask & (df["von_min"] // 60 == 2)]
    if sub.empty:
        return amb, 0, 0
    key = [sub["day"], sub["product"]]
    prev = sub.groupby(key, sort=False)["von_min"].shift()
    run = (sub["von_min"] != prev).groupby(key, sort=False).cumsum()
    rank = run.groupby([sub["day"], sub["product"], sub["von_min"]], sort=False).rank(method="dense")
    if (rank > 2).any():
        raise ValueError("more than two runs of one DST label - order not as expected")
    grp = [sub["day"], sub["product"], sub["von_min"]]
    single = rank.groupby(grp).transform("max") == 1
    first = (rank == 1).to_numpy()
    # Both passes of a label in ONE uninterrupted run (seen for the hourly
    # RR_TREnergie product: 02:00-03:00 CEST directly followed by 02:00-03:00 CET):
    # split the run in half by file order if it has an even number of rows.
    pos = sub.groupby(grp).cumcount()
    size = sub.groupby(grp)["von_min"].transform("size")
    halve = (single & (size % 2 == 0)).to_numpy()
    first = np.where(halve, (pos < size // 2).to_numpy(), first)
    amb[np.flatnonzero(df.index.isin(sub.index))] = first
    return amb, int(single.sum()), int(halve.sum())


def parse_month(path: Path, month: str) -> tuple[pd.DataFrame, dict]:
    df = read_month_file(path)
    info: dict = {"rows_raw": len(df)}
    units = (df["u1"] == "MW") & (df["u2"] == "MW") & (df["u3"] == "EUR/MWh")
    if not units.all():
        raise ValueError(f"{int((~units).sum())} rows with unexpected units")
    st = df["status"].map(STATUS)
    if st.isna().any():
        raise ValueError(f"unknown status {sorted(df.loc[st.isna(), 'status'].unique())[:5]}")
    pm = df["product"].str.extract(PRODUCT_RE)
    if pm["family"].isna().any():
        raise ValueError(f"unknown product {sorted(df.loc[pm['family'].isna(), 'product'].unique())[:5]}")
    a = df["auction"].str.extract(AUCTION_RE)
    if a.isna().any().any():
        raise ValueError(f"bad Ausschreibung, e.g. {df.loc[a.isna().any(axis=1), 'auction'].iloc[0]!r}")
    day = pd.to_datetime("20" + a[0] + "-" + a[1] + "-" + a[2], format="%Y-%m-%d")
    wrong = day.dt.strftime("%Y-%m") != month
    if wrong.any():
        raise ValueError(f"{int(wrong.sum())} rows outside {month}")
    num = {c: pd.to_numeric(df[c], errors="coerce") for c in ("offered_mw", "activated_mw", "price_eur_mwh")}
    for c, s in num.items():
        if s.isna().any():
            raise ValueError(f"{int(s.isna().sum())} non-numeric values in {c}")

    work = pd.DataFrame({"day": day, "product": df["product"],
                         "von_min": minutes(df["von"]), "bis_min": minutes(df["bis"])})
    # autumn DST days in this month (local 02:00 occurs twice)
    days = pd.Series(pd.to_datetime(sorted(day.unique())))
    two = days[[(pd.Timestamp(d).tz_localize(TZ) + pd.Timedelta(hours=24)).tz_convert(TZ).hour == 23
                for d in days]]
    fall = work["day"].isin(two)
    amb, single_runs, halved = dst_fold(work, fall)
    naive = work["day"] + pd.to_timedelta(work["von_min"], unit="min")
    start = pd.DatetimeIndex(naive).tz_localize(TZ, ambiguous=amb, nonexistent="shift_forward")
    dur = (work["bis_min"] - work["von_min"]) % 1440
    dur = dur.where(dur != 0, 1440)
    dur = dur.where(~(fall & (dur > 720)), dur - 1380)          # "02:45-02:00" on the DST day = 15 min
    end = start + pd.to_timedelta(dur.to_numpy(), unit="min")
    info["dst_fall_rows"] = int((fall & (work["von_min"] // 60 == 2)).sum())
    info["dst_single_run_rows"] = single_runs      # label seen in one run only
    info["dst_single_run_rows_halved"] = halved    # ...and split in half (CEST / CET)

    out = pd.DataFrame({
        "delivery_start": start.astype(f"datetime64[us, {TZ}]"),
        "delivery_end": end.astype(f"datetime64[us, {TZ}]"),
        "duration_min": dur.astype("int16").to_numpy(),
        "product": df["product"].astype("category"),
        "family": pm["family"].astype("category"),
        "activation": pm["act"].astype("category"),
        "direction": pm["dir"].map({"+": "up", "-": "down"}).astype("category"),
        "offered_mw": num["offered_mw"].astype("float32"),
        "activated_mw": num["activated_mw"].astype("float32"),
        "price_eur_mwh": num["price_eur_mwh"],
        "status": st.astype("category"),
    })
    over = out["activated_mw"] > out["offered_mw"] + 1e-6
    info["activated_gt_offered"] = int(over.sum())
    info["activated_but_status_not_activated"] = int(((out["activated_mw"] > 0) & (out["status"] != "activated")).sum())
    info["status_activated_but_zero_mw"] = int(((out["activated_mw"] == 0) & (out["status"] == "activated")).sum())
    y, m = map(int, month.split("-"))
    present = set(day.dt.day.unique())
    missing = [d for d in range(1, calendar.monthrange(y, m)[1] + 1) if d not in present]
    info.update(days=len(present), missing_days=" ".join(map(str, missing)),
                n_products=out["product"].nunique(),
                products=" ".join(sorted(out["product"].unique())),
                n_activated=int((out["status"] == "activated").sum()),
                activated_mwh=round(float((out["activated_mw"] * out["duration_min"] / 60).sum()), 1))
    return out, info


def qh_summary(b: pd.DataFrame) -> pd.DataFrame:
    act = b["activated_mw"] > 0
    b = b.assign(_pv=b["price_eur_mwh"] * b["activated_mw"],
                 _act_price=b["price_eur_mwh"].where(act), _isact=act.astype("int32"))
    g = b.groupby(["delivery_start", "delivery_end", "product", "family", "activation", "direction"],
                  observed=True, dropna=False, sort=True)
    s = g.agg(n_bids=("offered_mw", "size"), n_activated=("_isact", "sum"),
              offered_mw=("offered_mw", "sum"), activated_mw=("activated_mw", "sum"),
              price_min=("price_eur_mwh", "min"), price_max=("price_eur_mwh", "max"),
              _pv=("_pv", "sum"), act_max=("_act_price", "max"), act_min=("_act_price", "min")).reset_index()
    s["act_price_vwap"] = (s["_pv"] / s["activated_mw"]).where(s["activated_mw"] > 0)
    s["act_price_marginal"] = np.where(s["direction"] == "up", s["act_max"], s["act_min"])
    return s.drop(columns=["_pv", "act_max", "act_min"])


def atomic_write(df: pd.DataFrame, path: Path, csv: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        df.to_csv(tmp, index=False) if csv else df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)   # only our own temp file
        raise


def count_days(path: Path) -> int:
    days = set()
    with open(path, encoding="latin-1") as fh:
        next(fh)
        for line in fh:
            days.add(line[:12])
    return len(days)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Swissgrid TRE parser")
    ap.add_argument("--input", default="Swissgrid/Manual_Download/Tenders/Tertiary_control_energy")
    ap.add_argument("--output", default="Swissgrid/TRE/Data")
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--months", nargs="*", help="YYYY-MM")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.is_dir():
        print(f"input folder not found: {in_dir} (run from the thesis root)", file=sys.stderr)
        return 2

    by_month: dict[str, list[Path]] = {}
    for p in sorted(in_dir.rglob("*.csv")):
        mo = month_of(p.name)
        if mo:
            by_month.setdefault(mo, []).append(p)
    manifest_path = out_dir / "_parse_manifest.csv"
    old = pd.read_csv(manifest_path, dtype={"month": str}) if manifest_path.exists() else pd.DataFrame()
    rows, n_err = [], 0
    for mo, paths in sorted(by_month.items()):
        if args.years and int(mo[:4]) not in args.years:
            continue
        if args.months and mo not in args.months:
            continue
        bids_path = out_dir / "bids" / mo[:4] / f"{mo}.parquet"
        qh_path = out_dir / "qh" / mo[:4] / f"{mo}.parquet"
        if bids_path.exists() and qh_path.exists() and not args.force:
            print(f"[skip] {mo} (exists; --force to re-parse)")
            continue
        # pick the copy with most delivery days, then newest
        if len(paths) > 1:
            ranked = sorted(paths, key=lambda p: (count_days(p), p.stat().st_mtime))
        else:
            ranked = paths
        src, ignored = ranked[-1], ranked[:-1]
        try:
            bids, info = parse_month(src, mo)
            qh = qh_summary(bids)
            atomic_write(bids, bids_path)
            atomic_write(qh, qh_path)
            rows.append(dict(month=mo, source=str(src), status="ok", **info,
                             ignored_copies="; ".join(map(str, ignored)),
                             parsed_at=dt.datetime.now().isoformat(timespec="seconds")))
            miss = f", missing days: {info['missing_days']}" if info["missing_days"] else ""
            ign = f", {len(ignored)} other copy ignored" if ignored else ""
            print(f"[ok] {mo}: {info['rows_raw']:,} bids, {info['days']} days{miss}{ign}")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[ERROR] {mo} {src}: {e}")
            rows.append(dict(month=mo, source=str(src), status="error", error=str(e)))
    if rows:
        new = pd.DataFrame(rows)
        if not old.empty:
            new = pd.concat([old[~old["month"].isin(new["month"])], new], ignore_index=True)
        atomic_write(new.sort_values("month"), manifest_path, csv=True)
    print(f"done: {n_err} error(s). Manifest: {manifest_path}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
