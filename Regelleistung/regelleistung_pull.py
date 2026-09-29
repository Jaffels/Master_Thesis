#!/usr/bin/env python3
"""
regelleistung.net Datacenter — production pull (download raw day files, build parquet).

Default scope = what the thesis needs: German / FCR-cooperation CAPACITY tender
results (prices + demand) for FCR, aFRR, mFRR, 2018-07-01 -> 2026-08-31
(data cut-off 31 Aug 2026). Anonymous bid lists, demands and the ENERGY market
are available with --only.

Coverage (probe 2026-09-29): the datacenter starts with the daily auctions —
aFRR/mFRR from Jul 2018 (4h blocks), FCR from Jul 2019 (daily NEGPOS_00_24,
4h blocks from Jul 2020). Earlier (weekly) German tenders are NOT served;
requests before that return an empty workbook (-> .empty). Hence the default
start 2018-07-01; --start 2015-01-01 works but only adds empty days.
Schema breaks to handle in the clean layer (kept as separate columns here):
  - aFRR/mFRR capacity price unit: [EUR/MW] per block until 2021 ->
    [(EUR/MW)/h] from 2022 (columns *_eur_mw vs *_eur_mw_h); energy-price
    columns only until 2021 (separate energy market from Nov 2020).
  - FCR country prefix: AT_/DE_/CH_ ... until 2022 -> AUSTRIA_/GERMANY_/... from
    2023, and IMPORT(-)_EXPORT(+) -> DEFICIT(-)_SURPLUS(+) (check the sign).

Run from the thesis root (venv active). Run the probe first.
    python Regelleistung/regelleistung_pull.py --plan                       # what would be requested
    python Regelleistung/regelleistung_pull.py                              # download + build
    python Regelleistung/regelleistung_pull.py --only results:CAPACITY:FCR --start 2019-01-01
    python Regelleistung/regelleistung_pull.py --build-only                 # re-build parquet from raw
    python Regelleistung/regelleistung_pull.py --retry-errors               # only dates that failed before

Two stages
1. Download: one request per (task, delivery day). Each answer is kept as the
   original file, so parsing can be redone without the network:
       Data/raw/<report>_<market>/<product>/<year>/<YYYY-MM-DD>.xlsx
       Data/raw/<report>_<market>/<product>/<year>/<YYYY-MM-DD>.empty   (no data that day)
   Errors write nothing (the day is tried again on the next run) and go to
       Data/raw/_download_log.csv       (append-only, one row per request)
   Resumable: existing .xlsx/.empty files are skipped unless --force.
2. Build: per (task, year) all day files are stacked into
       Data/production/<report>_<market>/<product>/<year>.parquet
       Data/production/_build_manifest.csv
   - columns: `query_date` + the source columns, names made snake_case
     (e.g. GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h] -> germany_marginal_capacity_price_eur_mw_h)
   - numbers converted where every non-empty value parses (also decimal comma)
   - rows that repeat across query days (weekly products are returned for every
     day of their week) are dropped: dedup on all columns except `query_date`
   - `delivery_start` / `delivery_end` (datetime64[us, Europe/Zurich], interval
     start/end) where DATE_FROM/DATE_TO and a 4h/hourly product code can be read;
     NaT otherwise (e.g. pre-2018 HT/NT weekly products) — resolve in the clean layer.
   Each year keeps its own column set (schemas change with the reforms); the
   manifest lists a schema hash per year.

SAFETY: writes only under Regelleistung/Data/, never deletes. Parquet files are
replaced atomically. Needs: requests, pandas, pyarrow, openpyxl
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regelleistung_probe as probe  # noqa: E402

TZ = "Europe/Zurich"                 # same offsets as Europe/Berlin; thesis convention
LOG_COLUMNS = ["logged_at", "task", "date", "status", "http", "kind", "bytes",
               "rows", "attempts", "error"]


# ─── Download stage ─────────────────────────────────────────────────────────
def day_range(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days)]


def raw_dir(root: Path, task: probe.Task, day: dt.date) -> Path:
    return root / task.folder / f"{day.year}"


def existing(root: Path, task: probe.Task, day: dt.date) -> Path | None:
    d = raw_dir(root, task, day)
    for ext in ("xlsx", "csv", "empty"):
        p = d / f"{day.isoformat()}.{ext}"
        if p.exists():
            return p
    return None


def append_log(log_csv: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    df.to_csv(log_csv, mode="a", header=not log_csv.exists(), index=False)


def failed_dates(log_csv: Path) -> set[tuple[str, str]]:
    """(task, date) whose LAST logged status is error."""
    if not log_csv.exists():
        return set()
    log = pd.read_csv(log_csv, dtype=str)
    last = log.groupby(["task", "date"]).tail(1)
    return set(map(tuple, last.loc[last.status == "error", ["task", "date"]].values))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def download(tasks: list[probe.Task], days: list[dt.date], raw_root: Path, args) -> dict:
    log_csv = raw_root / "_download_log.csv"
    retry = failed_dates(log_csv) if args.retry_errors else None
    todo: list[tuple[probe.Task, dt.date]] = []
    for t in tasks:
        for day in days:
            if retry is not None:
                if (t.key, day.isoformat()) in retry:
                    todo.append((t, day))
            elif args.force or existing(raw_root, t, day) is None:
                todo.append((t, day))

    print(f"download: {len(todo)} request(s) to do "
          f"({len(tasks) * len(days) - len(todo) if retry is None else 0} already on disk)")
    for t in tasks:
        n = sum(1 for x, _ in todo if x == t)
        print(f"  {t.key:<28} {n:>6} days")
    est_h = len(todo) * (args.interval + 0.6) / 3600
    print(f"  rough runtime at --interval {args.interval}: {est_h:.1f} h")
    if args.plan or not todo:
        return {}

    raw_root.mkdir(parents=True, exist_ok=True)
    session, throttle = probe.new_session(), probe.Throttle(args.interval)
    counts = {"ok": 0, "empty": 0, "error": 0}
    buf: list[dict] = []
    t0 = time.monotonic()
    for i, (t, day) in enumerate(todo, 1):
        f = probe.fetch(session, throttle, t, day, args.country)
        d = raw_dir(raw_root, t, day)
        if f.status == "ok":
            atomic_write_bytes(d / f"{day.isoformat()}.{f.kind}", f.content)
        elif f.status == "empty":
            atomic_write_bytes(d / f"{day.isoformat()}.empty",
                               f"http={f.http} kind={f.kind} bytes={len(f.content)}\n".encode())
        counts[f.status] += 1
        buf.append({"logged_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "task": t.key, "date": day.isoformat(), "status": f.status,
                    "http": f.http, "kind": f.kind, "bytes": len(f.content),
                    "rows": f.rows, "attempts": f.attempts, "error": f.error})
        if f.status == "error":
            print(f"  ERROR {t.key} {day}: {f.error[:160]}")
        if i % 50 == 0 or i == len(todo):
            append_log(log_csv, buf)
            buf = []
            el = time.monotonic() - t0
            eta = el / i * (len(todo) - i) / 60
            print(f"  [{i}/{len(todo)}] {t.key} {day}  ok={counts['ok']} "
                  f"empty={counts['empty']} error={counts['error']}  ETA {eta:.0f} min", flush=True)
    append_log(log_csv, buf)
    return counts


# ─── Build stage ────────────────────────────────────────────────────────────
def snake(name: str) -> str:
    s = str(name).strip()
    s = s.replace("(-)", "_minus").replace("(+)", "_plus").replace("%", "pct")
    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").lower()
    return s or "unnamed"


def unique_names(cols) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        base = snake(c)
        n = seen.get(base, 0)
        out.append(base if n == 0 else f"{base}_{n + 1}")
        seen[base] = n + 1
    return out


def to_numeric_if_clean(s: pd.Series) -> pd.Series:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return s                            # already numeric / datetime
    txt = s.astype("string").str.strip()
    txt = txt.mask(txt.isin(probe.NULL_TOKENS))
    nonnull = txt.notna().sum()
    if nonnull == 0:
        return pd.to_numeric(txt, errors="coerce")
    num = pd.to_numeric(txt, errors="coerce")
    if num.notna().sum() == nonnull:
        return num
    # German format: 1.234,56 -> 1234.56
    alt = txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    num = pd.to_numeric(alt, errors="coerce")
    if num.notna().sum() == nonnull and txt.str.contains(",", regex=False).any():
        return num
    return s


DATE_COLS = ("date_from", "date_to", "datefrom", "dateto", "date")


def parse_date_col(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.normalize()
    txt = s.astype("string").str.strip()
    out = pd.to_datetime(txt, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M", "%d/%m/%Y"):
        miss = out.isna() & txt.notna()
        if not miss.any():
            break
        out = out.fillna(pd.to_datetime(txt.where(miss), format=fmt, errors="coerce"))
    return out.dt.normalize()


# hour window at the end of a product code: POS_000_004, NEGPOS_04_08, NEG_020_024 ...
BLOCK_RE = re.compile(r"(?<!\d)(\d{1,3})_(\d{1,3})$")


def add_delivery(df: pd.DataFrame) -> pd.DataFrame:
    """delivery_start / delivery_end from date_from/date_to + product hour window."""
    dcol_from = next((c for c in df.columns if c in ("date_from", "datefrom", "date")), None)
    dcol_to = next((c for c in df.columns if c in ("date_to", "dateto")), None)
    pcols = [c for c in df.columns if c in ("product", "productname", "product_name")]
    code_all = None
    if pcols:                                  # coalesce: name can change within a year
        code_all = df[pcols[0]].astype("string")
        for c in pcols[1:]:
            code_all = code_all.fillna(df[c].astype("string"))
    pcol = "_product_code" if code_all is not None else None
    if pcol:
        df = df.assign(_product_code=code_all)
    start = pd.Series(pd.NaT, index=df.index, dtype=f"datetime64[us, {TZ}]")
    end = start.copy()
    if dcol_from is None:
        df["delivery_start"], df["delivery_end"] = start, end
        return df.drop(columns=["_product_code"], errors="ignore")
    d_from = parse_date_col(df[dcol_from])
    d_to = parse_date_col(df[dcol_to]) if dcol_to else d_from
    hours = None
    if pcol is not None:
        m = df[pcol].astype("string").str.strip().str.extract(BLOCK_RE)
        h0, h1 = pd.to_numeric(m[0], errors="coerce"), pd.to_numeric(m[1], errors="coerce")
        valid = h0.between(0, 23) & h1.between(1, 24) & (h1 > h0)
        hours = (h0.where(valid), h1.where(valid))
    if hours is not None and hours[0].notna().any():
        h0, h1 = hours
        # block within one day (from == to): wall-clock hours on that day
        same_day = d_from.eq(d_to) & h0.notna()
        s = d_from + pd.to_timedelta(h0, unit="h")
        e = d_from + pd.to_timedelta(h1, unit="h")
        s = s.where(same_day)
        e = e.where(same_day)
    else:
        s = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        e = s.copy()
    # whole-period products without hour code (daily/weekly base): [from 00:00, to+1 00:00)
    whole = s.isna() & d_from.notna() & d_to.notna()
    if pcol is not None:
        code = df[pcol].astype("string").str.upper().fillna("")
        whole &= ~code.str.contains(r"HT|NT|PEAK|OFF", regex=True)   # HT/NT: not contiguous
    s = s.where(~whole, d_from)
    e = e.where(~whole, d_to + pd.Timedelta(days=1))
    for name, v in (("delivery_start", s), ("delivery_end", e)):
        df[name] = (pd.to_datetime(v)
                    .dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
                    .astype(f"datetime64[us, {TZ}]"))
    return df.drop(columns=["_product_code"], errors="ignore")


def read_day_file(p: Path) -> pd.DataFrame:
    kind = "xlsx" if p.suffix == ".xlsx" else "csv"
    frame, _ = probe.read_table(p.read_bytes(), kind)
    frame = frame.dropna(how="all")
    frame.columns = unique_names(frame.columns)
    frame.insert(0, "query_date", pd.Timestamp(p.stem).date())
    return frame


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".parquet")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)          # only our own temp file


def build(tasks: list[probe.Task], years: list[int], raw_root: Path, out_root: Path,
          plan: bool) -> None:
    man_rows = []
    for t in tasks:
        for y in years:
            d = raw_root / t.folder / f"{y}"
            files = sorted(list(d.glob("*.xlsx")) + list(d.glob("*.csv"))) if d.exists() else []
            n_empty = len(list(d.glob("*.empty"))) if d.exists() else 0
            if not files:
                if n_empty:
                    man_rows.append({"task": t.key, "year": y, "files": 0, "empty_days": n_empty,
                                     "rows_raw": 0, "rows": 0, "status": "empty"})
                continue
            if plan:
                print(f"  build {t.key} {y}: {len(files)} files")
                continue
            frames, bad = [], []
            for p in files:
                try:
                    frames.append(read_day_file(p))
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{p.name}: {type(e).__name__}: {e}")
            if not frames:
                man_rows.append({"task": t.key, "year": y, "files": len(files), "status": "error",
                                 "error": "; ".join(bad)[:500]})
                continue
            df = pd.concat(frames, ignore_index=True, sort=False)
            n_raw = len(df)
            if n_raw == 0:                      # header-only workbooks: nothing to write
                man_rows.append({"task": t.key, "year": y, "files": len(files),
                                 "empty_days": n_empty, "rows_raw": 0, "rows": 0,
                                 "status": "empty"})
                continue
            for c in df.columns:
                if c != "query_date":
                    df[c] = to_numeric_if_clean(df[c])
            key = [c for c in df.columns if c != "query_date"]
            df = (df.sort_values("query_date", kind="stable")
                    .drop_duplicates(subset=key, keep="first").reset_index(drop=True))
            df = add_delivery(df)
            df["query_date"] = pd.to_datetime(df["query_date"])
            for c in df.columns:
                if c in DATE_COLS:              # text in old files, Excel dates in new ones
                    df[c] = parse_date_col(df[c])
                elif pd.api.types.is_object_dtype(df[c]):
                    df[c] = df[c].astype("string")   # mixed-type leftovers -> text
            src_cols = [c for c in df.columns
                        if c not in ("query_date", "delivery_start", "delivery_end")]
            out = out_root / t.folder / f"{y}.parquet"
            atomic_parquet(df, out)
            man_rows.append({
                "task": t.key, "year": y, "files": len(files), "empty_days": n_empty,
                "bad_files": len(bad), "rows_raw": n_raw, "rows": len(df),
                "n_cols": len(src_cols), "schema": probe.schema_hash(src_cols),
                "delivery_parsed_share": round(df["delivery_start"].notna().mean(), 4),
                "first_delivery": df["delivery_start"].min(), "last_delivery": df["delivery_start"].max(),
                "first_query": df["query_date"].min().date(), "last_query": df["query_date"].max().date(),
                "status": "ok" if not bad else "partial",
                "error": "; ".join(bad)[:500], "columns": "|".join(src_cols)})
            print(f"  built {t.key:<28} {y}: {len(files)} files -> {len(df)} rows "
                  f"({n_raw - len(df)} repeats dropped), {len(src_cols)} cols, "
                  f"delivery parsed {df['delivery_start'].notna().mean():.0%}"
                  + (f", {len(bad)} BAD files" if bad else ""))
    if plan or not man_rows:
        return
    man = pd.DataFrame(man_rows)
    mpath = out_root / "_build_manifest.csv"
    if mpath.exists():                       # keep rows of tasks/years not rebuilt now
        old = pd.read_csv(mpath, dtype={"year": int})
        keep = ~old.set_index(["task", "year"]).index.isin(man.set_index(["task", "year"]).index)
        man = pd.concat([old[keep], man], ignore_index=True)
    man = man.sort_values(["task", "year"])
    out_root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_root, prefix=".tmp_", suffix=".csv")
    os.close(fd)
    man.to_csv(tmp, index=False)
    os.replace(tmp, mpath)
    print(f"manifest: {mpath}")


# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2018-07-01",
                    help="first delivery day, inclusive (datacenter has nothing earlier)")
    ap.add_argument("--end", default="2026-09-01", help="exclusive (default = cut-off 31 Aug 2026)")
    ap.add_argument("--only", default="",
                    help="tasks as in the probe: 'results', 'anonymous:CAPACITY:aFRR', 'ENERGY', "
                         "'all' (default: capacity results FCR/aFRR/mFRR)")
    ap.add_argument("--country", default=probe.DEFAULT_COUNTRY)
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--data", default=str(here / "Data"))
    ap.add_argument("--force", action="store_true", help="re-download days already on disk")
    ap.add_argument("--retry-errors", action="store_true",
                    help="only re-request (task, day) whose last logged status is error")
    ap.add_argument("--plan", action="store_true", help="show what would be done, request nothing")
    ap.add_argument("--build-only", action="store_true", help="skip download, rebuild parquet")
    ap.add_argument("--no-build", action="store_true", help="download only")
    args = ap.parse_args()

    tasks = probe.all_tasks() if args.only.strip().lower() == "all" else probe.select_tasks(args.only)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")
    days = day_range(start, end)
    years = sorted({d.year for d in days})
    raw_root = Path(args.data) / "raw"
    out_root = Path(args.data) / "production"
    print(f"tasks: {', '.join(t.key for t in tasks)}")
    print(f"days : {start} -> {end - dt.timedelta(days=1)} ({len(days)} days)")

    if not args.build_only:
        counts = download(tasks, days, raw_root, args)
        if counts:
            print(f"download done: {counts}")
    if not args.no_build:
        print("build:")
        build(tasks, years, raw_root, out_root, args.plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
