"""
Rebuild a domain's Data/production/_pull_manifest.csv from what is actually on
disk (fix for stale manifests found 2026-09-26: Balancing listed 18 entries
against 209 files, Generation 37 against 228).

Truth = the parquet files. For every <dataset>/<variant>/<area>/<year>.parquet:
rows, cols, first/last timestamp and na_share are read from the parquet
METADATA (row-group statistics) - no data is loaded, so the 14M-row per-unit
files take milliseconds. Definitions match probe.profile():
  cols     = number of data columns (index excluded)
  na_share = mean over numeric columns of that column's NaN share
area_code comes from the old manifest if it has one for that series, else from
the probe coverage CSVs (_coverage_union.csv preferred), else blank.

Years with NO file are kept from the old manifest as rows=0 entries (a real
"API returned no data" record) - they are not invented.

The old manifest is backed up to _pull_manifest.csv.bak_<YYYYMMDD> first
(never overwritten). Column order matches what each domain's pull writes, so
the pulls keep appending to the rebuilt file normally.

Usage (from the thesis root):
    python Entsoe/rebuild_manifest_from_disk.py Balancing            # dry run
    python Entsoe/rebuild_manifest_from_disk.py Balancing --write
    python Entsoe/rebuild_manifest_from_disk.py Generation --write
Only for the per-year layout (Balancing, Generation, Load). Transmission and
Outages keep their own manifest formats and are up to date.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
COLUMNS = {
    "Balancing": ["dataset", "variant", "area", "area_code", "year", "rows", "file",
                  "first_ts", "last_ts", "cols"],
    "Generation": ["dataset", "variant", "area", "area_code", "year", "rows",
                   "first_ts", "last_ts", "cols", "na_share", "file"],
    "Load": ["dataset", "variant", "area", "area_code", "year", "rows",
             "first_ts", "last_ts", "cols", "file"],
}
KEY = ["dataset", "variant", "area", "year"]


def file_stats(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    md, schema = pf.metadata, pf.schema_arrow
    idx = []
    if schema.pandas_metadata:
        idx = [c for c in schema.pandas_metadata.get("index_columns", []) if isinstance(c, str)]
    n = md.num_rows
    data_cols = [f for f in schema if f.name not in idx]
    numeric = [f.name for f in data_cols
               if pd.api.types.is_numeric_dtype(f.type.to_pandas_dtype())]
    nulls = {c: 0 for c in numeric}
    tmin = tmax = None
    import pyarrow as pa
    ts_candidates = [c for c in idx if pa.types.is_timestamp(schema.field(c).type)]
    ts_col = ts_candidates[0] if ts_candidates else None
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for i in range(rg.num_columns):
            col = rg.column(i)
            name, st = col.path_in_schema, col.statistics
            if st is None:
                continue
            if name in nulls:
                nulls[name] += st.null_count
            if name == ts_col and st.has_min_max:
                tmin = st.min if tmin is None else min(tmin, st.min)
                tmax = st.max if tmax is None else max(tmax, st.max)
    tz = getattr(schema.field(ts_col).type, "tz", None) if ts_col else None
    fmt = lambda t: str(pd.Timestamp(t).tz_convert(tz)) if (t is not None and tz) else ("" if t is None else str(pd.Timestamp(t)))
    na = round(sum(nulls[c] / n for c in numeric) / len(numeric), 3) if (numeric and n) else ""
    return {"rows": n, "cols": len(data_cols), "first_ts": fmt(tmin) if n else "",
            "last_ts": fmt(tmax) if n else "", "na_share": na}


def area_codes(data: Path, old: pd.DataFrame) -> dict:
    codes = {}
    covs = sorted(data.glob("_coverage_*.csv"), key=lambda p: (p.name == "_coverage_union.csv", p.stat().st_mtime))
    for c in covs:  # later (union / newest) wins
        df = pd.read_csv(c)
        df = df[df.get("status") == "ok"] if "status" in df else df
        for r in df.itertuples():
            codes[(r.dataset, r.variant, r.area)] = str(getattr(r, "area_code_used", "") or "")
    if not old.empty and "area_code" in old:
        for r in old.dropna(subset=["area_code"]).itertuples():
            codes[(r.dataset, r.variant, r.area)] = str(r.area_code)
    return codes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("domain", choices=sorted(COLUMNS))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    data = HERE / a.domain / "Data"
    prod = data / "production"
    man_path = prod / "_pull_manifest.csv"
    old = pd.read_csv(man_path) if man_path.exists() else pd.DataFrame(columns=COLUMNS[a.domain])
    codes = area_codes(data, old)

    rows = []
    for f in sorted(prod.glob("*/*/*/*.parquet")):
        if not f.stem.isdigit():
            continue
        ds, var, area = f.parent.parent.parent.name, f.parent.parent.name, f.parent.name
        st = file_stats(f)
        rows.append({"dataset": ds, "variant": var, "area": area,
                     "area_code": codes.get((ds, var, area), ""), "year": int(f.stem),
                     **st, "file": str(f.relative_to(prod))})
    disk = pd.DataFrame(rows)

    # keep old rows=0 records only for series-years that have no file on disk
    empties = old[(old["rows"] == 0)].copy() if not old.empty else old
    if not empties.empty:
        have = set(map(tuple, disk[KEY].values))
        empties = empties[[tuple(r) not in have for r in empties[KEY].values]]
    new = pd.concat([disk, empties], ignore_index=True)
    new = new.reindex(columns=COLUMNS[a.domain]).sort_values(KEY).reset_index(drop=True)
    new["year"] = new["year"].astype(int)

    print(f"{a.domain}: old manifest {len(old)} entries ({(old['rows'] > 0).sum() if len(old) else 0} with data) "
          f"-> rebuilt {len(new)} entries ({(new['rows'] > 0).sum()} files on disk, {len(empties)} kept empty records)")
    print(new.groupby(["dataset"]).agg(series=("area", "size"), rows=("rows", "sum")).to_string())
    if not a.write:
        print("dry run - add --write to replace the manifest (old one is backed up first)")
        return 0
    bak = man_path.with_name(f"{man_path.name}.bak_{dt.date.today():%Y%m%d}")
    if man_path.exists() and not bak.exists():
        shutil.copy2(man_path, bak)
        print(f"backup: {bak.name}")
    tmp = man_path.with_suffix(".csv.tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(man_path)
    print(f"written: {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
