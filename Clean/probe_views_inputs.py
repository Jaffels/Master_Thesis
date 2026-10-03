"""Probe the inputs of Clean/build_views.py. Read-only: writes nothing.

Run from Master_Thesis with .venv active:
    python Clean/probe_views_inputs.py > /tmp/probe_views.txt 2>&1
"""
import inspect
import re
import sys
import traceback
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

CLEAN = Path(__file__).resolve().parent
sys.path.insert(0, str(CLEAN))
DATA = CLEAN / "Data"
MASTER = DATA / "master"
AB = MASTER / "auction_blocks.parquet"
MT = MASTER / "master_15min.parquet"
DD = MASTER / "data_dictionary.csv"

pd.set_option("display.width", 220, "display.max_columns", 60,
              "display.max_rows", 400, "display.max_colwidth", 70)


def h(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def section(fn):
    try:
        fn()
    except Exception:
        print("!! section failed:\n" + traceback.format_exc())


# --------------------------------------------------------------------------- 1
def p_auction_blocks():
    h("1. auction_blocks — schema")
    df = pd.read_parquet(AB)
    print(f"rows={len(df):,}  cols={df.shape[1]}")
    print(df.dtypes.to_string())

    h("1b. low-cardinality columns (<= 40 distinct): value counts")
    for c in df.columns:
        n = df[c].nunique(dropna=False)
        if n <= 40 and not pd.api.types.is_float_dtype(df[c]):
            print(f"\n-- {c} ({n} distinct)")
            print(df[c].value_counts(dropna=False).to_string())
        else:
            print(f"\n-- {c}: {n:,} distinct (skipped)")

    h("1c. datetime columns: min / max / NaN")
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            print(f"{c:35s} {df[c].min()}  ->  {df[c].max()}  NaN={df[c].isna().sum():,}  tz={getattr(df[c].dt, 'tz', None)}")

    h("1d. 3 sample rows per source")
    src = next((c for c in ["source", "src", "market"] if c in df), None)
    if src:
        print(df.groupby(src, group_keys=False).head(3).T.to_string())
    else:
        print(df.sample(6, random_state=0).T.to_string())

    if {"block_start_utc", "block_end_utc"} <= set(df):
        h("1e. block length (h) by source x product x direction, with first/last block")
        d = df.copy()
        d["_hours"] = (d.block_end_utc - d.block_start_utc).dt.total_seconds() / 3600
        keys = [c for c in ["source", "product", "direction"] if c in d]
        g = d.groupby(keys + ["_hours"], dropna=False).agg(
            n=("_hours", "size"), first=("block_start_utc", "min"), last=("block_start_utc", "max"))
        print(g.to_string())

        h("1f. candidate auction / gate-closure / publication time columns")
        pat = re.compile(r"auction|gate|publ|creat|tender_date|award|closing|deadline", re.I)
        cands = [c for c in d.columns if pat.search(c)]
        print("candidates:", cands or "NONE")
        for c in cands:
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                lead = (d.block_start_utc - d[c]).dt.total_seconds() / 3600
                print(f"\nlead time block_start - {c} (h), by {keys}:")
                print(lead.groupby([d[k] for k in keys]).describe()[["count", "min", "50%", "max"]].to_string())
            else:
                print(f"{c}: dtype {d[c].dtype}, examples {d[c].dropna().unique()[:5]}")


# --------------------------------------------------------------------------- 2
def p_targets():
    h("2. Clean/targets.py — public functions")
    import targets
    for name, obj in inspect.getmembers(targets, inspect.isfunction):
        if obj.__module__ == "targets" and not name.startswith("_"):
            print(f"\n{name}{inspect.signature(obj)}\n  {inspect.getdoc(obj) or '(no docstring)'}")

    h("2b. combine_tenders source")
    print(inspect.getsource(targets.combine_tenders))

    h("2c. combine_tenders(auction_blocks) output")
    df = pd.read_parquet(AB)
    try:
        out = targets.combine_tenders(df)
    except TypeError as e:
        print(f"call with df failed ({e}); trying no-arg call")
        out = targets.combine_tenders()
    print(f"rows in={len(df):,}  out={len(out):,}")
    print(out.dtypes.to_string())
    print(out.head(3).T.to_string())


# --------------------------------------------------------------------------- 3
def p_dictionary():
    h("3. master/data_dictionary.csv")
    dd = pd.read_csv(DD)
    print(f"rows={len(dd):,}  cols={list(dd.columns)}")
    for c in ["table", "source", "resolution", "unit", "aggregation_rule", "availability_rule"]:
        if c in dd:
            print(f"\n-- {c}")
            print(dd[c].value_counts(dropna=False).to_string())
    if "availability_rule" in dd and "column" in dd:
        h("3b. example columns per availability_rule")
        for rule, g in dd.groupby("availability_rule", dropna=False):
            print(f"\n[{rule}] ({len(g)}) e.g. {list(g['column'].head(6))}")
    if "aggregation_rule" in dd and "column" in dd:
        h("3c. example columns per aggregation_rule")
        for rule, g in dd.groupby("aggregation_rule", dropna=False):
            print(f"\n[{rule}] ({len(g)}) e.g. {list(g['column'].head(6))}")


# --------------------------------------------------------------------------- 4
def p_master():
    h("4. master_15min — column groups")
    cols = pq.read_schema(MT).names
    print(f"{len(cols)} columns")
    groups = {
        "time": [c for c in cols if c in ("ts_utc", "ts_local") or c.startswith("ts_")],
        "flag": [c for c in cols if c.startswith("flag_") or c.endswith("_flag")],
        "regime": [c for c in cols if c.startswith("regime_")],
        "src": [c for c in cols if c.startswith("src_")],
        "xchk": [c for c in cols if "_xchk_" in c],
    }
    other = set(cols) - set(sum(groups.values(), []))
    for k, v in groups.items():
        print(f"\n-- {k} ({len(v)}): {v}")
    prefix = pd.Series([c.split("_")[0] if not c.startswith("it_nord") else "it_nord" for c in other])
    print("\n-- data columns by area prefix:\n" + prefix.value_counts().to_string())
    ch = sorted(c for c in other if c.startswith("ch_"))
    print(f"\n-- ch_* data columns ({len(ch)}):")
    for c in ch:
        print("  ", c)

    h("4b. regime / src columns: distinct values")
    sub = pd.read_parquet(MT, columns=groups["time"] + groups["regime"] + groups["src"])
    print(sub.dtypes.to_string())
    for c in groups["regime"] + groups["src"]:
        print(f"\n-- {c}\n{sub[c].value_counts(dropna=False).head(15).to_string()}")


# --------------------------------------------------------------------------- 5
def p_other_tables():
    h("5. outage_events / installed_capacity / gap_list")
    for pat in ["**/outage_events*.parquet", "**/installed_capacity*.parquet", "master/gap_list.parquet"]:
        for f in sorted(DATA.glob(pat)):
            df = pd.read_parquet(f)
            print(f"\n## {f.relative_to(CLEAN)}  rows={len(df):,}")
            print(df.dtypes.to_string())
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    print(f"   {c}: {df[c].min()} -> {df[c].max()}")
            print(df.head(2).T.to_string())


# --------------------------------------------------------------------------- 6
def p_common():
    h("6. Clean/common.py — public helpers")
    import common
    for name, obj in inspect.getmembers(common, inspect.isfunction):
        if obj.__module__ == "common" and not name.startswith("_"):
            doc = (inspect.getdoc(obj) or "").split("\n")[0]
            print(f"{name}{inspect.signature(obj)}  — {doc}")
    print("\nconstants:", [n for n in dir(common) if n.isupper()])


for fn in [p_auction_blocks, p_targets, p_dictionary, p_master, p_other_tables, p_common]:
    section(fn)
print("\nDONE")
