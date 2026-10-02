"""DuckDB query layer over the clean/master parquet files (Decision 4).

    from db import connect
    con = connect()
    df = con.sql("SELECT ... FROM master_15min").df()

Every parquet file in Clean/Data/master/ and Clean/Data/<domain>/ becomes a view
named after the file (domain tables as <domain>__<file>). Nothing is stored:
the connection lives in memory and reads the parquet files in place.
Install once:  pip install duckdb
"""
from __future__ import annotations

import config as C


def connect():
    import duckdb  # imported here so the rest of Clean/ works without DuckDB
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    for p in sorted(C.DATA_DIR.glob("*/*.parquet")):
        view = p.stem if p.parent.name == "master" else f"{p.parent.name}__{p.stem}"
        con.execute(f"CREATE OR REPLACE VIEW \"{view}\" AS SELECT * FROM read_parquet('{p.as_posix()}')")
    return con
