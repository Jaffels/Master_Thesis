"""
Build the coverage CSVs that drive the 2015-2020 pull (added 2026-09-28).

Reads the three probe windows in <Domain>/Data/_probe_pre2021/
(Jan 2015, Jun 2017, Jun 2019) and writes, next to them:

  _coverage_pre_split.csv   series + codes for 1 Jan 2015 -> 30 Sep 2018
  _coverage_post_split.csv  series + codes for 1 Oct 2018 -> 31 Dec 2020
  (Balancing: one _coverage_pre2021.csv - no German series, so no split)

Rules
  * pre  = status=ok in Jan 2015 or Jun 2017 (Jun 2017 verdict wins).
  * post = status=ok in Jun 2019, plus every pre series that was empty in
    Jun 2019 (event data can be empty in one month), with German pre-split
    codes mapped to DE_LU (DE / DE_AT_LU -> DE_LU).
  * A zonal DE_LU series that resolved to ONE German TSO (Amprion, TenneT,
    50Hertz, TransnetBW = a quarter of Germany or less) is dropped before
    the split and set back to DE_LU after it.
  * Exclusions: Generation per-unit (16.1.A) outside CH; Balancing DE.

Read-only on the probe files; writes only the CSVs above. Run from the
thesis root:  python Entsoe/build_pre2021_coverage.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WINDOWS = ["20150101_20150201", "20170601_20170701", "20190601_20190701"]
TSOS = {"DE_AMPRION", "DE_TENNET", "DE_50HZ", "DE_TRANSNET"}
POST_MAP = {"DE": "DE_LU", "DE_AT_LU": "DE_LU"}


def read(domain: str, w: str) -> pd.DataFrame:
    f = ROOT / domain / "Data" / "_probe_pre2021" / f"_coverage_{w}.csv"
    if not f.exists():  # e.g. Load was probed in 2015 and 2019 only
        print(f"  {domain}: no probe for {w} (skipped)")
        return pd.DataFrame()
    df = pd.read_csv(f)
    return df[df["status"] == "ok"].copy()


def map_code(code: str) -> str:
    return ">".join(POST_MAP.get(p, p) for p in str(code).split(">"))


def build(domain: str) -> None:
    key = ["dataset", "variant", "border" if domain == "Transmission" else "area"]
    codes = (["from_code_used", "to_code_used"] if domain == "Transmission"
             else ["area_code_used"])
    w15, w17, w19 = (read(domain, w) for w in WINDOWS)
    pre = pd.concat([w for w in (w15, w17) if not w.empty]).drop_duplicates(key, keep="last")
    post = w19.drop_duplicates(key, keep="last")
    carry = pre.merge(post[key], on=key, how="left", indicator=True)
    carry = carry[carry["_merge"] == "left_only"].drop(columns="_merge")
    for c in codes:
        carry[c] = carry[c].map(map_code)
    post = pd.concat([post, carry], ignore_index=True)

    if domain in ("Load", "Generation", "Outages"):
        tso_pre = (pre["area"] == "DE_LU") & pre["area_code_used"].isin(TSOS)
        tso_post = (post["area"] == "DE_LU") & post["area_code_used"].isin(TSOS)
        for _, r in pre[tso_pre].iterrows():
            print(f"  {domain}: drop before split (single TSO) "
                  f"{r.dataset}/{r.variant} <- {r.area_code_used}")
        pre = pre[~tso_pre]
        post.loc[tso_post, "area_code_used"] = "DE_LU"
    if domain == "Generation":
        pre = pre[~((pre["dataset"] == "actual_generation_unit") & (pre["area"] != "CH"))]
        post = post[~((post["dataset"] == "actual_generation_unit") & (post["area"] != "CH"))]

    out = ROOT / domain / "Data" / "_probe_pre2021"
    if domain == "Balancing":
        allw = pd.concat([pre, post]).drop_duplicates(key, keep="last")
        allw = allw[allw["area"] != "DE"]
        allw.to_csv(out / "_coverage_pre2021.csv", index=False)
        print(f"{domain:12s} one file: {len(allw)} series -> _coverage_pre2021.csv")
        return
    pre.to_csv(out / "_coverage_pre_split.csv", index=False)
    post.to_csv(out / "_coverage_post_split.csv", index=False)
    print(f"{domain:12s} pre-split {len(pre):3d} series | post-split {len(post):3d} "
          f"series ({len(carry)} carried over from pre)")


if __name__ == "__main__":
    for d in ["Load", "Generation", "Transmission", "Outages", "Balancing"]:
        build(d)
