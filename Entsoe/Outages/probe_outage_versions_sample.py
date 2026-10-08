"""
Sizing probe: how much of the ORIGINAL 2021-2025 outage history can be
recovered from older document revisions?  (open item 11, step 2, 8 Oct 2026)

Result of probe_outage_versions.py (8 Oct 2026): a query by mRID returns ALL
revisions of a document, but most revisions were re-stamped on 2-8 Oct 2025.
8 of 25 sampled documents (all 2022, none 2024) still had revisions with their
original creation time. This probe measures that share on a random sample per
zone and year, so we can decide whether a full re-pull is worth it.

Per sampled document (random, fixed seed; generation + production units,
planned + forced) it requests all revisions by mRID and classifies each
revision as
    restamped  created 1-15 Oct 2025 while the outage started before Sep 2025
    original   anything else
and reports per zone x year:
    docs, share with >= 1 original revision, share with an original revision
    created BEFORE the outage start (= usable ex ante), median lead time
    (start - first original created, days), mean revisions per document,
    and the API time a full re-pull of that zone-year would need.

Read-only on the data; writes only
    Entsoe/Outages/Data/_probe_versions/sample_revisions_<ts>.csv  (one row per revision)
    Entsoe/Outages/Data/_probe_versions/sample_report_<ts>.txt
Run from Master_Thesis, .venv active (default ~1,300 requests, ~15-25 min):
    python Entsoe/Outages/probe_outage_versions_sample.py
    python Entsoe/Outages/probe_outage_versions_sample.py --n-main 40 --n-other 15   # quicker
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import entsoe_outages_probe as P  # noqa: E402
from entsoe import EntsoePandasClient  # noqa: E402

PROD = HERE / "Data" / "production"
OUT = HERE / "Data" / "_probe_versions"
YEARS = [2021, 2022, 2023, 2024, 2025]
MAIN_ZONES = ["CH", "DE_LU"]
OTHER_ZONES = ["FR", "IT_NORD", "AT"]
# dataset folder -> documentType ; variant folder -> businessType
DATASETS = {"gen_unit_outages": "A80", "prod_unit_outages": "A77"}
VARIANTS = {"planned_A53": "A53", "forced_A54": "A54"}
RESTAMP_FROM = pd.Timestamp("2025-10-01", tz="UTC")
RESTAMP_TO = pd.Timestamp("2025-10-15", tz="UTC")
STARTED_BEFORE = pd.Timestamp("2025-09-01", tz="UTC")


def population(zone: str, year: int) -> pd.DataFrame:
    parts = []
    for ds, doctype in DATASETS.items():
        for var, btype in VARIANTS.items():
            f = PROD / ds / var / zone / f"{year}.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f, columns=["doc_mrid", "revision", "unavail_start", "unavail_end"])
            d = d.drop_duplicates("doc_mrid")
            d["unavail_start"] = pd.to_datetime(d["unavail_start"], utc=True)
            d["unavail_end"] = pd.to_datetime(d["unavail_end"], utc=True)
            d = d[d["unavail_start"].dt.year == year]
            d["doctype"], d["btype"], d["dataset"], d["variant"] = doctype, btype, ds, var
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).drop_duplicates("doc_mrid")


def revisions(client, row, zone: str) -> tuple[pd.DataFrame, str]:
    u0 = row["unavail_start"]
    w0 = (u0 - pd.Timedelta(days=90)).tz_convert("Europe/Brussels")
    w1 = (u0 + pd.Timedelta(days=270)).tz_convert("Europe/Brussels")
    params = {"documentType": row["doctype"], "biddingZone_Domain": P._eic(zone),
              "businessType": row["btype"], "mRID": row["doc_mrid"]}
    try:
        docs = P._fetch_window(client, params, w0, w1, P.PAGE_SIZE_DEFAULT)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), f"error: {P.redact(str(exc))[:120]}"
    if not docs:
        return pd.DataFrame(), "no data"
    df = P.parse_outage_documents(docs)
    if df.empty or "created" not in df:
        return pd.DataFrame(), "no data"
    df = df[["doc_mrid", "revision", "created"]].drop_duplicates()
    df = df[df["doc_mrid"] == row["doc_mrid"]]
    return df, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description="Outage revision sizing probe")
    ap.add_argument("--n-main", type=int, default=100, help="docs per zone-year, CH and DE_LU")
    ap.add_argument("--n-other", type=int, default=25, help="docs per zone-year, FR / IT_NORD / AT")
    ap.add_argument("--interval", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()

    client = EntsoePandasClient(api_key=P.load_api_key(), timeout=P.REQUEST_TIMEOUT_S)
    P.set_request_throttle(P.Throttle(args.interval))
    OUT.mkdir(parents=True, exist_ok=True)

    rows, pop_sizes = [], {}
    t0 = time.monotonic()
    for zone in MAIN_ZONES + OTHER_ZONES:
        n = args.n_main if zone in MAIN_ZONES else args.n_other
        for year in YEARS:
            pop = population(zone, year)
            pop_sizes[(zone, year)] = len(pop)
            if pop.empty:
                print(f"{zone} {year}: no documents")
                continue
            sample = pop.sample(min(n, len(pop)), random_state=args.seed)
            ok = 0
            for _, r in sample.iterrows():
                revs, status = revisions(client, r, zone)
                if status == "ok":
                    ok += 1
                    for _, v in revs.iterrows():
                        rows.append({"zone": zone, "year": year, "dataset": r["dataset"],
                                     "variant": r["variant"], "doc_mrid": r["doc_mrid"],
                                     "unavail_start": r["unavail_start"],
                                     "revision": v["revision"], "created": v["created"]})
                else:
                    rows.append({"zone": zone, "year": year, "dataset": r["dataset"],
                                 "variant": r["variant"], "doc_mrid": r["doc_mrid"],
                                 "unavail_start": r["unavail_start"], "status": status})
            print(f"{zone} {year}: {ok}/{len(sample)} documents answered "
                  f"(population {len(pop)}; {time.monotonic() - t0:.0f} s elapsed)")

    res = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    res.to_csv(OUT / f"sample_revisions_{stamp}.csv", index=False)

    rv = res[res["created"].notna()].copy() if "created" in res else pd.DataFrame()
    lines = [f"Outage revision sizing probe {stamp}",
             "restamped = created 1-15 Oct 2025 for an outage that started before Sep 2025", ""]
    if rv.empty:
        lines.append("No revisions returned.")
    else:
        rv["created"] = pd.to_datetime(rv["created"], utc=True)
        rv["unavail_start"] = pd.to_datetime(rv["unavail_start"], utc=True)
        rv["restamped"] = (rv["created"].between(RESTAMP_FROM, RESTAMP_TO)
                           & (rv["unavail_start"] < STARTED_BEFORE))
        rv["orig_before_start"] = (~rv["restamped"]) & (rv["created"] < rv["unavail_start"])
        per_doc = rv.groupby(["zone", "year", "doc_mrid"]).agg(
            n_rev=("revision", "nunique"),
            any_original=("restamped", lambda s: (~s).any()),
            exante=("orig_before_start", "any"),
            first_orig=("created", lambda s: s[~rv.loc[s.index, "restamped"]].min()),
            start=("unavail_start", "first")).reset_index()
        per_doc["lead_days"] = (per_doc["start"] - per_doc["first_orig"]).dt.total_seconds() / 86400
        sec_per_req = (time.monotonic() - t0) / max(res["doc_mrid"].nunique(), 1)
        tab = per_doc.groupby(["zone", "year"]).agg(
            docs=("doc_mrid", "size"),
            share_any_original=("any_original", "mean"),
            share_exante=("exante", "mean"),
            median_lead_days=("lead_days", "median"),
            mean_revisions=("n_rev", "mean")).reset_index()
        tab["population"] = [pop_sizes.get((z, y), 0) for z, y in zip(tab["zone"], tab["year"])]
        tab["full_repull_h"] = tab["population"] * sec_per_req / 3600
        lines.append(tab.round(3).to_string(index=False))
        tot_h = tab["full_repull_h"].sum()
        ex = per_doc.groupby("year")["exante"].mean()
        lines += ["", f"seconds per document (measured): {sec_per_req:.2f}",
                  f"full re-pull of all 2021-2025 documents, all zones: ~{tot_h:.1f} h",
                  "share usable ex ante by year (all zones): "
                  + ", ".join(f"{y} {v:.0%}" for y, v in ex.items()), "",
                  "Send this report to Claude to decide: full re-pull, CH-only re-pull, or limitation."]
    errs = int(res.get("status", pd.Series(dtype=str)).fillna("").str.startswith("error").sum())
    lines.append(f"errors: {errs}")
    (OUT / f"sample_report_{stamp}.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
