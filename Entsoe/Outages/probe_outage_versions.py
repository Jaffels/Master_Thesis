"""
Probe: do earlier versions of ENTSO-E outage documents (2021 -> mid 2025) exist
with their ORIGINAL creation times?  (open item 11 / to-do 7.6, 8 Oct 2026)

Background: in the production pull every outage document for 2021 -> Aug/Sep
2025 comes back as ONE revision whose `created` lies in 2025/26 (median lag
400-1,600 days), although revision numbers go up to 17. So the platform keeps
the revision counter but apparently re-stamped the documents. If older
revisions can still be requested, the ex-ante outage features for 2021-2025
could be rebuilt; if not, it stays a limitation (thesis 3.9).

Two tests per sampled document (CH and DE_LU, planned + forced generation
units, 2022 and 2024):
  A  query by mRID (the API's document id) over the outage period
     -> how many revisions come back, and with which `created` times?
  B  query by update window (PeriodStartUpdate / PeriodEndUpdate = the month
     before the outage starts) -> does the API know any version that was
     created/updated back then?

Read-only: reads the production parquet files, writes only
    Entsoe/Outages/Data/_probe_versions/versions_<timestamp>.csv
    Entsoe/Outages/Data/_probe_versions/report_<timestamp>.txt

Run from Master_Thesis with .venv active (~40-80 requests, 1-2 min):
    python Entsoe/Outages/probe_outage_versions.py
    python Entsoe/Outages/probe_outage_versions.py --per-file 3   # fewer docs
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import entsoe_outages_probe as P  # noqa: E402  (request machinery + parser)
from entsoe import EntsoePandasClient  # noqa: E402

PROD = HERE / "Data" / "production" / "gen_unit_outages"
OUT = HERE / "Data" / "_probe_versions"
SAMPLE = [  # (variant folder, businessType, area, year)
    ("planned_A53", "A53", "CH", 2022),
    ("forced_A54", "A54", "CH", 2022),
    ("planned_A53", "A53", "DE_LU", 2022),
    ("planned_A53", "A53", "CH", 2024),
    ("planned_A53", "A53", "DE_LU", 2024),
]
LATE = pd.Timestamp("2025-08-01", tz="UTC")   # 're-stamped' if created after this


def pick_docs(variant: str, area: str, year: int, n: int) -> pd.DataFrame:
    f = PROD / variant / area / f"{year}.parquet"
    if not f.exists():
        print(f"  missing {f.relative_to(HERE)} - skipped")
        return pd.DataFrame()
    d = pd.read_parquet(f, columns=["doc_mrid", "revision", "created",
                                    "unavail_start", "unavail_end"])
    d = d.drop_duplicates("doc_mrid")
    d = d[pd.to_datetime(d["unavail_start"], utc=True).dt.year == year]
    # highest revisions first: these had the most chances to leave old versions
    return d.sort_values("revision", ascending=False).head(n)


def run_query(client, params: dict, start: pd.Timestamp, end: pd.Timestamp
              ) -> tuple[pd.DataFrame, str]:
    try:
        docs = P._fetch_window(client, params, start, end, P.PAGE_SIZE_DEFAULT)
    except Exception as exc:  # noqa: BLE001 - probe: record and go on
        return pd.DataFrame(), f"error: {P.redact(str(exc))[:150]}"
    if not docs:
        return pd.DataFrame(), "no data"
    df = P.parse_outage_documents(docs)
    if df.empty:
        return df, "no data"
    keep = [c for c in ["doc_mrid", "revision", "created", "docstatus"] if c in df]
    return df[keep].drop_duplicates(), "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--per-file", type=int, default=5, help="docs per sample file")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    api_key = P.load_api_key()
    client = EntsoePandasClient(api_key=api_key, timeout=P.REQUEST_TIMEOUT_S)
    P.set_request_throttle(P.Throttle(args.interval))
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant, btype, area, year in SAMPLE:
        print(f"\n== {variant} {area} {year}")
        sample = pick_docs(variant, area, year, args.per_file)
        for _, s in sample.iterrows():
            u0 = pd.to_datetime(s["unavail_start"], utc=True)
            u1 = pd.to_datetime(s["unavail_end"], utc=True)
            win0 = (u0 - pd.Timedelta(days=1)).tz_convert("Europe/Brussels")
            win1 = min(u1 + pd.Timedelta(days=1), u0 + pd.Timedelta(days=360)
                       ).tz_convert("Europe/Brussels")
            base = {"documentType": "A80", "biddingZone_Domain": P._eic(area),
                    "businessType": btype}

            # A: by mRID
            a, a_status = run_query(client, {**base, "mRID": s["doc_mrid"]}, win0, win1)
            # B: by update window = the month before the outage starts
            up0 = (u0 - pd.DateOffset(months=1)).strftime("%Y%m%d%H00")
            up1 = u0.strftime("%Y%m%d%H00")
            b, b_status = run_query(client, {**base, "PeriodStartUpdate": up0,
                                             "PeriodEndUpdate": up1}, win0, win1)
            b_same = b[b["doc_mrid"] == s["doc_mrid"]] if not b.empty else b

            def summ(df):
                if df.empty:
                    return 0, None, None, ""
                c = pd.to_datetime(df["created"], utc=True)
                return (len(df), c.min(), c.max(),
                        ",".join(sorted(df["revision"].astype(str).unique())))

            na, amin, amax, arevs = summ(a)
            nb, bmin, bmax, _ = summ(b)
            nbs, bsmin, _, bsrevs = summ(b_same)
            rows.append({
                "area": area, "variant": variant, "year": year,
                "doc_mrid": s["doc_mrid"], "pull_revision": s["revision"],
                "pull_created": s["created"], "unavail_start": u0,
                "A_status": a_status, "A_versions": na, "A_revisions": arevs,
                "A_created_min": amin, "A_created_max": amax,
                "B_status": b_status, "B_docs_in_update_window": nb,
                "B_created_min": bmin, "B_created_max": bmax,
                "B_same_doc_versions": nbs, "B_same_doc_revisions": bsrevs,
                "B_same_doc_created_min": bsmin,
            })
            print(f"  {s['doc_mrid']} rev {s['revision']}: A {a_status} "
                  f"{na} version(s) [{arevs}] created {amin} .. {amax} | "
                  f"B {b_status} {nb} doc(s), created min {bmin}")

    res = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    res.to_csv(OUT / f"versions_{stamp}.csv", index=False)

    lines = [f"Outage version probe {stamp}", ""]
    if res.empty:
        lines.append("No documents sampled - check the production paths.")
    else:
        multi = (res["A_versions"] > 1).sum()
        early_a = (pd.to_datetime(res["A_created_min"], utc=True) < LATE).sum()
        early_b = (pd.to_datetime(res["B_created_min"], utc=True) < LATE).sum()
        lines += [
            f"documents probed: {len(res)}",
            f"A (by mRID): {multi} returned more than one version; "
            f"{early_a} have a version created before {LATE.date()}",
            f"B (update window = month before start): {early_b} windows returned "
            f"documents created before {LATE.date()}; "
            f"{(res['B_same_doc_versions'] > 0).sum()} returned the probed document",
            "",
            "Verdict:",
        ]
        if early_a or early_b:
            lines.append("  EARLIER VERSIONS EXIST -> worth a targeted re-pull of "
                         "2021-2025 outage history (send this report to Claude).")
        else:
            lines.append("  NO earlier versions available -> keep the limitation "
                         "(thesis 3.9); ex-ante outages only 2016-2020 and from "
                         "autumn 2025. Item closed.")
    (OUT / f"report_{stamp}.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwritten: {OUT.relative_to(HERE.parent.parent)}/versions_{stamp}.csv, report_{stamp}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
