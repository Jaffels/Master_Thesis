#!/usr/bin/env python3
"""
regelleistung.net Datacenter — coverage / schema probe (+ shared request layer).

Source: the public download API behind https://www.regelleistung.net/apps/datacenter/tenders/
    GET https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/<report>
        ?date=YYYY-MM-DD&exportFormat=xlsx&market=CAPACITY|ENERGY&productTypes=FCR|aFRR|mFRR
        [&countryCodeA2=DE]            (anonymous results only)
    <report> = resultsoverview   tender results (prices, demand, per country)
               anonymousresults  anonymous bid list (awarded bids)
               demands           tendered demand
No API key, no registration.

Why: ENTSO-E has no DE balancing data before 2021 and is late/patchy after
(workflow log, Balancing next step 4). regelleistung.net is the primary source
for German (and FCR-cooperation) capacity prices 2015+.

Run from the thesis root (venv active):
    python Regelleistung/regelleistung_probe.py                         # results, all products, 2015-2026
    python Regelleistung/regelleistung_probe.py --only results,demands --years 2015 2019 2025
    python Regelleistung/regelleistung_probe.py --only anonymous:CAPACITY:aFRR --dates 2024-03-01

The probe requests a few sample dates per year and task, and writes
    Regelleistung/Data/_probe/probe_<run>.csv       one row per request: status, rows, columns
    Regelleistung/Data/_probe/samples/...           the raw sample files (open them in Excel)
It prints, per task, the years with data and every distinct column schema.
Run it BEFORE the pull: the schema lines show which columns the files carry.

This module also owns ALL request logic (session, throttle, timeout, retry,
file sniffing); regelleistung_pull.py imports it and contains no fetch logic.

SAFETY: writes only under Regelleistung/Data/_probe/, never deletes anything.
Needs: requests, pandas, openpyxl
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders"
REPORTS = {                      # short name -> API path
    "results": "resultsoverview",
    "anonymous": "anonymousresults",
    "demands": "demands",
}
MARKETS = ("CAPACITY", "ENERGY")
PRODUCTS = ("FCR", "aFRR", "mFRR")
DEFAULT_COUNTRY = "DE"           # countryCodeA2 for anonymous results

REQUEST_TIMEOUT_S = 60           # hard timeout per request (lesson P4)
MAX_ATTEMPTS = 4
BACKOFF_S = (5, 20, 60)          # waits before attempt 2, 3, 4
RETRY_STATUS = {429, 500, 502, 503, 504}
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; master thesis data download)",
           "Accept": "*/*"}
NULL_TOKENS = {"", "-", "n/a", "N/A", "n.a.", "NaN", "nan", "None"}


# ─── Tasks ──────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class Task:
    report: str      # results | anonymous | demands
    market: str      # CAPACITY | ENERGY
    product: str     # FCR | aFRR | mFRR

    @property
    def key(self) -> str:
        return f"{self.report}:{self.market}:{self.product}"

    @property
    def folder(self) -> str:                     # results_CAPACITY/aFRR
        return f"{self.report}_{self.market}/{self.product}"

    def params(self, day: dt.date, country: str = DEFAULT_COUNTRY) -> dict:
        p = {"date": day.isoformat(), "exportFormat": "xlsx",
             "market": self.market, "productTypes": self.product}
        if self.report == "anonymous":
            p["countryCodeA2"] = country
        return p

    def url(self) -> str:
        return f"{BASE_URL}/{REPORTS[self.report]}"


def all_tasks() -> list[Task]:
    out = []
    for r in REPORTS:
        for m in MARKETS:
            for p in PRODUCTS:
                if m == "ENERGY" and (p == "FCR" or r == "demands"):
                    continue                     # FCR has no energy market
                out.append(Task(r, m, p))
    return out


DEFAULT_TASKS = [Task("results", "CAPACITY", p) for p in PRODUCTS]


def select_tasks(only: str) -> list[Task]:
    """--only accepts comma-separated filters; each is report[:market[:product]]
    or a bare product/market name. Empty -> DEFAULT_TASKS (capacity results)."""
    if not only.strip():
        return list(DEFAULT_TASKS)
    chosen: list[Task] = []
    for f in [x.strip() for x in only.split(",") if x.strip()]:
        parts = f.split(":")
        for t in all_tasks():
            fields = (t.report, t.market, t.product)
            if len(parts) == 1:                  # 'results', 'ENERGY', 'aFRR', ...
                ok = parts[0].lower() in {x.lower() for x in fields}
            else:
                ok = all(a.lower() == b.lower() for a, b in zip(parts, fields))
            if ok and t not in chosen:
                chosen.append(t)
    if not chosen:
        raise SystemExit(f"--only {only!r} matches no task. Valid: "
                         + ", ".join(t.key for t in all_tasks()))
    return chosen


# ─── Request layer ──────────────────────────────────────────────────────────
class Throttle:
    def __init__(self, interval: float):
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        d = self.interval - (time.monotonic() - self._last)
        if d > 0:
            time.sleep(d)
        self._last = time.monotonic()


@dataclasses.dataclass
class Fetch:
    status: str                 # ok | empty | error
    http: int | None
    content: bytes = b""
    kind: str = ""              # xlsx | csv | html | json | other
    frame: pd.DataFrame | None = None
    sheet: str = ""
    error: str = ""
    attempts: int = 1

    @property
    def rows(self) -> int:
        return 0 if self.frame is None else len(self.frame)


def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def sniff(content: bytes) -> str:
    head = content[:512].lstrip()
    if content[:2] == b"PK":
        return "xlsx"
    if head[:1] in (b"{", b"["):
        return "json"
    if head[:1] == b"<":
        return "html"
    if b";" in head or b"," in head:
        return "csv"
    return "other"


def read_table(content: bytes, kind: str) -> tuple[pd.DataFrame, str]:
    """Return (largest sheet as strings-preserving frame, sheet name)."""
    if kind == "xlsx":
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, engine="openpyxl")
        if not sheets:
            return pd.DataFrame(), ""
        name = max(sheets, key=lambda k: (len(sheets[k]), len(sheets[k].columns)))
        return sheets[name], name
    if kind == "csv":
        txt = content.decode("utf-8-sig", errors="replace")
        sep = ";" if txt.split("\n", 1)[0].count(";") > txt.split("\n", 1)[0].count(",") else ","
        return pd.read_csv(io.StringIO(txt), sep=sep), ""
    raise ValueError(f"not a table: {kind}")


def _short(msg: str, n: int = 300) -> str:
    return " ".join(str(msg).split())[:n]


def fetch(session: requests.Session, throttle: Throttle, task: Task, day: dt.date,
          country: str = DEFAULT_COUNTRY) -> Fetch:
    """One (task, day) request with timeout + retry. Retries are decided from the
    HTTP status code (429/5xx) or from the absence of a response (timeout,
    dropped connection) — never from text in the message (lesson P7)."""
    last: Fetch | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        throttle.wait()
        try:
            r = session.get(task.url(), params=task.params(day, country),
                            timeout=REQUEST_TIMEOUT_S)
        except (requests.Timeout, requests.ConnectionError) as e:
            last = Fetch("error", None, error=f"{type(e).__name__}: {_short(e)}",
                         attempts=attempt)
        else:
            code = r.status_code
            if code in RETRY_STATUS:
                last = Fetch("error", code, error=f"HTTP {code}: {_short(r.text)}",
                             attempts=attempt)
                if code == 429:
                    time.sleep(120)              # be polite on rate limit
            elif code in (204, 404):
                return Fetch("empty", code, error=f"HTTP {code}", attempts=attempt)
            elif code != 200:
                # 400 etc.: not retried; recorded as error so a re-run tries again
                return Fetch("error", code, error=f"HTTP {code}: {_short(r.text)}",
                             attempts=attempt)
            else:
                content = r.content
                kind = sniff(content)
                if not content:
                    return Fetch("empty", code, kind="", attempts=attempt)
                if kind in ("xlsx", "csv"):
                    try:
                        frame, sheet = read_table(content, kind)
                    except Exception as e:  # noqa: BLE001 — corrupt file = error
                        return Fetch("error", code, content, kind,
                                     error=f"parse: {type(e).__name__}: {_short(e)}",
                                     attempts=attempt)
                    frame = frame.dropna(how="all")
                    status = "ok" if len(frame) else "empty"
                    return Fetch(status, code, content, kind, frame, sheet, attempts=attempt)
                # JSON/HTML with 200 = usually a "no data" or error message
                return Fetch("error", code, content, kind,
                             error=f"unexpected {kind}: {_short(content[:400].decode('utf-8', 'replace'))}",
                             attempts=attempt)
        if attempt < MAX_ATTEMPTS:
            time.sleep(BACKOFF_S[attempt - 1])
    assert last is not None
    return last


def schema_hash(columns) -> str:
    return hashlib.md5("|".join(map(str, columns)).encode()).hexdigest()[:8]


# ─── Probe ──────────────────────────────────────────────────────────────────
def sample_dates(years: list[int], dates: list[str]) -> list[dt.date]:
    if dates:
        return [dt.date.fromisoformat(d) for d in dates]
    today = dt.date.today()
    out = []
    for y in years:
        for m, d in ((1, 15), (7, 15)):         # mid-month, winter + summer
            day = dt.date(y, m, d)
            if day < today:
                out.append(day)
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="",
                    help="tasks, e.g. 'results' | 'results,demands' | 'anonymous:CAPACITY:aFRR' "
                         "| 'ENERGY' | 'all' (default: capacity results FCR/aFRR/mFRR)")
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2015, 2027)))
    ap.add_argument("--dates", nargs="*", default=[], help="explicit YYYY-MM-DD (overrides --years)")
    ap.add_argument("--country", default=DEFAULT_COUNTRY, help="countryCodeA2 for anonymous results")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--data", default=str(here / "Data"))
    ap.add_argument("--no-samples", action="store_true", help="do not save raw sample files")
    args = ap.parse_args()

    tasks = all_tasks() if args.only.strip().lower() == "all" else select_tasks(args.only)
    days = sample_dates(args.years, args.dates)
    out_dir = Path(args.data) / "_probe"
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    run = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"probe: {len(tasks)} task(s) x {len(days)} date(s) = {len(tasks) * len(days)} requests")

    session, throttle, rows = new_session(), Throttle(args.interval), []
    for t in tasks:
        for day in days:
            f = fetch(session, throttle, t, day, args.country)
            cols = [] if f.frame is None else [str(c) for c in f.frame.columns]
            first = {} if f.frame is None or not len(f.frame) else f.frame.iloc[0].astype(str).to_dict()
            if f.content and not args.no_samples:
                ext = {"xlsx": "xlsx", "csv": "csv", "json": "json", "html": "html"}.get(f.kind, "bin")
                p = out_dir / "samples" / t.folder / f"{day.isoformat()}.{ext}"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(f.content)
            rows.append({"task": t.key, "date": day.isoformat(), "status": f.status,
                         "http": f.http, "kind": f.kind, "bytes": len(f.content),
                         "rows": f.rows, "n_cols": len(cols), "schema": schema_hash(cols) if cols else "",
                         "sheet": f.sheet, "columns": "|".join(cols),
                         "first_row": _short(first, 400) if first else "",
                         "error": f.error, "attempts": f.attempts})
            print(f"  {t.key:<28} {day}  {f.status:<5} http={f.http} rows={f.rows:<5} "
                  f"cols={len(cols):<3} {f.error[:80]}")

    df = pd.DataFrame(rows)
    csv = out_dir / f"probe_{run}.csv"
    df.to_csv(csv, index=False)

    print("\n=== Summary per task ===")
    for key, g in df.groupby("task", sort=False):
        ok_years = sorted({d[:4] for d in g.loc[g.status == "ok", "date"]})
        err = g[g.status == "error"]
        print(f"\n{key}: data in {', '.join(ok_years) or 'NO sample'}; "
              f"{(g.status == 'empty').sum()} empty, {len(err)} error")
        if len(err):
            print(f"  first error: {err.iloc[0]['date']} {err.iloc[0]['error'][:200]}")
        for h, s in g[g.status == "ok"].groupby("schema", sort=False):
            yrs = sorted({d[:4] for d in s["date"]})
            print(f"  schema {h} ({yrs[0]}-{yrs[-1]}, {int(s['rows'].median())} rows/day): "
                  f"{s.iloc[0]['columns'][:600]}")
    print(f"\nwritten: {csv}")
    print(f"samples: {out_dir / 'samples'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
