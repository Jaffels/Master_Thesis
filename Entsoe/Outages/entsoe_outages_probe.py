"""
ENTSO-E Outages probe — one month, thesis areas, coverage report.

Purpose: find out which Outages endpoints actually return data for the thesis-
relevant areas (CH and its four bordering bidding zones DE_LU / FR / IT_NORD /
AT) before committing to a multi-year download. Structurally identical to
entsoe_load_probe.py so entsoe_outages_pull.py can drive off the same coverage
CSV and reuse this module's request machinery.

Covers the Outages-domain articles requested:
    15.1.A  Planned unavailability of generation units   A80 / businessType A53
    15.1.B  Forced unavailability of generation units    A80 / businessType A54
    15.1.C  Planned unavailability of production units   A77 / businessType A53
    15.1.D  Forced unavailability of production units    A77 / businessType A54
    10.1.A  Planned unavailability in transmission grid  A78 / businessType A53
    10.1.B  Forced unavailability in transmission grid   A78 / businessType A54
    10.1.C  Unavailability of offshore grid              A79   (DE_LU only)
    IF aFRR 3.10  Fall-backs, aFRR platform              A53 / processType A51
    IF mFRR 3.11  Fall-backs, mFRR platform              A53 / processType A47
    IFs IN 7.2    Fall-backs, imbalance netting          A53 / processType A63
      each fall-back is split by the (mandatory) businessType:
        C47 disconnection, A53 planned maintenance, A54 unplanned outage,
        A83 auction cancellation

Why one raw series per article (not the "A&B" / "A&B&C&D" combined TP views):
    Same RAW-pull / reconcile-later choice as the Load scripts. Planned and
    forced are fetched as separate variants via the businessType filter, so each
    TP article maps to exactly one series and no document is fetched twice.

Why a custom request + parser layer instead of entsoe-py's unavailability
methods:
  * entsoe-py has no method at all for the fall-backs (docType A53).
  * Its unavailability parsers look domain codes up in a fixed dict and raise
    KeyError on any EIC it does not know (whole pull dies), drop the outage
    reason code/text, drop the asset list of transmission outages, and the
    offshore (A79) method is neither paginated nor window-limited.
  * documents_limited() stops silently after offset 4800 — a window with more
    than 5000 documents would be truncated without any warning.
  So every Outages method here goes through one path: monthly windows ->
  offset pagination (200 docs/page, 100 for fall-backs) -> if the offset
  ceiling is hit or the API reports a pagination error, the window is halved
  and re-queried -> one generic lxml parser that works for every Outages
  document type (and for ZIP as well as plain-XML responses).

Output schema (long format, one row per availability Point):
    doc_mrid, revision, created, docstatus, businesstype, unavail_start,
    unavail_end, period, start, end, resolution, position, quantity
    + every other document / TimeSeries leaf flattened as doc.<path> /
      ts.<path> (asset EICs, names, psrType, nominalP, Reason.code/text, ...).
    Repeated elements (e.g. several assets on one transmission outage) are
    joined with " | ". The index is a plain RangeIndex: outage documents are
    events, not a regular time series.

Outage documents are returned when their unavailability period OVERLAPS the
query window, so a long outage appears in several windows. Duplicates are
dropped within one call; across per-year files (pull) an outage spanning a
year boundary appears in both — dedupe on (doc_mrid, revision) downstream.

Transmission (10.1.A&B) is queried per directed border (out_Domain -> in_Domain)
plus CH internal (CH -> CH). The same asset outage may be published under both
directions; that is kept raw and deduped downstream on doc_mrid.

Usage (macOS), run from the thesis root:
    .venv/bin/python Entsoe/Outages/entsoe_outages_probe.py
    .venv/bin/python Entsoe/Outages/entsoe_outages_probe.py --retry-errors
    .venv/bin/python Entsoe/Outages/entsoe_outages_probe.py --only fallback
    .venv/bin/python Entsoe/Outages/entsoe_outages_probe.py \\
        --start 2025-06-01 --end 2025-07-01

Re-runs are cheap: every attempted (dataset, variant, area) is recorded in
Data/_probe_state.json and skipped next run. --force ignores the state file;
--retry-errors re-attempts only the combinations that failed. Pin the area code
that answered in PINNED_AREA_CODE before the production pull.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import etree
from entsoe import EntsoePandasClient
from entsoe.mappings import lookup_area

try:  # entsoe-py raises these for "no data" rather than returning empty
    from entsoe.exceptions import (
        NoMatchingDataError,
        InvalidBusinessParameterError,
        InvalidPSRTypeError,
        PaginationError,
    )
except ImportError:  # pragma: no cover - older entsoe-py
    class NoMatchingDataError(Exception): ...
    class InvalidBusinessParameterError(Exception): ...
    class InvalidPSRTypeError(Exception): ...
    class PaginationError(Exception): ...

NO_DATA_ERRORS = (
    NoMatchingDataError,
    InvalidBusinessParameterError,
    InvalidPSRTypeError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("outages-probe")

TZ = "Europe/Zurich"

# ---------------------------------------------------------------------------
# Areas
# ---------------------------------------------------------------------------
# Zonal datasets (15.1.x, fall-backs). Codes are tried in order; the next one
# only if the previous returned nothing.
AREAS: dict[str, list[str]] = {
    "CH": ["CH"],
    "DE_LU": ["DE_LU", "DE_AMPRION", "DE_TENNET", "DE_50HZ", "DE_TRANSNET"],
    "FR": ["FR"],
    "IT_NORD": ["IT_NORD", "IT"],
    "AT": ["AT"],
}

# Fall-backs are published per CTA / LFA / region, not per bidding zone, so the
# control-area level is tried first where it differs from the bidding zone.
FALLBACK_AREAS: dict[str, list[str]] = {
    "CH": ["CH"],
    "DE_LU": ["DE_LU", "DE", "DE_AMPRION", "DE_TENNET", "DE_50HZ", "DE_TRANSNET"],
    "FR": ["FR"],
    "IT_NORD": ["IT", "IT_NORD"],
    "AT": ["AT"],
}

# 10.1.C offshore grid: only relevant for the German North/Baltic Sea
# connections. Each German level is its own label so a bidding-zone answer can
# never hide a control-area one (TenneT and 50Hertz both operate offshore
# links) — dedupe on doc_mrid downstream.
OFFSHORE_AREAS: dict[str, list[str]] = {
    "DE_LU": ["DE_LU"],
    "DE_TENNET": ["DE_TENNET"],
    "DE_50HZ": ["DE_50HZ"],
    "DE_AMPRION": ["DE_AMPRION"],
}

# 10.1.A&B: directed borders "out>in" + CH internal grid. Label is filesystem-
# safe; the code string is what the method receives.
def _border_grid() -> dict[str, list[str]]:
    neighbours = {"DE_LU": ["DE_LU", "DE"], "FR": ["FR"],
                  "IT_NORD": ["IT_NORD", "IT"], "AT": ["AT"]}
    grid: dict[str, list[str]] = {"CH_to_CH": ["CH>CH"]}
    for nb, codes in neighbours.items():
        grid[f"CH_to_{nb}"] = [f"CH>{c}" for c in codes]
        grid[f"{nb}_to_CH"] = [f"{c}>CH" for c in codes]
    return grid


BORDERS = _border_grid()

# (dataset, area_label) -> the one code to use. Fill from the probe's
# "Pin these" hint before the production pull.
PINNED_AREA_CODE: dict[tuple[str, str], str] = {}

# ENTSO-E code reminders (Outages domain)
#   documentType: A77 production units  A78 transmission  A79 offshore grid
#                 A80 generation units  A53 outage publication (fall-backs)
#   businessType: A53 planned  A54 forced/unplanned  C47 disconnection
#                 A83 auction cancellation
#   processType (fall-backs): A51 aFRR  A47 mFRR  A63 imbalance netting
#   docStatus:    A05 active  A09 cancelled  A13 withdrawn (not returned by
#                 default)

PAGE_SIZE_DEFAULT = 200
PAGE_SIZE_FALLBACK = 100
MAX_OFFSET = 4800                    # API ceiling for the offset parameter
MIN_WINDOW = pd.Timedelta(hours=6)   # stop halving below this


# ---------------------------------------------------------------------------
# Throttle / errors / key
# ---------------------------------------------------------------------------
class Throttle:
    """Keep a minimum gap between API calls. ENTSO-E allows 400 req/min;
    we stay far below that so the probe never trips the limiter."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


# Outage methods paginate internally, so the throttle has to sit on every
# individual HTTP request, not on the method call. Set via set_request_throttle.
_REQ_THROTTLE: Throttle | None = None


def set_request_throttle(throttle: Throttle) -> None:
    global _REQ_THROTTLE
    _REQ_THROTTLE = throttle


class OutageApiError(Exception):
    """The API answered with an Acknowledgement document that is not 'no data'."""


def redact(text: str) -> str:
    """Never let the API token reach a log file, a terminal or a paste."""
    return re.sub(r"(securityToken=)[^&\s]*", r"\1REDACTED", str(text))


def describe_error(exc: Exception) -> str:
    body = getattr(getattr(exc, "response", None), "text", "") or ""
    body = " ".join(body.split())[:300]
    msg = f"{type(exc).__name__}: {exc}"
    if body:
        msg += f" | body: {body}"
    return redact(msg)[:400]


def _is_transient(msg: str) -> bool:
    return any(t in msg for t in ("429", "500", "502", "503", "504",
                                  "timeout", "Timeout", "Connection"))


def load_api_key() -> str:
    """ENTSOE_API_KEY from the environment, else from the nearest .env found by
    walking up from this file (then ENTSOE-New/.env)."""
    key = os.getenv("ENTSOE_API_KEY")
    if key:
        return key
    here = Path(__file__).resolve().parent
    candidates = [p / ".env" for p in [here, *here.parents][:5]]
    candidates += [p / "ENTSOE-New" / ".env" for p in [here, *here.parents][:5]]
    for env_file in candidates:
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ENTSOE_API_KEY") and "=" in line:
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    log.info("using ENTSOE_API_KEY from %s", env_file)
                    return value
    return ""


# ---------------------------------------------------------------------------
# Request layer: throttled GET -> pagination -> window halving -> months
# ---------------------------------------------------------------------------
def _eic(code: str) -> str:
    """entsoe-py area name ('CH', 'DE_LU') or a raw 16-char EIC."""
    try:
        return lookup_area(code).code
    except ValueError:
        if len(code) == 16:
            return code
        raise


def _ack_text(content: bytes) -> str | None:
    """Reason text of an Acknowledgement document, None for anything else."""
    if content[:2] == b"PK" or b"Acknowledgement_MarketDocument" not in content[:600]:
        return None
    try:
        root = etree.fromstring(content)
        texts = [e.text or "" for e in root.iter() if _local(e.tag) == "text"]
        return " ".join(texts).strip() or "acknowledgement without text"
    except etree.XMLSyntaxError:
        return "unparseable acknowledgement"


def _raw_get(client: EntsoePandasClient, params: dict[str, Any],
             start: pd.Timestamp, end: pd.Timestamp, attempts: int = 3) -> bytes:
    """One HTTP request, throttled, with backoff on transient failures.
    'No data' and pagination errors are answers, not failures: re-raised."""
    for i in range(attempts):
        if _REQ_THROTTLE is not None:
            _REQ_THROTTLE.wait()
        try:
            # _base_request mutates params — always hand it a fresh copy
            content = client._base_request(params=dict(params),
                                           start=start, end=end).content
        except (*NO_DATA_ERRORS, PaginationError):
            raise
        except Exception as exc:
            if not _is_transient(str(exc)) or i == attempts - 1:
                raise
            backoff = 5 * (2 ** i)
            log.warning("transient error (%s) — retrying in %ss",
                        redact(str(exc))[:80], backoff)
            time.sleep(backoff)
            continue
        ack = _ack_text(content)
        if ack is None:
            return content
        if "No matching data" in ack:
            raise NoMatchingDataError
        if "exceeds" in ack and "limit" in ack:
            raise PaginationError(ack)
        raise OutageApiError(ack[:300])
    raise RuntimeError("unreachable")


def _split_documents(content: bytes) -> list[bytes]:
    """ZIP of one-XML-per-outage, or a single plain XML document."""
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as arc:
            return [arc.read(f) for f in arc.infolist()
                    if f.filename.lower().endswith(".xml")]
    return [content]


def _page_count(content: bytes, docs: list[bytes]) -> int:
    """How many paginated items this page held: files in a ZIP, else
    TimeSeries in the plain XML (conservative — only decides whether to ask
    for the next offset)."""
    if content[:2] == b"PK":
        return len(docs)
    return content.count(b"<TimeSeries>") + content.count(b"<TimeSeries ")


def _fetch_window(client: EntsoePandasClient, params: dict[str, Any],
                  start: pd.Timestamp, end: pd.Timestamp,
                  page: int) -> list[bytes]:
    """All documents for one window. Offset-paginates; if the window holds more
    than the API's offset ceiling allows, halves it and recurses."""
    docs: list[bytes] = []
    offset, prev = 0, None
    try:
        while True:
            try:
                content = _raw_get(client, {**params, "offset": offset},
                                   start, end)
            except NoMatchingDataError:
                break
            if content == prev:          # API ignored the offset — stop
                break
            prev = content
            batch = _split_documents(content)
            docs.extend(batch)
            if _page_count(content, batch) < page:
                break
            offset += page
            if offset > MAX_OFFSET:
                raise PaginationError("offset ceiling reached")
    except PaginationError:
        if end - start <= MIN_WINDOW:
            raise
        mid = start + (end - start) / 2
        log.info("    window %s..%s too dense — halving", start, end)
        return (_fetch_window(client, params, start, mid, page)
                + _fetch_window(client, params, mid, end, page))
    return docs


def _month_windows(start: pd.Timestamp, end: pd.Timestamp
                   ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    edges = [start]
    m = (start.normalize().replace(day=1) + pd.DateOffset(months=1))
    while m < end:
        edges.append(m)
        m = m + pd.DateOffset(months=1)
    edges.append(end)
    return list(zip(edges[:-1], edges[1:]))


def fetch_outage_documents(client: EntsoePandasClient, params: dict[str, Any],
                           start: pd.Timestamp, end: pd.Timestamp,
                           page: int = PAGE_SIZE_DEFAULT) -> list[bytes]:
    docs: list[bytes] = []
    for w_start, w_end in _month_windows(start, end):
        docs.extend(_fetch_window(client, params, w_start, w_end, page))
    if not docs:
        raise NoMatchingDataError
    return docs


# ---------------------------------------------------------------------------
# Generic parser for every Outages document type
# ---------------------------------------------------------------------------
_XML_PARSER = etree.XMLParser(recover=True, huge_tree=True)

CORE_RENAME = {
    "doc.mrid": "doc_mrid",
    "doc.revisionnumber": "revision",
    "doc.createddatetime": "created",
    "doc.docstatus.value": "docstatus",
    "doc.unavailability_time_period.timeinterval.start": "unavail_start",
    "doc.unavailability_time_period.timeinterval.end": "unavail_end",
    "ts.businesstype": "businesstype",
}
DATETIME_COLS = ["created", "unavail_start", "unavail_end", "start", "end"]


def _local(tag: Any) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _is_period(el: etree._Element) -> bool:
    kids = {_local(c.tag) for c in el}
    return "Point" in kids or {"timeInterval", "resolution"} <= kids


def _put(out: dict[str, str], key: str, value: str) -> None:
    if not value:
        return
    if key in out and out[key] != value:
        out[key] = f"{out[key]} | {value}"
    else:
        out[key] = value


def _flatten(el: etree._Element, prefix: str, out: dict[str, str],
             periods: list[etree._Element]) -> None:
    """Leaf text under `el` into out[prefix.path]; period blocks collected."""
    for c in el:
        name = _local(c.tag)
        if not name:
            continue  # comments / processing instructions
        if _is_period(c):
            periods.append(c)
            continue
        key = f"{prefix}.{name.lower()}"
        if len(c):
            _flatten(c, key, out, periods)
        else:
            _put(out, key, (c.text or "").strip())


def _period_rows(period: etree._Element) -> list[dict[str, Any]]:
    kids = {_local(c.tag): c for c in period}
    ti = kids.get("timeInterval")
    p_start = p_end = None
    if ti is not None:
        for c in ti:
            if _local(c.tag) == "start":
                p_start = pd.Timestamp(c.text)
            elif _local(c.tag) == "end":
                p_end = pd.Timestamp(c.text)
    res = (kids["resolution"].text or "").strip() if "resolution" in kids else ""
    try:
        step = pd.Timedelta(res) if res else None
    except ValueError:
        step = None
    pts = [c for c in period if _local(c.tag) == "Point"]
    name = _local(period.tag).lower()
    if not pts:
        return [{"period": name, "start": p_start, "end": p_end,
                 "resolution": res}]
    parsed = []
    for p in pts:
        d: dict[str, Any] = {}
        for c in p:
            n = _local(c.tag)
            if n and not len(c):
                d[n.lower()] = (c.text or "").strip()
        parsed.append(d)
    rows = []
    for i, d in enumerate(parsed):
        pos = int(d.get("position", i + 1))
        start = (p_start + step * (pos - 1)
                 if (p_start is not None and step is not None) else p_start)
        if i + 1 < len(parsed) and p_start is not None and step is not None:
            end = p_start + step * (int(parsed[i + 1].get("position", i + 2)) - 1)
        else:
            end = p_end
        row = {"period": name, "start": start, "end": end,
               "resolution": res, "position": pos,
               "quantity": d.get("quantity")}
        for k, v in d.items():
            if k not in ("position", "quantity"):
                row[f"point.{k}"] = v
        rows.append(row)
    return rows


def parse_outage_documents(docs: list[bytes]) -> pd.DataFrame:
    """One row per availability Point (or per TimeSeries if it has none),
    carrying every document- and TimeSeries-level field."""
    records: list[dict[str, Any]] = []
    for raw in docs:
        root = etree.fromstring(raw, _XML_PARSER)
        if root is None:
            continue
        if "Acknowledgement" in _local(root.tag):
            continue  # stray ack inside a ZIP carries no outage
        doc: dict[str, str] = {"doc.root": _local(root.tag)}
        doc_periods: list[etree._Element] = []
        series = []
        for c in root:
            name = _local(c.tag)
            if name == "TimeSeries":
                series.append(c)
            elif name:
                if len(c):
                    _flatten(c, f"doc.{name.lower()}", doc, doc_periods)
                else:
                    _put(doc, f"doc.{name.lower()}", (c.text or "").strip())
        if not series:
            records.append(dict(doc))
            continue
        for ts in series:
            ts_fields: dict[str, str] = {}
            periods: list[etree._Element] = []
            _flatten(ts, "ts", ts_fields, periods)
            base = {**doc, **ts_fields}
            prows = [r for p in periods for r in _period_rows(p)]
            if not prows:
                records.append(base)
            else:
                records.extend({**base, **r} for r in prows)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df.rename(columns=CORE_RENAME)
    lead = [c for c in ["doc_mrid", "revision", "created", "docstatus",
                        "businesstype", "unavail_start", "unavail_end",
                        "period", "start", "end", "resolution", "position",
                        "quantity"] if c in df.columns]
    df = df[lead + sorted(c for c in df.columns if c not in lead)]
    return df


def tidy(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Typed, parquet-safe outage frame. Only known numeric / datetime columns
    are converted — IDs (mRIDs, EICs) stay strings, so no leading zero or
    numeric-looking code is ever mangled. Datetimes are converted to TZ."""
    df = df.copy()
    for c in DATETIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_convert(TZ)
    for c in ["revision", "position"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    num_like = [c for c in df.columns
                if c == "quantity" or c.endswith(".nominalp")
                or c.endswith(".quantity")]
    for c in num_like:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    df = df.drop_duplicates(ignore_index=True)
    sort_cols = [c for c in ["unavail_start", "doc_mrid", "revision", "start"]
                 if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ignore_index=True, na_position="last")
    df.index.name = "row"
    return df


def profile(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": len(df), "cols": df.shape[1]}
    out["docs"] = int(df["doc_mrid"].nunique()) if "doc_mrid" in df else ""
    s = df["unavail_start"] if "unavail_start" in df else df.get("start")
    e = df["unavail_end"] if "unavail_end" in df else df.get("end")
    out["first_ts"] = str(s.min()) if s is not None and s.notna().any() else ""
    out["last_ts"] = str(e.max()) if e is not None and e.notna().any() else ""
    out["na_share"] = (round(float(df["quantity"].isna().mean()), 3)
                       if "quantity" in df else "")
    return out


# ---------------------------------------------------------------------------
# Client methods (attached to EntsoePandasClient, called by name like native)
# ---------------------------------------------------------------------------
def _add_outage_methods() -> None:
    def zonal(doctype: str, page: int):
        def method(self, country_code: str, start: pd.Timestamp,
                   end: pd.Timestamp, business_type: str | None = None,
                   process_type: str | None = None,
                   doc_status: str | None = None) -> pd.DataFrame:
            params: dict[str, Any] = {"documentType": doctype,
                                      "biddingZone_Domain": _eic(country_code)}
            if business_type:
                params["businessType"] = business_type
            if process_type:
                params["processType"] = process_type
            if doc_status:
                params["docStatus"] = doc_status
            docs = fetch_outage_documents(self, params, start, end, page)
            df = parse_outage_documents(docs)
            if df.empty:
                raise NoMatchingDataError
            return df
        return method

    def transmission(self, country_code: str, start: pd.Timestamp,
                     end: pd.Timestamp, business_type: str | None = None,
                     doc_status: str | None = None) -> pd.DataFrame:
        """country_code is 'OUT>IN', e.g. 'CH>DE_LU' (10.1.A&B, A78)."""
        out_code, in_code = country_code.split(">")
        params: dict[str, Any] = {"documentType": "A78",
                                  "out_Domain": _eic(out_code),
                                  "in_Domain": _eic(in_code)}
        if business_type:
            params["businessType"] = business_type
        if doc_status:
            params["docStatus"] = doc_status
        docs = fetch_outage_documents(self, params, start, end, PAGE_SIZE_DEFAULT)
        df = parse_outage_documents(docs)
        if df.empty:
            raise NoMatchingDataError
        return df

    EntsoePandasClient.query_outages_generation_units = zonal("A80", PAGE_SIZE_DEFAULT)
    EntsoePandasClient.query_outages_production_units = zonal("A77", PAGE_SIZE_DEFAULT)
    EntsoePandasClient.query_outages_offshore_grid = zonal("A79", PAGE_SIZE_DEFAULT)
    EntsoePandasClient.query_outages_fallbacks = zonal("A53", PAGE_SIZE_FALLBACK)
    EntsoePandasClient.query_outages_transmission = transmission


_add_outage_methods()


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
@dataclass
class Task:
    """One endpoint + one parameter combination."""

    dataset: str
    variant: str
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    enabled: bool = True
    # area grid this task runs over (defaults to AREAS)
    areas: dict[str, list[str]] | None = None
    # event-driven series: the pull attempts the whole grid, not only probe-ok
    sparse: bool = False
    # parity with the Load/Balancing Task interface; windowing is internal
    chunk: str | None = None

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.variant}"

    @property
    def grid(self) -> dict[str, list[str]]:
        return self.areas if self.areas is not None else AREAS


def build_tasks() -> list[Task]:
    tasks: list[Task] = []
    planned_forced = [("planned_A53", "A53"), ("forced_A54", "A54")]

    # --- 15.1.A&B : generation units (>=100 MW) ----------------------------
    for (label, bt), art in zip(planned_forced, ("15.1.A", "15.1.B")):
        tasks.append(Task(
            "gen_unit_outages", label, "query_outages_generation_units",
            {"business_type": bt}, f"{art} unavailability of generation units"))

    # --- 15.1.C&D : production units (>=100 MW) ----------------------------
    for (label, bt), art in zip(planned_forced, ("15.1.C", "15.1.D")):
        tasks.append(Task(
            "prod_unit_outages", label, "query_outages_production_units",
            {"business_type": bt}, f"{art} unavailability of production units"))

    # --- 10.1.A&B : transmission grid, per directed border + CH internal ---
    for (label, bt), art in zip(planned_forced, ("10.1.A", "10.1.B")):
        tasks.append(Task(
            "transmission_outages", label, "query_outages_transmission",
            {"business_type": bt}, f"{art} unavailability in transmission grid",
            areas=BORDERS, sparse=True))

    # --- 10.1.C : offshore grid — German levels only -----------------------
    tasks.append(Task(
        "offshore_grid_outages", "all", "query_outages_offshore_grid", {},
        "10.1.C unavailability of offshore grid infrastructure (DE only)",
        areas=OFFSHORE_AREAS, sparse=True))

    # --- Fall-backs: IF aFRR 3.10, IF mFRR 3.11, IFs IN 7.2 ----------------
    fb_business = [("disconnection_C47", "C47"), ("planned_A53", "A53"),
                   ("unplanned_A54", "A54"), ("auction_cancel_A83", "A83")]
    for ds, pt, art in [("fallback_afrr", "A51", "IF aFRR 3.10"),
                        ("fallback_mfrr", "A47", "IF mFRR 3.11"),
                        ("fallback_in", "A63", "IFs IN 7.2")]:
        for label, bt in fb_business:
            tasks.append(Task(
                ds, label, "query_outages_fallbacks",
                {"process_type": pt, "business_type": bt},
                f"{art} fall-backs ({label})",
                areas=FALLBACK_AREAS, sparse=True))

    return tasks


def invoke(client: EntsoePandasClient, task: Task, code: str,
           start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Call a task's method for one area code; returns a tidy frame.
    Raises NO_DATA_ERRORS for a definitive 'no data'."""
    raw = getattr(client, task.method)(country_code=code, start=start,
                                       end=end, **task.kwargs)
    df = tidy(raw, task.dataset)
    if df.empty:
        raise NoMatchingDataError
    return df


def _log_task_summary(key: str, outcomes: list[dict[str, Any]]) -> None:
    ok = [o for o in outcomes if o["status"] == "ok"]
    empty = [o["area"] for o in outcomes if o["status"] == "empty"]
    errs = [o["area"] for o in outcomes if o["status"] == "error"]
    cached = [o["area"] for o in outcomes if str(o["status"]).startswith("cached")]
    parts = []
    if ok:
        parts.append("ok: " + " ".join(f"{o['area']}({o['docs']} docs)" for o in ok))
    if empty:
        parts.append("no data: " + ",".join(empty))
    if errs:
        parts.append("ERR: " + ",".join(errs))
    if cached:
        parts.append("cached: " + ",".join(cached))
    log.info("%-44s %s", key, "  |  ".join(parts) if parts else "(no areas)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2026-08-01", help="inclusive, YYYY-MM-DD")
    ap.add_argument("--end", default="2026-09-01", help="exclusive, YYYY-MM-DD")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "Data"),
                    help="output folder (default: Data/ next to this script)")
    ap.add_argument("--only", default="", help="substring filter on dataset name")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="minimum seconds between API requests (default 2.0)")
    ap.add_argument("--force", action="store_true",
                    help="re-attempt every combination in the state file")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-attempt only the combinations recorded as errors")
    ap.add_argument("--verbose", action="store_true",
                    help="log every area individually")
    args = ap.parse_args()

    api_key = load_api_key()
    if not api_key:
        log.error("ENTSOE_API_KEY not found — export it or put it in .env")
        return 1

    start = pd.Timestamp(args.start, tz=TZ)
    end = pd.Timestamp(args.end, tz=TZ)
    if end <= start:
        log.error("--end must be after --start")
        return 1

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "_probe_state.json"
    state: dict[str, str] = {}
    if state_path.exists() and not args.force:
        state = json.loads(state_path.read_text())

    client = EntsoePandasClient(api_key=api_key)
    set_request_throttle(Throttle(args.interval))
    tasks = [t for t in build_tasks()
             if t.enabled and args.only.lower() in t.dataset.lower()]

    stamp = f"{start:%Y%m%d}_{end:%Y%m%d}"
    n_combos = sum(len(t.grid) for t in tasks)
    log.info("probing %d datasets, %d (dataset, area) combos | %s -> %s",
             len(tasks), n_combos, start.date(), end.date())

    report: list[dict[str, Any]] = []

    for task in tasks:
        outcomes: list[dict[str, Any]] = []
        for area_label, codes in task.grid.items():
            state_key = f"{task.key}__{area_label}__{stamp}"
            prior = state.get(state_key)
            if prior and not (args.retry_errors and prior == "error"):
                outcomes.append({"area": area_label, "status": f"cached:{prior}"})
                continue

            pinned = PINNED_AREA_CODE.get((task.dataset, area_label))
            candidates = [pinned] if pinned else codes
            status, used_code, df, err = "empty", "", None, ""

            for code in candidates:
                try:
                    df = invoke(client, task, code, start, end)
                except NO_DATA_ERRORS as exc:
                    err, status, df = type(exc).__name__, "empty", None
                    continue
                except Exception as exc:
                    err, status, df = describe_error(exc), "error", None
                    break
                used_code, status, err = code, "ok", ""
                break

            row: dict[str, Any] = {
                "dataset": task.dataset, "variant": task.variant,
                "area": area_label, "area_code_used": used_code,
                "status": status, "rows": 0, "cols": 0, "docs": "",
                "first_ts": "", "last_ts": "", "na_share": "",
                "file": "", "note": task.note, "error": err,
            }

            if status == "ok" and df is not None:
                fname = f"{task.dataset}__{task.variant}__{area_label}__{stamp}.parquet"
                df.to_parquet(out_dir / fname, engine="pyarrow",
                              compression="snappy")
                row.update(profile(df))
                row["file"] = fname
                outcomes.append({"area": area_label, "status": "ok",
                                 "docs": row["docs"]})
                if args.verbose:
                    log.info("OK    %-40s %-12s -> %6d rows, %4s docs (%s)",
                             task.key, area_label, len(df), row["docs"], used_code)
            elif status == "empty":
                outcomes.append({"area": area_label, "status": "empty"})
                if args.verbose:
                    log.info("EMPTY %-40s %-12s", task.key, area_label)
            else:
                outcomes.append({"area": area_label, "status": "error"})
                log.warning("ERROR %-40s %-12s %s", task.key, area_label, err)

            report.append(row)
            state[state_key] = status
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True))

        if not args.verbose:
            _log_task_summary(task.key, outcomes)

    if not report:
        log.info("nothing to do — everything was already attempted "
                 "(use --force, or --retry-errors)")
        return 0

    rep = pd.DataFrame(report)
    rep_path = out_dir / f"_coverage_{stamp}.csv"
    if rep_path.exists():
        rep = pd.concat([pd.read_csv(rep_path), rep], ignore_index=True)
        rep = rep.drop_duplicates(subset=["dataset", "variant", "area"], keep="last")
    rep.to_csv(rep_path, index=False)

    print("\n" + "=" * 100)
    print(f"COVERAGE  {start.date()} -> {end.date()}")
    print("=" * 100)
    print(rep.pivot_table(index=["dataset", "variant"], columns="area",
                          values="status", aggfunc="first").fillna("-").to_string())

    ok = rep[rep.status == "ok"]
    print("\nUsable series (status=ok):")
    if ok.empty:
        print("  none")
    else:
        cols = ["dataset", "variant", "area", "area_code_used", "rows",
                "docs", "cols", "first_ts", "last_ts"]
        print(ok[cols].to_string(index=False))
        print("\nPin these in PINNED_AREA_CODE before the production pull:")
        default_code = {lbl: codes[0] for t in build_tasks()
                        for lbl, codes in t.grid.items()}
        for (ds, ar), code in (ok.groupby(["dataset", "area"])["area_code_used"]
                               .first().items()):
            if code != default_code.get(ar):
                print(f'    ("{ds}", "{ar}"): "{code}",')

    print("\nNote: sparse series (transmission, offshore, fall-backs) are "
          "event-driven —\n'empty' for one month is not a verdict; the pull "
          "attempts their full grid.")
    print(f"\nparquet + coverage csv written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
