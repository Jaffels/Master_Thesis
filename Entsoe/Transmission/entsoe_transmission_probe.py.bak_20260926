"""
ENTSO-E Transmission probe — one month, four Swiss borders, coverage report.

Purpose: find out which Transmission endpoints actually return data for the
thesis-relevant borders (CH against each of its four bordering bidding zones
DE_LU / FR / IT_NORD / AT, both directions) before committing to a multi-year
download. Structurally the sibling of entsoe_load_probe.py / the balancing
probe, so the pull can drive off the same coverage CSV and the same tidy() /
throttle / key-loading machinery.

Covers the Transmission-domain articles requested:
    11.1    Forecasted Transfer Capacity (NTC)   -> query_net_transfer_capacity_*
              day / week / month / year ahead      (A61, contract A01/A02/A03/A04)
    12.1.F  Scheduled Commercial Exchanges       -> query_scheduled_exchanges
              total (A05) and day-ahead (A01)
    12.1.G  Physical Flows                       -> query_crossborder_flows (A11)
    13.1.B  Countertrading                       -> query_countertrading (A91)*
    13.1.C  Costs of Congestion Management        -> query_congestion_costs (A92)*

    * 13.1.B and 13.1.C have no entsoe-py method; thin wrappers are added below
      (documentType A91 / A92), the same monkeypatch approach the load probe uses
      for its 8.1 forecast-margin wrapper. They lean on the library's own
      _base_request (so "No matching data found" -> NoMatchingDataError, i.e. a
      clean *empty*, never a fake error) and its generic timeseries parser.

Everything here is a CROSS-BORDER product: each endpoint takes a
country_code_from / country_code_to pair, not a single area. So the probe
iterates over directed borders (CH>DE_LU and DE_LU>CH, etc.) rather than over
single areas. All four Swiss borders the thesis names are covered, both
directions, because NTC / schedules / flows / countertrade are all directional.

empty vs error — the whole point of this probe for the sparse 13.1.x series:
    * "empty"  = the API definitively answered "No matching data found"
                 (NoMatchingDataError). Genuinely no data for that border/period.
    * "error"  = an HTTP 400/5xx or a parse failure. NOT the same as empty — it
                 means the request was rejected or malformed, and the coverage
                 CSV records the TP response body so the reason is visible.
    Transient failures (429/5xx/timeout) are retried before being called an
    error, and NoMatchingDataError is never retried, so an "empty" verdict is
    always a definitive answer and never a swallowed glitch. This is what lets
    the pull save an empty table as empty *and be sure it should be empty*.

Load/transmission methods are @month_limited or @year_limited in entsoe-py (a
too-long range is split internally) and return few TimeSeries per period, so the
100-TimeSeries response cap that forces adaptive chunking on the balancing side
does not arise here. No chunk logic is needed; every Task keeps chunk=None.

Writes one parquet per (dataset, variant, border) into Entsoe/Transmission/Data/
plus a coverage report. Lives in Master_Thesis/Entsoe/Transmission/ and writes
to its own Data/ subfolder.

Usage (macOS), run from the thesis root:
    # key is read from ENTSOE_API_KEY, or from the nearest .env walking up
    .venv/bin/python Entsoe/Transmission/entsoe_transmission_probe.py
    .venv/bin/python Entsoe/Transmission/entsoe_transmission_probe.py --retry-errors
    .venv/bin/python Entsoe/Transmission/entsoe_transmission_probe.py \
        --start 2021-06-01 --end 2021-07-01

Re-runs are cheap: every attempted (dataset, variant, border) is recorded in
Data/_probe_state.json and skipped next run. --force ignores the state file;
--retry-errors re-attempts only the combinations that failed. Once the probe has
told you which end-code serves a given border, pin it in PINNED_BORDER_CODE so
the production pull can never silently swap one area level for another.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from entsoe import EntsoePandasClient

try:  # entsoe-py raises these for "no data" rather than returning empty
    from entsoe.exceptions import (
        NoMatchingDataError,
        InvalidBusinessParameterError,
        InvalidPSRTypeError,
    )
except ImportError:  # pragma: no cover - older entsoe-py
    class NoMatchingDataError(Exception): ...
    class InvalidBusinessParameterError(Exception): ...
    class InvalidPSRTypeError(Exception): ...

NO_DATA_ERRORS = (
    NoMatchingDataError,
    InvalidBusinessParameterError,
    InvalidPSRTypeError,
)


def _add_congestion_methods() -> None:
    """Add query_countertrading (13.1.B, A91) and query_congestion_costs
    (13.1.C, A92) to the client — entsoe-py ships neither.

    Both are built from the raw request machinery the library already exposes:
    _base_request handles the security token, period formatting and the
    'No matching data found' -> NoMatchingDataError translation, so an empty
    period behaves like any other empty pull (a clean *empty*, never a fake
    error).

    Two quirks of these endpoints, both discovered from live 400 bodies and
    handled here:

    * 100-instance cap (A91). Countertrading is published as one TimeSeries per
      MTU, so a single month is ~1488 instances and the API rejects it with
      "The number of instances (N) exceeds the allowed maximum (100)". That exact
      wording is NOT one of the phrases entsoe-py's _base_request maps to
      PaginationError, so its built-in offset/pagination decorators never fire —
      it just re-raises a raw HTTPError. @window_on_overflow below catches that
      specific text and halves the period (down to an hour) until each sub-window
      is under the cap, then concatenates. @month_limited caps the initial window
      at a month and re-raises NoMatchingDataError only if every block is empty.

    * single-area shape (A92). Costs of congestion management are a per-area
      figure, not a border flow: the API rejects a border pair with "Area EICs
      shall be the same". So congestion_costs is queried with in_Domain ==
      out_Domain (the probe/pull iterate it over single AREAS, scope="area",
      calling with country_code_from == country_code_to); the same wrapper serves
      it because from == to makes in == out.

    Each A91/A92 TimeSeries is read with the library's own generic quantity
    parser and placed in its own column, keyed by businessType (+ flow direction
    where present) so overlapping-timestamp series from different business types
    do not collide. The value tag is auto-detected: 'quantity' (MW for
    countertrade, or the cost amount for A92 where costs are carried as
    quantities) falling back to 'price.amount'. Verify the A91/A92 column shape
    against the first live pull — these document types are less battle-tested
    than the A11/A61/A09 crossborder documents.
    """
    from functools import wraps
    import requests
    from entsoe.decorators import month_limited
    from entsoe.parsers import _extract_timeseries, _parse_timeseries_generic
    from entsoe.mappings import lookup_area

    def window_on_overflow(func: Callable) -> Callable:
        """Halve the request period when the API rejects it with 'exceeds the
        allowed maximum' — the 100-instance cap that _base_request does NOT
        translate to PaginationError — then concatenate. A sub-window with no
        data is skipped; NoMatchingDataError is re-raised only when every
        sub-window is empty. Errors other than the overflow propagate unchanged,
        so a real error is never silently turned into an empty.

        Splitting goes down to a 1-hour floor: a single dense day (multiple
        business types x MTU, or a revision-heavy day like some month-ends) can
        itself exceed 100 instances, so a day floor would surface it as an error.
        Multi-day windows split on a day boundary; sub-day windows split on the
        hour so boundaries stay MTU-aligned. Only a window that still overflows
        at <=1 hour is surfaced as an error (with its real reason)."""
        @wraps(func)
        def wrapper(*args, start, end, **kwargs):
            try:
                return func(*args, start=start, end=end, **kwargs)
            except NoMatchingDataError:
                raise
            except requests.HTTPError as exc:
                body = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                if "exceeds the allowed maximum" not in body:
                    raise
                overflow_exc = exc
            span = end - start
            if span <= pd.Timedelta(hours=1):
                raise overflow_exc  # at the floor: surface the real overflow error
            pivot = start + span / 2
            pivot = pivot.floor("D") if span > pd.Timedelta(days=1) else pivot.floor("h")
            if pivot <= start or pivot >= end:
                pivot = start + span / 2  # unaligned midpoint fallback
            frames = []
            for a, b in [(start, pivot), (pivot, end)]:
                try:
                    frames.append(wrapper(*args, start=a, end=b, **kwargs))
                except NoMatchingDataError:
                    pass
            frames = [f for f in frames
                      if f is not None and not getattr(f, "empty", False)]
            if not frames:
                raise NoMatchingDataError
            return pd.concat(frames).sort_index()
        return wrapper

    def _parse_congestion_mgmt(text: str) -> pd.DataFrame:
        """Parse an A91/A92 document into a tz-aware DataFrame, one column per
        businessType(+direction). Raises NoMatchingDataError when the document
        carries no usable TimeSeries, so empties propagate cleanly."""
        cols: dict[str, list[pd.Series]] = {}
        for ts in _extract_timeseries(text):
            point = ts.find("point")
            if point is None:
                continue
            label = next((tag for tag in ("quantity", "price.amount")
                          if point.find(tag) is not None), None)
            if label is None:
                continue
            s = _parse_timeseries_generic(ts, label=label, merge_series=True)
            bt = ts.find("businesstype")
            fd = ts.find("flowdirection.direction")
            parts = [t.text for t in (bt, fd) if t is not None]
            name = "_".join(parts) if parts else label
            cols.setdefault(name, []).append(s)
        if not cols:
            raise NoMatchingDataError
        # Each businessType becomes one column. Multiple TimeSeries under the
        # same type can repeat an MTU (revisions, or window-boundary overlap),
        # so keep the last value per timestamp to give every column a unique
        # index — otherwise assembling columns with differing coverage raises
        # "cannot reindex on an axis with duplicate labels".
        out: dict[str, pd.Series] = {}
        for k, v in cols.items():
            s = pd.concat(v).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            out[k] = s
        return pd.DataFrame(out)

    def _make_query(doctype: str):
        def _core(self, country_code_from, country_code_to,
                  start, end, **kwargs) -> pd.DataFrame:
            area_in = lookup_area(country_code_to)     # in_Domain
            area_out = lookup_area(country_code_from)  # out_Domain (== in for A92)
            params = {
                "documentType": doctype,
                "in_Domain": area_in.code,
                "out_Domain": area_out.code,
            }
            # _base_request raises NoMatchingDataError on an empty response, so
            # "no data for this period" surfaces as empty, not an error.
            text = self._base_request(params=params, start=start, end=end).text
            df = _parse_congestion_mgmt(text)
            # crossborder note: result is in the timezone of the origin (out) area
            df = df.tz_convert(area_out.tz)
            # half-open [start, end): truncate() is inclusive, which double-counts
            # the boundary timestamp when window_on_overflow stitches sub-windows
            # back together (a source of duplicate-index errors downstream).
            df = df[(df.index >= start) & (df.index < end)]
            return df
        # month blocks bound the initial window; window_on_overflow halves any
        # block the API still rejects for exceeding the 100-instance cap.
        return month_limited(window_on_overflow(_core))

    EntsoePandasClient.query_countertrading = _make_query("A91")     # 13.1.B
    EntsoePandasClient.query_congestion_costs = _make_query("A92")   # 13.1.C


_add_congestion_methods()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transmission-probe")

TZ = "Europe/Zurich"

# CH plus its four bordering areas. Each end lists the candidate area codes to
# try in order — the next code is only tried if the previous one returns a
# genuine empty (NoMatchingDataError), so first-match-empty-fallthrough resolves
# each product to whichever domain level actually publishes it.
#
#   The German end needs BOTH levels. CTA|DE(TransnetBW) = DE_TRANSNET
#   (10YDE-ENBW-----N) is the control area that physically borders Switzerland,
#   so the *physical* series (12.1.G flows, and NTC where published per CTA) live
#   there. DE_LU (10Y1001A1001A82H) is the German-Luxembourg *bidding zone*,
#   where the *commercial* series (12.1.F schedules) live. TransnetBW is listed
#   first because it is CH's physical counterpart; DE_LU catches the bidding-zone
#   products by fallthrough; the other three CTAs are last-resort only.
#   -> After the probe, PIN the resolved code per (dataset, border) below so the
#      pull never drifts between levels (see PINNED_BORDER_CODE).
#
#   Italy's CH-facing zone is IT_NORD (the border sits between CH and IT_NORD);
#   the national IT market area is kept as a fallback. Italy may carry the same
#   BZN-vs-CTA split as Germany — check the probe's resolved codes for CH-IT too.
CH_CODES: list[str] = ["CH"]
NEIGHBOURS: dict[str, list[str]] = {
    "DE": ["DE_TRANSNET", "DE_LU", "DE_AMPRION", "DE_TENNET", "DE_50HZ"],
    "FR": ["FR"],
    "IT_NORD": ["IT_NORD", "IT"],
    "AT": ["AT"],
}


@dataclass
class Border:
    """One directed border: from -> to, with candidate codes at each end."""
    label: str          # e.g. "CH-DE_LU"
    from_name: str
    to_name: str
    from_codes: list[str]
    to_codes: list[str]


def build_borders() -> list[Border]:
    """Both directions of each Swiss border. Border-scope products are directional
    (a flow / schedule / capacity from one area to another), so CH>N and N>CH are
    distinct series and both are probed."""
    borders: list[Border] = []
    for n, codes in NEIGHBOURS.items():
        borders.append(Border(f"CH-{n}", "CH", n, CH_CODES, codes))
        borders.append(Border(f"{n}-CH", n, "CH", codes, CH_CODES))
    return borders


# Single areas for scope="area" products (13.1.C congestion costs), which the
# API demands be queried with in_Domain == out_Domain ("Area EICs shall be the
# same"). Modelled as a Border with from == to so the same wrapper serves them.
# CH is the thesis focus; neighbours are cheap context.
AREAS_SINGLE: dict[str, list[str]] = {
    "CH": ["CH"],
    "DE_LU": ["DE_LU"],
    "FR": ["FR"],
    "IT_NORD": ["IT_NORD", "IT"],
    "AT": ["AT"],
}


def build_areas() -> list[Border]:
    """Single-area 'borders' (from == to) for scope='area' products."""
    return [Border(a, a, a, codes, codes) for a, codes in AREAS_SINGLE.items()]


def targets_for(task: "Task") -> list[Border]:
    """The iteration targets for a task: directed borders for border-scope
    products, single areas (in == out) for area-scope products."""
    return build_areas() if task.scope == "area" else build_borders()


# Which end-codes actually serve each (dataset, border), as established by the
# first probe run. A pinned pair is the ONLY one tried — no fallback, no silent
# level-mixing. Fill this from the coverage CSV's "Pin these" hint before the
# production pull. Empty for now; most borders answer on the primary codes.
#   key = (dataset, border_label) -> (from_code, to_code)
#
# Example for the German BZN-vs-CTA split — pin physical products to TransnetBW
# and commercial products to the DE_LU bidding zone once the probe confirms them:
#   ("physical_flows",      "CH-DE"): ("CH", "DE_TRANSNET"),
#   ("physical_flows",      "DE-CH"): ("DE_TRANSNET", "CH"),
#   ("scheduled_exchanges", "CH-DE"): ("CH", "DE_LU"),
#   ("scheduled_exchanges", "DE-CH"): ("DE_LU", "CH"),
PINNED_BORDER_CODE: dict[tuple[str, str], tuple[str, str]] = {}

# ENTSO-E code reminders (Transmission domain)
#   documentType: A61 Forecasted transfer capacity (NTC)   A11 Physical flows
#                 A09 Finalised/DA commercial schedules      A91 Counter trade
#                 A92 Congestion costs
#   contract_MarketAgreement.Type (NTC horizon):
#                 A01 day  A02 week  A03 month  A04 year
#   scheduled_exchanges: A05 total commercial schedule, A01 day-ahead


@dataclass
class Task:
    """One endpoint + one parameter combination (a cross-border series)."""

    dataset: str
    variant: str
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    enabled: bool = True
    # "border": iterate directed CH borders (from != to). "area": iterate single
    # areas with in == out (required by 13.1.C congestion costs).
    scope: str = "border"

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.variant}"


def build_tasks() -> list[Task]:
    """The Transmission-domain articles requested for the thesis."""
    tasks: list[Task] = []

    # --- 11.1 : Forecasted Transfer Capacity (NTC), four horizons ----------
    for horizon in ("dayahead", "weekahead", "monthahead", "yearahead"):
        tasks.append(Task(
            "ntc", horizon, f"query_net_transfer_capacity_{horizon}", {},
            f"11.1 forecasted net transfer capacity, {horizon} (MW)",
        ))

    # --- 12.1.F : Scheduled Commercial Exchanges --------------------------
    # total commercial schedule (A05) and the day-ahead schedule (A01).
    tasks.append(Task(
        "scheduled_exchanges", "total_A05", "query_scheduled_exchanges",
        {"dayahead": False}, "12.1.F total commercial schedules (MW)",
    ))
    tasks.append(Task(
        "scheduled_exchanges", "dayahead_A01", "query_scheduled_exchanges",
        {"dayahead": True}, "12.1.F day-ahead commercial schedules (MW)",
    ))

    # --- 12.1.G : Physical Flows ------------------------------------------
    tasks.append(Task(
        "physical_flows", "A11", "query_crossborder_flows", {},
        "12.1.G cross-border physical flows (MW)",
    ))

    # --- 13.1.B : Countertrading ------------------------------------------
    # Wrapper added above. Sparse/event-driven: expect many empty borders.
    tasks.append(Task(
        "countertrading", "A91", "query_countertrading", {},
        "13.1.B countertrading (MW) — A91 wrapper, verify output",
    ))

    # --- 13.1.C : Costs of Congestion Management --------------------------
    # Per-area figure (in_Domain == out_Domain), NOT a border product — the API
    # rejects a border pair with "Area EICs shall be the same". scope="area".
    tasks.append(Task(
        "congestion_costs", "A92", "query_congestion_costs", {},
        "13.1.C costs of congestion management (per area) — A92 wrapper",
        scope="area",
    ))

    return tasks


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


def redact(text: str) -> str:
    """Never let the API token reach a log file, a terminal or a paste."""
    return re.sub(r"(securityToken=)[^&\s]*", r"\1REDACTED", str(text))


def describe_error(exc: Exception) -> str:
    """Exception text plus the TP acknowledgement's <Reason> (code + text) — the
    part that actually names what went wrong. The Reason sits at the END of the
    document, after mRID/dates/participants, so a head-truncated body usually
    hides it; pull it out explicitly and fall back to a head slice otherwise."""
    body = getattr(getattr(exc, "response", None), "text", "") or ""
    body = " ".join(body.split())
    codes = re.findall(r"<code>(.*?)</code>", body)
    texts = re.findall(r"<text>(.*?)</text>", body)
    reason = "; ".join(f"{c}:{t}" for c, t in zip(codes, texts))
    msg = f"{type(exc).__name__}: {exc}"
    if reason:
        msg += f" | reason: {reason}"
    elif body:
        msg += f" | body: {body[:300]}"
    return redact(msg)[:500]


def tidy(obj: pd.Series | pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalise whatever entsoe-py returns into a tidy, parquet-safe frame.

    Crossborder methods return a named Series (framed under `name`); the A91/A92
    wrappers return a small DataFrame. Kept identical to the load/balancing
    tidy() so the on-disk schema conventions match — a Series is framed, any
    MultiIndex is stacked to long, numeric columns are coerced, duplicate labels
    are disambiguated, and the index is named 'timestamp' and sorted."""
    df = obj.to_frame(name=name) if isinstance(obj, pd.Series) else obj.copy()

    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels > 1:
        names = [n if n else f"level_{i}" for i, n in enumerate(df.columns.names)]
        df.columns = df.columns.set_names(names)
        outer = names[:-1]
        try:
            df = df.stack(outer, future_stack=True)
        except TypeError:  # pandas < 2.1
            df = df.stack(outer, dropna=False)
        value_cols = list(df.columns)
        df = df.reset_index(level=outer)
        df = df.dropna(subset=value_cols, how="all")
        df[outer] = df[outer].astype(str)

    df.columns = [str(c) for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) in ("str", "string"):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() and converted.isna().sum() == df[col].isna().sum():
                df[col] = converted

    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    df.columns = cols

    if isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "timestamp"
        df = df.sort_index()
    return df


def profile(df: pd.DataFrame) -> dict[str, Any]:
    """Diagnostics that survive long-format data: resolution is measured on
    unique timestamps, and duplicates are reported rather than hidden."""
    out: dict[str, Any] = {"rows": len(df), "cols": df.shape[1]}
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        uniq = df.index.unique().sort_values()
        out["unique_ts"] = len(uniq)
        out["dup_ts_share"] = round(1 - len(uniq) / len(df), 3)
        out["first_ts"] = str(uniq[0])
        out["last_ts"] = str(uniq[-1])
        if len(uniq) > 2:
            deltas = pd.Series(uniq).diff().dropna()
            out["resolution"] = str(deltas.mode().iloc[0]) if not deltas.empty else ""
    numeric = df.select_dtypes("number")
    out["na_share"] = (round(float(numeric.isna().mean().mean()), 3)
                       if not numeric.empty else "")
    return out


def call_with_retry(fn: Callable[..., Any], throttle: Throttle,
                    attempts: int = 3, **kwargs) -> Any:
    """Retry transient failures (429, 5xx, timeouts) with backoff.
    'No data' responses are NOT retried — they are an answer, not a failure —
    so an eventual 'empty' verdict is always definitive.

    Transient status is read from the response's status_code (429 or >=500),
    NOT by substring-matching the message: the request URL embeds the period
    (e.g. '202502...') and token, so a plain 'is "502" in the text' check
    spuriously fired on any February-2025 window and retried deterministic
    400s. A 400 is a client error and is never transient."""
    for i in range(attempts):
        throttle.wait()
        try:
            return fn(**kwargs)
        except NO_DATA_ERRORS:
            raise
        except Exception as exc:  # network / HTTP / parse
            status = getattr(getattr(exc, "response", None), "status_code", None)
            text = f"{type(exc).__name__}: {exc}"
            transient = (status == 429 or (status is not None and status >= 500)
                         or (status is None and any(t in text for t in (
                             "timeout", "Timeout", "Connection", "ConnectionError"))))
            if not transient or i == attempts - 1:
                raise
            backoff = 5 * (2 ** i)
            log.warning("transient error (%s) — retrying in %ss",
                        redact(text)[:80], backoff)
            time.sleep(backoff)


def load_api_key() -> str:
    """ENTSOE_API_KEY from the environment, else from the nearest .env found by
    walking up from this file (then ENTSOE-New/.env). Walking up rather than
    assuming a fixed depth means moving the script cannot break the lookup.
    Keeps the key out of the script and out of git."""
    key = os.getenv("ENTSOE_API_KEY")
    if key:
        return key
    here = Path(__file__).resolve().parent
    candidates = [p / ".env" for p in [here, *here.parents][:6]]
    candidates += [p / "ENTSOE-New" / ".env" for p in [here, *here.parents][:6]]
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


def _log_task_summary(key: str, outcomes: list[dict[str, Any]]) -> None:
    """One compact coverage line per product instead of a line per border."""
    ok = [o for o in outcomes if o["status"] == "ok"]
    empty = [o["border"] for o in outcomes if o["status"] == "empty"]
    errs = [o["border"] for o in outcomes if o["status"] == "error"]
    cached = [o["border"] for o in outcomes
              if str(o["status"]).startswith("cached")]
    parts = []
    if ok:
        parts.append("ok: " + " ".join(f"{o['border']}({o['rows']})" for o in ok))
    if empty:
        parts.append("no data: " + ",".join(empty))
    if errs:
        parts.append("ERR: " + ",".join(errs))
    if cached:
        parts.append("cached: " + ",".join(cached))
    log.info("%-46s %s", key, "  |  ".join(parts) if parts else "(no borders)")


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
                    help="minimum seconds between API calls (default 2.0)")
    ap.add_argument("--force", action="store_true",
                    help="re-attempt every combination in the state file")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-attempt only the combinations recorded as errors")
    ap.add_argument("--verbose", action="store_true",
                    help="log every border individually instead of one summary "
                         "line per product (empties are hidden by default)")
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
    throttle = Throttle(args.interval)
    tasks = [t for t in build_tasks()
             if t.enabled and args.only.lower() in t.dataset.lower()]

    stamp = f"{start:%Y%m%d}_{end:%Y%m%d}"
    est = sum(len(targets_for(t)) for t in tasks)
    log.info("probing %d datasets | %s -> %s | up to %d (dataset x target) calls",
             len(tasks), start.date(), end.date(), est)

    report: list[dict[str, Any]] = []

    for task in tasks:
        outcomes: list[dict[str, Any]] = []  # one per target, for the summary
        for border in targets_for(task):
            state_key = f"{task.key}__{border.label}__{stamp}"
            prior = state.get(state_key)
            if prior and not (args.retry_errors and prior == "error"):
                outcomes.append({"border": border.label,
                                 "status": f"cached:{prior}"})
                if args.verbose:
                    log.info("skip (already attempted: %s)  %s", prior, state_key)
                continue

            pinned = PINNED_BORDER_CODE.get((task.dataset, border.label))
            if pinned:
                combos = [pinned]
            elif task.scope == "area":
                # single-area products need in_Domain == out_Domain; try each
                # candidate code on the diagonal only, never a cross-pair.
                combos = [(c, c) for c in border.from_codes]
            else:
                combos = [(fc, tc) for fc in border.from_codes
                          for tc in border.to_codes]

            status, used_from, used_to, df, err = "empty", "", "", None, ""

            for fc, tc in combos:
                try:
                    raw = call_with_retry(
                        getattr(client, task.method), throttle,
                        country_code_from=fc, country_code_to=tc,
                        start=start, end=end, **task.kwargs)
                except NO_DATA_ERRORS as exc:
                    err, status = type(exc).__name__, "empty"
                    continue
                except Exception as exc:
                    err, status = describe_error(exc), "error"
                    break

                if raw is None or (hasattr(raw, "empty") and raw.empty):
                    status = "empty"
                    continue

                df = tidy(raw, task.dataset)
                # the API end boundary is inclusive: drop the timestamp that
                # belongs to the next period, or multi-period pulls double-count
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df[df.index < end]
                if df.empty:
                    status = "empty"
                    continue
                used_from, used_to, status, err = fc, tc, "ok", ""
                break

            row: dict[str, Any] = {
                "dataset": task.dataset,
                "variant": task.variant,
                "border": border.label,
                "from_code_used": used_from,
                "to_code_used": used_to,
                "status": status,
                "rows": 0,
                "cols": 0,
                "unique_ts": "",
                "dup_ts_share": "",
                "resolution": "",
                "first_ts": "",
                "last_ts": "",
                "na_share": "",
                "file": "",
                "note": task.note,
                "error": err,
            }

            if status == "ok" and df is not None:
                fname = (f"{task.dataset}__{task.variant}__"
                         f"{border.label}__{stamp}.parquet")
                df.to_parquet(out_dir / fname, engine="pyarrow",
                              compression="snappy")
                row.update(profile(df))
                row["file"] = fname
                outcomes.append({"border": border.label, "status": "ok",
                                 "rows": len(df)})
                if args.verbose:
                    log.info("OK    %-42s %-9s -> %5d rows, %2d cols (%s>%s)",
                             task.key, border.label, len(df), df.shape[1],
                             used_from, used_to)
            elif status == "empty":
                outcomes.append({"border": border.label, "status": "empty"})
                if args.verbose:
                    log.info("EMPTY %-42s %-9s (no data)", task.key, border.label)
            else:
                outcomes.append({"border": border.label, "status": "error",
                                 "err": err})
                log.warning("ERROR %-42s %-9s %s", task.key, border.label, err)

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
        rep = rep.drop_duplicates(subset=["dataset", "variant", "border"],
                                  keep="last")
    rep.to_csv(rep_path, index=False)

    print("\n" + "=" * 108)
    print(f"COVERAGE  {start.date()} -> {end.date()}")
    print("=" * 108)
    print(rep.pivot_table(index=["dataset", "variant"], columns="border",
                          values="status", aggfunc="first").fillna("-").to_string())

    ok = rep[rep.status == "ok"]
    empty = rep[rep.status == "empty"]
    errs = rep[rep.status == "error"]
    print(f"\nSummary: {len(ok)} ok, {len(empty)} empty (genuine no-data), "
          f"{len(errs)} error.")
    if not errs.empty:
        print("\nERRORS — these are NOT empty; inspect before pulling:")
        print(errs[["dataset", "variant", "border", "error"]].to_string(index=False))

    if not ok.empty:
        cols = ["dataset", "variant", "border", "from_code_used", "to_code_used",
                "rows", "unique_ts", "cols", "resolution", "na_share"]
        print("\nUsable series (status=ok):")
        print(ok[cols].to_string(index=False))
        print("\nPin these in PINNED_BORDER_CODE if an end resolved to a "
              "fallback code:")
        for (ds, bl), grp in ok.groupby(["dataset", "border"]):
            fc, tc = grp["from_code_used"].iloc[0], grp["to_code_used"].iloc[0]
            print(f'    ("{ds}", "{bl}"): ("{fc}", "{tc}"),')

    print(f"\nparquet + coverage csv written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
