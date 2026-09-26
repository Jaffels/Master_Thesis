"""
ENTSO-E Balancing probe — one month, four areas, coverage report.

Purpose: find out which Balancing endpoints actually return data for
CH / DE / FR / IT before committing to a multi-year download.

Writes one parquet per (dataset, variant, area) into Entsoe/Balancing/Data/
plus a coverage report so you can see at a glance what is empty,
what is thin, and what is usable.

Lives in Master_Thesis/Entsoe/Balancing/ and writes to its own Data/ subfolder,
so it can be run from anywhere.

Usage (macOS):
    # key is read from ENTSOE_API_KEY, or from the nearest .env walking up
    # from this file (falling back to ENTSOE-New/.env) if it is not exported
    .venv/bin/python Entsoe/Balancing/entsoe_balancing_probe.py
    .venv/bin/python Entsoe/Balancing/entsoe_balancing_probe.py --retry-errors
    .venv/bin/python Entsoe/Balancing/entsoe_balancing_probe.py \
        --start 2021-06-01 --end 2021-07-01

Re-runs are cheap: every attempted (dataset, variant, area) is recorded in
Data/_probe_state.json and skipped on the next run, so you
can stop and restart without hammering the API. --force ignores the state
file; --retry-errors re-attempts only the combinations that failed.

Once the probe has told you which area code serves a given dataset, pin it in
PINNED_AREA_CODE below so the production pull can never silently mix a control
area with a bidding zone.
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
from requests.exceptions import HTTPError

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


def _patch_symmetric_direction() -> None:
    """Replace entsoe-py's procured-balancing-capacity per-timeseries parser.

    Two fixes:
      * the stock parser maps only flow directions A01 (Up) and A02 (Down), so
        symmetrically procured Swiss FCR (A03) raised KeyError: 'A03';
      * it discards original_MarketProduct and psrType. If a market publishes
        more than one product type, summing across them without separating
        would silently mix them, so both are kept as index levels.

    One TimeSeries in an A15 document is ONE ACCEPTED AWARD (businessType B95):
    a single quantity at the clearing price for one product block. Summing
    across TimeSeries at a timestamp gives procured volume — do not dedupe on
    (Price, Volume), because under marginal pricing equal-sized awards at the
    same clearing price are legitimately distinct.

    NOTE: TimeSeries mRID is numbered per document, and a document covers one
    day, so mrid restarts at 1 every day. Group by timestamp only; mrid never
    identifies the same provider across days."""
    import entsoe.parsers as parsers

    direction = {"A01": "Up", "A02": "Down", "A03": "Symmetric"}

    def _tag(soup, name, default="NA"):
        node = soup.find(name)
        return node.text if node is not None else default

    def _parse(soup, tz):
        flow_direction = direction[soup.find("flowdirection.direction").text]
        mr_id = int(soup.find("mrid").text)
        product = _tag(soup, "original_marketproduct.marketproducttype")
        psr = _tag(soup, "mktpsrtype.psrtype")
        df = pd.DataFrame({
            "Price": parsers._parse_timeseries_generic(
                soup, label="procurement_price.amount", merge_series=True),
            "Volume": parsers._parse_timeseries_generic(
                soup, label="quantity", merge_series=True),
        })
        df.columns = pd.MultiIndex.from_product(
            [[flow_direction], [mr_id], [product], [psr], df.columns],
            names=("direction", "mrid", "product", "psr", "unit"))
        return df

    parsers._parse_procured_balancing_capacity = _parse


_patch_symmetric_direction()


def _patch_contracted_reserve_skip_bad() -> None:
    """Make entsoe-py's contracted-reserve parser tolerate CH's document shape.

    17.1.B&C publishes VOLUME and PRICE as separate TimeSeries. When only one
    of the two exists for a given product/direction/period (common for Swiss
    reserves — e.g. daily mFRR is price-only, some FCR rows volume-only), the
    stock parser hits `point.find('quantity')` (or the price tag) → None →
    'NoneType has no attribute text' and the whole request dies. Here we skip
    the TimeSeries that lacks the tag being parsed and keep the rest, so a
    quantity pull returns the volume-bearing series and a price pull the
    price-bearing series. Reuses the library's exact concat/groupby, so results
    for well-formed series are unchanged. A fully-unparseable document degrades
    to NoMatchingDataError ('empty') rather than an error."""
    import entsoe.parsers as parsers
    import entsoe.entsoe as client_mod
    from entsoe.exceptions import NoMatchingDataError

    def _safe(xml_text, tz, label):
        frames = []
        for soup in parsers._extract_timeseries(xml_text):
            try:
                frames.append(
                    parsers._parse_contracted_reserve_series(soup, tz, label))
            except (AttributeError, KeyError):
                # TimeSeries missing the parsed tag / businessType / direction
                continue
        if not frames:
            raise NoMatchingDataError
        df = pd.concat(frames, axis=1)
        df = df.T.groupby(level=[0, 1]).mean().T  # identical to the original
        df.sort_index(inplace=True)
        return df

    parsers.parse_contracted_reserve = _safe      # zip path calls this by name
    client_mod.parse_contracted_reserve = _safe    # text path (bound at import)


_patch_contracted_reserve_skip_bad()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("probe")

TZ = "Europe/Zurich"

# Starting (ceiling) sub-window for endpoints entsoe-py does NOT internally
# paginate. The API caps every XML response at 100 TimeSeries elements (a
# documented hard limit, independent of the 1-year request-range limit).
# entsoe-py only offsets past that cap where a method carries @paginated +
# @documents_limited; the balancing methods with only @year_limited /
# @month_limited return the truncated first 100 TimeSeries for the whole block.
# chunked_call() starts from this window and adaptively halves any window whose
# response looks capped, confirming a cap only when splitting actually recovers
# more rows — so it settles on the LARGEST window that stays under the cap for
# each endpoint, and reports it so you can pin --chunk for the production pull.
UNPAGINATED_CHUNK = "31D"

# Smallest window the adaptive descent will go down to before giving up and
# flagging a still-capped response. Reached only if an endpoint caps below a
# few hours, which none of the balancing series here are expected to.
CHUNK_FLOOR = "6h"

# Clean, pin-friendly window sizes (descending). The adaptive descent may
# settle on an odd span like 1.9D; snapping DOWN to the nearest ladder value
# gives a valid --chunk alias that is guaranteed safe (any window no larger
# than a known-uncapped one also stays under the cap). Conservative by design:
# the true largest-safe window can sit up to one ladder step higher.
_PIN_LADDER = ["31D", "30D", "14D", "10D", "7D", "5D", "3D", "2D", "1D",
               "12h", "6h"]


def _snap_down(td: pd.Timedelta) -> str:
    """Largest ladder alias whose duration is <= td (guaranteed-safe pin)."""
    for alias in _PIN_LADDER:
        if pd.Timedelta(alias) <= td:
            return alias
    return _PIN_LADDER[-1]

# Areas to probe. Each entry lists the area codes to try in order — the next
# code is only tried if the previous one returns nothing.
#   Germany: the 17.1.x family keys on CONTROL AREA, not bidding zone, so
#   DE_LU alone returns nothing for contracted reserves / imbalance even though
#   it works for 12.3.F. Let the probe find the one that serves each dataset.
#   Italy: publishes partly at control-area level ("IT"), partly per zone.
AREAS: dict[str, list[str]] = {
    "CH": ["CH"],
    "DE": ["DE_LU", "DE", "DE_AMPRION", "DE_TENNET", "DE_50HZ", "DE_TRANSNET"],
    "FR": ["FR"],
    "IT": ["IT", "IT_NORD"],
}

# Which area code actually serves each dataset, as established by the
# 2026-08 probe. A pinned code is the ONLY one tried — no fallback, no silent
# level-mixing between a control area and a bidding zone.
#   Germany publishes the 17.1.x family per TSO control area; FCR/aFRR/mFRR are
#   jointly procured, so the PRICE is identical across all four (verified:
#   4.52 EUR for CH-FCR day 1 on TenneT, 50Hertz and TransnetBW alike) and any
#   one of them is representative. VOLUME is per TSO and must be summed —
#   see SUM_AREAS.
PINNED_AREA_CODE: dict[tuple[str, str], str] = {
    ("activated_balancing_energy_prices", "DE"): "DE_TENNET",
    ("activated_balancing_energy_prices", "IT"): "IT_NORD",
    ("aggregated_bids", "DE"): "DE_TENNET",
    ("aggregated_bids", "IT"): "IT_NORD",
    ("contracted_reserve_prices", "DE"): "DE_TENNET",
    ("imbalance_prices", "DE"): "DE",
    ("imbalance_prices", "IT"): "IT_NORD",
    ("imbalance_volumes", "DE"): "DE_AMPRION",
    ("imbalance_volumes", "IT"): "IT",
    ("procured_balancing_capacity", "DE"): "DE_LU",
}

# Datasets whose values must be SUMMED across several area codes rather than
# taken from one. German contracted reserve volume is split across the four
# TSO control areas: a single code gives roughly a quarter of the national
# figure (DE_AMPRION alone was 247 of ~666 MW for FCR), which would look like
# a plausible number while being badly wrong.
SUM_AREAS: dict[tuple[str, str], list[str]] = {
    ("contracted_reserve_amount", "DE"):
        ["DE_AMPRION", "DE_TENNET", "DE_50HZ", "DE_TRANSNET"],
}

# ENTSO-E code reminders
#   process_type:  A52=FCR  A51=aFRR  A47=mFRR  A46=RR  A16=Realised
#   business_type: A95=FCR  A96=aFRR  A97=mFRR  A98=RR
#   type_marketagreement_type: A01=Daily  A13=Hourly  A02=Weekly  A03=Monthly


@dataclass
class Task:
    """One endpoint + one parameter combination."""

    dataset: str
    variant: str
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    enabled: bool = True
    # Sub-window size (pandas offset alias, e.g. "1D", "7D") to split the
    # request into before calling. Set ONLY for endpoints entsoe-py does not
    # internally paginate: those carry @year_limited/@month_limited but NOT
    # @paginated + @documents_limited(100), so a whole-month call silently
    # truncates at the API's 100-TimeSeries-per-response cap. None = one call,
    # letting the library's own pagination handle the cap.
    chunk: str | None = None

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.variant}"


def build_tasks() -> list[Task]:
    """The Tier-1/Tier-2 balancing endpoints worth probing."""
    tasks: list[Task] = []

    # --- 17.1.B / 17.1.C : contracted reserve amounts and prices -----------
    # entsoe-py >= 0.8 requires process_type here (one call per reserve type).
    # Swiss (and other) pre-reform procurement is WEEKLY (A02), so a probe that
    # only asks A01/A13 misses it entirely — CH aFRR/mFRR/FCR in 2021 are
    # largely weekly. A13 (hourly) came back empty everywhere in Aug 2026; A03
    # (monthly) is rare but cheap to check. Kept together so an older-month probe
    # shows which agreement types a given market actually used at the time.
    for pt, pt_label in [("A52", "FCR"), ("A51", "aFRR"), ("A47", "mFRR")]:
        for mat, mat_label in [("A01", "daily"), ("A02", "weekly"),
                               ("A03", "monthly"), ("A13", "hourly")]:
            tasks.append(Task(
                "contracted_reserve_amount", f"{pt_label}_{mat}_{mat_label}",
                "query_contracted_reserve_amount",
                {"process_type": pt, "type_marketagreement_type": mat},
                "17.1.B procured reserve volume (MW)",
            ))
            tasks.append(Task(
                "contracted_reserve_prices", f"{pt_label}_{mat}_{mat_label}",
                "query_contracted_reserve_prices",
                {"process_type": pt, "type_marketagreement_type": mat},
                "17.1.C procured reserve capacity price",
            ))

    # --- 12.3.F : procured balancing capacity (newer EB-GL endpoint) -------
    # Returns a (direction, mrid, unit) MultiIndex — one column block per
    # auction — which tidy() reshapes to long. Do not read the raw width.
    for pt, label in [("A52", "FCR"), ("A51", "aFRR"), ("A47", "mFRR")]:
        tasks.append(Task(
            "procured_balancing_capacity", f"{pt}_{label}",
            "query_procured_balancing_capacity",
            {"process_type": pt, "type_marketagreement_type": "A01"},
            "12.3.F procured capacity volume + price",
        ))

    # --- 17.1.G / 17.1.H : imbalance -------------------------------------
    tasks.append(Task(
        "imbalance_prices", "all", "query_imbalance_prices", {},
        "17.1.G imbalance prices",
        chunk=UNPAGINATED_CHUNK,  # year_limited only, no @paginated
    ))
    tasks.append(Task(
        "imbalance_volumes", "all", "query_imbalance_volumes", {},
        "17.1.H total imbalance volumes",
        chunk=UNPAGINATED_CHUNK,  # month_limited only, no @paginated
    ))

    # --- 17.1.E : activated balancing energy volumes ----------------------
    # DISABLED: entsoe-py sends documentType=A83 with no processType and TP
    # returns HTTP 400 for every area. A83 looks retired — the current TP view
    # is "activationAndActivatedBalancingReserves" (A84). The A84 endpoint
    # below returns Direction/Price/ReserveType but NO quantity, so activated
    # VOLUMES are not available through this wrapper at all. If you need them:
    #   EntsoeFileClient(...).list_folder('ActivatedBalancingEnergy_17.1.E')
    # Run with --include-broken to re-test (e.g. after an entsoe-py upgrade).
    for bt, label in [("A95", "FCR"), ("A96", "aFRR"), ("A97", "mFRR")]:
        tasks.append(Task(
            "activated_balancing_energy", f"{bt}_{label}",
            "query_activated_balancing_energy",
            {"business_type": bt},
            "17.1.E activated energy volume (A83 returns HTTP 400)",
            enabled=False,
            chunk=UNPAGINATED_CHUNK,  # year_limited only, no @paginated
        ))

    # --- 17.1.F : prices of activated balancing energy --------------------
    # One call covers every reserve type: the frame carries a ReserveType
    # column, so no per-business-type loop is needed.
    tasks.append(Task(
        "activated_balancing_energy_prices", "A16_realised",
        "query_activated_balancing_energy_prices",
        {"process_type": "A16"},
        "17.1.F activated energy price (long: Direction x ReserveType)",
        chunk=UNPAGINATED_CHUNK,  # year_limited only, no @paginated
    ))

    # --- 12.3.E : aggregated balancing energy bids (merit-order depth) ----
    for pt, label in [("A51", "aFRR"), ("A47", "mFRR")]:
        tasks.append(Task(
            "aggregated_bids", f"{pt}_{label}",
            "query_aggregated_bids",
            {"process_type": pt},
            "12.3.E aggregated bid volumes",
            chunk=UNPAGINATED_CHUNK,  # year_limited only, no @paginated —
            # this is the endpoint that returned "2 rows for a whole month":
            # 12.3.E emits many TimeSeries/day, so a month blew past the
            # 100-TimeSeries response cap and the wrapper kept only the first 100.
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
    """Exception text plus the TP response body, which names the offending
    parameter — the 'Bad Request' line on its own never does."""
    body = getattr(getattr(exc, "response", None), "text", "") or ""
    body = " ".join(body.split())[:300]
    msg = f"{type(exc).__name__}: {exc}"
    if body:
        msg += f" | body: {body}"
    return redact(msg)[:400]


def tidy(obj: pd.Series | pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalise whatever entsoe-py returns into a tidy, parquet-safe frame.

    A MultiIndex column frame (procured_balancing_capacity: direction x mrid x
    unit) is stacked to long, keeping the innermost level — Price/Volume — as
    columns. Without this you get hundreds of mostly-NaN columns, and the NaN
    share looks like missing data when it is really just sparse wide layout."""
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

    # some endpoints (aggregated_bids) return numbers as objects, which makes
    # every numeric diagnostic silently empty. Convert a column only when the
    # whole column converts, so genuine labels (Direction, ReserveType) stay.
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) in ("str", "string"):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() and converted.isna().sum() == df[col].isna().sum():
                df[col] = converted

    # de-duplicate column names (some endpoints repeat labels)
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
    'No data' responses are not retried — they are an answer, not a failure."""
    for i in range(attempts):
        throttle.wait()
        try:
            return fn(**kwargs)
        except NO_DATA_ERRORS:
            raise
        except Exception as exc:  # network / HTTP / parse
            msg = str(exc)
            transient = any(t in msg for t in ("429", "500", "502", "503",
                                               "504", "timeout", "Timeout",
                                               "Connection"))
            if not transient or i == attempts - 1:
                raise
            backoff = 5 * (2 ** i)
            log.warning("transient error (%s) — retrying in %ss",
                        redact(msg)[:80], backoff)
            time.sleep(backoff)


def _windows(start: pd.Timestamp, end: pd.Timestamp,
             window: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) into consecutive half-open windows of at most `window`.
    Half-open with no overlap, so concatenating the pieces never double-counts a
    boundary timestamp."""
    step = pd.Timedelta(window)
    out, cur = [], start
    while cur < end:
        nxt = min(cur + step, end)
        out.append((cur, nxt))
        cur = nxt
    return out


_UNSET = object()  # identity sentinel: DataFrame __eq__ is elementwise, so a
                   # value-based default would raise inside `resolve`
_OVERFLOW = object()  # a window the API couldn't serve (HTTP 400 "too large",
                      # or a 5xx/599 server hiccup on a dense response) — split
                      # it further; distinct from "no data" (None)


def _reaches_end(part: pd.DataFrame | None, w_end: pd.Timestamp,
                 tol: pd.Timedelta) -> bool:
    """True if `part`'s data extends to within `tol` of the window end. A full,
    uncapped response reaches its end (bar one MTU); a capped one stops short."""
    if part is None or not isinstance(part.index, pd.DatetimeIndex) or part.empty:
        return True  # empty is not "short"; there is simply nothing to recover
    return part.index.max() >= w_end - tol


def chunked_call(fn: Callable[..., Any], throttle: Throttle,
                 start: pd.Timestamp, end: pd.Timestamp, window: str,
                 dataset: str, floor: str = CHUNK_FLOOR,
                 **kwargs) -> tuple[pd.DataFrame | None, str, pd.Timedelta | None]:
    """Call `fn` over sub-windows and concatenate, adaptively shrinking any
    window whose response is capped at the API's 100-TimeSeries limit.

    Strategy: start from `window`; whenever a window's data stops well before
    its end (the fingerprint of a cap), split it in half and refetch. Keep the
    split ONLY if the two halves together return more rows than the whole did —
    that empirical gain is what tells a real cap apart from genuinely sparse
    data, where splitting recovers nothing and is discarded. Descent stops as
    soon as splitting stops helping, so each stretch settles on the largest
    window that stays under the cap. `floor` bounds the descent.

    Returns (tidy DataFrame or None, hint, largest_safe_window). `hint` is set
    only if a window is still capped at the floor. `largest_safe_window` is the
    widest window accepted without needing a split — pin it as --chunk for the
    production pull. Half-open windows never double-count a boundary timestamp.
    """
    floor_td = pd.Timedelta(floor)
    tol = min(pd.Timedelta("1h"), floor_td)  # "reaches end" slack (one MTU-ish)
    warnings: list[str] = []
    safe_spans: list[pd.Timedelta] = []      # spans accepted without splitting

    def fetch(a: pd.Timestamp, b: pd.Timestamp) -> Any:
        try:
            raw = call_with_retry(fn, throttle, start=a, end=b, **kwargs)
        except NO_DATA_ERRORS:
            # a window with genuinely no data is "empty", not a failure — over a
            # multi-year range some windows (before a series starts, gaps) will
            # be empty; skip them instead of aborting the whole pull
            return None
        except HTTPError as exc:
            # The API rejected THIS window — too many TimeSeries (400 "exceeds
            # limit") or a server/timeout hiccup on an oversized response (ENTSO-E
            # emits 599 and assorted 5xx for dense queries like Italian bids).
            # Signal that the window must be split; at the floor it's skipped with
            # a warning, so one bad window never aborts a multi-year pull. Auth
            # failures (401/403) are global — every request would fail — so those
            # re-raise instead of silently skipping the whole series.
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
            if status in (401, 403):
                raise
            return _OVERFLOW
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            return None
        part = tidy(raw, dataset)
        if isinstance(part.index, pd.DatetimeIndex) and not part.empty:
            part = part[part.index < b]  # own the exclusive upper boundary only
        return part if (part is not None and not part.empty) else None

    def rows(p: Any) -> int:
        return 0 if (p is None or p is _OVERFLOW) else len(p)

    def resolve(a: pd.Timestamp, b: pd.Timestamp,
                prefetched: Any = _UNSET) -> list[pd.DataFrame]:
        part = fetch(a, b) if prefetched is _UNSET else prefetched
        span = b - a
        if part is _OVERFLOW:
            # window rejected as too large — must split, cannot keep it
            if span <= floor_td:
                warnings.append(f"{a.date()}->{b.date()} unservable even at "
                                f"floor {floor}; sub-window skipped")
                return []
            mid = a + span / 2
            return resolve(a, mid) + resolve(mid, b)
        if _reaches_end(part, b, tol):
            if part is not None:
                safe_spans.append(span)  # this width was not capped here
            return [part] if part is not None else []
        if span <= floor_td:
            last = part.index.max() if rows(part) else "nothing"
            warnings.append(f"{a.date()}->{b.date()} still capped at floor "
                            f"{floor} (data to {last})")
            return [part] if part is not None else []
        # looks capped: split once and confirm by whether we recover more rows
        mid = a + span / 2
        left, right = fetch(a, mid), fetch(mid, b)
        if left is _OVERFLOW or right is _OVERFLOW:
            # a half overflowed -> splitting definitely helps; recurse both
            # (resolve handles _OVERFLOW, DataFrame and None halves alike)
            return resolve(a, mid, left) + resolve(mid, b, right)
        if rows(left) + rows(right) <= rows(part):
            # no gain -> genuine sparsity, not a cap; keep the whole window
            if part is not None:
                safe_spans.append(span)
            return [part] if part is not None else []
        # cap confirmed: recurse into each half, reusing the halves we fetched
        return resolve(a, mid, left) + resolve(mid, b, right)

    parts: list[pd.DataFrame] = []
    for w_start, w_end in _windows(start, end, window):
        parts.extend(resolve(w_start, w_end))

    largest_safe = max(safe_spans) if safe_spans else None
    if not parts:
        return None, "; ".join(warnings), largest_safe
    df = pd.concat(parts)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    return df, "; ".join(warnings), largest_safe


def load_api_key() -> str:
    """ENTSOE_API_KEY from the environment, else from the nearest .env found by
    walking up from this file (then ENTSOE-New/.env). Walking up rather than
    assuming a fixed depth means moving the script cannot break the lookup.
    Keeps the key out of the script and out of git."""
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


def _log_task_summary(key: str, outcomes: list[dict[str, Any]]) -> None:
    """One compact coverage line per product instead of a line per area.

    'empty' means the area does not publish that series (an expected, benign
    outcome for most area x product combinations) — it is a coverage fact, not
    an error, so it is summarised, never printed per area with an exception
    name. Genuine failures were already logged individually as ERROR."""
    ok = [o for o in outcomes if o["status"] == "ok"]
    empty = [o["area"] for o in outcomes if o["status"] == "empty"]
    errs = [o["area"] for o in outcomes if o["status"] == "error"]
    cached = [o["area"] for o in outcomes
              if str(o["status"]).startswith("cached")]
    parts = []
    if ok:
        parts.append("ok: " + " ".join(f"{o['area']}({o['rows']})" for o in ok))
    if empty:
        parts.append("no data: " + ",".join(empty))
    if errs:
        parts.append("ERR: " + ",".join(errs))
    if cached:
        parts.append("cached: " + ",".join(cached))
    log.info("%-46s %s", key, "  |  ".join(parts) if parts else "(no areas)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2026-08-01", help="inclusive, YYYY-MM-DD")
    ap.add_argument("--end", default="2026-09-01", help="exclusive, YYYY-MM-DD")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "Data"),
                    help="output folder (default: Data/ next to this script)")
    ap.add_argument("--only", default="", help="substring filter on dataset name")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="minimum seconds between API calls (default 2.0)")
    ap.add_argument("--chunk", default=UNPAGINATED_CHUNK,
                    help="starting/ceiling sub-window (pandas offset, e.g. 31D, "
                         "7D) for endpoints entsoe-py does not paginate — "
                         "imbalance, activated energy prices, aggregated bids. "
                         "The probe adaptively halves any window whose response "
                         "is capped and reports the largest safe window it finds "
                         f"(default {UNPAGINATED_CHUNK}).")
    ap.add_argument("--chunk-floor", default=CHUNK_FLOOR,
                    help="smallest window the adaptive descent will try before "
                         f"flagging a still-capped response (default {CHUNK_FLOOR})")
    ap.add_argument("--force", action="store_true",
                    help="re-attempt every combination in the state file")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-attempt only the combinations recorded as errors")
    ap.add_argument("--include-broken", action="store_true",
                    help="also probe endpoints known to be broken in entsoe-py")
    ap.add_argument("--verbose", action="store_true",
                    help="log every area individually instead of one summary "
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
             if (t.enabled or args.include_broken)
             and args.only.lower() in t.dataset.lower()]
    # let --chunk retune the window for every endpoint that needs one, without
    # touching the ones the library already paginates (chunk stays None there)
    for t in tasks:
        if t.chunk:
            t.chunk = args.chunk

    stamp = f"{start:%Y%m%d}_{end:%Y%m%d}"
    log.info("probing %d datasets x %d areas | %s -> %s | up to %d calls",
             len(tasks), len(AREAS), start.date(), end.date(),
             len(tasks) * len(AREAS))

    report: list[dict[str, Any]] = []

    for task in tasks:
        outcomes: list[dict[str, Any]] = []  # one per area, for the summary line
        for area_label, codes in AREAS.items():
            state_key = f"{task.key}__{area_label}__{stamp}"
            prior = state.get(state_key)
            if prior and not (args.retry_errors and prior == "error"):
                outcomes.append({"area": area_label, "status": f"cached:{prior}"})
                if args.verbose:
                    log.info("skip (already attempted: %s)  %s",
                             prior, state_key)
                continue

            pinned = PINNED_AREA_CODE.get((task.dataset, area_label))
            summed = SUM_AREAS.get((task.dataset, area_label))
            candidates = [pinned] if pinned else codes

            status, used_code, df, err = "empty", "", None, ""
            safe_window = ""  # largest un-capped window the descent settled on

            if summed:
                # sum numeric values across control areas; keep a part only if
                # every area answered, so a partial sum can never masquerade
                # as a national figure
                parts, missing = [], []
                for code in summed:
                    try:
                        raw = call_with_retry(
                            getattr(client, task.method), throttle,
                            country_code=code, start=start, end=end,
                            **task.kwargs,
                        )
                    except NO_DATA_ERRORS:
                        missing.append(code)
                        continue
                    except Exception as exc:
                        err, status = describe_error(exc), "error"
                        missing.append(code)
                        break
                    if raw is not None and not raw.empty:
                        parts.append(tidy(raw, task.dataset))
                    else:
                        missing.append(code)
                if parts and not missing:
                    df = (pd.concat(parts).groupby(level=0).sum(numeric_only=True))
                    df = df[df.index < end]
                    used_code, status, err = "+".join(summed), "ok", ""
                elif status != "error":
                    status = "empty"
                    err = f"incomplete sum, missing {','.join(missing)}"
                candidates = []

            for code in candidates:
                already_tidy = False
                try:
                    if task.chunk:
                        # endpoint entsoe-py does not paginate: adaptively split
                        # the range so no response hits the 100-TS cap, settling
                        # on the largest un-capped window on its own
                        raw, hint, safe = chunked_call(
                            getattr(client, task.method), throttle,
                            start, end, task.chunk, task.dataset,
                            floor=args.chunk_floor,
                            country_code=code, **task.kwargs,
                        )
                        already_tidy = True
                        if safe is not None:
                            safe_window = _snap_down(safe)
                        if hint:
                            log.warning("STILL CAPPED %-38s %-6s %s",
                                        task.key, code, hint)
                    else:
                        raw = call_with_retry(
                            getattr(client, task.method), throttle,
                            country_code=code, start=start, end=end,
                            **task.kwargs,
                        )
                except NO_DATA_ERRORS as exc:
                    err, status = type(exc).__name__, "empty"
                    continue
                except Exception as exc:
                    err, status = describe_error(exc), "error"
                    break

                if raw is None or (hasattr(raw, "empty") and raw.empty):
                    status = "empty"
                    continue

                df = raw if already_tidy else tidy(raw, task.dataset)
                # the API end boundary is inclusive: drop the timestamp that
                # belongs to the next period, or multi-period pulls double-count
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df[df.index < end]
                if df.empty:
                    status = "empty"
                    continue
                used_code, status, err = code, "ok", ""
                break

            row: dict[str, Any] = {
                "dataset": task.dataset,
                "variant": task.variant,
                "area": area_label,
                "area_code_used": used_code,
                "status": status,
                "rows": 0,
                "cols": 0,
                "unique_ts": "",
                "dup_ts_share": "",
                "resolution": "",
                "first_ts": "",
                "last_ts": "",
                "na_share": "",
                "safe_window": safe_window,
                "file": "",
                "note": task.note,
                "error": err,
            }

            if status == "ok" and df is not None:
                fname = f"{task.dataset}__{task.variant}__{area_label}__{stamp}.parquet"
                df.to_parquet(out_dir / fname, engine="pyarrow",
                              compression="snappy")
                row.update(profile(df))
                row["file"] = fname
                outcomes.append({"area": area_label, "status": "ok",
                                 "rows": len(df), "code": used_code,
                                 "safe": safe_window})
                if args.verbose:
                    log.info("OK    %-42s %-6s -> %5d rows, %2d cols (%s%s%s)",
                             task.key, area_label, len(df), df.shape[1],
                             used_code, ", dup ts" if row.get("dup_ts_share")
                             else "", f", safe={safe_window}" if safe_window
                             else "")
            elif status == "empty":
                outcomes.append({"area": area_label, "status": "empty"})
                if args.verbose:
                    log.info("EMPTY %-42s %-6s (%s)", task.key, area_label,
                             "no data")
            else:
                outcomes.append({"area": area_label, "status": "error",
                                 "err": err})
                # real failures are always shown, verbose or not
                log.warning("ERROR %-42s %-6s %s", task.key, area_label, err)

            report.append(row)
            state[state_key] = status
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True))

        # one compact line per product: what answered, what was blank, what broke
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
        rep = rep.drop_duplicates(subset=["dataset", "variant", "area"],
                                  keep="last")
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
                "unique_ts", "dup_ts_share", "cols", "resolution", "na_share"]
        if "safe_window" in ok.columns:
            cols.append("safe_window")
        print(ok[cols].to_string(index=False))

        codes = (ok.groupby(["dataset", "area"])["area_code_used"]
                   .nunique().reset_index())
        mixed = codes[codes.area_code_used > 1]
        if not mixed.empty:
            print("\nWARNING — one dataset answered on more than one area code:")
            print(mixed.to_string(index=False))
        print("\nPin these in PINNED_AREA_CODE before the production pull:")
        for (ds, ar), code in (ok.groupby(["dataset", "area"])["area_code_used"]
                               .first().items()):
            if code != ar:
                print(f'    ("{ds}", "{ar}"): "{code}",')

        # largest --chunk that stayed under the 100-TimeSeries cap everywhere.
        # A global --chunk must satisfy the tightest endpoint, so recommend the
        # smallest window discovered across the chunked series. Paginated-by-lib
        # datasets have no window; after a coverage CSV round-trip their blank
        # becomes NaN, so filter on "is a non-empty string" (bool(NaN) is True).
        if "safe_window" in ok.columns:
            valid = ok["safe_window"].apply(
                lambda w: isinstance(w, str) and w.strip() != "")
            if valid.any():
                okv = ok[valid]
                tightest = min(okv["safe_window"],
                               key=lambda w: pd.Timedelta(w))
                print("\nLargest safe --chunk for the production pull "
                      "(tightest across chunked endpoints):")
                print(f"    --chunk {tightest}")
                per_ds = (okv.groupby("dataset")["safe_window"]
                          .agg(lambda s: min(s, key=lambda w: pd.Timedelta(w))))
                for ds, w in per_ds.items():
                    print(f"        {ds:34s} safe up to {w}")

    print(f"\nparquet + coverage csv written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
