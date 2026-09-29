"""
ENTSO-E Load probe — one month, five areas, coverage report.

Purpose: find out which Load endpoints actually return data for the thesis-
relevant areas (CH and its four bordering bidding zones DE_LU / FR / IT_NORD /
AT) before committing to a multi-year download. Structurally identical to
entsoe_balancing_probe.py so the pull can drive off the same coverage CSV.

Covers the Load-domain articles requested:
    6.1.A   Actual Total Load                 -> query_load
    6.1.B   Day-Ahead Total Load Forecast     -> query_load_forecast(A01)
    6.1.C   Week-Ahead Total Load Forecast    -> query_load_forecast(A31)
    6.1.D   Month-Ahead Total Load Forecast   -> query_load_forecast(A32)
    6.1.E   Year-Ahead Total Load Forecast    -> query_load_forecast(A33)
    8.1     Year-Ahead Forecast Margin        -> query_load_forecast_margin(A33)

Why one raw series per article (not the "A&B" / "A&C&D&E" combined TP views):
    The TP web views 6.1.A&B and 6.1.A&C&D&E simply overlay actual load on the
    forecasts; 6.1.A appears in both. Pulling each article whole — actual once,
    each forecast horizon once — is the same RAW-pull / reconcile-later choice
    the balancing scripts make: nothing is double-counted at pull time, and the
    A&B / A&C&D&E overlays are trivial downstream joins on the timestamp index.
    (entsoe-py's query_load_and_forecast exists as a convenience for 6.1.A&B, but
    it would re-fetch 6.1.A, so it is deliberately not used here.)

8.1 is not exposed by entsoe-py: query_load_forecast_margin is added below via a
thin raw-request wrapper (documentType A70), the same monkeypatch approach the
balancing probe uses for its parsers. Verify its output once against a live pull
— the A70 document shape is less battle-tested than the A65 load documents.

Forecast-parser fix (6.1.C/D/E): entsoe-py's parse_loads keeps ONLY the min /
max forecast TimeSeries (businessType A60 / A61) for the week/month/year-ahead
horizons and silently drops everything else. Some zones/years publish the
horizon as a single forecast series under a different (or absent) businessType —
e.g. AT month-ahead 2021-2024 — so the stock parser returns an all-empty frame
and the pull records the year as 'empty' even though the document has data.
_patch_load_forecast_keep_all() below replaces the non-A01/A16 branch to keep
A60/A61 as Min/Max AND any other series as a plain 'Forecasted Load' column,
leaving the min/max case unchanged. A year may therefore carry Min/Max columns,
a Forecasted Load column, or both, across its per-year files — downstream joins
align on the timestamp index.

Load endpoints are @month_limited in entsoe-py (a >1-month range is split into
monthly sub-requests internally) and return few TimeSeries per period, so the
100-TimeSeries response cap that forces adaptive chunking on the balancing side
does not arise here. No chunk logic is needed; every Task keeps chunk=None.

Writes one parquet per (dataset, variant, area) into Entsoe/Load/Data/ plus a
coverage report so you can see at a glance what is empty and what is usable.
Lives in Master_Thesis/Entsoe/Load/ and writes to its own Data/ subfolder.

Usage (macOS), run from the thesis root:
    # key is read from ENTSOE_API_KEY, or from the nearest .env walking up
    .venv/bin/python Entsoe/Load/entsoe_load_probe.py
    .venv/bin/python Entsoe/Load/entsoe_load_probe.py --retry-errors
    .venv/bin/python Entsoe/Load/entsoe_load_probe.py \
        --start 2021-06-01 --end 2021-07-01

Re-runs are cheap: every attempted (dataset, variant, area) is recorded in
Data/_probe_state.json and skipped next run. --force ignores the state file;
--retry-errors re-attempts only the combinations that failed. Once the probe
has told you which area code serves a given dataset, pin it in PINNED_AREA_CODE
so the production pull can never silently swap one area level for another.
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


def _add_forecast_margin_method() -> None:
    """Add query_load_forecast_margin to the client (article 8.1, docType A70).

    entsoe-py ships no method for the Year-Ahead Forecast Margin, so we build one
    from the raw request machinery it already exposes: _base_request handles the
    security token, period formatting and the 'No matching data found' ->
    NoMatchingDataError translation, so an empty year behaves exactly like an
    empty load pull. Each A70 TimeSeries is read with the library's own generic
    quantity parser and labelled by businessType where present, defaulting to a
    single 'Forecast Margin' column. Attached to the class so the Task machinery
    can call it by name just like a native method."""
    from entsoe.parsers import _extract_timeseries, _parse_load_timeseries
    from entsoe.mappings import lookup_area

    bt_label = {
        "A91": "Forecast Margin",       # positive/expected margin
        "A92": "Forecast Margin",
        "A60": "Min Forecast Margin",
        "A61": "Max Forecast Margin",
    }

    def query_load_forecast_margin(self, country_code, start, end,
                                   process_type: str = "A33") -> pd.DataFrame:
        area = lookup_area(country_code)
        params = {
            "documentType": "A70",           # Load forecast margin
            "processType": process_type,     # A33 = year ahead
            "outBiddingZone_Domain": area.code,
        }
        # _base_request raises NoMatchingDataError on an empty response, so no
        # data for a year is surfaced as "empty", not an error.
        text = self._base_request(params=params, start=start, end=end).text

        cols: dict[str, list[pd.Series]] = {}
        for soup in _extract_timeseries(text):
            s = _parse_load_timeseries(soup)
            bt = soup.find("businesstype")
            name = bt_label.get(bt.text, "Forecast Margin") if bt else "Forecast Margin"
            cols.setdefault(name, []).append(s)
        if not cols:
            raise NoMatchingDataError

        df = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in cols.items()})
        df = df.tz_convert(area.tz).truncate(before=start, after=end)
        return df

    EntsoePandasClient.query_load_forecast_margin = query_load_forecast_margin




_add_forecast_margin_method()


def _patch_load_forecast_keep_all() -> None:
    """Keep every week/month/year-ahead forecast TimeSeries, not just min/max.

    entsoe-py's parse_loads, for any process_type other than A01/A16, keeps only
    businessType A60 (min) and A61 (max) and discards the rest. A horizon that a
    zone publishes as a single forecast series under another/absent businessType
    then parses to an empty frame, and the pull marks the year 'empty'. This
    replacement delegates A01/A16 to the original parser (unchanged) and, for the
    forecast horizons, keeps A60/A61 as Min/Max plus any other series as
    'Forecasted Load'. Unparseable TimeSeries are skipped rather than aborting
    the document; a document with nothing usable degrades to NoMatchingDataError
    ('empty'), which month_limited handles per block."""
    import entsoe.parsers as parsers
    import entsoe.entsoe as client_mod

    _orig_parse_loads = parsers.parse_loads
    label = {"A60": "Min Forecasted Load", "A61": "Max Forecasted Load"}

    def parse_loads(xml_text, process_type="A01"):
        if process_type in ("A01", "A16"):
            return _orig_parse_loads(xml_text, process_type=process_type)
        cols: dict[str, list[pd.Series]] = {}
        for soup in parsers._extract_timeseries(xml_text):
            try:
                s = parsers._parse_load_timeseries(soup)
            except (AttributeError, KeyError):
                continue  # TimeSeries missing curveType / an expected tag
            bt = soup.find("businesstype")
            name = label.get(bt.text, "Forecasted Load") if bt else "Forecasted Load"
            cols.setdefault(name, []).append(s)
        if not cols:
            raise NoMatchingDataError
        return pd.DataFrame({k: pd.concat(v).sort_index() for k, v in cols.items()})

    parsers.parse_loads = parse_loads      # zip / by-name path
    client_mod.parse_loads = parse_loads   # module global bound at import


_patch_load_forecast_keep_all()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("load-probe")

TZ = "Europe/Zurich"

# Areas to probe. Each entry lists the area codes to try in order — the next
# code is only tried if the previous one returns nothing. CH plus the four
# bordering bidding zones the thesis names (Germany/Luxembourg, France,
# Italy North, Austria).
#   DE_LU is the aggregate German-Luxembourg bidding zone and is what carries
#   total load; the control areas are listed only as a fallback.
#   Italy publishes load per zone; IT_NORD is the CH-bordering zone the thesis
#   uses for prices, with the national IT market area as a fallback.
AREAS: dict[str, list[str]] = {
    "CH": ["CH"],
    "DE_LU": ["DE_LU", "DE_AMPRION", "DE_TENNET", "DE_50HZ", "DE_TRANSNET"],
    "FR": ["FR"],
    "IT_NORD": ["IT_NORD", "IT"],
    "AT": ["AT"],
}

# Which area code actually serves each dataset, as established by the first
# probe run. A pinned code is the ONLY one tried — no fallback, no silent
# level-mixing. Fill this in from the coverage CSV's "Pin these" hint before the
# production pull. Empty for now; load usually answers on the primary code.
PINNED_AREA_CODE: dict[tuple[str, str], str] = {}

# ENTSO-E code reminders (Load domain)
#   documentType:  A65=System total load   A70=Load forecast margin
#   processType:   A16=Realised  A01=Day ahead  A31=Week ahead
#                  A32=Month ahead  A33=Year ahead


@dataclass
class Task:
    """One endpoint + one parameter combination."""

    dataset: str
    variant: str
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    enabled: bool = True
    # Kept for parity with the balancing scripts' Task/pull interface. Load
    # endpoints are @month_limited and cap-free, so this stays None (one call
    # per calendar year; the library splits it into months internally).
    chunk: str | None = None

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.variant}"


def build_tasks() -> list[Task]:
    """The Load-domain articles requested for the thesis."""
    tasks: list[Task] = []

    # --- 6.1.A : Actual Total Load ----------------------------------------
    tasks.append(Task(
        "actual_load", "A16_realised", "query_load", {},
        "6.1.A actual total load (MW)",
    ))

    # --- 6.1.B/C/D/E : Total Load Forecast at four horizons ----------------
    # A01 returns a single 'Forecasted Load' column; A31/A32/A33 normally return
    # 'Min Forecasted Load' / 'Max Forecasted Load' (businessType A60/A61), but
    # a zone/year that publishes a single forecast series is now kept as
    # 'Forecasted Load' too (see _patch_load_forecast_keep_all).
    for pt, label, art in [
        ("A01", "day_ahead",   "6.1.B"),
        ("A31", "week_ahead",  "6.1.C"),
        ("A32", "month_ahead", "6.1.D"),
        ("A33", "year_ahead",  "6.1.E"),
    ]:
        tasks.append(Task(
            "load_forecast", f"{label}_{pt}", "query_load_forecast",
            {"process_type": pt},
            f"{art} total load forecast, {label.replace('_', '-')} (MW)",
        ))

    # --- 8.1 : Year-Ahead Forecast Margin ---------------------------------
    # Not a native entsoe-py method; served by the A70 wrapper added above.
    tasks.append(Task(
        "forecast_margin", "year_ahead_A33", "query_load_forecast_margin",
        {"process_type": "A33"},
        "8.1 year-ahead forecast margin (MW) — A70 wrapper, verify output",
    ))

    return tasks


# Hard per-request socket timeout (seconds) for every ENTSO-E call (fix P4,
# added 2026-09-26). entsoe-py sends requests with no timeout by default; one
# stalled connection hung the Balancing pull for ~13 h. A timeout raises
# requests.Timeout, which call_with_retry treats as transient and retries.
REQUEST_TIMEOUT_S = 60


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

    Load frames come back already tidy (a DatetimeIndex and one or two named
    value columns), so this is mostly a no-op for them; it is kept identical to
    the balancing probe's tidy() so the on-disk schema conventions match — a
    Series is framed under `name`, any MultiIndex is stacked to long, numeric
    columns are coerced, duplicate column labels are disambiguated, and the
    index is named 'timestamp' and sorted."""
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
    """One compact coverage line per product instead of a line per area."""
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
    log.info("%-42s %s", key, "  |  ".join(parts) if parts else "(no areas)")


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

    client = EntsoePandasClient(api_key=api_key, timeout=REQUEST_TIMEOUT_S)
    throttle = Throttle(args.interval)
    tasks = [t for t in build_tasks()
             if t.enabled and args.only.lower() in t.dataset.lower()]

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
                    log.info("skip (already attempted: %s)  %s", prior, state_key)
                continue

            pinned = PINNED_AREA_CODE.get((task.dataset, area_label))
            candidates = [pinned] if pinned else codes

            status, used_code, df, err = "empty", "", None, ""

            for code in candidates:
                try:
                    raw = call_with_retry(
                        getattr(client, task.method), throttle,
                        country_code=code, start=start, end=end, **task.kwargs)
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
                                 "rows": len(df), "code": used_code})
                if args.verbose:
                    log.info("OK    %-38s %-7s -> %5d rows, %2d cols (%s)",
                             task.key, area_label, len(df), df.shape[1], used_code)
            elif status == "empty":
                outcomes.append({"area": area_label, "status": "empty"})
                if args.verbose:
                    log.info("EMPTY %-38s %-7s (no data)", task.key, area_label)
            else:
                outcomes.append({"area": area_label, "status": "error", "err": err})
                log.warning("ERROR %-38s %-7s %s", task.key, area_label, err)

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

    print(f"\nparquet + coverage csv written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
