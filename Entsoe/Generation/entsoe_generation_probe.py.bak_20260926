"""
ENTSO-E Generation probe — one window, five areas, coverage report.

Purpose: find out which Generation endpoints actually return data for the
thesis-relevant areas (CH and its four bordering bidding zones DE_LU / FR /
IT_NORD / AT) before committing to a multi-year download. Structurally
identical to entsoe_load_probe.py / entsoe_balancing_probe.py so the pull can
drive off the same coverage CSV.

Covers the Generation-domain articles requested (and a couple of close
neighbours — "not limited to"):

    14.1.A  Installed Capacity per Production Type   -> query_installed_generation_capacity          [annual]
    14.1.B  Installed Capacity per Production Unit    -> query_installed_generation_capacity_per_unit  [annual]
    14.1.C  Day-ahead Aggregated Generation           -> query_generation_forecast (processType A01)
    14.1.D  Gen. Forecast Wind & Solar (day-ahead)    -> query_wind_and_solar_forecast (A01)
    14.1.D  Gen. Forecast Wind & Solar (intraday)     -> query_intraday_wind_and_solar_forecast   (bonus)
    16.1.A  Actual Generation per Generation Unit     -> query_generation_per_plant               [day-limited, heavy]
    16.1.B&C Actual Generation per Production Type     -> query_generation
    16.1.D  Water Reservoirs & Hydro Storage Plants    -> query_aggregate_water_reservoirs_and_hydro_storage

Why one raw series per article (psr_type=None, i.e. ALL production types in one
wide frame) rather than iterating per fuel:
    Passing psr_type=None returns every production type as its own column (or,
    for query_generation, a (type, Actual Aggregated / Actual Consumption)
    column MultiIndex that tidy() stacks to long). That is the same RAW-pull /
    reconcile-later choice the Load and balancing scripts make: pull each
    article whole, split by fuel downstream. It also keeps the request count
    low and the on-disk schema identical to the other domains.

Expect a LOT of empty / N/A here, and that is fine — it is the whole reason the
probe exists:
  * A small country like CH reports only a handful of production types (hydro,
    nuclear, solar, a little thermal), so most fuel columns in 16.1.B&C /
    14.1.A come back entirely NaN. They are kept, not dropped, so the on-disk
    schema is stable across areas.
  * 14.1.B (per production unit) and 16.1.A (per generation unit) are not
    published by every TSO; where they are missing, those combinations surface
    as status=empty in the coverage CSV, exactly as intended. Do NOT assume in
    advance which areas are empty — the probe is what decides that. (16.1.D
    water reservoirs IS published for CH — it is the weekly filling rate of the
    Swiss storage fleet, a central hydro series for this thesis, not an
    afterthought.)

Unlike the Load probe there is NO missing endpoint to monkeypatch: every
article above has a native entsoe-py method, so this file is pure request
machinery + a build_tasks() for the Generation domain.

Rate-limit / windowing notes (entsoe-py 0.8.x decorators):
  * query_generation, query_generation_forecast, query_wind_and_solar_forecast
    are @month_limited and cap-free at CH scale -> one call per calendar year
    is fine (the library splits it into monthly sub-requests). Same as Load.
  * query_aggregate_water_reservoirs_and_hydro_storage is @year_limited
    @paginated -> one call per year, offset handled internally.
  * query_installed_generation_capacity[_per_unit] are @year_limited ANNUAL
    snapshots: the yearly value sits at the start of the year, so a 1-month
    probe window reads empty. These Tasks carry annual=True and the probe
    widens their window to the whole calendar year of --start so coverage is
    not a false negative. (The production pull already works per calendar year,
    so it needs no such special-casing.)
  * query_generation_per_plant (16.1.A) is @day_limited: a multi-day range is
    split into DAILY sub-requests internally, and it is NOT paginated, so a
    single day that exceeds the 100-TimeSeries cap (possible for large zones
    like DE) is silently truncated. For CH it is well under the cap. It is left
    enabled but flagged heavy; probe it over a short window first.

Writes one parquet per (dataset, variant, area) into Entsoe/Generation/Data/
plus a coverage report. Lives in Master_Thesis/Entsoe/Generation/ and writes to
its own Data/ subfolder.

Usage (macOS), run from the thesis root:
    # key is read from ENTSOE_API_KEY, or from the nearest .env walking up
    .venv/bin/python Entsoe/Generation/entsoe_generation_probe.py
    .venv/bin/python Entsoe/Generation/entsoe_generation_probe.py --retry-errors
    .venv/bin/python Entsoe/Generation/entsoe_generation_probe.py \
        --start 2024-06-01 --end 2024-07-01
    # per-plant is heavy; probe it on its own over a short window:
    .venv/bin/python Entsoe/Generation/entsoe_generation_probe.py \
        --only per_unit --start 2024-06-01 --end 2024-06-08

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

try:
    from requests.exceptions import HTTPError as _HTTPError
except ImportError:  # pragma: no cover
    class _HTTPError(Exception): ...  # type: ignore[no-redef]

def _is_client_error(exc: Exception) -> bool:
    """True for 4xx HTTP errors — bad parameter / not published, not transient."""
    resp = getattr(exc, "response", None)
    return resp is not None and 400 <= resp.status_code < 500


def _is_parser_error(exc: Exception) -> bool:
    """True for entsoe-py XML parser crashes — e.g. a generation unit with no
    <name> tag triggers AttributeError: 'NoneType' object has no attribute 'text'.
    These are non-retryable bugs in the response data for a specific day/area,
    not transient network issues."""
    return isinstance(exc, (AttributeError, TypeError, KeyError))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gen-probe")

TZ = "Europe/Zurich"

# Areas to probe. Each entry lists the area codes to try in order — the next
# code is only tried if the previous one returns nothing. CH plus the four
# bordering bidding zones the thesis names (Germany/Luxembourg, France,
# Italy North, Austria).
#   Generation per type / per unit for Germany is historically published on the
#   four control areas (Amprion / TenneT / 50Hertz / TransnetBW) as well as the
#   DE_LU bidding zone, so those are listed as fallbacks.
#   Italy publishes generation per zone; IT_NORD is the CH-bordering zone, with
#   the national IT market area as a fallback.
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
# production pull. Empty for now.
PINNED_AREA_CODE: dict[tuple[str, str], str] = {}

# ENTSO-E code reminders (Generation domain)
#   documentType:  A68=Installed capacity aggregated  A71=Gen forecast
#                  A69=Wind/solar forecast  A73=Actual per unit
#                  A75=Actual per type  A72=Water reservoirs
#   processType:   A16=Realised  A01=Day ahead  A40=Intraday total
#   psr_type=None  -> ALL production types returned as columns (what we want)


@dataclass
class Task:
    """One endpoint + one parameter combination."""

    dataset: str
    variant: str
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    enabled: bool = True
    # Kept for parity with the balancing scripts' Task/pull interface. All the
    # Generation endpoints are @month_limited / @year_limited / @day_limited and
    # cap-free at CH scale, so this stays None (the library handles splitting).
    chunk: str | None = None
    # Annual snapshot (14.1.A / 14.1.B): the value sits at the start of the
    # year, so the probe widens a short --start/--end window to the full
    # calendar year before querying. The production pull works per calendar
    # year already and ignores this flag.
    annual: bool = False
    # @day_limited endpoint (16.1.A): entsoe-py fans out into daily sub-requests
    # internally, but its loop only catches NoMatchingDataError — a 400 on any
    # single day (e.g. a DST-transition boundary) kills the entire call. When
    # this flag is True the pull can fall back to its own per-day loop that
    # catches 400s per day, losing only the bad day(s) instead of the year.
    day_limited: bool = False

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.variant}"


def build_tasks() -> list[Task]:
    """The Generation-domain articles requested for the thesis."""
    tasks: list[Task] = []

    # --- 14.1.A : Installed Capacity per Production Type (aggregated) ------
    tasks.append(Task(
        "installed_capacity", "per_type", "query_installed_generation_capacity",
        {},  # psr_type=None -> all production types as columns
        "14.1.A installed capacity per production type (MW, annual)",
        annual=True,
    ))

    # --- 14.1.B : Installed Capacity per Production Unit -------------------
    tasks.append(Task(
        "installed_capacity_unit", "per_unit",
        "query_installed_generation_capacity_per_unit", {},
        "14.1.B installed capacity per production unit (MW, annual) — "
        "not published by every TSO",
        annual=True,
    ))

    # --- 14.1.C : Day-ahead Aggregated Generation forecast ----------------
    tasks.append(Task(
        "generation_forecast", "day_ahead", "query_generation_forecast",
        {"process_type": "A01"},
        "14.1.C day-ahead aggregated generation forecast (MW)",
    ))

    # --- 14.1.D : Generation Forecast for Wind and Solar ------------------
    # Day-ahead (A01) and the intraday variant (bonus, "not limited to").
    tasks.append(Task(
        "wind_solar_forecast", "day_ahead", "query_wind_and_solar_forecast",
        {"process_type": "A01"},
        "14.1.D day-ahead wind & solar generation forecast (MW)",
    ))
    tasks.append(Task(
        "wind_solar_forecast", "intraday",
        "query_intraday_wind_and_solar_forecast", {},
        "14.1.D intraday wind & solar generation forecast (MW) — bonus",
    ))

    # --- 16.1.B&C : Actual Generation per Production Type -----------------
    # nett=False -> both 'Actual Aggregated' (B) and 'Actual Consumption' (C)
    # for storage-capable types (pumped hydro); tidy() stacks the fuel level.
    tasks.append(Task(
        "actual_generation", "per_type", "query_generation", {},
        "16.1.B&C actual generation per production type (MW), incl. consumption",
    ))

    # --- 16.1.A : Actual Generation per Generation Unit ------------------
    # @day_limited and NOT paginated: heavy, and a single day over the
    # 100-series cap (large zones) is truncated. Fine for CH; probe short.
    tasks.append(Task(
        "actual_generation_unit", "per_unit", "query_generation_per_plant", {},
        "16.1.A actual generation per generation unit (MW) — DAY-LIMITED, heavy",
        day_limited=True,
    ))

    # --- 16.1.D : Water Reservoirs & Hydro Storage Plants ---------------
    tasks.append(Task(
        "water_reservoirs", "weekly",
        "query_aggregate_water_reservoirs_and_hydro_storage", {},
        "16.1.D aggregate weekly average filling rate of water reservoirs & "
        "hydro storage (MWh stored energy) — published for CH, core hydro series",
    ))

    return tasks


class Throttle:
    """Keep a minimum gap between API calls. ENTSO-E allows 400 req/min;
    we stay far below that so the probe never trips the limiter.

    Note: for @day_limited endpoints (16.1.A) entsoe-py fires its per-day
    sub-requests inside a single library call, so the Throttle only spaces the
    OUTER calls. Keep --interval sane and the probe window short for per-unit."""

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

    Identical to the Load/balancing probe's tidy() so the on-disk schema matches
    across domains. A Series is framed under `name`; any column MultiIndex (e.g.
    query_generation's (production_type, Actual Aggregated/Consumption), or
    query_generation_per_plant's (unit, type)) is stacked to long with the outer
    levels moved into columns; numeric columns are coerced; duplicate column
    labels are disambiguated; the index is named 'timestamp' and sorted."""
    df = obj.to_frame(name=name) if isinstance(obj, pd.Series) else obj.copy()

    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels > 1:
        names = [n if n else f"level_{i}" for i, n in enumerate(df.columns.names)]
        df.columns = df.columns.set_names(names)
        outer = names[:-1]
        try:
            df = df.stack(outer, future_stack=True)
        except TypeError:  # pandas < 2.1
            df = df.stack(outer, dropna=False)
        df = df.reset_index(level=outer)
        # do NOT dropna here — all-NaN rows are n/e fields from ENTSO-E and
        # must be preserved so the on-disk schema is stable across areas/years
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
    unique timestamps, and duplicates are reported rather than hidden.

    na_share is expected to be high for generation-per-type frames: fuels a
    country does not operate are all-NaN columns, by design."""
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
    'No data' and 4xx client errors are not retried — they are answers, not
    failures. A 400 means the TSO does not publish this series for this area;
    it is raised immediately so the caller records it as status=error and skips
    without wasting further retries."""
    for i in range(attempts):
        throttle.wait()
        try:
            return fn(**kwargs)
        except NO_DATA_ERRORS:
            raise
        except Exception as exc:  # network / HTTP / parse
            if _is_client_error(exc) or _is_parser_error(exc):
                raise  # 4xx or parser crash — don't retry, let caller handle
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


def _effective_window(task: Task, start: pd.Timestamp, end: pd.Timestamp
                      ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Widen a short probe window to the whole calendar year for annual
    snapshots (14.1.A / 14.1.B), so their yearly value is not missed. All other
    tasks probe exactly the requested [start, end)."""
    if not task.annual:
        return start, end
    y0 = pd.Timestamp(f"{start.year}-01-01", tz=start.tz)
    y1 = pd.Timestamp(f"{start.year + 1}-01-01", tz=start.tz)
    return y0, y1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2024-06-01", help="inclusive, YYYY-MM-DD")
    ap.add_argument("--end", default="2024-07-01", help="exclusive, YYYY-MM-DD")
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

    client = EntsoePandasClient(api_key=api_key)
    throttle = Throttle(args.interval)
    tasks = [t for t in build_tasks()
             if t.enabled and args.only.lower() in t.dataset.lower()]

    stamp = f"{start:%Y%m%d}_{end:%Y%m%d}"
    log.info("probing %d datasets x %d areas | %s -> %s | up to %d calls",
             len(tasks), len(AREAS), start.date(), end.date(),
             len(tasks) * len(AREAS))

    report: list[dict[str, Any]] = []

    for task in tasks:
        eff_start, eff_end = _effective_window(task, start, end)
        if task.annual and (eff_start, eff_end) != (start, end):
            log.info("%-42s annual snapshot — widening window to %s..%s",
                     task.key, eff_start.date(), eff_end.date())

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
                        country_code=code, start=eff_start, end=eff_end,
                        **task.kwargs)
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
                    df = df[df.index < eff_end]
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
