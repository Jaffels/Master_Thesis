"""
energycharts_probe.py — coverage / resolution / range probe for the Energy-Charts API
(Fraunhofer ISE, https://api.energy-charts.info, data licence CC BY 4.0).

What it checks
  0. Schema:    reads openapi.json, lists all endpoints, prints the parameters of
                /public_power and /frequency (names, defaults, allowed values).
  1. CH solar:  /public_power?country=ch&subtype=solarlog
                - resolution, series names, NaN share for one day and one full year
                - how far back the series goes (2015 / 2018 / 2021 test days)
  2. Frequency: /frequency
                - resolution and history (2015 / 2018 / 2021 / recent test days)
                - largest range one call can return (1 day -> 7 days -> 31 days)
                - estimated full-pull time and raw size for 2021-01 -> 2026-09
                  and 2015-01 -> 2026-09

Rate limits (API docs): 2 req/min per endpoint and client IP, burst 4; /price 2/min.
The probe waits 31 s between calls to the same endpoint and honours HTTP 429
Retry-After. Expected run time: ~6-8 min (~12 requests).

Run from the thesis root (needs requests, pandas):
    python EnergyCharts/energycharts_probe.py
    python EnergyCharts/energycharts_probe.py --skip-solar      # frequency only
    python EnergyCharts/energycharts_probe.py --skip-frequency  # solar only

Writes only to EnergyCharts/probe_output/ (creates it if missing, never deletes):
    energycharts_openapi.json        full API schema
    energycharts_probe_report.txt    human-readable report (also printed)
    energycharts_probe_results.csv   one row per request
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.energy-charts.info"
OUT_DIR = Path("EnergyCharts") / "probe_output"
MIN_GAP_S = 31          # seconds between calls to the same endpoint (2 req/min)
TIMEOUT_S = 180         # large frequency ranges can be slow
MAX_429_RETRIES = 3

# Full-pull windows used for the time estimate
PULL_WINDOWS = {
    "2021-01 -> 2026-09": (date(2021, 1, 1), date(2026, 9, 1)),
    "2015-01 -> 2026-09": (date(2015, 1, 1), date(2026, 9, 1)),
}

_last_call: dict[str, float] = {}
_report: list[str] = []
_rows: list[dict] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _report.append(msg)


# --------------------------------------------------------------------------- HTTP

def throttled_get(endpoint: str, params: dict) -> tuple[requests.Response | None, float, str]:
    """GET with per-endpoint spacing and 429 handling. Returns (response, seconds, error)."""
    wait = MIN_GAP_S - (time.time() - _last_call.get(endpoint, 0))
    if wait > 0:
        print(f"    ...waiting {wait:.0f} s (rate limit)", flush=True)
        time.sleep(wait)

    for attempt in range(MAX_429_RETRIES + 1):
        t0 = time.time()
        try:
            r = requests.get(f"{BASE}{endpoint}", params=params, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            _last_call[endpoint] = time.time()
            return None, time.time() - t0, f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0
        _last_call[endpoint] = time.time()
        if r.status_code == 429 and attempt < MAX_429_RETRIES:
            retry = int(r.headers.get("Retry-After", 60))
            print(f"    429 rate-limited, retrying in {retry} s", flush=True)
            time.sleep(retry)
            continue
        return r, elapsed, ""
    return None, 0.0, "429 after retries"


# ------------------------------------------------------------------ response parsing

def parse_series(payload) -> tuple[pd.DatetimeIndex, dict[str, list]]:
    """Extract timestamps and value series from a v1 response, whatever its layout.

    Known layouts:
      public_power: {"unix_seconds": [...], "production_types": [{"name", "data"}]}
      simple:       {"unix_seconds": [...], "data": [...]}
    Falls back to any other list the same length as unix_seconds.
    """
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict) or "unix_seconds" not in payload:
        return pd.DatetimeIndex([]), {}

    ts = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    n = len(ts)
    series: dict[str, list] = {}

    for pt in payload.get("production_types", []) or []:
        if isinstance(pt, dict) and isinstance(pt.get("data"), list):
            series[str(pt.get("name"))] = pt["data"]
    for key, val in payload.items():
        if key in ("unix_seconds", "production_types"):
            continue
        if isinstance(val, list) and len(val) == n and n > 0:
            series[key] = val
    return ts, series


def describe(ts: pd.DatetimeIndex, series: dict[str, list]) -> dict:
    if len(ts) == 0:
        return {"n_points": 0}
    diffs = pd.Series(ts).diff().dropna().dt.total_seconds()
    out = {
        "n_points": len(ts),
        "first_utc": ts.min().isoformat(),
        "last_utc": ts.max().isoformat(),
        "resolution_s": float(diffs.median()) if len(diffs) else float("nan"),
        "n_series": len(series),
    }
    nan_share = {}
    for name, vals in series.items():
        s = pd.to_numeric(pd.Series(vals), errors="coerce")
        nan_share[name] = round(float(s.isna().mean()), 4)
    out["nan_share"] = nan_share
    return out


def probe_call(label: str, endpoint: str, params: dict) -> dict:
    log(f"  [{label}] GET {endpoint} {params}")
    r, elapsed, err = throttled_get(endpoint, params)
    row = {"label": label, "endpoint": endpoint, "params": json.dumps(params),
           "status": None, "seconds": round(elapsed, 1), "bytes": 0, "error": err}

    if r is None:
        log(f"    -> FAILED ({err})")
        _rows.append(row)
        return row

    row["status"] = r.status_code
    row["bytes"] = len(r.content)
    if r.status_code != 200:
        row["error"] = r.text[:300].replace("\n", " ")
        log(f"    -> HTTP {r.status_code} after {elapsed:.1f} s: {row['error']}")
        _rows.append(row)
        return row

    try:
        payload = r.json()
    except ValueError:
        row["error"] = "response is not JSON"
        log("    -> 200 but response is not JSON")
        _rows.append(row)
        return row

    ts, series = parse_series(payload)
    info = describe(ts, series)
    row.update({k: v for k, v in info.items() if k != "nan_share"})
    row["nan_share"] = json.dumps(info.get("nan_share", {}))
    if isinstance(payload, dict) and payload.get("deprecated"):
        row["error"] = "endpoint flagged deprecated"

    if info["n_points"] == 0:
        keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        log(f"    -> 200, {row['bytes']/1e6:.2f} MB, {elapsed:.1f} s, NO time series found (keys: {keys})")
    else:
        log(f"    -> 200, {row['bytes']/1e6:.2f} MB, {elapsed:.1f} s, "
            f"{info['n_points']:,} points, resolution {info['resolution_s']:.0f} s, "
            f"{info['first_utc']} -> {info['last_utc']}")
        nan_txt = ", ".join(f"{k}: {v:.1%}" for k, v in info["nan_share"].items())
        log(f"       series ({info['n_series']}): {nan_txt}")
    _rows.append(row)
    return row


def is_ok(row: dict) -> bool:
    return row.get("status") == 200 and row.get("n_points", 0) > 0


# ------------------------------------------------------------------------- probes

def probe_schema() -> dict:
    log("=" * 78)
    log("0. API SCHEMA")
    log("=" * 78)
    try:
        r = requests.get(f"{BASE}/openapi.json", timeout=60)
        r.raise_for_status()
        schema = r.json()
    except Exception as e:  # noqa: BLE001
        log(f"  Could not load openapi.json ({e}); continuing with default parameters.")
        return {}

    (OUT_DIR / "energycharts_openapi.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    info = schema.get("info", {})
    log(f"  API version: {info.get('version')}")
    paths = sorted(schema.get("paths", {}))
    log(f"  {len(paths)} endpoints: {', '.join(paths)}")

    for ep in ("/public_power", "/frequency"):
        op = schema.get("paths", {}).get(ep, {}).get("get")
        if not op:
            log(f"  {ep}: NOT in schema")
            continue
        log(f"  {ep} parameters:")
        for p in op.get("parameters", []):
            sch = p.get("schema", {})
            extra = []
            if "default" in sch:
                extra.append(f"default={sch['default']!r}")
            if "enum" in sch:
                extra.append(f"enum={sch['enum']}")
            log(f"    - {p.get('name')} ({'required' if p.get('required') else 'optional'}) {' '.join(extra)}")
        desc = (op.get("description") or "").split("Response schema")[0]
        desc = " ".join(desc.split())[:400]
        if desc:
            log(f"    description: {desc}")
    return schema


def probe_solar() -> None:
    log("")
    log("=" * 78)
    log("1. CH SOLAR — /public_power?country=ch&subtype=solarlog")
    log("=" * 78)
    base = {"country": "ch", "subtype": "solarlog"}

    log(" a) One day and one full year")
    probe_call("solar_1day", "/public_power", {**base, "start": "2024-06-15", "end": "2024-06-15"})
    year = probe_call("solar_1year", "/public_power", {**base, "start": "2023-01-01", "end": "2023-12-31"})

    log(" b) History (single test days)")
    for d in ("2015-06-15", "2018-06-15", "2021-06-15"):
        probe_call(f"solar_hist_{d[:4]}", "/public_power", {**base, "start": d, "end": d})

    log(" c) Comparison baseline without subtype (to see what 'solarlog' changes)")
    probe_call("public_power_ch_default", "/public_power", {"country": "ch", "start": "2024-06-15", "end": "2024-06-15"})

    log("")
    if is_ok(year):
        log("  -> A full year works in one call: full 2015-2026 solar pull is ~12 calls, ~6 min.")
    else:
        log("  -> Full-year call failed; pull month by month instead (~140 calls, ~70 min for 2015-2026).")


def probe_frequency() -> None:
    log("")
    log("=" * 78)
    log("2. GRID FREQUENCY — /frequency (default region)")
    log("=" * 78)

    log(" a) History (single test days)")
    hist = {}
    for d in ("2015-06-15", "2018-06-15", "2021-01-04", "2026-09-01"):
        hist[d] = probe_call(f"freq_hist_{d[:4]}", "/frequency", {"start": d, "end": d})

    log(" b) Largest range per call (stops at the first failure)")
    start = date(2026, 8, 1)
    ranges = {1: None, 7: None, 31: None}
    best_days, best_row = 0, None
    for days in ranges:
        end = start + timedelta(days=days - 1)
        row = probe_call(f"freq_range_{days}d", "/frequency",
                         {"start": start.isoformat(), "end": end.isoformat()})
        # A call "works" only if it returns the whole requested span, not a truncated slice
        covered = False
        if is_ok(row):
            span_days = (pd.Timestamp(row["last_utc"]) - pd.Timestamp(row["first_utc"])).total_seconds() / 86400
            covered = span_days >= days - 1.05
            if not covered:
                log(f"    -> returned only {span_days:.2f} days of the requested {days} (truncated)")
        if covered:
            best_days, best_row = days, row
        else:
            break

    log("")
    log(" c) Full-pull estimate")
    if not best_row:
        log("  -> No frequency range call succeeded; cannot estimate. Check the errors above.")
        return

    res_s = best_row["resolution_s"]
    secs_per_call = best_row["seconds"]
    mb_per_day = best_row["bytes"] / 1e6 / best_days
    points_per_day = best_row["n_points"] / best_days
    log(f"  Resolution {res_s:.0f} s, ~{points_per_day:,.0f} points/day, "
        f"~{mb_per_day:.1f} MB JSON/day, max tested range per call: {best_days} days "
        f"(larger ranges untested{'' if best_days < 31 else ' — may work too'}).")

    for label, (a, b) in PULL_WINDOWS.items():
        n_days = (b - a).days
        n_calls = math.ceil(n_days / best_days)
        # each call costs max(rate-limit gap, download time)
        hours = n_calls * max(MIN_GAP_S, secs_per_call) / 3600
        raw_gb = n_days * mb_per_day / 1000
        parquet_gb = n_days * points_per_day * 12 / 1e9  # ~timestamp + float, compressed-ish upper bound
        log(f"  {label}: {n_days} days -> {n_calls:,} calls -> ~{hours:.1f} h, "
            f"~{raw_gb:.1f} GB raw JSON, ~{parquet_gb:.1f} GB as parquet (upper bound)")

    first_ok = [d for d, r in hist.items() if is_ok(r)]
    if first_ok:
        log(f"  History: data returned for test days {first_ok}; earliest confirmed {first_ok[0]}.")
    else:
        log("  History: none of the test days returned data.")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-solar", action="store_true")
    ap.add_argument("--skip-frequency", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Energy-Charts probe — {datetime.now():%Y-%m-%d %H:%M}")

    probe_schema()
    if not args.skip_solar:
        probe_solar()
    if not args.skip_frequency:
        probe_frequency()

    pd.DataFrame(_rows).to_csv(OUT_DIR / "energycharts_probe_results.csv", index=False)
    (OUT_DIR / "energycharts_probe_report.txt").write_text("\n".join(_report) + "\n", encoding="utf-8")
    log("")
    log(f"Written: {OUT_DIR}/energycharts_probe_report.txt, energycharts_probe_results.csv, energycharts_openapi.json")


if __name__ == "__main__":
    main()
