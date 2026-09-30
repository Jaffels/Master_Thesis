"""
energycharts_pull.py — production pull from the Energy-Charts API (Fraunhofer ISE).
Data licence: CC BY 4.0, attribution "Energy-Charts.info" required.

Two tasks
  solar      /public_power?country=ch&subtype=solarlog, one call per year (5-min, MW).
             Years that return 404 are logged as 'empty' -> the first 'ok' year is the
             series start. Also pulls the default CH /public_power mix (hourly) per year
             for comparison with the ENTSO-E / Swissgrid solar series.
  frequency  /frequency (1-s, measured in Freiburg, RG Continental Europe; available
             from 1 May 2022). Max 3 days per call -> 3-day windows. Each window is
             aggregated on the fly to 15-min and 4-h (Europe/Zurich) features; the raw
             1-s data is kept as parquet per window unless --no-raw.

Rate limiting: 33 s between calls to the same endpoint (API limit 2 req/min + 2 s margin),
HTTP 429 honours Retry-After, timeouts / 5xx retried with backoff.

Resume: every request is recorded in _pull_manifest.csv. Items with status 'ok'
(file present) or 'empty' are skipped on re-run; 'error' items are retried.
Safe to stop with Ctrl-C and restart.

Run from the thesis root (needs requests, pandas, pyarrow, numpy):
    python EnergyCharts/energycharts_pull.py --dry-run          # plan + ETA, no requests
    python EnergyCharts/energycharts_pull.py                    # solar, then frequency
    python EnergyCharts/energycharts_pull.py --task solar
    python EnergyCharts/energycharts_pull.py --task frequency --start 2022-05-01 --end 2026-08-31
    python EnergyCharts/energycharts_pull.py --combine-only     # rebuild combined files

Writes only below EnergyCharts/Data/ (atomic writes, never deletes):
    Data/_pull_manifest.csv
    Data/solar/ch_solarlog_<YYYY>.parquet             5-min, UTC index
    Data/solar/ch_public_power_<YYYY>.parquet         hourly CH mix, UTC index
    Data/frequency/raw/<YYYY>/freq_<start>_<end>.parquet   1-s (unless --no-raw)
    Data/frequency/agg/freq_agg_<start>_<end>.parquet      per-window 15-min + 4-h rows
    Data/frequency/frequency_15min.parquet             combined (after the pull)
    Data/frequency/frequency_4h.parquet                combined (after the pull)
    Data/solar/ch_solarlog_all.parquet                 combined (after the pull)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://api.energy-charts.info"
DATA = Path("EnergyCharts") / "Data"
MANIFEST = DATA / "_pull_manifest.csv"
LOCAL_TZ = "Europe/Zurich"

MIN_GAP_S = 33            # 2 req/min per endpoint + 2 s margin
TIMEOUT_S = 180
MAX_RETRIES = 4           # for 429 / 5xx / network errors
FREQ_START = date(2022, 5, 1)   # documented start of 1-s data
FREQ_WINDOW_DAYS = 3            # documented maximum timespan per call
SOLAR_FIRST_YEAR = 2021

# Frequency feature thresholds (mHz). FCR: ±10 mHz insensitivity band,
# full activation at ±200 mHz.
THRESHOLDS_MHZ = (10, 50, 100)
FCR_FULL_MHZ = 200.0

MANIFEST_COLS = ["task", "key", "status", "http", "n_points", "first_utc", "last_utc",
                 "file", "seconds", "message", "pulled_at"]

_last_call: dict[str, float] = {}


# ------------------------------------------------------------------------ helpers

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return s or "value"


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


def load_manifest() -> dict[tuple[str, str], dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return {(r["task"], r["key"]): r for r in csv.DictReader(f)}


def save_manifest(man: dict[tuple[str, str], dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        for k in sorted(man):
            w.writerow({c: man[k].get(c, "") for c in MANIFEST_COLS})
    tmp.replace(MANIFEST)


def is_done(man: dict, task: str, key: str) -> bool:
    r = man.get((task, key))
    if not r:
        return False
    if r["status"] == "empty":
        return True
    if r["status"] == "ok":
        return not r["file"] or Path(r["file"]).exists()
    return False


def record(man: dict, task: str, key: str, **kw) -> None:
    row = {"task": task, "key": key, "pulled_at": now_str()}
    row.update({k: ("" if v is None else v) for k, v in kw.items()})
    man[(task, key)] = row
    save_manifest(man)


# --------------------------------------------------------------------------- HTTP

def get_json(endpoint: str, params: dict) -> tuple[int | None, object, str, float]:
    """Throttled GET. Returns (http_status, json_or_None, message, seconds)."""
    for attempt in range(MAX_RETRIES + 1):
        wait = MIN_GAP_S - (time.time() - _last_call.get(endpoint, 0.0))
        if wait > 0:
            time.sleep(wait)
        t0 = time.time()
        try:
            r = requests.get(f"{BASE}{endpoint}", params=params, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            _last_call[endpoint] = time.time()
            msg = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                back = 60 * (attempt + 1)
                log(f"    network error ({msg[:80]}), retry in {back} s")
                time.sleep(back)
                continue
            return None, None, msg, time.time() - t0
        secs = time.time() - t0
        _last_call[endpoint] = time.time()

        if r.status_code == 429 and attempt < MAX_RETRIES:
            retry = int(r.headers.get("Retry-After", 60))
            log(f"    429 rate-limited, retry in {retry} s")
            time.sleep(retry)
            continue
        if r.status_code >= 500 and attempt < MAX_RETRIES:
            back = 60 * (attempt + 1)
            log(f"    HTTP {r.status_code}, retry in {back} s")
            time.sleep(back)
            continue
        if r.status_code != 200:
            return r.status_code, None, r.text[:300].replace("\n", " "), secs
        try:
            return 200, r.json(), "", secs
        except ValueError:
            return 200, None, "response is not JSON", secs
    return None, None, "retries exhausted", 0.0


# ------------------------------------------------------------------------ parsing

def to_frame(payload) -> pd.DataFrame:
    """v1 response -> DataFrame with UTC DatetimeIndex 'timestamp_utc'.
    Handles {"unix_seconds", "production_types": [{name, data}]} and
    {"unix_seconds", "data"} layouts."""
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict) or "unix_seconds" not in payload:
        return pd.DataFrame()
    idx = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    n = len(idx)
    cols: dict[str, list] = {}
    for pt in payload.get("production_types") or []:
        if isinstance(pt, dict) and isinstance(pt.get("data"), list) and len(pt["data"]) == n:
            cols[snake(str(pt.get("name")))] = pt["data"]
    for k, v in payload.items():
        if k in ("unix_seconds", "production_types"):
            continue
        if isinstance(v, list) and len(v) == n and n > 0:
            cols[snake(k)] = v
    df = pd.DataFrame(cols, index=idx).apply(pd.to_numeric, errors="coerce")
    df.index.name = "timestamp_utc"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# ------------------------------------------------------------------ frequency features

def frequency_features(f_hz: pd.Series, freq: str) -> pd.DataFrame:
    """Aggregate a 1-s frequency series (UTC index) into interval features.

    Intervals are labelled in Europe/Zurich local time (4-h FCR blocks 00-04, 04-08, ...).
    df = f - 50 Hz in mHz; FCR activation share a = clip(-df / 200 mHz, -1, 1)
    (positive = under-frequency = upward activation).
    Reusable for other 1-s sources (e.g. TSO downloads) with the same input format.
    """
    s = f_hz.dropna()
    s = s[(s > 45) & (s < 55)]          # drop obvious sensor glitches
    if s.empty:
        return pd.DataFrame()
    key = interval_key(s.index, freq)
    df = (s - 50.0) * 1000.0
    act = (-df / FCR_FULL_MHZ).clip(-1, 1)
    g_df, g_abs = df.groupby(key), df.abs().groupby(key)

    parts = {
        "n_seconds": g_df.size(),
        "mean_f_hz": s.groupby(key).mean(),
        "mean_df_mhz": g_df.mean(),
        "mean_abs_df_mhz": g_abs.mean(),
        "std_df_mhz": g_df.std(),
        "max_abs_df_mhz": g_abs.max(),
        "min_df_mhz": g_df.min(),
        "max_df_mhz": g_df.max(),
        # FCR energy proxies, in "hours at full FCR activation" per interval
        "fcr_up_equiv_h": act.clip(lower=0).groupby(key).sum() / 3600.0,
        "fcr_down_equiv_h": (-act.clip(upper=0)).groupby(key).sum() / 3600.0,
    }
    for t in THRESHOLDS_MHZ:
        parts[f"share_outside_{t}mhz"] = (df.abs() > t).astype(float).groupby(key).mean()
    out = pd.DataFrame(parts).sort_index()
    out.index.name = "interval_start_local"
    return out


def interval_key(idx_utc: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Interval start in Europe/Zurich wall-clock time.
    15min: plain floor (offsets are whole hours, so UTC and local floors agree).
    4h:    blocks 00-04, 04-08, ... on the local clock, also on DST days
           (the block containing the switch has 3 or 5 hours; see n_seconds)."""
    if freq == "15min":
        return idx_utc.floor("15min").tz_convert(LOCAL_TZ)
    if freq == "4h":
        wall = idx_utc.tz_convert(LOCAL_TZ).tz_localize(None)
        start = wall.floor("D") + pd.to_timedelta((wall.hour // 4) * 4, unit="h")
        return start.tz_localize(LOCAL_TZ)   # 00/04/08/... are never ambiguous or missing
    raise ValueError(freq)


# ------------------------------------------------------------------------- tasks

def solar_items(last_year: int) -> list[tuple[str, dict, str, Path]]:
    items = []
    for y in range(SOLAR_FIRST_YEAR, last_year + 1):
        s, e = f"{y}-01-01", f"{y}-12-31"
        items.append((f"solarlog_{y}", {"country": "ch", "subtype": "solarlog", "start": s, "end": e},
                      "/public_power", DATA / "solar" / f"ch_solarlog_{y}.parquet"))
        items.append((f"public_power_{y}", {"country": "ch", "start": s, "end": e},
                      "/public_power", DATA / "solar" / f"ch_public_power_{y}.parquet"))
    return items


def freq_windows(start: date, end: date) -> list[tuple[date, date]]:
    out, a = [], max(start, FREQ_START)
    while a <= end:
        b = min(a + timedelta(days=FREQ_WINDOW_DAYS - 1), end)
        out.append((a, b))
        a = b + timedelta(days=1)
    return out


def run_solar(man: dict, last_year: int) -> None:
    items = [i for i in solar_items(last_year) if not is_done(man, "solar", i[0])]
    log(f"SOLAR: {len(items)} requests to do (~{len(items) * MIN_GAP_S / 60:.0f} min)")
    for key, params, ep, path in items:
        log(f"  {key}")
        http, payload, msg, secs = get_json(ep, params)
        if http == 404:
            record(man, "solar", key, status="empty", http=404, message=msg, seconds=round(secs, 1))
            log("    -> empty (404, before series start)")
            continue
        if http != 200 or payload is None:
            record(man, "solar", key, status="error", http=http, message=msg, seconds=round(secs, 1))
            log(f"    -> ERROR {http}: {msg[:120]}")
            continue
        df = to_frame(payload)
        if df.empty:
            record(man, "solar", key, status="error", http=200, message="no time series in response")
            log("    -> ERROR: no time series in response")
            continue
        atomic_parquet(df, path)
        record(man, "solar", key, status="ok", http=200, n_points=len(df),
               first_utc=df.index.min().isoformat(), last_utc=df.index.max().isoformat(),
               file=str(path), seconds=round(secs, 1))
        log(f"    -> ok, {len(df):,} rows, {df.index.min()} -> {df.index.max()}, cols {list(df.columns)}")


def run_frequency(man: dict, start: date, end: date, keep_raw: bool) -> None:
    wins = [w for w in freq_windows(start, end) if not is_done(man, "frequency", f"{w[0]}_{w[1]}")]
    eta_h = len(wins) * (MIN_GAP_S + 2) / 3600
    log(f"FREQUENCY: {len(wins)} windows to do, ETA ~{eta_h:.1f} h "
        f"(finish ~{datetime.now() + timedelta(hours=eta_h):%a %H:%M})")
    t_start = time.time()
    for i, (a, b) in enumerate(wins, 1):
        key = f"{a}_{b}"
        http, payload, msg, secs = get_json("/frequency", {"start": a.isoformat(), "end": b.isoformat()})
        prefix = f"  [{i}/{len(wins)}] {key}"
        if http == 404:
            record(man, "frequency", key, status="empty", http=404, message=msg, seconds=round(secs, 1))
            log(f"{prefix} -> empty (404)")
            continue
        if http != 200 or payload is None:
            record(man, "frequency", key, status="error", http=http, message=msg, seconds=round(secs, 1))
            log(f"{prefix} -> ERROR {http}: {msg[:120]}")
            continue
        df = to_frame(payload)
        if df.empty or df.shape[1] == 0:
            record(man, "frequency", key, status="error", http=200, message="no time series in response")
            log(f"{prefix} -> ERROR: no time series in response")
            continue
        f = df.iloc[:, 0].rename("frequency_hz")

        if keep_raw:
            raw_path = DATA / "frequency" / "raw" / str(a.year) / f"freq_{a}_{b}.parquet"
            atomic_parquet(f.astype("float32").to_frame(), raw_path)

        q = frequency_features(f, "15min").assign(resolution="15min")
        h = frequency_features(f, "4h").assign(resolution="4h")
        agg_path = DATA / "frequency" / "agg" / f"freq_agg_{a}_{b}.parquet"
        atomic_parquet(pd.concat([q, h]), agg_path)

        w0 = pd.Timestamp(a).tz_localize(LOCAL_TZ)
        w1 = pd.Timestamp(b + timedelta(days=1)).tz_localize(LOCAL_TZ)
        expected = (w1 - w0).total_seconds()        # DST-aware (23 h / 25 h days)
        cov = len(f) / expected
        record(man, "frequency", key, status="ok", http=200, n_points=len(f),
               first_utc=f.index.min().isoformat(), last_utc=f.index.max().isoformat(),
               file=str(agg_path), seconds=round(secs, 1), message=f"coverage {cov:.4f}")
        done_rate = (time.time() - t_start) / i
        left = (len(wins) - i) * done_rate / 3600
        log(f"{prefix} -> ok, {len(f):,} s ({cov:.2%}), {secs:.1f} s, ~{left:.1f} h left")


# ----------------------------------------------------------------------- combine

def combine() -> None:
    agg_files = sorted((DATA / "frequency" / "agg").glob("freq_agg_*.parquet"))
    if agg_files:
        all_agg = pd.concat([pd.read_parquet(p) for p in agg_files])
        for res, name in (("15min", "frequency_15min.parquet"), ("4h", "frequency_4h.parquet")):
            part = all_agg[all_agg["resolution"] == res].drop(columns="resolution")
            part = part[~part.index.duplicated(keep="first")].sort_index()
            atomic_parquet(part, DATA / "frequency" / name)
            log(f"COMBINE: {name}: {len(part):,} rows, {part.index.min()} -> {part.index.max()}")

    solar_files = sorted((DATA / "solar").glob("ch_solarlog_[0-9]*.parquet"))
    if solar_files:
        s = pd.concat([pd.read_parquet(p) for p in solar_files]).sort_index()
        s = s[~s.index.duplicated(keep="first")]
        atomic_parquet(s, DATA / "solar" / "ch_solarlog_all.parquet")
        log(f"COMBINE: ch_solarlog_all.parquet: {len(s):,} rows, {s.index.min()} -> {s.index.max()}")


# -------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["all", "solar", "frequency"], default="all")
    ap.add_argument("--start", default=FREQ_START.isoformat(), help="frequency start (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-08-31", help="frequency end, inclusive (YYYY-MM-DD)")
    ap.add_argument("--no-raw", action="store_true", help="do not keep raw 1-s parquet files")
    ap.add_argument("--dry-run", action="store_true", help="print plan and ETA only")
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    last_year = end.year
    man = load_manifest()

    if args.combine_only:
        combine()
        return

    n_solar = sum(not is_done(man, "solar", k) for k, *_ in solar_items(last_year)) \
        if args.task in ("all", "solar") else 0
    n_freq = sum(not is_done(man, "frequency", f"{a}_{b}") for a, b in freq_windows(start, end)) \
        if args.task in ("all", "frequency") else 0
    log(f"Plan: {n_solar} solar calls, {n_freq} frequency calls "
        f"({start} -> {end}, {FREQ_WINDOW_DAYS}-day windows), gap {MIN_GAP_S} s, raw={'no' if args.no_raw else 'yes'}")
    log(f"ETA ~{(n_solar * MIN_GAP_S + n_freq * (MIN_GAP_S + 2)) / 3600:.1f} h")
    if args.dry_run:
        return

    try:
        if args.task in ("all", "solar"):
            run_solar(man, last_year)
        if args.task in ("all", "frequency"):
            run_frequency(man, start, end, keep_raw=not args.no_raw)
    except KeyboardInterrupt:
        log("Interrupted — manifest is saved; re-run the same command to resume.")
        sys.exit(1)

    combine()
    errs = [r for r in man.values() if r["status"] == "error"]
    log(f"DONE. Errors: {len(errs)}" + (" -> re-run the same command to retry them." if errs else ""))


if __name__ == "__main__":
    main()
