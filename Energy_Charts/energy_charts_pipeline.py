"""
Energy-Charts API Pipeline
===========================
Downloads historical data (2016-01-01 → 2026-05-31) for all available
countries and bidding zones across all supported endpoints.

Endpoints collected:
  - /public_power   → production by type (country-level)
  - /price          → day-ahead spot prices (bidding-zone-level)
  - /cbet           → cross-border electricity trading (country-level)
  - /cbpf           → cross-border physical flows (country-level)
  - /installed_power → installed capacity (country-level, yearly)

Rate limiting: 1 request every 2 seconds + exponential backoff on errors.
Data is chunked in 90-day windows to keep responses manageable.
Output: parquet files under ./Energy_Charts/<endpoint>/<key>.parquet
"""

import time
import logging
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL      = "https://api.energy-charts.info"
OUTPUT_DIR    = Path("Energy_Charts")
START_DATE    = date(2016, 1, 1)
END_DATE      = date(2026, 5, 31)
CHUNK_DAYS    = 90          # days per request window
SLEEP_BETWEEN = 2.0         # seconds between requests (conservative)
MAX_RETRIES   = 5
BACKOFF_BASE  = 10          # seconds; doubles on each retry

# All countries supported by the API
ALL_COUNTRIES = [
    "de", "ch", "eu", "all",
    "al", "am", "at", "az", "ba", "be", "bg", "by",
    "cy", "cz", "dk", "ee", "es", "fi", "fr", "ge",
    "gr", "hr", "hu", "ie", "it", "lt", "lu", "lv",
    "md", "me", "mk", "mt", "nie", "nl", "no", "pl",
    "pt", "ro", "rs", "ru", "se", "si", "sk", "tr",
    "ua", "uk", "xk",
]

# Bidding zones for /price endpoint
ALL_BZN = [
    "AT", "BE", "BG", "CH", "CZ",
    "DE-LU", "DE-AT-LU",
    "DK1", "DK2",
    "EE", "ES", "FI", "FR", "GR", "HR", "HU",
    "IT-Calabria", "IT-Centre-North", "IT-Centre-South",
    "IT-North", "IT-SACOAC", "IT-SACODC",
    "IT-Sardinia", "IT-Sicily", "IT-South",
    "LT", "LV", "ME", "NL",
    "NO1", "NO2", "NO2NSL", "NO3", "NO4", "NO5",
    "PL", "PT", "RO", "RS",
    "SE1", "SE2", "SE3", "SE4",
    "SI", "SK",
]

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── HTTP helper ───────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update({"Accept": "application/json"})

def get(endpoint: str, params: dict) -> Optional[dict]:
    """GET with retry + exponential backoff. Returns None on permanent failure."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                # No data for this combination – not an error
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                if resp.status_code == 429:
                    log.warning("429 response headers: %s", dict(resp.headers))
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                log.warning("HTTP %s on attempt %d/%d – retrying in %ds",
                            resp.status_code, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                continue
            # 422 or other client error – skip silently
            log.debug("HTTP %s for %s %s – skipping", resp.status_code, endpoint, params)
            return None
        except requests.RequestException as exc:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            log.warning("Request error attempt %d/%d: %s – retrying in %ds",
                        attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)
    log.error("Permanent failure for %s %s after %d attempts", endpoint, params, MAX_RETRIES)
    return None

# ── Date chunking ─────────────────────────────────────────────────────────────

def date_chunks(start: date, end: date, chunk_days: int):
    """Yield (chunk_start, chunk_end) pairs of at most chunk_days each."""
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)

# ── Parquet helpers ───────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Append to an existing parquet file or create a new one."""
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pq.read_table(path)
        combined = pa.concat_tables([existing, pa.Table.from_pandas(df, preserve_index=False)])
        # Deduplicate on timestamp + any key columns
        combined_df = combined.to_pandas()
        ts_cols = [c for c in combined_df.columns if "timestamp" in c or c == "date"]
        if ts_cols:
            combined_df = combined_df.drop_duplicates(subset=ts_cols).sort_values(ts_cols)
        pq.write_table(pa.Table.from_pandas(combined_df, preserve_index=False), path)
    else:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

# ── Response parsers ──────────────────────────────────────────────────────────

def parse_production(data: dict, key: str) -> pd.DataFrame:
    """Parse /public_power and /total_power responses into a wide DataFrame."""
    if not data or not data.get("unix_seconds") or not data.get("production_types"):
        return pd.DataFrame()
    timestamps = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)
    df = pd.DataFrame({"timestamp_utc": timestamps})
    for pt in data["production_types"]:
        name = pt["name"].replace(" ", "_").replace("/", "_").lower()
        df[name] = pt["data"]
    df.insert(0, "key", key)
    return df

def parse_price(data: dict, bzn: str) -> pd.DataFrame:
    """Parse /price response."""
    if not data or not data.get("unix_seconds") or not data.get("price"):
        return pd.DataFrame()
    df = pd.DataFrame({
        "bzn": bzn,
        "timestamp_utc": pd.to_datetime(data["unix_seconds"], unit="s", utc=True),
        "price_eur_mwh": data["price"],
    })
    return df

def parse_cross_border(data: dict, key: str) -> pd.DataFrame:
    """Parse /cbet and /cbpf responses."""
    if not data or not data.get("unix_seconds") or not data.get("countries"):
        return pd.DataFrame()
    timestamps = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)
    df = pd.DataFrame({"timestamp_utc": timestamps})
    for c in data["countries"]:
        col = c["name"].replace(" ", "_").replace("/", "_").lower()
        df[col] = c["data"]
    df.insert(0, "key", key)
    return df

def parse_installed(data: dict, key: str) -> pd.DataFrame:
    """Parse /installed_power response (yearly, no date chunking needed)."""
    if not data or not data.get("time") or not data.get("production_types"):
        return pd.DataFrame()
    df = pd.DataFrame({"time": data["time"]})
    for pt in data["production_types"]:
        name = pt["name"].replace(" ", "_").replace("/", "_").lower()
        df[name] = pt["data"]
    df.insert(0, "key", key)
    return df

# ── Endpoint collectors ───────────────────────────────────────────────────────

def collect_public_power(country: str) -> None:
    out = OUTPUT_DIR / "public_power" / f"{country}.parquet"
    frames = []
    for s, e in date_chunks(START_DATE, END_DATE, CHUNK_DAYS):
        data = get("/public_power", {"country": country, "start": str(s), "end": str(e)})
        time.sleep(SLEEP_BETWEEN)
        df = parse_production(data, country) if data else pd.DataFrame()
        if not df.empty:
            frames.append(df)
    if frames:
        save_parquet(pd.concat(frames, ignore_index=True), out)
        log.info("  ✓ public_power/%s  → %d rows", country, sum(len(f) for f in frames))
    else:
        log.debug("  – public_power/%s  → no data", country)


def collect_price(bzn: str) -> None:
    out = OUTPUT_DIR / "price" / f"{bzn.replace('-', '_')}.parquet"
    frames = []
    for s, e in date_chunks(START_DATE, END_DATE, CHUNK_DAYS):
        data = get("/price", {"bzn": bzn, "start": str(s), "end": str(e)})
        time.sleep(SLEEP_BETWEEN)
        df = parse_price(data, bzn) if data else pd.DataFrame()
        if not df.empty:
            frames.append(df)
    if frames:
        save_parquet(pd.concat(frames, ignore_index=True), out)
        log.info("  ✓ price/%s  → %d rows", bzn, sum(len(f) for f in frames))
    else:
        log.debug("  – price/%s  → no data", bzn)


def collect_cbet(country: str) -> None:
    out = OUTPUT_DIR / "cbet" / f"{country}.parquet"
    frames = []
    for s, e in date_chunks(START_DATE, END_DATE, CHUNK_DAYS):
        data = get("/cbet", {"country": country, "start": str(s), "end": str(e)})
        time.sleep(SLEEP_BETWEEN)
        df = parse_cross_border(data, country) if data else pd.DataFrame()
        if not df.empty:
            frames.append(df)
    if frames:
        save_parquet(pd.concat(frames, ignore_index=True), out)
        log.info("  ✓ cbet/%s  → %d rows", country, sum(len(f) for f in frames))
    else:
        log.debug("  – cbet/%s  → no data", country)


def collect_cbpf(country: str) -> None:
    out = OUTPUT_DIR / "cbpf" / f"{country}.parquet"
    frames = []
    for s, e in date_chunks(START_DATE, END_DATE, CHUNK_DAYS):
        data = get("/cbpf", {"country": country, "start": str(s), "end": str(e)})
        time.sleep(SLEEP_BETWEEN)
        df = parse_cross_border(data, country) if data else pd.DataFrame()
        if not df.empty:
            frames.append(df)
    if frames:
        save_parquet(pd.concat(frames, ignore_index=True), out)
        log.info("  ✓ cbpf/%s  → %d rows", country, sum(len(f) for f in frames))
    else:
        log.debug("  – cbpf/%s  → no data", country)


def collect_installed_power(country: str) -> None:
    """Installed power is yearly – fetch once per country, no chunking needed."""
    out = OUTPUT_DIR / "installed_power" / f"{country}.parquet"
    data = get("/installed_power", {"country": country, "time_step": "yearly"})
    time.sleep(SLEEP_BETWEEN)
    df = parse_installed(data, country) if data else pd.DataFrame()
    if not df.empty:
        save_parquet(df, out)
        log.info("  ✓ installed_power/%s  → %d rows", country, len(df))
    else:
        log.debug("  – installed_power/%s  → no data", country)


# ── Progress tracking ─────────────────────────────────────────────────────────

PROGRESS_FILE = OUTPUT_DIR / ".progress.txt"

def load_done() -> set:
    if PROGRESS_FILE.exists():
        return set(PROGRESS_FILE.read_text().splitlines())
    return set()

def mark_done(task: str) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_FILE.open("a") as f:
        f.write(task + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    done = load_done()

    # Build full task list
    tasks = []
    for country in ALL_COUNTRIES:
        tasks.append(("public_power", country, collect_public_power))
        tasks.append(("cbet",         country, collect_cbet))
        tasks.append(("cbpf",         country, collect_cbpf))
        tasks.append(("installed",    country, collect_installed_power))
    for bzn in ALL_BZN:
        tasks.append(("price", bzn, collect_price))

    total = len(tasks)
    log.info("Pipeline start: %d tasks  (%s → %s)  chunk=%dd  sleep=%.1fs",
             total, START_DATE, END_DATE, CHUNK_DAYS, SLEEP_BETWEEN)

    for i, (endpoint, key, fn) in enumerate(tasks, 1):
        task_id = f"{endpoint}::{key}"
        if task_id in done:
            log.info("[%d/%d] SKIP  %s::%s (already done)", i, total, endpoint, key)
            continue
        log.info("[%d/%d] START %s::%s", i, total, endpoint, key)
        try:
            fn(key)
            mark_done(task_id)
        except Exception as exc:
            log.error("FAILED %s::%s — %s", endpoint, key, exc)

    log.info("Pipeline complete. Data saved to: %s/", OUTPUT_DIR)

if __name__ == "__main__":
    main()
