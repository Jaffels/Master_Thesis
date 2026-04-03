"""
Pipeline 04: Day-Ahead Spot Prices — SwissIX / ENTSO-E
Source:  ENTSO-E Transparency Platform (REST API)
         https://transparency.entsoe.eu
         Bidding zone: 10YCH-SWISSGRIDZ (Switzerland)
Format:  XML via REST API, parsed to DataFrame
Thesis:  aFRR energy price benchmark / exogenous feature for price forecasting model

SETUP:
    1. Register free account at https://transparency.entsoe.eu
    2. Email transparency@entsoe.eu with subject "Restful API access" to get API key
    3. Pass your key via --api-key argument or ENTSOE_API_KEY env variable
    4. pip install entsoe-py
"""

import os, time, logging
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timezone
from dateutil.relativedelta import relativedelta

try:
    from entsoe import EntsoePandasClient
    ENTSOE_PY = True
except ImportError:
    ENTSOE_PY = False
    logging.warning("entsoe-py not installed. pip install entsoe-py")

# Fallback: raw requests if entsoe-py unavailable
import requests
from io import StringIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/dayahead")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWITZERLAND_ZONE = "10YCH-SWISSGRIDZ"     # EIC code for Switzerland
ENTSOE_BASE      = "https://web-api.tp.entsoe.eu/api"


# ── via entsoe-py (preferred) ─────────────────────────────────────────────────
def _fetch_entsoe_py(
    api_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    country_code: str = SWITZERLAND_ZONE,
) -> pd.Series | None:
    """
    Fetch day-ahead prices using the entsoe-py client.
    Returns a Series indexed by UTC timestamp with EUR/MWh values.
    """
    client = EntsoePandasClient(api_key=api_key)
    try:
        prices = client.query_day_ahead_prices(
            country_code=country_code,
            start=start,
            end=end,
        )
        log.info(f"  entsoe-py: {len(prices)} hourly prices fetched")
        return prices
    except Exception as e:
        log.error(f"  entsoe-py query failed: {e}")
        return None


# ── via raw REST API (fallback) ───────────────────────────────────────────────
def _fetch_raw_api(
    api_key: str,
    start: date,
    end: date,
    zone: str = SWITZERLAND_ZONE,
) -> pd.DataFrame | None:
    """
    Fallback: directly call ENTSO-E REST API (Document type A44 — Day-ahead prices).
    """
    HEADERS = {"User-Agent": "SwissgridThesisBot/1.0 (thesis; contact@uni.ch)"}
    # ENTSO-E expects UTC timestamps in format YYYYMMDDHHSS
    start_str = start.strftime("%Y%m%d0000")
    end_str   = end.strftime("%Y%m%d2300")

    params = {
        "securityToken": api_key,
        "documentType":  "A44",      # Day-ahead prices
        "in_Domain":     zone,
        "out_Domain":    zone,
        "periodStart":   start_str,
        "periodEnd":     end_str,
    }
    log.info(f"  REST API: {start} → {end}")
    try:
        r = requests.get(ENTSOE_BASE, params=params,
                         headers=HEADERS, timeout=60)
        r.raise_for_status()
    except Exception as e:
        log.error(f"  REST request failed: {e}")
        return None

    # Parse XML response
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.content)
        ns   = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
        rows = []
        for ts in root.findall(".//ns:TimeSeries", ns):
            period = ts.find("ns:Period", ns)
            if period is None:
                continue
            start_el = period.find("ns:timeInterval/ns:start", ns)
            res_el   = period.find("ns:resolution", ns)
            if start_el is None or res_el is None:
                continue
            period_start = pd.Timestamp(start_el.text, tz="UTC")
            resolution   = pd.Timedelta(res_el.text.replace("PT","").replace("H","h").replace("M","min"))
            for pt in period.findall("ns:Point", ns):
                pos   = int(pt.find("ns:position", ns).text)
                price = float(pt.find("ns:price.amount", ns).text)
                ts_val = period_start + (pos - 1) * resolution
                rows.append({"timestamp": ts_val, "price_eur_mwh": price})
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        log.info(f"  XML parsed: {len(df)} rows")
        return df
    except Exception as e:
        log.error(f"  XML parse error: {e}")
        return None


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(
    api_key: str | None = None,
    start: date = date(2020, 1, 1),
    end: date | None = None,
    zone: str = SWITZERLAND_ZONE,
    chunk_months: int = 3,       # Split into chunks to avoid API timeout
) -> pd.DataFrame:
    """
    Fetch day-ahead spot prices for Switzerland from ENTSO-E.

    Parameters
    ----------
    api_key       : ENTSO-E API token (or set ENTSOE_API_KEY env var)
    start         : First date to fetch
    end           : Last date (default: yesterday)
    zone          : EIC bidding zone code (default: Switzerland 10YCH-SWISSGRIDZ)
    chunk_months  : Months per API call (API has a 1-year limit per request)

    Returns
    -------
    DataFrame columns: timestamp (UTC), price_eur_mwh
    """
    api_key = api_key or os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key. Set ENTSOE_API_KEY env var or pass --api-key.\n"
            "Register at https://transparency.entsoe.eu and email transparency@entsoe.eu"
        )
    if end is None:
        end = date.today() - relativedelta(days=1)

    log.info(f"=== Pipeline 04: Day-Ahead Prices {start} → {end} | zone={zone} ===")
    frames = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(
            (chunk_start + relativedelta(months=chunk_months) - relativedelta(days=1)),
            end
        )

        if ENTSOE_PY:
            ts_start = pd.Timestamp(chunk_start, tz="UTC")
            ts_end   = pd.Timestamp(chunk_end + relativedelta(days=1), tz="UTC")
            series   = _fetch_entsoe_py(api_key, ts_start, ts_end, zone)
            if series is not None:
                df = series.reset_index()
                df.columns = ["timestamp", "price_eur_mwh"]
                frames.append(df)
        else:
            df = _fetch_raw_api(api_key, chunk_start, chunk_end, zone)
            if df is not None:
                frames.append(df)

        log.info(f"  Chunk {chunk_start} → {chunk_end} done")
        chunk_start = chunk_end + relativedelta(days=1)
        time.sleep(1.0)   # ENTSO-E rate limit: max 400 req/hour

    if not frames:
        log.warning("No day-ahead data retrieved.")
        return pd.DataFrame()

    out = (pd.concat(frames, ignore_index=True)
             .sort_values("timestamp")
             .drop_duplicates(subset=["timestamp"], keep="last")
             .reset_index(drop=True))

    # Resample to 15-min to align with Swissgrid 15-min resolution
    # (day-ahead prices are hourly — forward-fill within each hour)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out_15 = (out.set_index("timestamp")
                 .resample("15min")
                 .ffill()
                 .reset_index())
    out_15.rename(columns={"price_eur_mwh": "dayahead_price_eur_mwh"}, inplace=True)

    log.info(f"Records: {len(out_15):,} | Range: {out_15['timestamp'].min()} → {out_15['timestamp'].max()}")

    out.to_parquet(OUTPUT_DIR / "dayahead_hourly.parquet", index=False)
    out_15.to_parquet(OUTPUT_DIR / "dayahead_15min.parquet", index=False)
    out_15.to_csv(OUTPUT_DIR / "dayahead_15min.csv", index=False)
    log.info(f"Saved → {OUTPUT_DIR}/dayahead_15min.{{parquet,csv}}")
    return out_15


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--api-key",      default=None,          help="ENTSO-E API token")
    p.add_argument("--start",        default="2020-01-01",  help="YYYY-MM-DD")
    p.add_argument("--end",          default=None,          help="YYYY-MM-DD")
    p.add_argument("--zone",         default=SWITZERLAND_ZONE)
    p.add_argument("--chunk-months", type=int, default=3)
    args = p.parse_args()

    df = run(
        api_key=args.api_key,
        start=datetime.strptime(args.start, "%Y-%m-%d").date(),
        end=datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None,
        zone=args.zone,
        chunk_months=args.chunk_months,
    )
    print(df.head(10).to_string())
    print(df.dtypes)