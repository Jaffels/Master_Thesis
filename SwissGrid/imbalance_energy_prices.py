"""
Pipeline 01: aFRR/mFRR Imbalance Energy Prices
Source:  Swissgrid — Monthly XLSX per quarter-hour
URL:     https://www.swissgrid.ch/en/home/customers/topics/bgm/balance-energy.html
Format:  .xlsx (and .xml), monthly publication, 15-min resolution
Thesis:  TARGET VARIABLE — positive & negative imbalance price forecasting
"""

import time, logging
import requests
import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/imbalance_prices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.swissgrid.ch"

# Monthly XLSX URL pattern (Swissgrid naming convention)
XLSX_TMPL = (
    "{base}/dam/swissgrid/customers/topics/bgm/{year}/AE_{year}{month:02d}.xlsx"
)

HEADERS = {"User-Agent": "SwissgridThesisBot/1.0 (thesis research; contact@uni.ch)"}

# Column name normalization: raw Swissgrid names → thesis schema
COL_MAP = {
    # Date/time columns (vary by file vintage)
    "Date":                               "date_raw",
    "Datum":                              "date_raw",
    "Time":                               "time_raw",
    "Zeit":                               "time_raw",
    # Price columns — both EUR/MWh (current) and EURct/kWh (legacy pre-2022)
    "Positive Imbalance Price [EUR/MWh]": "price_pos_eur_mwh",
    "Negative Imbalance Price [EUR/MWh]": "price_neg_eur_mwh",
    "Positive Ausgleichsenergie [EURct/kWh]": "price_pos_raw",   # needs ×10 conversion
    "Negative Ausgleichsenergie [EURct/kWh]": "price_neg_raw",
    "Positive Imbalance Price [EURct/kWh]":   "price_pos_raw",
    "Negative Imbalance Price [EURct/kWh]":   "price_neg_raw",
    # Volume columns (present in newer files)
    "aFRR pos. [MW]":  "afrr_pos_mw",
    "aFRR neg. [MW]":  "afrr_neg_mw",
    "mFRR pos. [MW]":  "mfrr_pos_mw",
    "mFRR neg. [MW]":  "mfrr_neg_mw",
    "Net Position [MW]": "net_position_mw",
}


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
def build_url(year: int, month: int) -> str:
    return XLSX_TMPL.format(base=BASE_URL, year=year, month=month)


def download_month(year: int, month: int, retries: int = 3) -> pd.DataFrame | None:
    """Download one monthly XLSX from Swissgrid and parse it."""
    url = build_url(year, month)
    log.info(f"Fetching {year}-{month:02d} → {url}")
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            # Cache raw bytes locally
            (OUTPUT_DIR / f"raw_AE_{year}{month:02d}.xlsx").write_bytes(resp.content)
            return _parse_xlsx(BytesIO(resp.content), year, month)
        except requests.HTTPError as e:
            if resp.status_code == 404:
                log.warning(f"  404 – file not published yet for {year}-{month:02d}")
                return None
            log.warning(f"  HTTP {resp.status_code}, attempt {attempt+1}/{retries}: {e}")
            time.sleep(2 ** attempt)
    log.error(f"  Giving up on {year}-{month:02d} after {retries} retries")
    return None


def load_local(path: Path, year: int, month: int) -> pd.DataFrame | None:
    """Parse a manually-downloaded XLSX file (useful when behind a VPN/paywall)."""
    log.info(f"Loading local: {path}")
    with open(path, "rb") as f:
        return _parse_xlsx(f, year, month)


# ── PARSE ─────────────────────────────────────────────────────────────────────
def _parse_xlsx(file_obj, year: int, month: int) -> pd.DataFrame | None:
    """
    Parse Swissgrid imbalance XLSX.
    Handles two sheet-name conventions: 'AE' (current) or sheet index 0 (legacy).
    Also handles two unit formats: EUR/MWh (post-2022) vs EURct/kWh (pre-2022).
    """
    # Try named sheet first, fall back to sheet 0
    for sheet in ("AE", 0):
        try:
            raw = pd.read_excel(file_obj, sheet_name=sheet, header=None)
            break
        except Exception:
            continue
    else:
        log.error(f"  Cannot open {year}-{month:02d}: no readable sheet")
        return None

    # Detect header row (first row with date/time keyword)
    header_row = next(
        (i for i, row in raw.iterrows()
         if any(str(c).strip().lower() in ("date","datum","time","zeit") for c in row)),
        None
    )
    if header_row is None:
        log.warning(f"  No header row found in {year}-{month:02d}")
        return None

    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row]]
    df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns}, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Build unified timestamp ──────────────────────────────────────────────
    if "date_raw" in df.columns and "time_raw" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["date_raw"].astype(str) + " " + df["time_raw"].astype(str),
            errors="coerce", dayfirst=True
        )
        df.drop(columns=["date_raw", "time_raw"], inplace=True)
    elif "date_raw" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date_raw"], errors="coerce", dayfirst=True)
        df.drop(columns=["date_raw"], inplace=True)
    df.dropna(subset=["timestamp"], inplace=True)

    # ── Unit conversion: EURct/kWh → EUR/MWh (multiply by 10) ───────────────
    # Heuristic: if the column is still named price_pos_raw, it came from a
    # legacy ct/kWh column header and needs conversion.
    for raw_col, final_col in [("price_pos_raw","price_pos_eur_mwh"),
                                ("price_neg_raw","price_neg_eur_mwh")]:
        if raw_col in df.columns:
            df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
            log.info(f"  Converting {raw_col} ct/kWh → EUR/MWh (×10)")
            df[final_col] = df[raw_col] * 10
            df.drop(columns=[raw_col], inplace=True)

    # ── Numeric coercion for remaining columns ───────────────────────────────
    num_cols = ["price_pos_eur_mwh","price_neg_eur_mwh",
                "afrr_pos_mw","afrr_neg_mw","mfrr_pos_mw","mfrr_neg_mw","net_position_mw"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("timestamp").reset_index(drop=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(
    start: date = date(2020, 1, 1),
    end: date | None = None,
    local_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Collect Swissgrid imbalance prices for a date range.

    Parameters
    ----------
    start     : First month (default Jan 2020)
    end       : Last month  (default: last completed month)
    local_dir : If provided, look for pre-downloaded files named AE_YYYYMM.xlsx

    Returns
    -------
    DataFrame columns:
        timestamp (CET/CEST), price_pos_eur_mwh, price_neg_eur_mwh,
        afrr_pos_mw, afrr_neg_mw, mfrr_pos_mw, mfrr_neg_mw, net_position_mw
    """
    if end is None:
        today = date.today()
        end = today.replace(day=1) - relativedelta(months=1)

    log.info(f"=== Pipeline 01: Imbalance Prices {start} → {end} ===")
    frames, current = [], start.replace(day=1)

    while current <= end:
        y, m = current.year, current.month
        df = None

        # Prefer local files (faster, no rate-limit risk)
        if local_dir:
            p = Path(local_dir) / f"AE_{y}{m:02d}.xlsx"
            if p.exists():
                df = load_local(p, y, m)

        if df is None:
            df = download_month(y, m)
            time.sleep(1.0)  # polite crawl delay (1 req/sec)

        if df is not None:
            frames.append(df)
        current += relativedelta(months=1)

    if not frames:
        log.warning("No data collected — check URL pattern or supply --local-dir")
        return pd.DataFrame()

    out = (pd.concat(frames, ignore_index=True)
             .sort_values("timestamp")
             .drop_duplicates(subset=["timestamp"], keep="last")
             .reset_index(drop=True))

    log.info(f"Records: {len(out):,} | Range: {out['timestamp'].min()} → {out['timestamp'].max()}")
    log.info(f"Missing prices:\n{out[['price_pos_eur_mwh','price_neg_eur_mwh']].isna().sum()}")

    out.to_parquet(OUTPUT_DIR / "imbalance_prices.parquet", index=False)
    out.to_csv(OUTPUT_DIR / "imbalance_prices.csv", index=False)
    log.info(f"Saved → {OUTPUT_DIR}/imbalance_prices.{{parquet,csv}}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start",     default="2020-01", help="YYYY-MM")
    p.add_argument("--end",       default=None,      help="YYYY-MM (default: last completed month)")
    p.add_argument("--local-dir", default=None,      help="Folder with pre-downloaded AE_YYYYMM.xlsx")
    args = p.parse_args()

    df = run(
        start=datetime.strptime(args.start, "%Y-%m").date(),
        end=datetime.strptime(args.end, "%Y-%m").date() if args.end else None,
        local_dir=Path(args.local_dir) if args.local_dir else None,
    )
    print(df.head(10).to_string())
    print(df.dtypes)