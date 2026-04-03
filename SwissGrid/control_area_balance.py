"""
Pipeline 03: Control Area Balance & Activation Volumes
Source:  Swissgrid Energy Overview / Energy Statistic Switzerland
         https://www.swissgrid.ch/en/home/customers/topics/energy-data-ch.html
Format:  CSV + XLSX (daily and yearly), 15-min resolution
Thesis:  Response time & volume modeling — aFRR/mFRR activation volumes,
         net control area position, imbalance netting
"""

import time, logging
import requests
import pandas as pd
from io import BytesIO, StringIO
from pathlib import Path
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/control_area")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.swissgrid.ch"
HEADERS  = {"User-Agent": "SwissgridThesisBot/1.0 (thesis; contact@uni.ch)"}

# ── Direct download URLs ──────────────────────────────────────────────────────
# Yearly CSV (confirmed from Swissgrid page, updated annually)
YEARLY_CSV_TMPL  = BASE_URL + "/dam/swissgrid/customers/topics/energy-data-ch/{year}/control_area_balance_{year}.csv"
YEARLY_XLSX_TMPL = BASE_URL + "/dam/swissgrid/customers/topics/energy-data-ch/{year}/control_area_balance_{year}.xlsx"

# Total System Imbalance (real-time indicator, published daily)
TSI_CSV_URL  = BASE_URL + "/dam/swissgrid/customers/topics/energy-data-ch/total_system_imbalance.csv"
TSI_XLSX_URL = BASE_URL + "/dam/swissgrid/customers/topics/energy-data-ch/total_system_imbalance.xlsx"

# Control Area Balance (daily rolling file)
CAB_CSV_URL  = BASE_URL + "/dam/swissgrid/customers/topics/energy-data-ch/control_area_balance.csv"
CAB_XLSX_URL = BASE_URL + "/dam/swissgrid/customers/topics/energy-data-ch/control_area_balance.xlsx"

COL_MAP = {
    "Date":                       "date_raw",
    "Datum":                      "date_raw",
    "Time":                       "time_raw",
    "Zeit":                       "time_raw",
    "Timestamp":                  "timestamp",
    "Net Position [MW]":          "net_position_mw",
    "Netto-Position [MW]":        "net_position_mw",
    "Control Area Balance [MW]":  "control_area_balance_mw",
    "aFRR Activation positive [MW]": "afrr_act_pos_mw",
    "aFRR Activation negative [MW]": "afrr_act_neg_mw",
    "mFRR Activation positive [MW]": "mfrr_act_pos_mw",
    "mFRR Activation negative [MW]": "mfrr_act_neg_mw",
    "FCR Activation [MW]":        "fcr_act_mw",
    "Imbalance Price pos [EUR/MWh]": "imbalance_price_pos",
    "Imbalance Price neg [EUR/MWh]": "imbalance_price_neg",
    "Imbalance [MW]":             "imbalance_mw",
    "Total System Imbalance [MW]": "total_system_imbalance_mw",
}


def _get(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning(f"  GET failed {url}: {e}")
        return None


def _parse_content(content: bytes, filename: str) -> pd.DataFrame | None:
    """Auto-detect CSV or XLSX and parse into DataFrame."""
    fname = filename.lower()
    try:
        if fname.endswith(".csv"):
            # Try UTF-8, fall back to latin-1 (Swissgrid uses both)
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    text = content.decode(enc)
                    # Detect delimiter: comma or semicolon
                    delim = ";" if text.count(";") > text.count(",") else ","
                    df = pd.read_csv(StringIO(text), sep=delim, header=None)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None
        else:
            df = pd.read_excel(BytesIO(content), sheet_name=0, header=None)
    except Exception as e:
        log.error(f"  Parse error: {e}")
        return None

    # Find header row
    hdr = next(
        (i for i, row in df.iterrows()
         if any(str(c).strip().lower() in ("date","datum","time","zeit","timestamp")
                for c in row)),
        None
    )
    if hdr is None:
        # Try using row 0 as header directly
        hdr = 0

    df.columns = [str(c).strip() for c in df.iloc[hdr]]
    df = df.iloc[hdr + 1:].reset_index(drop=True)
    df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns}, inplace=True)

    # Build timestamp
    if "timestamp" not in df.columns:
        if "date_raw" in df.columns and "time_raw" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["date_raw"].astype(str) + " " + df["time_raw"].astype(str),
                errors="coerce", dayfirst=True
            )
            df.drop(columns=["date_raw","time_raw"], inplace=True)
        elif "date_raw" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date_raw"], errors="coerce", dayfirst=True)
            df.drop(columns=["date_raw"], inplace=True)

    df.dropna(subset=["timestamp"], inplace=True)

    # Numeric coercion
    num_cols = [c for c in df.columns if c != "timestamp"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("timestamp").reset_index(drop=True)


def _fetch_and_parse(url: str, label: str) -> pd.DataFrame | None:
    log.info(f"Fetching {label}: {url}")
    content = _get(url)
    if content is None:
        return None
    fname = url.split("/")[-1]
    (OUTPUT_DIR / f"raw_{label}_{fname}").write_bytes(content)
    return _parse_content(content, fname)


def run(
    start_year: int = 2019,
    end_year: int | None = None,
    include_realtime: bool = True,
) -> pd.DataFrame:
    """
    Collect Swissgrid Control Area Balance & activation volume data.

    Strategy:
    1. Download confirmed yearly CSV/XLSX per year (most reliable for historical)
    2. Download daily rolling files (TSI + CAB) for recent data
    3. Merge, deduplicate, sort chronologically

    Parameters
    ----------
    start_year       : First year to fetch (default 2019)
    end_year         : Last year to fetch (default: current year)
    include_realtime : Also fetch daily-updated TSI/CAB rolling files

    Returns
    -------
    DataFrame columns (subset, depending on file vintage):
        timestamp, net_position_mw, control_area_balance_mw,
        afrr_act_pos_mw, afrr_act_neg_mw, mfrr_act_pos_mw, mfrr_act_neg_mw,
        fcr_act_mw, total_system_imbalance_mw, imbalance_price_pos, imbalance_price_neg
    """
    if end_year is None:
        end_year = datetime.today().year
    log.info(f"=== Pipeline 03: Control Area Balance {start_year}–{end_year} ===")
    frames = []

    # Step 1 — Yearly historical files
    for year in range(start_year, end_year + 1):
        for tmpl, label in [(YEARLY_CSV_TMPL, "CAB_CSV"), (YEARLY_XLSX_TMPL, "CAB_XLSX")]:
            url = tmpl.format(year=year)
            df  = _fetch_and_parse(url, f"{label}_{year}")
            if df is not None and not df.empty:
                frames.append(df)
                break   # If CSV works, skip XLSX for the same year
            time.sleep(0.5)

    # Step 2 — Daily rolling files (most recent data, real-time indicator)
    if include_realtime:
        for url, label in [
            (CAB_CSV_URL,  "CAB_daily_CSV"),
            (CAB_XLSX_URL, "CAB_daily_XLSX"),
            (TSI_CSV_URL,  "TSI_CSV"),
            (TSI_XLSX_URL, "TSI_XLSX"),
        ]:
            df = _fetch_and_parse(url, label)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(0.5)

    if not frames:
        log.warning("No control area data collected.")
        return pd.DataFrame()

    out = (pd.concat(frames, ignore_index=True)
             .sort_values("timestamp")
             .drop_duplicates(subset=["timestamp"], keep="last")
             .reset_index(drop=True))

    log.info(f"Records: {len(out):,} | Range: {out['timestamp'].min()} → {out['timestamp'].max()}")

    out.to_parquet(OUTPUT_DIR / "control_area_balance.parquet", index=False)
    out.to_csv(OUTPUT_DIR / "control_area_balance.csv", index=False)
    log.info(f"Saved → {OUTPUT_DIR}/control_area_balance.{{parquet,csv}}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year",   type=int, default=None)
    p.add_argument("--no-realtime", action="store_true")
    args = p.parse_args()
    df = run(start_year=args.start_year, end_year=args.end_year,
             include_realtime=not args.no_realtime)
    print(df.head(10).to_string())
    print(df.dtypes)