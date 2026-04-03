"""
Pipeline 02: FCR / aFRR / mFRR Capacity Tender Results
Source:  Swissgrid procurement results pages (one per product)
         https://www.swissgrid.ch/en/home/operation/procurement-results/fcr.html
         https://www.swissgrid.ch/en/home/operation/procurement-results/afrr.html
         https://www.swissgrid.ch/en/home/operation/procurement-results/mfrr.html
Format:  HTML tables, XLSX downloads, and/or linked PDFs
Thesis:  Capacity price analysis — clearing prices per tender period (weekly/daily)
"""

import time, logging, re
import requests
import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import datetime

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("pip install beautifulsoup4")

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/tender_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL  = "https://www.swissgrid.ch"
HEADERS   = {"User-Agent": "SwissgridThesisBot/1.0 (thesis; contact@uni.ch)"}

# Procurement results page URLs per product
PRODUCT_URLS = {
    "FCR":  f"{BASE_URL}/en/home/operation/procurement-results/fcr.html",
    "aFRR": f"{BASE_URL}/en/home/operation/procurement-results/afrr.html",
    "mFRR": f"{BASE_URL}/en/home/operation/procurement-results/mfrr.html",
}

# Direct annual XLSX patterns (try before scraping — more reliable)
# Adjust sub-path if Swissgrid changes their DAM folder structure
DIRECT_XLSX = {
    "FCR":  BASE_URL + "/dam/swissgrid/operation/procurement/fcr/{year}/FCR_{year}.xlsx",
    "aFRR": BASE_URL + "/dam/swissgrid/operation/procurement/afrr/{year}/aFRR_{year}.xlsx",
    "mFRR": BASE_URL + "/dam/swissgrid/operation/procurement/mfrr/{year}/mFRR_{year}.xlsx",
}

COL_MAP = {
    "Tender Period":                  "tender_period",
    "Date":                           "date",
    "Start":                          "period_start",
    "End":                            "period_end",
    "Procurement Volume [MW]":        "volume_mw",
    "Procured Volume [MW]":           "volume_mw",
    "Marginal Price [EUR/MW/h]":      "clearing_price_eur_mwh",
    "Clearing Price [EUR/MW/h]":      "clearing_price_eur_mwh",
    "Average Price [EUR/MW/h]":       "avg_price_eur_mwh",
    "Weighted Average Price [EUR/MW/h]": "wavg_price_eur_mwh",
    "Number of Bids":                 "num_bids",
    "Accepted Bids":                  "accepted_bids",
    "Min Price [EUR/MW/h]":           "min_price_eur_mwh",
    "Max Price [EUR/MW/h]":           "max_price_eur_mwh",
}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _get(url: str) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r
    except Exception as e:
        log.warning(f"  GET failed {url}: {e}")
        return None


def _normalize(df: pd.DataFrame, product: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns}, inplace=True)
    df["product"] = product
    # Parse date columns
    for col in ("date", "period_start", "tender_period"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    # Numeric coercion
    for col in ("volume_mw","clearing_price_eur_mwh","avg_price_eur_mwh",
                "wavg_price_eur_mwh","num_bids","accepted_bids",
                "min_price_eur_mwh","max_price_eur_mwh"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _try_direct_xlsx(product: str, year: int) -> pd.DataFrame | None:
    url = DIRECT_XLSX.get(product, "").format(year=year)
    if not url:
        return None
    r = _get(url)
    if r is None:
        return None
    log.info(f"  ✓ Direct XLSX: {product} {year}")
    df = pd.read_excel(BytesIO(r.content), sheet_name=0)
    (OUTPUT_DIR / f"raw_{product}_{year}.xlsx").write_bytes(r.content)
    return _normalize(df, product)


def _scrape_page(product: str) -> list[pd.DataFrame]:
    """Scrape the product's procurement results HTML page."""
    url = PRODUCT_URLS[product]
    r = _get(url)
    if r is None:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    frames = []

    # 1. HTML tables on the page
    for tbl in soup.find_all("table"):
        try:
            df = pd.read_html(str(tbl))[^1_0]
            frames.append(_normalize(df, product))
            log.info(f"  Found HTML table with {len(df)} rows")
        except Exception:
            continue

    # 2. XLSX / CSV links embedded in the page
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not any(href.lower().endswith(ext) for ext in (".xlsx", ".csv")):
            continue
        full_url = href if href.startswith("http") else BASE_URL + href
        r2 = _get(full_url)
        if r2 is None:
            continue
        try:
            if href.lower().endswith(".csv"):
                df = pd.read_csv(BytesIO(r2.content))
            else:
                df = pd.read_excel(BytesIO(r2.content), sheet_name=0)
            frames.append(_normalize(df, product))
            log.info(f"  Downloaded linked file: {Path(href).name} ({len(df)} rows)")
        except Exception as e:
            log.warning(f"  Parse error {href}: {e}")
        time.sleep(0.5)

    # 3. PDF fallback (only if pdfplumber available)
    if PDF_SUPPORT:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.lower().endswith(".pdf"):
                continue
            full_url = href if href.startswith("http") else BASE_URL + href
            r2 = _get(full_url)
            if r2 is None:
                continue
            rows = []
            with pdfplumber.open(BytesIO(r2.content)) as pdf:
                for page in pdf.pages:
                    for tbl in (page.extract_tables() or []):
                        if not tbl:
                            continue
                        hdr = [str(c).strip() if c else f"c{i}" for i, c in enumerate(tbl[^1_0])]
                        for row in tbl[1:]:
                            rows.append(dict(zip(hdr, [str(c).strip() if c else "" for c in row])))
            if rows:
                df = _normalize(pd.DataFrame(rows), product)
                frames.append(df)
                log.info(f"  Parsed PDF: {Path(href).name} ({len(df)} rows)")
            time.sleep(0.5)

    return frames


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(
    products: list[str] | None = None,
    years: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Collect tender clearing prices for FCR, aFRR, mFRR.

    Parameters
    ----------
    products : ['FCR','aFRR','mFRR'] (default: all three)
    years    : list of calendar years (default: 2019 to current year)

    Returns
    -------
    dict[product] → DataFrame with columns:
        product, date/period_start, volume_mw, clearing_price_eur_mwh,
        avg_price_eur_mwh, wavg_price_eur_mwh, num_bids, accepted_bids
    """
    products = products or ["FCR", "aFRR", "mFRR"]
    years    = years or list(range(2019, datetime.today().year + 1))
    log.info(f"=== Pipeline 02: Tender Results — {products} ===")
    results = {}

    for product in products:
        frames = []

        # Step 1 – Direct annual XLSX (fastest)
        for year in years:
            df = _try_direct_xlsx(product, year)
            if df is not None:
                frames.append(df)
            time.sleep(0.3)

        # Step 2 – Scrape procurement results page
        page_frames = _scrape_page(product)
        frames.extend(page_frames)
        time.sleep(1.0)

        if not frames:
            log.warning(f"  No data for {product}")
            continue

        combined = (pd.concat(frames, ignore_index=True)
                      .drop_duplicates()
                      .reset_index(drop=True))

        # Sort by best available date column
        sort_col = next((c for c in ("period_start","date","tender_period")
                         if c in combined.columns), None)
        if sort_col:
            combined.sort_values(sort_col, inplace=True)
            combined.reset_index(drop=True, inplace=True)

        combined.to_parquet(OUTPUT_DIR / f"tender_{product.lower()}.parquet", index=False)
        combined.to_csv(OUTPUT_DIR / f"tender_{product.lower()}.csv", index=False)
        log.info(f"  {product}: {len(combined)} rows saved")
        results[product] = combined

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--products", nargs="+", default=["FCR","aFRR","mFRR"])
    p.add_argument("--years", nargs="+", type=int, default=None)
    args = p.parse_args()
    results = run(products=args.products, years=args.years)
    for prod, df in results.items():
        print(f"\n=== {prod} ({len(df)} rows) ===")
        print(df.head(5).to_string())