"""
Swissgrid-Specific Data Fetcher (Excluding ENTSO-E)
Master Thesis: Price Forecasting for Ancillary Services and Response Time Modeling

This script focuses on data ONLY available from Swissgrid directly:
1. Ancillary services tender results (auction prices)
2. Tertiary control energy (TRE) activation data with timestamps
3. Monthly Energy Overview Excel files (15-min resolution)
4. Active power losses data

Since you already have ENTSO-E data, this complements it.
"""

import requests
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
import time
import zipfile
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SwissgridDataCollector:
    """
    Collects Swissgrid-specific data not available via ENTSO-E
    """
    
    BASE_URL = "https://www.swissgrid.ch"
    TENDERS_URL = f"{BASE_URL}/en/home/customers/topics/ancillary-services/tenders.html"
    ENERGY_DATA_URL = f"{BASE_URL}/en/home/customers/topics/energy-data-ch.html"
    
    def __init__(self, output_dir="swissgrid_specific_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "tenders").mkdir(exist_ok=True)
        (self.output_dir / "tre_activation").mkdir(exist_ok=True)
        (self.output_dir / "energy_overview").mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research/Academic)'
        })
    
    def scrape_tender_page_links(self):
        """
        Scrape the tenders page to find all current download links
        Returns dict of {description: url}
        """
        logger.info("Scraping tender page for download links...")
        
        try:
            response = self.session.get(self.TENDERS_URL, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links to CSV and ZIP files
            links = {}
            for link in soup.find_all('a', href=True):
                href = link['href']
                if any(ext in href.lower() for ext in ['.csv', '.zip']):
                    # Get link text as description
                    text = link.get_text(strip=True)
                    if text:
                        full_url = href if href.startswith('http') else f"{self.BASE_URL}{href}"
                        links[text] = full_url
            
            logger.info(f"Found {len(links)} download links")
            return links
        
        except Exception as e:
            logger.error(f"Failed to scrape tender page: {e}")
            return {}
    
    def download_file(self, url, filename=None, subdirectory="tenders"):
        """
        Download a file from URL
        """
        if not filename:
            filename = url.split('/')[-1]
        
        output_path = self.output_dir / subdirectory / filename
        
        logger.info(f"Downloading: {filename}")
        
        try:
            response = self.session.get(url, timeout=120)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Extract if ZIP
            if filename.endswith('.zip'):
                self._extract_zip(output_path, subdirectory)
            
            logger.info(f"Downloaded: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return None
    
    def _extract_zip(self, zip_path, subdirectory):
        """Extract ZIP file to subdirectory"""
        try:
            extract_dir = self.output_dir / subdirectory / zip_path.stem
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"Extracted to: {extract_dir}")
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
    
    def download_all_current_tenders(self):
        """
        Download all current tender and TRE files
        """
        logger.info("=" * 70)
        logger.info("DOWNLOADING SWISSGRID TENDER DATA")
        logger.info("=" * 70)
        
        links = self.scrape_tender_page_links()
        
        if not links:
            logger.warning("No links found. You may need to update the scraping logic.")
            return []
        
        downloaded = []
        for description, url in links.items():
            # Determine subdirectory based on file type
            if 'TRE' in description or 'Tertiary control energy' in description:
                subdir = "tre_activation"
            else:
                subdir = "tenders"
            
            filepath = self.download_file(url, subdirectory=subdir)
            if filepath:
                downloaded.append(filepath)
            
            time.sleep(1)  # Rate limiting
        
        return downloaded
    
    def download_energy_overview_files(self, start_year=2022, end_year=None):
        """
        Download monthly Energy Overview Excel files
        These contain 15-minute resolution data for:
        - System imbalance
        - Balancing energy prices
        - aFRR/mFRR activations
        - Cross-border flows
        
        Note: File naming and URL patterns may change!
        """
        logger.info("=" * 70)
        logger.info("DOWNLOADING ENERGY OVERVIEW FILES")
        logger.info("=" * 70)
        
        if end_year is None:
            end_year = datetime.now().year
        
        # Common URL patterns for Energy Overview files
        # Pattern 1: Direct year-month format
        base_patterns = [
            "/dam/swissgrid/customers/topics/energy-statistics/Energieübersicht_Schweiz_{year}_{month:02d}.xlsx",
            "/dam/swissgrid/expert/grid_data/energy_data_ch/Energieübersicht_Schweiz_{year}_{month:02d}.xlsx",
        ]
        
        downloaded = []
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Skip future months
                if datetime(year, month, 1) > datetime.now():
                    break
                
                for pattern in base_patterns:
                    url = self.BASE_URL + pattern.format(year=year, month=month)
                    filename = f"Energieübersicht_Schweiz_{year}_{month:02d}.xlsx"
                    
                    filepath = self.download_file(url, filename, subdirectory="energy_overview")
                    
                    if filepath:
                        downloaded.append(filepath)
                        break  # Found it, move to next month
                    
                    time.sleep(0.5)
        
        if not downloaded:
            logger.warning("No Energy Overview files downloaded. URLs may have changed.")
            logger.info("Please visit manually: " + self.ENERGY_DATA_URL)
        
        return downloaded
    
    def parse_csv(self, csv_path, sep=';'):
        """
        Parse Swissgrid CSV file
        Default delimiter is semicolon
        """
        try:
            df = pd.read_csv(csv_path, sep=sep, encoding='utf-8-sig')
            logger.info(f"Parsed {csv_path.name}: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to parse {csv_path}: {e}")
            return None
    
    def parse_energy_overview_excel(self, excel_path):
        """
        Parse Energy Overview Excel file
        Returns dict of DataFrames (one per sheet)
        """
        try:
            xl_file = pd.ExcelFile(excel_path)
            
            data = {}
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                data[sheet_name] = df
                logger.info(f"  Sheet '{sheet_name}': {len(df)} rows")
            
            logger.info(f"Parsed {excel_path.name}: {len(data)} sheets")
            return data
        
        except Exception as e:
            logger.error(f"Failed to parse {excel_path}: {e}")
            return None
    
    def analyze_tre_response_times(self, tre_csv_path):
        """
        Analyze TRE file for response time metrics
        
        TRE files contain:
        - Bid submission timestamps
        - Call/activation timestamps
        - Volume delivered
        
        Response time = Call timestamp - Bid acceptance timestamp
        """
        df = self.parse_csv(tre_csv_path)
        
        if df is None:
            return None
        
        logger.info(f"\nAnalyzing TRE response times from {tre_csv_path.name}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Note: Actual column names may vary - adjust accordingly
        # Common columns: Zeit (Time), Produkt (Product), Menge (Volume), 
        #                 Preis (Price), Angebot/Abruf (Offer/Call)
        
        logger.info(f"First few rows:\n{df.head()}")
        
        return df
    
    def create_master_dataset(self):
        """
        Combine all downloaded data into master datasets
        """
        logger.info("=" * 70)
        logger.info("CREATING MASTER DATASETS")
        logger.info("=" * 70)
        
        # 1. Combine tender auction results
        tender_files = list((self.output_dir / "tenders").glob("*PRL-SRL-TRL*.csv"))
        if tender_files:
            logger.info(f"\nFound {len(tender_files)} tender result files")
            
            all_tenders = []
            for file in tender_files:
                df = self.parse_csv(file)
                if df is not None:
                    df['source_file'] = file.name
                    all_tenders.append(df)
            
            if all_tenders:
                master_tenders = pd.concat(all_tenders, ignore_index=True)
                output_path = self.output_dir / "MASTER_tender_results.csv"
                master_tenders.to_csv(output_path, index=False, sep=';')
                logger.info(f"Saved master tender dataset: {output_path}")
        
        # 2. Combine TRE activation data
        tre_files = list((self.output_dir / "tre_activation").glob("*.csv"))
        if tre_files:
            logger.info(f"\nFound {len(tre_files)} TRE files")
            
            all_tre = []
            for file in tre_files:
                df = self.parse_csv(file)
                if df is not None:
                    df['source_file'] = file.name
                    all_tre.append(df)
            
            if all_tre:
                master_tre = pd.concat(all_tre, ignore_index=True)
                output_path = self.output_dir / "MASTER_tre_activations.csv"
                master_tre.to_csv(output_path, index=False, sep=';')
                logger.info(f"Saved master TRE dataset: {output_path}")
        
        # 3. Energy Overview files remain separate (too large/complex to combine)
        energy_files = list((self.output_dir / "energy_overview").glob("*.xlsx"))
        if energy_files:
            logger.info(f"\nFound {len(energy_files)} Energy Overview files")
            logger.info("These contain 15-min resolution data - process individually as needed")
    
    def generate_data_summary(self):
        """
        Generate summary report of all downloaded data
        """
        logger.info("=" * 70)
        logger.info("DATA SUMMARY REPORT")
        logger.info("=" * 70)
        
        summary = {
            'Tender Files': len(list((self.output_dir / "tenders").glob("*.csv"))),
            'TRE Activation Files': len(list((self.output_dir / "tre_activation").glob("*.csv"))),
            'Energy Overview Files': len(list((self.output_dir / "energy_overview").glob("*.xlsx"))),
        }
        
        for category, count in summary.items():
            logger.info(f"{category}: {count}")
        
        logger.info(f"\nAll data saved to: {self.output_dir.absolute()}")
        
        return summary


def main():
    """
    Main execution
    """
    logger.info("=" * 70)
    logger.info("SWISSGRID-SPECIFIC DATA COLLECTION")
    logger.info("(Excluding ENTSO-E data you already have)")
    logger.info("=" * 70)
    
    collector = SwissgridDataCollector(output_dir="swissgrid_thesis_data")
    
    # Step 1: Download tender and TRE data
    logger.info("\n### STEP 1: TENDER & TRE DATA ###")
    tender_files = collector.download_all_current_tenders()
    
    # Step 2: Download Energy Overview files
    logger.info("\n### STEP 2: ENERGY OVERVIEW FILES (15-min data) ###")
    logger.info("Attempting to download 2022-2026...")
    energy_files = collector.download_energy_overview_files(start_year=2022)
    
    if not energy_files:
        logger.warning("\nEnergy Overview files could not be auto-downloaded.")
        logger.info("MANUAL DOWNLOAD INSTRUCTIONS:")
        logger.info("1. Visit: https://www.swissgrid.ch/en/home/customers/topics/energy-data-ch.html")
        logger.info("2. Look for 'Energieübersicht Schweiz' or 'Energy Overview Switzerland'")
        logger.info("3. Download monthly Excel files (typically under 'Downloads' section)")
        logger.info("4. Save to: " + str(collector.output_dir / "energy_overview"))
    
    # Step 3: Create master datasets
    logger.info("\n### STEP 3: CREATING MASTER DATASETS ###")
    collector.create_master_dataset()
    
    # Step 4: Summary
    logger.info("\n### STEP 4: SUMMARY ###")
    summary = collector.generate_data_summary()
    
    # Final instructions
    logger.info("\n" + "=" * 70)
    logger.info("NEXT STEPS FOR YOUR THESIS")
    logger.info("=" * 70)
    logger.info("""
    1. CHECK DOWNLOADED DATA:
       - Tender results: For capacity price forecasting
       - TRE files: For response time analysis (bid/call timestamps)
       - Energy Overview: For 15-min system state data
    
    2. KEY FILES FOR YOUR THESIS:
       - MASTER_tender_results.csv → Auction prices (FCR/aFRR/mFRR)
       - MASTER_tre_activations.csv → Activation data with timestamps
       - Energy Overview Excel files → System imbalance, prices, activations
    
    3. COMBINE WITH YOUR ENTSO-E DATA:
       - Use ENTSO-E for: Load, generation, imbalance prices
       - Use Swissgrid for: Auction prices, TRE timestamps, detailed activations
    
    4. RESPONSE TIME ANALYSIS:
       - Parse TRE CSV files for bid submission → call timestamps
       - Calculate time deltas for response time distributions
       - Correlate with system imbalance from Energy Overview
    
    5. PRICE FORECASTING FEATURES:
       - Historical auction prices (from tenders)
       - System imbalance (from Energy Overview)
       - Load/generation (from ENTSO-E you already have)
       - Time features (hour, day, season)
       - Lagged prices
    """)
    
    logger.info("\nData collection complete!")
    logger.info(f"Output directory: {collector.output_dir.absolute()}")


if __name__ == "__main__":
    main()
