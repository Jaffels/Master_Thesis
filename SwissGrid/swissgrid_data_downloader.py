"""
Swissgrid Ancillary Services Data Downloader
Master Thesis: Price Forecasting for Ancillary Services and Response Time Modeling

This script downloads REAL data files from Swissgrid's public website:
- Auction results for Primary (PRL), Secondary (SRL), and Tertiary (TRL) control power
- Tertiary control energy bids and activations
- Active power losses compensation data
- Historical data archives

Data source: https://www.swissgrid.ch/en/home/customers/topics/ancillary-services/tenders.html
"""

import requests
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import time
import zipfile
import io

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SwissgridDataDownloader:
    """
    Downloads ancillary services data from Swissgrid's public website
    
    Data includes:
    - PRL (Primary Reserve/FCR - Frequency Containment Reserve)
    - SRL (Secondary Reserve/aFRR - automatic Frequency Restoration Reserve)  
    - TRL (Tertiary Reserve/mFRR - manual Frequency Restoration Reserve)
    - Tertiary Control Energy (TRE) - activation data
    - Active Power Losses Compensation
    """
    
    # Base URL for Swissgrid data downloads
    BASE_URL = "https://www.swissgrid.ch/dam/jcr:"
    
    # Current data file IDs (these change when Swissgrid updates files)
    # You need to inspect the website to get current file IDs
    CURRENT_FILES = {
        'auction_results_2026': '330cb638-0ce3-40b4-a9dc-ca26b7ad46c5/2026-PRL-SRL-TRL-Ergebnis.csv',
        'active_power_losses': 'd6feb681-332c-4b11-ab3a-3a48cea70871/KompWV-Ergebnis.csv',
        'tre_current': 'eb7a5d1d-e86f-46f3-a4ec-0bd0249d01c9/2026-TRE-Ergebnis-Aktuell.csv',
        'tre_may_2026': '848dbc3c-53f1-457d-beaf-c48065642eb4/2026-05-TRE-Ergebnis.csv.zip',
        'tre_april_2026': 'b52a18f2-2da0-4962-9a17-8efedeca2b32/2026-04-TRE-Ergebnis.csv.zip',
        'archive_2015_2025': '7cff53d2-b292-4f8f-b9b0-61804f791123/Archiv-Ergebnisse.zip',
        'tre_archive_2025': '706888f2-08e9-4df7-accb-8d3e4edcc207/TRE-Ergebnisse-2025.zip',
    }
    
    def __init__(self, output_dir="swissgrid_data"):
        """Initialize the downloader with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research/Academic)'
        })
    
    def download_file(self, file_key, custom_filename=None):
        """
        Download a file from Swissgrid
        
        Parameters:
        -----------
        file_key : str
            Key from CURRENT_FILES dictionary
        custom_filename : str, optional
            Custom filename to save as (otherwise uses original)
            
        Returns:
        --------
        Path to downloaded file or None if failed
        """
        if file_key not in self.CURRENT_FILES:
            logger.error(f"Unknown file key: {file_key}")
            return None
        
        file_path = self.CURRENT_FILES[file_key]
        url = self.BASE_URL + file_path
        
        # Determine filename
        if custom_filename:
            filename = custom_filename
        else:
            filename = file_path.split('/')[-1]
        
        output_path = self.output_dir / filename
        
        logger.info(f"Downloading {file_key} from Swissgrid...")
        
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Save file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded: {output_path}")
            
            # If it's a ZIP file, extract it
            if filename.endswith('.zip'):
                self.extract_zip(output_path)
            
            return output_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {file_key}: {e}")
            return None
    
    def extract_zip(self, zip_path):
        """Extract a ZIP file to the output directory"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extract_dir = self.output_dir / zip_path.stem
                extract_dir.mkdir(exist_ok=True)
                zip_ref.extractall(extract_dir)
                logger.info(f"Extracted {zip_path} to {extract_dir}")
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
    
    def download_all_current(self):
        """Download all current/recent data files"""
        logger.info("=" * 60)
        logger.info("Downloading current Swissgrid ancillary services data")
        logger.info("=" * 60)
        
        downloaded = []
        
        # Download current auction results (PRL/SRL/TRL)
        logger.info("\n--- Primary, Secondary, Tertiary Control Power Auction Results ---")
        path = self.download_file('auction_results_2026')
        if path:
            downloaded.append(path)
        time.sleep(1)
        
        # Download active power losses
        logger.info("\n--- Active Power Losses Compensation ---")
        path = self.download_file('active_power_losses')
        if path:
            downloaded.append(path)
        time.sleep(1)
        
        # Download tertiary control energy (activation data)
        logger.info("\n--- Tertiary Control Energy (Current) ---")
        path = self.download_file('tre_current')
        if path:
            downloaded.append(path)
        time.sleep(1)
        
        logger.info("\n--- Tertiary Control Energy (Monthly Archives) ---")
        path = self.download_file('tre_may_2026')
        if path:
            downloaded.append(path)
        time.sleep(1)
        
        path = self.download_file('tre_april_2026')
        if path:
            downloaded.append(path)
        time.sleep(1)
        
        return downloaded
    
    def download_historical(self):
        """Download historical archives"""
        logger.info("=" * 60)
        logger.info("Downloading historical archives")
        logger.info("=" * 60)
        
        downloaded = []
        
        # Historical auction results 2015-2025
        logger.info("\n--- Historical Auction Results 2015-2025 ---")
        path = self.download_file('archive_2015_2025')
        if path:
            downloaded.append(path)
        time.sleep(2)
        
        # Historical TRE 2025
        logger.info("\n--- Historical TRE 2025 ---")
        path = self.download_file('tre_archive_2025')
        if path:
            downloaded.append(path)
        time.sleep(2)
        
        return downloaded
    
    def load_auction_results(self, csv_path):
        """
        Load and parse auction results CSV
        
        Returns:
        --------
        pandas.DataFrame with parsed auction data
        """
        try:
            # Swissgrid CSVs use semicolon as delimiter
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
            logger.info(f"Loaded {len(df)} rows from {csv_path}")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {csv_path}: {e}")
            return None
    
    def analyze_auction_data(self, df):
        """
        Perform basic analysis on auction data
        """
        if df is None or df.empty:
            logger.warning("No data to analyze")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"\nDataset shape: {df.shape}")
        logger.info(f"\nColumn names and types:")
        for col, dtype in df.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        logger.info(f"\nFirst few rows:")
        logger.info(df.head().to_string())
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            logger.info(f"\nNumeric statistics:")
            logger.info(df[numeric_cols].describe().to_string())


def main():
    """
    Main execution function
    """
    downloader = SwissgridDataDownloader(output_dir="swissgrid_thesis_data")
    
    # Download current data
    current_files = downloader.download_all_current()
    
    # Optionally download historical data
    logger.info("\n\nDo you want to download historical archives? (Large files)")
    logger.info("Commenting out for now - uncomment if needed:")
    # historical_files = downloader.download_historical()
    
    # Load and analyze the main auction results file
    logger.info("\n\n" + "=" * 60)
    logger.info("LOADING AND ANALYZING AUCTION RESULTS")
    logger.info("=" * 60)
    
    auction_file = downloader.output_dir / "2026-PRL-SRL-TRL-Ergebnis.csv"
    if auction_file.exists():
        df = downloader.load_auction_results(auction_file)
        downloader.analyze_auction_data(df)
    
    logger.info("\n\n" + "=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"All files saved to: {downloader.output_dir.absolute()}")
    logger.info("\nIMPORTANT NOTES:")
    logger.info("1. File URLs may change when Swissgrid updates their website")
    logger.info("2. Visit https://www.swissgrid.ch/en/home/customers/topics/ancillary-services/tenders.html")
    logger.info("   to find the latest file links")
    logger.info("3. For balancing/frequency data, visit:")
    logger.info("   https://www.swissgrid.ch/en/home/operation/grid-data/balance.html")
    logger.info("4. CSV files use semicolon (;) as delimiter, not comma")


if __name__ == "__main__":
    main()
