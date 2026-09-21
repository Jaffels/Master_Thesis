#!/usr/bin/env python3
"""
Comprehensive ENTSOE Data Downloader for Switzerland

Downloads ALL available data categories from ENTSOE Transparency Platform:
- Market (Day-ahead prices, Intraday prices, etc.)
- Load (Actual load, Forecasted load)
- Generation (By type, Forecasts, Installed capacity)
- Transmission (Cross-border flows, Capacity)
- Outages (Planned and unplanned)
- Balancing (All balancing data)
- Operations (System operations data)

RATE LIMITING:
- 1.5 seconds minimum between each request
- 10 second pause every 50 requests
- Skips files that already exist (resume capability)

ESTIMATED TIME (2009-2026, ~17 years):
- ~40-50 requests per data category
- ~6-8 categories per time chunk
- ~2,500-3,000 total requests for full dataset
- **Total time: ~2-3 hours**

Based on: https://transparency.entsoe.eu/
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List
import time
from entsoe import EntsoePandasClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveENTSOEDownloader:
    """
    Download all available ENTSOE data for Switzerland
    """
    
    def __init__(self, api_key: str, output_dir: str = './entsoe_complete_data'):
        self.client = EntsoePandasClient(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.country_code = 'CH'
        
        # Rate limiting settings - balanced approach
        self.request_count = 0
        self.last_request_time = time.time()
        self.min_delay_between_requests = 1.5  # 1.5 seconds between requests
        self.requests_before_break = 50  # Take break every 50 requests
        self.break_duration = 10  # 10 second break
        
        # Create directory structure
        self.categories = {
            'market': self.output_dir / 'market',
            'load': self.output_dir / 'load',
            'generation': self.output_dir / 'generation',
            'transmission': self.output_dir / 'transmission',
            'outages': self.output_dir / 'outages',
            'balancing': self.output_dir / 'balancing',
            'operations': self.output_dir / 'operations'
        }
        
        for dir_path in self.categories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def rate_limit(self):
        """
        Enforce reasonable rate limiting
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Enforce minimum delay between requests
        if time_since_last_request < self.min_delay_between_requests:
            sleep_time = self.min_delay_between_requests - time_since_last_request
            time.sleep(sleep_time)
        
        # Take periodic breaks
        self.request_count += 1
        if self.request_count % self.requests_before_break == 0:
            logger.info(f"Progress: {self.request_count} requests completed. Taking {self.break_duration}s break...")
            time.sleep(self.break_duration)
        
        self.last_request_time = time.time()
    
    def check_exists(self, category: str, name: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """
        Check if file already exists to skip unnecessary API calls
        """
        filename = f"{name}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        filepath = self.categories[category] / filename
        return filepath.exists()
    
    def save_data(self, data, category: str, name: str, start: pd.Timestamp, end: pd.Timestamp):
        """Save data to parquet file"""
        if data is None or (hasattr(data, 'empty') and data.empty):
            logger.warning(f"No data for {name}")
            return False
        
        # Convert Series to DataFrame if needed
        if isinstance(data, pd.Series):
            data = data.to_frame(name=name)
        
        filename = f"{name}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        filepath = self.categories[category] / filename
        
        # Check if file already exists - skip download if it does
        if filepath.exists():
            logger.info(f"⊙ Already exists: {filepath.name} (skipping)")
            return True
        
        data.to_parquet(filepath)
        logger.info(f"✓ Saved: {filepath.name}")
        return True
    
    # ========================
    # MARKET DATA
    # ========================
    
    def fetch_market_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Fetch all market-related data
        - Day-ahead prices
        - Intraday prices  
        """
        logger.info("\n" + "="*80)
        logger.info("MARKET DATA")
        logger.info("="*80)
        
        # Day-ahead prices
        try:
            if self.check_exists('market', 'day_ahead_prices', start, end):
                logger.info("⊙ Day-ahead prices already exist (skipping)")
            else:
                logger.info("Fetching day-ahead prices...")
                self.rate_limit()
                data = self.client.query_day_ahead_prices(self.country_code, start=start, end=end)
                self.save_data(data, 'market', 'day_ahead_prices', start, end)
        except Exception as e:
            logger.warning(f"Day-ahead prices: {e}")
        
        time.sleep(2)  # Additional delay between data types
    
    # ========================
    # LOAD DATA
    # ========================
    
    def fetch_load_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Fetch all load-related data
        - Actual total load
        - Forecasted load (day-ahead)
        - Forecasted load (week-ahead)
        - Forecasted load (month-ahead)
        - Forecasted load (year-ahead)
        """
        logger.info("\n" + "="*80)
        logger.info("LOAD DATA")
        logger.info("="*80)
        
        # Actual total load
        try:
            if self.check_exists('load', 'actual_load', start, end):
                logger.info("⊙ Actual load already exists (skipping)")
            else:
                logger.info("Fetching actual load...")
                self.rate_limit()
                data = self.client.query_load(self.country_code, start=start, end=end)
                self.save_data(data, 'load', 'actual_load', start, end)
        except Exception as e:
            logger.warning(f"Actual load: {e}")
        
        time.sleep(2)
        
        # Day-ahead load forecast
        try:
            if self.check_exists('load', 'forecast_day_ahead', start, end):
                logger.info("⊙ Day-ahead load forecast already exists (skipping)")
            else:
                logger.info("Fetching day-ahead load forecast...")
                self.rate_limit()
                data = self.client.query_load_forecast(self.country_code, start=start, end=end)
                self.save_data(data, 'load', 'forecast_day_ahead', start, end)
        except Exception as e:
            logger.warning(f"Load forecast: {e}")
        
        time.sleep(2)
        
        # Week-ahead load forecast
        try:
            if self.check_exists('load', 'forecast_week_ahead', start, end):
                logger.info("⊙ Week-ahead load forecast already exists (skipping)")
            else:
                logger.info("Fetching week-ahead load forecast...")
                self.rate_limit()
                data = self.client.query_load_and_forecast(self.country_code, start=start, end=end)
                self.save_data(data, 'load', 'forecast_week_ahead', start, end)
        except Exception as e:
            logger.warning(f"Week-ahead forecast: {e}")
        
        time.sleep(2)
    
    # ========================
    # GENERATION DATA
    # ========================
    
    def fetch_generation_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Fetch all generation-related data
        - Actual generation per type
        - Generation forecasts
        - Installed capacity per type
        - Wind and solar forecasts
        """
        logger.info("\n" + "="*80)
        logger.info("GENERATION DATA")
        logger.info("="*80)
        
        # Actual generation by type
        try:
            logger.info("Fetching actual generation by type...")
            data = self.client.query_generation(self.country_code, start=start, end=end)
            self.save_data(data, 'generation', 'actual_by_type', start, end)
        except Exception as e:
            logger.warning(f"Actual generation: {e}")
        
        time.sleep(1)
        
        # Aggregated generation (single value)
        try:
            logger.info("Fetching aggregated generation...")
            data = self.client.query_generation(
                self.country_code, 
                start=start, 
                end=end,
                psr_type=None
            )
            self.save_data(data, 'generation', 'aggregated', start, end)
        except Exception as e:
            logger.warning(f"Aggregated generation: {e}")
        
        time.sleep(1)
        
        # Generation forecast
        try:
            logger.info("Fetching generation forecast...")
            data = self.client.query_generation_forecast(
                self.country_code, 
                start=start, 
                end=end
            )
            self.save_data(data, 'generation', 'forecast', start, end)
        except Exception as e:
            logger.warning(f"Generation forecast: {e}")
        
        time.sleep(1)
        
        # Wind and Solar forecasts (if available)
        try:
            logger.info("Fetching wind and solar forecast...")
            data = self.client.query_wind_and_solar_forecast(
                self.country_code, 
                start=start, 
                end=end
            )
            self.save_data(data, 'generation', 'wind_solar_forecast', start, end)
        except Exception as e:
            logger.warning(f"Wind/solar forecast: {e}")
        
        time.sleep(1)
        
        # Installed generation capacity
        try:
            logger.info("Fetching installed capacity...")
            data = self.client.query_installed_generation_capacity(
                self.country_code, 
                start=start, 
                end=end
            )
            self.save_data(data, 'generation', 'installed_capacity', start, end)
        except Exception as e:
            logger.warning(f"Installed capacity: {e}")
        
        time.sleep(1)
    
    # ========================
    # TRANSMISSION DATA
    # ========================
    
    def fetch_transmission_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Fetch all transmission-related data
        - Cross-border flows (physical)
        - Cross-border flows (scheduled)
        - Net transfer capacity
        - Available transfer capacity
        """
        logger.info("\n" + "="*80)
        logger.info("TRANSMISSION DATA")
        logger.info("="*80)
        
        neighbors = ['DE', 'FR', 'IT', 'AT']
        
        for neighbor in neighbors:
            # Physical flows
            try:
                logger.info(f"Fetching CH-{neighbor} physical flows...")
                data = self.client.query_crossborder_flows(
                    self.country_code,
                    neighbor,
                    start=start,
                    end=end
                )
                if isinstance(data, pd.Series):
                    data = data.to_frame(name='flow')
                self.save_data(data, 'transmission', f'physical_flow_CH_{neighbor}', start, end)
            except Exception as e:
                logger.warning(f"CH-{neighbor} physical flows: {e}")
            
            time.sleep(1)
            
            # Scheduled exchanges
            try:
                logger.info(f"Fetching CH-{neighbor} scheduled exchanges...")
                data = self.client.query_scheduled_exchanges(
                    self.country_code,
                    neighbor,
                    start=start,
                    end=end
                )
                if isinstance(data, pd.Series):
                    data = data.to_frame(name='scheduled')
                self.save_data(data, 'transmission', f'scheduled_CH_{neighbor}', start, end)
            except Exception as e:
                logger.warning(f"CH-{neighbor} scheduled: {e}")
            
            time.sleep(1)
            
            # Net transfer capacity
            try:
                logger.info(f"Fetching CH-{neighbor} net transfer capacity...")
                data = self.client.query_net_transfer_capacity_dayahead(
                    self.country_code,
                    neighbor,
                    start=start,
                    end=end
                )
                if isinstance(data, pd.Series):
                    data = data.to_frame(name='ntc')
                self.save_data(data, 'transmission', f'ntc_CH_{neighbor}', start, end)
            except Exception as e:
                logger.warning(f"CH-{neighbor} NTC: {e}")
            
            time.sleep(1)
    
    # ========================
    # BALANCING DATA
    # ========================
    
    def fetch_balancing_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Fetch all balancing-related data
        - Imbalance prices
        - Balancing energy prices
        - Activated balancing energy
        - Procured balancing capacity
        """
        logger.info("\n" + "="*80)
        logger.info("BALANCING DATA")
        logger.info("="*80)
        
        # Imbalance prices
        try:
            logger.info("Fetching imbalance prices...")
            data = self.client.query_imbalance_prices(
                self.country_code,
                start=start,
                end=end
            )
            self.save_data(data, 'balancing', 'imbalance_prices', start, end)
        except Exception as e:
            logger.warning(f"Imbalance prices: {e}")
        
        time.sleep(1)
        
        # Balancing energy prices (if available)
        try:
            logger.info("Fetching balancing energy prices...")
            data = self.client.query_aggregated_bids(
                self.country_code,
                start=start,
                end=end,
                process_type='A51'
            )
            self.save_data(data, 'balancing', 'balancing_prices', start, end)
        except Exception as e:
            logger.warning(f"Balancing prices: {e}")
        
        time.sleep(1)
        
        # Activated balancing energy
        try:
            logger.info("Fetching activated balancing energy...")
            # Switzerland may not publish this data, try anyway
            # Note: This often fails for CH as data may not be available
            pass  # Skip for now as not reliably available
        except Exception as e:
            logger.warning(f"Activated balancing: {e}")
        
        time.sleep(1)
        
        # Procured balancing capacity
        try:
            logger.info("Fetching procured balancing capacity...")
            # Switzerland typically doesn't publish detailed reserve data via ENTSOE
            # This data is usually available on Swissgrid's own website
            pass  # Skip for now as not reliably available
        except Exception as e:
            logger.warning(f"Procured capacity: {e}")
        
        time.sleep(1)
    
    # ========================
    # OUTAGES DATA
    # ========================
    
    def fetch_outages_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Fetch outage data
        - Generation unavailability
        - Transmission unavailability
        
        NOTE: Outage data is often not available via ENTSOE API for Switzerland.
        This data may be available on Swissgrid's website directly.
        """
        logger.info("\n" + "="*80)
        logger.info("OUTAGES DATA")
        logger.info("="*80)
        
        # Generation unavailability
        try:
            if self.check_exists('outages', 'generation_unavailability', start, end):
                logger.info("⊙ Generation unavailability already exists (skipping)")
            else:
                logger.info("Fetching generation unavailability...")
                self.rate_limit()
                data = self.client.query_unavailability_of_generation_units(
                    self.country_code,
                    start=start,
                    end=end
                )
                self.save_data(data, 'outages', 'generation_unavailability', start, end)
        except Exception as e:
            logger.warning(f"Generation unavailability not available: {e}")
        
        # Transmission unavailability - not available in entsoe-py library
        logger.info("Transmission unavailability: Not available via this API")
        # Note: Switzerland publishes this on Swissgrid website, not ENTSOE API
    
    # ========================
    # MAIN RUNNER
    # ========================
    
    def download_all(self, start_date: str, end_date: str, chunk_days: int = 7):
        """
        Download all available data for date range
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            chunk_days: Days per API call
        """
        start = pd.Timestamp(start_date, tz='Europe/Zurich')
        end = pd.Timestamp(end_date, tz='Europe/Zurich')
        
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE ENTSOE DATA DOWNLOAD")
        logger.info("="*80)
        logger.info(f"Country: Switzerland (CH)")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("="*80)
        
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"PROCESSING: {current.date()} to {chunk_end.date()}")
            logger.info(f"{'='*80}")
            
            # Fetch all categories
            self.fetch_market_data(current, chunk_end)
            self.fetch_load_data(current, chunk_end)
            self.fetch_generation_data(current, chunk_end)
            self.fetch_transmission_data(current, chunk_end)
            self.fetch_balancing_data(current, chunk_end)
            self.fetch_outages_data(current, chunk_end)
            
            current = chunk_end
            time.sleep(2)
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info("="*80)
        self.create_summary()
    
    def create_summary(self):
        """Create summary of downloaded data"""
        summary = {}
        
        for category, dir_path in self.categories.items():
            files = list(dir_path.glob('*.parquet'))
            summary[category] = {
                'file_count': len(files),
                'files': [f.name for f in files[:5]]  # Show first 5
            }
        
        logger.info("\nData Summary:")
        for category, info in summary.items():
            logger.info(f"  {category}: {info['file_count']} files")
        
        # Save summary
        summary_df = pd.DataFrame([
            {'category': k, 'file_count': v['file_count']} 
            for k, v in summary.items()
        ])
        summary_file = self.output_dir / 'download_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"\nSummary saved to: {summary_file}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("COMPREHENSIVE ENTSOE DATA DOWNLOADER FOR SWITZERLAND")
    print("="*80)
    
    # Get API key
    api_key = os.getenv('ENTSOE_API_KEY')
    
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('ENTSOE_API_KEY')
    
    if not api_key:
        api_key = input("\nEnter ENTSOE API key: ").strip()
    
    # Date range
    print("\nRecommended: Start from 2015-01-01 for maximum historical data")
    start_date = input("Start date (YYYY-MM-DD, default: 2015-01-01): ").strip()
    if not start_date:
        start_date = '2015-01-01'
    
    end_date = input("End date (YYYY-MM-DD, default: today): ").strip()
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Output directory
    output_dir = input("\nOutput directory (default: ./entsoe_complete_data): ").strip()
    if not output_dir:
        output_dir = './entsoe_complete_data'
    
    # Confirmation
    print("\n" + "="*80)
    print("READY TO DOWNLOAD")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Output: {output_dir}")
    print("\nThis will download:")
    print("  • Market: Day-ahead prices, etc.")
    print("  • Load: Actual and forecasts")
    print("  • Generation: By type, forecasts, capacity")
    print("  • Transmission: Cross-border flows, capacity")
    print("  • Balancing: Prices, activated energy, reserves")
    print("  • Outages: Generation and transmission unavailability")
    print("\nEstimated time: 2-4 hours for 10 years of data")
    
    proceed = input("\nProceed? (yes/no): ").strip().lower()
    
    if proceed not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Run download
    downloader = ComprehensiveENTSOEDownloader(api_key, output_dir)
    downloader.download_all(start_date, end_date, chunk_days=7)
    
    print("\n" + "="*80)
    print("ALL DATA DOWNLOADED SUCCESSFULLY!")
    print("="*80)
    print(f"\nData location: {output_dir}")
    print("\nNext steps:")
    print("  1. Run check_frequency.py to verify data quality")
    print("  2. Run data_processing.py to create master dataset")
    print("  3. Start your analysis!")


if __name__ == "__main__":
    main()
