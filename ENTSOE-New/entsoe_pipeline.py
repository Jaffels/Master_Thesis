"""
ENTSOE API Data Pipeline for Ancillary Services Analysis
Research focus: Price forecasting and response time modeling (Swissgrid)

This pipeline collects:
- Balancing energy prices (activation prices)
- Reserve capacity prices (FCR, aFRR, mFRR)
- Imbalance prices
- Actual generation and load data
- Cross-border flows
- Balancing energy volumes and directions
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time

# ENTSOE API client
from entsoe import EntsoePandasClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ENTSOEAncillaryServicesPipeline:
    """
    Data pipeline for collecting ENTSOE ancillary services data
    focused on Swiss market (Swissgrid)
    """
    
    def __init__(self, api_key: str, output_dir: str = "./data"):
        """
        Initialize pipeline with ENTSOE API credentials
        
        Args:
            api_key: ENTSOE API key
            output_dir: Directory to store collected data
        """
        self.client = EntsoePandasClient(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Switzerland country code
        self.country_code = 'CH'  # Switzerland
        
        # Create subdirectories for different data types
        self.dirs = {
            'balancing': self.output_dir / 'balancing',
            'reserves': self.output_dir / 'reserves',
            'imbalance': self.output_dir / 'imbalance',
            'generation': self.output_dir / 'generation',
            'load': self.output_dir / 'load',
            'crossborder': self.output_dir / 'crossborder',
            'processed': self.output_dir / 'processed'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def fetch_balancing_prices(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch balancing energy activation prices
        These are the prices paid for upward/downward regulation
        
        Critical for: Price forecasting models
        """
        logger.info(f"Fetching balancing prices from {start} to {end}")
        
        try:
            # Try to get aggregated balancing prices
            # Note: This endpoint may not be available for all countries/periods
            data = self.client.query_aggregated_bids(
                self.country_code, 
                start=start, 
                end=end,
                process_type='A51'  # Balancing energy
            )
            
            if data is not None and not data.empty:
                # Convert Series to DataFrame if needed
                if isinstance(data, pd.Series):
                    data = data.to_frame(name='balancing_price')
                
                output_file = self.dirs['balancing'] / f'balancing_prices_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}.parquet'
                data.to_parquet(output_file)
                logger.info(f"Saved balancing prices: {output_file}")
                return data
            else:
                logger.warning("No balancing price data available")
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"Balancing prices not available: {e}")
            return pd.DataFrame()
    
    def fetch_imbalance_prices(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch imbalance prices (penalties for imbalance)
        
        Critical for: Understanding balancing market dynamics
        """
        logger.info(f"Fetching imbalance prices from {start} to {end}")
        
        try:
            data = self.client.query_imbalance_prices(
                self.country_code,
                start=start,
                end=end,
                psr_type=None
            )
            
            if data is not None and not data.empty:
                output_file = self.dirs['imbalance'] / f'imbalance_prices_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}.parquet'
                data.to_parquet(output_file)
                logger.info(f"Saved imbalance prices: {output_file}")
                return data
            else:
                logger.warning("No imbalance price data available")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching imbalance prices: {e}")
            return pd.DataFrame()
    
    def fetch_procured_balancing_capacity(self, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """
        Fetch procured balancing capacity (reserve volumes and prices)
        
        Types:
        - FCR (Frequency Containment Reserves) - primary control
        - aFRR (automatic Frequency Restoration Reserves) - secondary control  
        - mFRR (manual Frequency Restoration Reserves) - tertiary control
        
        Critical for: Reserve capacity price forecasting
        """
        logger.info(f"Fetching procured balancing capacity from {start} to {end}")
        
        # Try different methods to get reserve data
        results = {}
        
        # Method 1: Try procured balancing capacity (if available)
        try:
            data = self.client.query_procured_balancing_capacity(
                self.country_code,
                start=start,
                end=end
            )
            
            if data is not None and not data.empty:
                # Convert Series to DataFrame if needed
                if isinstance(data, pd.Series):
                    data = data.to_frame(name='reserve_capacity')
                
                output_file = self.dirs['reserves'] / f'reserves_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}.parquet'
                data.to_parquet(output_file)
                logger.info(f"Saved reserve data: {output_file}")
                results['reserves'] = data
                return results
                
        except Exception as e:
            logger.warning(f"Procured balancing capacity not available: {e}")
        
        # Method 2: Try reserve prices via aggregated bids
        for reserve_name in ['FCR', 'aFRR', 'mFRR']:
            try:
                # Try to get reserve auction results
                # Note: Not all countries publish this data
                logger.info(f"Attempting to fetch {reserve_name} data...")
                
            except Exception as e:
                logger.warning(f"Reserve type {reserve_name} not available: {e}")
        
        if not results:
            logger.warning("No reserve capacity data available for this period")
        
        return results
    
    def fetch_activated_balancing_energy(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch actual activated balancing energy volumes
        
        Critical for: Understanding actual system needs and response patterns
        """
        logger.info(f"Fetching activated balancing energy from {start} to {end}")
        
        try:
            # Try to get activated balancing energy volumes
            data = self.client.query_activated_balancing_energy(
                self.country_code,
                start=start,
                end=end
            )
            
            if data is not None and not data.empty:
                # Convert Series to DataFrame if needed
                if isinstance(data, pd.Series):
                    data = data.to_frame(name='activated_energy')
                
                output_file = self.dirs['balancing'] / f'activated_volumes_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}.parquet'
                data.to_parquet(output_file)
                logger.info(f"Saved activated balancing energy: {output_file}")
                return data
            else:
                logger.warning("No activated balancing energy data available")
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"Activated balancing energy not available: {e}")
            return pd.DataFrame()
    
    def fetch_generation_data(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch actual generation per production type
        
        Useful for: Understanding system state and predicting reserve needs
        """
        logger.info(f"Fetching generation data from {start} to {end}")
        
        try:
            data = self.client.query_generation(
                self.country_code,
                start=start,
                end=end
            )
            
            if data is not None and not data.empty:
                output_file = self.dirs['generation'] / f'generation_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}.parquet'
                data.to_parquet(output_file)
                logger.info(f"Saved generation data: {output_file}")
                return data
            else:
                logger.warning("No generation data available")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching generation data: {e}")
            return pd.DataFrame()
    
    def fetch_load_data(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch actual total load
        
        Useful for: System state features
        """
        logger.info(f"Fetching load data from {start} to {end}")
        
        try:
            data = self.client.query_load(
                self.country_code,
                start=start,
                end=end
            )
            
            if data is not None and not data.empty:
                output_file = self.dirs['load'] / f'load_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}.parquet'
                data.to_parquet(output_file)
                logger.info(f"Saved load data: {output_file}")
                return data
            else:
                logger.warning("No load data available")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching load data: {e}")
            return pd.DataFrame()
    
    def fetch_crossborder_flows(self, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """
        Fetch cross-border flows
        Switzerland borders: DE, FR, IT, AT
        
        Useful for: Understanding interconnection constraints affecting balancing
        """
        logger.info(f"Fetching cross-border flows from {start} to {end}")
        
        neighboring_countries = ['DE', 'FR', 'IT', 'AT']
        results = {}
        
        for neighbor in neighboring_countries:
            try:
                data = self.client.query_crossborder_flows(
                    self.country_code,
                    neighbor,
                    start=start,
                    end=end
                )
                
                if data is not None and not data.empty:
                    # Convert Series to DataFrame if needed
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name='flow')
                    
                    output_file = self.dirs['crossborder'] / f'flow_CH_{neighbor}_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}.parquet'
                    data.to_parquet(output_file)
                    logger.info(f"Saved CH-{neighbor} flow data: {output_file}")
                    results[f'CH_{neighbor}'] = data
                else:
                    logger.warning(f"No flow data for CH-{neighbor}")
                
                # Rate limiting
                time.sleep(0.5)
                    
            except Exception as e:
                logger.warning(f"Cross-border flows for CH-{neighbor} not available: {e}")
        
        return results
    
    def run_full_collection(self, start_date: str, end_date: str, chunk_days: int = 7):
        """
        Run complete data collection pipeline
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            chunk_days: Number of days per API call (to avoid timeouts)
        """
        start = pd.Timestamp(start_date, tz='Europe/Zurich')
        end = pd.Timestamp(end_date, tz='Europe/Zurich')
        
        logger.info(f"Starting full data collection from {start} to {end}")
        
        # Split into chunks to avoid API timeouts
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing chunk: {current} to {chunk_end}")
            logger.info(f"{'='*60}\n")
            
            # Collect all data types
            self.fetch_balancing_prices(current, chunk_end)
            time.sleep(1)
            
            self.fetch_imbalance_prices(current, chunk_end)
            time.sleep(1)
            
            self.fetch_procured_balancing_capacity(current, chunk_end)
            time.sleep(1)
            
            self.fetch_activated_balancing_energy(current, chunk_end)
            time.sleep(1)
            
            self.fetch_generation_data(current, chunk_end)
            time.sleep(1)
            
            self.fetch_load_data(current, chunk_end)
            time.sleep(1)
            
            self.fetch_crossborder_flows(current, chunk_end)
            time.sleep(1)
            
            current = chunk_end
        
        logger.info("Data collection completed!")
        self.create_data_summary()
    
    def create_data_summary(self):
        """
        Create a summary of collected data
        """
        summary = {}
        
        for data_type, dir_path in self.dirs.items():
            if data_type == 'processed':
                continue
                
            files = list(dir_path.glob('*.parquet'))
            summary[data_type] = {
                'file_count': len(files),
                'files': [f.name for f in files]
            }
        
        summary_df = pd.DataFrame([
            {'data_type': k, 'file_count': v['file_count']} 
            for k, v in summary.items()
        ])
        
        summary_file = self.output_dir / 'data_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"\nData Summary:")
        logger.info(summary_df.to_string())
        logger.info(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    # Example usage
    API_KEY = os.getenv('ENTSOE_API_KEY', 'your-api-key-here')
    
    pipeline = ENTSOEAncillaryServicesPipeline(
        api_key=API_KEY,
        output_dir='./swissgrid_ancillary_data'
    )
    
    # Collect data for the last 2 years (adjust as needed)
    # For thesis, you might want 3-5 years of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    pipeline.run_full_collection(
        start_date=start_date,
        end_date=end_date,
        chunk_days=7  # Weekly chunks
    )
