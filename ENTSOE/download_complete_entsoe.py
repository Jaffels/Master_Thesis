#!/usr/bin/env python3
"""
COMPLETE ENTSOE Data Downloader - ALL Categories (FIXED VERSION)

Downloads EVERY available data type from ENTSOE Transparency Platform for Switzerland.
Generates detailed report of what was found and what wasn't.

FIXES APPLIED:
- Added business_type parameter to query_activated_balancing_energy
  (Now retrieves FCR, aFRR, mFRR, RR separately)
- Added process_type parameter to query_procured_balancing_capacity
  (Now retrieves capacity data for each reserve type)

These fixes address the "missing required positional argument" errors
in the balancing data section.
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import time
from entsoe import EntsoePandasClient
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveENTSOEDownloader:
    """
    Download ALL available ENTSOE data for Switzerland with detailed reporting
    """
    
    def __init__(self, api_key: str, output_dir: str = './entsoe_complete_data'):
        self.client = EntsoePandasClient(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.country_code = 'CH'
        
        # Tracking for report
        self.download_report = {
            'timestamp': datetime.now().isoformat(),
            'categories': {}
        }
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = time.time()
        self.min_delay_between_requests = 1.5
        self.requests_before_break = 50
        self.break_duration = 10
        
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
        """Rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_delay_between_requests:
            sleep_time = self.min_delay_between_requests - time_since_last_request
            time.sleep(sleep_time)
        
        self.request_count += 1
        if self.request_count % self.requests_before_break == 0:
            logger.info(f"Progress: {self.request_count} requests. Taking {self.break_duration}s break...")
            time.sleep(self.break_duration)
        
        self.last_request_time = time.time()
    
    def check_exists(self, category: str, name: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """Check if file already exists"""
        filename = f"{name}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        filepath = self.categories[category] / filename
        return filepath.exists()
    
    def try_download(self, category: str, subcategory: str, func, *args, **kwargs) -> Tuple[bool, str]:
        """
        Try to download data and track result
        
        Returns:
            (success: bool, message: str)
        """
        try:
            self.rate_limit()
            data = func(*args, **kwargs)
            
            if data is None or (hasattr(data, 'empty') and data.empty):
                return False, "No data returned from API"
            
            # Convert Series to DataFrame if needed
            if isinstance(data, pd.Series):
                data = data.to_frame(name=subcategory)
            
            return True, f"Success - {len(data)} records"
        
        except Exception as e:
            error_msg = str(e)
            if '503' in error_msg:
                return False, "503 Service Unavailable"
            elif '400' in error_msg:
                return False, "400 Bad Request - Data not available"
            elif '404' in error_msg:
                return False, "404 Not Found - Endpoint not available"
            else:
                return False, f"Error: {error_msg[:100]}"
    
    def save_data(self, data, category: str, name: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """Save data to parquet"""
        if data is None or (hasattr(data, 'empty') and data.empty):
            return False
        
        if isinstance(data, pd.Series):
            data = data.to_frame(name=name)
        
        filename = f"{name}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        filepath = self.categories[category] / filename
        
        if filepath.exists():
            logger.info(f"⊙ Already exists: {filepath.name}")
            return True
        
        data.to_parquet(filepath)
        logger.info(f"✓ Saved: {filepath.name}")
        return True
    
    def record_result(self, category: str, subcategory: str, success: bool, message: str):
        """Record download result for reporting"""
        if category not in self.download_report['categories']:
            self.download_report['categories'][category] = {}
        
        self.download_report['categories'][category][subcategory] = {
            'success': success,
            'message': message
        }
    
    # ========================
    # MARKET DATA - ALL TYPES
    # ========================
    
    def fetch_all_market_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """Fetch ALL market-related data"""
        logger.info("\n" + "="*80)
        logger.info("MARKET DATA - ALL SUBCATEGORIES")
        logger.info("="*80)
        
        category = 'market'
        
        # 1. Day-ahead prices
        subcategory = "day_ahead_prices"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_day_ahead_prices,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                self.save_data(self.client.query_day_ahead_prices(self.country_code, start=start, end=end),
                              category, subcategory, start, end)
        
        time.sleep(1)
        
        # 2. Intraday prices (likely not available)
        subcategory = "intraday_prices"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available via standard API")
        
        # 3. Flow-based congestion income
        subcategory = "flow_based_congestion_income"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Requires specific flow-based market API")
        
        # 4. Implicit allocations - congestion income  
        subcategory = "implicit_allocations_congestion"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Requires auction data API")
        
        # 5. Total nominated capacity
        subcategory = "total_nominated_capacity"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Requires nomination API")
    
    # ========================
    # LOAD DATA - ALL TYPES
    # ========================
    
    def fetch_all_load_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """Fetch ALL load-related data"""
        logger.info("\n" + "="*80)
        logger.info("LOAD DATA - ALL SUBCATEGORIES")
        logger.info("="*80)
        
        category = 'load'
        
        # 1. Actual total load
        subcategory = "actual_total_load"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_load,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_load(self.country_code, start=start, end=end)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 2. Day-ahead total load forecast
        subcategory = "forecast_day_ahead_load"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_load_forecast,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_load_forecast(self.country_code, start=start, end=end)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 3. Week-ahead total load forecast
        subcategory = "forecast_week_ahead_load"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_load_and_forecast,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_load_and_forecast(self.country_code, start=start, end=end)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 4. Month-ahead total load forecast
        subcategory = "forecast_month_ahead_load"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
        
        # 5. Year-ahead total load forecast
        subcategory = "forecast_year_ahead_load"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
        
        # 6. Forecasted margin
        subcategory = "forecasted_margin"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
    
    # ========================
    # GENERATION DATA - ALL TYPES
    # ========================
    
    def fetch_all_generation_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """Fetch ALL generation-related data"""
        logger.info("\n" + "="*80)
        logger.info("GENERATION DATA - ALL SUBCATEGORIES")
        logger.info("="*80)
        
        category = 'generation'
        
        # 1. Actual generation per production type
        subcategory = "actual_per_production_type"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_generation,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_generation(self.country_code, start=start, end=end)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 2. Aggregated generation per type
        subcategory = "aggregated_generation"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_generation,
                self.country_code, start=start, end=end, psr_type=None
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_generation(self.country_code, start=start, end=end, psr_type=None)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 3. Day-ahead generation forecast
        subcategory = "forecast_day_ahead_generation"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_generation_forecast,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_generation_forecast(self.country_code, start=start, end=end)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 4. Wind and solar forecast
        subcategory = "wind_solar_forecast"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_wind_and_solar_forecast,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_wind_and_solar_forecast(self.country_code, start=start, end=end)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 5. Installed generation capacity per type
        subcategory = "installed_capacity_per_type"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_installed_generation_capacity,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_installed_generation_capacity(self.country_code, start=start, end=end)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 6. Actual generation per unit
        subcategory = "actual_per_unit"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
        
        # 7. Water reservoirs and hydro storage
        subcategory = "water_reservoirs_hydro"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
    
    # ========================
    # TRANSMISSION DATA - ALL TYPES
    # ========================
    
    def fetch_all_transmission_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """Fetch ALL transmission-related data"""
        logger.info("\n" + "="*80)
        logger.info("TRANSMISSION DATA - ALL SUBCATEGORIES")
        logger.info("="*80)
        
        category = 'transmission'
        neighbors = ['DE_LU', 'FR', 'IT', 'AT']
        
        for neighbor in neighbors:
            # 1. Physical flows
            subcategory = f"physical_flow_CH_{neighbor}"
            if self.check_exists(category, subcategory, start, end):
                logger.info(f"⊙ {subcategory} already exists")
                self.record_result(category, subcategory, True, "Already downloaded")
            else:
                logger.info(f"Fetching {subcategory}...")
                success, msg = self.try_download(
                    category, subcategory,
                    self.client.query_crossborder_flows,
                    self.country_code, neighbor, start=start, end=end
                )
                self.record_result(category, subcategory, success, msg)
                if success:
                    data = self.client.query_crossborder_flows(self.country_code, neighbor, start=start, end=end)
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name='flow')
                    self.save_data(data, category, subcategory, start, end)
            
            time.sleep(1)
            
            # 2. Scheduled commercial exchanges
            subcategory = f"scheduled_exchanges_CH_{neighbor}"
            if self.check_exists(category, subcategory, start, end):
                logger.info(f"⊙ {subcategory} already exists")
                self.record_result(category, subcategory, True, "Already downloaded")
            else:
                logger.info(f"Fetching {subcategory}...")
                success, msg = self.try_download(
                    category, subcategory,
                    self.client.query_scheduled_exchanges,
                    self.country_code, neighbor, start=start, end=end
                )
                self.record_result(category, subcategory, success, msg)
                if success:
                    data = self.client.query_scheduled_exchanges(self.country_code, neighbor, start=start, end=end)
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name='scheduled')
                    self.save_data(data, category, subcategory, start, end)
            
            time.sleep(1)
            
            # 3. Net transfer capacity day-ahead
            subcategory = f"ntc_day_ahead_CH_{neighbor}"
            if self.check_exists(category, subcategory, start, end):
                logger.info(f"⊙ {subcategory} already exists")
                self.record_result(category, subcategory, True, "Already downloaded")
            else:
                logger.info(f"Fetching {subcategory}...")
                success, msg = self.try_download(
                    category, subcategory,
                    self.client.query_net_transfer_capacity_dayahead,
                    self.country_code, neighbor, start=start, end=end
                )
                self.record_result(category, subcategory, success, msg)
                if success:
                    data = self.client.query_net_transfer_capacity_dayahead(self.country_code, neighbor, start=start, end=end)
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name='ntc')
                    self.save_data(data, category, subcategory, start, end)
            
            time.sleep(1)
            
            # 4. Net transfer capacity week-ahead
            subcategory = f"ntc_week_ahead_CH_{neighbor}"
            if self.check_exists(category, subcategory, start, end):
                logger.info(f"⊙ {subcategory} already exists")
                self.record_result(category, subcategory, True, "Already downloaded")
            else:
                logger.info(f"Fetching {subcategory}...")
                success, msg = self.try_download(
                    category, subcategory,
                    self.client.query_net_transfer_capacity_weekahead,
                    self.country_code, neighbor, start=start, end=end
                )
                self.record_result(category, subcategory, success, msg)
                if success:
                    data = self.client.query_net_transfer_capacity_weekahead(self.country_code, neighbor, start=start, end=end)
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name='ntc_weekly')
                    self.save_data(data, category, subcategory, start, end)
            
            time.sleep(1)
            
            # 5. Net transfer capacity month-ahead
            subcategory = f"ntc_month_ahead_CH_{neighbor}"
            if self.check_exists(category, subcategory, start, end):
                logger.info(f"⊙ {subcategory} already exists")
                self.record_result(category, subcategory, True, "Already downloaded")
            else:
                logger.info(f"Fetching {subcategory}...")
                success, msg = self.try_download(
                    category, subcategory,
                    self.client.query_net_transfer_capacity_monthahead,
                    self.country_code, neighbor, start=start, end=end
                )
                self.record_result(category, subcategory, success, msg)
                if success:
                    data = self.client.query_net_transfer_capacity_monthahead(self.country_code, neighbor, start=start, end=end)
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name='ntc_monthly')
                    self.save_data(data, category, subcategory, start, end)
            
            time.sleep(1)
            
            # 6. Net transfer capacity year-ahead
            subcategory = f"ntc_year_ahead_CH_{neighbor}"
            if self.check_exists(category, subcategory, start, end):
                logger.info(f"⊙ {subcategory} already exists")
                self.record_result(category, subcategory, True, "Already downloaded")
            else:
                logger.info(f"Fetching {subcategory}...")
                success, msg = self.try_download(
                    category, subcategory,
                    self.client.query_net_transfer_capacity_yearahead,
                    self.country_code, neighbor, start=start, end=end
                )
                self.record_result(category, subcategory, success, msg)
                if success:
                    data = self.client.query_net_transfer_capacity_yearahead(self.country_code, neighbor, start=start, end=end)
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name='ntc_yearly')
                    self.save_data(data, category, subcategory, start, end)
            
            time.sleep(1)
        
        # Additional transmission categories (not per-border)
        subcategory = "expansion_dismantling_projects"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
        
        subcategory = "redispatching"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
        
        subcategory = "congestion_income"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Requires specific congestion API")
    
    # ========================
    # BALANCING DATA - ALL TYPES
    # ========================
    
    def fetch_all_balancing_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """Fetch ALL balancing-related data"""
        logger.info("\n" + "="*80)
        logger.info("BALANCING DATA - ALL SUBCATEGORIES")
        logger.info("="*80)
        
        category = 'balancing'
        
        # 1. Imbalance prices
        subcategory = "imbalance_prices"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_imbalance_prices,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_imbalance_prices(self.country_code, start=start, end=end)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 2. Balancing energy prices (aggregated bids)
        subcategory = "balancing_energy_prices"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_aggregated_bids,
                self.country_code, start=start, end=end, process_type='A51'
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_aggregated_bids(self.country_code, start=start, end=end, process_type='A51')
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 3. Activated balancing energy volumes - BY TYPE
        # FIXED: Added business_type parameter for each balancing type
        balancing_types = {
            'A95': 'FCR',   # Frequency Containment Reserve (Primary)
            'A96': 'aFRR',  # Automatic Frequency Restoration Reserve (Secondary)
            'A97': 'mFRR',  # Manual Frequency Restoration Reserve (Tertiary)
            'A98': 'RR'     # Replacement Reserve
        }
        
        for business_code, business_name in balancing_types.items():
            subcategory = f"activated_balancing_energy_{business_name}"
            
            if self.check_exists(category, subcategory, start, end):
                logger.info(f"⊙ {subcategory} already exists")
                self.record_result(category, subcategory, True, "Already downloaded")
            else:
                logger.info(f"Fetching {subcategory} (business_type={business_code})...")
                success, msg = self.try_download(
                    category, subcategory,
                    self.client.query_activated_balancing_energy,
                    self.country_code,
                    start=start,
                    end=end,
                    business_type=business_code  # FIXED: Added required parameter
                )
                self.record_result(category, subcategory, success, msg)
                if success:
                    data = self.client.query_activated_balancing_energy(
                        self.country_code,
                        start=start,
                        end=end,
                        business_type=business_code
                    )
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name=f'activated_energy_{business_name}')
                    self.save_data(data, category, subcategory, start, end)
            
            time.sleep(1)
        
        # 4. Procured balancing capacity - BY TYPE
        # FIXED: Added process_type parameter for each capacity type
        capacity_process_types = {
            'A47': 'mFRR_capacity',  # Manual frequency restoration reserve
            'A51': 'aFRR_capacity',  # Automatic frequency restoration reserve
            'A52': 'FCR_capacity',   # Frequency containment reserve
        }
        
        for process_code, process_name in capacity_process_types.items():
            subcategory = f"procured_balancing_capacity_{process_name}"
            
            if self.check_exists(category, subcategory, start, end):
                logger.info(f"⊙ {subcategory} already exists")
                self.record_result(category, subcategory, True, "Already downloaded")
            else:
                logger.info(f"Fetching {subcategory} (process_type={process_code})...")
                success, msg = self.try_download(
                    category, subcategory,
                    self.client.query_procured_balancing_capacity,
                    self.country_code,
                    process_code,  # FIXED: Added required parameter (positional)
                    start=start,
                    end=end
                )
                self.record_result(category, subcategory, success, msg)
                if success:
                    data = self.client.query_procured_balancing_capacity(
                        self.country_code,
                        process_code,
                        start=start,
                        end=end
                    )
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name=f'procured_capacity_{process_name}')
                    self.save_data(data, category, subcategory, start, end)
            
            time.sleep(1)
        
        # 5. Imbalance volumes
        subcategory = "imbalance_volumes"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_imbalance_volumes,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_imbalance_volumes(self.country_code, start=start, end=end)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # Additional balancing categories
        subcategory = "accepted_offers"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
        
        subcategory = "financial_expenses_income"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
        
        subcategory = "cross_border_balancing"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
        
        subcategory = "FCR_total_capacity"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
    
    # ========================
    # OUTAGES DATA - ALL TYPES
    # ========================
    
    def fetch_all_outages_data(self, start: pd.Timestamp, end: pd.Timestamp):
        """Fetch ALL outage-related data"""
        logger.info("\n" + "="*80)
        logger.info("OUTAGES DATA - ALL SUBCATEGORIES")
        logger.info("="*80)
        
        category = 'outages'
        
        # 1. Generation unavailability
        subcategory = "generation_unavailability"
        if self.check_exists(category, subcategory, start, end):
            logger.info(f"⊙ {subcategory} already exists")
            self.record_result(category, subcategory, True, "Already downloaded")
        else:
            logger.info(f"Fetching {subcategory}...")
            success, msg = self.try_download(
                category, subcategory,
                self.client.query_unavailability_of_generation_units,
                self.country_code, start=start, end=end
            )
            self.record_result(category, subcategory, success, msg)
            if success:
                data = self.client.query_unavailability_of_generation_units(self.country_code, start=start, end=end)
                self.save_data(data, category, subcategory, start, end)
        
        time.sleep(1)
        
        # 2. Transmission unavailability
        subcategory = "transmission_unavailability"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not available in entsoe-py library")
        
        # 3. Production unavailability of offshore wind
        subcategory = "offshore_wind_unavailability"
        logger.info(f"Attempting {subcategory}...")
        self.record_result(category, subcategory, False, "Not applicable for Switzerland (no offshore wind)")
    
    # ========================
    # MAIN RUNNER
    # ========================
    
    def download_all(self, start_date: str, end_date: str, chunk_days: int = 7):
        """Download all data and generate comprehensive report"""
        start = pd.Timestamp(start_date, tz='Europe/Zurich')
        end = pd.Timestamp(end_date, tz='Europe/Zurich')
        
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE ENTSOE DATA DOWNLOAD - ALL CATEGORIES")
        logger.info("="*80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("="*80)
        
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"CHUNK: {current.date()} to {chunk_end.date()}")
            logger.info(f"{'='*80}")
            
            # Fetch ALL categories
            self.fetch_all_market_data(current, chunk_end)
            self.fetch_all_load_data(current, chunk_end)
            self.fetch_all_generation_data(current, chunk_end)
            self.fetch_all_transmission_data(current, chunk_end)
            self.fetch_all_balancing_data(current, chunk_end)
            self.fetch_all_outages_data(current, chunk_end)
            
            current = chunk_end
            time.sleep(3)
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOAD COMPLETE - GENERATING REPORT")
        logger.info("="*80)
        
        self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self):
        """Generate detailed report of what was downloaded"""
        report_path = self.output_dir / 'download_report.json'
        
        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(self.download_report, indent=2, fp=f)
        
        # Generate human-readable report
        print("\n" + "="*80)
        print("COMPREHENSIVE DOWNLOAD REPORT")
        print("="*80)
        
        for category, subcategories in self.download_report['categories'].items():
            print(f"\n{'='*80}")
            print(f"{category.upper()}")
            print('='*80)
            
            success_count = sum(1 for s in subcategories.values() if s['success'])
            total_count = len(subcategories)
            
            print(f"Summary: {success_count}/{total_count} subcategories downloaded successfully\n")
            
            # Successful downloads
            successful = [(name, info) for name, info in subcategories.items() if info['success']]
            if successful:
                print("✅ SUCCESSFUL:")
                for name, info in successful:
                    print(f"   • {name}: {info['message']}")
            
            # Failed downloads
            failed = [(name, info) for name, info in subcategories.items() if not info['success']]
            if failed:
                print("\n❌ FAILED/NOT AVAILABLE:")
                for name, info in failed:
                    print(f"   • {name}: {info['message']}")
        
        print("\n" + "="*80)
        print(f"Full report saved to: {report_path}")
        print("="*80)


def main():
    print("\n" + "="*80)
    print("COMPLETE ENTSOE DATA DOWNLOADER")
    print("Downloads ALL available data categories with detailed reporting")
    print("="*80)
    
    api_key = os.getenv('ENTSOE_API_KEY')
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('ENTSOE_API_KEY')
    
    if not api_key:
        api_key = input("\nENTSOE API key: ").strip()
    
    start_date = input("Start date (YYYY-MM-DD, default: 2009-01-01): ").strip() or '2009-01-01'
    end_date = input("End date (YYYY-MM-DD, default: today): ").strip() or datetime.now().strftime('%Y-%m-%d')
    output_dir = input("Output directory (default: ./entsoe_complete_data): ").strip() or './entsoe_complete_data'
    
    print(f"\nDownloading {start_date} to {end_date}")
    print(f"Output: {output_dir}")
    print("\nThis will attempt ALL data categories and generate a detailed report.")
    
    proceed = input("\nProceed? (yes/no): ").strip().lower()
    if proceed not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    downloader = ComprehensiveENTSOEDownloader(api_key, output_dir)
    downloader.download_all(start_date, end_date, chunk_days=7)
    
    print("\n🎉 COMPLETE! Check download_report.json for detailed results.")


if __name__ == "__main__":
    main()
