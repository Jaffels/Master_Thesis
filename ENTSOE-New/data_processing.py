"""
Data Processing and Feature Engineering for Ancillary Services Forecasting

This module:
1. Loads collected ENTSOE data
2. Combines multiple data sources
3. Engineers features for ML models
4. Prepares time series for forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AncillaryServicesFeatureEngineer:
    """
    Feature engineering for ancillary services price and response time forecasting
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize with data directory
        
        Args:
            data_dir: Root directory containing collected data
        """
        self.data_dir = Path(data_dir)
        self.dirs = {
            'balancing': self.data_dir / 'balancing',
            'reserves': self.data_dir / 'reserves',
            'imbalance': self.data_dir / 'imbalance',
            'generation': self.data_dir / 'generation',
            'load': self.data_dir / 'load',
            'crossborder': self.data_dir / 'crossborder',
            'processed': self.data_dir / 'processed'
        }
    
    def load_all_files(self, data_type: str) -> pd.DataFrame:
        """
        Load and concatenate all parquet files of a given type
        
        Args:
            data_type: Type of data (balancing, reserves, imbalance, etc.)
        
        Returns:
            Combined DataFrame
        """
        dir_path = self.dirs[data_type]
        files = list(dir_path.glob('*.parquet'))
        
        if not files:
            logger.warning(f"No files found for {data_type}")
            return pd.DataFrame()
        
        logger.info(f"Loading {len(files)} files for {data_type}")
        
        dfs = []
        for file in sorted(files):
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if dfs:
            combined = pd.concat(dfs, axis=0)
            combined = combined.sort_index()
            # Remove duplicates if any
            combined = combined[~combined.index.duplicated(keep='first')]
            logger.info(f"Loaded {len(combined)} records for {data_type}")
            return combined
        else:
            return pd.DataFrame()
    
    def create_master_dataset(self) -> pd.DataFrame:
        """
        Create master dataset combining all data sources
        
        Returns:
            Master DataFrame with all features aligned by timestamp
        """
        logger.info("Creating master dataset...")
        
        # Load all data types
        balancing_prices = self.load_all_files('balancing')
        imbalance_prices = self.load_all_files('imbalance')
        generation = self.load_all_files('generation')
        load = self.load_all_files('load')
        
        # Start with a time index (15-min resolution typical for balancing)
        if not balancing_prices.empty:
            time_index = balancing_prices.index
        elif not imbalance_prices.empty:
            time_index = imbalance_prices.index
        elif not load.empty:
            time_index = load.index
        else:
            logger.error("No data available to create master dataset")
            return pd.DataFrame()
        
        # Create master dataframe
        master = pd.DataFrame(index=time_index)
        
        # Add balancing prices (if multiple columns, rename them)
        if not balancing_prices.empty:
            if isinstance(balancing_prices, pd.DataFrame):
                for col in balancing_prices.columns:
                    master[f'balancing_price_{col}'] = balancing_prices[col]
            else:
                master['balancing_price'] = balancing_prices
        
        # Add imbalance prices
        if not imbalance_prices.empty:
            if isinstance(imbalance_prices, pd.DataFrame):
                for col in imbalance_prices.columns:
                    master[f'imbalance_price_{col}'] = imbalance_prices[col]
            else:
                master['imbalance_price'] = imbalance_prices
        
        # Add generation data
        if not generation.empty:
            if isinstance(generation, pd.DataFrame):
                for col in generation.columns:
                    master[f'generation_{col}'] = generation[col]
            else:
                master['generation_total'] = generation
        
        # Add load data
        if not load.empty:
            master['load'] = load
        
        # Load reserve data (FCR, aFRR, mFRR)
        for reserve_type in ['FCR', 'aFRR', 'mFRR']:
            reserve_files = list(self.dirs['reserves'].glob(f'{reserve_type}_*.parquet'))
            if reserve_files:
                reserve_dfs = [pd.read_parquet(f) for f in reserve_files]
                reserve_data = pd.concat(reserve_dfs, axis=0).sort_index()
                reserve_data = reserve_data[~reserve_data.index.duplicated(keep='first')]
                
                if isinstance(reserve_data, pd.DataFrame):
                    for col in reserve_data.columns:
                        master[f'{reserve_type}_{col}'] = reserve_data[col]
                else:
                    master[f'{reserve_type}'] = reserve_data
        
        # Load cross-border flows
        crossborder_files = list(self.dirs['crossborder'].glob('*.parquet'))
        for file in crossborder_files:
            neighbor = file.stem.split('_')[2]  # Extract country code
            flow_data = pd.read_parquet(file)
            
            # Handle both Series and DataFrame
            if isinstance(flow_data, pd.DataFrame):
                # Use first column or 'flow' column
                if 'flow' in flow_data.columns:
                    master[f'flow_{neighbor}'] = flow_data['flow']
                else:
                    master[f'flow_{neighbor}'] = flow_data.iloc[:, 0]
            else:
                master[f'flow_{neighbor}'] = flow_data
        
        logger.info(f"Master dataset created with shape: {master.shape}")
        logger.info(f"Columns: {master.columns.tolist()}")
        
        return master
    
    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features (hour, day, week, month, season, etc.)
        
        Critical for: Capturing seasonality patterns in prices
        """
        df = df.copy()
        
        # Basic temporal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Time of day categories
        df['is_peak_hour'] = df['hour'].isin(range(8, 20)).astype(int)  # 8am-8pm
        df['is_super_peak'] = df['hour'].isin(range(17, 21)).astype(int)  # 5pm-9pm
        df['is_night'] = df['hour'].isin(range(0, 6)).astype(int)  # midnight-6am
        
        # Weekend/weekday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Season (meteorological)
        df['season'] = df['month'] % 12 // 3 + 1  # 1=winter, 2=spring, 3=summer, 4=fall
        
        # Cyclical encoding for hour (important for ML models)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        logger.info("Added temporal features")
        return df
    
    def engineer_lag_features(self, df: pd.DataFrame, 
                            target_cols: List[str],
                            lags: List[int] = [1, 2, 4, 24, 96]) -> pd.DataFrame:
        """
        Create lagged features for autoregressive patterns
        
        Args:
            df: Input dataframe
            target_cols: Columns to create lags for
            lags: List of lag periods (in number of timesteps)
                  For 15-min data: 4 = 1 hour, 96 = 1 day, 672 = 1 week
        
        Critical for: Capturing autocorrelation in prices
        """
        df = df.copy()
        
        for col in target_cols:
            if col not in df.columns:
                continue
                
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"Added lag features for {len(target_cols)} columns")
        return df
    
    def engineer_rolling_features(self, df: pd.DataFrame,
                                 cols: List[str],
                                 windows: List[int] = [4, 24, 96]) -> pd.DataFrame:
        """
        Create rolling statistics (mean, std, min, max)
        
        Args:
            df: Input dataframe
            cols: Columns to compute rolling stats for
            windows: Rolling window sizes (in timesteps)
        
        Critical for: Capturing recent trends and volatility
        """
        df = df.copy()
        
        for col in cols:
            if col not in df.columns:
                continue
            
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                
                # Rolling std (volatility)
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                
                # Rolling min/max
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
        
        logger.info(f"Added rolling features for {len(cols)} columns")
        return df
    
    def engineer_system_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features describing system state
        
        Critical for: Understanding drivers of ancillary service needs
        """
        df = df.copy()
        
        # Renewable penetration (if generation data available)
        renewable_cols = [col for col in df.columns if any(x in col.lower() for x in ['solar', 'wind', 'hydro'])]
        if renewable_cols:
            df['renewable_generation'] = df[renewable_cols].sum(axis=1)
            if 'generation_total' in df.columns:
                df['renewable_share'] = df['renewable_generation'] / df['generation_total']
        
        # Load-generation balance (proxy for system stress)
        if 'load' in df.columns and 'generation_total' in df.columns:
            df['load_gen_delta'] = df['load'] - df['generation_total']
            df['load_gen_ratio'] = df['load'] / (df['generation_total'] + 1e-6)
        
        # Net cross-border position
        flow_cols = [col for col in df.columns if col.startswith('flow_')]
        if flow_cols:
            df['net_import'] = df[flow_cols].sum(axis=1)
        
        # Reserve scarcity indicators (if reserve data available)
        for reserve_type in ['FCR', 'aFRR', 'mFRR']:
            capacity_cols = [col for col in df.columns if reserve_type in col and 'capacity' in col.lower()]
            if capacity_cols:
                df[f'{reserve_type}_total_capacity'] = df[capacity_cols].sum(axis=1)
        
        logger.info("Added system state features")
        return df
    
    def prepare_forecasting_dataset(self, 
                                   target_col: str,
                                   forecast_horizon: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare final dataset for forecasting
        
        Args:
            target_col: Column to forecast
            forecast_horizon: Number of timesteps ahead to forecast
        
        Returns:
            X (features), y (target)
        """
        # Create master dataset
        df = self.create_master_dataset()
        
        # Add all features
        df = self.engineer_temporal_features(df)
        
        # Identify price and volume columns for lag/rolling features
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        volume_cols = [col for col in df.columns if any(x in col.lower() for x in ['load', 'generation', 'flow'])]
        
        df = self.engineer_lag_features(df, price_cols + volume_cols, lags=[1, 2, 4, 24, 96])
        df = self.engineer_rolling_features(df, price_cols + volume_cols, windows=[4, 24, 96])
        df = self.engineer_system_state_features(df)
        
        # Create forecast target
        df[f'{target_col}_target'] = df[target_col].shift(-forecast_horizon)
        
        # Drop rows with NaN
        df_clean = df.dropna()
        
        # Separate features and target
        target = df_clean[f'{target_col}_target']
        features = df_clean.drop(columns=[f'{target_col}_target'])
        
        # Save processed data
        output_file = self.dirs['processed'] / f'forecasting_dataset_{target_col}.parquet'
        df_clean.to_parquet(output_file)
        logger.info(f"Saved forecasting dataset: {output_file}")
        
        logger.info(f"Final dataset shape: {features.shape}")
        logger.info(f"Target shape: {target.shape}")
        
        return features, target
    
    def create_response_time_dataset(self) -> pd.DataFrame:
        """
        Create dataset for response time modeling
        
        Focuses on activation timing of different reserve types
        """
        # Load activated balancing energy
        activated_files = list(self.dirs['balancing'].glob('activated_volumes_*.parquet'))
        
        if not activated_files:
            logger.warning("No activated balancing energy data available")
            return pd.DataFrame()
        
        dfs = [pd.read_parquet(f) for f in activated_files]
        activated = pd.concat(dfs, axis=0).sort_index()
        activated = activated[~activated.index.duplicated(keep='first')]
        
        # Add temporal features
        activated_df = pd.DataFrame(index=activated.index)
        
        if isinstance(activated, pd.DataFrame):
            for col in activated.columns:
                activated_df[f'activated_{col}'] = activated[col]
        else:
            activated_df['activated_volume'] = activated
        
        activated_df = self.engineer_temporal_features(activated_df)
        
        # Add system state at activation time
        master = self.create_master_dataset()
        
        # Merge with master to get system state
        response_df = activated_df.join(master, how='left')
        
        # Save
        output_file = self.dirs['processed'] / 'response_time_dataset.parquet'
        response_df.to_parquet(output_file)
        logger.info(f"Saved response time dataset: {output_file}")
        
        return response_df


if __name__ == "__main__":
    # Example usage
    engineer = AncillaryServicesFeatureEngineer(
        data_dir='./swissgrid_ancillary_data'
    )
    
    # Create forecasting dataset for balancing prices
    X, y = engineer.prepare_forecasting_dataset(
        target_col='balancing_price',  # Adjust based on actual column name
        forecast_horizon=4  # 1 hour ahead for 15-min data
    )
    
    # Create response time dataset
    response_df = engineer.create_response_time_dataset()
