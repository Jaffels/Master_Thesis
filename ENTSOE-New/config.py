"""
Configuration file for ENTSOE Ancillary Services Data Pipeline
"""

from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
ENTSOE_API_KEY = os.getenv('ENTSOE_API_KEY', '')

# Data Collection Parameters
COUNTRY_CODE = 'CH'  # Switzerland (Swissgrid)

# Neighboring countries for cross-border flow analysis
NEIGHBORING_COUNTRIES = ['DE', 'FR', 'IT', 'AT']

# Date range for data collection
# Recommended: 3-5 years for robust forecasting models
DEFAULT_START_DATE = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')

# API call parameters
CHUNK_DAYS = 7  # Number of days per API call (to avoid timeouts)
RATE_LIMIT_DELAY = 1  # Seconds between API calls

# Data Storage
OUTPUT_DIR = './swissgrid_ancillary_data'

# Directory structure
DATA_DIRS = {
    'balancing': 'balancing',
    'reserves': 'reserves',
    'imbalance': 'imbalance',
    'generation': 'generation',
    'load': 'load',
    'crossborder': 'crossborder',
    'processed': 'processed'
}

# Reserve Types
RESERVE_TYPES = {
    'A95': 'FCR',   # Frequency Containment Reserve (Primary Control)
    'A96': 'aFRR',  # Automatic Frequency Restoration Reserve (Secondary Control)
    'A97': 'mFRR'   # Manual Frequency Restoration Reserve (Tertiary Control)
}

# Feature Engineering Parameters

# Lag features (in number of timesteps)
# For 15-minute resolution:
# - 4 = 1 hour
# - 24 = 6 hours
# - 96 = 1 day
# - 672 = 1 week
LAG_PERIODS = [1, 2, 4, 24, 96, 672]

# Rolling window sizes (in timesteps)
ROLLING_WINDOWS = [4, 24, 96, 672]  # 1 hour, 6 hours, 1 day, 1 week

# Forecast Horizons (in timesteps)
# For price forecasting experiments
FORECAST_HORIZONS = {
    '15min': 1,
    '1hour': 4,
    '6hours': 24,
    '1day': 96
}

# Model Configuration

# Train/test split ratio
TRAIN_TEST_SPLIT = 0.8

# Cross-validation folds
CV_FOLDS = 5

# Target Variables for Forecasting
TARGET_VARIABLES = [
    'balancing_price',
    'imbalance_price',
    'FCR_price',
    'aFRR_price',
    'mFRR_price'
]

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Research-specific Parameters

# Response time analysis bins (in seconds)
# For analyzing activation speed of different reserve types
RESPONSE_TIME_BINS = [
    0,      # Instant
    30,     # FCR activation time (typically < 30s)
    300,    # aFRR activation time (typically < 5 min)
    900,    # mFRR activation time (typically < 15 min)
    1800    # 30 minutes
]

# Price spike threshold (for anomaly detection)
# Prices above this percentile are considered spikes
PRICE_SPIKE_PERCENTILE = 95

# Minimum data quality threshold
# Percentage of non-null values required
MIN_DATA_QUALITY = 0.95
