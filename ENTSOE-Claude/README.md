# ENTSOE Ancillary Services Data Pipeline

**Master's Thesis Project: Price Forecasting for Ancillary Services and Response Time Modeling**

This pipeline collects, processes, and prepares data from the ENTSOE Transparency Platform specifically for analyzing Swissgrid's ancillary services market.

## 🎯 Research Focus

- **Price Forecasting**: Predict balancing energy prices, imbalance prices, and reserve capacity prices (FCR, aFRR, mFRR)
- **Response Time Modeling**: Analyze activation patterns and response characteristics of different reserve types
- **Market Dynamics**: Understanding drivers of ancillary service needs and pricing

## 📊 Data Sources

### Primary Data (ENTSOE API)
- **Balancing Energy Prices**: Activation prices for upward/downward regulation
- **Imbalance Prices**: Penalties for system imbalances
- **Reserve Capacity**: FCR, aFRR, mFRR volumes and prices
- **Activated Balancing Energy**: Actual volumes activated
- **Generation Data**: Actual generation by production type
- **Load Data**: Total system load
- **Cross-Border Flows**: Imports/exports with DE, FR, IT, AT

### Reserve Types
1. **FCR** (Frequency Containment Reserve) - Primary control, ~30s activation
2. **aFRR** (Automatic Frequency Restoration Reserve) - Secondary control, <5 min
3. **mFRR** (Manual Frequency Restoration Reserve) - Tertiary control, <15 min

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Key Setup

Get your ENTSOE API key from: https://transparency.entsoe.eu/

Create a `.env` file:
```bash
ENTSOE_API_KEY=your-api-key-here
```

### 3. Run Data Collection

```python
from entsoe_pipeline import ENTSOEAncillaryServicesPipeline
from datetime import datetime, timedelta

# Initialize pipeline
pipeline = ENTSOEAncillaryServicesPipeline(
    api_key='your-api-key',
    output_dir='./swissgrid_ancillary_data'
)

# Collect last 3 years of data
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')

pipeline.run_full_collection(
    start_date=start_date,
    end_date=end_date,
    chunk_days=7
)
```

### 4. Process and Engineer Features

```python
from data_processing import AncillaryServicesFeatureEngineer

# Initialize feature engineer
engineer = AncillaryServicesFeatureEngineer(
    data_dir='./swissgrid_ancillary_data'
)

# Prepare forecasting dataset
X, y = engineer.prepare_forecasting_dataset(
    target_col='balancing_price',
    forecast_horizon=4  # 1 hour ahead for 15-min data
)

# Create response time dataset
response_df = engineer.create_response_time_dataset()
```

### 5. Exploratory Analysis

Open the Jupyter notebook:
```bash
jupyter notebook exploratory_analysis.ipynb
```

## 📁 Project Structure

```
.
├── entsoe_pipeline.py          # Main data collection pipeline
├── data_processing.py          # Feature engineering module
├── config.py                   # Configuration parameters
├── requirements.txt            # Python dependencies
├── exploratory_analysis.ipynb  # Jupyter notebook for EDA
├── .env                        # API credentials (create this)
└── swissgrid_ancillary_data/   # Output directory
    ├── balancing/              # Balancing energy data
    ├── reserves/               # FCR, aFRR, mFRR data
    ├── imbalance/              # Imbalance prices
    ├── generation/             # Generation by type
    ├── load/                   # System load
    ├── crossborder/            # Cross-border flows
    └── processed/              # Feature-engineered datasets
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Date ranges**: Adjust `DEFAULT_START_DATE` and `DEFAULT_END_DATE`
- **Lag features**: Modify `LAG_PERIODS` for different autoregressive lags
- **Rolling windows**: Change `ROLLING_WINDOWS` for trend analysis
- **Forecast horizons**: Set `FORECAST_HORIZONS` for different prediction timeframes

## 📈 Available Features

### Temporal Features
- Hour, day of week, month, season
- Peak/off-peak indicators
- Weekend/weekday flags
- Cyclical encodings (sin/cos transformations)

### Lag Features
- Autoregressive lags: 1, 2, 4, 24, 96, 672 timesteps
- For 15-min data: 15min, 30min, 1hr, 6hr, 1day, 1week

### Rolling Statistics
- Mean, standard deviation, min, max
- Windows: 1 hour, 6 hours, 1 day, 1 week

### System State Features
- Renewable generation share
- Load-generation balance
- Net import/export position
- Reserve capacity availability

## 🎓 Research Applications

### 1. Price Forecasting Models

**Recommended approaches:**
- **LSTM/GRU**: Capture long-term dependencies in time series
- **XGBoost/LightGBM**: Handle non-linearities and feature interactions
- **ARIMAX/SARIMAX**: Classical time series with exogenous variables
- **Ensemble**: Combine multiple model types

**Example workflow:**
```python
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# Prepare data
X, y = engineer.prepare_forecasting_dataset(
    target_col='balancing_price',
    forecast_horizon=4
)

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Train model
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.8
)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    # Evaluate...
```

### 2. Response Time Analysis

**Key questions:**
- How quickly do different reserve types activate?
- What system conditions trigger faster/slower responses?
- Are there seasonal patterns in activation timing?

**Example analysis:**
```python
# Load response time data
response_df = engineer.create_response_time_dataset()

# Analyze activation patterns by reserve type
# Group by system stress indicators
# Model response time as a function of system state
```

### 3. Price Spike Detection

**Anomaly detection:**
```python
from sklearn.ensemble import IsolationForest

# Identify abnormal pricing events
iso_forest = IsolationForest(contamination=0.05)
anomalies = iso_forest.fit_predict(X)

# Analyze characteristics of price spikes
spike_conditions = X[anomalies == -1]
```

## 📊 Data Quality Considerations

- **Missing data**: ENTSOE API may have gaps for certain data types
- **Frequency**: Most data at 15-minute resolution (96 data points/day)
- **Availability**: Some reserve data may be reported at hourly or daily resolution
- **Updates**: Historical data is stable, but recent data may be revised

## 🔬 Advanced Topics

### Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_regression

# Select top K features
selector = SelectKBest(f_regression, k=50)
X_selected = selector.fit_transform(X, y)
```

### Regime Detection
```python
from hmmlearn import hmm

# Hidden Markov Model for market regimes
model = hmm.GaussianHMM(n_components=3)
model.fit(price_data)
regimes = model.predict(price_data)
```

### Causality Testing
```python
from statsmodels.tsa.stattools import grangercausalitytests

# Test if renewable generation Granger-causes price volatility
grangercausalitytests(data[['price', 'renewable_share']], maxlag=96)
```

## 📝 Thesis Structure Recommendations

1. **Introduction**: Ancillary services market overview, research questions
2. **Literature Review**: Existing forecasting approaches, reserve markets
3. **Data & Methodology**: This pipeline, feature engineering, model selection
4. **Results**: 
   - Price forecasting performance by horizon
   - Feature importance analysis
   - Response time patterns
   - Comparative model evaluation
5. **Discussion**: Market insights, policy implications, limitations
6. **Conclusion**: Contributions, future work

## 🐛 Troubleshooting

### API Rate Limits
If you hit rate limits, increase `RATE_LIMIT_DELAY` in `config.py`.

### Missing Data
Some ENTSOE endpoints may not have Swiss data for all periods. Check data availability first:
```python
# Test availability
from entsoe import EntsoePandasClient
client = EntsoePandasClient(api_key='your-key')

# Try a small date range first
test_data = client.query_balancing_prices('CH', start, end)
```

### Memory Issues
For large datasets, process in smaller chunks:
```python
# Process monthly instead of yearly
pipeline.run_full_collection(
    start_date='2023-01-01',
    end_date='2023-01-31',
    chunk_days=7
)
```

## 📚 References

- ENTSOE Transparency Platform: https://transparency.entsoe.eu/
- ENTSOE API Documentation: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
- Swissgrid: https://www.swissgrid.ch/
- entsoe-py library: https://github.com/EnergieID/entsoe-py

## 📧 Support

For issues with:
- **ENTSOE API**: Check their documentation and status page
- **This pipeline**: Review code comments and configuration
- **Data interpretation**: Consult ENTSOE terminology and Swissgrid documentation

## 🎉 Good luck with your thesis!

This pipeline provides a solid foundation for your research. Focus on:
1. Understanding the data patterns
2. Selecting appropriate forecasting models
3. Interpreting results in the context of ancillary services markets
4. Drawing meaningful conclusions for grid operators and policy makers

Happy researching! 🚀
