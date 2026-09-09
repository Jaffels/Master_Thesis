# Data Dictionary - ENTSOE Ancillary Services

## Overview
This document describes the data collected from ENTSOE Transparency Platform for Swissgrid ancillary services analysis.

---

## 1. Balancing Energy Data

### Balancing Prices
**File location**: `balancing/balancing_prices_*.parquet`

**Description**: Prices paid for activated balancing energy (upward and downward regulation).

**Resolution**: Typically 15-minute intervals

**Columns**:
- `timestamp`: DateTime index
- Various direction/product combinations (up/down regulation)

**Units**: EUR/MWh

**Use case**: Primary target variable for price forecasting

**Notes**: 
- Positive prices indicate cost of upward regulation (increasing generation)
- May include separate columns for different balancing products

---

### Activated Balancing Energy Volumes
**File location**: `balancing/activated_volumes_*.parquet`

**Description**: Actual volumes of balancing energy activated by the TSO.

**Resolution**: 15-minute intervals

**Columns**:
- `timestamp`: DateTime index
- Volume by direction (upward/downward)

**Units**: MWh

**Use case**: Understanding actual system imbalances and activation patterns

**Notes**:
- High activation volumes indicate system stress
- Useful for response time modeling

---

## 2. Reserve Capacity Data

### FCR (Frequency Containment Reserve)
**File location**: `reserves/FCR_*.parquet`

**Description**: Primary control reserve procured to maintain frequency within ±200 mHz.

**Activation time**: < 30 seconds

**Resolution**: Variable (hourly or daily auction results)

**Columns**:
- `timestamp`: DateTime index
- `capacity`: Procured capacity (MW)
- `price`: Capacity payment (EUR/MW/period)

**Use case**: 
- Reserve capacity price forecasting
- Understanding primary control market

**Notes**:
- Always activated automatically via droop control
- Symmetric product (must provide both up and down)

---

### aFRR (Automatic Frequency Restoration Reserve)
**File location**: `reserves/aFRR_*.parquet`

**Description**: Secondary control reserve activated automatically to restore frequency to 50 Hz.

**Activation time**: < 5 minutes (full activation)

**Resolution**: Variable (hourly or daily auction results)

**Columns**:
- `timestamp`: DateTime index
- `capacity`: Procured capacity (MW)
- `price`: Capacity payment (EUR/MW/period)
- May have separate upward/downward products

**Use case**:
- Reserve capacity price forecasting
- Response time modeling
- Understanding automatic secondary control

**Notes**:
- Activated based on Area Control Error (ACE)
- May be asymmetric (separate up/down products)

---

### mFRR (Manual Frequency Restoration Reserve)
**File location**: `reserves/mFRR_*.parquet`

**Description**: Tertiary control reserve activated manually by TSO operators.

**Activation time**: < 15 minutes

**Resolution**: Variable (hourly or daily auction results)

**Columns**:
- `timestamp`: DateTime index
- `capacity`: Procured capacity (MW)
- `price`: Capacity payment (EUR/MW/period)
- Separate upward/downward products

**Use case**:
- Reserve capacity price forecasting
- Manual activation pattern analysis

**Notes**:
- Slowest reserve type but largest volumes
- Used for longer-duration imbalances
- Energy payments separate from capacity payments

---

## 3. Imbalance Pricing

### Imbalance Prices
**File location**: `imbalance/imbalance_prices_*.parquet`

**Description**: Prices charged/paid to balance responsible parties for their imbalances.

**Resolution**: 15-minute intervals (imbalance settlement period)

**Columns**:
- `timestamp`: DateTime index
- `imbalance_price_up`: Price for short positions (EUR/MWh)
- `imbalance_price_down`: Price for long positions (EUR/MWh)

**Use case**:
- Understanding incentives for balanced schedules
- Relationship with balancing activation prices

**Notes**:
- Single pricing vs. dual pricing depends on market design
- Often related to marginal balancing energy price
- High imbalance prices encourage accurate forecasting

---

## 4. Generation Data

### Actual Generation
**File location**: `generation/generation_*.parquet`

**Description**: Actual electricity generation by production type.

**Resolution**: 15-minute or hourly intervals

**Columns**:
- `timestamp`: DateTime index
- Multiple columns for different production types:
  - `Biomass`: Biomass generation
  - `Fossil Brown coal/Lignite`: Lignite generation
  - `Fossil Gas`: Natural gas generation
  - `Fossil Hard coal`: Hard coal generation
  - `Fossil Oil`: Oil generation
  - `Geothermal`: Geothermal generation
  - `Hydro Pumped Storage`: Pumped hydro (generating)
  - `Hydro Run-of-river and poundage`: Run-of-river hydro
  - `Hydro Water Reservoir`: Reservoir hydro
  - `Nuclear`: Nuclear generation
  - `Other`: Other generation types
  - `Other renewable`: Other renewable sources
  - `Solar`: Solar PV generation
  - `Waste`: Waste-to-energy
  - `Wind Onshore`: Onshore wind
  - `Wind Offshore`: Offshore wind

**Units**: MW (power) or MWh (energy)

**Use case**:
- Calculate renewable penetration
- Understand system flexibility
- Predictor for balancing needs

**Notes**:
- Not all production types available in Switzerland
- High renewable share correlates with higher balancing needs
- Nuclear provides baseload

---

## 5. Load Data

### Total Load
**File location**: `load/load_*.parquet`

**Description**: Total electricity consumption in Switzerland.

**Resolution**: 15-minute intervals

**Columns**:
- `timestamp`: DateTime index
- `Actual Load`: Total load (MW)

**Units**: MW

**Use case**:
- System state indicator
- Load forecast errors drive imbalances
- Strong daily/weekly/seasonal patterns

**Notes**:
- Peak demand typically 17:00-19:00 on weekdays
- Seasonal pattern: higher in winter
- Weekend demand significantly lower

---

## 6. Cross-Border Flows

### Physical Flows
**File location**: `crossborder/flow_CH_XX_*.parquet` (where XX = DE, FR, IT, AT)

**Description**: Scheduled and actual cross-border electricity flows.

**Resolution**: 15-minute or hourly intervals

**Columns**:
- `timestamp`: DateTime index
- `flow`: Power flow (MW)

**Units**: MW (positive = import, negative = export)

**Convention**: 
- Positive values = import to Switzerland
- Negative values = export from Switzerland

**Use case**:
- Net import position as system stress indicator
- Interconnection constraints affect balancing
- Flow forecast errors contribute to imbalances

**Neighbors**:
- **DE (Germany)**: Largest electricity market, high renewable penetration
- **FR (France)**: Nuclear-heavy system
- **IT (Italy)**: High demand, limited domestic generation
- **AT (Austria)**: Hydro-heavy system, similar to CH

**Notes**:
- Unplanned flow deviations require balancing actions
- Cross-border balancing cooperation exists
- Interconnection capacity is limited

---

## 7. Processed/Engineered Features

### Temporal Features
Created by `data_processing.py`:

- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `day_of_month`: Day of month (1-31)
- `week_of_year`: Week number (1-52/53)
- `month`: Month (1-12)
- `quarter`: Quarter (1-4)
- `year`: Year
- `is_peak_hour`: Binary (1 if 8:00-20:00)
- `is_super_peak`: Binary (1 if 17:00-21:00)
- `is_night`: Binary (1 if 0:00-6:00)
- `is_weekend`: Binary (1 if Saturday/Sunday)
- `season`: Season (1=winter, 2=spring, 3=summer, 4=fall)
- `hour_sin`, `hour_cos`: Cyclical encoding of hour
- `dow_sin`, `dow_cos`: Cyclical encoding of day of week
- `month_sin`, `month_cos`: Cyclical encoding of month

### Lag Features
Format: `{column}_lag_{n}`

Examples:
- `balancing_price_lag_1`: Previous 15-min value
- `balancing_price_lag_4`: 1 hour ago (4 × 15 min)
- `balancing_price_lag_96`: 1 day ago (96 × 15 min)
- `balancing_price_lag_672`: 1 week ago (672 × 15 min)

### Rolling Window Features
Format: `{column}_rolling_{stat}_{window}`

Statistics:
- `mean`: Rolling average
- `std`: Rolling standard deviation (volatility)
- `min`: Rolling minimum
- `max`: Rolling maximum

Windows (in timesteps):
- 4: 1 hour
- 24: 6 hours
- 96: 1 day
- 672: 1 week

Examples:
- `balancing_price_rolling_mean_4`: 1-hour moving average
- `balancing_price_rolling_std_96`: 1-day price volatility

### System State Features

- `renewable_generation`: Sum of solar, wind, hydro
- `renewable_share`: Renewable gen / total gen
- `load_gen_delta`: Load - generation (imbalance proxy)
- `load_gen_ratio`: Load / generation
- `net_import`: Sum of all cross-border flows
- `{FCR/aFRR/mFRR}_total_capacity`: Total procured reserve

---

## Data Quality Notes

### Completeness
- **Load data**: Usually 99%+ complete
- **Generation data**: 95-99% complete
- **Balancing prices**: 90-95% complete (depends on activations)
- **Reserve data**: Variable, auction-based (daily/hourly resolution)
- **Cross-border flows**: 95-99% complete

### Resolution
- Most real-time data: 15-minute intervals (96 points/day)
- Some reserve auctions: Hourly or daily
- Need to handle mixed resolutions in modeling

### Timezone
- All data in UTC by default
- Swiss local time: UTC+1 (winter) or UTC+2 (summer)
- Important for temporal feature engineering

### Data Revisions
- Near-real-time data may be revised
- Historical data (> 1 week old) is stable
- For thesis: focus on stable historical data

---

## Key Relationships

### Price Drivers
1. **Imbalance magnitude** → Higher balancing activation
2. **Renewable variability** → Higher forecast errors → More imbalances
3. **Time of day** → Demand patterns affect prices
4. **Reserve scarcity** → Higher reserve prices
5. **Cross-border constraints** → Limited balancing resources

### Response Time Patterns
1. **FCR**: Continuous automatic activation
2. **aFRR**: Activated within minutes for ACE correction
3. **mFRR**: Manual activation for predictable needs

### Seasonal Patterns
- **Winter**: Higher load, higher prices
- **Summer**: Higher solar, more renewable variability
- **Spring/Fall**: Hydro availability, moderate prices

---

## References

- ENTSOE Glossary: https://www.entsoe.eu/data/glossary/
- Swissgrid Ancillary Services: https://www.swissgrid.ch/en/home/operation/ancillary-services.html
- ENTSOE Data Portal: https://transparency.entsoe.eu/

---

## For Thesis Use

### Recommended Variables for Price Forecasting

**Target variables**:
- Balancing activation prices (most granular, 15-min)
- Reserve capacity prices (strategic, hourly/daily)

**Key predictors**:
- Lagged prices (autocorrelation)
- Temporal features (seasonality)
- Load and generation (system state)
- Renewable share (variability driver)
- Net imports (interconnection stress)

### Recommended Variables for Response Time Modeling

**Target variable**:
- Time from imbalance to full activation

**Key predictors**:
- Reserve type (FCR/aFRR/mFRR)
- Magnitude of imbalance
- Available reserve capacity
- Time of day
- System stress indicators

---

## Version
Last updated: 2024
Based on ENTSOE Transparency Platform API v1
