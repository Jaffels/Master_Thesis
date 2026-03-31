# thesis_weather_data.py
# Extracts weather features relevant for electricity demand & ancillary services
# price forecasting using MeteoSwissDataFactory.

import pandas as pd
from meteoswiss_api import MeteoSwissDataFactory

# ============================================================
# CONFIGURATION — edit these values
# ============================================================

YEAR_START = 2019
YEAR_END   = 2024   # Adjust to match your Swissgrid price data range

# Swiss load centres — one station will be auto-selected per city
LOCATIONS = {
    "Zurich":  (47.3769, 8.5417),
    "Bern":    (46.9481, 7.4474),
    "Geneva":  (46.2044, 6.1432),
    "Basel":   (47.5596, 7.5886),
    "Luzern":  (47.0502, 8.3093),
}

OUTPUT_FILE = "meteoswiss_features_for_thesis.csv"

# ============================================================
# HELPER: Compute derived features useful for demand modeling
# ============================================================

def compute_features(df_T: pd.DataFrame, df_I: pd.DataFrame) -> pd.DataFrame:
    """
    Merge temperature + irradiation and compute thesis-relevant features:
    - HDD / CDD  : Heating/Cooling Degree Hours (base 18 °C)
    - T_range    : Intraday temperature swing
    - irr_ramp   : Hour-on-hour irradiation change (PV volatility proxy)
    """
    df = df_T.join(df_I, how="outer")

    BASE_TEMP = 18.0  # Standard base temperature for degree-day calculation

    df["HDD"] = (BASE_TEMP - df["T_mean_degC"]).clip(lower=0)
    df["CDD"] = (df["T_mean_degC"] - BASE_TEMP).clip(lower=0)
    df["T_range_degC"] = df["T_max_degC"] - df["T_min_degC"]
    df["irr_ramp_W_per_m2"] = df["I_global_mean_W_per_m2"].diff()

    return df

# ============================================================
# MAIN LOOP — fetch data for all cities and combine
# ============================================================

all_dfs = []

for city, (lat, lon) in LOCATIONS.items():
    print(f"\n>>> Fetching data for {city} ({lat}, {lon}) ...")

    try:
        factory = MeteoSwissDataFactory(
            latitude=lat,
            longitude=lon,
            year_start=YEAR_START,
            year_end=YEAR_END,
            avoid_crest=True,   # Exclude mountain stations — not representative
        )

        station_info = factory.info_selected_station().iloc[0]
        print(f"    Selected station: {station_info['station_abbr']} — {station_info['station_name']}")

        df_T = factory.get_hourly_data_temperature()   # T_mean, T_min, T_max [°C]
        df_I = factory.get_hourly_data_irradiation()   # I_global, I_diff [W/m²]

        df_city = compute_features(df_T, df_I)
        df_city["city"]    = city
        df_city["station"] = station_info["station_abbr"]

        all_dfs.append(df_city)

    except Exception as e:
        print(f"    WARNING: Failed for {city}: {e}")

# ============================================================
# COMBINE & EXPORT
# ============================================================

df_all = pd.concat(all_dfs).reset_index()
df_all = df_all.rename(columns={"time": "timestamp"})
df_all = df_all.sort_values(["city", "timestamp"])

df_all.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved {len(df_all):,} rows to '{OUTPUT_FILE}'")
print(df_all.head(10))
print("\nColumns:", list(df_all.columns))
