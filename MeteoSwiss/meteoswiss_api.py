# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:10:28 2026

@author: UeliSchilt
"""

import pandas as pd
import numpy as np
import requests

class MeteoSwissDataFactory:
    """
    Automatic download of MeteoSwiss data from the Automatic Weather Staions data
    via the STAC-API.

    Return hourly weather data from an Automatic MeteoSwiss weather station. The station
    closest to the provided coordinates is selected automatically and data for the
    selected year returned.
    
    Info on the automatically selected station can be found via the
    info_selected_station() function.

    Links:

    General information:
    https://opendatadocs.meteoswiss.ch/a-data-groundbased/a1-automatic-weather-stations    

    How to download files (implemented here):
    https://opendatadocs.meteoswiss.ch/general/download#how-to-download-files-automatically

    STAC-API:
    https://www.geo.admin.ch/en/rest-interface-stac-api


    Some info on the query structure:
    (more details in links above)

    Data is structured into collections, items, assets.
    Here we only use the collection for "Automatic weather stations"
    (ch.meteoschweiz.ogd-smn). Items in this case are the different weather
    stations, which can be accessed via the featureId (e.g., for the station
    Adelboden (ABO), the featureId is 'abo'). A specific dataset within an item
    is called an assed. For example, the asset 'ogd-smn_abo_d_historical.csv'
    contains daily histrocial data. The key 'href' provides the URL for downloading
    the data file (which is read here as a df).

    Parameters:
    A complete list of parameters can be obtained via the
    info_parameters_meta_data() function.
        
    tre200h0:	Lufttemperatur 2 m über Boden; Stundenmittel
    tre200hn:	Lufttemperatur 2 m über Boden; Stundenminimum
    tre200hx:	Lufttemperatur 2 m über Boden; Stundenmaximum

    gre000h0:	Globalstrahlung; Stundenmittel
    """
    
    def __init__(
            self,
            latitude,
            longitude,
            year_start,
            year_end,
            avoid_crest=False,
            avoid_stations=[],
            ):
                
        self._latitude = latitude
        self._longitude = longitude
        self._year_start = year_start
        self._year_end = year_end
        
        self._df_hourly = None
        self._df_daily = None
        self._df_10min = None
        self._df_monthly = None
        self._df_yearly = None
        
        self._BASE = "https://data.geo.admin.ch/api/stac/v1"
        self._collectionId = 'ch.meteoschweiz.ogd-smn'

        collection_response = (
            requests.get(f"{self._BASE}/collections/{self._collectionId}")
            )
        collection_response.raise_for_status()

        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # Server responses:
        # 200 → OK
        # 404 → Not found
        # 403 → Forbidden
        # 500 → Server error
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        # Get URL to station file:
        stations_url = (
            collection_response
            .json()['assets']['ogd-smn_meta_stations.csv']['href']
            )
        # Read weather station meta file:
        self._df_meta_stations = (
            pd.read_csv(stations_url, sep=";", encoding="cp1252")
            )
        
        # Get URL to data inventory meta file:        
        datainventory_url = (
            collection_response
            .json()['assets']['ogd-smn_meta_datainventory.csv']['href']
            )
        # Read datainventory meta file:
        self._df_meta_datainventory = (
            pd.read_csv(datainventory_url, sep=";", encoding="cp1252")
            )
        
        # Get URL to parameters meta file:        
        parameters_url = (
            collection_response
            .json()['assets']['ogd-smn_meta_parameters.csv']['href']
            )
        # Read parameters meta file:
        self._df_meta_parameters = (
            pd.read_csv(parameters_url, sep=";", encoding="cp1252")
            )
        
        # Find weather station closest to provided coordinates:
        self._station = self.__find_closest_station(
            latitude=self._latitude,
            longitude=longitude,
            df_meta_stations=self._df_meta_stations,
            avoid_crest=avoid_crest,
            avoid_stations=avoid_stations,
            )       
        
        self._featureId = self._station.lower() # Example: Station Adelboden (ABO) has featureId 'abo'
        
    def __fetch_raw_data(self, resolution):
        
        decade_start = self.__decade_start(self._year_start)
        decade_end = int(decade_start + 9)
        decade_check = self.__decade_start(self._year_end)

        loop_stop_flag = 0
        df_data = None
        
        while loop_stop_flag == 0:
            
            if resolution=='h': # hourly
                assetId = f'ogd-smn_{self._featureId}_h_historical_{decade_start}-{decade_end}.csv'
            elif resolution=='d': # daily
                assetId = f'ogd-smn_{self._featureId}_d_historical.csv'
            elif resolution=='t': # 10min
                assetId = f'ogd-smn_{self._featureId}_t_historical_{decade_start}-{decade_end}.csv'
            elif resolution=='m': # monthly
                assetId = f'ogd-smn_{self._featureId}_m.csv'
            elif resolution=='y': # annual
                assetId = f'ogd-smn_{self._featureId}_y.csv'
                
            request_path = (
                f"{self._BASE}/collections/{self._collectionId}/items/{self._featureId}/assets/{assetId}"
                )
            asset_response = requests.get(request_path)
            asset_response.raise_for_status()
            
            download_url = asset_response.json()['href']
            df_decade = pd.read_csv(download_url, delimiter=';')
            
            if df_data is None:
                pass
            else:
                df_decade = pd.concat(
                    [df_data, df_decade],
                    ignore_index=True
                    )

            if decade_start == decade_check:
                loop_stop_flag = 1
            elif resolution in ('d','m','y'):
                loop_stop_flag = 1
            else:
                decade_start += 10
                decade_end += 10
                
            df_data = df_decade.copy()   
        
        df_data['reference_timestamp'] = (
            pd.to_datetime(
                df_data['reference_timestamp'],
                format="%d.%m.%Y %H:%M"
                )
            )
        
        df_data = df_data[
            (df_data["reference_timestamp"].dt.year >= self._year_start) & 
            (df_data["reference_timestamp"].dt.year <= self._year_end)
            ].reset_index(drop=True)
        
        return df_data
        

    def __find_closest_station(
            self,
            latitude,
            longitude,
            df_meta_stations,
            avoid_crest,
            avoid_stations,
            ):
        """
        Find weather station located closest to provided coordinates.

        Parameters
        ----------
        latitude : TYPE
            DESCRIPTION.
        longitude : TYPE
            DESCRIPTION.
        df_meta_stations : TYPE
            DESCRIPTION.
        avoid_crest : bool
            Reject stations that are located on a crest/mountain, as these
            might not be representative for an area (e.g., station on Pilatus
            is not representative for city of Luzern).
        avoid_stations: list
            List of stations to be excluded, as station abbreviation (e.g.,
            'QUI' for Quinten). This can for example be done if a dataset
            contains NaN values for the selected period.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        df = df_meta_stations.copy()
        
        if avoid_crest:
            # Remove all stations located on crest/mountain (i.e. "Gipfellage"):
            df = df[df['station_exposition_en'] != 'crest']

        # Remove stations:
        df = df[~df['station_abbr'].isin(avoid_stations)]
        
        # Earth radius in km
        R = 6371.0
    
        # Convert to radians
        lat1 = np.radians(latitude)
        lon1 = np.radians(longitude)
        lat2 = np.radians(
            df["station_coordinates_wgs84_lat"].astype(float)
            )
        lon2 = np.radians(
            df["station_coordinates_wgs84_lon"].astype(float)
            )
    
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
    
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = R * c
    
        # Get index of closest station
        idx_min = distances.idxmin()
    
        return df.loc[idx_min, "station_abbr"]
    
    def __decade_start(self, year: int) -> int:
        return (year // 10) * 10
    
    def info_stations_meta_data(self):
        """
        Provide overview of all available stations along with meta data.
        """
        return self._df_meta_stations
        
    def info_parameters_meta_data(self):
        """
        Provide overview of all available parameters along with meta data.
        """
        return self._df_meta_parameters
        
    def info_selected_station(self):
        """
        Povide info about selected weather station along with meta data.
        """
        df_station = (
            self._df_meta_stations
            .loc[self._df_meta_stations['station_abbr']==self._station]
            .copy()
            )        
        return df_station
        
    def info_selected_station_available_data(self):
        """
        Povide info about selected weather station along with meta data.
        """
        df_available_data = (
            self._df_meta_datainventory
            .loc[self._df_meta_datainventory['station_abbr']==self._station]
            .copy()
            )
        return df_available_data
    
    def get_raw_data_hourly(self):
        if self._df_hourly is None:
            self._df_hourly = self.__fetch_raw_data(resolution='h')        
        return self._df_hourly.copy()
    
    def get_raw_data_daily(self):
        if self._df_daily is None:
            self._df_daily = self.__fetch_raw_data(resolution='d')        
        return self._df_daily.copy()
    
    def get_raw_data_10min(self):
        if self._df_10min is None:
            self._df_10min = self.__fetch_raw_data(resolution='t')        
        return self._df_10min.copy()
    
    def get_raw_data_monthly(self):
        if self._df_monthly is None:
            self._df_monthly = self.__fetch_raw_data(resolution='m')        
        return self._df_monthly.copy()
    
    def get_raw_data_yearly(self):
        if self._df_yearly is None:
            self._df_yearly = self.__fetch_raw_data(resolution='y')        
        return self._df_yearly.copy()

    def get_hourly_data_temperature(self):
        
        df_raw = self.get_raw_data_hourly()
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # tre200h0:	Lufttemperatur 2 m über Boden; Stundenmittel [°C]
        # tre200hn:	Lufttemperatur 2 m über Boden; Stundenminimum [°C]
        # tre200hx:	Lufttemperatur 2 m über Boden; Stundenmaximum [°C]
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        df_raw = df_raw[['reference_timestamp','tre200h0', 'tre200hn', 'tre200hx']]
        
        columns_renaming = {
            'reference_timestamp':'time',
            'tre200h0':'T_mean_degC',
            'tre200hn':'T_min_degC',
            'tre200hx':'T_max_degC',
            }
        
        df_T = df_raw.rename(columns=columns_renaming)
        df_T.set_index('time', inplace=True)
        
        return df_T
    
    def get_hourly_data_irradiation(self):
        
        df_raw = self.get_raw_data_hourly()
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # gre000h0:	Globalstrahlung; Stundenmittel [W/m2]
        # ods000h0:	Diffusstrahlung; Stundenmittel [W/m2]
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        df_raw = df_raw[['reference_timestamp','gre000h0', 'ods000h0']]
        
        columns_renaming = {
            'reference_timestamp':'time',
            'gre000h0':'I_global_mean_W_per_m2',
            'ods000h0':'I_diff_mean_W_per_m2',
            }
        
        df_I = df_raw.rename(columns=columns_renaming)
        df_I.set_index('time', inplace=True)
        
        return df_I
    
    def get_daily_data_temperature(self):
        
        df_raw = self.get_raw_data_daily()
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # tre200d0:	Lufttemperatur 2 m über Boden; Tagesmittel [°C]
        # tre200dn:	Lufttemperatur 2 m über Boden; Tagesminimum [°C]
        # tre200dx:	Lufttemperatur 2 m über Boden; Tagesmaximum [°C]
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        df_raw = df_raw[['reference_timestamp','tre200d0', 'tre200dn', 'tre200dx']]
        
        columns_renaming = {
            'reference_timestamp':'time',
            'tre200d0':'T_mean_degC',
            'tre200dn':'T_min_degC',
            'tre200dx':'T_max_degC',
            }
        
        df_T = df_raw.rename(columns=columns_renaming)
        df_T.set_index('time', inplace=True)
        
        return df_T
    
    def get_daily_data_irradiation(self):
        
        df_raw = self.get_raw_data_daily()
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # gre000d0:	Globalstrahlung; Tagesmittel [W/m2]
        # ods000d0:	Diffusstrahlung; Tagesmittel [W/m2]
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        df_raw = df_raw[['reference_timestamp','gre000d0', 'ods000d0']]
        
        columns_renaming = {
            'reference_timestamp':'time',
            'gre000d0':'I_global_mean_W_per_m2',
            'ods000d0':'I_diff_mean_W_per_m2',
            }
        
        df_I = df_raw.rename(columns=columns_renaming)
        df_I.set_index('time', inplace=True)
        
        return df_I

#%% Run example:

if __name__ == "__main__":
    
    # Location:
    # Example: Bern
    latitude = 46.947144
    longitude = 7.444751
    
    # Metadata: SG	47.20571524	9.231518063	QUI	Quinten	SG	23.07.1992	419	47.128739	9.21605	Südhang	https://www.meteoschweiz.admin.ch/service-und-publikationen/applikationen/messwerte-und-messnetze.html#param=messnetz-automatisch&station=QUI

    # Example: St. Gallen
    # latitude = 47.20571524
    # longitude = 9.231518063
    
    # Years for which data is required (incl. start and end years):
    year_start = 2016
    year_end = 2016 # Can be same as year_start if only 1 year is required
    
    # Generate class instance:
    weather_factory = MeteoSwissDataFactory(
        latitude=latitude,
        longitude=longitude,
        year_start=year_start,
        year_end=year_end,
        # avoid_crest=True,
        # avoid_stations=['QUI'],
        )
    
    
    # Get raw data from MeteoSwiss:
    # ----------------------------
    
    df_hourly = weather_factory.get_raw_data_hourly()    
    print(df_hourly)
    
    # df_daily = weather_factory.get_raw_data_daily()    
    # print(df_daily)
    
    # df_10min = weather_factory.get_raw_data_10min()    
    # print(df_10min)
    
    # df_monthly = weather_factory.get_raw_data_monthly()
    # print(df_monthly)
    
    # df_yearly = weather_factory.get_raw_data_yearly()
    # print(df_yearly)
    
    
    # Extract specific data with more intuitive header naming:
    # -------------------------------------------------------
    
    df_T_hourly = weather_factory.get_hourly_data_temperature()
    print(df_T_hourly)
    
    # df_I_hourly = weather_factory.get_hourly_data_irradiation()
    # print(df_I_hourly)

    # df_T_daily = weather_factory.get_daily_data_temperature()
    # print(df_T_daily)
    
    # df_I_daily = weather_factory.get_daily_data_irradiation()
    # print(df_I_daily)
    
    
    # Extract meta data:
    # -----------------
    info_stations = weather_factory.info_stations_meta_data()
    info_parameters = weather_factory.info_parameters_meta_data()
    info_station = weather_factory.info_selected_station()
    info_data = weather_factory.info_selected_station_available_data()
    
    print("\n ----------------------------------------------------------")
    print(" Overview of available stations:\n")
    print(info_stations)
    print("\n ----------------------------------------------------------")
    print(" Overview of parameters:\n")
    print(info_parameters)
    print("\n ----------------------------------------------------------")
    print(" Selected station:\n")
    print(info_station.iloc[0])
    print("\n ----------------------------------------------------------")
    print(" Available data at selected station:\n")
    print(info_data)









