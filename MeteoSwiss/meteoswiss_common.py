"""
Shared settings and helpers for the MeteoSwiss weather pipeline.

Pipeline (run from the thesis root, venv active):
    python MeteoSwiss/meteoswiss_probe.py   # metadata + station selection
    python MeteoSwiss/meteoswiss_pull.py    # hourly data per selected station
    python MeteoSwiss/meteoswiss_build.py   # weighted Swiss series + features

Source: MeteoSwiss Open Data, automatic weather stations (SwissMetNet),
STAC collection `ch.meteoschweiz.ogd-smn`.

Timestamp convention (MeteoSwiss docs, "Download"):
  * all reference timestamps are UTC;
  * 10-min and hourly values are labelled at the END of the interval
    (16:00 = 15:00-16:00). The pipeline converts them to interval START in
    Europe/Zurich, the convention used by every other thesis pipeline.
  * `historical` files run to 31 Dec of last year; `recent` files cover
    1 Jan of this year to yesterday. Both are pulled.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent            # .../Master_Thesis/MeteoSwiss
DATA = ROOT / "Data"
META_DIR = DATA / "meta"
RAW_DIR = DATA / "raw"
SELECTION_CSV = DATA / "station_selection.csv"
CANDIDATES_CSV = DATA / "station_candidates.csv"
ENERGY_OVERVIEW_QH = ROOT.parent / "SwissGrid" / "EnergyOverview" / "Data" / "qh"

# ---------------------------------------------------------------- API
BASE = "https://data.geo.admin.ch/api/stac/v1"
COLLECTION = "ch.meteoschweiz.ogd-smn"
TIMEOUT_S = 120
RETRIES = 4

# ---------------------------------------------------------------- period
# Thesis range starts 2015-01-01 (Swissgrid auctions, ENTSO-E pre-2021).
# Pull from Nov 2014 so 30-day rolling features are complete on 1 Jan 2015.
PULL_START_LOCAL = "2014-11-01"
OUTPUT_START_LOCAL = "2015-01-01"
TZ = "Europe/Zurich"

# ---------------------------------------------------------------- parameters
# Hourly SwissMetNet parameter short names.
P_T_MEAN = "tre200h0"    # air temperature 2 m, hourly mean [°C]
P_T_MIN = "tre200hn"     # hourly minimum [°C]
P_T_MAX = "tre200hx"     # hourly maximum [°C]
P_GHI = "gre000h0"       # global radiation, hourly mean [W/m2]
P_PRECIP = "rre150h0"    # precipitation, hourly total [mm]
P_RH = "ure200h0"        # relative humidity 2 m, hourly mean [%]
P_WIND = "fkl010h0"      # wind speed scalar, hourly mean [m/s]

REQUIRED = {
    "load": [P_T_MEAN, P_GHI],        # temperature (demand) + radiation (rooftop PV)
    "hydro": [P_T_MEAN, P_PRECIP],    # precipitation + temperature (snowmelt)
}
OPTIONAL = [P_T_MIN, P_T_MAX, P_RH, P_WIND]
# Hourly snow depth is added automatically if the parameter list has one
# (detected by the probe, stored in Data/meta/snow_parameter.txt).

# ---------------------------------------------------------------- selection rules
# LOAD series: one anchor per Swissgrid Energy Overview consumption group,
# placed on the group's main load centre. The nearest eligible station to each
# anchor is used; groups are weighted by their share of annual consumption.
# (lat, lon) in WGS84. Groups with two large centres get two anchors (equal split).
LOAD_ANCHORS = {
    "sh_zh":    [("Zurich", 47.3769, 8.5417)],
    "be_ju":    [("Bern", 46.9481, 7.4474)],
    "ge_vd":    [("Geneva", 46.2044, 6.1432), ("Lausanne", 46.5197, 6.6323)],
    "ag":       [("Aarau/Baden", 47.4400, 8.1500)],
    "vs":       [("Sion", 46.2331, 7.3606)],
    "sg":       [("St. Gallen", 47.4245, 9.3767)],
    "lu":       [("Luzern", 47.0502, 8.3093)],
    "bl_bs":    [("Basel", 47.5596, 7.5886)],
    "ti":       [("Lugano", 46.0037, 8.9511)],
    "fr":       [("Fribourg", 46.8065, 7.1620)],
    "gr":       [("Chur", 46.8499, 9.5329)],
    "gl":       [("Glarus", 47.0404, 9.0679)],
    "tg":       [("Frauenfeld", 47.5536, 8.8987)],
    "sz_zg":    [("Zug", 47.1662, 8.5155)],
    "so":       [("Solothurn/Olten", 47.2600, 7.7000)],
    "ow_nw_ur": [("Stans/Sarnen", 46.9300, 8.3300)],
    "ne":       [("Neuchatel", 46.9900, 6.9293)],
    "ai_ar":    [("Herisau", 47.3861, 9.2792)],
}
LOAD_MAX_HEIGHT_M = 1000     # load-centre stations must be valley/plateau stations
LOAD_MAX_DISTANCE_KM = 40    # flag (not drop) anything further away

# HYDRO series: all eligible Alpine stations in the main hydro cantons, equal
# weight within a canton group; groups weighted by their share of annual
# production among these groups. BE/JU is left out because its production
# includes the Mühleberg nuclear plant until Dec 2019 (weights would shift).
HYDRO_GROUPS = {
    "vs": ["VS"],
    "gr": ["GR"],
    "ti": ["TI"],
    "gl": ["GL"],
    "ow_nw_ur": ["OW", "NW", "UR"],
}
HYDRO_HEIGHT_RANGE_M = (1000, 2600)   # Alpine, but below crest/glacier stations

# Manual overrides after reviewing station_candidates.csv:
# {"load:<group>": ["ABC", ...]} or {"hydro:<group>": [...]}. Empty = rules only.
OVERRIDES: dict[str, list[str]] = {
    # GL has no eligible station in 1000-2600 m; Elm (958 m, Sernftal) is the
    # only Alpine-valley station (hourly data since Apr 2011).
    "hydro:gl": ["ELM"],
    # TI: only Robièi (1898 m) passes the rule; add Piotta (990 m, Leventina,
    # next to the Ritom/Lucendro schemes) so one station isn't the whole canton.
    "hydro:ti": ["ROE", "PIO"],
}
EXCLUDE_STATIONS: list[str] = []


# ---------------------------------------------------------------- helpers
def http_get(url: str, stream: bool = False) -> requests.Response:
    """GET with timeout and retries (network errors and 5xx/429 only)."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT_S, stream=stream)
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code} for {url}")
            else:
                r.raise_for_status()
                return r
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
        time.sleep(5 * attempt)
    raise RuntimeError(f"GET failed after {RETRIES} attempts: {url}") from last


def read_semicolon_csv(content: bytes) -> pd.DataFrame:
    """MeteoSwiss CSV: ';' separator, '.' decimals, Windows-1252, empty = missing."""
    return pd.read_csv(io.BytesIO(content), sep=";", encoding="cp1252", low_memory=False)


def find_col(df: pd.DataFrame, *keywords: str, exclude: tuple[str, ...] = ()) -> str:
    """First column whose lower-cased name contains all keywords and none of exclude."""
    for c in df.columns:
        lc = c.lower()
        if all(k in lc for k in keywords) and not any(x in lc for x in exclude):
            return c
    raise KeyError(f"No column with {keywords} in {list(df.columns)}")


def parse_ms_time(s: pd.Series) -> pd.Series:
    """MeteoSwiss 'dd.mm.yyyy HH:MM' (UTC) -> tz-aware UTC."""
    return pd.to_datetime(s, format="%d.%m.%Y %H:%M", utc=True)


def end_utc_to_start_local(ts_end_utc: pd.Series | pd.DatetimeIndex, hours: int = 1):
    """Hourly label at interval end (UTC) -> interval start (Europe/Zurich)."""
    return (ts_end_utc - pd.Timedelta(hours=hours)).tz_convert(TZ)
