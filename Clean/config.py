"""Shared constants for the clean layer (Decisions 1-4, 2 Oct 2026).

Every clean-layer script imports these; no dates or paths are hard-coded elsewhere.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # Master_Thesis
CLEAN_DIR = ROOT / "Clean"
DATA_DIR = CLEAN_DIR / "Data"
MASTER_DIR = DATA_DIR / "master"

SAMPLE_START = "2015-01-01 00:00"   # Europe/Zurich, first quarter-hour (= 2014-12-31 23:00 UTC)
CUTOFF = "2026-08-31 23:45"         # Europe/Zurich, last quarter-hour  (= 2026-08-31 21:45 UTC)
TZ_LABEL = "Europe/Zurich"
FREQ = "15min"
FLOAT_DTYPE = "float32"


def start_utc() -> pd.Timestamp:
    """First quarter-hour of the sample, as a UTC timestamp."""
    return pd.Timestamp(SAMPLE_START, tz=TZ_LABEL).tz_convert("UTC")


def end_utc() -> pd.Timestamp:
    """End of the sample (exclusive): the cut-off quarter-hour + 15 min, UTC."""
    return (pd.Timestamp(CUTOFF, tz=TZ_LABEL) + pd.Timedelta(FREQ)).tz_convert("UTC")
