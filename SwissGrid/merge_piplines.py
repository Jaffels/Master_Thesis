"""
Pipeline 05: Master Merge — Model-Ready Dataset
Merges all four data sources on a unified 15-min CET timestamp.
Adds thesis-specific engineered features for forecasting models.

Output: data/merged/master_dataset.parquet  (and .csv)

Feature groups produced:
  - Temporal:    hour, weekday, month, is_weekend, quarter, is_dst
  - Lag features: imbalance price lags (1h, 4h, 1d, 1w)
  - Rolling stats: 1h/4h/24h rolling mean + std of price and volume
  - Spread:       price_pos − price_neg (imbalance spread)
  - Ratio:        aFRR_act / (aFRR_act + mFRR_act)  (response mix)
  - Capacity ref: latest available capacity clearing price per tender period
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Input paths (written by pipelines 01–04)
P_IMBALANCE  = Path("data/imbalance_prices/imbalance_prices.parquet")
P_TENDER_FCR = Path("data/tender_results/tender_fcr.parquet")
P_TENDER_AFRR= Path("data/tender_results/tender_afrr.parquet")
P_TENDER_MFRR= Path("data/tender_results/tender_mfrr.parquet")
P_CONTROL    = Path("data/control_area/control_area_balance.parquet")
P_DAYAHEAD   = Path("data/dayahead/dayahead_15min.parquet")

OUTPUT_DIR   = Path("data/merged")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── LOAD ──────────────────────────────────────────────────────────────────────
def _load(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        log.warning(f"  Missing: {path} ({label}) — run pipeline first")
        return None
    df = pd.read_parquet(path)
    log.info(f"  Loaded {label}: {len(df):,} rows, {list(df.columns)}")
    return df


def _to_utc(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Normalize timestamp to UTC-aware pandas Timestamp."""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].dt.tz is None:
        # Swissgrid publishes in CET (UTC+1) / CEST (UTC+2)
        # Localize as Europe/Zurich then convert to UTC for alignment
        df[ts_col] = (df[ts_col]
                      .dt.tz_localize("Europe/Zurich", ambiguous="infer",
                                      nonexistent="shift_forward")
                      .dt.tz_convert("UTC"))
    else:
        df[ts_col] = df[ts_col].dt.tz_convert("UTC")
    return df


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    ts = df[ts_col].dt.tz_convert("Europe/Zurich")
    df["hour"]       = ts.dt.hour
    df["minute"]     = ts.dt.minute
    df["weekday"]    = ts.dt.weekday          # 0=Monday
    df["month"]      = ts.dt.month
    df["quarter"]    = ts.dt.quarter
    df["is_weekend"] = (ts.dt.weekday >= 5).astype(int)
    df["is_dst"]     = ts.dt.dst().apply(lambda x: 1 if x.seconds > 0 else 0)
    # Cyclical encoding for hour and weekday (useful for ML models)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["weekday"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["weekday"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, cols: list[str],
                     lags_15min: list[int]) -> pd.DataFrame:
    """
    Add lag features. Lags are specified in 15-min periods.
    Common lags:  4 = 1h,  16 = 4h,  96 = 1d,  672 = 1w
    """
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags_15min:
            suffix = {4:"1h", 16:"4h", 96:"1d", 672:"1w"}.get(lag, f"{lag}p")
            df[f"{col}_lag_{suffix}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list[str],
                          windows_15min: list[int]) -> pd.DataFrame:
    """
    Rolling mean and std. Windows: 4=1h, 16=4h, 96=1d
    """
    for col in cols:
        if col not in df.columns:
            continue
        for w in windows_15min:
            label = {4:"1h", 16:"4h", 96:"1d"}.get(w, f"{w}p")
            df[f"{col}_roll_mean_{label}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll_std_{label}"]  = df[col].rolling(w, min_periods=1).std()
    return df


def merge_capacity_prices(df_main: pd.DataFrame,
                          df_tender: pd.DataFrame | None,
                          product: str,
                          price_col: str = "clearing_price_eur_mwh") -> pd.DataFrame:
    """
    Forward-fill capacity clearing price onto the 15-min grid.
    Each tender period's price applies from its start date until the next tender.
    """
    if df_tender is None or df_tender.empty:
        return df_main

    date_col = next((c for c in ("period_start","date","tender_period")
                     if c in df_tender.columns), None)
    if date_col is None or price_col not in df_tender.columns:
        return df_main

    cap = df_tender[[date_col, price_col]].dropna().copy()
    cap[date_col] = pd.to_datetime(cap[date_col], utc=True, errors="coerce")
    cap = cap.dropna(subset=[date_col]).sort_values(date_col)
    cap.rename(columns={date_col: "timestamp",
                         price_col: f"cap_price_{product.lower()}_eur_mwh"}, inplace=True)
    cap["timestamp"] = cap["timestamp"].dt.tz_convert("UTC")

    df_main = df_main.sort_values("timestamp")
    df_main = pd.merge_asof(df_main, cap, on="timestamp", direction="backward")
    return df_main


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(
    lag_cols: list[str] | None = None,
    lag_periods: list[int] | None = None,
    roll_cols: list[str] | None = None,
    roll_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Merge all four pipelines into a single model-ready DataFrame.

    Steps:
    1. Load each source; normalize timestamps to UTC
    2. Create a 15-min master index (inner join on timestamp)
    3. Merge day-ahead prices (15-min, forward-filled from hourly)
    4. Forward-fill capacity clearing prices per tender period
    5. Add temporal, lag, rolling, and ratio features
    6. Save parquet + CSV

    Parameters
    ----------
    lag_cols    : Columns to lag (default: price + volume cols)
    lag_periods : Lag lengths in 15-min steps (default: [4,16,96,672])
    roll_cols   : Columns for rolling stats (default: same as lag_cols)
    roll_windows: Rolling window lengths in 15-min steps (default: [4,16,96])

    Returns
    -------
    Model-ready DataFrame at 15-min resolution (UTC-indexed)
    """
    log.info("=== Pipeline 05: Master Merge ===")

    # 1 — Load
    df_imb  = _load(P_IMBALANCE,  "Imbalance Prices")
    df_ctrl = _load(P_CONTROL,    "Control Area Balance")
    df_da   = _load(P_DAYAHEAD,   "Day-Ahead Prices")
    df_fcr  = _load(P_TENDER_FCR, "FCR Tenders")
    df_afrr = _load(P_TENDER_AFRR,"aFRR Tenders")
    df_mfrr = _load(P_TENDER_MFRR,"mFRR Tenders")

    if df_imb is None:
        raise FileNotFoundError("Run pipeline_01 first — imbalance prices are required.")

    # 2 — Normalize timestamps to UTC
    df_imb  = _to_utc(df_imb)
    if df_ctrl is not None:
        df_ctrl = _to_utc(df_ctrl)
    if df_da is not None:
        df_da = _to_utc(df_da)

    # 3 — Build master index from imbalance prices (the primary source)
    master = df_imb.copy()

    # 4 — Merge control area balance (activation volumes)
    if df_ctrl is not None:
        merge_cols = [c for c in df_ctrl.columns if c != "timestamp"]
        master = pd.merge(
            master, df_ctrl[["timestamp"] + merge_cols],
            on="timestamp", how="left", suffixes=("", "_ctrl")
        )
        log.info(f"  Merged control area: {len(merge_cols)} columns")

    # 5 — Merge day-ahead prices (as_of merge handles hourly → 15-min alignment)
    if df_da is not None:
        master = master.sort_values("timestamp")
        df_da  = df_da.sort_values("timestamp")
        master = pd.merge_asof(
            master, df_da[["timestamp","dayahead_price_eur_mwh"]],
            on="timestamp", direction="backward"
        )
        log.info("  Merged day-ahead prices")

    # 6 — Merge capacity tender prices (forward-filled per tender period)
    master = merge_capacity_prices(master, df_fcr,  "FCR")
    master = merge_capacity_prices(master, df_afrr, "aFRR")
    master = merge_capacity_prices(master, df_mfrr, "mFRR")
    log.info("  Merged capacity tender clearing prices")

    # 7 — Derived signals
    if "price_pos_eur_mwh" in master.columns and "price_neg_eur_mwh" in master.columns:
        master["imbalance_price_spread"] = (
            master["price_pos_eur_mwh"] - master["price_neg_eur_mwh"]
        )

    # aFRR response ratio = aFRR activation / (aFRR + mFRR) activation
    afrr_p = master.get("afrr_act_pos_mw", master.get("afrr_pos_mw"))
    mfrr_p = master.get("mfrr_act_pos_mw", master.get("mfrr_pos_mw"))
    if afrr_p is not None and mfrr_p is not None:
        total = afrr_p.abs() + mfrr_p.abs()
        master["afrr_response_ratio"] = afrr_p.abs() / total.replace(0, np.nan)

    # Day-ahead spread vs imbalance price (arbitrage signal)
    if "dayahead_price_eur_mwh" in master.columns and "price_pos_eur_mwh" in master.columns:
        master["da_vs_imbalance_spread"] = (
            master["price_pos_eur_mwh"] - master["dayahead_price_eur_mwh"]
        )

    # 8 — Temporal features
    master = add_temporal_features(master)

    # 9 — Lag features
    default_lag_cols = ["price_pos_eur_mwh","price_neg_eur_mwh",
                        "net_position_mw","afrr_act_pos_mw","mfrr_act_pos_mw",
                        "dayahead_price_eur_mwh"]
    master = add_lag_features(
        master,
        cols=lag_cols or [c for c in default_lag_cols if c in master.columns],
        lags_15min=lag_periods or [4, 16, 96, 672]
    )

    # 10 — Rolling statistics
    master = add_rolling_features(
        master,
        cols=roll_cols or [c for c in default_lag_cols if c in master.columns],
        windows_15min=roll_windows or [4, 16, 96]
    )

    # 11 — Final sort, de-duplicate, quality summary
    master = (master.sort_values("timestamp")
                    .drop_duplicates(subset=["timestamp"], keep="last")
                    .reset_index(drop=True))

    log.info(f"Master dataset: {len(master):,} rows × {len(master.columns)} columns")
    log.info(f"  Date range: {master['timestamp'].min()} → {master['timestamp'].max()}")
    log.info(f"  Completeness:\n{master.notna().mean().sort_values().tail(10).to_string()}")

    # 12 — Save
    master.to_parquet(OUTPUT_DIR / "master_dataset.parquet", index=False)
    master.to_csv(OUTPUT_DIR / "master_dataset.csv", index=False)
    log.info(f"Saved → {OUTPUT_DIR}/master_dataset.{{parquet,csv}}")
    return master


if __name__ == "__main__":
    df = run()
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head(5).to_string())
    print("\nFeature list:")
    for c in df.columns:
        print(f"  {c}")