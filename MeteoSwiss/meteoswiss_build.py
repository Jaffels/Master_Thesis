"""
MeteoSwiss build — two weighted Swiss weather series + thesis features.

LOAD-weighted series (demand, rooftop PV):
  station weight = group share of annual consumption (Swissgrid Energy
  Overview, cons_<group>_kwh, per calendar year) x within-group weight.
HYDRO-weighted series (inflows, snowmelt):
  station weight = group share of annual production among the hydro groups
  (prod_<group>_kwh) x within-group weight.
Weights are renormalised every hour over the stations that have a value, and
the share of weight actually available is kept as a coverage column.

Output: MeteoSwiss/Data/weather_ch_hourly.parquet (hourly, 2015-01-01 ->),
timestamp = interval start, Europe/Zurich.
  t_lw_degc, tmin_lw_degc, tmax_lw_degc   load-weighted temperature
  hdh_lw, cdh_lw                          heating/cooling degree-hours
                                          (base 18 °C / 22 °C)
  ghi_lw_wm2, ghi_ramp_lw_wm2             load-weighted global radiation, 1-h change
  rh_lw_pct, wind_lw_ms                   optional, if measured
  t_hydro_degc, precip_hydro_mm           hydro-weighted temperature, precipitation
  melt_dh_hydro                           positive degree-hours (snowmelt proxy)
  snow_hydro_cm                           hydro-weighted snow depth, if available
  precip_hydro_7d_mm / _30d_mm, melt_dh_hydro_7d   trailing sums (past only)
  coverage_lw, coverage_hydro             share of weight with data (0-1)
Also writes Data/weights_by_year.csv and Data/_build_report.txt.

Rolling sums only look backwards, but for forecasting (RQ2) every feature
must still be lagged to what was known at auction time — do that in the
modelling layer, not here.

Usage (from the thesis root, venv active):
    python MeteoSwiss/meteoswiss_build.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meteoswiss_common as C  # noqa: E402

HDD_BASE = 18.0
CDD_BASE = 22.0

LOAD_VARS = {  # output name -> parameter
    "t_lw_degc": C.P_T_MEAN, "tmin_lw_degc": C.P_T_MIN, "tmax_lw_degc": C.P_T_MAX,
    "ghi_lw_wm2": C.P_GHI, "rh_lw_pct": C.P_RH, "wind_lw_ms": C.P_WIND,
}
HYDRO_VARS = {"t_hydro_degc": C.P_T_MEAN, "precip_hydro_mm": C.P_PRECIP}


def group_shares() -> pd.DataFrame:
    """Annual consumption/production share per group (rows = year)."""
    files = sorted(C.ENERGY_OVERVIEW_QH.glob("*.parquet"))
    if not files:
        sys.exit(f"Energy Overview parquet not found in {C.ENERGY_OVERVIEW_QH}")
    cols = ["timestamp"] + [f"cons_{g}_kwh" for g in C.LOAD_ANCHORS] + [f"prod_{g}_kwh" for g in C.HYDRO_GROUPS]
    yearly = []
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        s = df.drop(columns="timestamp").sum(min_count=1)
        s.name = int(df["timestamp"].dt.year.mode()[0])
        yearly.append(s)
    y = pd.DataFrame(yearly).sort_index()
    load = y[[f"cons_{g}_kwh" for g in C.LOAD_ANCHORS]]
    load.columns = [f"load:{g}" for g in C.LOAD_ANCHORS]
    hydro = y[[f"prod_{g}_kwh" for g in C.HYDRO_GROUPS]]
    hydro.columns = [f"hydro:{g}" for g in C.HYDRO_GROUPS]
    load = load.div(load.sum(axis=1), axis=0)
    hydro = hydro.div(hydro.sum(axis=1), axis=0)
    return pd.concat([load, hydro], axis=1).dropna(how="all")


def station_weights(sel: pd.DataFrame, shares: pd.DataFrame, role: str, years) -> pd.DataFrame:
    """Rows = year, columns = station, value = weight (sums to 1 per year)."""
    sub = sel[sel["role"] == role]
    w = {}
    for yr in years:
        yr_src = yr if yr in shares.index else shares.index[shares.index <= yr].max()
        if pd.isna(yr_src):
            yr_src = shares.index.min()
        row = {}
        for r in sub.itertuples():
            share = shares.at[yr_src, f"{role}:{r.group}"]
            row[r.station_abbr] = row.get(r.station_abbr, 0.0) + share * r.weight_in_group
        w[yr] = row
    w = pd.DataFrame(w).T.fillna(0.0)
    return w.div(w.sum(axis=1), axis=0)


def weighted(panel: dict[str, pd.DataFrame], param: str, w_year: pd.DataFrame, index) -> tuple[pd.Series, pd.Series]:
    """Hourly weighted mean over stations, renormalised over available values."""
    vals = pd.DataFrame({s: panel[s][param] for s in w_year.columns if s in panel and param in panel[s]},
                        index=index)
    if vals.empty:
        return pd.Series(np.nan, index=index), pd.Series(0.0, index=index)
    w = w_year.loc[index.year, vals.columns].to_numpy()
    x = vals.to_numpy()
    avail = ~np.isnan(x)
    wsum = (w * avail).sum(axis=1)
    mean = np.where(wsum > 0, np.nansum(w * np.nan_to_num(x), axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
    return pd.Series(mean, index=index), pd.Series(wsum, index=index)


def main():
    sel = pd.read_csv(C.SELECTION_CSV)
    snow_file = C.META_DIR / "snow_parameter.txt"
    snow = snow_file.read_text().strip() if snow_file.exists() else ""

    panel, missing = {}, []
    for abbr in sel["station_abbr"].astype(str).unique():
        path = C.RAW_DIR / f"{abbr}.parquet"
        if not path.exists():
            missing.append(abbr)
            continue
        panel[abbr] = pd.read_parquet(path).set_index("timestamp")
    if not panel:
        sys.exit("No station files in Data/raw — run meteoswiss_pull.py first.")

    idx = pd.DatetimeIndex(sorted(set().union(*[p.index for p in panel.values()])), name="timestamp")
    for k in panel:
        panel[k] = panel[k].reindex(idx)

    shares = group_shares()
    years = sorted(set(idx.year))
    w_load = station_weights(sel, shares, "load", years)
    w_hydro = station_weights(sel, shares, "hydro", years)

    out = pd.DataFrame(index=idx)
    for name, p in LOAD_VARS.items():
        out[name], cov = weighted(panel, p, w_load, idx)
        if name == "t_lw_degc":
            out["coverage_lw"] = cov
    for name, p in HYDRO_VARS.items():
        out[name], cov = weighted(panel, p, w_hydro, idx)
        if name == "precip_hydro_mm":
            out["coverage_hydro"] = cov
    if snow:
        out["snow_hydro_cm"], _ = weighted(panel, snow, w_hydro, idx)

    out["hdh_lw"] = (HDD_BASE - out["t_lw_degc"]).clip(lower=0)
    out["cdh_lw"] = (out["t_lw_degc"] - CDD_BASE).clip(lower=0)
    out["ghi_ramp_lw_wm2"] = out["ghi_lw_wm2"].diff()
    out["melt_dh_hydro"] = out["t_hydro_degc"].clip(lower=0)
    out["precip_hydro_7d_mm"] = out["precip_hydro_mm"].rolling(168, min_periods=150).sum()
    out["precip_hydro_30d_mm"] = out["precip_hydro_mm"].rolling(720, min_periods=648).sum()
    out["melt_dh_hydro_7d"] = out["melt_dh_hydro"].rolling(168, min_periods=150).sum()

    out = out.dropna(axis=1, how="all")
    out = out[out.index >= pd.Timestamp(C.OUTPUT_START_LOCAL, tz=C.TZ)].reset_index()
    out.to_parquet(C.DATA / "weather_ch_hourly.parquet", index=False)

    wy = pd.concat({"load": w_load, "hydro": w_hydro}, names=["role", "year"]).round(4)
    wy.to_csv(C.DATA / "weights_by_year.csv")

    rep = [f"MeteoSwiss build — {pd.Timestamp.now():%Y-%m-%d %H:%M}",
           f"Stations used: {len(panel)}; missing raw files: {missing or 'none'}",
           f"Rows: {len(out):,}; {out.timestamp.min()} .. {out.timestamp.max()}",
           "", "Missing share per column:",
           out.drop(columns="timestamp").isna().mean().round(4).to_string(),
           "", "Hours with coverage < 0.8:",
           f"  load: {int((out['coverage_lw'] < 0.8).sum())}, hydro: {int((out['coverage_hydro'] < 0.8).sum())}",
           "", "Annual mean load-weighted temperature (°C):",
           out.groupby(out.timestamp.dt.year)["t_lw_degc"].mean().round(2).to_string()]
    text = "\n".join(rep)
    (C.DATA / "_build_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
