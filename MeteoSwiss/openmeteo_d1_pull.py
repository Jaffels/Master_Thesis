"""
Pull archived DAY-AHEAD (D-1) weather forecasts from the Open-Meteo Previous
Runs API for all 43 selected SwissMetNet stations and build the same weighted
Swiss series as meteoswiss_build.py, but from forecasts.
(Weather sub-period check, decided 8 Oct 2026.)

Why: the views use observed MeteoSwiss weather only as a perfect forecast
(`__pf`). The check of 8 Oct 2026 (openmeteo_forecast_check.py) showed that
forecasts made one day earlier (`<var>_previous_day1`) exist for temperature
from 2022 (2021: 78 % of hours) and for radiation and wind from 2024. They are
used for a robustness run, not for the main models.

Variables (Open-Meteo `<var>_previous_day1` -> output, same names as the
observed series with suffix `_d1fc`):
  load-weighted (17 stations):  t_lw_degc, ghi_lw_wm2, rh_lw_pct, wind_lw_ms
  hydro-weighted (26 stations): t_hydro_degc, precip_hydro_mm
Weights: Data/weights_by_year.csv (identical to the observed build).
Station elevation is passed to the API so that temperature is height-corrected.
Timestamps: hourly, interval start, Europe/Zurich (radiation and precipitation
are preceding-hour values in Open-Meteo and are shifted to interval start).
`coverage_*` = weight share of stations with a value in that hour.

Writes only MeteoSwiss/Data/forecast_d1/:
  raw/<ABBR>.parquet           one file per station (re-used unless --force)
  weather_ch_d1_hourly.parquet weighted series + coverage
  _pull_manifest.csv, _build_report.txt
Run from Master_Thesis, .venv active (~260 requests, ~5-10 min):
    python MeteoSwiss/openmeteo_d1_pull.py
    python MeteoSwiss/openmeteo_d1_pull.py --force          # re-download all stations
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA = HERE / "Data"
OUT = DATA / "forecast_d1"
RAW = OUT / "raw"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
TIMEOUT_S = 120
START_YEAR = 2021
TZ = "Europe/Zurich"

OM_VARS = ["temperature_2m", "shortwave_radiation", "relative_humidity_2m",
           "wind_speed_10m", "precipitation"]
SHIFT_TO_START = {"shortwave_radiation", "precipitation"}   # preceding-hour values
LOAD_OUT = {"t_lw_degc_d1fc": "temperature_2m", "ghi_lw_wm2_d1fc": "shortwave_radiation",
            "rh_lw_pct_d1fc": "relative_humidity_2m", "wind_lw_ms_d1fc": "wind_speed_10m"}
HYDRO_OUT = {"t_hydro_degc_d1fc": "temperature_2m", "precip_hydro_mm_d1fc": "precipitation"}


def fetch_year(lat: float, lon: float, elev: float, y: int) -> pd.DataFrame | str:
    end = min(date(y, 12, 31), date.today() - timedelta(days=2))
    params = {"latitude": lat, "longitude": lon, "elevation": elev,
              "start_date": f"{y}-01-01", "end_date": end.isoformat(),
              "hourly": ",".join(f"{v}_previous_day1" for v in OM_VARS),
              "timezone": "UTC", "wind_speed_unit": "ms"}
    for i in range(4):
        try:
            r = requests.get(URL, params=params, timeout=TIMEOUT_S)
            if r.status_code == 429:
                time.sleep(60)
                continue
            if r.status_code == 400:
                return f"400: {r.text[:150]}"
            r.raise_for_status()
            df = pd.DataFrame(r.json()["hourly"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").rename(columns=lambda c: c.replace("_previous_day1", ""))
            for v in SHIFT_TO_START:
                df[v] = df[v].shift(-1)
            return df
        except (requests.RequestException, ValueError, KeyError) as exc:
            if i == 3:
                return f"error: {exc}"[:150]
            time.sleep(10 * (i + 1))
    return "error: rate limited"


def pull(sel: pd.DataFrame, force: bool) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    man = []
    for _, s in sel.iterrows():
        f = RAW / f"{s['station_abbr']}.parquet"
        if f.exists() and not force:
            man.append({"station": s["station_abbr"], "status": "cached"})
            continue
        parts, notes = [], []
        for y in range(START_YEAR, date.today().year + 1):
            res = fetch_year(s["lat"], s["lon"], s["height_m"], y)
            time.sleep(0.4)
            if isinstance(res, str):
                notes.append(f"{y} {res}")
            else:
                parts.append(res)
        if parts:
            df = pd.concat(parts)
            df = df[~df.index.duplicated()]
            tmp = f.with_suffix(".tmp")
            df.to_parquet(tmp)
            tmp.replace(f)
        man.append({"station": s["station_abbr"], "status": "ok" if parts else "failed",
                    "rows": sum(len(p) for p in parts), "notes": "; ".join(notes)})
        print(f"  {s['station_abbr']}: {man[-1]['status']} {man[-1].get('rows', '')} {man[-1].get('notes', '')}")
    m = pd.DataFrame(man)
    m.to_csv(OUT / "_pull_manifest.csv", index=False)
    return m


def weighted(role: str, out_map: dict, sel: pd.DataFrame, wy: pd.DataFrame) -> pd.DataFrame:
    stations = sel.loc[sel["role"] == role, "station_abbr"].unique().tolist()
    panel = {a: pd.read_parquet(RAW / f"{a}.parquet") for a in stations
             if (RAW / f"{a}.parquet").exists()}
    idx = pd.DatetimeIndex(sorted(set().union(*[p.index for p in panel.values()])))
    w = wy[wy["role"] == role].set_index("year")
    year = idx.tz_convert(TZ).year
    out = {}
    for name, var in out_map.items():
        vals = pd.DataFrame({a: p[var].reindex(idx) for a, p in panel.items()})
        W = pd.DataFrame({a: w[a].reindex(year).to_numpy() if a in w else np.nan
                          for a in vals.columns}, index=idx).fillna(0.0)
        Wv = W.where(vals.notna(), 0.0)
        tot = Wv.sum(axis=1)
        out[name] = (vals.fillna(0.0) * Wv).sum(axis=1) / tot.replace(0, np.nan)
        out[f"coverage_{name}"] = tot / W.sum(axis=1).replace(0, np.nan)
    return pd.DataFrame(out, index=idx)


def main() -> int:
    ap = argparse.ArgumentParser(description="Open-Meteo D-1 forecast pull + weighted build")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(DATA / "station_selection.csv").drop_duplicates(["role", "station_abbr"])
    wy = pd.read_csv(DATA / "weights_by_year.csv")
    stations = sel.drop_duplicates("station_abbr")
    print(f"pulling {len(stations)} stations, {START_YEAR} -> {date.today() - timedelta(days=2)}")
    man = pull(stations, args.force)

    df = weighted("load", LOAD_OUT, sel, wy).join(weighted("hydro", HYDRO_OUT, sel, wy), how="outer")
    df.index = df.index.tz_convert(TZ)
    df = df.rename_axis("timestamp").reset_index()
    tmp = OUT / "weather_ch_d1_hourly.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT / "weather_ch_d1_hourly.parquet")

    # report: coverage per year + agreement with the observed series
    obs = pd.read_parquet(DATA / "weather_ch_hourly.parquet")
    obs["timestamp"] = pd.to_datetime(obs["timestamp"]).dt.tz_convert(TZ)
    j = df.merge(obs, on="timestamp", how="left")
    j["year"] = j["timestamp"].dt.year
    lines = ["Open-Meteo D-1 forecast build", "",
             "stations: " + ", ".join(f"{k} {v}" for k, v in man["status"].value_counts().items()), ""]
    rows = []
    for fc in list(LOAD_OUT) + list(HYDRO_OUT):
        ob = fc.replace("_d1fc", "")
        for y, g in j.groupby("year"):
            has = g[fc].notna() & (g[f"coverage_{fc}"] >= 0.9)
            both = has & g[ob].notna() if ob in g else has & False
            rows.append({"series": fc, "year": y,
                         "hours_cov90": round(has.mean(), 3),
                         "corr_vs_obs": round(g.loc[both, fc].corr(g.loc[both, ob]), 3) if both.sum() > 100 else np.nan,
                         "mae_vs_obs": round((g.loc[both, fc] - g.loc[both, ob]).abs().mean(), 2) if both.sum() > 100 else np.nan})
    rep = pd.DataFrame(rows)
    lines.append(rep.to_string(index=False))
    lines += ["", "hours_cov90 = share of hours with a value from stations holding >= 90 % of the weight.",
              "Usable for the sub-period check where hours_cov90 >= 0.95 (expected: temperature 2022->, all 2024->).",
              "Send this report to Claude for the clean / view integration."]
    (OUT / "_build_report.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
