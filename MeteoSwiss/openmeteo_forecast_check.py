"""
Check: do archived weather FORECASTS from Open-Meteo cover 2021+ well enough for
an ex-ante weather test in RQ2 / RQ3b?  (open item 21 / to-do 8.3, 8 Oct 2026)

Today the weather block enters the views only as observed MeteoSwiss values,
labelled as a perfect forecast (`__pf`). A true ex-ante test needs forecasts
that were available before gate closure (FCR D-1 08:00; RQ3 origin D-1 18:00).

Two Open-Meteo archives are checked for five load-weighted SwissMetNet stations:
  HF  Historical Forecast API  (historical-forecast-api.open-meteo.com)
      = stitched short-lead forecasts (first hours of each run). Close to an
        analysis, so it is NOT a D-1 forecast; listed for coverage only.
  PR  Previous Runs API        (previous-runs-api.open-meteo.com)
      = `<var>_previous_day1`: the value forecast one day earlier. This is the
        one that fits a D-1 origin.
For each archive, year and variable: share of hours with a value, and against
the MeteoSwiss observation the correlation and the mean absolute error.

Read-only on the thesis data; writes only
    MeteoSwiss/Data/_openmeteo_check/coverage_<timestamp>.csv
    MeteoSwiss/Data/_openmeteo_check/report_<timestamp>.txt
No API key needed (free tier, ~60 requests). Run from Master_Thesis, .venv on:
    python MeteoSwiss/openmeteo_forecast_check.py
    python MeteoSwiss/openmeteo_forecast_check.py --start 2021 --end 2026
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA = HERE / "Data"
OUT = DATA / "_openmeteo_check"
TIMEOUT_S = 120

PREFERRED = ["SMA", "GVE", "BAS", "BER", "LUG"]   # Zurich, Geneva, Basel, Bern, Lugano
# Open-Meteo variable -> MeteoSwiss raw column
VARS = {"temperature_2m": "tre200h0",
        "shortwave_radiation": "gre000h0",
        "wind_speed_10m": "fkl010h0"}
APIS = {
    "HF": ("https://historical-forecast-api.open-meteo.com/v1/forecast", ""),
    "PR": ("https://previous-runs-api.open-meteo.com/v1/forecast", "_previous_day1"),
}


def stations() -> pd.DataFrame:
    sel = pd.read_csv(DATA / "station_selection.csv")
    load = sel[sel["role"] == "load"].drop_duplicates("station_abbr")
    pick = load[load["station_abbr"].isin(PREFERRED)]
    if len(pick) < 5:
        rest = load[~load["station_abbr"].isin(pick["station_abbr"])]
        pick = pd.concat([pick, rest.head(5 - len(pick))])
    return pick[["station_abbr", "station_name", "lat", "lon"]].reset_index(drop=True)


def fetch(api: str, lat: float, lon: float, y: int) -> pd.DataFrame:
    url, suffix = APIS[api]
    end = min(date(y, 12, 31), date.today() - timedelta(days=2))
    params = {"latitude": lat, "longitude": lon,
              "start_date": f"{y}-01-01", "end_date": end.isoformat(),
              "hourly": ",".join(v + suffix for v in VARS),
              "timezone": "UTC", "wind_speed_unit": "ms"}
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT_S)
            if r.status_code == 400:            # e.g. date before archive start
                return pd.DataFrame({"note": [r.text[:200]]})
            r.raise_for_status()
            h = r.json()["hourly"]
            df = pd.DataFrame(h)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.rename(columns={v + suffix: v for v in VARS})
            # radiation is the mean of the PRECEDING hour -> label at interval start
            df["shortwave_radiation"] = df["shortwave_radiation"].shift(-1)
            return df.set_index("time")
        except (requests.RequestException, ValueError, KeyError) as exc:
            if i == 2:
                return pd.DataFrame({"note": [f"error: {exc}"[:200]]})
            time.sleep(10 * (i + 1))
    return pd.DataFrame()


def observed(abbr: str) -> pd.DataFrame:
    d = pd.read_parquet(DATA / "raw" / f"{abbr}.parquet")
    d["time"] = pd.to_datetime(d["timestamp"], utc=True)   # interval start
    return d.set_index("time")[list(VARS.values())].astype(float)


def main() -> int:
    ap = argparse.ArgumentParser(description="Open-Meteo forecast archive check")
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--end", type=int, default=date.today().year)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    st = stations()
    print("stations:", ", ".join(st["station_abbr"]))
    rows = []
    for _, s in st.iterrows():
        obs = observed(s["station_abbr"])
        for api in APIS:
            for y in range(args.start, args.end + 1):
                fc = fetch(api, s["lat"], s["lon"], y)
                time.sleep(0.5)
                if "note" in fc.columns:
                    print(f"  {s['station_abbr']} {api} {y}: {fc['note'].iloc[0][:90]}")
                    for v in VARS:
                        rows.append({"station": s["station_abbr"], "api": api,
                                     "year": y, "variable": v, "hours": 0,
                                     "coverage": 0.0, "note": fc["note"].iloc[0]})
                    continue
                for v, col in VARS.items():
                    j = fc[[v]].join(obs[[col]], how="left")
                    has = j[v].notna()
                    both = has & j[col].notna()
                    first = j.index[has].min() if has.any() else None
                    rows.append({
                        "station": s["station_abbr"], "api": api, "year": y,
                        "variable": v, "hours": len(j),
                        "coverage": round(has.mean(), 4),
                        "first_valid": first,
                        "corr_vs_obs": round(j.loc[both, v].corr(j.loc[both, col]), 3)
                        if both.sum() > 100 else np.nan,
                        "mae_vs_obs": round((j.loc[both, v] - j.loc[both, col]).abs().mean(), 2)
                        if both.sum() > 100 else np.nan,
                        "note": ""})
                print(f"  {s['station_abbr']} {api} {y}: ok")

    res = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    res.to_csv(OUT / f"coverage_{stamp}.csv", index=False)

    lines = [f"Open-Meteo forecast archive check {stamp}",
             "coverage = share of hours with a value; corr / MAE vs MeteoSwiss station",
             ""]
    if not res.empty:
        tab = (res.groupby(["api", "variable", "year"])
                  .agg(coverage=("coverage", "mean"), corr=("corr_vs_obs", "mean"),
                       mae=("mae_vs_obs", "mean"))
                  .round(3).reset_index())
        lines.append(tab.to_string(index=False))
        pr = tab[(tab["api"] == "PR") & (tab["coverage"] >= 0.95)]
        first_ok = int(pr["year"].min()) if not pr.empty else None
        lines += ["", "Verdict (PR = forecast made one day earlier):"]
        if first_ok is None:
            lines.append("  No year with >= 95 % coverage -> keep weather as __pf only "
                         "(perfect-forecast upper bound); item closed.")
        elif first_ok <= 2021:
            lines.append("  D-1 forecasts cover the whole RQ2 test window -> build a true "
                         "ex-ante weather block (send this report to Claude).")
        else:
            lines.append(f"  D-1 forecasts usable from {first_ok} only -> ex-ante weather "
                         f"test possible as a sub-period check ({first_ok}->), main "
                         "models keep __pf as upper bound (send this report to Claude).")
    (OUT / f"report_{stamp}.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
