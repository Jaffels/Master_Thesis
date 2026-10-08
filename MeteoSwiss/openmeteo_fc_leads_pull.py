"""
Pull archived weather forecasts with lead times of 2 to 5 days from the Open-Meteo
Previous Runs API and build weighted Swiss series per lead day (1 to 5).
(Weather sub-period check, 8 Oct 2026; lead 1 comes from openmeteo_d1_pull.py.)

Why more than one lead: `<var>_previous_dayK` for hour h comes from a model run of
about K days before h. Which K is known at a gate closure t0 depends on the product:
  FCR 4h (D-1 08:00) and RQ3 d1 (D-1 18:00): K <= 2
  FCR daily (D-2 15:00):                     K <= 3
  aFRR / mFRR daily (D-2 to D-4, 14:30/15:30): K <= 5
  weekly products (Tue 13:00 W-1):           K up to 13 -> not covered (NaN)
Clean/build_weather_fc_features.py picks, per hour and gate closure, the most recent
lead that was published by t0.

Requests are kept within the Open-Meteo free limits (10,000 weighted calls per day,
5,000 per hour, 600 per minute; weight = variables/10 x days/14): 2021-2023 only
temperature (radiation, humidity, wind and precipitation forecasts start in 2024).
Total about 7,500 weighted calls. The script is resumable: stations already on disk
are skipped, and on a daily-limit answer it stops cleanly - just run it again later
(the next day if the daily limit was hit).

Writes only MeteoSwiss/Data/forecast_d1/:
  raw_leads/<ABBR>.parquet        columns <var>_ld2 .. <var>_ld5 (UTC hourly, interval start)
  weather_ch_fc_leads_hourly.parquet  weighted series <name>_ld1 .. _ld5 + coverage, Europe/Zurich
  _pull_leads_manifest.csv, _build_leads_report.txt
Run from Master_Thesis, .venv active (1.5-3 h because of the hourly limit):
    python MeteoSwiss/openmeteo_fc_leads_pull.py
    python MeteoSwiss/openmeteo_fc_leads_pull.py --build-only   # rebuild series from disk
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
sys.path.insert(0, str(HERE))
import openmeteo_d1_pull as D1  # noqa: E402  (URL, weighting, station handling)

OUT = D1.OUT
RAW_LEADS = OUT / "raw_leads"
LEADS = [2, 3, 4, 5]
TEMP_ONLY_BEFORE = 2024
ALL_VARS = D1.OM_VARS
TZ = D1.TZ


class DailyLimit(Exception):
    pass


def fetch(lat, lon, elev, y, variables) -> pd.DataFrame | str:
    end = min(date(y, 12, 31), date.today() - timedelta(days=2))
    hourly = [f"{v}_previous_day{k}" for v in variables for k in LEADS]
    params = {"latitude": lat, "longitude": lon, "elevation": elev,
              "start_date": f"{y}-01-01", "end_date": end.isoformat(),
              "hourly": ",".join(hourly), "timezone": "UTC", "wind_speed_unit": "ms"}
    for i in range(6):
        try:
            r = requests.get(D1.URL, params=params, timeout=D1.TIMEOUT_S)
            if r.status_code == 429:
                txt = r.text.lower()
                if "daily" in txt:
                    raise DailyLimit(r.text[:200])
                wait = 900 if "hourly" in txt else 65
                print(f"    rate limit ({'hourly' if wait == 900 else 'minutely'}) - waiting {wait} s")
                time.sleep(wait)
                continue
            if r.status_code == 400:
                return f"400: {r.text[:150]}"
            r.raise_for_status()
            df = pd.DataFrame(r.json()["hourly"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
            ren = {}
            for v in variables:
                for k in LEADS:
                    ren[f"{v}_previous_day{k}"] = f"{v}_ld{k}"
            df = df.rename(columns=ren)
            for v in variables:
                if v in D1.SHIFT_TO_START:
                    for k in LEADS:
                        df[f"{v}_ld{k}"] = df[f"{v}_ld{k}"].shift(-1)
            return df
        except DailyLimit:
            raise
        except (requests.RequestException, ValueError, KeyError) as exc:
            if i == 5:
                return f"error: {exc}"[:150]
            time.sleep(15 * (i + 1))
    return "error: rate limited"


def pull(stations: pd.DataFrame) -> pd.DataFrame:
    RAW_LEADS.mkdir(parents=True, exist_ok=True)
    man = []
    try:
        for n, (_, s) in enumerate(stations.iterrows(), 1):
            f = RAW_LEADS / f"{s['station_abbr']}.parquet"
            if f.exists():
                man.append({"station": s["station_abbr"], "status": "cached"})
                continue
            parts, notes = [], []
            for y in range(D1.START_YEAR, date.today().year + 1):
                variables = ["temperature_2m"] if y < TEMP_ONLY_BEFORE else ALL_VARS
                res = fetch(s["lat"], s["lon"], s["height_m"], y, variables)
                time.sleep(6 if len(variables) > 1 else 2)   # stay below 600 weighted calls / min
                if isinstance(res, str):
                    notes.append(f"{y} {res}")
                else:
                    parts.append(res)
            if parts:
                df = pd.concat(parts)
                df = df[~df.index.duplicated()]
                cols = [f"{v}_ld{k}" for v in ALL_VARS for k in LEADS]
                df = df.reindex(columns=cols)
                tmp = f.with_suffix(".tmp")
                df.to_parquet(tmp)
                tmp.replace(f)
            man.append({"station": s["station_abbr"], "status": "ok" if parts else "failed",
                        "notes": "; ".join(notes)})
            print(f"  [{n}/{len(stations)}] {s['station_abbr']}: {man[-1]['status']} {man[-1]['notes']}")
    except DailyLimit as exc:
        print(f"\nDaily Open-Meteo limit reached ({exc}). Stations done so far are saved; "
              "run the script again tomorrow to continue.")
        man.append({"station": "-", "status": "stopped_daily_limit"})
    m = pd.DataFrame(man)
    m.to_csv(OUT / "_pull_leads_manifest.csv", index=False)
    return m


def build(sel: pd.DataFrame, wy: pd.DataFrame) -> list[str]:
    lead1 = D1.RAW
    have = [a for a in sel["station_abbr"].unique() if (RAW_LEADS / f"{a}.parquet").exists()]
    missing = sorted(set(sel["station_abbr"]) - set(have))
    if missing:
        return [f"not built: {len(missing)} stations still missing lead files ({', '.join(missing[:10])} ...)"]
    frames = []
    for k in [1] + LEADS:
        # write a temporary per-lead raw folder view: rename <var>_ldK -> <var>
        tmpdir = OUT / f"_tmp_ld{k}"
        tmpdir.mkdir(exist_ok=True)
        for a in sel["station_abbr"].unique():
            if k == 1:
                d = pd.read_parquet(lead1 / f"{a}.parquet")[ALL_VARS]
            else:
                d = pd.read_parquet(RAW_LEADS / f"{a}.parquet")
                d = d[[f"{v}_ld{k}" for v in ALL_VARS]].rename(columns=lambda c: c.rsplit("_ld", 1)[0])
            d.to_parquet(tmpdir / f"{a}.parquet")
        D1.RAW = tmpdir
        w = D1.weighted("load", D1.LOAD_OUT, sel, wy).join(D1.weighted("hydro", D1.HYDRO_OUT, sel, wy), how="outer")
        w.columns = [c.replace("_d1fc", f"_ld{k}") for c in w.columns]
        frames.append(w)
        for f in tmpdir.glob("*.parquet"):
            f.unlink()
        tmpdir.rmdir()
    D1.RAW = lead1
    df = pd.concat(frames, axis=1)
    df.index = df.index.tz_convert(TZ)
    df = df.rename_axis("timestamp").reset_index()
    tmp = OUT / "weather_ch_fc_leads_hourly.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT / "weather_ch_fc_leads_hourly.parquet")

    obs = pd.read_parquet(D1.DATA / "weather_ch_hourly.parquet")
    obs["timestamp"] = pd.to_datetime(obs["timestamp"]).dt.tz_convert(TZ)
    j = df.merge(obs, on="timestamp", how="left")
    j["year"] = j["timestamp"].dt.year
    rows = []
    for base in list(D1.LOAD_OUT) + list(D1.HYDRO_OUT):
        ob = base.replace("_d1fc", "")
        for k in [1] + LEADS:
            fc = base.replace("_d1fc", f"_ld{k}")
            for y, g in j.groupby("year"):
                has = g[fc].notna() & (g[f"coverage_{fc}"] >= 0.9)
                both = has & g[ob].notna()
                rows.append({"series": ob, "lead": k, "year": y, "hours_cov90": round(has.mean(), 3),
                             "mae_vs_obs": round((g.loc[both, fc] - g.loc[both, ob]).abs().mean(), 2)
                             if both.sum() > 100 else np.nan})
    rep = pd.DataFrame(rows)
    piv = rep.pivot_table(index=["series", "year"], columns="lead", values=["hours_cov90", "mae_vs_obs"])
    return ["coverage (share of hours, >= 90 % of weight) and MAE vs observed, by lead day:",
            piv.round(3).to_string()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Open-Meteo forecasts, lead days 2-5")
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    sel = pd.read_csv(D1.DATA / "station_selection.csv").drop_duplicates(["role", "station_abbr"])
    wy = pd.read_csv(D1.DATA / "weights_by_year.csv")
    lines = ["Open-Meteo forecast leads 1-5 build", ""]
    if not a.build_only:
        man = pull(sel.drop_duplicates("station_abbr"))
        lines.append("pull: " + ", ".join(f"{k} {v}" for k, v in man["status"].value_counts().items()))
    lines += build(sel, wy)
    lines += ["", "Send this report to Claude, then run: python Clean/build_weather_fc_features.py --write"]
    (OUT / "_build_leads_report.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
