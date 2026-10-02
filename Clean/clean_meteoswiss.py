"""Clean layer: MeteoSwiss weather (to-do 3.3, 2 Oct 2026).

Run from Master_Thesis with .venv active:
    python Clean/clean_meteoswiss.py            # build, check, print summary
    python Clean/clean_meteoswiss.py --write    # also write Clean/Data/weather/

Input   MeteoSwiss/Data/weather_ch_hourly.parquet (43 SwissMetNet stations, built by
        meteoswiss_build.py; hourly, Europe/Zurich interval start; two weightings:
        lw = load-weighted, hydro = hydro-production-weighted, weights from the
        Swissgrid Energy Overview canton data)
Output  Clean/Data/weather/weather.parquet + weather_dictionary.csv (15-min grid)

Hourly -> 15 min (table design, to-do 3.3)
- levels / averages (temperature, irradiance, humidity, wind, snow depth, coverage,
  running 7-/30-day sums, irradiance ramp): repeated over the four quarter-hours
- per-hour amounts (precipitation, heating / cooling / melt degree-hours):
  divided by 4, so sums over any window stay correct
Observed weather is known only after the fact: in RQ2 views it must be lagged to
what was known at D-1, or labelled as a perfect-forecast upper bound.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import config as C  # noqa: E402

SRC = "MeteoSwiss SwissMetNet (43 stations), weighted by meteoswiss_build.py"
AVAIL = "ex post (observed); lag to D-1 or treat as perfect forecast in RQ2 views"
# source column -> (clean name, unit, how, aggregation, note)
MAP = {
    "t_lw_degc":           ("ch_temp_lw_degc", "°C", "repeat", "mean", "load-weighted air temperature"),
    "tmin_lw_degc":        ("ch_temp_min_lw_degc", "°C", "repeat", "min", "load-weighted hourly minimum temperature"),
    "tmax_lw_degc":        ("ch_temp_max_lw_degc", "°C", "repeat", "max", "load-weighted hourly maximum temperature"),
    "ghi_lw_wm2":          ("ch_ghi_lw_wm2", "W/m²", "repeat", "mean", "load-weighted global horizontal irradiance"),
    "ghi_ramp_lw_wm2":     ("ch_ghi_ramp_lw_wm2", "W/m² per h", "repeat", "mean", "hour-on-hour change of ghi_lw"),
    "rh_lw_pct":           ("ch_rh_lw_pct", "%", "repeat", "mean", "load-weighted relative humidity"),
    "wind_lw_ms":          ("ch_wind_lw_ms", "m/s", "repeat", "mean", "load-weighted wind speed"),
    "hdh_lw":              ("ch_hdh_lw_kh", "K·h", "split", "sum", "heating degree-hours, load-weighted (per quarter-hour after ÷4)"),
    "cdh_lw":              ("ch_cdh_lw_kh", "K·h", "split", "sum", "cooling degree-hours, load-weighted (per quarter-hour after ÷4)"),
    "coverage_lw":         ("ch_wx_coverage_lw_share", "share", "repeat", "min", "share of load weight covered by reporting stations"),
    "t_hydro_degc":        ("ch_temp_hydro_degc", "°C", "repeat", "mean", "hydro-weighted air temperature"),
    "precip_hydro_mm":     ("ch_precip_hydro_mm", "mm", "split", "sum", "hydro-weighted precipitation (per quarter-hour after ÷4)"),
    "precip_hydro_7d_mm":  ("ch_precip_7d_hydro_mm", "mm", "repeat", "mean", "running 7-day precipitation sum (level)"),
    "precip_hydro_30d_mm": ("ch_precip_30d_hydro_mm", "mm", "repeat", "mean", "running 30-day precipitation sum (level)"),
    "snow_hydro_cm":       ("ch_snow_hydro_cm", "cm", "repeat", "mean", "hydro-weighted snow depth (1000-2500 m stations; melt-timing proxy, not snow-water volume); small negative values = sensor noise, kept"),
    "melt_dh_hydro":       ("ch_melt_dh_hydro_kh", "K·h", "split", "sum", "melt degree-hours, hydro-weighted (per quarter-hour after ÷4)"),
    "melt_dh_hydro_7d":    ("ch_melt_dh_7d_hydro_kh", "K·h", "repeat", "mean", "running 7-day melt degree-hours (level)"),
    "coverage_hydro":      ("ch_wx_coverage_hydro_share", "share", "repeat", "min", "share of hydro weight covered by reporting stations"),
}


def build():
    w = pd.read_parquet(C.ROOT / "MeteoSwiss" / "Data" / "weather_ch_hourly.parquet")
    missing = set(MAP) - set(w.columns)
    extra = set(w.columns) - set(MAP) - {"timestamp"}
    if missing or extra:
        raise ValueError(f"weather columns changed: missing {sorted(missing)}, new {sorted(extra)}")
    w.index = K.to_utc(w.pop("timestamp"))
    w = w.astype("float64")
    out = pd.DataFrame(index=K.grid().index)
    dic = []
    for src, (name, unit, how, agg, note) in MAP.items():
        out[name] = K.hourly_to_qh(w[src], how)
        dic.append(dict(column=name, source=SRC, source_series=f"weather_ch_hourly.{src}", unit=unit,
                        resolution_native="1h", aggregation_rule=agg, availability_rule=AVAIL, notes=note))
    return w, out, dic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    w, out, dic = build()
    assert not K.check_names(out.columns), K.check_names(out.columns)
    assert len(out) == 409_052
    # checks: split columns keep their hourly totals, repeat columns their hourly means
    wt = K.trim(w)
    for src, (name, _, how, _, _) in MAP.items():
        if how == "split":
            r = out[name].sum() / wt[src].sum()
            assert abs(r - 1) < 1e-6, (name, r)
        else:
            r = out[name].mean() / wt[src].mean() if wt[src].mean() else 1
            assert abs(r - 1) < 1e-3, (name, r)
    s = out.describe().T[["count", "min", "50%", "max"]]
    s["nan_share"] = out.isna().mean().round(5)
    print(s.round(2).to_string())
    print(f"\nrows {len(out)} | first {out.index[0]} | last {out.index[-1]} | totals / means preserved: ok")
    if a.write:
        p = K.write_clean(out, "weather", "weather", dic)
        print(f"written: {p} + weather_dictionary.csv")


if __name__ == "__main__":
    main()
