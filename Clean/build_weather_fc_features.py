"""Ex-ante weather FORECAST features for the weather sub-period check (8 Oct 2026).

The main views carry weather only as observed values: `__pf` (perfect forecast, NOT ex
ante) and lookbacks. This script adds what a provider could really have known: the
Open-Meteo archived forecasts (MeteoSwiss/openmeteo_d1_pull.py, lead day 1, and
openmeteo_fc_leads_pull.py, lead days 2-5), weighted like the observed series.

Rule per hour h of the delivery window and forecast origin t0:
  a lead-K value for hour h comes from a model run of about h - K days; it counts as
  published at h - K days + PUB_LAG (6 h, conservative for run time + upload).
  K = smallest lead in 1..5 with h - K days + PUB_LAG <= t0; none -> NaN.
Block features (fcr / afrr / mfrr): mean over the delivery hours (precipitation: sum),
only if every hour has a value; t0 = gate closure (gate_closure.py). Weekly blocks need
leads of up to 13 days -> always NaN (documented, not an error).
RQ3 features: value of the quarter-hour's hour for origin d1 (t0_d1_utc, D-1 18:00) and
h1 (t0_h1_utc, start - 1 h).

Coverage (expected): temperature from 2022 (2021 ~78 %), radiation / humidity / wind /
precipitation from 2024. Use only for the sub-period robustness run (RQ2 2022->/2024->,
RQ3b), never as a replacement of the main feature set.

Reads  MeteoSwiss/Data/forecast_d1/weather_ch_fc_leads_hourly.parquet (falls back to
       weather_ch_d1_hourly.parquet = lead 1 only), Clean/Data/views/*_exante.parquet,
       Clean/Data/views/rq3_15min.parquet
Writes Clean/Data/views/{fcr,afrr,mfrr}_wxfc.parquet, rq3_wxfc.parquet,
       _wxfc_dictionary.csv, build_weather_fc_features_report.txt   (only with --write)
Load   views.load(view, ..., weather_fc=True) / views.load_rq3(weather_fc=True)

Run from Master_Thesis with .venv active:
    python Clean/build_weather_fc_features.py            # build + checks, no files
    python Clean/build_weather_fc_features.py --write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path(__file__).resolve().parent
sys.path.insert(0, str(CLEAN))
from gate_closure import gate_closure  # noqa: E402

ROOT = CLEAN.parent
FC_DIR = ROOT / "MeteoSwiss" / "Data" / "forecast_d1"
VIEWS = CLEAN / "Data" / "views"
PUB_LAG = pd.Timedelta(hours=6)
MAX_LEAD = 5
DAY_NS = 86_400 * 10**9
# forecast series base name -> (feature name, aggregation over a block)
SERIES = {
    "t_lw_degc": ("ch_temp_lw_degc", "mean"),
    "ghi_lw_wm2": ("ch_ghi_lw_wm2", "mean"),
    "rh_lw_pct": ("ch_rh_lw_pct", "mean"),
    "wind_lw_ms": ("ch_wind_lw_ms", "mean"),
    "t_hydro_degc": ("ch_temp_hydro_degc", "mean"),
    "precip_hydro_mm": ("ch_precip_hydro_mm", "sum"),
}
KEYS = ["product", "direction", "procurement", "block_start_utc", "block_end_utc", "auction_id"]


def load_forecasts() -> tuple[np.ndarray, dict[str, np.ndarray], list[int]]:
    """hour index (int64 ns, UTC) and per series an array [n_hours, MAX_LEAD] (NaN where
    the lead is missing or its weight coverage < 90 %)."""
    f = FC_DIR / "weather_ch_fc_leads_hourly.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        leads = list(range(1, MAX_LEAD + 1))
        col = lambda b, k: f"{b}_ld{k}"  # noqa: E731
    else:
        df = pd.read_parquet(FC_DIR / "weather_ch_d1_hourly.parquet")
        leads = [1]
        col = lambda b, k: f"{b}_d1fc"  # noqa: E731
        print("note: lead files not built yet -> lead 1 only (most blocks will be NaN)")
    hours = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    arr = {}
    for base in SERIES:
        a = np.full((len(df), MAX_LEAD), np.nan)
        for k in leads:
            v = df[col(base, k)].to_numpy("float64")
            cov = df[f"coverage_{col(base, k)}"].to_numpy("float64")
            a[:, k - 1] = np.where(cov >= 0.9, v, np.nan)
        arr[base] = a
    return hours, arr, leads


def pick(hours_needed: np.ndarray, t0: np.ndarray, hours: np.ndarray, arr: np.ndarray
         ) -> tuple[np.ndarray, np.ndarray]:
    """value and lead used for each (hour, t0) pair. Lead K is allowed if
    hour - K days + PUB_LAG <= t0; the smallest allowed K with a value is taken."""
    pos = np.searchsorted(hours, hours_needed)
    ok_pos = (pos < len(hours)) & (hours[np.minimum(pos, len(hours) - 1)] == hours_needed)
    pos = np.minimum(pos, len(hours) - 1)
    val = np.full(len(hours_needed), np.nan)
    lead = np.zeros(len(hours_needed), dtype="int8")
    for k in range(1, MAX_LEAD + 1):
        allowed = (hours_needed - k * DAY_NS + PUB_LAG.value <= t0) & ok_pos & np.isnan(val)
        v = arr[pos, k - 1]
        take = allowed & ~np.isnan(v)
        val[take] = v[take]
        lead[take] = k
    return val, lead


def block_features(view: str, hours: np.ndarray, arr: dict) -> tuple[pd.DataFrame, list[dict]]:
    v = pd.read_parquet(VIEWS / f"{view}_exante.parquet", columns=KEYS)
    gc = gate_closure(v)
    t0 = gc["gate_closure_utc"].dt.as_unit("ns").astype("int64").to_numpy()
    s = v["block_start_utc"].dt.as_unit("ns").astype("int64").to_numpy()
    e = v["block_end_utc"].dt.as_unit("ns").astype("int64").to_numpy()
    nh = ((e - s) // 3_600_000_000_000).astype("int64")
    rep = np.repeat(np.arange(len(v)), nh)
    offs = np.arange(nh.sum()) - np.repeat(np.cumsum(nh) - nh, nh)
    h = s[rep] + offs * 3_600_000_000_000
    out = v[KEYS].copy()
    dic = []
    lead_max = np.zeros(len(v), dtype="int8")
    for base, (name, agg) in SERIES.items():
        val, lead = pick(h, t0[rep], hours, arr[base])
        g = pd.DataFrame({"b": rep, "v": val, "l": lead})
        n_ok = g.groupby("b")["v"].count().reindex(range(len(v)), fill_value=0).to_numpy()
        a = (g.groupby("b")["v"].sum() if agg == "sum" else g.groupby("b")["v"].mean()
             ).reindex(range(len(v))).to_numpy()
        full = n_ok == nh
        out[f"{name}__wxfc"] = np.where(full, a, np.nan)
        lm = g.groupby("b")["l"].max().reindex(range(len(v)), fill_value=0).to_numpy()
        lead_max = np.maximum(lead_max, np.where(full, lm, 0)).astype("int8")
        dic.append(dict(column=f"{name}__wxfc", view=view, role="weather_forecast", aggregation=agg,
                        source="Open-Meteo Previous Runs API, weighted like MeteoSwiss series",
                        rule="per delivery hour: freshest lead (1-5 d) published by gate closure "
                             "(run + 6 h); NaN unless every hour is covered"))
    out["wxfc_lead_max_days"] = lead_max
    dic.append(dict(column="wxfc_lead_max_days", view=view, role="meta", aggregation="max",
                    source="", rule="largest lead day used in the block (0 = no forecast)"))
    return out, dic


def rq3_features(hours: np.ndarray, arr: dict) -> tuple[pd.DataFrame, list[dict]]:
    r = pd.read_parquet(VIEWS / "rq3_15min.parquet", columns=["ts_utc", "t0_d1_utc", "t0_h1_utc"])
    ts = r["ts_utc"].dt.as_unit("ns").astype("int64").to_numpy()
    h = ts - ts % 3_600_000_000_000
    out = r[["ts_utc"]].copy()
    dic = []
    for origin in ["d1", "h1"]:
        t0 = r[f"t0_{origin}_utc"].dt.as_unit("ns").astype("int64").to_numpy()
        for base, (name, agg) in SERIES.items():
            val, lead = pick(h, t0, hours, arr[base])
            if agg == "sum":
                val = val / 4.0          # hourly amount -> per quarter-hour (as in the clean layer)
            out[f"{name}__wxfc_{origin}"] = val
            dic.append(dict(column=f"{name}__wxfc_{origin}", view="rq3", role="weather_forecast",
                            aggregation=agg, source="Open-Meteo Previous Runs API",
                            rule=f"hour of the QH, freshest lead published by t0_{origin}_utc (run + 6 h)",
                            origin=origin))
    return out, dic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    hours, arr, leads = load_forecasts()
    lines = [f"weather forecast features (leads available: {leads})", ""]
    dic, outs = [], {}
    for view in ["fcr", "afrr", "mfrr"]:
        df, d = block_features(view, hours, arr)
        outs[view] = df
        dic += d
        df["year"] = df["block_start_utc"].dt.tz_convert("Europe/Zurich").dt.year
        cov = (df[df["year"] >= 2021].groupby(["procurement", "year"])
               [[c for c in df.columns if c.endswith("__wxfc")]].apply(lambda g: g.notna().mean()))
        lines += [f"{view}: share of blocks with a forecast value", cov.round(2).to_string(),
                  f"lead days used: {df['wxfc_lead_max_days'].value_counts().sort_index().to_dict()}", ""]
        df.drop(columns="year", inplace=True)
    r, d = rq3_features(hours, arr)
    outs["rq3"] = r
    dic += d
    yr = r["ts_utc"].dt.tz_convert("Europe/Zurich").dt.year
    cov = r[yr >= 2021].groupby(yr[yr >= 2021])[[c for c in r.columns if "__wxfc_" in c]].apply(lambda g: g.notna().mean())
    lines += ["rq3: share of quarter-hours with a forecast value", cov.round(2).T.to_string(), ""]
    # leakage check: no value may come from a lead published after t0
    lines.append("leakage check: by construction (lead K allowed only if hour - K d + 6 h <= t0)")
    print("\n".join(lines))
    if a.write:
        for name, df in outs.items():
            tmp = VIEWS / f"{name}_wxfc.tmp"
            df.to_parquet(tmp, index=False)
            tmp.replace(VIEWS / f"{name}_wxfc.parquet")
        pd.DataFrame(dic).to_csv(VIEWS / "_wxfc_dictionary.csv", index=False)
        (VIEWS / "build_weather_fc_features_report.txt").write_text("\n".join(lines) + "\n")
        print(f"\nwritten: {', '.join(f'{n}_wxfc.parquet' for n in outs)}, _wxfc_dictionary.csv, report")


if __name__ == "__main__":
    main()
