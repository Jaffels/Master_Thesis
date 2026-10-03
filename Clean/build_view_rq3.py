"""RQ3 modelling view: one row per quarter-hour, spike targets + ex-ante features (to-do 5, item 6).

Decided 3 Oct 2026:
- Rows: every quarter-hour of the master from the RQ3 window start (31 Mar 2016, CH imbalance
  prices start) to the data cut-off.
- Targets (role target / threshold / spike / episode), from the CH imbalance prices:
    tgt_price_short_eur_mwh, tgt_price_long_eur_mwh   ch_imb_price_short / _long (2026: both =
                                                     single price, see regime_imb_pricing)
    thr_short_q{975,990,995}   upper quantile of the SHORT price over the previous 365 days
    thr_long_q{025,010,005}    lower quantile of the LONG price over the previous 365 days
                               (computed per local day from data up to the end of D-1;
                               >= MIN_THR_DAYS x 48 quarter-hour prices, i.e. 60 full days,
                               else NaN -> first threshold 31 May 2016)
    spike_short_q*             short price > threshold   (undersupply, BG-short)
    spike_long_q*              long price  < threshold   (oversupply, BG-long)
                               2026 (single price): direction from ch_system_imbalance_mw
                               (< 0 short, > 0 long), as in thesis 3.74
    episode_{short,long}_id / _len_qh   consecutive spike quarter-hours (q990 / q010)
  Values are 1.0 / 0.0 / NaN (NaN where the price or the threshold is missing).
- Two forecast origins; every feature carries its origin in the suffix:
    __d1  t0 = D-1 18:00 local for all quarter-hours of day D (day-ahead risk model, all
          ENTSO-E day-ahead forecasts and the day-ahead result are public by then)
    __h1  t0 = quarter-hour start - 1 h (intraday gate closure, nowcast)
  Publication times follow the availability rules of build_views_exante.py (column_plan:
  observed values public `lag` after the quarter-hour end, monthly Swissgrid files on the 15th
  of the following month, D-1 forecasts by 18:00).
- Curated RQ3 feature set (not all master columns):
    cal_*                          calendar of the quarter-hour (known in advance), incl. CH
                                   holidays, bridge days, Christmas period, DST switch days
    regime_imb_resolution / _pricing  copied (market-design dates)
    {col}__fc_d1                   day-ahead forecasts for the quarter-hour itself (load, solar,
                                   wind, total generation per zone; CH border DA NTC and DA
                                   schedules), plus derived CH residual load and ramps
    {series}__{lb24h,lb7d}_d1      lookbacks at D-1 18:00: imbalance prices (mean, max, min),
                                   imbalance volume, aFRR / mFRR activation, CH load, forecast
                                   errors (actual - DA forecast) per zone, frequency deviation
    spikehist_*__{lb24h,lb7d,lb30d}_d1   share of spike quarter-hours (q990 / q010) known at t0
    {price}__lag{2d,7d}_d1         same quarter-hour 2 / 7 days earlier
    {outage col}__exante_d1        outages of day D announced by t0 (outage_events, as in the
                                   block views; NaN in months with late publication)
    {weather}__pf / __lb24h_d1     weather of the quarter-hour as 'perfect forecast' (NOT ex ante,
                                   role perfect_forecast) and observed lookbacks
    {series}__{last,lb1h,lb4h}_h1  nowcast at t0 = start - 1 h: latest published values and
                                   1 h / 4 h means of imbalance prices / volume, aFRR activation,
                                   load and VRES forecast errors, intraday - day-ahead VRES updates
    {price}__lag1d_h1, spikehist_*__lb4h_h1
Nothing is imputed (NaN stays NaN). Load with views.load_rq3().

Outputs (only with --write), in Clean/Data/views/:
    rq3_15min.parquet, _rq3_dictionary.csv, build_view_rq3_report.txt

Run from Master_Thesis with .venv active (after build_master.py --write):
    python Clean/build_view_rq3.py            # build + checks, writes nothing
    python Clean/build_view_rq3.py --write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path(__file__).resolve().parent
sys.path.insert(0, str(CLEAN))
import build_views_exante as BX  # noqa: E402

TZ = BX.TZ
VIEWS_DIR = BX.VIEWS_DIR
MT = BX.MT
START = "2016-03-31"                       # RQ3 window (views.WINDOWS["RQ3"]), local date
QH = pd.Timedelta("15min").value
D1_HOUR = 18                               # __d1 origin: D-1 18:00 local
H1_LEAD = pd.Timedelta("1h")               # __h1 origin: quarter-hour start - 1 h
THR_DAYS = 365
MIN_THR_DAYS = 120                         # threshold needs >= MIN_THR_DAYS * 48 prices (= 60 full days)
Q_SHORT = {"q975": 0.975, "q990": 0.990, "q995": 0.995}
Q_LONG = {"q025": 0.025, "q010": 0.010, "q005": 0.005}
MIN_COVER = 0.5                            # lookback needs >= 50 % non-NaN quarter-hours
ZONES = ["ch", "de_lu", "fr", "it_nord", "at"]
BORDERS = ["ch_de", "de_ch", "ch_fr", "fr_ch", "ch_it_nord", "it_nord_ch", "ch_at", "at_ch"]

PRICE_S, PRICE_L = "ch_imb_price_short_eur_mwh", "ch_imb_price_long_eur_mwh"
SYS_IMB = "ch_system_imbalance_mw"

# day-ahead forecasts used for the quarter-hour itself (__fc_d1)
FC_D1 = ([f"{z}_load_da_fc_mw" for z in ZONES]
         + [f"{z}_gen_{g}_da_fc_mw" for z in ZONES for g in ("solar", "wind_on", "total")]
         + [f"{b}_ntc_da_mw" for b in BORDERS] + [f"{b}_sched_da_mw" for b in BORDERS])
# observed series for lookbacks: name -> (master column or None for derived, aggregation)
OBS = {
    PRICE_S: "mean", PRICE_L: "mean",
    f"{PRICE_S}|max": "max", f"{PRICE_L}|min": "min",
    "ch_imb_vol_mwh": "sum",
    "ch_afrr_up_act_mw": "mean", "ch_afrr_down_act_mw": "mean",
    "ch_afrr_up_act_price_eur_mwh": "mean", "ch_afrr_down_act_price_eur_mwh": "mean",
    "ch_mfrr_up_act_all_mw": "mean", "ch_mfrr_down_act_all_mw": "mean",
    "ch_load_actual_mw": "mean",
    "ce_freq_mean_abs_df_mhz": "mean",
}
# forecast errors (actual - DA forecast), published with the actual
ERR = {f"err_{z}_load_mw": (f"{z}_load_actual_mw", f"{z}_load_da_fc_mw") for z in ZONES}
ERR.update({f"err_{z}_{g}_mw": (f"{z}_gen_{g}_mw", f"{z}_gen_{g}_da_fc_mw")
            for z in ZONES for g in ("solar", "wind_on")})
# intraday - day-ahead VRES forecast updates (nowcast only)
IDUP = {f"idup_{z}_{g}_mw": (f"{z}_gen_{g}_id_fc_mw", f"{z}_gen_{g}_da_fc_mw")
        for z in ("de_lu", "fr", "it_nord", "at") for g in ("solar", "wind_on")}
WEATHER = ["ch_temp_lw_degc", "ch_ghi_lw_wm2", "ch_ghi_ramp_lw_wm2", "ch_wind_lw_ms", "ch_hdh_lw_kh",
           "ch_cdh_lw_kh", "ch_precip_hydro_mm", "ch_snow_hydro_cm", "ch_melt_dh_hydro_kh"]
NOWCAST = [PRICE_S, PRICE_L, "ch_imb_vol_mwh", "ch_afrr_up_act_mw", "ch_afrr_down_act_mw",
           "ch_load_actual_mw", "err_ch_load_mw", "err_de_lu_load_mw", "err_de_lu_solar_mw",
           "err_de_lu_wind_on_mw", "err_ch_solar_mw"]

_report: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _report.append(s)


def h(t):
    log(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


# ------------------------------------------------------------------ helpers
def local(ts_ns: np.ndarray) -> pd.Series:
    return BX.naive_local(ts_ns)


def pub_times(ts: np.ndarray, plan: pd.DataFrame, col: str) -> np.ndarray:
    """Publication time (UTC ns) of every quarter-hour of a master column."""
    r = plan.loc[col]
    return BX.lb_pub(ts, r.avail_class, r.avail_param)


def window_agg(x: np.ndarray, pub: np.ndarray, t0: np.ndarray, n: int, kind: str) -> np.ndarray:
    """Aggregate the n latest quarter-hours published by t0 (pub must be non-decreasing)."""
    if (np.diff(pub) < 0).any():
        raise AssertionError("publication times not monotone")
    e = np.searchsorted(pub, t0, side="right")
    s = np.maximum(e - n, 0)
    val, cnt = BX.range_agg(x, kind, s, e)
    val[cnt < MIN_COVER * n] = np.nan
    return val


def last_value(x: np.ndarray, pub: np.ndarray, t0: np.ndarray) -> np.ndarray:
    e = np.searchsorted(pub, t0, side="right")
    out = np.full(len(t0), np.nan)
    ok = e > 0
    out[ok] = x[e[ok] - 1]
    return out


def lag_value(ts: np.ndarray, x: np.ndarray, rows: np.ndarray, days: int) -> np.ndarray:
    """Value of the same quarter-hour `days` days earlier (UTC shift; DST days off by 1 h)."""
    tgt = ts[rows] - days * 86_400 * 10**9
    i = np.searchsorted(ts, tgt)
    ok = (i < len(ts)) & (ts[np.minimum(i, len(ts) - 1)] == tgt)
    out = np.full(len(rows), np.nan)
    out[ok] = x[i[ok]]
    return out


# ------------------------------------------------------------------ targets
def thresholds(ts: np.ndarray, price: np.ndarray, qs: dict, day_of: np.ndarray) -> pd.DataFrame:
    """Per local day d: quantiles of the price over [d - THR_DAYS, d)."""
    days = np.unique(day_of)
    first = np.searchsorted(day_of, days)                    # first row of each day
    out = {k: np.full(len(days), np.nan) for k in qs}
    for j, d in enumerate(days):
        lo = np.searchsorted(day_of, d - np.timedelta64(THR_DAYS, "D"))
        x = price[lo:first[j]]
        x = x[~np.isnan(x)]
        if len(x) < MIN_THR_DAYS * 96 * 0.5:              # hourly-dominant years count per QH too
            continue
        v = np.quantile(x, list(qs.values()))
        for k, q in zip(qs, v):
            out[k][j] = q
    return pd.DataFrame(out, index=days)


def episodes(spike: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.nan_to_num(spike, nan=0.0) > 0.5
    start = s & ~np.concatenate([[False], s[:-1]])
    eid = np.cumsum(start).astype("float64")
    eid[~s] = np.nan
    ln = pd.Series(eid).map(pd.Series(eid).value_counts()).to_numpy(dtype="float64")
    return eid, ln


# ------------------------------------------------------------------ calendar
def calendar(loc: pd.Series) -> pd.DataFrame:
    day = loc.dt.normalize()
    hol = BX.holidays_ch(range(day.dt.year.min() - 1, day.dt.year.max() + 2))
    is_hol = day.isin(hol)
    wd = day.dt.weekday
    prev_h, next_h = (day - pd.Timedelta(days=1)).isin(hol), (day + pd.Timedelta(days=1)).isin(hol)
    # bridge day: working day between a holiday and the weekend (Mon after Tue holiday? no:
    # Monday before a Tuesday holiday, Friday after a Thursday holiday)
    bridge = ((wd == 0) & next_h) | ((wd == 4) & prev_h)
    md = day.dt.month * 100 + day.dt.day
    xmas = (md >= 1224) | (md <= 106)
    # DST switch days: local day with 92 or 100 quarter-hours
    n_qh = loc.groupby(day).transform("size")
    return pd.DataFrame({
        "cal_qh": (loc.dt.hour * 4 + loc.dt.minute // 15).astype("int8"),
        "cal_hour": loc.dt.hour.astype("int8"),
        "cal_dow": wd.astype("int8"),
        "cal_month": loc.dt.month.astype("int8"),
        "cal_year": loc.dt.year.astype("int16"),
        "cal_doy_sin": np.sin(2 * np.pi * loc.dt.dayofyear / 365.25).astype("float32"),
        "cal_doy_cos": np.cos(2 * np.pi * loc.dt.dayofyear / 365.25).astype("float32"),
        "cal_holiday": is_hol.astype("int8"),
        "cal_day_before_holiday": next_h.astype("int8"),
        "cal_day_after_holiday": prev_h.astype("int8"),
        "cal_bridge_day": bridge.astype("int8"),
        "cal_xmas_period": xmas.astype("int8"),
        "cal_dst_switch_day": (n_qh != 96).astype("int8"),
    }, index=loc.index)


# ------------------------------------------------------------------ main
def main(write: bool) -> None:
    dd = pd.read_csv(BX.DD)
    plan = BX.column_plan(dd)
    need = sorted({c.split("|")[0] for c in OBS} | {c for p in ERR.values() for c in p}
                  | {c for p in IDUP.values() for c in p} | set(FC_D1) | set(WEATHER)
                  | {SYS_IMB, "regime_imb_resolution", "regime_imb_pricing"}
                  | set(plan.index[plan["avail_class"] == "outage"]))
    import pyarrow.parquet as pq
    cols_m = set(pq.read_schema(MT).names)
    missing = sorted(c for c in need if c not in cols_m)
    if missing:
        raise KeyError(f"master columns missing: {missing}")
    m = pd.read_parquet(MT, columns=["ts_utc"] + need)
    ts = m["ts_utc"].dt.as_unit("ns").astype("int64").to_numpy()
    if not (np.diff(ts) == QH).all():
        raise AssertionError("master is not a complete 15-min grid")
    loc_all = local(ts)
    day_all = loc_all.dt.normalize().to_numpy()
    start = np.searchsorted(day_all, np.datetime64(START))
    rows = np.arange(start, len(ts))
    log(f"rows: {len(rows):,} quarter-hours from {loc_all.iloc[start]} to {loc_all.iloc[-1]}")

    out: dict[str, np.ndarray] = {}
    roles: dict[str, tuple] = {}            # column -> (role, origin, base, rule)

    def put(name, val, role, origin="", base="", rule=""):
        if isinstance(val, np.ndarray) and val.dtype == np.float64:
            val = val.astype("float32")            # keeps memory in check (~360k rows x ~250 cols)
        out[name] = val
        roles[name] = (role, origin, base, rule)

    # ---------------- targets
    h("1. Targets: prices, rolling thresholds, spikes, episodes")
    ps, pl = m[PRICE_S].to_numpy("float64"), m[PRICE_L].to_numpy("float64")
    sysimb = m[SYS_IMB].to_numpy("float64")
    single = (m["regime_imb_pricing"].astype(str) == "single").to_numpy()
    put("tgt_price_short_eur_mwh", ps[rows], "target", base=PRICE_S)
    put("tgt_price_long_eur_mwh", pl[rows], "target", base=PRICE_L)
    put("tgt_system_imbalance_mw", sysimb[rows], "target", base=SYS_IMB, rule="2026 only")
    put("tgt_imb_vol_mwh", m["ch_imb_vol_mwh"].to_numpy("float64")[rows], "target", base="ch_imb_vol_mwh")
    ts_short = thresholds(ts, ps, Q_SHORT, day_all)
    ts_long = thresholds(ts, pl, Q_LONG, day_all)
    spikes_full = {}
    for side, thr, qs, price, cmp, sgn in (("short", ts_short, Q_SHORT, ps, np.greater, -1),
                                           ("long", ts_long, Q_LONG, pl, np.less, 1)):
        for k in qs:
            t = thr[k].reindex(day_all).to_numpy()
            with np.errstate(invalid="ignore"):
                sp = cmp(price, t).astype("float64")
                # 2026 single price: the side must match the sign of the system imbalance
                wrong = single & ~(np.sign(sysimb) == sgn)
                sp[wrong & ~np.isnan(price)] = 0.0
            sp[np.isnan(price) | np.isnan(t) | (single & np.isnan(sysimb))] = np.nan
            put(f"thr_{side}_{k}", t[rows], "threshold", base=PRICE_S if side == "short" else PRICE_L,
                rule=f"quantile {qs[k]} of the previous {THR_DAYS} days")
            put(f"spike_{side}_{k}", sp[rows], "spike", base=f"thr_{side}_{k}")
            spikes_full[f"{side}_{k}"] = sp
    for side, k in (("short", "q990"), ("long", "q010")):
        eid, ln = episodes(spikes_full[f"{side}_{k}"][rows])
        put(f"episode_{side}_id", eid, "episode", base=f"spike_{side}_{k}")
        put(f"episode_{side}_len_qh", ln, "episode", base=f"spike_{side}_{k}")
    yr = loc_all.iloc[rows].dt.year.to_numpy()
    summ = pd.DataFrame({f"{s}_{k}": pd.Series(out[f"spike_{s}_{k}"]).groupby(yr).mean()
                         for s, qs in (("short", Q_SHORT), ("long", Q_LONG)) for k in qs})
    log("share of quarter-hours flagged as spike, by year:")
    log(summ.round(4).to_string())
    for side in ("short", "long"):
        e = pd.DataFrame({"id": out[f"episode_{side}_id"], "len": out[f"episode_{side}_len_qh"], "yr": yr}).dropna()
        g = e.drop_duplicates("id").groupby("yr")["len"].agg(["size", "median", "max"])
        log(f"\nepisodes ({side}, main threshold): count / median / max length in quarter-hours")
        log(g.to_string())

    # ---------------- calendar + regimes
    h("2. Calendar and regimes")
    cal = calendar(loc_all.iloc[rows].reset_index(drop=True))
    for c in cal:
        put(c, cal[c].to_numpy(), "calendar")
    for c in ("regime_imb_resolution", "regime_imb_pricing"):
        put(c, m[c].astype(str).to_numpy()[rows], "regime")
    log(f"holiday QH share {cal['cal_holiday'].mean():.3f}, bridge {cal['cal_bridge_day'].mean():.4f}, "
        f"DST switch days {cal['cal_dst_switch_day'].mean():.4f}")

    # ---------------- origins
    loc_r = loc_all.iloc[rows]
    t0_d1 = BX.to_ns(loc_r.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=D1_HOUR))
    t0_h1 = ts[rows] - H1_LEAD.value
    # distinct d1 origins (one per day) -> compute once, broadcast
    d1_u, d1_inv = np.unique(t0_d1, return_inverse=True)
    put("t0_d1_utc", t0_d1.astype("datetime64[ns]"), "origin")
    put("t0_h1_utc", t0_h1.astype("datetime64[ns]"), "origin")

    # ---------------- day-ahead forecasts for the QH itself
    h("3. Day-ahead forecasts (__fc_d1)")
    n_ok = 0
    for c in FC_D1:
        r = plan.loc[c]
        if r.avail_class != "fc" or not str(r.avail_param).startswith("D-1"):
            raise AssertionError(f"{c}: not a D-1 forecast ({r.avail_class} {r.avail_param})")
        pub = BX.dw_pub(ts, r.avail_param)[rows]
        known = pub <= t0_d1
        if not known.all():
            raise AssertionError(f"{c}: published after D-1 {D1_HOUR}:00 ({r.avail_param})")
        put(f"{c}__fc_d1", m[c].to_numpy("float64")[rows], "forecast", "d1", c, r.availability_rule)
        n_ok += 1
    x = {c: m[c].to_numpy("float64") for c in ("ch_load_da_fc_mw", "ch_gen_solar_da_fc_mw", "ch_gen_wind_on_da_fc_mw",
                                              "de_lu_load_da_fc_mw", "de_lu_gen_solar_da_fc_mw", "de_lu_gen_wind_on_da_fc_mw")}
    resid_ch = x["ch_load_da_fc_mw"] - np.nan_to_num(x["ch_gen_solar_da_fc_mw"]) - np.nan_to_num(x["ch_gen_wind_on_da_fc_mw"])
    resid_de = x["de_lu_load_da_fc_mw"] - x["de_lu_gen_solar_da_fc_mw"] - x["de_lu_gen_wind_on_da_fc_mw"]
    for name, v in (("ch_resid_load_da_fc_mw", resid_ch), ("de_lu_resid_load_da_fc_mw", resid_de)):
        put(f"{name}__fc_d1", v[rows], "forecast", "d1", name, "derived: load - solar - wind DA forecasts")
        ramp = np.concatenate([[np.nan], np.diff(v)])
        put(f"{name}_ramp__fc_d1", ramp[rows], "forecast", "d1", name, "derived: change vs previous QH")
    for c in ("ch_gen_solar_da_fc_mw", "de_lu_gen_solar_da_fc_mw"):
        ramp = np.concatenate([[np.nan], np.diff(m[c].to_numpy("float64"))])
        put(f"{c.removesuffix('_mw')}_ramp_mw__fc_d1", ramp[rows], "forecast", "d1", c, "derived: change vs previous QH")
    log(f"{n_ok} DA forecast columns + residual load / ramps")

    # ---------------- observed series (lookbacks + nowcast)
    h("4. Lookbacks at D-1 18:00 (__d1) and nowcast at start - 1 h (__h1)")
    series: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}   # name -> (values, pub, kind)
    for key, kind in OBS.items():
        c = key.split("|")[0]
        name = c if "|" not in key else c.replace("_eur_mwh", f"_{kind}_eur_mwh")
        series[name] = (m[c].to_numpy("float64"), pub_times(ts, plan, c), kind)
    for name, (a, f) in ERR.items():
        series[name] = (m[a].to_numpy("float64") - m[f].to_numpy("float64"), pub_times(ts, plan, a), "mean")
    for name, (a, f) in IDUP.items():
        series[name] = (m[a].to_numpy("float64") - m[f].to_numpy("float64"), pub_times(ts, plan, a), "mean")
    p_price = pub_times(ts, plan, PRICE_S)
    for key in ("short_q990", "long_q010"):
        series[f"spikehist_{key}"] = (spikes_full[key], p_price, "mean")
    for c in WEATHER:
        series[c] = (m[c].to_numpy("float64"), pub_times(ts, plan, c), plan.at[c, "kind"])

    for name, (v, pub, kind) in series.items():
        if name.startswith("idup_"):
            continue
        wins = {"lb24h": 96, "lb7d": 672}
        if name.startswith("spikehist_"):
            wins["lb30d"] = 2880
        if name in WEATHER:
            wins = {"lb24h": 96}
        for w, n in wins.items():
            val = window_agg(v, pub, d1_u, n, kind)[d1_inv]
            put(f"{name}__{w}_d1", val, "lookback", "d1", name, f"{kind} of the {n} latest QH published by t0")
    for c in (PRICE_S, PRICE_L, "ch_imb_vol_mwh"):
        v = m[c].to_numpy("float64")
        for d in (2, 7):
            put(f"{c}__lag{d}d_d1", lag_value(ts, v, rows, d), "lag", "d1", c, f"same QH {d} days earlier")
    # nowcast
    for name in NOWCAST + list(IDUP):
        v, pub, kind = series[name]
        put(f"{name}__last_h1", last_value(v, pub, t0_h1), "nowcast", "h1", name, "latest QH published by t0")
        for w, n in (("lb1h", 4), ("lb4h", 16)):
            put(f"{name}__{w}_h1", window_agg(v, pub, t0_h1, n, kind if kind != "sum" else "sum"),
                "nowcast", "h1", name, f"{kind} of the {n} latest QH published by t0")
    for key in ("short_q990", "long_q010"):
        v, pub, _ = series[f"spikehist_{key}"]
        put(f"spikehist_{key}__lb4h_h1", window_agg(v, pub, t0_h1, 16, "mean"), "nowcast", "h1",
            f"spikehist_{key}", "share of spike QH among the 16 latest published")
    for c in (PRICE_S, PRICE_L):
        put(f"{c}__lag1d_h1", lag_value(ts, m[c].to_numpy("float64"), rows, 1), "lag", "h1", c, "same QH 1 day earlier")
    log(f"lookback / nowcast series: {len(series)}")

    # spot check: recompute a few d1 lookbacks with pandas from first principles
    rng = np.random.default_rng(3)
    bad = 0
    tsd = pd.to_datetime(ts, utc=True)
    for i in rng.choice(len(rows), 6, replace=False):
        t0 = pd.Timestamp(t0_d1[i], tz="UTC")
        for name in (PRICE_S, "err_ch_load_mw", "ch_afrr_up_act_mw"):
            v, pub, kind = series[name]
            ok = pd.to_datetime(pub, utc=True) <= t0
            vals = v[np.asarray(ok)][-96:]
            ref = np.nanmean(vals) if np.isfinite(vals).sum() >= MIN_COVER * 96 else np.nan
            got = out[f"{name}__lb24h_d1"][i]
            if not np.isclose(ref, got, rtol=1e-6, atol=1e-6, equal_nan=True):
                bad += 1
                log(f"  !! {t0} {name}: view={got} pandas={ref}")
        # leakage check: nothing published after t0 can enter (h1)
        t1 = pd.Timestamp(t0_h1[i], tz="UTC")
        v, pub, _ = series[PRICE_S]
        e = np.searchsorted(pub, t0_h1[i], side="right")
        if e > 0 and pub[e - 1] > t0_h1[i]:
            bad += 1
            log(f"  !! leakage at {t1}")
    log(f"spot check lookbacks / leakage: mismatches={bad}")
    if bad:
        raise AssertionError("spot check failed")

    # ---------------- outages announced by t0 (day windows)
    h("5. Outages of day D announced by D-1 18:00 (__exante_d1)")
    days = np.unique(day_all[rows])
    dl = pd.Series(pd.DatetimeIndex(days))
    win = pd.DataFrame({"block_start_utc": dl.dt.tz_localize(TZ).dt.tz_convert("UTC"),
                        "block_end_utc": (dl + pd.Timedelta(days=1)).dt.tz_localize(TZ).dt.tz_convert("UTC"),
                        "gate_closure_utc": (dl - pd.Timedelta(days=1) + pd.Timedelta(hours=D1_HOUR))
                        .dt.tz_localize(TZ).dt.tz_convert("UTC")})
    win = win.astype({c: "datetime64[ns, UTC]" for c in win.columns})
    oe = BX.outages_exante(win, plan)
    lm = BX.late_months()
    lm_bad = lm[lm["late"]]
    oe = win.merge(oe, how="left", on=["block_start_utc", "block_end_utc", "gate_closure_utc"])
    oe["day"] = days
    dmonth = pd.Series(pd.DatetimeIndex(days)).dt.to_period("M").values
    pos = np.searchsorted(days, day_all[rows])
    for c in [c for c in oe.columns if c.endswith("__exante")]:
        basec = c.removesuffix("__exante")
        zone, kind = basec.split("_outage_")[0], basec.split("_outage_")[1].split("_")[0]
        badm = set(lm_bad.loc[(lm_bad.zone == zone) & (lm_bad.kind == kind), "month"])
        late = pd.Series(dmonth).isin(badm).to_numpy()
        v = oe[c].fillna(0.0).to_numpy(dtype="float64", copy=True)
        v[late] = np.nan
        has = m[basec].notna().to_numpy()[rows]
        put(f"{c}_d1", np.where(has, v[pos], np.nan), "outage_exante", "d1", basec,
            "day mean of outages announced by t0 (outage_events.created_utc)")
    log(f"days {len(days):,}; outage columns {sum(r[0] == 'outage_exante' for r in roles.values())}")

    # ---------------- weather as perfect forecast (not ex ante)
    for c in WEATHER:
        put(f"{c}__pf", m[c].to_numpy("float64")[rows], "perfect_forecast", "", c,
            "observed weather of the QH: upper bound, NOT ex ante")

    # ---------------- assemble
    h("6. View")
    idx = pd.DatetimeIndex(ts[rows].astype("datetime64[ns]"), tz="UTC", name="ts_utc")
    df = pd.DataFrame({k: (v.astype("float32") if isinstance(v, np.ndarray) and v.dtype == np.float64 else v)
                       for k, v in out.items()}, index=idx)
    for c in ("t0_d1_utc", "t0_h1_utc"):
        df[c] = pd.to_datetime(df[c]).dt.tz_localize("UTC")
    df.insert(0, "ts_local", idx.tz_convert(TZ))
    allnan = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=allnan)
    log(f"rq3_15min: {len(df):,} rows x {df.shape[1]} columns; dropped all-NaN: {allnan}")
    log(f"memory {df.memory_usage(deep=True).sum() / 1e6:,.0f} MB")
    fam = pd.Series([roles[c][0] + (f" ({roles[c][1]})" if roles[c][1] else "") for c in df.columns if c in roles]
                    ).value_counts()
    log(fam.to_string())

    dict_rows = [{"table": "rq3_15min", "column": c, "role": roles.get(c, ("key",))[0],
                  "origin": roles.get(c, ("", ""))[1], "base_column": roles.get(c, ("", "", ""))[2],
                  "rule": roles.get(c, ("", "", "", ""))[3],
                  "ex_ante": roles.get(c, ("key",))[0] not in ("perfect_forecast",)}
                 for c in df.columns]
    if not write:
        log("\nDry run: nothing written. Rerun with --write.")
        return
    df.reset_index().to_parquet(VIEWS_DIR / "rq3_15min.parquet", index=False)
    pd.DataFrame(dict_rows).to_csv(VIEWS_DIR / "_rq3_dictionary.csv", index=False)
    log(f"\nWritten to {VIEWS_DIR.relative_to(CLEAN.parent)}: rq3_15min.parquet, _rq3_dictionary.csv")
    (VIEWS_DIR / "build_view_rq3_report.txt").write_text("\n".join(_report) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
