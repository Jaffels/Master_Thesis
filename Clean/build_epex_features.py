"""RQ3 add-on features from the EPEX SPOT CH data (sensitivity analysis), written 16 Nov 2026.

Path A (30 Oct 2026): rq3_15min.parquet and the main models stay frozen. This script writes a
SEPARATE table, keyed by ts_utc, that is joined onto the RQ3 view only for the EPEX
sensitivity run (appendix): baseline features vs baseline + EPEX features, both estimated and
evaluated on the same EPEX window.

EPEX window (meta_epex_window = 1): 18 Apr 2019 (first CH intraday auction; trades from
1 Jan 2019) to 31 Dec 2025 (licence: historical package up to Y-1, 2026 not licensed).
All features are NaN outside their source coverage - nothing is imputed, nothing for 2026.

Input  Clean/Data/epex/{da,ida,idc_trades}.parquet          (clean_epex.py --write)
       Clean/Data/epex/da_curves.parquet                    (clean_epex_curves.py --write; optional)
       Clean/Data/epex/{idc_results,idc_index}.parquet      (optional, descriptive columns)
       Clean/Data/views/rq3_15min.parquet                   (optional: row set + spike check)
Output (only with --write), Clean/Data/views/:
       rq3_epex.parquet, _rq3_epex_dictionary.csv, build_epex_features_report.txt

Origins (same as build_view_rq3.py)
  __*_d1   t0 = D-1 18:00 local for every quarter-hour of day D
  __*_h1   t0 = quarter-hour start - 1 h (nowcast); h1 models may also use all d1 columns

Features (roles as in _rq3_dictionary.csv, so the same selection logic applies)
  forecast  (d1)  ch_vol_da_epex_mw__fc_d1                  DA traded volume of the QH
                  ch_{price,vol}_ida1_*__fc_d1, ch_spread_ida1_da_eur_mwh__fc_d1
                                                            IDA1 result for the QH and IDA1 - DA
                  ch_idc_{vwap_pre,spread_pre_da,vol_pre,ntrades_pre}_*__fc_d1
                                                            continuous trades for the QH executed by t0
  lookback  (d1)  ch_idc_da_spread_eur_mwh__{lb24h,lb7d}_d1  realised (intraday VWAP - DA) of QHs
                  ch_idc_da_absspread_eur_mwh__{lb24h,lb7d}_d1  delivered by t0, signed / absolute
                  ch_idc_vol_mwh__lb24h_d1                   traded energy of QHs delivered by t0
                  ch_idc_mkt_spread_eur_mwh__lb3h_d1         VWAP(price - DA of the traded QH) of ALL
                  ch_idc_mkt_vol_mwh__lb3h_d1                trades executed in the 3 h before t0
  nowcast   (h1)  ch_{price,vol}_ida2_*__nc_h1, ch_spread_ida2_da_eur_mwh__nc_h1
                  ch_idc_{vwap_pre,last_pre,vwap_3h,range_pre,vol_pre,ntrades_pre}_*__nc_h1
                  ch_idc_spread_{pre_da,last_da,pre_ida1}_eur_mwh__nc_h1
                  ch_idc_da_spread / _absspread __{lb1h,lb4h}_h1, ch_idc_vol_mwh__lb4h_h1
                  ch_idc_mkt_spread_eur_mwh__lb1h_h1, ch_idc_mkt_vol_mwh__lb1h_h1
  forecast  (d1)  ch_da_curve_*__fc_d1 (added 16 Nov 2026): shape of the Swiss day-ahead order book of
                  the hour itself from the aggregated curves (clean_epex_curves.py): price change for
                  +/-100 and +/-500 MW, spare sell / purchase volume, price-taking volumes, curve volume.
                  Published with the day-ahead result (D-1 ~11:10 to 13:00), so known at the d1 origin.
                  Available from 2016 (curves file), i.e. for the whole RQ3 window to 2025.
  descriptive     ex-post values for RQ3a (mechanisms) and figures - NEVER features:
                  ch_idc_{vwap,da_spread,vol}_*__expost (trades), ch_idc_h_*__expost (EOD results,
                  from 2015), ch_idc_{id1,id3}_h_eur_mwh__expost (indices, from Jul 2020)
  meta            meta_epex_window

Timing rules (all Europe/Zurich wall clock)
- Day-ahead auction: D-1 11:00, results ~11:10 -> always before the d1 origin.
- CH-IDA1 (all hours of D) on D-1, CH-IDA2 (afternoon/evening of D) on D: 16:30 / 11:15 until
  20 Sep 2021, 17:40 / 10:30 from 21 Sep 2021 (ECC clearing circulars No. 12/2019, No. 35/2021).
  A result counts as known PUB_LAG (15 min) after the auction. IDA1 therefore just makes the
  18:00 origin from 21 Sep 2021 (17:55); with a larger PUB_LAG it drops out - the report shows
  the share. Later timetable changes (2022-2025) are not modelled: check if needed.
- IDA prices with < 1 MW traded are set to NaN (no-trade clearing prices, incl. the 1750 / 4000 /
  -500 EUR/MWh extremes); the IDA volume stays (0 MW is information).
- Continuous trades: known at their execution time (EPEX trades are public in real time).
  Products used: 60-, 30- and 15-min contracts that contain the QH, no user blocks, no
  self-trades; VWAP weights = traded MW. Volume features = MWh traded for that QH
  (MW x 0.25 h), 0 if nothing traded (inside the trade coverage 2019-2025).
- Realised spreads count once the QH has ended (QH end <= t0); lookbacks need >= 25 % of the
  window's QHs with trades (thin market in 2019/2020), else NaN.

Run from Master_Thesis with .venv active (after clean_epex.py --write):
    python Clean/build_epex_features.py            # build + checks, writes nothing
    python Clean/build_epex_features.py --write

Use in the sensitivity run:
    from build_epex_features import load_rq3_epex, epex_feature_columns
    df = load_rq3_epex()                                   # rq3_15min + EPEX columns, EPEX window only
    base = views.rq3_feature_columns(df, origin="h1")      # frozen baseline set
    extra = epex_feature_columns(df, origin="h1")          # EPEX add-on set
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path(__file__).resolve().parent
sys.path.insert(0, str(CLEAN))
import common as K  # noqa: E402
import config as C  # noqa: E402

TZ = K.TZ
EPEX_DIR = C.DATA_DIR / "epex"
VIEWS_DIR = C.DATA_DIR / "views"
OUT, DICT, REPORT = "rq3_epex.parquet", "_rq3_epex_dictionary.csv", "build_epex_features_report.txt"
RQ3_START = "2016-03-31"                       # build_view_rq3.START (local date)
WINDOW = ("2019-04-18", "2026-01-01")          # EPEX window [start, end) local
TRADES_FROM = "2019-01-01"                     # first trade file (local delivery date)
QH_NS = pd.Timedelta("15min").value
H_NS = pd.Timedelta("1h").value
D1_HOUR = 18
PUB_LAG = pd.Timedelta("15min")                # auction result known PUB_LAG after the auction
IDA_SWITCH = pd.Timestamp("2021-09-21")        # new CH-IDA timetable from this auction date
IDA_TIMES = {1: ("16:30", "17:40"), 2: ("11:15", "10:30")}   # (before, from switch)
MIN_COVER = 0.25
IDA_MIN_MW = 1.0                               # IDA price used only with >= 1 MW traded

_report: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _report.append(s)


def head(t):
    log(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


def read_grid_table(name: str, cols: list[str] | None = None) -> pd.DataFrame | None:
    p = EPEX_DIR / f"{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=None if cols is None else ["ts_utc", *cols])
    df.index = K.to_utc(df.pop("ts_utc"))
    return df.reindex(K.grid().index)


# ------------------------------------------------------------------ time helpers
def local_naive(ts: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(ts.astype("datetime64[ns]")).tz_localize("UTC").tz_convert(TZ).tz_localize(None)


def to_utc_ns(naive_local: pd.DatetimeIndex) -> np.ndarray:
    return naive_local.tz_localize(TZ, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC").asi8


def ida_pub(day: pd.DatetimeIndex, n: int) -> np.ndarray:
    """UTC ns at which the CH-IDA{n} result for local delivery day `day` is known."""
    auction_day = day - pd.Timedelta(days=1) if n == 1 else day
    before, after = IDA_TIMES[n]
    hhmm = np.where(auction_day < IDA_SWITCH, before, after)
    t = auction_day + pd.to_timedelta(pd.Series(hhmm).add(":00").to_numpy()) + PUB_LAG
    return to_utc_ns(pd.DatetimeIndex(t))


# ------------------------------------------------------------------ window sums on the grid
def window_mean(x: np.ndarray, end_ns: np.ndarray, t0: np.ndarray, n: int, how: str = "mean",
                min_cover: float = MIN_COVER) -> np.ndarray:
    """Aggregate the n latest QHs whose end <= t0 (x on the full grid, end_ns sorted)."""
    ok = ~np.isnan(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    e = np.searchsorted(end_ns, t0, side="right")
    s = np.maximum(e - n, 0)
    tot, cnt = cs[e] - cs[s], cn[e] - cn[s]
    if how == "sum":
        out = tot.astype("float64")
        out[cnt < np.ceil(min_cover * n)] = np.nan
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        out = tot / cnt
    out[cnt < max(1, np.ceil(min_cover * n))] = np.nan
    return out


# ------------------------------------------------------------------ trades -> QH level
def explode_trades(tr: pd.DataFrame, g0: int, n_grid: int) -> pd.DataFrame:
    """One row per (trade, covered QH) for 15/30/60-min contracts; pos = grid row."""
    t = tr[~tr["is_block"] & ~tr["self_trade"] & tr["dur_min"].isin([15, 30, 60])]
    st = K.to_utc(t["delivery_start_utc"]).asi8
    k = (t["dur_min"].to_numpy() // 15).astype("int64")
    rep = np.repeat(np.arange(len(t)), k)
    within = np.arange(len(rep)) - np.repeat(np.cumsum(k) - k, k)
    pos = (np.repeat(st, k) - g0) // QH_NS + within
    ex = K.to_utc(t["exec_utc"]).asi8
    out = pd.DataFrame({"pos": pos, "exec": ex[rep], "p": t["price_eur_mwh"].to_numpy("float64")[rep],
                        "w": t["volume_mw"].to_numpy("float64")[rep]})
    out = out[(out["pos"] >= 0) & (out["pos"] < n_grid) & (out["w"] > 0)]
    return out.sort_values(["pos", "exec"], kind="stable").reset_index(drop=True)


def qh_agg(ex: pd.DataFrame, mask: np.ndarray, n_grid: int, extra: bool = False) -> dict:
    """VWAP, MWh, count (and last / range) per grid QH over the rows in mask."""
    e = ex[mask]
    pos = e["pos"].to_numpy()
    sw = np.bincount(pos, weights=e["w"].to_numpy(), minlength=n_grid)
    swp = np.bincount(pos, weights=(e["w"] * e["p"]).to_numpy(), minlength=n_grid)
    cnt = np.bincount(pos, minlength=n_grid).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(sw > 0, swp / sw, np.nan)
    out = {"vwap": vwap, "mwh": sw * 0.25, "n": cnt}
    if extra:
        g = e.groupby("pos")["p"]
        for key, s in (("last", g.last()), ("hi", g.max()), ("lo", g.min())):
            a = np.full(n_grid, np.nan)
            a[s.index.to_numpy()] = s.to_numpy()
            out[key] = a
    return out


def market_window(ex: pd.DataFrame, da: np.ndarray, t0: np.ndarray, width_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """All trades executed in (t0 - width, t0]: VWAP of (price - DA of the traded QH), MWh."""
    e = ex.assign(sp=ex["p"].to_numpy() - da[ex["pos"].to_numpy()]).dropna(subset=["sp"]).sort_values("exec")
    xs = e["exec"].to_numpy()
    cw = np.concatenate([[0.0], np.cumsum(e["w"].to_numpy())])
    cwp = np.concatenate([[0.0], np.cumsum((e["w"] * e["sp"]).to_numpy())])
    hi = np.searchsorted(xs, t0, side="right")
    lo = np.searchsorted(xs, t0 - width_ns, side="right")
    sw, swp = cw[hi] - cw[lo], cwp[hi] - cwp[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sw > 0, swp / sw, np.nan), sw * 0.25


# ------------------------------------------------------------------ build
def build():
    g = K.grid()
    ts = g.index.asi8
    n = len(ts)
    end_ns = ts + QH_NS
    loc = local_naive(ts)
    day = loc.normalize()
    t0_d1 = to_utc_ns(day - pd.Timedelta(days=1) + pd.Timedelta(hours=D1_HOUR))
    t0_h1 = ts - H_NS
    in_win = (loc >= pd.Timestamp(WINDOW[0])) & (loc < pd.Timestamp(WINDOW[1]))
    tr_cov = (loc >= pd.Timestamp(TRADES_FROM)) & (loc < pd.Timestamp(WINDOW[1]))

    cols: dict[str, np.ndarray] = {}
    roles: dict[str, tuple] = {}

    def put(name, val, role, origin="", base="", rule=""):
        cols[name] = np.asarray(val, dtype="float64")
        roles[name] = (role, origin, base, rule)

    put("meta_epex_window", in_win.astype("float64"), "meta", rule=f"1 inside [{WINDOW[0]}, {WINDOW[1]}) local")

    # --- day-ahead (EPEX) ---------------------------------------------------------------
    da_t = read_grid_table("da")
    if da_t is None:
        raise SystemExit("Clean/Data/epex/da.parquet missing: run clean_epex.py --write first")
    da = da_t["ch_price_da_epex_eur_mwh"].to_numpy("float64")
    put("ch_vol_da_epex_mw__fc_d1", da_t["ch_vol_da_epex_mw"], "forecast", "d1", "ch_vol_da_epex_mw",
        "DA auction volume of the QH (results D-1 ~11:10)")

    # --- day-ahead aggregated curves (optional table) --------------------------------------
    CURVE = ["dp_up100_eur_mwh", "dp_up500_eur_mwh", "dp_down100_eur_mwh", "dp_down500_eur_mwh", "head_sell_mw",
             "head_buy_mw", "inel_sell_mw", "inel_buy_mw", "q_mw"]
    cv = read_grid_table("da_curves", [f"ch_da_curve_{c}" for c in CURVE])
    if cv is None:
        print("notice: Clean/Data/epex/da_curves.parquet missing - curve features skipped "
              "(run python Clean/clean_epex_curves.py --write)")
    else:
        for c in CURVE:
            put(f"ch_da_curve_{c}__fc_d1", cv[f"ch_da_curve_{c}"], "forecast", "d1", f"ch_da_curve_{c}",
                "day-ahead aggregated curves of the QH's hour (published with the D-1 auction result)")

    # --- intraday auctions ---------------------------------------------------------------
    ida = read_grid_table("ida")
    if ida is None:
        raise SystemExit("Clean/Data/epex/ida.parquet missing: run clean_epex.py --write first")
    pub1, pub2 = ida_pub(day, 1), ida_pub(day, 2)
    known1, known2 = pub1 <= t0_d1, pub2 <= t0_h1
    p1, v1 = ida["ch_price_ida1_eur_mwh"].to_numpy("float64"), ida["ch_vol_ida1_mw"].to_numpy("float64")
    p2, v2 = ida["ch_price_ida2_eur_mwh"].to_numpy("float64"), ida["ch_vol_ida2_mw"].to_numpy("float64")
    # thin auctions (volume < IDA_MIN_MW): the published price is no market signal -> NaN
    # (volume itself stays: 0 MW traded is information)
    p1 = np.where(v1 >= IDA_MIN_MW, p1, np.nan)
    p2 = np.where(v2 >= IDA_MIN_MW, p2, np.nan)
    r1 = f"CH-IDA1 result, known {PUB_LAG} after the auction (16:30 / 17:40 from 21 Sep 2021, D-1)"
    r2 = f"CH-IDA2 result, known {PUB_LAG} after the auction (11:15 / 10:30 from 21 Sep 2021, D)"
    put("ch_price_ida1_eur_mwh__fc_d1", np.where(known1, p1, np.nan), "forecast", "d1", "ch_price_ida1_eur_mwh", r1)
    put("ch_vol_ida1_mw__fc_d1", np.where(known1, v1, np.nan), "forecast", "d1", "ch_vol_ida1_mw", r1)
    put("ch_spread_ida1_da_eur_mwh__fc_d1", np.where(known1, p1 - da, np.nan), "forecast", "d1",
        "ch_price_ida1_eur_mwh", "IDA1 - EPEX DA price of the QH; " + r1)
    put("ch_price_ida2_eur_mwh__nc_h1", np.where(known2, p2, np.nan), "nowcast", "h1", "ch_price_ida2_eur_mwh", r2)
    put("ch_vol_ida2_mw__nc_h1", np.where(known2, v2, np.nan), "nowcast", "h1", "ch_vol_ida2_mw", r2)
    put("ch_spread_ida2_da_eur_mwh__nc_h1", np.where(known2, p2 - da, np.nan), "nowcast", "h1",
        "ch_price_ida2_eur_mwh", "IDA2 - EPEX DA price of the QH; " + r2)
    raw1 = ida["ch_price_ida1_eur_mwh"].notna().to_numpy()
    raw2 = ida["ch_price_ida2_eur_mwh"].notna().to_numpy()
    ida_info = {"IDA1 published QH known at the d1 origin": known1[raw1].mean() if raw1.any() else np.nan,
                "IDA2 published QH known at the h1 origin": known2[raw2].mean() if raw2.any() else np.nan,
                f"IDA1 published QH with >= {IDA_MIN_MW} MW (price kept)": (~np.isnan(p1))[raw1].mean() if raw1.any() else np.nan,
                f"IDA2 published QH with >= {IDA_MIN_MW} MW (price kept)": (~np.isnan(p2))[raw2].mean() if raw2.any() else np.nan}

    # --- continuous trades ---------------------------------------------------------------
    p = EPEX_DIR / "idc_trades.parquet"
    if not p.exists():
        raise SystemExit("Clean/Data/epex/idc_trades.parquet missing: run clean_epex.py --write first")
    tr = pd.read_parquet(p)
    ex = explode_trades(tr, int(ts[0]), n)
    pos, exe = ex["pos"].to_numpy(), ex["exec"].to_numpy()
    zero_cov = lambda a: np.where(tr_cov, np.nan_to_num(a, nan=0.0), np.nan)  # noqa: E731
    nan_cov = lambda a: np.where(tr_cov, a, np.nan)                           # noqa: E731

    post = qh_agg(ex, np.ones(len(ex), bool), n)
    s_post = post["vwap"] - da
    vol_post = zero_cov(post["mwh"])

    # d1: trades for the QH executed by D-1 18:00
    a = qh_agg(ex, exe <= t0_d1[pos], n)
    rd1 = "continuous trades for the QH (60/30/15-min contracts) executed by t0"
    put("ch_idc_vwap_pre_eur_mwh__fc_d1", nan_cov(a["vwap"]), "forecast", "d1", "idc_trades", "VWAP; " + rd1)
    put("ch_idc_spread_pre_da_eur_mwh__fc_d1", nan_cov(a["vwap"] - da), "forecast", "d1", "idc_trades", "VWAP - DA; " + rd1)
    put("ch_idc_vol_pre_mwh__fc_d1", zero_cov(a["mwh"]), "forecast", "d1", "idc_trades", "MWh; " + rd1)
    put("ch_idc_ntrades_pre_n__fc_d1", zero_cov(a["n"]), "forecast", "d1", "idc_trades", "count; " + rd1)
    for w, nq in (("lb24h", 96), ("lb7d", 672)):
        put(f"ch_idc_da_spread_eur_mwh__{w}_d1", window_mean(s_post, end_ns, t0_d1, nq), "lookback", "d1",
            "idc_trades", f"mean (QH VWAP - DA) of the {nq} latest QHs delivered by t0")
        put(f"ch_idc_da_absspread_eur_mwh__{w}_d1", window_mean(np.abs(s_post), end_ns, t0_d1, nq), "lookback", "d1",
            "idc_trades", f"mean |QH VWAP - DA| of the {nq} latest QHs delivered by t0")
    put("ch_idc_vol_mwh__lb24h_d1", window_mean(vol_post, end_ns, t0_d1, 96, "sum", 1.0), "lookback", "d1",
        "idc_trades", "MWh traded for the 96 latest QHs delivered by t0")
    ms, mv = market_window(ex, da, t0_d1, 3 * H_NS)
    put("ch_idc_mkt_spread_eur_mwh__lb3h_d1", nan_cov(ms), "lookback", "d1", "idc_trades",
        "VWAP of (price - DA of the traded QH) over ALL trades executed in (t0 - 3 h, t0]")
    put("ch_idc_mkt_vol_mwh__lb3h_d1", zero_cov(mv), "lookback", "d1", "idc_trades",
        "MWh of ALL trades executed in (t0 - 3 h, t0]")

    # h1: trades for the QH executed by start - 1 h
    b = qh_agg(ex, exe <= t0_h1[pos], n, extra=True)
    c3 = qh_agg(ex, (exe <= t0_h1[pos]) & (exe > ts[pos] - 3 * H_NS), n)
    rh1 = "continuous trades for the QH executed by t0 = start - 1 h"
    put("ch_idc_vwap_pre_eur_mwh__nc_h1", nan_cov(b["vwap"]), "nowcast", "h1", "idc_trades", "VWAP; " + rh1)
    put("ch_idc_last_pre_eur_mwh__nc_h1", nan_cov(b["last"]), "nowcast", "h1", "idc_trades", "last trade price; " + rh1)
    put("ch_idc_vwap_3h_eur_mwh__nc_h1", nan_cov(c3["vwap"]), "nowcast", "h1", "idc_trades",
        "VWAP of trades executed 3 h to 1 h before the QH start")
    put("ch_idc_range_pre_eur_mwh__nc_h1", nan_cov(b["hi"] - b["lo"]), "nowcast", "h1", "idc_trades", "max - min price; " + rh1)
    put("ch_idc_vol_pre_mwh__nc_h1", zero_cov(b["mwh"]), "nowcast", "h1", "idc_trades", "MWh; " + rh1)
    put("ch_idc_ntrades_pre_n__nc_h1", zero_cov(b["n"]), "nowcast", "h1", "idc_trades", "count; " + rh1)
    put("ch_idc_spread_pre_da_eur_mwh__nc_h1", nan_cov(b["vwap"] - da), "nowcast", "h1", "idc_trades", "VWAP - DA; " + rh1)
    put("ch_idc_spread_last_da_eur_mwh__nc_h1", nan_cov(b["last"] - da), "nowcast", "h1", "idc_trades", "last - DA; " + rh1)
    put("ch_idc_spread_pre_ida1_eur_mwh__nc_h1", nan_cov(b["vwap"] - p1), "nowcast", "h1", "idc_trades",
        "VWAP - IDA1 price of the QH (IDA1 known since D-1 evening); " + rh1)
    for w, nq in (("lb1h", 4), ("lb4h", 16)):
        put(f"ch_idc_da_spread_eur_mwh__{w}_h1", window_mean(s_post, end_ns, t0_h1, nq), "nowcast", "h1",
            "idc_trades", f"mean (QH VWAP - DA) of the {nq} latest QHs delivered by t0")
        put(f"ch_idc_da_absspread_eur_mwh__{w}_h1", window_mean(np.abs(s_post), end_ns, t0_h1, nq), "nowcast", "h1",
            "idc_trades", f"mean |QH VWAP - DA| of the {nq} latest QHs delivered by t0")
    put("ch_idc_vol_mwh__lb4h_h1", window_mean(vol_post, end_ns, t0_h1, 16, "sum", 1.0), "nowcast", "h1",
        "idc_trades", "MWh traded for the 16 latest QHs delivered by t0")
    ms, mv = market_window(ex, da, t0_h1, H_NS)
    put("ch_idc_mkt_spread_eur_mwh__lb1h_h1", nan_cov(ms), "nowcast", "h1", "idc_trades",
        "VWAP of (price - DA of the traded QH) over ALL trades executed in (t0 - 1 h, t0]")
    put("ch_idc_mkt_vol_mwh__lb1h_h1", zero_cov(mv), "nowcast", "h1", "idc_trades",
        "MWh of ALL trades executed in (t0 - 1 h, t0]")

    # --- descriptive (ex post, never features) -----------------------------------------
    put("ch_idc_vwap_eur_mwh__expost", nan_cov(post["vwap"]), "descriptive", "", "idc_trades", "final QH VWAP (all trades)")
    put("ch_idc_da_spread_eur_mwh__expost", nan_cov(s_post), "descriptive", "", "idc_trades", "final QH VWAP - DA")
    put("ch_idc_vol_mwh__expost", vol_post, "descriptive", "", "idc_trades", "MWh traded for the QH")
    res = read_grid_table("idc_results", ["ch_idc_h_vwap_eur_mwh", "ch_idc_h_vol_mw"])
    if res is not None:
        hv = res["ch_idc_h_vwap_eur_mwh"].to_numpy("float64")
        put("ch_idc_h_vwap_eur_mwh__expost", hv, "descriptive", "", "ch_idc_h_vwap_eur_mwh", "EOD VWAP of the hourly contract (from 2015)")
        put("ch_idc_h_da_spread_eur_mwh__expost", hv - da, "descriptive", "", "ch_idc_h_vwap_eur_mwh", "hourly contract VWAP - DA (from 2015)")
        put("ch_idc_h_vol_mw__expost", res["ch_idc_h_vol_mw"], "descriptive", "", "ch_idc_h_vol_mw", "hourly contract traded MW (from 2015)")
    idx = read_grid_table("idc_index", ["ch_idc_id1_h_eur_mwh", "ch_idc_id3_h_eur_mwh"])
    if idx is not None:
        for c in idx.columns:
            put(f"{c}__expost", idx[c], "descriptive", "", c, "EPEX index (window ends 30 min before delivery: after the h1 origin)")

    df = pd.DataFrame(cols, index=g.index)
    # licence: no EPEX-derived value on 2026 rows (lookbacks in early January 2026 would
    # otherwise carry late-2025 values into unlicensed rows)
    after = loc >= pd.Timestamp(WINDOW[1])
    df.loc[after, [c for c in df.columns if c != "meta_epex_window"]] = np.nan
    info = dict(ida=ida_info, n_trade_rows=len(ex), n_trades=len(tr))
    return df, roles, info


# ------------------------------------------------------------------ checks
def auc(score: np.ndarray, y: np.ndarray) -> float:
    ok = ~np.isnan(score) & ~np.isnan(y)
    s, t = score[ok], y[ok] > 0.5
    n1, n0 = t.sum(), (~t).sum()
    if n1 < 20 or n0 < 20:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def report(df: pd.DataFrame, roles: dict, info: dict, rows: pd.DatetimeIndex) -> None:
    head("Coverage")
    log(f"rows {len(df):,} (RQ3 window from {RQ3_START}); trades {info['n_trades']:,} -> {info['n_trade_rows']:,} trade x QH rows")
    for k, v in info["ida"].items():
        log(f"{k}: {v:.3f}" if pd.notna(v) else f"{k}: n/a")
    yr = df.index.tz_convert(TZ).year
    feats = [c for c in df.columns if roles[c][0] not in ("meta",)]
    cov = df[feats].notna().groupby(yr).mean().T.round(2)
    log("non-NaN share per year:\n" + cov.to_string())
    lic = df.loc[df.index.tz_convert(TZ).year >= 2026, feats].notna().sum().sum()
    log(f"\nlicence check: non-NaN values in 2026 = {int(lic)} (must be 0)")
    if lic:
        raise AssertionError("EPEX values found in 2026 (not licensed)")
    head("Distribution inside the EPEX window")
    w = df["meta_epex_window"] == 1
    log(df.loc[w, feats].describe(percentiles=[0.01, 0.5, 0.99]).T.round(2).to_string())

    rq3 = VIEWS_DIR / "rq3_15min.parquet"
    if not rq3.exists():
        log("\n(rq3_15min.parquet not found: univariate spike check skipped)")
        return
    head("Univariate check vs RQ3 spike targets (EPEX window, rank AUC; 0.5 = no signal)")
    tg = ["spike_short_q990", "spike_long_q010"]
    t = pd.read_parquet(rq3, columns=["ts_utc", *tg])
    t.index = K.to_utc(t.pop("ts_utc"))
    j = df.loc[w].join(t, how="inner")
    log(f"rows with targets: {len(j):,}; spike rates: "
        + ", ".join(f"{c} {j[c].mean():.4f}" for c in tg))
    out = []
    for c in feats:
        if roles[c][0] == "descriptive" and not c.endswith("da_spread_eur_mwh__expost"):
            continue
        x = j[c].to_numpy("float64")
        out.append({"feature": c, "origin": roles[c][1] or "-", "non_nan": round(float(np.mean(~np.isnan(x))), 3),
                    **{f"auc_{k}": round(auc(x, j[k].to_numpy("float64")), 3) for k in tg}})
    o = pd.DataFrame(out)
    o["max_dev"] = o[[f"auc_{k}" for k in tg]].sub(0.5).abs().max(axis=1)
    log(o.sort_values("max_dev", ascending=False).drop(columns="max_dev").to_string(index=False))
    log("AUC < 0.5 = higher values go with fewer spikes (sign matters, e.g. long spikes with negative spreads).")
    log("Univariate screening only; the decision comes from the A/B model comparison.")


# ------------------------------------------------------------------ helpers for the sensitivity run
def load_rq3_epex(columns: list[str] | None = None, epex_window_only: bool = True) -> pd.DataFrame:
    """rq3_15min (frozen view) + EPEX add-on columns, joined on ts_utc."""
    import views  # noqa: PLC0415
    base = views.load_rq3(columns)
    ep = pd.read_parquet(VIEWS_DIR / OUT)
    n = len(base)
    df = base.merge(ep, on="ts_utc", how="left", validate="one_to_one")
    assert len(df) == n
    if epex_window_only:
        df = df[df["meta_epex_window"] == 1].reset_index(drop=True)
    return df


def epex_feature_columns(df: pd.DataFrame, origin: str = "d1") -> list[str]:
    """EPEX add-on features known at the origin ('d1', or 'h1' = d1 + h1 columns)."""
    d = pd.read_csv(VIEWS_DIR / DICT).set_index("column")
    ok_roles = {"forecast", "lookback", "lag", "nowcast"}
    origins = {"d1": {"d1"}, "h1": {"d1", "h1"}}[origin]
    return [c for c in df.columns if c in d.index and d.at[c, "role"] in ok_roles and d.at[c, "origin"] in origins]


def main_() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    df, roles, info = build()
    rq3 = VIEWS_DIR / "rq3_15min.parquet"
    if rq3.exists():
        rows = K.to_utc(pd.read_parquet(rq3, columns=["ts_utc"])["ts_utc"])
        df = df.reindex(rows)
    else:
        start = pd.Timestamp(RQ3_START, tz=TZ).tz_convert("UTC")
        df = df[df.index >= start]
    report(df, roles, info, df.index)
    if a.write:
        VIEWS_DIR.mkdir(parents=True, exist_ok=True)
        out = df.astype("float32")
        out["meta_epex_window"] = out["meta_epex_window"].astype("int8")
        out.index.name = "ts_utc"
        out.reset_index().to_parquet(VIEWS_DIR / OUT, index=False)
        pd.DataFrame([{"table": "rq3_epex", "column": c, "role": r[0], "origin": r[1], "base_column": r[2],
                       "rule": r[3], "ex_ante": r[0] in ("forecast", "lookback", "lag", "nowcast")}
                      for c, r in roles.items()]).to_csv(VIEWS_DIR / DICT, index=False)
        (VIEWS_DIR / REPORT).write_text("\n".join(_report))
        log(f"\nwritten: Clean/Data/views/{OUT}, {DICT}, {REPORT}")
    else:
        log("\n(dry run: nothing written; add --write)")


if __name__ == "__main__":
    main_()
