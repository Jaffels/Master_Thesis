"""Modelling views, step 5.3 (+ masks of 5.4): ex-ante feature set per CH target block.

Builds on the ex-post views of build_views.py (step 5.1). Every block gets a forecast
origin t0 = auction gate closure (gate_closure.py), and every feature is restricted to
what was published at or before t0, using each column's availability_rule from
master/data_dictionary.csv (mapped to a class in AVAIL below).

Feature families (column suffix after a double underscore):
  {col}__lb24h, {col}__lb7d   lookback: the column's aggregation_rule over the last 24 h / 7 d
                              of values published by t0 (observed data: quarter-hour end + lag)
  {col}__dw                   delivery-window value of a forecast (D-1, W-1, M-1, Y-1) - only
                              where the whole window's forecast was published by t0, else NaN
  {col}__pf                   'perfect forecast' upper bound, NOT ex ante: delivery-window value
                              of weather and of forecasts published after t0 (RQ2 comparison)
  {zone}_outage_{kind}_{type}_mw__exante
                              delivery-window mean of outages announced by t0
                              (outage_events.created_utc <= t0)
  xm_{de_afrr|de_mfrr}_{dir}_{avg|marg}__{prevday|prevday_slot|lb7d}, xm_fcr_coop_settle__*
                              DE regelleistung.net / FCR cooperation prices of the latest delivery
                              days known at t0 (results of day D known D-1 12:00 local)
  tgt_prev_slot__{x}          same slot (start hour) of the latest auction known at t0
  tgt_prev_auction__{x}       mean over the blocks of the latest auction known at t0
  cal_*                       calendar of the delivery window (known in advance)
  regime_*, src_*             copied from the ex-post view (market-design dates, known in advance)
Keys, targets and auction meta columns are copied from the ex-post view unchanged.

Masks (5.4): TRE activation columns -> NaN where flag_tre_extra_activation_* (always);
*_suspect values -> NaN only with --drop-suspect. Gaps stay NaN (no imputation).

Outputs (only with --write), in Clean/Data/views/:
  {fcr,afrr,mfrr}_exante.parquet       one row per block (same rows as *_expost)
  _exante_dictionary.csv               one row per column per view (role, base column, rule)
  build_views_exante_report.txt

Run from Master_Thesis with .venv active (after python Clean/build_views.py --write):
    python Clean/build_views_exante.py            # build + checks, writes nothing
    python Clean/build_views_exante.py --write
"""
import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CLEAN = Path(__file__).resolve().parent
sys.path.insert(0, str(CLEAN))
from gate_closure import TZ, gate_closure  # noqa: E402

MASTER = CLEAN / "Data" / "master"
MT = MASTER / "master_15min.parquet"
DD = MASTER / "data_dictionary.csv"
EV = CLEAN / "Data" / "outages" / "outage_events.parquet"
VIEWS_DIR = CLEAN / "Data" / "views"
VIEWS = ["fcr", "afrr", "mfrr"]

LOOKBACKS = {"lb24h": 96, "lb7d": 672}       # window length in quarter-hours
MIN_COVERAGE_LB = 0.5                        # share of non-NaN quarter-hours, else NaN
RESULT_LAG = pd.Timedelta("1h")              # auction result known 1 h after gate closure
TGT_LAG_COLS = ["price_settle_ch", "price_bid_vwap", "price_settle_max", "awarded_mw",
                "offered_mw", "n_bids"]
ZERO_INFO = ["at_gen_oil_mw", "de_ch_countertrade_mw", "at_ch_countertrade_mw",
             "it_nord_outage_prod_forced_mw"]
FALLBACK_PUBLISHED = pd.Timestamp("2025-04-15", tz=TZ)   # 2022-24 fall-back data published

# availability_rule -> (class, parameter). Lags are after the end of the quarter-hour.
#   obs     observed value, published `lag` after the quarter-hour       -> lookbacks
#   obs_pf  as obs, plus a perfect-forecast delivery-window column       -> lookbacks + __pf
#   fc      forecast for the delivery day/week/month/year, published at  -> __dw (+ __pf);
#           the given time (D-1 forecasts also get lookbacks with lag 0)
#   fallback  aFRR platform fall-back: next day, 2022-24 only from 15 Apr 2025 -> lookbacks
#   outage  rebuilt from outage_events by created_utc                    -> __exante
#   advance known in advance (regimes)                                   -> copied
#   obs_monthly  published in Swissgrid's monthly files only: value of month M public on the
#           15th of M+1, 00:00 local -> lookbacks use the latest published values (decided 3 Oct 2026)
# Energy Overview columns (*): 1 h only where ENTSO-E published a near-real-time equivalent
# (NRT_TWIN below), otherwise obs_monthly (decided 3 Oct 2026). Swissgrid imbalance prices:
# ENTSO-E 17.1.G near real time -> 1 h.
AVAIL = {
    "ex post (about 1 h after delivery)": ("obs", "1h"),
    "ex post": ("obs", "1h"),
    "ex post (final schedule)": ("obs", "1h"),
    "ex post as final": ("obs", "1h"),
    "ex post (Energy Overview, published with a delay)": ("obs_monthly", None),         # (*) see NRT_TWIN
    "published after delivery (monthly files, mid following month); "
    "ENTSO-E near real time": ("obs", "1h"),                                            # (*)
    "near real time on ENTSO-E (Swissgrid files: monthly, after delivery)": ("obs", "1h"),
    "ex post (weekly CSV, 2026 only)": ("obs", "8D"),
    "ex post (UTC CSV, 2026 only)": ("obs", "1D"),
    "bids: before delivery (gate closure); activations: ex post": ("obs", "1D"),
    "ex post (measured); for RQ2 use lagged values only": ("obs", "1D"),
    "published the following week": ("obs", "8D"),
    "intraday (updated during the day)": ("obs", "0h"),
    "intraday (published hourly)": ("obs", "0h"),
    "ex post (observed); lag to D-1 or treat as perfect forecast in RQ2 views": ("obs_pf", "1h"),
    "D-1": ("fc", "D-1 10:00"),
    "D-1 (daily auction result)": ("fc", "D-1 10:00"),
    "D-1, before day-ahead gate closure (12:00)": ("fc", "D-1 12:00"),
    "D-1 after day-ahead coupling": ("fc", "D-1 13:00"),
    "D-1 (day-ahead)": ("fc", "D-1 13:00"),
    "D-1 (by 18:00)": ("fc", "D-1 18:00"),
    "week W-1": ("fc", "W-1 Fri 18:00"),
    "month M-1": ("fc", "M-1 -7D"),
    "year Y-1": ("fc", "Y-1 15 Dec"),
    "ex post as final; ex-ante views must use outage_events.created": ("outage", None),
    "from Apr 2025 next day; 2022-2024 published 15 Apr 2025 (not known in real time)":
        ("fallback", None),
    "see share": ("fallback", None),
    "known in advance (market-design date)": ("advance", None),
}

# Energy Overview columns with an ENTSO-E near-real-time equivalent -> ("obs", "1h")
NRT_TWIN = {
    "ch_afrr_up_act_mw": "ENTSO-E 17.1.E activated aFRR", "ch_afrr_down_act_mw": "ENTSO-E 17.1.E activated aFRR",
    "ch_afrr_up_act_price_eur_mwh": "ENTSO-E 17.1.F aFRR activation price",
    "ch_afrr_down_act_price_eur_mwh": "ENTSO-E 17.1.F aFRR activation price",
    "ch_cons_mw": "ENTSO-E 6.1.A actual load (same level, r 0.94-0.96)",
    "ch_import_mw": "ENTSO-E 12.1.G physical flows", "ch_export_mw": "ENTSO-E 12.1.G physical flows",
}
MONTHLY_PUB_DAY = 15

_report: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _report.append(s)


def h(t):
    log(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


# ------------------------------------------------------------------ column plan
def agg_kind(rule) -> str:
    r = str(rule).strip().lower()
    if r.startswith("volume-weighted"):
        return "vwmean"
    if r.startswith("mean"):
        return "mean"
    return r


def column_plan(dd: pd.DataFrame) -> pd.DataFrame:
    """One row per master data column: class, parameter, aggregation, weight column."""
    m = dd[(dd["table"] == "master_15min") & (dd["class"] == "data")].copy()
    m = m[~m["column"].isin(ZERO_INFO)]
    unknown = sorted(set(m["availability_rule"].dropna()) - set(AVAIL))
    if unknown or m["availability_rule"].isna().any():
        raise KeyError(f"availability_rule without class in AVAIL: {unknown} "
                       f"(NaN rules: {m.loc[m.availability_rule.isna(), 'column'].tolist()})")
    m["avail_class"] = m["availability_rule"].map(lambda r: AVAIL[r][0])
    m["avail_param"] = m["availability_rule"].map(lambda r: AVAIL[r][1])
    twin = m["column"].isin(list(NRT_TWIN)) & (m["avail_class"] == "obs_monthly")
    m.loc[twin, "avail_class"], m.loc[twin, "avail_param"] = "obs", "1h"
    m["kind"] = m["aggregation_rule"].map(agg_kind)
    bad = m[~m["kind"].isin(["mean", "sum", "max", "min", "vwmean"])]
    if len(bad):
        raise ValueError(f"aggregation not supported for lookbacks:\n{bad[['column', 'kind']]}")
    m["weight"] = np.where(m["kind"] == "vwmean",
                           m["column"].str.replace("_price_eur_mwh", "_mw", regex=False), None)
    return m.set_index("column")[["avail_class", "avail_param", "kind", "weight",
                                  "availability_rule", "flags", "source"]]


# ------------------------------------------------------------------ publication times
def naive_local(ts) -> pd.Series:
    return pd.Series(pd.to_datetime(ts, utc=True)).dt.tz_convert(TZ).dt.tz_localize(None)


def to_ns(naive: pd.Series) -> np.ndarray:
    return naive.dt.tz_localize(TZ).dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()


def dw_pub(ts_ns: np.ndarray, param: str) -> np.ndarray:
    """Publication time (UTC ns) of the forecast value for each quarter-hour."""
    day = naive_local(ts_ns).dt.normalize()
    horizon, rest = param.split(" ", 1)
    if horizon == "D-1":
        pub = day - pd.Timedelta(days=1) + pd.Timedelta(rest + ":00")
    elif horizon == "W-1":            # Friday 18:00 before the delivery week
        pub = day - pd.to_timedelta(day.dt.weekday, unit="D") - pd.Timedelta(days=3) \
            + pd.Timedelta(hours=18)
    elif horizon == "M-1":            # one week before the delivery month
        pub = day.dt.to_period("M").dt.start_time - pd.Timedelta(days=7)
    elif horizon == "Y-1":            # 15 Dec of the year before
        pub = pd.to_datetime((day.dt.year - 1).astype(str) + "-12-15")
    else:
        raise ValueError(param)
    return to_ns(pub)


def lb_pub(ts_ns: np.ndarray, cls: str, param) -> np.ndarray:
    """Publication time (UTC ns) used for lookbacks."""
    end = ts_ns + pd.Timedelta("15min").value
    if cls in ("obs", "obs_pf"):
        return end + pd.Timedelta(param).value
    if cls == "fc":
        return end
    if cls == "obs_monthly":      # 15th of the following month, 00:00 local
        loc = naive_local(ts_ns)
        pub = loc.dt.to_period("M").dt.start_time + pd.offsets.MonthBegin(1) + pd.Timedelta(days=MONTHLY_PUB_DAY - 1)
        return to_ns(pub)
    if cls == "fallback":         # next day (known at the end of it), never before 15 Apr 2025
        nxt = to_ns(naive_local(ts_ns).dt.normalize() + pd.Timedelta(days=2))
        return np.maximum(nxt, FALLBACK_PUBLISHED.tz_convert("UTC").value)
    raise ValueError(cls)


# ------------------------------------------------------------------ range aggregation
def range_agg(x: np.ndarray, kind: str, s: np.ndarray, e: np.ndarray,
              w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate x over [s, e) for every pair; returns (value, non-NaN count)."""
    valid = ~np.isnan(x)
    if kind == "vwmean":
        valid &= ~np.isnan(w)
    cnt = np.concatenate([[0], np.cumsum(valid)])
    c = cnt[e] - cnt[s]
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind in ("mean", "sum"):
            cs = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
            val = cs[e] - cs[s]
            if kind == "mean":
                val = val / c
        elif kind == "vwmean":
            num = np.concatenate([[0.0], np.cumsum(np.where(valid, x * w, 0.0))])
            den = np.concatenate([[0.0], np.cumsum(np.where(valid, w, 0.0))])
            d = den[e] - den[s]
            val = np.where(d != 0, (num[e] - num[s]) / d, np.nan)
        else:
            f = np.fmax if kind == "max" else np.fmin
            idx = np.empty(2 * len(s), dtype=np.int64)
            idx[0::2], idx[1::2] = s, e
            val = f.reduceat(np.append(x, np.nan), idx)[0::2]
    val = np.asarray(val, dtype="float64")
    val[c == 0] = np.nan
    return val, c


# ------------------------------------------------------------------ feature blocks
def load_master(plan: pd.DataFrame, drop_suspect: bool) -> tuple[np.ndarray, pd.DataFrame]:
    flags = sorted({f for fl in plan["flags"].dropna() for f in fl.split(";")})
    cols = ["ts_utc"] + list(plan.index) + flags
    m = pd.read_parquet(MT, columns=cols)
    ts = m["ts_utc"].dt.as_unit("ns").astype("int64").to_numpy()
    if not (np.diff(ts) == pd.Timedelta("15min").value).all():
        raise AssertionError("master grid is not a complete 15-min grid")
    n_masked = {}
    for c, fl in plan["flags"].dropna().items():
        for f in fl.split(";"):
            if f.startswith("flag_tre_extra_activation") or (drop_suspect and f.endswith("_suspect")):
                mask = m[f].fillna(False).astype(bool) & m[c].notna()
                m.loc[mask, c] = np.nan
                n_masked[f"{c} <- {f}"] = int(mask.sum())
    log("masked values (5.4):")
    for k, v in n_masked.items():
        log(f"  {k:75s} {v:>7,}")
    return ts, m.drop(columns=["ts_utc"] + flags)


def lookbacks(ts: np.ndarray, m: pd.DataFrame, plan: pd.DataFrame, t0: np.ndarray) -> pd.DataFrame:
    lb = plan[plan["avail_class"].isin(["obs", "obs_pf", "obs_monthly", "fallback"])
              | ((plan["avail_class"] == "fc") & plan["avail_param"].str.startswith("D-1"))]
    out, iend_cache = {}, {}
    for c, r in lb.iterrows():
        key = (r.avail_class if r.avail_class != "obs_pf" else "obs", r.avail_param
               if r.avail_class in ("obs", "obs_pf") else None)
        if key not in iend_cache:
            pub = lb_pub(ts, r.avail_class, r.avail_param)
            if (np.diff(pub) < 0).any():
                raise AssertionError(f"publication times not monotone for {key}")
            iend_cache[key] = np.searchsorted(pub, t0, side="right")
        i_end = iend_cache[key]
        x = m[c].to_numpy(dtype="float64")
        w = m[r.weight].to_numpy(dtype="float64") if r.kind == "vwmean" else None
        for name, n in LOOKBACKS.items():
            s = np.maximum(i_end - n, 0)
            val, cnt = range_agg(x, r.kind, s, i_end, w)
            val[cnt < MIN_COVERAGE_LB * n] = np.nan
            out[f"{c}__{name}"] = val.astype("float32")
    log(f"lookback features: {len(lb)} columns x {len(LOOKBACKS)} windows = {len(out)}; "
        f"distinct publication rules {len(iend_cache)}")
    return pd.DataFrame(out, index=pd.DatetimeIndex(t0.astype("datetime64[ns]"), tz="UTC",
                                                    name="gate_closure_utc"))


def dw_availability(ts: np.ndarray, plan: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """bool per block x forecast column: whole delivery window published by t0."""
    fc = plan[plan["avail_class"] == "fc"]
    be = blocks["block_end_utc"].dt.as_unit("ns").astype("int64").to_numpy()
    last = np.searchsorted(ts, be, side="left") - 1          # last quarter-hour in the window
    t0 = blocks["gate_closure_utc"].dt.as_unit("ns").astype("int64").to_numpy()
    av, cache = {}, {}
    for c, r in fc.iterrows():
        if r.avail_param not in cache:
            cache[r.avail_param] = dw_pub(ts, r.avail_param)[last] <= t0
        av[c] = cache[r.avail_param]
    return pd.DataFrame(av, index=blocks.index)


LATE_DAYS = 30   # month x zone x kind whose median outage document was created more than
                 # this many days after the outage started -> not reconstructable, __exante NaN
                 # (2021 - mid 2025: all zones' documents carry creation dates of 2025/26)


def outages_exante(windows: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Delivery-window mean of unavailable MW as known at gate closure: only documents
    created <= t0; per unit the latest-created of those documents wins (as in the master).
    Uses the latest revision of each document (clean layer), so a revision made after t0
    leaks its content into earlier origins: accepted limitation."""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("w", windows)
    df = con.execute(f"""
        WITH e AS (SELECT zone, kind, unit_mrid, outage_type, start_utc, end_utc,
                          unavailable_mw AS mw, created_utc,
                          -- tie-break like the master: equal creation time -> later row wins
                          epoch_ns(created_utc)::HUGEINT * 10000000 + file_row_number AS k
                   FROM read_parquet('{EV}', file_row_number = true)),
        j AS (SELECT w.block_start_utc AS bs, w.block_end_utc AS be, w.gate_closure_utc AS t0,
                     e.zone, e.kind, e.unit_mrid, e.outage_type, e.mw, e.k,
                     greatest(e.start_utc, w.block_start_utc) AS s,
                     least(e.end_utc, w.block_end_utc) AS en
              FROM w JOIN e ON e.start_utc < w.block_end_utc AND e.end_utc > w.block_start_utc
                           AND e.created_utc <= w.gate_closure_utc),
        p AS (SELECT bs, be, t0, zone, kind, unit_mrid, s AS p FROM j
              UNION SELECT bs, be, t0, zone, kind, unit_mrid, en AS p FROM j),
        seg AS (SELECT *, lead(p) OVER (PARTITION BY bs, be, t0, zone, kind, unit_mrid
                                        ORDER BY p) AS q FROM p),
        win AS (SELECT seg.bs, seg.be, seg.t0, seg.zone, seg.kind, seg.unit_mrid, seg.p, seg.q,
                       arg_max(j.mw, j.k) AS mw,
                       arg_max(j.outage_type, j.k) AS typ
                FROM seg JOIN j ON seg.bs = j.bs AND seg.be = j.be AND seg.t0 = j.t0
                     AND seg.zone = j.zone AND seg.kind = j.kind AND seg.unit_mrid = j.unit_mrid
                     AND j.s <= seg.p AND j.en >= seg.q
                WHERE seg.q IS NOT NULL
                GROUP BY seg.bs, seg.be, seg.t0, seg.zone, seg.kind, seg.unit_mrid, seg.p, seg.q)
        SELECT bs AS block_start_utc, be AS block_end_utc, t0 AS gate_closure_utc,
               zone || '_outage_' || kind || '_' || typ || '_mw' AS col,
               sum(mw * epoch(q - p)) / epoch(be - bs) AS mw
        FROM win GROUP BY bs, be, t0, col""").df()
    wide = df.pivot_table(index=["block_start_utc", "block_end_utc", "gate_closure_utc"],
                          columns="col", values="mw", aggfunc="sum")
    cols = [c for c in plan.index[plan["avail_class"] == "outage"]]
    missing = sorted(set(wide.columns) - set(cols) - set(ZERO_INFO))
    if missing:
        raise KeyError(f"outage_events give columns not in the master: {missing}")
    wide = wide.reindex(columns=cols)
    wide.columns = [f"{c}__exante" for c in wide.columns]
    wide = wide.reset_index()
    for c in ["block_start_utc", "block_end_utc", "gate_closure_utc"]:
        wide[c] = pd.to_datetime(wide[c], utc=True).astype("datetime64[ns, UTC]")
    return wide


def late_months() -> pd.DataFrame:
    """Median publication lag (created - start, days) per zone x kind x local month of the
    outage start; 'late' if above LATE_DAYS."""
    e = pd.read_parquet(EV, columns=["zone", "kind", "start_utc", "created_utc"])
    e = e.assign(month=naive_local(e["start_utc"].values).dt.to_period("M").values,
                 lag_d=(e["created_utc"] - e["start_utc"]).dt.total_seconds().values / 86400)
    g = e.groupby(["zone", "kind", "month"])["lag_d"].median().rename("median_lag_d").reset_index()
    g["late"] = g["median_lag_d"] > LATE_DAYS
    return g


def holidays_ch(years) -> set:
    from dateutil.easter import easter
    out = set()
    for y in years:
        e = pd.Timestamp(easter(y))
        out |= {pd.Timestamp(y, 1, 1), pd.Timestamp(y, 1, 2), e - pd.Timedelta(days=2),
                e + pd.Timedelta(days=1), e + pd.Timedelta(days=39), e + pd.Timedelta(days=50),
                pd.Timestamp(y, 8, 1), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26)}
    return out


def calendar(blocks: pd.DataFrame) -> pd.DataFrame:
    start = naive_local(blocks["block_start_utc"].values)
    start.index = blocks.index
    day = start.dt.normalize()
    ndays = np.ceil(blocks["duration_h"].to_numpy() / 24).astype(int).clip(min=1)
    hol = holidays_ch(range(day.dt.year.min() - 1, day.dt.year.max() + 2))
    wkend = np.zeros(len(blocks))
    holi = np.zeros(len(blocks))
    for k in range(ndays.max()):
        d = day + pd.Timedelta(days=k)
        use = k < ndays
        wkend += use * (d.dt.weekday >= 5).to_numpy()
        holi += use * d.isin(hol).to_numpy()
    doy = start.dt.dayofyear.to_numpy()
    return pd.DataFrame({
        "cal_hour_local": start.dt.hour.astype("int8"),
        "cal_dow": start.dt.weekday.astype("int8"),
        "cal_month": start.dt.month.astype("int8"),
        "cal_year": start.dt.year.astype("int16"),
        "cal_doy_sin": np.sin(2 * np.pi * doy / 365.25).astype("float32"),
        "cal_doy_cos": np.cos(2 * np.pi * doy / 365.25).astype("float32"),
        "cal_weekend_share": (wkend / ndays).astype("float32"),
        "cal_holiday_share": (holi / ndays).astype("float32"),
    }, index=blocks.index)


def lagged_targets(v: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in TGT_LAG_COLS if c in v.columns]
    by = ["direction", "procurement"]
    b = v[by + ["block_start_utc", "gate_closure_utc"] + cols].copy()
    b["slot"] = np.where(b["procurement"] == "4h",
                         naive_local(b["block_start_utc"].values).dt.hour.to_numpy(), 0)
    b["known_at"] = b["gate_closure_utc"] + RESULT_LAG
    b["_row"] = np.arange(len(b))
    left = b[by + ["slot", "gate_closure_utc", "block_start_utc", "_row"]].sort_values("gate_closure_utc")

    right = b[by + ["slot", "known_at", "block_start_utc"] + cols].sort_values("known_at")
    right = right.rename(columns={"block_start_utc": "prev_block_start_utc",
                                  **{c: f"tgt_prev_slot__{c}" for c in cols}})
    a = pd.merge_asof(left, right, left_on="gate_closure_utc", right_on="known_at",
                      by=by + ["slot"], direction="backward", allow_exact_matches=True)
    if (a["prev_block_start_utc"] >= a["block_start_utc"]).any():
        raise AssertionError("lagged target from a block that does not start earlier")

    auc = b.groupby(by + ["known_at"], as_index=False)[cols].mean().sort_values("known_at")
    auc = auc.rename(columns={c: f"tgt_prev_auction__{c}" for c in cols})
    a2 = pd.merge_asof(left, auc, left_on="gate_closure_utc", right_on="known_at",
                       by=by, direction="backward", allow_exact_matches=True)
    if (a2["known_at"] > a2["gate_closure_utc"]).any() or (a["known_at"] > a["gate_closure_utc"]).any():
        raise AssertionError("lagged target known after gate closure")
    out = a.drop(columns=by + ["slot", "gate_closure_utc", "block_start_utc", "known_at",
                               "prev_block_start_utc"]).merge(
        a2[["_row"] + [f"tgt_prev_auction__{c}" for c in cols]], on="_row")
    return out.sort_values("_row").drop(columns="_row").set_index(v.index)


# ------------------------------------------------------------------ cross-market drivers
AB = MASTER / "auction_blocks.parquet"
XM_KNOWN = pd.Timedelta(hours=12)   # results of delivery day D treated as known at D-1 12:00 local:
# regelleistung.net daily aFRR / mFRR capacity auctions close D-1 09:00 / 10:00 (4h blocks from
# 12 Jul 2018), FCR cooperation D-1 08:00 (4h from Jul 2020; Jul 2019 - Jun 2020 daily, closing
# D-2) -> 12:00 is a conservative publication time. The same-day FCR cooperation result IS the
# CH FCR price, so only earlier delivery days are used (decided 3 Oct 2026).
XM_SERIES = {   # name -> (market, product, direction, price column)
    **{f"xm_de_{p.lower()}_{d}_{k}": ("de_regelleistung", p, d, f"price_{col}_de")
       for p in ("aFRR", "mFRR") for d in ("up", "down") for k, col in (("avg", "average"), ("marg", "marginal"))},
    "xm_fcr_coop_settle": ("fcr_coop", "FCR", "sym", "price_settle_coop"),
}


def cross_market(ex: pd.DataFrame) -> pd.DataFrame:
    """DE regelleistung.net and FCR cooperation prices known at gate closure (EUR/MW/h):
    {name}__prevday       mean over the latest delivery day known at t0
    {name}__prevday_slot  same local start hour on that day (CH 4h blocks only)
    {name}__lb7d          mean over the 7 latest delivery days known at t0 (>= 4 days with data)"""
    ab = pd.read_parquet(AB, columns=["market", "product", "direction", "block_start_utc",
                                      *sorted({v[3] for v in XM_SERIES.values()})])
    loc = naive_local(ab["block_start_utc"].values)
    ab["day"], ab["slot"] = loc.dt.normalize().values, loc.dt.hour.values
    t0 = ex["gate_closure_utc"].dt.as_unit("ns")
    slot = np.where(ex["procurement"] == "4h", naive_local(ex["block_start_utc"].values).dt.hour.to_numpy(), -1)
    left = pd.DataFrame({"t0": t0.reset_index(drop=True), "slot": slot,
                         "_row": np.arange(len(ex))}).sort_values("t0")
    out = {}
    for name, (mk, prod, d, col) in XM_SERIES.items():
        x = ab[(ab["market"] == mk) & (ab["product"] == prod) & (ab["direction"] == d)]
        daily = x.groupby("day")[col].mean().rename("prevday").to_frame()
        slots = x.pivot_table(index="day", columns="slot", values=col, aggfunc="mean")
        daily["lb7d"] = daily["prevday"].rolling("7D", min_periods=4).mean()
        daily["known_at"] = pd.Series(to_ns(pd.Series(daily.index - pd.Timedelta(days=1) + XM_KNOWN)),
                                      index=daily.index).astype("datetime64[ns]").dt.tz_localize("UTC")
        r = daily.join(slots).reset_index().sort_values("known_at")
        j = pd.merge_asof(left, r, left_on="t0", right_on="known_at", direction="backward")
        if (j["known_at"] > j["t0"]).any():
            raise AssertionError(f"{name}: value known after gate closure")
        j = j.sort_values("_row")
        out[f"{name}__prevday"] = j["prevday"].to_numpy()
        sv = np.full(len(j), np.nan)
        for h in slots.columns:
            sel = (j["slot"] == h).to_numpy()
            sv[sel] = j.loc[sel, h].to_numpy() if h in j else np.nan
        out[f"{name}__prevday_slot"] = sv
        out[f"{name}__lb7d"] = j["lb7d"].to_numpy()
        # stale: no delivery day within the last 3 days before t0 -> NaN (series not running)
        age = (j["t0"] - j["known_at"]).dt.total_seconds().to_numpy() / 86400
        for k in ("prevday", "prevday_slot", "lb7d"):
            out[f"{name}__{k}"] = np.where(age <= 3, out[f"{name}__{k}"], np.nan).astype("float32")
    return pd.DataFrame(out, index=ex.index)


# ------------------------------------------------------------------ checks
def spot_check_lookback(ts, m, plan, lbf, name, n=5, seed=1):
    """Recompute a few lookback values with pandas from first principles."""
    rng = np.random.default_rng(seed)
    cands = [c for c in ["ch_load_actual_mw", "ch_imb_vol_mwh", "ch_temp_lw_degc",
                         "ch_hydro_reservoir_mwh", "ch_load_da_fc_mw"] if c in plan.index]
    ser_ts = pd.to_datetime(ts, utc=True)
    bad = 0
    for t0 in rng.choice(lbf.index, size=min(n, len(lbf)), replace=False):
        for c in cands:
            r = plan.loc[c]
            lag = pd.Timedelta(r.avail_param) if r.avail_class in ("obs", "obs_pf") else pd.Timedelta(0)
            ok = (ser_ts + pd.Timedelta("15min") + lag) <= t0
            vals = m[c].to_numpy("float64")[np.asarray(ok)][-LOOKBACKS["lb24h"]:]
            ref = (np.nansum(vals) if r.kind == "sum" else np.nanmean(vals)) \
                if np.isfinite(vals).sum() >= MIN_COVERAGE_LB * LOOKBACKS["lb24h"] else np.nan
            got = lbf.at[t0, f"{c}__lb24h"]
            if not np.isclose(ref, got, rtol=1e-4, atol=1e-3, equal_nan=True):
                bad += 1
                log(f"  !! {name} {t0} {c}: view={got} pandas={ref}")
    log(f"  spot check lookbacks ({name}): {n} origins x {len(cands)} cols, mismatches={bad}")
    if bad:
        raise AssertionError("lookback spot check failed")


# ------------------------------------------------------------------ main
def main(write: bool, drop_suspect: bool) -> None:
    dd = pd.read_csv(DD)
    plan = column_plan(dd)
    for v in VIEWS:
        p = VIEWS_DIR / f"{v}_expost.parquet"
        if not p.exists() or p.stat().st_mtime < MT.stat().st_mtime:
            raise RuntimeError(f"{p.name} missing or older than the master: "
                               "run python Clean/build_views.py --write first")
    vdict = pd.read_csv(VIEWS_DIR / "_dictionary.csv")

    h("1. Availability classes")
    log(plan.groupby(["avail_class", "avail_param"], dropna=False).size().to_string())

    h("2. Gate closure")
    views = {}
    for v in VIEWS:
        ex = pd.read_parquet(VIEWS_DIR / f"{v}_expost.parquet")
        ex = pd.concat([ex, gate_closure(ex)], axis=1)
        views[v] = ex
        t = ex.assign(year=ex.block_start_utc.dt.year).pivot_table(
            index=["product", "procurement", "gc_rule", "gc_confidence"], columns="year",
            values="direction", aggfunc="size", fill_value=0)
        log(t.to_string())
        lead = (ex["block_start_utc"] - ex["gate_closure_utc"]).dt.total_seconds() / 3600
        log(f"  {v}: lead time block start - gate closure [h]: "
            f"min {lead.min():.1f}, median {lead.median():.1f}, max {lead.max():.1f}")

    h("3. Master, masks, lookbacks")
    ts, m = load_master(plan, drop_suspect)
    t0_all = np.unique(np.concatenate([v["gate_closure_utc"].dt.as_unit("ns").astype("int64").to_numpy()
                                       for v in views.values()]))
    log(f"distinct forecast origins: {len(t0_all):,}")
    lbf = lookbacks(ts, m, plan, t0_all)
    spot_check_lookback(ts, m, plan, lbf, "all")

    h("4. Outages announced by gate closure")
    win = pd.concat([v[["block_start_utc", "block_end_utc", "gate_closure_utc"]]
                     for v in views.values()]).drop_duplicates().reset_index(drop=True)
    # validation: with t0 far in the future the rebuild must reproduce the master (ex post)
    ex0 = views["mfrr"].sample(min(3000, len(views["mfrr"])), random_state=0)
    w0 = ex0[["block_start_utc", "block_end_utc"]].drop_duplicates().assign(
        gate_closure_utc=pd.Timestamp("2100-01-01", tz="UTC"))
    chk = ex0[["block_start_utc", "block_end_utc"]].merge(
        outages_exante(w0, plan).drop(columns="gate_closure_utc"),
        how="left", on=["block_start_utc", "block_end_utc"])
    bad, tot = 0, 0
    for c in [c for c in chk.columns if c.endswith("__exante")]:
        ref = ex0[c.removesuffix("__exante")].to_numpy()
        got = chk[c].fillna(0.0).to_numpy()
        ok = ~np.isnan(ref)
        diff = np.abs(got[ok] - ref[ok])
        bad += int((diff > 1.0 + 0.02 * np.abs(ref[ok])).sum())
        tot += int(ok.sum())
    log(f"validation vs master (t0 = 2100): {bad:,} of {tot:,} block-columns differ by > 1 MW + 2 %")
    if bad > 0.01 * tot:
        raise AssertionError("outage rebuild does not reproduce the master")

    out_ex = outages_exante(win, plan)
    log(f"windows {len(win):,}; with announced outages {len(out_ex):,}")
    lm = late_months()
    lm_bad = lm[lm["late"]]
    log(f"months with median outage publication lag > {LATE_DAYS} d (-> __exante NaN):")
    log(lm_bad.groupby(["zone", "kind"])["month"].agg(
        lambda x: f"{len(x)} months, {x.min()} .. {x.max()}").to_string())

    h("5. Views")
    out, dict_rows = {}, []
    fc_cols = list(plan.index[plan["avail_class"] == "fc"])
    pf_weather = list(plan.index[plan["avail_class"] == "obs_pf"])
    for v, ex in views.items():
        roles = vdict[vdict["table"] == f"{v}_expost"].set_index("column")["role"]
        keep = [c for c in ex.columns if roles.get(c) in ("key", "meta", "target", "regime")
                or c.startswith("src_")]
        base = ex[keep + ["gate_closure_utc", "gc_rule", "gc_confidence"]].copy()
        parts = [base, calendar(ex), lagged_targets(ex), cross_market(ex)]

        av = dw_availability(ts, plan, ex)
        fcp = [c for c in fc_cols if c in ex.columns]
        dw = pd.DataFrame({f"{c}__dw": ex[c].where(av[c]) for c in fcp}, index=ex.index)
        pf_cols = [c for c in fcp if not av[c].all()] + [c for c in pf_weather if c in ex.columns]
        pf = pd.DataFrame({f"{c}__pf": ex[c] for c in pf_cols}, index=ex.index)
        parts += [dw, pf]

        lbv = lbf.reindex(pd.DatetimeIndex(ex["gate_closure_utc"]))
        lbv.index = ex.index
        parts.append(lbv)

        o = ex[["block_start_utc", "block_end_utc", "gate_closure_utc"]].merge(
            out_ex, how="left", on=["block_start_utc", "block_end_utc", "gate_closure_utc"],
            validate="many_to_one").drop(columns=["block_start_utc", "block_end_utc",
                                                  "gate_closure_utc"])
        o.index = ex.index
        bmonth = naive_local(ex["block_start_utc"].values).dt.to_period("M").values
        for c in o.columns:   # 0 = no announced outage; NaN where the master has no data
            basec = c.removesuffix("__exante")    # or where announcements were published late
            zone, kind = basec.split("_outage_")[0], basec.split("_outage_")[1].split("_")[0]
            badm = set(lm_bad.loc[(lm_bad.zone == zone) & (lm_bad.kind == kind), "month"])
            late = pd.Series(bmonth).isin(badm).to_numpy()
            o[c] = o[c].fillna(0.0).where(ex[basec].notna() & ~late).astype("float32")
        parts.append(o)

        view = pd.concat(parts, axis=1)
        if view.columns.duplicated().any():
            raise AssertionError(f"{v}: duplicate columns {view.columns[view.columns.duplicated()].tolist()}")
        allnan = [c for c in view.columns if view[c].isna().all()]
        view = view.drop(columns=allnan)

        # checks
        if len(view) != len(ex):
            raise AssertionError(f"{v}: row count changed")
        oc = [c for c in view.columns if c.endswith("__exante")]
        over = sum(int((view[c] > ex[c.removesuffix("__exante")] + 1e-3).sum()) for c in oc)
        known = pd.DataFrame({c: view[c] / ex[c.removesuffix("__exante")].replace(0, np.nan)
                              for c in oc if c.startswith("ch_")})
        log(f"\n{v}_exante: {len(view):,} blocks x {view.shape[1]} cols; dropped all-NaN: {len(allnan)}")
        fam = pd.Series({
            "lookback": sum(c.split("__")[-1] in LOOKBACKS for c in view.columns),
            "forecast __dw": sum(c.endswith("__dw") for c in view.columns),
            "perfect forecast __pf": sum(c.endswith("__pf") for c in view.columns),
            "outage __exante": len(oc),
            "lagged target": sum(c.startswith("tgt_prev_") for c in view.columns),
            "cross-market": sum(c.startswith("xm_") for c in view.columns),
            "calendar": sum(c.startswith("cal_") for c in view.columns)})
        log(fam.to_string())
        log(f"  forecasts available at gate closure (share of blocks): "
            + ", ".join(f"{p}={av[[c for c in fcp if plan.at[c, 'avail_param'] == p]].mean().mean():.2f}"
                        for p in sorted({plan.at[c, 'avail_param'] for c in fcp})))
        log(f"  outages: ex-ante > ex-post in {over} block-columns (announced, later reduced); "
            f"CH announced share by year (ex-ante / ex-post, mean over CH columns):")
        if len(known.columns):
            log("   " + known.groupby(ex["block_start_utc"].dt.year).mean().mean(axis=1)
                .round(2).to_string().replace("\n", "\n   "))
        for c in view.columns:
            basec = c.split("__")[0]
            r = (roles.get(c) if c in keep else
                 "gate_closure" if c.startswith("g") and c in ("gate_closure_utc", "gc_rule", "gc_confidence") else
                 "calendar" if c.startswith("cal_") else
                 "lagged_target" if c.startswith("tgt_prev_") else
                 "cross_market" if c.startswith("xm_") else
                 "perfect_forecast" if c.endswith("__pf") else
                 "forecast" if c.endswith("__dw") else
                 "outage_exante" if c.endswith("__exante") else
                 "lookback" if "__lb" in c else "other")
            if c.startswith("src_"):
                r = "source"
            pr = plan.loc[basec] if basec in plan.index else None
            dict_rows.append({
                "table": f"{v}_exante", "column": c, "role": r, "base_column": basec,
                "ex_ante": r != "perfect_forecast",
                "window": c.split("__")[-1] if "__" in c else "",
                "aggregation": None if pr is None else pr.kind,
                "availability_rule": None if pr is None else pr.availability_rule,
                "availability_class": None if pr is None else pr.avail_class,
                "publication": None if pr is None else pr.avail_param,
                "source": None if pr is None else pr.source})
        out[v] = view

    if not write:
        log("\nDry run: nothing written. Rerun with --write.")
        return
    for v, view in out.items():
        view.to_parquet(VIEWS_DIR / f"{v}_exante.parquet", index=False)
    pd.DataFrame(dict_rows).to_csv(VIEWS_DIR / "_exante_dictionary.csv", index=False)
    log(f"\nWritten to {VIEWS_DIR.relative_to(CLEAN.parent)}")
    (VIEWS_DIR / "build_views_exante_report.txt").write_text("\n".join(_report) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--drop-suspect", action="store_true",
                    help="set *_suspect values to NaN before aggregating (default: keep)")
    a = ap.parse_args()
    main(a.write, a.drop_suspect)
