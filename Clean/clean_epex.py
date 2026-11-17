"""Clean layer: EPEX SPOT CH market data (historical package, up to 2025), written 16 Nov 2026.

Licence: the university package covers CH, historical years only (up to Y-1 = 2025).
2026 data are NOT licensed and not used: anything delivered on/after LICENCE_END is dropped
(none expected). Raw EPEX files stay in EpexSpot_histo/ and must not be redistributed;
only derived features / aggregates go into the thesis.

Input (read in place, zips are not extracted):  EpexSpot_histo/
  Day-Ahead Auction/Hourly/Historical/Prices_Volumes/auction_spot_{prices,volumes}_switzerland_YYYY.csv
  Intraday Auction/CH IDA{1,2}/Historical/Prices_Volumes/intraday_auction_spot_{prices,volumes}_CH-IDA{1,2}_YYYY.csv
  Intraday Continuous/EOD/Historical/Results/      intraday_results_switzerland_YYYY.zip   (2013-2020, old)
                                                   Continuous_Statistics-CH-YYYY.zip       (2021-2025, new)
  Intraday Continuous/EOD/Historical/Transactions/ Continuous_Trades-CH-YYYY.zip           (2019-2025;
                                                   old daily csv to 2020, new csv.zip from 2021; the
                                                   2021 zip holds both -> new format wins per day)
  Intraday Continuous/Indices/Historical/Intraday indices/Continuous_Index-CH-YYYY.zip    (from 21 Jul 2020)
Not used: aggregated curves and block bids (order-book detail, not needed for RQ3); the
quarter-hourly day-ahead folder (empty in the download; the hourly file continues after the
15-min day-ahead go-live on 1 Oct 2025, and ENTSO-E 12.1.D already has the 15-min prices).

Output (Clean/Data/epex/, only with --write)
  da.parquet           ch_price_da_epex_eur_mwh, ch_vol_da_epex_mw                  15-min grid
  ida.parquet          ch_price_ida{1,2}_eur_mwh, ch_vol_ida{1,2}_mw               15-min grid
  idc_results.parquet  end-of-day statistics per product, 60-min (h) and 15-min (qh)
                       contracts: ch_idc_{h,qh}_{vwap,low,high,last}_eur_mwh, ch_idc_{h,qh}_vol_mw
                       EX POST (published after the trading day) -> descriptive only
  idc_index.parquet    EPEX indices ch_idc_{idfull,id3,id1}_{h,qh}_eur_mwh + flag_*_no_trades
                       (index published without traded volume = EPEX fallback value)
  idc_trades.parquet   long table, one row per continuous trade (2019-2025): delivery start/end,
                       execution time, price, volume (MW), product length, block / self-trade flags
  *_dictionary.csv, clean_epex_report.txt
Not added to the master (Path A, 30 Oct 2026: master and main models stay frozen).
build_epex_features.py turns these tables into ex-ante RQ3 features.

Rules
- Local delivery hours (Hour 1 = 00:00-01:00 Europe/Zurich; 3A / 3B = first / second 02:00 hour
  on the October DST day) -> UTC. Hourly values are repeated over their four quarter-hours.
- Volumes are converted to MW (energy per contract / contract length). The 2015 results file
  reports MW, later files MWh: the unit is read from each header.
- Old results files report 0.00 prices for contracts without trades -> prices NaN where the
  traded volume is 0.
- Trades: new-format files list every trade once per side (BUY + SELL) and some trades in two
  daily files -> de-duplicated on TradeId. Old-format execution times: local wall-clock in the
  2019 files (repeated October hour read as CEST, < 1 h per year), UTC in the 2020/2021 files
  (checked against the new format on the 2021 overlap). Delivery hours always local.
- IDA prices are also published for hours with (almost) no volume; those hours carry the
  extreme clearing prices -> flag_ch_ida{1,2}_thin (volume < 1 MW). Prices kept as published.
- Missing stays NaN; raw files are never modified.

Run from Master_Thesis with .venv active:
    python Clean/clean_epex.py            # parse + checks, writes nothing
    python Clean/clean_epex.py --write    # also write Clean/Data/epex/
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import config as C  # noqa: E402

DOMAIN = "epex"
SRC = "EPEX SPOT market data (university historical package, CH)"
RAW = C.ROOT / "EpexSpot_histo"
LICENCE_END = pd.Timestamp("2026-01-01", tz=K.TZ).tz_convert("UTC")   # exclusive: 2026 not licensed
DA_DIR = RAW / "Day-Ahead Auction" / "Hourly" / "Historical" / "Prices_Volumes"
IDA_DIR = RAW / "Intraday Auction"
IDC_RES = RAW / "Intraday Continuous" / "EOD" / "Historical" / "Results"
IDC_TRD = RAW / "Intraday Continuous" / "EOD" / "Historical" / "Transactions"
IDC_IDX = RAW / "Intraday Continuous" / "Indices" / "Historical" / "Intraday indices"
YEAR_RE = re.compile(r"(20\d\d)")
IDA_MIN_MW = 1.0          # IDA price counts as a market signal only with >= 1 MW traded
CODE_RE = re.compile(r"^(\d{1,2})([AB]?)(?:(qh|hh)(\d))?$")

_report: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _report.append(s)


def head(t):
    log(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


def year_of(p: Path) -> int:
    m = YEAR_RE.findall(p.name)
    return int(m[-1]) if m else 0


# ------------------------------------------------------------------ local hour codes -> UTC
def local_starts(day: pd.Series, code: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """EPEX delivery codes ('7', '3A', '3B', '14qh2', '20hh1') of local delivery days ->
    UTC start and contract length in minutes. Codes that do not parse -> NaT."""
    c = code.astype(str).str.strip().str.replace("Hour ", "", regex=False)
    m = c.str.extract(CODE_RE)
    hour = pd.to_numeric(m[0], errors="coerce")
    sub_kind, sub = m[2], pd.to_numeric(m[3], errors="coerce")
    dur = np.where(sub_kind == "qh", 15, np.where(sub_kind == "hh", 30, 60)).astype("int64")
    offs_min = (hour - 1) * 60 + np.where(sub.notna(), (sub.fillna(1) - 1) * dur, 0)
    naive = pd.to_datetime(day, dayfirst=True, format="%d/%m/%Y") + pd.to_timedelta(offs_min, unit="min")
    amb = (m[1] != "B").to_numpy()                       # 3A / plain = CEST on the October day
    ok = hour.notna().to_numpy()
    idx = pd.DatetimeIndex(naive.where(ok))
    loc = idx.tz_localize(K.TZ, ambiguous=amb, nonexistent="NaT")
    return loc.tz_convert("UTC").as_unit("ns"), dur


def read_epex_csv(path_or_buf, **kw) -> pd.DataFrame:
    """EPEX csv with one '#' comment line on top (sometimes trailing empty columns)."""
    df = pd.read_csv(path_or_buf, comment=None, skiprows=1, **kw)
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]


# ------------------------------------------------------------------ wide hourly files (DA, IDA)
def read_wide(files: list[Path]) -> pd.Series:
    """Wide 'Delivery day, Hour 1..24 (3A/3B)' files -> one hourly UTC series. Later files win."""
    parts = []
    for f in sorted(files, key=year_of):
        df = read_epex_csv(f)
        hcols = [c for c in df.columns if str(c).startswith("Hour ")]
        lg = df.melt(id_vars=["Delivery day"], value_vars=hcols, var_name="code", value_name="v")
        lg["v"] = pd.to_numeric(lg["v"], errors="coerce")
        lg = lg.dropna(subset=["v"])
        ts, dur = local_starts(lg["Delivery day"], lg["code"])
        s = pd.Series(lg["v"].to_numpy("float64"), index=ts)
        parts.append(s[s.index.notna()])
    s = pd.concat(parts).sort_index(kind="stable")
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "ts_utc"
    return s[s.index < LICENCE_END]


def build_da():
    p = sorted(DA_DIR.glob("auction_spot_prices_switzerland_*.csv"))
    v = sorted(DA_DIR.glob("auction_spot_volumes_switzerland_*.csv"))
    if not p:
        raise FileNotFoundError(f"no day-ahead files in {DA_DIR}")
    price, vol = read_wide(p), read_wide(v)
    out = pd.DataFrame(index=K.grid().index)
    out["ch_price_da_epex_eur_mwh"] = K.hourly_to_qh(price, "repeat")
    out["ch_vol_da_epex_mw"] = K.hourly_to_qh(vol, "repeat")       # MWh per hour = average MW
    dd = [dict(column="ch_price_da_epex_eur_mwh", source=SRC, source_series="Day-Ahead Auction hourly prices CH",
               unit="EUR/MWh", resolution_native="1h", aggregation_rule="mean",
               availability_rule="D-1 after CH day-ahead auction (order book closes 11:00, results from ~11:10)",
               notes="from 1 Oct 2025 the hourly file of the 15-min day-ahead market (hourly value); "
                     "ENTSO-E 12.1.D has the 15-min prices"),
          dict(column="ch_vol_da_epex_mw", source=SRC, source_series="Day-Ahead Auction hourly volumes CH",
               unit="MW", resolution_native="1h", aggregation_rule="mean",
               availability_rule="D-1 after CH day-ahead auction", notes="MWh per hour = average MW")]
    return out, dd


def build_ida():
    out = pd.DataFrame(index=K.grid().index)
    dd = []
    for n in (1, 2):
        d = IDA_DIR / f"CH IDA{n}" / "Historical" / "Prices_Volumes"
        p = sorted(d.glob(f"intraday_auction_spot_prices_CH-IDA{n}_*.csv"))
        v = sorted(d.glob(f"intraday_auction_spot_volumes_CH-IDA{n}_*.csv"))
        if not p:
            raise FileNotFoundError(f"no IDA{n} files in {d}")
        out[f"ch_price_ida{n}_eur_mwh"] = K.hourly_to_qh(read_wide(p), "repeat")
        out[f"ch_vol_ida{n}_mw"] = K.hourly_to_qh(read_wide(v), "repeat")
        out[f"flag_ch_ida{n}_thin"] = (out[f"ch_price_ida{n}_eur_mwh"].notna()
                                       & ~(out[f"ch_vol_ida{n}_mw"] >= IDA_MIN_MW))
        when = ("D-1 (16:30 until 20 Sep 2021, 17:40 from 21 Sep 2021), all hours of D" if n == 1 else
                "D (11:15 until 20 Sep 2021, 10:30 from 21 Sep 2021), afternoon / evening hours of D")
        dd += [dict(column=f"ch_price_ida{n}_eur_mwh", source=SRC, source_series=f"CH-IDA{n} prices", unit="EUR/MWh",
                    resolution_native="1h", aggregation_rule="mean", availability_rule=f"after CH-IDA{n} auction: {when}",
                    notes="starts 18 Apr 2019 (CH-IT implicit intraday auctions); price also set when volume is 0"),
               dict(column=f"ch_vol_ida{n}_mw", source=SRC, source_series=f"CH-IDA{n} volumes", unit="MW",
                    resolution_native="1h", aggregation_rule="mean", availability_rule=f"after CH-IDA{n} auction",
                    notes="MWh per hour = average MW"),
               dict(column=f"flag_ch_ida{n}_thin", source="derived", source_series=f"ch_vol_ida{n}_mw", unit="bool",
                    resolution_native="1h", aggregation_rule="any", availability_rule="-",
                    notes=f"price set with volume < {IDA_MIN_MW} MW (or volume missing): not a market signal - "
                          "these hours carry the extreme clearing prices (1750 / 4000 / -500 EUR/MWh)")]
    return out, dd


# ------------------------------------------------------------------ continuous: EOD statistics
def _vol_to_mw(vol: pd.Series, unit_is_mwh: bool | np.ndarray, dur_min: np.ndarray) -> np.ndarray:
    v = pd.to_numeric(vol, errors="coerce").to_numpy("float64")
    return np.where(unit_is_mwh, v / (dur_min / 60.0), v)


def read_results_old(z: Path) -> pd.DataFrame:
    with zipfile.ZipFile(z) as Z:
        name = [n for n in Z.namelist() if "results_hours" in n]
        if not name:
            return pd.DataFrame()
        return _results_old_frame(read_epex_csv(Z.open(name[0])))


def _results_old_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Old-format 'Hour Values' results (2013-2021) -> one row per contract."""
    cols = {c: c for c in df.columns}
    vb = [c for c in df.columns if c.startswith("Volume Buy")][0]
    unit_mwh = "MWh" in vb
    same = df["Hour from"].astype(str) == df["Hour to"].astype(str)
    df = df[same].copy()
    start, dur = local_starts(df["Delivery day"], df["Hour from"])
    pick = lambda pre: [c for c in cols if c.startswith(pre)][0]  # noqa: E731
    out = pd.DataFrame({
        "start_utc": start, "dur_min": dur,
        "low": pd.to_numeric(df[pick("Low Price")], errors="coerce").to_numpy(),
        "high": pd.to_numeric(df[pick("High Price")], errors="coerce").to_numpy(),
        "last": pd.to_numeric(df[pick("Last Price")], errors="coerce").to_numpy(),
        "vwap": pd.to_numeric(df[pick("Weighted Average")], errors="coerce").to_numpy(),
        "vol_mw": _vol_to_mw(df[vb], unit_mwh, dur),
    })
    out["src"] = "old"
    return out[out["start_utc"].notna()]


def read_statistics_new(z: Path) -> pd.DataFrame:
    parts, olds = [], []
    with zipfile.ZipFile(z) as Z:
        for n in Z.namelist():
            if not n.endswith(".csv"):
                continue
            d = read_epex_csv(Z.open(n))
            if "DeliveryStart" in d.columns:
                parts.append(d)
            elif "results_hours" in n:                      # 2021 zip also holds the old yearly file
                olds.append(_results_old_frame(d))
    old = pd.concat(olds, ignore_index=True) if olds else pd.DataFrame()
    if not parts:
        return old
    df = pd.concat(parts, ignore_index=True)
    df = df[df["DeliveryStart"].notna() & df["DeliveryEnd"].notna()]
    st = pd.to_datetime(df["DeliveryStart"], utc=True, format="ISO8601")
    en = pd.to_datetime(df["DeliveryEnd"], utc=True, format="ISO8601")
    dur = ((en - st).dt.total_seconds() / 60).round().astype("int64").to_numpy()
    out = pd.DataFrame({
        "start_utc": st.dt.as_unit("ns"), "dur_min": dur,
        "low": pd.to_numeric(df["LowPrice"], errors="coerce"),
        "high": pd.to_numeric(df["HighPrice"], errors="coerce"),
        "last": pd.to_numeric(df["LastPrice"], errors="coerce"),
        "vwap": pd.to_numeric(df["WeightedAveragePrice"], errors="coerce"),
        "vol_mw": _vol_to_mw(df["VolumeBuy"], df["VolumeUnit"].astype(str).str.upper().eq("MWH").to_numpy(), dur),
    })
    out["src"] = "new"
    return pd.concat([old, out], ignore_index=True) if len(old) else out


def build_idc_results():
    old = [read_results_old(z) for z in sorted(IDC_RES.glob("intraday_results_switzerland_*.zip"))]
    new = [read_statistics_new(z) for z in sorted(IDC_RES.glob("Continuous_Statistics-CH-*.zip"))]
    df = pd.concat([d for d in old + new if len(d)], ignore_index=True)
    df = df[df["start_utc"] < LICENCE_END]
    df["_new"] = (df["src"] == "new").astype("int8")
    df = df.sort_values(["_new", "start_utc"], kind="stable").drop(columns="_new")
    df = df.drop_duplicates(["start_utc", "dur_min"], keep="last")        # new over old
    no_trade = ~(df["vol_mw"] > 0)
    df.loc[no_trade, ["low", "high", "last", "vwap"]] = np.nan
    out = pd.DataFrame(index=K.grid().index)
    dd = []
    for tag, dur in (("h", 60), ("qh", 15)):
        b = df[df["dur_min"] == dur].copy()
        b["block_start_utc"] = b["start_utc"]
        b["block_end_utc"] = b["start_utc"] + pd.Timedelta(minutes=dur)
        vals = ["vwap", "low", "high", "last", "vol_mw"]
        q = K.blocks_to_qh(b, vals)
        for v in vals:
            col = f"ch_idc_{tag}_{v}" + ("" if v == "vol_mw" else "_eur_mwh")
            out[col] = q[v]
            dd.append(dict(column=col, source=SRC,
                           source_series=f"Intraday Continuous EOD results ({dur}-min contracts)",
                           unit="MW" if v == "vol_mw" else "EUR/MWh", resolution_native=f"{dur}min",
                           aggregation_rule="mean", availability_rule="EX POST: after the trading day (EOD file)",
                           notes=("volume buy, converted to MW (0 = no trades)" if v == "vol_mw"
                                  else f"{v} price of the contract; NaN without trades")))
    return out, dd, df


# ------------------------------------------------------------------ continuous: indices
def build_idc_index():
    parts = []
    for z in sorted(IDC_IDX.glob("Continuous_Index-CH-*.zip")):
        with zipfile.ZipFile(z) as Z:
            for n in Z.namelist():
                if not n.endswith(".csv"):
                    continue
                raw = Z.read(n)
                first = raw[:200].decode("utf-8", "ignore")
                if first.startswith("#"):                      # new format
                    d = read_epex_csv(io.BytesIO(raw))
                    vol = d["IndexVolume"]
                    fmt = 1
                else:                                          # old format (Jul 2020 - 2021)
                    d = pd.read_csv(io.BytesIO(raw), sep=";" if ";" in first.splitlines()[0] else ",")
                    vol = d["Volume"]
                    fmt = 0
                parts.append(pd.DataFrame({
                    "name": d["IndexName"].astype(str).str.upper(),
                    "res": d["TimeResolution"].astype(str).str.upper(),
                    "start": d["DeliveryStart"].astype(str), "end": d["DeliveryEnd"].astype(str),
                    "price": pd.to_numeric(d["IndexPrice"], errors="coerce"),
                    "vol": pd.to_numeric(vol, errors="coerce"), "fmt": fmt,
                    "vol_reported": pd.to_numeric(vol, errors="coerce").notna().any()}))
    if not parts:
        raise FileNotFoundError(f"no index files in {IDC_IDX}")
    df = pd.concat(parts, ignore_index=True)
    df["res_min"] = df["res"].map({"H": 60, "60MIN": 60, "HH": 30, "30MIN": 30, "QH": 15, "15MIN": 15})
    df = df[df["res_min"].isin([60, 15]) & df["name"].isin(["IDFULL", "ID3", "ID1"])].copy()
    df["start_utc"] = pd.to_datetime(df["start"], utc=True, format="ISO8601").dt.as_unit("ns")
    df["end_utc"] = pd.to_datetime(df["end"], utc=True, format="ISO8601").dt.as_unit("ns")
    df = df[df["start_utc"] < LICENCE_END]
    df = df.sort_values("fmt").drop_duplicates(["name", "res_min", "start_utc"], keep="last")
    out = pd.DataFrame(index=K.grid().index)
    dd = []
    for name in ("IDFULL", "ID3", "ID1"):
        for tag, rm in (("h", 60), ("qh", 15)):
            b = df[(df["name"] == name) & (df["res_min"] == rm)].rename(
                columns={"start_utc": "block_start_utc", "end_utc": "block_end_utc"})
            # files that report no volume at all (some old-format days) -> 'no trades' unknown
            b = b.assign(no_tr=np.where(b["vol_reported"], (~(b["vol"] > 0)).astype("float64"), np.nan))
            q = K.blocks_to_qh(b, ["price", "no_tr"])
            col, fl = f"ch_idc_{name.lower()}_{tag}_eur_mwh", f"flag_ch_idc_{name.lower()}_{tag}_no_trades"
            out[col] = q["price"]
            out[fl] = (q["no_tr"] == 1.0)
            win = {"IDFULL": "all trades of the contract", "ID3": "trades 3 h to 30 min before delivery",
                   "ID1": "trades 1 h to 30 min before delivery"}[name]
            dd += [dict(column=col, source=SRC, source_series=f"EPEX intraday index {name} ({rm}-min)",
                        unit="EUR/MWh", resolution_native=f"{rm}min", aggregation_rule="mean",
                        availability_rule="EX POST for RQ3 (window ends 30 min before delivery, after the 1 h nowcast origin)",
                        notes=f"VWAP of {win}; from 21 Jul 2020"),
                   dict(column=fl, source="derived", source_series=col, unit="bool", resolution_native=f"{rm}min",
                        aggregation_rule="any", availability_rule="-",
                        notes="index published without traded volume (EPEX fallback value), use with care")]
    return out, dd


# ------------------------------------------------------------------ continuous: trades
def _trades_old(buf, src_name: str) -> pd.DataFrame:
    d = read_epex_csv(buf)
    if d.empty:
        return pd.DataFrame()
    # 2019 files ('Volume (MW)') stamp executions in local time; 2020/2021 files ('Quantity (MW)')
    # in UTC - verified on 167,650 trades of 2021 that appear in both formats (old - new = exactly
    # -1 h / -2 h when read as local), and by the minimum lead time per month (0.5 h gate closure).
    exec_is_utc = any(str(c).startswith("Quantity") for c in d.columns)
    d.columns = ["day", "area_buy", "area_sell", "h_from", "h_to", "vol", "price", "exec", "trade_id"][:len(d.columns)]
    st, dur = local_starts(d["day"], d["h_from"])
    st_to, dur_to = local_starts(d["day"], d["h_to"])
    block = (d["h_from"].astype(str) != d["h_to"].astype(str)).to_numpy()
    en = np.where(block, (st_to + pd.to_timedelta(dur_to, unit="min")).asi8, (st + pd.to_timedelta(dur, unit="min")).asi8)
    ex_naive = pd.DatetimeIndex(pd.to_datetime(d["exec"], dayfirst=True, format="mixed"))
    if exec_is_utc:
        ex = ex_naive.tz_localize("UTC").as_unit("ns")
    else:
        ex = ex_naive.tz_localize(K.TZ, ambiguous=np.ones(len(d), bool),
                                  nonexistent="shift_forward").tz_convert("UTC").as_unit("ns")
    out = pd.DataFrame({
        "trade_id": pd.to_numeric(d["trade_id"], errors="coerce").astype("Int64"),
        "delivery_start_utc": st, "delivery_end_utc": pd.to_datetime(en, utc=True).as_unit("ns"),
        "exec_utc": ex,
        "price_eur_mwh": pd.to_numeric(d["price"], errors="coerce").to_numpy("float64"),
        "volume_mw": pd.to_numeric(d["vol"], errors="coerce").to_numpy("float64"),
        "is_block": block, "self_trade": False,
        "cross_border": ((d["area_buy"].astype(str) != "CH") | (d["area_sell"].astype(str) != "CH")).to_numpy(),
        "src_format": "old"})
    return out


def _trades_new(buf) -> pd.DataFrame:
    d = read_epex_csv(buf)
    if d.empty:
        return pd.DataFrame()
    d = d.drop_duplicates("TradeId", keep="first")
    st = pd.to_datetime(d["DeliveryStart"], utc=True, format="ISO8601").dt.as_unit("ns")
    en = pd.to_datetime(d["DeliveryEnd"], utc=True, format="ISO8601").dt.as_unit("ns")
    dur_h = (en - st).dt.total_seconds().to_numpy() / 3600.0
    v = pd.to_numeric(d["Volume"], errors="coerce").to_numpy("float64")
    mwh = d["VolumeUnit"].astype(str).str.upper().eq("MWH").to_numpy()
    out = pd.DataFrame({
        "trade_id": pd.to_numeric(d["TradeId"], errors="coerce").astype("Int64").to_numpy(),
        "delivery_start_utc": st.to_numpy(), "delivery_end_utc": en.to_numpy(),
        "exec_utc": pd.to_datetime(d["ExecutionTime"], utc=True, format="ISO8601").dt.as_unit("ns").to_numpy(),
        "price_eur_mwh": pd.to_numeric(d["Price"], errors="coerce").to_numpy("float64"),
        "volume_mw": np.where(mwh, v / np.where(dur_h > 0, dur_h, np.nan), v),
        "is_block": (d["UserDefinedBlock"].astype(str).str.upper().eq("Y")
                     | ~np.isin(np.round(dur_h * 60), [15, 30, 60])).to_numpy(),
        "self_trade": d["SelfTrade"].astype(str).str.upper().eq("Y").to_numpy(),
        "cross_border": ~d["DeliveryArea"].astype(str).eq("CH").to_numpy(),
        "trade_phase": d["TradePhase"].astype(str).to_numpy(),
        "src_format": "new"})
    out = out[out["trade_phase"] == "CONT"].drop(columns="trade_phase")
    return out


def build_trades():
    old, new = [], []
    for z in sorted(IDC_TRD.glob("Continuous_Trades-CH-*.zip")):
        with zipfile.ZipFile(z) as Z:
            for n in Z.namelist():
                if n.endswith(".csv.zip"):
                    with zipfile.ZipFile(io.BytesIO(Z.read(n))) as I:
                        for m in I.namelist():
                            new.append(_trades_new(I.open(m)))
                elif n.endswith(".zip"):                       # old yearly zip inside (2021)
                    with zipfile.ZipFile(io.BytesIO(Z.read(n))) as I:
                        for m in I.namelist():
                            if m.endswith(".csv"):
                                old.append(_trades_old(I.open(m), m))
                elif n.endswith(".csv"):
                    first = Z.read(n)[:120].decode("utf-8", "ignore")
                    (new.append(_trades_new(Z.open(n))) if first.startswith("# Continuous")
                     else old.append(_trades_old(Z.open(n), n)))
    o = pd.concat([x for x in old if len(x)], ignore_index=True) if old else pd.DataFrame()
    w = pd.concat([x for x in new if len(x)], ignore_index=True) if new else pd.DataFrame()
    n_old_raw, n_new_raw = len(o), len(w)
    if len(w):
        w = w.drop_duplicates("trade_id", keep="first")
    if len(o):
        o = o.dropna(subset=["delivery_start_utc"]).drop_duplicates("trade_id", keep="first")
        if len(w):                                            # new format wins per local delivery day
            day = lambda s: pd.to_datetime(s, utc=True).dt.tz_convert(K.TZ).dt.date  # noqa: E731
            new_days = set(day(w["delivery_start_utc"]))
            o = o[~day(o["delivery_start_utc"]).isin(new_days)]
    df = pd.concat([o, w], ignore_index=True)
    df = df[df["delivery_start_utc"] < LICENCE_END]
    df["dur_min"] = ((df["delivery_end_utc"] - df["delivery_start_utc"]).dt.total_seconds() / 60).round().astype("int32")
    df["src_format"] = df["src_format"].astype("category")
    for c in ("is_block", "self_trade", "cross_border"):
        df[c] = df[c].astype(bool)
    df = df.sort_values(["delivery_start_utc", "exec_utc", "trade_id"]).reset_index(drop=True)
    info = dict(old_raw=n_old_raw, new_raw=n_new_raw, kept=len(df))
    return df, info


TRADE_DICT = [
    dict(column="trade_id", unit="-", notes="EPEX trade id (unique after de-duplication)"),
    dict(column="delivery_start_utc", unit="UTC", notes="contract delivery start"),
    dict(column="delivery_end_utc", unit="UTC", notes="contract delivery end"),
    dict(column="exec_utc", unit="UTC", notes="execution time (old format: local minutes/seconds -> UTC)"),
    dict(column="price_eur_mwh", unit="EUR/MWh", notes="trade price"),
    dict(column="volume_mw", unit="MW", notes="traded power (MWh / contract length)"),
    dict(column="is_block", unit="bool", notes="user-defined block or non 15/30/60-min contract (not used in features)"),
    dict(column="self_trade", unit="bool", notes="EPEX self-trade flag (new format only)"),
    dict(column="cross_border", unit="bool", notes="one side outside CH"),
    dict(column="src_format", unit="-", notes="old (daily csv to 2020/21) or new (Continuous_Trades csv.zip)"),
    dict(column="dur_min", unit="min", notes="contract length"),
]


# ------------------------------------------------------------------ main
def yearly(df: pd.DataFrame, cols: list[str], fn: str = "notna") -> pd.DataFrame:
    """Per local year: non-NaN share (fn='notna') or mean (fn='mean')."""
    yr = df.index.tz_convert(K.TZ).year
    x = df[cols].notna() if fn == "notna" else df[cols]
    return x.groupby(yr).mean().round(3)


def main_() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--only", nargs="+", choices=["da", "ida", "results", "index", "trades"],
                    help="build only these tables")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    todo = set(a.only or ["da", "ida", "results", "index", "trades"])
    if not RAW.exists():
        raise SystemExit(f"raw folder not found: {RAW}")
    written = []

    if "da" in todo:
        head("Day-ahead auction (EPEX CH, hourly)")
        da, dd = build_da()
        log("non-NaN share per year:\n" + yearly(da, list(da.columns)).to_string())
        log("yearly mean:\n" + yearly(da, list(da.columns), "mean").to_string())
        ent = C.DATA_DIR / "prices" / "day_ahead.parquet"
        if ent.exists():
            e = pd.read_parquet(ent, columns=["ts_utc", "ch_price_da_eur_mwh"]).set_index("ts_utc")
            e.index = K.to_utc(e.index)
            j = da[["ch_price_da_epex_eur_mwh"]].join(e, how="inner").dropna()
            j["d"] = (j["ch_price_da_epex_eur_mwh"] - j["ch_price_da_eur_mwh"]).abs()
            g = j.groupby(j.index.tz_convert(K.TZ).year)
            log("cross-check vs ENTSO-E ch_price_da_eur_mwh (|diff| EUR/MWh; share > 0.01):\n"
                + pd.DataFrame({"mean_abs_diff": g["d"].mean().round(3), "share_diff": g["d"].apply(lambda x: (x > 0.01).mean()).round(4),
                                "n": g.size()}).to_string())
            log("note: from 1 Oct 2025 EPEX hourly vs ENTSO-E 15-min values differ by construction")
        else:
            log(f"(ENTSO-E clean day-ahead table not found at {ent}: cross-check skipped)")
        if a.write:
            written.append(K.write_clean(da, DOMAIN, "da", dd))

    if "ida" in todo:
        head("Intraday auctions CH-IDA1 / CH-IDA2")
        ida, dd = build_ida()
        log("non-NaN share per year:\n" + yearly(ida, list(ida.columns)).to_string())
        log("yearly mean:\n" + yearly(ida, list(ida.columns), "mean").to_string())
        for n in (1, 2):
            c = f"ch_price_ida{n}_eur_mwh"
            nn = ida[c].dropna()
            if len(nn):
                hrs = nn.index.tz_convert(K.TZ).hour.value_counts().sort_index()
                log(f"IDA{n}: first {nn.index.min().tz_convert(K.TZ)}, last {nn.index.max().tz_convert(K.TZ)}, "
                    f"delivery hours covered (local start hour: QH count) {hrs.to_dict()}")
                th = ida[f"flag_ch_ida{n}_thin"]
                yr = ida.index.tz_convert(K.TZ).year
                log(f"IDA{n}: share of priced QH with volume < {IDA_MIN_MW} MW per year: "
                    f"{th[ida[c].notna()].groupby(yr[ida[c].notna()]).mean().round(3).to_dict()}")
                ext = ida[c].notna() & ((ida[c] <= -400) | (ida[c] >= 1000))
                log(f"IDA{n}: extreme prices (<= -400 or >= 1000) {int(ext.sum())} QH, of which thin "
                    f"{int((ext & th).sum())}")
        if a.write:
            written.append(K.write_clean(ida, DOMAIN, "ida", dd))

    if "results" in todo:
        head("Intraday continuous: end-of-day results (EX POST)")
        res, dd, long = build_idc_results()
        log("contracts per year and length (min):\n"
            + long.groupby([long["start_utc"].dt.tz_convert(K.TZ).dt.year, "dur_min"]).size().unstack().to_string())
        log("share of contracts with trades (vol > 0) per year and length:\n"
            + long.assign(tr=long["vol_mw"] > 0).groupby([long["start_utc"].dt.tz_convert(K.TZ).dt.year, "dur_min"])["tr"]
            .mean().unstack().round(3).to_string())
        log("non-NaN share per year (h / qh VWAP):\n" + yearly(res, ["ch_idc_h_vwap_eur_mwh", "ch_idc_qh_vwap_eur_mwh"]).to_string())
        log("mean traded MW per year:\n" + yearly(res, ["ch_idc_h_vol_mw", "ch_idc_qh_vol_mw"], "mean").to_string())
        if a.write:
            written.append(K.write_clean(res, DOMAIN, "idc_results", dd))

    if "index" in todo:
        head("Intraday continuous: EPEX indices (IDFULL / ID3 / ID1)")
        idx, dd = build_idc_index()
        pc = [c for c in idx.columns if c.endswith("_eur_mwh")]
        log("non-NaN share per year:\n" + yearly(idx, pc).to_string())
        fl = [c for c in idx.columns if c.startswith("flag_")]
        yr = idx.index.tz_convert(K.TZ).year
        log("share of QH where the index had no trades (fallback):\n"
            + idx[fl].where(idx[[c.replace("flag_", "").replace("_no_trades", "_eur_mwh") for c in fl]].notna().to_numpy())
            .groupby(yr).mean().round(3).to_string())
        if a.write:
            written.append(K.write_clean(idx, DOMAIN, "idc_index", dd))

    if "trades" in todo:
        head("Intraday continuous: trades (2019-2025)")
        tr, info = build_trades()
        log(f"rows read: old format {info['old_raw']:,}, new format {info['new_raw']:,} (incl. BUY+SELL duplicates); "
            f"kept after de-duplication / new-wins-per-day: {info['kept']:,}")
        y = tr["delivery_start_utc"].dt.tz_convert(K.TZ).dt.year
        log("trades per year x contract length (min):\n" + tr.groupby([y, "dur_min"]).size().unstack(fill_value=0).to_string())
        log("traded energy per year [GWh]:\n"
            + (tr["volume_mw"] * tr["dur_min"] / 60).groupby(y).sum().div(1000).round(1).to_string())
        lead = (tr["delivery_start_utc"] - tr["exec_utc"]).dt.total_seconds() / 3600
        log("lead time (delivery start - execution) [h], quantiles per year:\n"
            + lead.groupby(y).quantile([0.01, 0.1, 0.5, 0.9]).unstack().round(2).to_string())
        lmin = lead.groupby([y, tr["delivery_start_utc"].dt.month]).min().unstack().round(2)
        log("minimum lead time per year x month [h] (should be ~0.5 h until 2021 / >= 0.08 h later; "
            "a jump of 1-2 h would mean a time-zone error):\n" + lmin.to_string())
        log("share executed >= 1 h before delivery (= known at the RQ3 nowcast origin) per year:\n"
            + (lead >= 1).groupby(y).mean().round(3).to_string())
        log(f"blocks {int(tr['is_block'].sum()):,}, self-trades {int(tr['self_trade'].sum()):,}, "
            f"cross-border {int(tr['cross_border'].sum()):,}, format per year:\n"
            + tr.groupby([y, "src_format"], observed=True).size().unstack(fill_value=0).to_string())
        bad = (lead < 0).sum()
        log(f"checks: executions after delivery start {int(bad)}, duplicate trade_id {int(tr['trade_id'].duplicated().sum())}, "
            f"NaN price {int(tr['price_eur_mwh'].isna().sum())}, NaN volume {int(tr['volume_mw'].isna().sum())}")
        if a.write:
            dd = [dict(r, source=SRC, source_series="Intraday Continuous transactions CH") for r in TRADE_DICT]
            written.append(K.write_long(tr, DOMAIN, "idc_trades", dd))

    head("Licence check")
    log(f"all tables cut at delivery < {LICENCE_END.tz_convert(K.TZ):%Y-%m-%d} local (2026 not licensed)")
    if a.write:
        for p in written:
            log(f"written: {p.relative_to(C.ROOT)}")
        (C.DATA_DIR / DOMAIN).mkdir(parents=True, exist_ok=True)
        (C.DATA_DIR / DOMAIN / "clean_epex_report.txt").write_text("\n".join(_report))
        log(f"report: Clean/Data/{DOMAIN}/clean_epex_report.txt")
    else:
        log("\n(dry run: nothing written; add --write)")


if __name__ == "__main__":
    main_()
