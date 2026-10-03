"""Shared helpers for the clean layer (to-do 3.1, 2 Oct 2026).

Every Clean/clean_<domain>.py script uses these, so all domains follow the same
rules (table design, Sections 2-4):

  grid()               the complete 15-min master grid (409,052 rows), UTC + local
  read_entsoe()        one ENTSO-E series from production_pre2021/ + production/
  to_utc()             any timestamp index -> UTC (ns), naive = Europe/Zurich
  trim()               cut to [SAMPLE_START, CUTOFF]
  hourly_to_qh()       hourly values onto the 15-min grid (repeat, or split for energy)
  blocks_to_qh()       block values (week/day/4h) as step functions on the 15-min grid
  check_names()        enforce the column-name grammar {area}_{variable}[_{qual}]_{unit}
  write_clean()        save one clean table + its data-dictionary rows

Rules carried by the helpers
- Missing stays NaN. Nothing is forward-filled across a real gap: hourly values
  only fill their own hour, block values only their own block.
- Clean tables are stored as parquet with a `ts_utc` column (interval start,
  UTC) and float32 values; flag and regime columns keep their own dtype.
- Raw source files are never modified.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

TZ = C.TZ_LABEL
QH = pd.Timedelta(C.FREQ)

ENTSOE_TREES = ("production_pre2021", "production")

# ---------------------------------------------------------------- names
AREAS = ("ch", "de_lu", "de", "de_at_lu", "at", "fr", "it_nord", "it",
         "de_amprion", "de_transnet", "ce")
UNITS = ("mw", "mwh", "eur_mwh", "chf_mwh", "eur_mw_h", "chf_mw_h", "eur", "chf",
         "mhz", "hz", "degc", "wm2", "mm", "cm", "share", "n", "h", "kwh",
         "pct", "ms", "kh")   # kh = degree-hours (K*h); ms = m/s
_AREA_RE = "|".join(sorted(AREAS, key=len, reverse=True))
_UNIT_RE = "|".join(sorted(UNITS, key=len, reverse=True))
NAME_RE = re.compile(
    rf"^(?:(?:{_AREA_RE})_(?:(?:{_AREA_RE})_)?[a-z0-9_]+_(?:{_UNIT_RE})(?:_xchk_[a-z0-9]+)?"
    rf"|flag_[a-z0-9_]+|regime_[a-z0-9_]+|src_[a-z0-9_]+|eda_[a-z0-9_]+"
    rf"|ch_afrr_platform_fallback)$")


def check_names(cols) -> list[str]:
    """Return the column names that break the grammar (empty list = all fine)."""
    return [c for c in cols if c not in ("ts_utc", "ts_local") and not NAME_RE.match(c)]


# ---------------------------------------------------------------- time
def to_utc(idx) -> pd.DatetimeIndex:
    """Any timestamps -> tz-aware UTC DatetimeIndex in ns. Naive = Europe/Zurich.
    Text with mixed UTC offsets (e.g. '+01:00' / '+02:00' from CSV) is parsed as such."""
    try:
        idx = pd.DatetimeIndex(idx)
    except (TypeError, ValueError):
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(idx), utc=True))
    if idx.tz is None:
        idx = idx.tz_localize(TZ, ambiguous="infer", nonexistent="raise")
    return idx.tz_convert("UTC").as_unit("ns")


def grid() -> pd.DataFrame:
    """The complete 15-min master grid: ts_utc (index) and ts_local."""
    idx = pd.date_range(C.start_utc(), C.end_utc(), freq=C.FREQ, inclusive="left",
                        name="ts_utc").as_unit("ns")
    return pd.DataFrame({"ts_local": idx.tz_convert(TZ)}, index=idx)


def trim(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with index in [SAMPLE_START, CUTOFF] (index must be UTC)."""
    return df[(df.index >= C.start_utc()) & (df.index < C.end_utc())]


def native_step(idx: pd.DatetimeIndex) -> pd.Timedelta:
    """Median step of a sorted index (the series' native resolution)."""
    if len(idx) < 3:
        return QH
    return pd.Timedelta(np.median(np.diff(idx.asi8)), unit="ns")


# ---------------------------------------------------------------- reading
def read_entsoe(domain: str, dataset: str, variant: str, area: str,
                columns: list[str] | None = None, dedup: bool = True) -> pd.DataFrame:
    """One ENTSO-E series from both trees, UTC index, sorted, de-duplicated.

    domain   Load | Generation | Transmission | Balancing | Outages
    area     folder name, e.g. CH, DE_LU, CH-DE
    Later files win on duplicate timestamps (production over production_pre2021).
    dedup=False keeps repeated timestamps (long layouts, e.g. one row per production
    type); the caller then de-duplicates on its own keys.
    """
    parts = []
    for tree in ENTSOE_TREES:
        folder = C.ROOT / "Entsoe" / domain / "Data" / tree / dataset / variant / area
        for f in sorted(folder.glob("[0-9]*.parquet")):
            df = pd.read_parquet(f, columns=columns)
            df.index = to_utc(df.index)
            parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no ENTSO-E files for {domain}/{dataset}/{variant}/{area}")
    out = pd.concat(parts).sort_index(kind="stable")
    if dedup:
        out = out[~out.index.duplicated(keep="last")]
    out.index.name = "ts_utc"
    return out


# ---------------------------------------------------------------- onto the grid
def hourly_to_qh(s: pd.Series | pd.DataFrame, how: str = "repeat") -> pd.Series | pd.DataFrame:
    """Hourly (or coarser-than-15-min, sub-hourly-mixed) values onto the 15-min grid.

    how = "repeat"  power / price / level: each value fills the quarter-hours of its
                    own interval (until the next timestamp, max. 1 h).
        = "split"   energy per hour (MWh): value / 4 per quarter-hour.
    Quarter-hours already present (15-min parts of a mixed series) are kept as they
    are. Values never spread beyond their own hour, so gaps stay NaN.
    """
    if how not in ("repeat", "split"):
        raise ValueError(how)
    g = grid().index
    x = s.sort_index()
    x = x[~x.index.duplicated(keep="last")]
    # length of each source interval = time to next timestamp, capped at 1 h
    nxt = x.index.to_series().shift(-1)
    length = (nxt - x.index.to_series()).fillna(pd.Timedelta("1h")).clip(upper=pd.Timedelta("1h"))
    n_qh = (length / QH).round().astype(int).clip(lower=1)
    if how == "split":
        x = x.div(n_qh, axis=0)
    out = x.reindex(g)
    # forward-fill each value at most (n_qh - 1) quarter-hours: only within its interval
    pos = np.searchsorted(x.index.asi8, g.asi8, side="right") - 1
    valid = pos >= 0
    src_ts = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")
    src_ts[valid] = x.index.asi8[pos[valid]].astype("datetime64[ns]")
    src_n = np.zeros(len(g), dtype=int)
    src_n[valid] = n_qh.to_numpy()[pos[valid]]
    age = (g.asi8 - src_ts.astype("int64")) // QH.value
    inside = valid & (age >= 0) & (age < src_n)
    filled = x.iloc[np.where(valid, pos, 0)]
    filled.index = g
    if isinstance(out, pd.Series):
        out = out.where(out.notna() | ~inside, filled)
    else:
        out = out.where(out.notna() | ~pd.Series(inside, index=g).to_numpy()[:, None], filled)
    return out


def blocks_to_qh(blocks: pd.DataFrame, value_cols: list[str],
                 start_col: str = "block_start_utc", end_col: str = "block_end_utc") -> pd.DataFrame:
    """Block products (week / day / 4h) as step functions on the 15-min grid.

    Each quarter-hour gets the value of the block that contains it, i.e.
    block_start <= ts < block_end (DST-correct, works for 3 h / 5 h blocks and
    167 h / 169 h weeks). Quarter-hours not covered by any block stay NaN.
    Overlapping blocks of the same line are an error: split them by product first.
    """
    g = grid().index
    b = blocks.sort_values(start_col)
    st, en = to_utc(b[start_col]), to_utc(b[end_col])
    if (st[1:] < en[:-1]).any():
        raise ValueError("overlapping blocks: split by product / direction / procurement first")
    pos = np.searchsorted(st.asi8, g.asi8, side="right") - 1
    ok = (pos >= 0) & (g.asi8 < en.asi8[np.clip(pos, 0, None)])
    out = pd.DataFrame(index=g, columns=value_cols, dtype="float64")
    vals = b[value_cols].to_numpy(dtype="float64")
    out.loc[ok, value_cols] = vals[pos[ok]]
    return out


# ---------------------------------------------------------------- writing
def write_clean(df: pd.DataFrame, domain: str, name: str, dictionary: list[dict]) -> Path:
    """Save one clean table and its data-dictionary rows.

    df          index = ts_utc (UTC), columns follow the name grammar
    dictionary  one dict per column with keys: column, source, source_series,
                unit, resolution_native, aggregation_rule, availability_rule, notes
    Writes Clean/Data/<domain>/<name>.parquet and <name>_dictionary.csv.
    """
    bad = check_names(df.columns)
    if bad:
        raise ValueError(f"column names break the grammar: {bad}")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("index must be tz-aware UTC")
    if df.index.duplicated().any():
        raise ValueError("duplicate timestamps")
    out = trim(df).copy()
    num = [c for c in out.columns if pd.api.types.is_float_dtype(out[c])]
    out[num] = out[num].astype(C.FLOAT_DTYPE)
    folder = C.DATA_DIR / domain
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.parquet"
    out.reset_index().rename(columns={"index": "ts_utc"}).to_parquet(path, index=False)
    d = pd.DataFrame(dictionary)
    if len(d):
        missing = set(out.columns) - set(d["column"])
        if missing:
            raise ValueError(f"no dictionary entry for: {sorted(missing)}")
        firsts, lasts = {}, {}
        for c in d["column"]:
            nn = out[c].dropna().index
            firsts[c] = nn.min() if len(nn) else pd.NaT
            lasts[c] = nn.max() if len(nn) else pd.NaT
        d["start_utc"] = d["column"].map(firsts)
        d["end_utc"] = d["column"].map(lasts)
        d["nan_share"] = d["column"].map(lambda c: round(float(out[c].isna().mean()), 5))
        d.to_csv(folder / f"{name}_dictionary.csv", index=False)
    return path


# ---------------------------------------------------------------- regimes
# Break dates (local, first day of the new regime). Sources: regime file A/B/C4.
REGIME_DATES = {
    "afrr_split": "2018-06-11",      # A1 aFRR symmetric -> up/down (delivery ISO week 24)
    "de_zone_split": "2018-10-01",   # B1 DE-AT-LU -> DE-LU + AT
    "fcr_daily": "2019-07-01",       # A2 FCR weekly pay-as-bid -> daily marginal
    "fcr_4h": "2020-07-01",          # A3 FCR daily -> 4h blocks
    "imb_hourly": "2018-06-11",      # C4b CH imbalance price mostly constant within the hour (EDA 3c, 3 Oct 2026)
    "imb_qh": "2022-06-01",          # C4 CH imbalance price per quarter-hour
    "mfrr_merged": "2025-09-29",     # A7 TRL+/TRL- -> TRL (first delivery of merged series)
    "afrr_daily": "2025-09-30",      # A6 daily 4h aFRR auctions added
    "imb_single": "2026-01-01",      # A8 dual -> single imbalance price
}


def _after(idx_utc: pd.DatetimeIndex, key: str) -> np.ndarray:
    t = pd.Timestamp(REGIME_DATES[key], tz=TZ).tz_convert("UTC")
    return np.asarray(idx_utc >= t)


def regimes(idx_utc) -> pd.DataFrame:
    """The regime columns of table design 4.5 for any UTC timestamps.
    (ch_afrr_platform_fallback comes from the outage documents, not from dates.)"""
    idx = to_utc(idx_utc)
    out = pd.DataFrame(index=idx)
    out["regime_afrr_dir"] = np.where(_after(idx, "afrr_split"), "split", "sym")
    out["regime_fcr"] = np.select([_after(idx, "fcr_4h"), _after(idx, "fcr_daily")],
                                  ["4h_marginal", "daily_marginal"], "weekly_pab")
    out["regime_afrr_daily"] = _after(idx, "afrr_daily").astype("int8")
    out["regime_mfrr_merged"] = _after(idx, "mfrr_merged").astype("int8")
    out["regime_de_zone"] = np.where(_after(idx, "de_zone_split"), "post_split", "pre_split")
    # mixed (~45 % of hours with 4 equal QH prices) -> hourly_dominant (~78 %) -> qh (~3 %)
    out["regime_imb_resolution"] = np.select([_after(idx, "imb_qh"), _after(idx, "imb_hourly")],
                                             ["qh", "hourly_dominant"], "mixed")
    out["regime_imb_pricing"] = np.where(_after(idx, "imb_single"), "single", "dual")
    for c in ("regime_afrr_dir", "regime_fcr", "regime_de_zone", "regime_imb_resolution", "regime_imb_pricing"):
        out[c] = out[c].astype("category")
    return out


def write_long(df: pd.DataFrame, domain: str, name: str, dictionary: list[dict]) -> Path:
    """Save a long table (one row per block / event / bid, not on the 15-min grid).
    No name grammar here (keys like product, direction), but every column needs a
    dictionary row. Writes Clean/Data/<domain>/<name>.parquet + _dictionary.csv."""
    d = pd.DataFrame(dictionary)
    missing = set(df.columns) - set(d.get("column", []))
    if missing:
        raise ValueError(f"no dictionary entry for: {sorted(missing)}")
    folder = C.DATA_DIR / domain
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.parquet"
    df.to_parquet(path, index=False)
    d["nan_share"] = d["column"].map(lambda c: round(float(df[c].isna().mean()), 5) if c in df else np.nan)
    d.to_csv(folder / f"{name}_dictionary.csv", index=False)
    return path


def local_steps(df: pd.DataFrame, days: int, cols: list[str]) -> pd.DataFrame:
    """Values stamped at a local day / week start -> step function over `days` local
    days on the 15-min grid. Stamps are rounded to the nearest local midnight (some
    ENTSO-E stamps sit at 23:00 / 01:00 because of DST offsets). A later block cuts
    an earlier overlapping one short."""
    loc = to_utc(df.index).tz_convert(TZ)
    start_local = (loc.tz_localize(None) + pd.Timedelta(hours=12)).normalize()
    end_local = start_local + pd.Timedelta(days=days)
    b = df[cols].copy().reset_index(drop=True)
    b["block_start_utc"] = start_local.tz_localize(TZ, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
    b["block_end_utc"] = pd.DatetimeIndex(end_local).tz_localize(TZ, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
    b = b.sort_values("block_start_utc").drop_duplicates("block_start_utc", keep="last").reset_index(drop=True)
    st, en = b["block_start_utc"].to_numpy(), b["block_end_utc"].to_numpy()
    over = st[1:] < en[:-1]
    if over.any():
        en[:-1][over] = st[1:][over]
        b["block_end_utc"] = en
    return blocks_to_qh(b, cols)
