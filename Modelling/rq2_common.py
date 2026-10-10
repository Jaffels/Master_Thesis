"""Shared helpers for RQ2 (todo 9.2): series definition, targets, naive baselines, rolling-origin
fold plan, error measures and the Diebold-Mariano test. numpy / pandas only (no scipy).

Design (9.2 round, 10 Oct 2026; thesis 3.8):
  * Unit of analysis = series = product x direction x procurement (4h / day / week blocks).
  * Targets: FCR price_settle_ch, aFRR price_bid_vwap (VWAP, EDA 5), mFRR price_settle_ch
    (procured blocks only, views.load default). Models are trained on asinh(price) (like log, but
    defined for the ~2 % zero / negative mFRR-up prices); errors are reported in EUR/MWh.
  * Test period from 1 Jan 2021 (RQ2 split). FCR (4h from Jul 2020) and aFRR (4h from Sep 2025,
    weekly up/down from 2018) have little history before 2021, so evaluation is ROLLING-ORIGIN with
    an expanding window: every `refit_months` the model is refit on all blocks delivered before
    the fold start and predicts the next fold. Folds with too little training data (MIN_TRAIN)
    are skipped.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Clean"))
import views as V  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

VIEWS = ["fcr", "afrr", "mfrr"]
TARGET = {"FCR": "price_settle_ch", "aFRR": "price_bid_vwap", "mFRR": "price_settle_ch"}
TZ = V.TZ
TEST_START = pd.Timestamp("2021-01-01", tz=TZ)
MIN_TRAIN = {"4h": 500, "day": 200, "week": 100}
BASELINES = ["b_prev_slot", "b_prev_auction", "b_roll_med"]
BASE_LABEL = {"b_prev_slot": "same slot, previous day / week", "b_prev_auction": "previous auction",
              "b_roll_med": "median of last 7 days (4 weeks for weekly)"}


def load_view(view: str, impute: bool = False, weather_fc: bool = False) -> pd.DataFrame:
    """Ex-ante view, RQ2 window, procured blocks, sorted by series and time, with columns
    series, slot, y (target) and the naive baselines b_*."""
    df = V.load(view, "exante", rq="RQ2", impute=impute, weather_fc=weather_fc).copy()
    df["series"] = (df["product"].astype(str) + "|" + df["direction"].astype(str) + "|"
                    + df["procurement"].astype(str))
    df = df.sort_values(["series", "block_start_utc"]).reset_index(drop=True)
    t = TARGET[str(df["product"].iloc[0])]
    df["y"] = df[t]
    loc = df["block_start_utc"].dt.tz_convert(TZ)
    df["slot"] = np.where(df["procurement"].astype(str) == "4h", loc.dt.hour, 0)
    df["b_prev_slot"] = df[f"tgt_prev_slot__{t}"]
    df["b_prev_auction"] = df[f"tgt_prev_auction__{t}"]
    roll = pd.Series(np.nan, index=df.index)
    for (s, _), g in df.groupby(["series", "slot"], sort=False):
        win = 4 if s.endswith("|week") else 7
        roll.loc[g.index] = g[f"tgt_prev_slot__{t}"].rolling(win, min_periods=max(2, win // 2)).median()
    df["b_roll_med"] = roll
    return df


def fold_plan(g: pd.DataFrame, refit_months: int, min_train: int):
    """Rolling-origin folds for one series (g sorted by time). Yields (fold_id, fold_start,
    train_pos, test_pos): integer positions into g. Training = blocks whose delivery ended before
    the fold start (conservative); test = blocks starting in the fold. Folds with fewer than
    min_train training blocks are skipped."""
    loc = g["block_start_utc"].dt.tz_convert(TZ)
    m = (loc.dt.year * 12 + loc.dt.month - 1).to_numpy()
    m0 = TEST_START.year * 12 + TEST_START.month - 1
    in_test = (g["block_start_utc"] >= TEST_START.tz_convert("UTC")).to_numpy()
    fold = np.where(in_test, (m - m0) // refit_months, -1)
    end = g["block_end_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    for f in sorted(set(fold[fold >= 0])):
        mm = m0 + int(f) * refit_months
        start = pd.Timestamp(year=mm // 12, month=mm % 12 + 1, day=1, tz=TZ).tz_convert("UTC")
        train = np.flatnonzero(end <= np.datetime64(start.tz_localize(None)))
        if len(train) < min_train:
            continue
        yield int(f), start, train, np.flatnonzero(fold == f)


# ---------------------------------------------------------------- error measures
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    den = (np.abs(y) + np.abs(p)) / 2
    smape = np.where(den > 0, np.abs(e) / np.where(den > 0, den, 1), 0.0).mean() * 100
    return {"n": len(y), "MAE": np.abs(e).mean(), "RMSE": math.sqrt((e ** 2).mean()),
            "sMAPE_%": smape, "MAE_asinh": np.abs(np.arcsinh(p) - np.arcsinh(y)).mean()}


def dm_test(y, p1, p2, lags: int) -> tuple[float, float]:
    """Diebold-Mariano test, absolute-error loss, one-step. d = |e1| - |e2|; H0: equal accuracy.
    Newey-West variance with `lags`. Returns (statistic, two-sided normal p). Statistic < 0:
    model 1 is more accurate."""
    y, p1, p2 = (np.asarray(v, float) for v in (y, p1, p2))
    d = np.abs(p1 - y) - np.abs(p2 - y)
    n = len(d)
    if n < 30:
        return float("nan"), float("nan")
    d0 = d - d.mean()
    v = d0 @ d0 / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (d0[l:] @ d0[:-l]) / n
    if v <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / math.sqrt(v / n)
    return stat, math.erfc(abs(stat) / math.sqrt(2))


def fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.3f}")
