"""Auction gate closure (forecast origin t0) per Swissgrid capacity block (to-do 5.3).

The ex-ante views use only information published at or before the gate closure of the
auction that sets the block's price. Swissgrid has no gate-closure column in its
result files, so the times come from rules:

  product  procurement  rule                                  confidence
  FCR      week         Tuesday 13:00 of the week before      assumed (same as SRL/TRL weekly)
  FCR      day          D-2 15:00 (FCR cooperation daily)     assumed
  FCR      4h           D-1 08:00                             calendar (SDL Ausschreibungskalender 2025, 2026)
  aFRR/mFRR week        Tuesday 13:00 of the week before      calendar from 2025, assumed before
  aFRR/mFRR 4h          closing table by delivery weekday     calendar from 2025, assumed before
                        (DAILY_FRR below: Mon<-Fri 14:30, Tue<-Fri 15:30,
                        Wed<-Mon 14:30, Thu<-Tue 14:30, Fri<-Wed 14:30,
                        Sat<-Thu 14:30, Sun<-Thu 15:30)

All times are Europe/Zurich wall-clock times. Holiday shifts in the calendar (e.g.
KW1 2025 closed Mon 23 Dec instead of Tue 24 Dec) are NOT modelled: t0 can be up to
one day too late around Christmas / New Year. Refine DAILY_FRR and the 'assumed'
periods once the SDL calendars for 2015-2024 are checked.

    from gate_closure import gate_closure
    gc = gate_closure(blocks)   # DataFrame: gate_closure_utc, gc_rule, gc_confidence
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"
CALENDAR_FROM = pd.Timestamp("2025-01-01")   # first year with a checked SDL calendar (local date)

# delivery weekday (Mon=0) -> (days before delivery day, closing time)
DAILY_FRR = {0: (3, "14:30"), 1: (4, "15:30"), 2: (2, "14:30"), 3: (2, "14:30"),
             4: (2, "14:30"), 5: (2, "14:30"), 6: (3, "15:30")}


def _naive_local(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert(TZ).dt.tz_localize(None)


def _to_utc(naive_local: pd.Series) -> pd.Series:
    # closing times (08:00-15:30) never fall into a DST switch hour
    return naive_local.dt.tz_localize(TZ).dt.tz_convert("UTC")


def gate_closure(blocks: pd.DataFrame) -> pd.DataFrame:
    """blocks needs product, procurement, block_start_utc. Returns one row per input row
    (same index) with gate_closure_utc, gc_rule, gc_confidence."""
    start = _naive_local(blocks["block_start_utc"])
    day = start.dt.normalize()
    prod, proc = blocks["product"], blocks["procurement"]
    gc = pd.Series(pd.NaT, index=blocks.index, dtype="datetime64[ns]")
    rule = pd.Series("", index=blocks.index, dtype=object)

    # weekly: block starts Monday 00:00; closing Tuesday 13:00 of the week before
    wk = proc == "week"
    monday = day - pd.to_timedelta(day.dt.weekday, unit="D")
    gc[wk] = monday[wk] - pd.Timedelta(days=6) + pd.Timedelta(hours=13)
    rule[wk] = "week_tue_1300_w-1"

    m = (prod == "FCR") & (proc == "day")
    gc[m] = day[m] - pd.Timedelta(days=2) + pd.Timedelta(hours=15)
    rule[m] = "fcr_day_d-2_1500"

    m = (prod == "FCR") & (proc == "4h")
    gc[m] = day[m] - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    rule[m] = "fcr_4h_d-1_0800"

    m = prod.isin(["aFRR", "mFRR"]) & (proc == "4h")
    if m.any():
        wd = day[m].dt.weekday
        back = wd.map({k: v[0] for k, v in DAILY_FRR.items()})
        hhmm = wd.map({k: v[1] for k, v in DAILY_FRR.items()})
        gc[m] = day[m] - pd.to_timedelta(back, unit="D") + pd.to_timedelta(hhmm + ":00")
        rule[m] = "frr_daily_table"

    if gc.isna().any():
        bad = blocks.loc[gc.isna(), ["product", "procurement"]].drop_duplicates()
        raise ValueError(f"no gate-closure rule for:\n{bad}")

    conf = np.where(rule == "fcr_4h_d-1_0800", "calendar",
                    np.where(rule.isin(["week_tue_1300_w-1", "frr_daily_table"])
                             & (prod != "FCR") & (day >= CALENDAR_FROM), "calendar", "assumed"))
    out = pd.DataFrame({"gate_closure_utc": _to_utc(gc).astype("datetime64[ns, UTC]"),
                        "gc_rule": rule, "gc_confidence": conf}, index=blocks.index)
    late = out["gate_closure_utc"] >= pd.to_datetime(blocks["block_start_utc"], utc=True)
    if late.any():
        raise AssertionError(f"{int(late.sum())} blocks with gate closure at/after delivery start")
    return out
