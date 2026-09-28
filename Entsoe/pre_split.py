"""
Helpers for pulling across the German bidding-zone split (1 Oct 2018).

Before 1 Oct 2018 Germany, Austria and Luxembourg formed one bidding zone
(DE_AT_LU); from 1 Oct 2018 it is DE_LU (+ AT on its own). A German series
therefore answers on a DIFFERENT area code before and after the split
(e.g. Load/Generation: DE -> DE_LU; Outages / CH-DE NTC: DE_AT_LU -> DE_LU).

The pulls use ONE area code per series and write ONE parquet per calendar year.
With a second coverage CSV (--coverage-pre) a pull instead cuts each year at
the split date and queries each part on its own code; the parts are joined
into the usual single year file. Without --coverage-pre nothing changes.

Added 2026-09-28 for the 2015-2020 extension. Stdlib + pandas only.
"""

from __future__ import annotations

import pandas as pd

DE_LU_SPLIT = "2018-10-01"

# Tokens that only exist on one side of the split.
_PRE_ONLY = {"DE_AT_LU"}


def split_ts(split: str, tz: str) -> pd.Timestamp:
    return pd.Timestamp(split, tz=tz)


def segments(y_start: pd.Timestamp, y_end: pd.Timestamp, split: pd.Timestamp,
             pre_code: str | None, post_code: str | None
             ) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Cut [y_start, y_end) at the split. A part is kept only when a code is
    known for its side (None = series not available on that side)."""
    out: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if pre_code is not None and y_start < split:
        s, e = y_start, min(y_end, split)
        if s < e:
            out.append((s, e, pre_code))
    if post_code is not None and y_end > split:
        s, e = max(y_start, split), y_end
        if s < e:
            out.append((s, e, post_code))
    return out


def pre_split_code(code: str) -> str:
    """Map a post-split German code to its pre-split equivalent, for series
    the probe never resolved before the split (sparse event tasks). Works on
    plain codes ('DE_LU') and border strings ('CH>DE_LU')."""
    parts = code.split(">")
    return ">".join("DE_AT_LU" if p == "DE_LU" else p for p in parts)


def join_parts(frames: list[pd.DataFrame | None], dedupe_rows: bool = False
               ) -> pd.DataFrame | None:
    """Concatenate the per-segment frames of one year (None/empty skipped)."""
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    df = pd.concat(parts)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index(kind="stable")
    if dedupe_rows:
        try:
            df = df[~df.reset_index().duplicated().to_numpy()]
        except TypeError:  # unhashable cells -> leave as is
            pass
    return df


def codes_label(segs: list[tuple[pd.Timestamp, pd.Timestamp, str]]) -> str:
    """'DE|DE_LU' style label of the codes used in a year (manifest column)."""
    seen: list[str] = []
    for _, _, c in segs:
        if c not in seen:
            seen.append(c)
    return "|".join(seen)
