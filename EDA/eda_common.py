"""Shared helpers for the EDA scripts (Phase 3, started 3 Oct 2026).

Every EDA script reads the clean layer / master / ex-post views and writes only to
EDA/Output/<step>/ (figures as PNG, tables as CSV, a short report .txt). Nothing in
Clean/Data is changed.

    python EDA/eda_1_quality.py               # -> EDA/Output/1_quality/
    python EDA/eda_1_quality.py --out /tmp/x  # write somewhere else (test runs)

Only pandas / numpy / matplotlib (all in .venv).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

EDA_DIR = Path(__file__).resolve().parent
ROOT = EDA_DIR.parent
CLEAN = ROOT / "Clean"
sys.path.insert(0, str(CLEAN))
import config as C  # noqa: E402
import views as V   # noqa: E402

MASTER_DIR = C.MASTER_DIR
VIEWS_DIR = V.VIEWS_DIR
TZ = C.TZ_LABEL

# ── market-design breaks (common.REGIME_DATES) and other dated events ───────────
# kind 'market' = the auction / price formation changed; 'reporting' = only the
# way a driver is published changed (kept apart, todo 6.3).
BREAKS = pd.DataFrame([
    ("afrr_split",    "2018-06-11", "market",    "aFRR symmetric -> up/down"),
    ("de_zone_split", "2018-10-01", "market",    "DE-AT-LU -> DE-LU + AT bidding zones"),
    ("fcr_daily",     "2019-07-01", "market",    "FCR weekly pay-as-bid -> daily marginal"),
    ("fcr_4h",        "2020-07-01", "market",    "FCR daily -> 4h blocks"),
    ("imb_qh",        "2022-06-01", "market",    "CH imbalance price per quarter-hour"),
    ("afrr_fallback", "2024-02-09", "market",    "CH aFRR platform fall-back (first full day)"),
    ("mfrr_merged",   "2025-09-29", "market",    "TRL+/TRL- merged (mFRR)"),
    ("afrr_daily",    "2025-09-30", "market",    "daily 4h aFRR auctions added"),
    ("imb_single",    "2026-01-01", "market",    "single imbalance price"),
], columns=["key", "date", "kind", "label"])
BREAKS["ts"] = pd.to_datetime(BREAKS["date"]).dt.tz_localize(TZ)

# ── target series (one per market x product x direction x procurement) ──────────
PRICE = "price_settle_ch"            # price paid in CH (= VWAP for pay-as-bid)
PRICE_COLS = ["price_settle_ch", "price_bid_vwap", "price_bid_max", "price_bid_min"]
UNIT = {"FCR": "EUR/MW/h", "aFRR": "CHF/MW/h", "mFRR": "CHF/MW/h"}

# ── plot style: reference palette (dataviz skill), light surface, recessive grid ─
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9cc3ef", "#2a78d6", "#0d3a73"])
DIV_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ee", "#e34948"])
DIR_COLOR = {"up": SERIES_COLORS[1], "down": SERIES_COLORS[0], "sym": SERIES_COLORS[2]}
# one fixed style per target series in every figure: colour = direction x procurement,
# line style = product (FCR dotted, aFRR dashed, mFRR solid)
_COL = {("up", "week"): SERIES_COLORS[1], ("down", "week"): SERIES_COLORS[0],
        ("up", "4h"): SERIES_COLORS[3], ("down", "4h"): SERIES_COLORS[6],
        ("sym", "week"): SERIES_COLORS[2], ("sym", "day"): SERIES_COLORS[5], ("sym", "4h"): SERIES_COLORS[4],
        ("up+down", "week"): SERIES_COLORS[7]}
_LS = {"FCR": ":", "aFRR": "--", "mFRR": "-"}


def style(series: str) -> dict:
    prod, direction, proc = series.split()
    return {"color": _COL[(direction, proc)], "ls": _LS[prod]}
BREAK_COLOR = {"market": "#e34948", "reporting": MUTED}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.fontsize": 8,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES_COLORS),
})


# ── CLI / output ─────────────────────────────────────────────────────────────────
class Out:
    """Output folder + report collector for one EDA step."""

    def __init__(self, step: str, description: str):
        ap = argparse.ArgumentParser(description=description)
        ap.add_argument("--out", type=Path, default=None, help="output folder (default EDA/Output/<step>)")
        self.args, _ = ap.parse_known_args()
        self.dir = (self.args.out or EDA_DIR / "Output") / step
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step = step
        self.lines: list[str] = [f"EDA {step}  -  run {pd.Timestamp.now():%Y-%m-%d %H:%M}", ""]

    def p(self, *text: str) -> None:
        line = " ".join(str(t) for t in text)
        print(line)
        self.lines.append(line)

    def h(self, title: str) -> None:
        self.p("")
        self.p(title)
        self.p("-" * len(title))

    def table(self, df: pd.DataFrame, name: str, show: int | None = 15, index: bool = True) -> None:
        df.to_csv(self.dir / f"{name}.csv", index=index)
        if show:
            self.p(df.head(show).to_string(index=index))
            if len(df) > show:
                self.p(f"... ({len(df)} rows, full table in {name}.csv)")

    def fig(self, fig, name: str) -> None:
        fig.savefig(self.dir / f"{name}.png")
        plt.close(fig)
        self.p(f"[figure] {name}.png")

    def close(self) -> None:
        f = self.dir / f"report_{self.step}.txt"
        f.write_text("\n".join(self.lines) + "\n")
        print(f"\nwritten: {self.dir}")


# ── loading ──────────────────────────────────────────────────────────────────────
def view_dictionary() -> pd.DataFrame:
    return pd.read_csv(VIEWS_DIR / "_dictionary.csv")


def master_dictionary() -> pd.DataFrame:
    return pd.read_csv(MASTER_DIR / "data_dictionary.csv")


def load_expost(view: str, rq: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Ex-post view, CH Swissgrid blocks, partial blocks dropped, + 'series' and local time."""
    df = V.load(view, kind="expost", rq=rq, columns=columns)
    df = df[df["market"] == "ch_swissgrid"].copy()
    df["series"] = series_name(df)
    df["t"] = df["block_start_utc"].dt.tz_convert(TZ)
    return df.sort_values(["series", "t"]).reset_index(drop=True)


def load_targets(rq: str | None = None) -> pd.DataFrame:
    """Target columns of all three views stacked: one row per CH block."""
    keep = ["market", "product", "direction", "procurement", "block_start_utc", "block_end_utc",
            "duration_h", "partial_in_sample", "currency", "n_tenders", "n_bids", "n_accepted",
            "offered_mw", "awarded_mw", "awarded_ch_mw", "cost", *PRICE_COLS, "price_settle_max"]
    return pd.concat([load_expost(v, rq, keep) for v in ("fcr", "afrr", "mfrr")], ignore_index=True)


def series_name(df: pd.DataFrame) -> pd.Series:
    return df["product"] + " " + df["direction"] + " " + df["procurement"]


def load_master(columns: list[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(["ts_utc", *columns]))
    df = pd.read_parquet(MASTER_DIR / "master_15min.parquet", columns=cols)
    df["t"] = pd.to_datetime(df["ts_utc"], utc=True).dt.tz_convert(TZ)
    return df


# ── plotting helpers ─────────────────────────────────────────────────────────────
def mark_breaks(ax, keys=None, kinds=("market",), label=True, xlim=None) -> None:
    """Vertical lines for the breaks (only those inside the current x range)."""
    b = BREAKS[BREAKS["kind"].isin(kinds)]
    if keys is not None:
        b = b[b["key"].isin(keys)]
    lo, hi = ax.get_xlim() if xlim is None else xlim
    b = b[[lo <= mdates.date2num(t) <= hi for t in b["ts"]]].sort_values("ts")
    last = None
    for _, r in b.iterrows():
        ax.axvline(r["ts"], color=BREAK_COLOR[r["kind"]], lw=0.8, ls="--", zorder=1)
        if not label:
            continue
        # breaks closer than ~3 % of the axis share one label
        if last is not None and (mdates.date2num(r["ts"]) - mdates.date2num(last[0])) < 0.03 * (hi - lo):
            last[1].set_text(last[1].get_text() + " / " + r["key"])
            continue
        txt = ax.text(r["ts"], 1.0, " " + r["key"], transform=ax.get_xaxis_transform(), rotation=90,
                      va="top", ha="right", fontsize=6.5, color=INK2)
        last = (r["ts"], txt)


def date_axis(ax, years: bool = True) -> None:
    if years:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def spearman(a: pd.Series, b: pd.Series, min_n: int = 30) -> tuple[float, int]:
    """Rank correlation without scipy; NaN pairs dropped."""
    ok = a.notna() & b.notna()
    n = int(ok.sum())
    if n < min_n or a[ok].nunique() < 3 or b[ok].nunique() < 3:
        return np.nan, n
    return float(a[ok].rank().corr(b[ok].rank())), n


def bootstrap_median_diff(x: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 0):
    """median(y) - median(x) with a 95 % percentile bootstrap interval (iid; for blocks
    with autocorrelation this is optimistic, so read it as descriptive)."""
    rng = np.random.default_rng(seed)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 5 or len(y) < 5:
        return np.nan, np.nan, np.nan
    d = np.median(rng.choice(y, (n, len(y))), axis=1) - np.median(rng.choice(x, (n, len(x))), axis=1)
    return float(np.median(y) - np.median(x)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
