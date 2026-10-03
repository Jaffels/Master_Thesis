"""EDA 2 - Distributions and seasonality of the reserve prices (todo 6.2).

Target price: price_settle_ch (CH price paid per MW and hour of delivery; FCR EUR, aFRR /
mFRR CHF). One series per product x direction x procurement, CH Swissgrid blocks after
combine_tenders, partial blocks dropped (ex-post views).

Outputs (EDA/Output/2_prices/)
  2a  price history per product (log scale, 4h series as weekly means), market breaks marked
  2b  summary statistics per series and per series x year; yearly boxplots
  2c  seasonality of the 4h series: delivery slot, weekday, month - each relative to the
      series' median of the same calendar year (removes the level shifts between years)
  2d  seasonality of the weekly series: month of year, relative to the yearly median
  2e  persistence: autocorrelation of log price (4h: lags in blocks up to 2 weeks;
      weekly: lags up to 12 weeks)

    python EDA/eda_2_prices.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eda_common import (Out, plt, style, load_targets, mark_breaks, date_axis, PRICE, UNIT, DIR_COLOR,
                        SERIES_COLORS, INK2, SEQ_CMAP, DIV_CMAP)

out = Out("2_prices", __doc__)
tg = load_targets()
tg = tg[tg[PRICE].notna()].copy()
tg["year"] = tg["t"].dt.year
tg["slot"] = tg["t"].dt.hour
tg["weekday"] = tg["t"].dt.dayofweek
tg["month"] = tg["t"].dt.month
# blocks of 3 / 5 h (DST days) belong to the 4h series; slot = local start hour
ORDER = ["FCR sym week", "FCR sym day", "FCR sym 4h", "aFRR sym week", "aFRR up week", "aFRR down week",
         "aFRR up 4h", "aFRR down 4h", "mFRR up week", "mFRR down week", "mFRR up 4h", "mFRR down 4h"]
ORDER = [s for s in ORDER if s in set(tg["series"])]
FOUR_H = [s for s in ORDER if s.endswith("4h")]
WEEK = [s for s in ORDER if s.endswith("week")]
FLOOR = 0.01                                     # floor for log scales (prices of 0 exist)


# ── 2a  price history ───────────────────────────────────────────────────────────
out.h("2a  Price history")
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for ax, prod in zip(axes, ["FCR", "aFRR", "mFRR"]):
    for s in [x for x in ORDER if x.startswith(prod + " ")]:
        d = tg[tg["series"] == s].set_index("t")[PRICE]
        # weekly grid, gaps stay NaN so that the line is interrupted (no bridging)
        d = d.resample("W-MON", label="left", closed="left").mean()
        lab = f"{s} (weekly mean)" if (s.endswith("4h") or s.endswith("day")) else s
        ax.plot(d.index, d.clip(lower=FLOOR), label=lab, lw=1.0, **style(s))
    ax.set_yscale("log")
    ax.set_ylabel(f"{prod}  [{UNIT[prod]}]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
    mark_breaks(ax, label=(prod == "FCR"))
date_axis(axes[-1])
axes[0].set_title("Capacity prices paid in CH (price_settle_ch), log scale; dashed = market-design breaks")
out.fig(fig, "2a_price_history")

# ── 2b  summary statistics ──────────────────────────────────────────────────────
out.h("2b  Summary statistics of price_settle_ch")


def stats(x: pd.Series) -> pd.Series:
    x = x.dropna()
    return pd.Series({"n": len(x), "mean": x.mean(), "median": x.median(), "std": x.std(),
                      "cv": x.std() / x.mean() if x.mean() else np.nan,
                      "p05": x.quantile(.05), "p95": x.quantile(.95), "p99": x.quantile(.99), "max": x.max(),
                      "skew": x.skew(), "share_zero": (x <= 0).mean(),
                      "p99_over_median": x.quantile(.99) / x.median() if x.median() else np.nan})


st = tg.groupby("series")[PRICE].apply(stats).unstack().reindex(ORDER)
st.insert(0, "from", tg.groupby("series")["t"].min().dt.strftime("%Y-%m-%d"))
st.insert(1, "to", tg.groupby("series")["t"].max().dt.strftime("%Y-%m-%d"))
out.table(st.round(3), "2b_stats_by_series", show=None)
out.p(st.drop(columns=["std"]).round(2).to_string())
sty = tg.groupby(["series", "year"])[PRICE].apply(stats).unstack()
out.table(sty.round(3), "2b_stats_by_series_year", show=None)
out.p("\nMedian by series x year:")
out.p(sty["median"].unstack().reindex(ORDER).round(2).to_string())
out.p("\nCoefficient of variation by series x year:")
out.p(sty["cv"].unstack().reindex(ORDER).round(2).to_string())

ncol = 4
nrow = int(np.ceil(len(ORDER) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.8 * nrow), squeeze=False)
for ax, s in zip(axes.flat, ORDER):
    d = tg[tg["series"] == s]
    yrs = sorted(d["year"].unique())
    data = [d.loc[d["year"] == y, PRICE].clip(lower=FLOOR).to_numpy() for y in yrs]
    bp = ax.boxplot(data, positions=range(len(yrs)), widths=0.6, whis=(5, 95), showfliers=True,
                    patch_artist=True, flierprops=dict(marker=".", ms=2, alpha=0.3, mec=INK2),
                    medianprops=dict(color="#0b0b0b", lw=1.2))
    for b in bp["boxes"]:
        b.set(facecolor=style(s)["color"], alpha=0.55, lw=0.6)
    ax.set_xticks(range(len(yrs)), [str(y)[2:] for y in yrs], fontsize=7)
    ax.set_yscale("log")
    ax.set_title(s, fontsize=9)
for ax in axes.flat[len(ORDER):]:
    ax.axis("off")
fig.suptitle("price_settle_ch per year (box = IQR, whiskers = 5-95 %, dots beyond; log scale; FCR EUR, others CHF per MW/h)",
             fontsize=10, fontweight="bold")
fig.tight_layout()
out.fig(fig, "2b_boxplots_by_year")

# ── 2c  seasonality of the 4h series ────────────────────────────────────────────
out.h("2c  Seasonality of the 4h series (price / median of the same series and calendar year)")
h = tg[tg["series"].isin(FOUR_H)].copy()
h["rel"] = h[PRICE] / h.groupby(["series", "year"])[PRICE].transform("median")
h = h[np.isfinite(h["rel"])]
prof = {}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, (key, labels, title) in zip(axes, [
        ("slot", None, "delivery slot (local start hour)"),
        ("weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "weekday"),
        ("month", list("JFMAMJJASOND"), "month")]):
    hh = h[h["slot"].isin([0, 4, 8, 12, 16, 20])] if key == "slot" else h
    p = hh.groupby(["series", key])["rel"].median().unstack(0)
    prof[key] = p
    for s in FOUR_H:
        if s in p:
            ax.plot(p.index, p[s], marker="o", ms=4, label=s, **style(s))
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_title(title)
    if labels:
        ax.set_xticks(p.index, labels)
    else:
        ax.set_xticks(p.index, [f"{x:02d}" for x in p.index])
axes[0].set_ylabel("median of price / yearly median")
axes[0].legend(fontsize=7)
fig.suptitle("4h products: relative price profiles (all years)", fontsize=10, fontweight="bold")
fig.tight_layout()
out.fig(fig, "2c_4h_profiles")
for k, p in prof.items():
    out.p(f"\nmedian relative price by {k}:")
    out.table(p.round(2), f"2c_4h_profile_{k}", show=None)
    out.p(p.round(2).to_string())

# slot x month heatmaps (does the daily shape move with the season? solar)
fig, axes = plt.subplots(1, len(FOUR_H), figsize=(3.2 * len(FOUR_H), 3.2), squeeze=False)
for ax, s in zip(axes[0], FOUR_H):
    d = h[(h["series"] == s) & h["slot"].isin([0, 4, 8, 12, 16, 20])]
    p = d.groupby(["slot", "month"])["rel"].median().unstack()
    im = ax.imshow(np.log2(p.to_numpy()), cmap=DIV_CMAP, vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(p.shape[1]), [list("JFMAMJJASOND")[m - 1] for m in p.columns], fontsize=7)
    ax.set_yticks(range(p.shape[0]), [f"{x:02d}h" for x in p.index], fontsize=7)
    ax.set_title(s, fontsize=9)
    ax.grid(False)
cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
cb.set_label("log2(price / yearly median)")
fig.suptitle("4h products: slot x month (red = above the yearly median, blue = below; aFRR 4h: Oct 2025 - Aug 2026 only)",
             fontsize=10, fontweight="bold", y=1.06)
out.fig(fig, "2c_4h_slot_month_heatmap")

# same slot profile per year (stability of the daily shape)
fig, axes = plt.subplots(1, len(FOUR_H), figsize=(3.2 * len(FOUR_H), 3.2), squeeze=False, sharey=True)
for ax, s in zip(axes[0], FOUR_H):
    d = h[(h["series"] == s) & h["slot"].isin([0, 4, 8, 12, 16, 20])]
    p = d.groupby(["slot", "year"])["rel"].median().unstack()
    for i, y in enumerate(p.columns):
        ax.plot(p.index, p[y], color=SEQ_CMAP(0.25 + 0.75 * i / max(1, len(p.columns) - 1)), lw=1, label=str(y))
    ax.set_title(s, fontsize=9)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.axhline(1, color=INK2, lw=0.6)
axes[0][0].set_ylabel("median price / yearly median")
axes[0][-1].legend(fontsize=6, ncol=2)
fig.suptitle("4h products: daily slot profile by year (light = early, dark = recent)", fontsize=10, fontweight="bold")
fig.tight_layout()
out.fig(fig, "2c_4h_slot_profile_by_year")

# ── 2d  weekly series: month of year ────────────────────────────────────────────
out.h("2d  Seasonality of the weekly series (price / yearly median)")
w = tg[tg["series"].isin(WEEK)].copy()
w["rel"] = w[PRICE] / w.groupby(["series", "year"])[PRICE].transform("median")
w = w[np.isfinite(w["rel"])]
pm = w.groupby(["series", "month"])["rel"].median().unstack(0)
out.table(pm.round(2), "2d_week_profile_month", show=None)
out.p(pm.round(2).to_string())
fig, ax = plt.subplots(figsize=(8, 3.6))
for s in WEEK:
    ax.plot(pm.index, pm[s], marker="o", ms=4, label=s, **style(s))
ax.axhline(1, color=INK2, lw=0.6)
ax.set_xticks(pm.index, list("JFMAMJJASOND"))
ax.set_ylabel("median of price / yearly median")
ax.set_title("Weekly products: month-of-year profile")
ax.legend(fontsize=7, ncol=2)
out.fig(fig, "2d_week_profile_month")

# ── 2e  persistence ─────────────────────────────────────────────────────────────
out.h("2e  Persistence: autocorrelation of log(price)")


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    x = x - np.nanmean(x)
    v = np.nansum(x * x)
    return np.array([np.nansum(x[k:] * x[:-k]) / v if k else 1.0 for k in range(lags + 1)])


rows = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for s in FOUR_H:
    # lag k = k blocks in delivery order (6 blocks per day; DST days have 3 / 5 h blocks but
    # still 6 blocks; blocks without a price are dropped, so a lag can span a gap)
    d = tg[tg["series"] == s].set_index("t")[PRICE].clip(lower=FLOOR).sort_index()
    a = acf(np.log(d.to_numpy()), 84)
    rows[s] = {f"lag{k}": a[k] for k in (1, 6, 42, 84)}
    axes[0].plot(np.arange(85) / 6, a, label=s, **style(s))
axes[0].set_xlabel("lag [days] (4h blocks)")
axes[0].set_title("4h series")
axes[0].set_xticks(range(0, 15, 1))
for s in WEEK:
    d = tg[tg["series"] == s].set_index("t")[PRICE].clip(lower=FLOOR)
    a = acf(np.log(d.to_numpy()), 12)
    rows[s] = {f"lag{k}": a[k] for k in (1, 4, 12)}
    axes[1].plot(range(13), a, marker="o", ms=3, label=s, **style(s))
axes[1].set_xlabel("lag [weeks]")
axes[1].set_title("weekly series")
for ax in axes:
    ax.axhline(0, color=INK2, lw=0.6)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
axes[0].set_ylabel("autocorrelation of log price")
fig.tight_layout()
out.fig(fig, "2e_autocorrelation")
acft = pd.DataFrame(rows).T.reindex(ORDER)
out.table(acft.round(3), "2e_autocorrelation", show=None)
out.p(acft.round(2).to_string())
out.p("(4h: lag1 = previous block, lag6 = same slot previous day, lag42 = same slot one week earlier)")

out.close()
