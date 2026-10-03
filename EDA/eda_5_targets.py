"""EDA 5 - Target definition (todo 6.5; open aFRR decision in V0.3, 3.6).

Candidate price targets per CH block (all per MW and hour of delivery):
  price_settle_ch  price paid in CH (pay-as-bid: = VWAP of accepted bids; FCR: marginal
                   cooperation / CH settlement price)
  price_bid_vwap   volume-weighted average accepted bid
  price_bid_max    marginal (highest) accepted bid
  price_bid_min    lowest accepted bid
Plus the volume side: offered_mw, awarded_mw, n_bids, n_accepted (competition).

Outputs (EDA/Output/5_targets/)
  5a  how far apart the candidates are: ratio marginal / VWAP and settle / VWAP by series x year
  5b  how similar they move: Spearman between the candidates (levels and changes)
  5c  aFRR: VWAP vs marginal accepted bid over time (weekly and 4h)
  5d  volumes and competition: offered / awarded MW, bids per block, by series x year
  5e  blocks without procurement (price NaN) by series x year and slot

    python EDA/eda_5_targets.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eda_common import Out, plt, style, load_targets, mark_breaks, date_axis, PRICE, SERIES_COLORS, INK2, TZ

out = Out("5_targets", __doc__)
tg = load_targets()
tg["year"] = tg["t"].dt.year
ORDER = ["FCR sym week", "FCR sym day", "FCR sym 4h", "aFRR sym week", "aFRR up week", "aFRR down week",
         "aFRR up 4h", "aFRR down 4h", "mFRR up week", "mFRR down week", "mFRR up 4h", "mFRR down 4h"]
ORDER = [s for s in ORDER if s in set(tg["series"])]
p = tg[tg["price_bid_vwap"] > 0].copy()
p["marg_over_vwap"] = p["price_bid_max"] / p["price_bid_vwap"]
p["settle_over_vwap"] = p["price_settle_ch"] / p["price_bid_vwap"]
p["min_over_vwap"] = p["price_bid_min"] / p["price_bid_vwap"]

# ── 5a  distance between the candidates ─────────────────────────────────────────
out.h("5a  Median ratios between the candidate targets")
r = p.groupby("series")[["marg_over_vwap", "settle_over_vwap", "min_over_vwap"]].median().reindex(ORDER)
r["share_settle_eq_vwap"] = p.groupby("series").apply(
    lambda g: (np.isclose(g["price_settle_ch"], g["price_bid_vwap"], rtol=1e-3)).mean()).reindex(ORDER)
out.table(r.round(3), "5a_ratios_by_series", show=None)
out.p(r.round(2).to_string())
ry = p.groupby(["series", "year"])["marg_over_vwap"].median().unstack().reindex(ORDER)
out.p("\nmedian marginal / VWAP by year:")
out.table(ry.round(2), "5a_marginal_over_vwap_by_year", show=None)
out.p(ry.round(2).to_string())

# ── 5b  co-movement ─────────────────────────────────────────────────────────────
out.h("5b  Spearman between the candidates (log; levels and block-to-block changes)")
rows = []
for s in ORDER:
    g = p[p["series"] == s].sort_values("t")
    L = np.log(g[["price_settle_ch", "price_bid_vwap", "price_bid_max", "price_bid_min"]].clip(lower=0.01))
    lv, ch = L.rank().corr(), L.diff().rank().corr()
    rows.append({"series": s, "vwap~marginal": lv.at["price_bid_vwap", "price_bid_max"],
                 "vwap~marginal (changes)": ch.at["price_bid_vwap", "price_bid_max"],
                 "settle~vwap": lv.at["price_settle_ch", "price_bid_vwap"],
                 "settle~marginal": lv.at["price_settle_ch", "price_bid_max"],
                 "vwap~min": lv.at["price_bid_vwap", "price_bid_min"]})
cm = pd.DataFrame(rows).set_index("series")
out.table(cm.round(3), "5b_candidate_correlations", show=None)
out.p(cm.round(2).to_string())

# ── 5c  aFRR VWAP vs marginal ───────────────────────────────────────────────────
out.h("5c  aFRR: VWAP vs marginal accepted bid")
af = ["aFRR up week", "aFRR down week", "aFRR up 4h", "aFRR down 4h"]
fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex="col")
for ax, s in zip(axes.T.flat, af):
    g = p[p["series"] == s].set_index("t").sort_index()
    if s.endswith("4h"):
        g = g[["price_bid_vwap", "price_bid_max", "price_bid_min"]].resample("D").mean()
    ax.fill_between(g.index, g["price_bid_min"], g["price_bid_max"], color=style(s)["color"], alpha=0.18, lw=0,
                    label="min - marginal accepted bid")
    ax.plot(g.index, g["price_bid_vwap"], color=style(s)["color"], lw=1.1, label="VWAP")
    ax.plot(g.index, g["price_bid_max"], color=INK2, lw=0.7, ls="--", label="marginal")
    ax.set_yscale("log")
    ax.set_title(s + (" (daily means)" if s.endswith("4h") else ""), fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    if s.endswith("week"):
        mark_breaks(ax, keys=["afrr_fallback", "afrr_daily"], label=False)
for ax in axes[1]:
    date_axis(ax, years=ax is axes[1][0])
axes[0][0].set_ylabel("CHF/MW/h (log)")
axes[1][0].set_ylabel("CHF/MW/h (log)")
fig.suptitle("aFRR accepted bids: VWAP (line), marginal (dashed), range of accepted bids (band)",
             fontsize=10, fontweight="bold")
fig.tight_layout()
out.fig(fig, "5c_afrr_vwap_vs_marginal")

fig, ax = plt.subplots(figsize=(8, 3.6))
for s in [x for x in ORDER if x.split()[0] in ("aFRR", "mFRR")]:
    if s in ry.index:
        ax.plot(ry.columns, ry.loc[s], marker="o", ms=3, label=s, **style(s))
ax.set_yscale("log")
ax.set_ylabel("median marginal / VWAP")
ax.set_title("Spread between marginal accepted bid and VWAP (aFRR / mFRR, pay-as-bid)")
ax.legend(fontsize=6.5, ncol=2)
out.fig(fig, "5c_marginal_over_vwap_by_year")

# ── 5d  volumes and competition ─────────────────────────────────────────────────
out.h("5d  Volumes and competition (median per block)")
tg["offer_ratio"] = tg["offered_mw"] / tg["awarded_mw"]
tg["accept_share"] = tg["n_accepted"] / tg["n_bids"]
vol = tg.groupby(["series", "year"])[["awarded_mw", "offered_mw", "offer_ratio", "n_bids", "accept_share"]].median()
out.table(vol.round(2), "5d_volumes_by_series_year", show=None)
for c in ["awarded_mw", "offer_ratio", "n_bids"]:
    out.p(f"\nmedian {c}:")
    out.p(vol[c].unstack().reindex(ORDER).round(1).to_string())
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, c, t in zip(axes, ["awarded_mw", "offer_ratio", "n_bids"],
                    ["awarded MW (CH procured)", "offered / awarded MW", "bids per block"]):
    v = vol[c].unstack()
    for s in ORDER:
        if s in v.index:
            ax.plot(v.columns, v.loc[s], marker="o", ms=3, label=s, **style(s))
    ax.set_title(t)
    if c == "offer_ratio":
        ax.axhline(1, color=INK2, lw=0.6)
axes[0].legend(fontsize=6, ncol=2)
fig.suptitle("Procured volume and competition per block (median by year)", fontsize=10, fontweight="bold")
fig.tight_layout()
out.fig(fig, "5d_volumes_competition")

# competition vs price (within series and year, blocks)
out.p("\nSpearman(offered / awarded, log price) per series, 2021 ->:")
for s in ORDER:
    g = tg[(tg["series"] == s) & (tg["t"] >= pd.Timestamp("2021-01-01", tz=TZ))].dropna(subset=["offer_ratio", PRICE])
    if len(g) <= 50:
        continue
    if g["offer_ratio"].nunique() < 3:
        out.p(f"  {s:16s} offered = awarded in the source (only accepted bids reported) -> no competition measure")
        continue
    rho = g["offer_ratio"].rank().corr(np.log(g[PRICE].clip(lower=0.01)).rank())
    out.p(f"  {s:16s} rho = {rho:+.2f}  (n = {len(g)})")

# ── 5e  blocks without procurement ──────────────────────────────────────────────
out.h("5e  Blocks without procurement (price_settle_ch NaN)")
tg["no_proc"] = tg[PRICE].isna()
np_y = tg.groupby(["series", "year"])["no_proc"].mean().unstack().reindex(ORDER)
out.table(np_y.round(3), "5e_no_procurement_share", show=None)
out.p(np_y.loc[np_y.max(axis=1) > 0].round(3).to_string())
m4 = tg[tg["series"].isin(["mFRR up 4h", "mFRR down 4h"])]
sl = m4.groupby(["series", m4["t"].dt.hour])["no_proc"].mean().unstack(0)
out.p("\nmFRR 4h: share without procurement by delivery slot (all years):")
out.p(sl.round(3).to_string())
out.p("-> a model of the mFRR 4h price needs a rule for these blocks (drop, or a two-part model: "
      "procured yes/no, then price)")

out.close()
