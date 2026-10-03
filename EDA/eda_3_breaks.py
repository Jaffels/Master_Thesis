"""EDA 3 - Structural breaks (todo 6.3, RQ1b).

Market-design breaks (auction / price formation changed) are kept apart from reporting
breaks (only the way a driver is published changed).

Outputs (EDA/Output/3_breaks/)
  3a  before / after comparison of the capacity prices at each market break
      - window: up to WINDOW weeks on each side, cut at the neighbouring break of the same series
      - prices aggregated to the coarser procurement period of the two sides
        (e.g. FCR day -> weekly mean when compared with FCR week)
      - change = log(median after / median before) with a bootstrap 95 % interval
      - placebo = the same comparison on the same calendar dates one year earlier
        (same product, no break) -> separates the break from seasonality / drift
  3b  event plots: weekly prices +- 1 year around each break
  3c  CH imbalance price: resolution change 1 Jun 2022 (share of hours with 4 identical
      quarter-hour prices, intra-hour spread) and single price 1 Jan 2026 (long - short)
  3d  CH aFRR platform fall-back (2024 ->): fall-back share vs aFRR capacity prices
  3e  reporting breaks: CH ENTSO-E generation vs Swissgrid production, CH solar / run-of-river
      coverage regimes (regime_ch_gen_reporting) - not market breaks

    python EDA/eda_3_breaks.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eda_common import (Out, plt, style, mdates, load_targets, load_master, mark_breaks, date_axis, bootstrap_median_diff,
                        BREAKS, PRICE, SERIES_COLORS, INK2, MUTED, TZ)

out = Out("3_breaks", __doc__)
WINDOW = pd.Timedelta(weeks=26)
FLOOR = 0.01

tg = load_targets()
tg = tg[tg[PRICE].notna()].copy()

# synthetic series: symmetric aFRR equivalent after the split = up + down of the same week
ud = tg[tg["series"].isin(["aFRR up week", "aFRR down week"])]
ud = ud.pivot_table(index="t", columns="direction", values=PRICE).dropna()
syn = pd.DataFrame({"t": ud.index, PRICE: (ud["up"] + ud["down"]).to_numpy(), "series": "aFRR up+down week",
                    "procurement": "week"})
tg = pd.concat([tg, syn], ignore_index=True)

PERIOD = {"week": "W-MON", "day": "D", "4h": None}
# (break, series before, series after); same name = same series on both sides
COMPARE = [
    ("afrr_split", "aFRR sym week", "aFRR up+down week"),
    ("afrr_split", "aFRR sym week", "aFRR up week"),
    ("afrr_split", "aFRR sym week", "aFRR down week"),
    *[("de_zone_split", s, s) for s in ["FCR sym week", "aFRR up week", "aFRR down week", "mFRR up week",
                                        "mFRR down week", "mFRR up 4h", "mFRR down 4h"]],
    ("fcr_daily", "FCR sym week", "FCR sym day"),
    ("fcr_4h", "FCR sym day", "FCR sym 4h"),
    *[("afrr_fallback", s, s) for s in ["aFRR up week", "aFRR down week"]],
    *[("mfrr_merged", s, s) for s in ["mFRR up week", "mFRR down week", "mFRR up 4h", "mFRR down 4h"]],
    *[("afrr_daily", s, s) for s in ["aFRR up week", "aFRR down week"]],
]
bts = BREAKS.set_index("key")["ts"]
# neighbouring breaks per series (window is cut there)
series_breaks: dict[str, list[pd.Timestamp]] = {}
for k, a, b in COMPARE:
    for s in (a, b):
        series_breaks.setdefault(s, []).append(bts[k])


def window(s: str, t0: pd.Timestamp, side: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    others = [t for t in series_breaks.get(s, []) if t != t0]
    if side == "pre":
        lo = max([t0 - WINDOW] + [t for t in others if t < t0])
        return lo, t0
    hi = min([t0 + WINDOW] + [t for t in others if t > t0])
    return t0, hi


def prices(s: str, lo, hi, period: str | None) -> pd.Series:
    d = tg[(tg["series"] == s) & (tg["t"] >= lo) & (tg["t"] < hi)].set_index("t")[PRICE].sort_index()
    if period and len(d):
        d = d.resample(period, label="left", closed="left").mean().dropna()
    return d


def coarser(a: str, b: str) -> str | None:
    rank = {"4h": 0, "day": 1, "week": 2}
    pa, pb = a.split()[-1], b.split()[-1]
    p = pa if rank[pa] >= rank[pb] else pb
    return PERIOD[p]


def compare(k, a, b, shift=pd.Timedelta(0)):
    t0 = bts[k] - shift
    lo, _ = window(a, bts[k], "pre")
    _, hi = window(b, bts[k], "post")
    lo, hi = lo - shift, hi - shift
    per = coarser(a, b)
    sa, sb = (a, b) if shift == pd.Timedelta(0) else (a, a)        # placebo: before-series on both sides
    x = prices(sa, lo, t0, per)
    y = prices(sb, t0, hi, per)
    lx, ly = np.log(x.clip(lower=FLOOR).to_numpy()), np.log(y.clip(lower=FLOOR).to_numpy())
    d, d_lo, d_hi = bootstrap_median_diff(lx, ly)
    return {"n_pre": len(x), "n_post": len(y), "median_pre": x.median(), "median_post": y.median(),
            "iqr_pre": x.quantile(.75) - x.quantile(.25), "iqr_post": y.quantile(.75) - y.quantile(.25),
            "cv_pre": x.std() / x.mean() if len(x) else np.nan, "cv_post": y.std() / y.mean() if len(y) else np.nan,
            "dlog_median": d, "dlog_lo": d_lo, "dlog_hi": d_hi,
            "from": lo.strftime("%Y-%m-%d"), "to": hi.strftime("%Y-%m-%d"), "aggregated_to": per or "block"}


# ── 3a  before / after ──────────────────────────────────────────────────────────
out.h("3a  Capacity prices before / after each market break")
rows = []
for k, a, b in COMPARE:
    r = {"break": k, "date": bts[k].strftime("%Y-%m-%d"), "before": a, "after": b, **compare(k, a, b)}
    p = compare(k, a, b, shift=pd.Timedelta(weeks=52)) if tg.loc[tg["series"] == a, "t"].min() <= bts[k] - WINDOW - pd.Timedelta(weeks=52) else {}
    r["placebo_dlog"] = p.get("dlog_median", np.nan)
    r["placebo_lo"], r["placebo_hi"] = p.get("dlog_lo", np.nan), p.get("dlog_hi", np.nan)
    rows.append(r)
ba = pd.DataFrame(rows)
ba["ratio_post_pre"] = np.exp(ba["dlog_median"])
ba["ratio_placebo"] = np.exp(ba["placebo_dlog"])
ba["excess_vs_placebo"] = ba["dlog_median"] - ba["placebo_dlog"]
out.table(ba.round(3), "3a_before_after", show=None, index=False)
show = ba[["break", "before", "after", "aggregated_to", "n_pre", "n_post", "median_pre", "median_post",
           "ratio_post_pre", "dlog_lo", "dlog_hi", "ratio_placebo", "cv_pre", "cv_post"]]
out.p(show.round(2).to_string(index=False))
out.p("ratio = median after / median before; dlog_lo / hi = 95 % bootstrap interval of log(ratio) "
      "(iid, so optimistic for autocorrelated prices); ratio_placebo = same dates one year earlier.")

# forest plot
fig, ax = plt.subplots(figsize=(8, 0.32 * len(ba) + 1))
y = np.arange(len(ba))[::-1]
ax.errorbar(ba["dlog_median"], y, xerr=[ba["dlog_median"] - ba["dlog_lo"], ba["dlog_hi"] - ba["dlog_median"]],
            fmt="o", color=SERIES_COLORS[0], ms=5, lw=1.2, label="break")
ok = ba["placebo_dlog"].notna()
ax.errorbar(ba.loc[ok, "placebo_dlog"], y[ok.to_numpy()] - 0.25,
            xerr=[(ba["placebo_dlog"] - ba["placebo_lo"])[ok], (ba["placebo_hi"] - ba["placebo_dlog"])[ok]],
            fmt="s", color=MUTED, ms=4, lw=0.8, label="placebo (one year earlier)")
ax.axvline(0, color=INK2, lw=0.7)
ax.set_yticks(y, [f"{r['break']}: {r['before']}" + ("" if r["before"] == r["after"] else f" -> {r['after']}")
                  for _, r in ba.iterrows()], fontsize=7)
ax.set_xlabel("log(median after / median before)   (+0.69 = doubling, -0.69 = halving)")
ax.set_title("Price level change at the market breaks (+- 26 weeks, cut at neighbouring breaks)")
ax.legend(loc="lower right")
out.fig(fig, "3a_before_after_forest")

# ── 3b  event plots ─────────────────────────────────────────────────────────────
out.h("3b  Event plots")
keys = list(dict.fromkeys(k for k, _, _ in COMPARE))
fig, axes = plt.subplots(int(np.ceil(len(keys) / 2)), 2, figsize=(12, 2.8 * np.ceil(len(keys) / 2)), squeeze=False)
for ax, k in zip(axes.flat, keys):
    t0 = bts[k]
    ss = list(dict.fromkeys([s for kk, a, b in COMPARE if kk == k for s in (a, b)]))
    for i, s in enumerate(ss):
        d = tg[(tg["series"] == s) & (tg["t"] >= t0 - pd.Timedelta(weeks=52)) & (tg["t"] < t0 + pd.Timedelta(weeks=52))]
        d = d.set_index("t")[PRICE].resample("W-MON", label="left", closed="left").mean()
        ax.plot(d.index, d.clip(lower=FLOOR), lw=1, label=s, **style(s))
    ax.axvline(t0, color="#e34948", ls="--", lw=0.9)
    ax.axvspan(t0 - WINDOW, t0 + WINDOW, color="#eda100", alpha=0.06, lw=0)
    ax.set_yscale("log")
    ax.set_title(f"{k}  ({t0:%d %b %Y})", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
for ax in axes.flat[len(keys):]:
    ax.axis("off")
fig.suptitle("Weekly mean capacity prices +- 1 year around each market break (shaded = +- 26 weeks)",
             fontsize=10, fontweight="bold")
fig.tight_layout()
out.fig(fig, "3b_event_plots")

# ── 3c  imbalance price ─────────────────────────────────────────────────────────
out.h("3c  CH imbalance price: resolution (1 Jun 2022) and single price (1 Jan 2026)")
im = load_master(["ch_imb_price_long_eur_mwh", "ch_imb_price_short_eur_mwh", "src_ch_imb_price"])
im = im[im["ch_imb_price_long_eur_mwh"].notna()].copy()
im["hour"] = pd.to_datetime(im["ts_utc"], utc=True).dt.floor("h")
g = im.groupby("hour")["ch_imb_price_short_eur_mwh"]
hr = pd.DataFrame({"n": g.size(), "nuniq": g.nunique(), "range": g.max() - g.min()})
hr = hr[hr["n"] == 4]
hr["flat"] = hr["nuniq"] == 1
mon = hr.groupby(hr.index.tz_convert(TZ).tz_localize(None).to_period("M")).agg(flat_share=("flat", "mean"), mean_intrahour_range=("range", "mean"))
im["spread"] = im["ch_imb_price_short_eur_mwh"] - im["ch_imb_price_long_eur_mwh"]
mon2 = im.groupby(im["t"].dt.tz_localize(None).dt.to_period("M")).agg(
    spread_mean=("spread", "mean"), spread_zero_share=("spread", lambda s: (s.abs() < 1e-6).mean()),
    short_mean=("ch_imb_price_short_eur_mwh", "mean"), short_std=("ch_imb_price_short_eur_mwh", "std"))
mon = mon.join(mon2, how="outer")
out.table(mon.round(3), "3c_imbalance_monthly", show=None)
for k in ("imb_qh", "imb_single"):
    t0 = bts[k].tz_localize(None).to_period("M")
    pre = mon[(mon.index < t0) & (mon.index >= t0 - 6)]
    post = mon[(mon.index >= t0) & (mon.index < t0 + 6)]
    out.p(f"{k}: 6 months before / after")
    out.p(pd.DataFrame({"before": pre.mean(), "after": post.mean()}).round(3).to_string())
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
x = mon.index.to_timestamp().tz_localize(TZ)
axes[0].plot(x, mon["flat_share"], color=SERIES_COLORS[0])
axes[0].set_ylabel("share of hours with\n4 equal QH prices")
axes[1].plot(x, mon["mean_intrahour_range"], color=SERIES_COLORS[1])
axes[1].set_ylabel("mean intra-hour\nrange [EUR/MWh]")
axes[2].plot(x, mon["spread_mean"], color=SERIES_COLORS[6])
axes[2].set_ylabel("mean short - long\n[EUR/MWh]")
for ax in axes:
    mark_breaks(ax, keys=["imb_qh", "imb_single"], label=ax is axes[0])
date_axis(axes[-1])
axes[0].set_title("CH imbalance price (short): resolution and dual / single pricing, monthly")
out.fig(fig, "3c_imbalance_price_regimes")
del im

# ── 3d  aFRR fall-back ──────────────────────────────────────────────────────────
out.h("3d  CH aFRR platform fall-back vs aFRR capacity prices (weekly)")
fb = load_master(["ch_afrr_fallback_share", "ch_afrr_platform_fallback"])
fbw = fb.set_index("t")[["ch_afrr_fallback_share", "ch_afrr_platform_fallback"]].resample(
    "W-MON", label="left", closed="left").mean()
del fb
pw = tg[tg["series"].isin(["aFRR up week", "aFRR down week"])].pivot_table(index="t", columns="series", values=PRICE)
pw.index = pw.index.floor("D")
j = fbw.join(pw, how="inner")
j = j[j.index >= pd.Timestamp("2021-01-01", tz=TZ)]
out.table(j.round(3), "3d_fallback_weekly", show=None)
for s in ["aFRR up week", "aFRR down week"]:
    for per, jj in [("2021 ->", j), ("2024 ->", j[j.index >= pd.Timestamp("2024-01-01", tz=TZ)])]:
        ok = jj[["ch_afrr_fallback_share", s]].dropna()
        if len(ok) > 10:
            r = ok["ch_afrr_fallback_share"].rank().corr(ok[s].rank())
            out.p(f"  Spearman(fall-back share, {s}) {per}: {r:.2f}  (n={len(ok)} weeks)")
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
for s in ["aFRR up week", "aFRR down week"]:
    ax.plot(j.index, j[s], label=s, **style(s))
ax.set_yscale("log")
ax.set_ylabel("CHF/MW/h (log)")
ax.legend(loc="upper left")
mark_breaks(ax, keys=["afrr_fallback", "afrr_daily"])
ax.set_title("aFRR weekly capacity prices and CH aFRR platform fall-back share")
ax2.fill_between(j.index, j["ch_afrr_fallback_share"], step="post", color=MUTED, alpha=0.5, lw=0)
ax2.set_ylim(0, 1)
ax2.set_ylabel("fall-back share\nof the week")
mark_breaks(ax2, keys=["afrr_fallback", "afrr_daily"], label=False)
date_axis(ax2)
out.fig(fig, "3d_afrr_fallback")

# ── 3e  reporting breaks ────────────────────────────────────────────────────────
out.h("3e  Reporting breaks (not market breaks): CH ENTSO-E generation")
gen = load_master(["ch_gen_total_mw_xchk_entsoe", "ch_gen_total_mw", "ch_gen_solar_mw", "ch_gen_hydro_ror_mw",
                   "ch_gen_solar_da_fc_mw", "ch_gen_wind_on_da_fc_mw", "regime_ch_gen_reporting"])
gm = gen.drop(columns=["ts_utc", "regime_ch_gen_reporting"]).set_index("t").resample("MS").mean()
gm["entsoe_over_swissgrid"] = gm["ch_gen_total_mw_xchk_entsoe"] / gm["ch_gen_total_mw"]
reg = gen.groupby("regime_ch_gen_reporting", observed=True)["t"].min().sort_values()
out.p("regime_ch_gen_reporting starts: " + ", ".join(f"{k} {v:%Y-%m-%d}" for k, v in reg.items()))
yr = gm.groupby(gm.index.year).mean()
out.table(yr.round(3), "3e_ch_generation_yearly", show=None)
out.p(yr[["entsoe_over_swissgrid", "ch_gen_solar_mw", "ch_gen_hydro_ror_mw", "ch_gen_solar_da_fc_mw"]].round(2).to_string())
del gen
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
axes[0].plot(gm.index, gm["entsoe_over_swissgrid"], color=SERIES_COLORS[0])
axes[0].set_ylabel("ENTSO-E total /\nSwissgrid production")
axes[1].plot(gm.index, gm["ch_gen_solar_mw"], color=SERIES_COLORS[3], label="solar actual")
axes[1].plot(gm.index, gm["ch_gen_solar_da_fc_mw"], color=SERIES_COLORS[1], label="solar DA forecast")
axes[1].set_ylabel("MW (monthly mean)")
axes[1].legend(loc="upper left")
axes[2].plot(gm.index, gm["ch_gen_hydro_ror_mw"], color=SERIES_COLORS[2], label="run-of-river actual")
axes[2].set_ylabel("MW (monthly mean)")
axes[2].legend(loc="upper left")
for ax in axes:
    for k, v in reg.items():
        if v > gm.index.min():
            ax.axvline(v, color=MUTED, ls=":", lw=1)
            if ax is axes[0]:
                ax.text(v, 1.0, " " + str(k), transform=ax.get_xaxis_transform(), va="top", fontsize=7, color=INK2)
date_axis(axes[-1])
axes[0].set_title("CH ENTSO-E generation reporting regimes (dotted) - reporting breaks, not market breaks")
out.fig(fig, "3e_ch_generation_reporting")

out.close()
