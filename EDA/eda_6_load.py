"""EDA 6 - CH load check (todo 6.6).

The ENTSO-E CH day-ahead load forecast fits the ENTSO-E CH actual load worse than the
neighbours do (corr ~0.84). Question: is the CH actual (or the forecast) the problem, and which
Swissgrid series is the better CH load reference?

Series (master, 15 min): ENTSO-E <zone>_load_actual_mw / <zone>_load_da_fc_mw for CH, DE_LU,
FR, IT_NORD, AT; Swissgrid ch_cons_mw (total consumption, control block), ch_cons_enduse_mw
(end-user consumption), ch_tn_vertical_feedin_mw (vertical feed-in to the transmission grid),
ch_tn_vertical_load_mw (hourly vertical grid load, ends 2022).

Outputs (EDA/Output/6_load/)
  6a  forecast quality per zone x year: Pearson r, MAPE, bias (fc - actual) / actual
  6b  CH: correlation of the ENTSO-E actual and DA forecast with each Swissgrid series, by year;
      level ratio ENTSO-E actual / Swissgrid
  6c  split of the error: daily means (level) vs within-day shape (deviation from the daily mean)
  6d  example weeks (winter / summer, latest full year): all CH series, each scaled to its own week mean

    python EDA/eda_6_load.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eda_common import Out, plt, load_master, date_axis, SERIES_COLORS, INK2, TZ

out = Out("6_load", __doc__)
ZONES = ["ch", "de_lu", "fr", "it_nord", "at"]
SG = ["ch_cons_mw", "ch_cons_enduse_mw", "ch_tn_vertical_feedin_mw", "ch_tn_vertical_load_mw"]
cols = [f"{z}_load_{k}_mw" for z in ZONES for k in ("actual", "da_fc")] + SG
m = load_master(cols)
m["year"] = m["t"].dt.year
m["day"] = m["t"].dt.tz_localize(None).dt.floor("D")


def quality(a: pd.Series, f: pd.Series) -> dict:
    ok = a.notna() & f.notna() & (a > 0)
    a, f = a[ok], f[ok]
    if len(a) < 100:
        return {"r": np.nan, "mape": np.nan, "bias": np.nan, "n": len(a)}
    return {"r": a.corr(f), "mape": (f - a).abs().div(a).mean(), "bias": (f - a).sum() / a.sum(), "n": len(a),
            "identical": (np.isclose(a, f, rtol=0, atol=0.5)).mean()}   # actual published = forecast


# ── 6a  forecast quality per zone ───────────────────────────────────────────────
out.h("6a  ENTSO-E day-ahead load forecast vs actual, per zone x year")
rows = []
for z in ZONES:
    for y, g in m.groupby("year"):
        rows.append({"zone": z, "year": y, **quality(g[f"{z}_load_actual_mw"], g[f"{z}_load_da_fc_mw"])})
qa = pd.DataFrame(rows)
qa.to_csv(out.dir / "6a_forecast_quality_zone_year.csv", index=False)
for k in ("r", "mape", "bias", "identical"):
    out.p(f"\n{k}:" + ("  (share of quarter-hours where actual = forecast, +-0.5 MW)" if k == "identical" else ""))
    out.p(qa.pivot(index="zone", columns="year", values=k).reindex(ZONES).round(3).to_string())
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for i, z in enumerate(ZONES):
    q = qa[qa["zone"] == z]
    kw = dict(color=SERIES_COLORS[i], marker="o", ms=3, lw=2 if z == "ch" else 1, label=z.upper())
    axes[0].plot(q["year"], q["r"], **kw)
    axes[1].plot(q["year"], q["mape"], **kw)
axes[0].set_title("Pearson r (DA forecast vs actual, 15 min)")
axes[1].set_title("MAPE")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0].legend()
fig.tight_layout()
out.fig(fig, "6a_forecast_quality_zones")

eqm = pd.Series(np.isclose(m["ch_load_actual_mw"], m["ch_load_da_fc_mw"], rtol=0, atol=0.5), index=m["t"])
eqm = eqm.groupby(m["t"].dt.tz_localize(None).dt.to_period("M").to_numpy()).mean()
bad = eqm[eqm > 0.9]
out.p("\nCH months in which the published ENTSO-E 'actual' load is the day-ahead forecast (> 90 % of quarter-hours): "
      + (", ".join(str(x) for x in bad.index) if len(bad) else "none"))
out.p("-> ch_load_actual_mw carries no information beyond the forecast in these months; "
      "prefer Swissgrid ch_cons_mw as CH load, or flag / mask these months.")
eqm.round(3).to_csv(out.dir / "6a_ch_actual_equals_forecast_monthly.csv")

sw = []
for y, g in m.groupby("year"):
    ok = g["ch_load_actual_mw"].notna() & g["ch_cons_mw"].notna()
    sw.append({"year": y, "share_actual_eq_swissgrid_cons": np.isclose(g.loc[ok, "ch_load_actual_mw"],
                                                                        g.loc[ok, "ch_cons_mw"], rtol=0.005).mean()})
out.p("\nCH ENTSO-E actual within 0.5 % of Swissgrid ch_cons_mw (share of quarter-hours):")
out.p(pd.DataFrame(sw).set_index("year").T.round(3).to_string())

# ── 6b  CH vs Swissgrid ─────────────────────────────────────────────────────────
out.h("6b  CH: ENTSO-E actual / DA forecast vs Swissgrid series (Pearson r, 15 min; hourly for vertical load)")
pairs = [(e, s) for e in ("ch_load_actual_mw", "ch_load_da_fc_mw") for s in SG] + [("ch_load_da_fc_mw", "ch_load_actual_mw")]
rows = []
for y, g in m.groupby("year"):
    for e, s in pairs:
        ok = g[e].notna() & g[s].notna()
        if ok.sum() < 1000:
            continue
        rows.append({"year": y, "pair": f"{e.replace('_mw', '')} ~ {s.replace('_mw', '')}",
                     "r": g.loc[ok, e].corr(g.loc[ok, s]), "level_ratio": g.loc[ok, e].mean() / g.loc[ok, s].mean()})
cb = pd.DataFrame(rows)
cb.to_csv(out.dir / "6b_ch_vs_swissgrid.csv", index=False)
out.p("Pearson r:")
out.p(cb.pivot(index="pair", columns="year", values="r").round(3).to_string())
out.p("\nlevel ratio (mean ENTSO-E / mean Swissgrid):")
out.p(cb.pivot(index="pair", columns="year", values="level_ratio").round(3).to_string())
fig, ax = plt.subplots(figsize=(10, 4))
pv = cb.pivot(index="year", columns="pair", values="r")
for i, c in enumerate(pv.columns):
    ax.plot(pv.index, pv[c], marker="o", ms=3, color=SERIES_COLORS[i % 8], label=c,
            ls="-" if c.startswith("ch_load_actual") else "--")
ax.set_ylabel("Pearson r")
ax.set_title("CH: which load series agree? (solid = ENTSO-E actual, dashed = ENTSO-E DA forecast)")
ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1))
out.fig(fig, "6b_ch_vs_swissgrid")

# ── 6c  level vs shape ──────────────────────────────────────────────────────────
out.h("6c  Daily level vs within-day shape (2021 ->)")
r = m[m["t"] >= pd.Timestamp("2021-01-01", tz=TZ)]
rows = []
for a, f in [(f"{z}_load_actual_mw", f"{z}_load_da_fc_mw") for z in ZONES] + \
            [("ch_cons_mw", "ch_load_da_fc_mw"), ("ch_cons_enduse_mw", "ch_load_da_fc_mw"),
             ("ch_tn_vertical_feedin_mw", "ch_load_da_fc_mw"), ("ch_cons_mw", "ch_load_actual_mw")]:
    d = r[["day", a, f]].dropna()
    dm = d.groupby("day")[[a, f]].transform("mean")
    daily = d.groupby("day")[[a, f]].mean()
    rows.append({"actual": a, "forecast / other": f,
                 "r_15min": d[a].corr(d[f]), "r_daily_mean": daily[a].corr(daily[f]),
                 "r_within_day": (d[a] - dm[a]).corr(d[f] - dm[f])})
lv = pd.DataFrame(rows)
out.table(lv.round(3), "6c_level_vs_shape", show=None, index=False)
out.p(lv.round(3).to_string(index=False))
out.p("r_daily_mean = agreement of the day-to-day level; r_within_day = agreement of the daily shape.")

# ── 6d  example weeks ───────────────────────────────────────────────────────────
out.h("6d  Example weeks")
last = int(m.loc[m["ch_load_actual_mw"].notna(), "year"].max()) - 1
weeks = [pd.Timestamp(f"{last}-01-13", tz=TZ), pd.Timestamp(f"{last}-07-14", tz=TZ)]
show = ["ch_load_actual_mw", "ch_load_da_fc_mw", "ch_cons_mw", "ch_cons_enduse_mw", "ch_tn_vertical_feedin_mw"]
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
for ax, w0 in zip(axes, weeks):
    w0 = w0 - pd.Timedelta(days=w0.dayofweek)
    d = m[(m["t"] >= w0) & (m["t"] < w0 + pd.Timedelta(days=7))].set_index("t")
    for i, c in enumerate(show):
        x = d[c]
        if x.notna().any():
            ax.plot(x.index, x / x.mean(), color=SERIES_COLORS[i], lw=1.4 if c.startswith("ch_load") else 0.9,
                    ls="--" if "fc" in c else "-", label=f"{c} (mean {x.mean():,.0f} MW)")
    ax.set_title(f"week from {w0:%d %b %Y}", fontsize=9)
    ax.set_ylabel("value / week mean")
    date_axis(ax, years=False)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.suptitle("CH load series, each scaled to its own week mean", fontsize=10, fontweight="bold")
fig.tight_layout()
out.fig(fig, "6d_example_weeks")

out.close()
