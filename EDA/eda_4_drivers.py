"""EDA 4 - Relationships between drivers and capacity prices (todo 6.4; ex-post views).

Ex-post = every driver aggregated over the delivery window (actuals, not what was known at
gate closure). This shows which drivers co-move with the price; whether they can be used
ex ante is the job of the ex-ante views (RQ2).

For each CH target series (price_settle_ch) and every feature column of its view:
  level         Spearman rank correlation of price and driver
  within-month  Spearman after removing the calendar-month level of both (price: log minus the
                month median; driver: minus the month mean; 4h series: per month x delivery
                slot, so the daily solar / load shape is removed too) -> short-run co-movement,
                not driven by the common 2021-23 price level / trend or the time of day
Window: RQ1a (2021 ->) for series that exist then, otherwise the series' own life (RQ1b).
Columns with < 70 % of blocks covered in the window, flags, keys, regimes and targets are skipped.

Outputs (EDA/Output/4_drivers/)
  4a  driver ranking per series (csv: all columns; report: top 15 per series)
  4b  heatmap: top drivers x series (level and within-month)
  4c  stability: yearly Spearman of the top drivers per 4h series
  4d  cross-market: weekly CH prices vs DE regelleistung.net (aFRR / mFRR, 4h, from Jul 2018)
      and FCR cooperation prices, levels and week-on-week changes

    python EDA/eda_4_drivers.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eda_common import (Out, plt, load_expost, load_targets, view_dictionary, spearman, MASTER_DIR, PRICE,
                        DIV_CMAP, INK2, TZ)

out = Out("4_drivers", __doc__)
MIN_COVER = 0.7
TOP = 15
vd = view_dictionary()


def features(view: str) -> list[str]:
    d = vd[(vd["table"] == f"{view}_expost") & (vd["role"] == "feature")]
    return d["column"].tolist()


rows, per_series = [], {}
for view in ("fcr", "afrr", "mfrr"):
    df = load_expost(view, None)
    feats = [c for c in features(view) if c in df and pd.api.types.is_numeric_dtype(df[c])]
    for s, g in df.groupby("series"):
        g = g[g[PRICE].notna()]
        w = g[g["t"] >= pd.Timestamp("2021-01-01", tz=TZ)]
        win = "RQ1a"
        if len(w) < 100:
            w, win = g, "own life"
        if len(w) < 60:
            continue
        y = np.log(w[PRICE].clip(lower=0.01))
        ym = w["t"].dt.tz_localize(None).dt.to_period("M").astype(str)
        if s.endswith("4h"):                      # 4h: also remove the daily slot pattern
            ym = ym + "_" + w["t"].dt.hour.astype(str)
        y_dm = y - y.groupby(ym).transform("median")
        for c in feats:
            x = w[c].astype("float64")
            cov = x.notna().mean()
            if cov < MIN_COVER or x.nunique() < 5:
                continue
            r_lvl, n = spearman(y, x)
            x_dm = x - x.groupby(ym).transform("mean")
            r_dm, _ = spearman(y_dm, x_dm)
            rows.append({"series": s, "window": win, "column": c, "n": n, "coverage": cov,
                         "rho_level": r_lvl, "rho_within_month": r_dm})
        per_series[s] = w
dr = pd.DataFrame(rows)
fam = vd.drop_duplicates("column").set_index("column")["source"].str.split(" \\(").str[0]
dr["source"] = dr["column"].map(fam)
dr["abs_level"] = dr["rho_level"].abs()
dr.to_csv(out.dir / "4a_driver_correlations_all.csv", index=False)

# ── 4a  ranking ─────────────────────────────────────────────────────────────────
out.h("4a  Top drivers per series (|Spearman| on the price level; within-month alongside)")
ORDER = ["FCR sym week", "FCR sym day", "FCR sym 4h", "aFRR sym week", "aFRR up week", "aFRR down week",
         "aFRR up 4h", "aFRR down 4h", "mFRR up week", "mFRR down week", "mFRR up 4h", "mFRR down 4h"]
ORDER = [s for s in ORDER if s in set(dr["series"])]
for s in ORDER:
    d = dr[dr["series"] == s].sort_values("abs_level", ascending=False)
    out.p(f"\n{s}  (window {d['window'].iat[0]}, n = {d['n'].max()} blocks, {len(d)} columns tested)")
    out.p(d.head(TOP)[["column", "rho_level", "rho_within_month", "coverage"]].round(2).to_string(index=False))
    dw = d.assign(a=d["rho_within_month"].abs()).sort_values("a", ascending=False).head(5)
    out.p("  strongest within-month: " + ", ".join(f"{c} ({r:+.2f})" for c, r in zip(dw["column"], dw["rho_within_month"])))

# ── 4b  heatmap ─────────────────────────────────────────────────────────────────
out.h("4b  Heatmap of the top drivers")
top_cols = list(dict.fromkeys(c for s in ORDER for c in
                              dr[dr["series"] == s].nlargest(6, "abs_level")["column"]))
for kind in ("rho_level", "rho_within_month"):
    hm = dr[dr["column"].isin(top_cols)].pivot_table(index="column", columns="series", values=kind)
    hm = hm.reindex(index=top_cols, columns=ORDER)
    fig, ax = plt.subplots(figsize=(1.0 * len(ORDER) + 4, 0.24 * len(hm) + 1.6))
    im = ax.imshow(hm.to_numpy(dtype=float), cmap=DIV_CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(hm)), hm.index, fontsize=6.5)
    ax.grid(False)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(v) > 0.6 else INK2)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label("Spearman rho")
    ax.set_title(f"Top-6 drivers per series ({'price level' if kind == 'rho_level' else 'within-month'}), ex-post views")
    out.fig(fig, f"4b_top_drivers_{kind}")
    hm.round(3).to_csv(out.dir / f"4b_top_drivers_{kind}.csv")

# ── 4c  stability by year (4h series) ───────────────────────────────────────────
out.h("4c  Yearly Spearman of the top-5 drivers (4h series)")
four = [s for s in ORDER if s.endswith("4h")]
fig, axes = plt.subplots(1, len(four), figsize=(3.4 * len(four), 3.4), squeeze=False, sharey=True)
stab = []
for ax, s in zip(axes[0], four):
    w = per_series[s]
    y = np.log(w[PRICE].clip(lower=0.01))
    cols = dr[dr["series"] == s].nlargest(5, "abs_level")["column"].tolist()
    for i, c in enumerate(cols):
        r = {yr: spearman(y[w["t"].dt.year == yr], w.loc[w["t"].dt.year == yr, c].astype("float64"))[0]
             for yr in sorted(w["t"].dt.year.unique())}
        stab.append({"series": s, "column": c, **r})
        ax.plot(list(r), list(r.values()), marker="o", ms=3, label=c, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i])
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title(s, fontsize=9)
    ax.legend(fontsize=5.5, loc="lower left")
axes[0][0].set_ylabel("Spearman rho within the year")
fig.suptitle("Do the strongest drivers keep their sign and size from year to year?", fontsize=10, fontweight="bold")
fig.tight_layout()
out.fig(fig, "4c_driver_stability_by_year")
out.table(pd.DataFrame(stab).round(2), "4c_driver_stability_by_year", show=None, index=False)
out.p(pd.DataFrame(stab).round(2).to_string(index=False))

# ── 4d  cross-market ────────────────────────────────────────────────────────────
out.h("4d  Cross-market: weekly CH prices vs DE (regelleistung.net) and FCR cooperation")
ab = pd.read_parquet(MASTER_DIR / "auction_blocks.parquet")
ab["t"] = ab["block_start_utc"].dt.tz_convert(TZ)
wk = lambda s: s.resample("W-MON", label="left", closed="left").mean()  # noqa: E731
cols = {}
for s, g in load_targets().groupby("series"):
    cols[f"CH {s}"] = wk(g.set_index("t")[PRICE])
de = ab[ab["market"] == "de_regelleistung"]
for (p, d), g in de.groupby(["product", "direction"]):
    cols[f"DE {p} {d} 4h (avg)"] = wk(g.set_index("t")["price_average_de"])
fc = ab[ab["market"] == "fcr_coop"]
cols["FCR coop settle"] = wk(fc.set_index("t")["price_settle_coop"])
cols["FCR DE settle"] = wk(fc.set_index("t")["price_settle_de"])
X = pd.DataFrame(cols)
X = X[X.index >= pd.Timestamp("2018-07-16", tz=TZ)]
keep = [c for c in X.columns if X[c].notna().sum() >= 52]
X = np.log(X[keep].clip(lower=0.01))
lvl = X.rank().corr(min_periods=52)
chg = X.diff().rank().corr(min_periods=52)
out.p("Spearman of weekly log prices (levels), 2018-07 ->:")
ch_rows = [c for c in keep if c.startswith("CH ")]
oth = [c for c in keep if not c.startswith("CH ")]
out.table(lvl.loc[ch_rows, oth].round(2), "4d_crossmarket_levels", show=None)
out.p(lvl.loc[ch_rows, oth].round(2).to_string())
out.p("\nSpearman of week-on-week changes of log prices:")
out.table(chg.loc[ch_rows, oth].round(2), "4d_crossmarket_changes", show=None)
out.p(chg.loc[ch_rows, oth].round(2).to_string())
fig, axes = plt.subplots(1, 2, figsize=(13, 0.4 * len(ch_rows) + 2))
for ax, m, t in zip(axes, [lvl, chg], ["levels", "week-on-week changes"]):
    mm = m.loc[ch_rows, oth]
    im = ax.imshow(mm.to_numpy(dtype=float), cmap=DIV_CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(oth)), oth, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(ch_rows)), ch_rows, fontsize=7)
    ax.grid(False)
    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            v = mm.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.6 else INK2)
    ax.set_title(f"weekly log prices, {t}")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01).set_label("Spearman rho")
fig.suptitle("CH capacity prices vs German / FCR-cooperation prices (2018-07 ->)", fontsize=10, fontweight="bold")
out.fig(fig, "4d_crossmarket")

out.close()
