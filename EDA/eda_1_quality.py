"""EDA 1 - Data quality: coverage, gaps and flags (todo 6.1).

Reads (nothing is written outside EDA/Output/1_quality/):
  Clean/Data/master/data_dictionary.csv, nan_share_by_year.csv, gap_series_summary.csv,
  gap_list.csv, nan_runs_unexplained.csv, master_15min.parquet (flag columns only),
  Clean/Data/views/{fcr,afrr,mfrr}_expost.parquet and *_expost_coverage.parquet

Outputs
  1a  master availability by column family x year     (heatmap + csv)
  1b  columns with partial availability, per year       (heatmap of the worst columns + csv)
  1c  raw-series gaps: share missing per series, missing hours by source x year
  1d  unexplained NaN runs in the master (not in the gap list)
  1e  flags: share of quarter-hours flagged per flag x year
  1f  view coverage: share of blocks with complete / no data per column, RQ1a and RQ1b windows
  1g  targets: blocks and missing prices per series x year

    python EDA/eda_1_quality.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eda_common import (Out, plt, master_dictionary, view_dictionary, load_master, load_expost,
                        MASTER_DIR, VIEWS_DIR, SEQ_CMAP, SERIES_COLORS, INK2, PRICE, V)

out = Out("1_quality", __doc__)
YEARS = [str(y) for y in range(2015, 2027)]


def area_group(a) -> str:
    if pd.isna(a) or a in ("ce", "de_amprion"):
        return "CE / other"
    if "->" in str(a):
        return "cross-border"
    return "CH" if a == "ch" else "neighbours"


def family_table(dic: pd.DataFrame) -> pd.Series:
    m = dic[dic["table"] == "master_15min"].set_index("column")
    tab = m["clean_table"].fillna("-").str.split(" + ", regex=False).str[0]
    return (tab + "  |  " + m["area"].map(area_group)).rename("family")


def heat(ax, data: pd.DataFrame, fmt="{:.0%}", vmin=0, vmax=1, cmap=SEQ_CMAP, annotate=True):
    im = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(data.shape[1]), data.columns, rotation=0)
    ax.set_yticks(range(data.shape[0]), data.index)
    ax.grid(False)
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data.iat[i, j]
                if pd.notna(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                            color="white" if (v - vmin) / (vmax - vmin + 1e-12) > 0.6 else INK2)
    return im


dic = master_dictionary()
fam = family_table(dic)

# ── 1a  availability by family x year ───────────────────────────────────────────
out.h("1a  Master availability (share of non-NaN quarter-hours) by column family x year")
ny = pd.read_csv(MASTER_DIR / "nan_share_by_year.csv").set_index("column")
avail = 1 - ny[YEARS]
avail = avail.join(fam)
fam_avail = avail.groupby("family")[YEARS].mean()
fam_avail.insert(0, "n_cols", avail.groupby("family").size())
fam_avail = fam_avail.sort_index()
out.table(fam_avail.round(3), "1a_availability_by_family_year", show=40)
fig, ax = plt.subplots(figsize=(10, 0.28 * len(fam_avail) + 1.2))
heat(ax, fam_avail[YEARS])
ax.set_yticks(range(len(fam_avail)), [f"{f}  ({n})" for f, n in zip(fam_avail.index, fam_avail["n_cols"])])
ax.set_title("Master table: mean share of available quarter-hours per column family (n columns)")
out.fig(fig, "1a_availability_by_family_year")

# ── 1b  columns with partial availability ───────────────────────────────────────
out.h("1b  Columns that are not (almost) complete in every year they exist")
part = avail[YEARS]
# a series that starts / ends during the sample is not a gap: only full live years count
# (live span = dictionary start_utc .. end_utc; the start and end year are dropped unless
# the span covers them from the first / to the last week)
m = dic[dic["table"] == "master_15min"].set_index("column")
s0 = pd.to_datetime(m["start_utc"], utc=True).reindex(part.index)
s1 = pd.to_datetime(m["end_utc"], utc=True).reindex(part.index)
yr = np.array([int(y) for y in YEARS])
first_full = np.where(s0.dt.dayofyear <= 7, s0.dt.year, s0.dt.year + 1)
last_full = np.where((s1.dt.month == 12) & (s1.dt.day >= 24) | (s1.dt.year >= 2026), s1.dt.year, s1.dt.year - 1)
live = (yr[None, :] >= first_full[:, None]) & (yr[None, :] <= last_full[:, None])
exists = pd.DataFrame(live, index=part.index, columns=YEARS) & part.gt(0.0)
in_life = part.where(exists)
worst = (1 - in_life).max(axis=1)                      # worst missing share in a live year
first_year = exists.idxmax(axis=1).where(exists.any(axis=1))
cols_b = pd.DataFrame({"family": avail["family"], "first_year": first_year,
                       "worst_live_year_missing": worst.round(4),
                       "mean_missing_live": (1 - in_life).mean(axis=1).round(4)})
cols_b = cols_b[cols_b["worst_live_year_missing"] > 0.01].sort_values("worst_live_year_missing", ascending=False)
out.p(f"{len(cols_b)} of {len(part)} master data columns miss > 1 % in at least one full year of their live span "
      "(start / end years of a series excluded)")
out.table(cols_b, "1b_partial_columns", show=30)
top = part.loc[cols_b.index[:60]]
fig, ax = plt.subplots(figsize=(10, 0.2 * len(top) + 1.2))
heat(ax, top, annotate=False)
ax.tick_params(axis="y", labelsize=6)
ax.set_title("Columns with > 1 % missing in a live year (top 60): share available per year")
fig.colorbar(ax.images[0], ax=ax, fraction=0.02, pad=0.01)
out.fig(fig, "1b_partial_columns_heatmap")

# ── 1c  raw series gaps ─────────────────────────────────────────────────────────
out.h("1c  Raw-series gaps (gap list)")
gs = pd.read_csv(MASTER_DIR / "gap_series_summary.csv")
by_src = gs.groupby("source").agg(series=("series", "size"),
                                  median_missing=("share_missing", "median"),
                                  max_missing=("share_missing", "max"),
                                  sparse=("sparse", lambda s: int(s.astype(str).eq("True").sum())))
out.table(by_src.round(4), "1c_gap_summary_by_source")
g30 = gs.dropna(subset=["share_missing"]).nlargest(30, "share_missing")
fig, ax = plt.subplots(figsize=(8, 0.22 * len(g30) + 1))
srcs = list(gs["source"].unique())
ax.barh(range(len(g30)), g30["share_missing"], color=[SERIES_COLORS[srcs.index(s) % 8] for s in g30["source"]],
        height=0.7)
ax.set_yticks(range(len(g30)), g30["series"].str.replace("entsoe/", "", regex=False), fontsize=6.5)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("share of expected time stamps missing (within the series' own span)")
ax.set_title("30 raw series with the largest missing share")
handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES_COLORS[srcs.index(s) % 8]) for s in g30["source"].unique()]
ax.legend(handles, list(g30["source"].unique()), loc="lower right")
out.fig(fig, "1c_series_missing_top30")

gl = pd.read_csv(MASTER_DIR / "gap_list.csv")
real = gl[~gl["kind"].isin(["resolution", "head", "tail", "series_start"])].copy()
real["year"] = pd.to_datetime(real["start_local"], utc=True).dt.tz_convert("Europe/Zurich").dt.year
hrs = real.pivot_table(index="source", columns="year", values="hours", aggfunc="sum", fill_value=0)
out.p("Missing hours summed over series (kind gap / source_gap / not_published ..., by gap start year):")
out.table(hrs.round(0), "1c_gap_hours_source_year")
out.p("gap-list rows by kind: " + ", ".join(f"{k} {v}" for k, v in gl["kind"].value_counts().items()))

# ── 1d  unexplained NaN runs ────────────────────────────────────────────────────
out.h("1d  NaN runs in the master that the gap list does not explain")
nr = pd.read_csv(MASTER_DIR / "nan_runs_unexplained.csv")
un = nr[~nr["explained"]]
out.p(f"{len(nr):,} NaN runs >= 1 day; {len(un):,} unexplained ({un['days'].sum():,.0f} column-days)")
un_c = un.groupby("column").agg(runs=("days", "size"), days=("days", "sum"),
                                first=("run_start_utc", "min"), last=("run_end_utc", "max"))
out.table(un_c.sort_values("days", ascending=False).round(1), "1d_unexplained_nan_runs_by_column", show=20)

# ── 1e  flags ───────────────────────────────────────────────────────────────────
out.h("1e  Flags: share of quarter-hours flagged, by year")
flag_cols = dic.loc[(dic["table"] == "master_15min") & (dic["class"] == "flag"), "column"].tolist()
fm = load_master(flag_cols)
fy = fm[flag_cols].astype("float32").groupby(fm["t"].dt.year).mean().T
fy.columns = fy.columns.astype(str)
fy["total_qh"] = fm[flag_cols].astype("float32").sum().astype(int)
fy = fy.sort_values("total_qh", ascending=False)
out.table(fy.round(4), "1e_flags_by_year", show=30)
fig, ax = plt.subplots(figsize=(10, 0.28 * len(fy) + 1.2))
fyv = fy[[c for c in fy.columns if c != "total_qh"]]
heat(ax, fyv, fmt="{:.1%}", vmax=max(0.05, float(np.nanmax(fyv.to_numpy()))))
ax.set_title("Share of quarter-hours flagged per year (values kept unless masked in the views)")
out.fig(fig, "1e_flags_by_year")
del fm

# ── 1f  view coverage ───────────────────────────────────────────────────────────
out.h("1f  Ex-post view coverage (share of the block's quarter-hours with data, per column)")
vd = view_dictionary()
rows = []
for view in ("fcr", "afrr", "mfrr"):
    cov = pd.read_parquet(VIEWS_DIR / f"{view}_expost_coverage.parquet")
    cov = cov[cov["market"] == "ch_swissgrid"] if "market" in cov else cov
    cov = V.apply_window(cov, None)                                # drops partial blocks
    feat = [c for c in cov.columns if c in set(vd.loc[(vd["table"] == f"{view}_expost") &
                                                       (vd["role"] == "feature"), "column"])]
    for rq in ("RQ1a", "RQ1b"):
        w = V.apply_window(cov, rq)
        x = w[feat]
        r = pd.DataFrame({"view": view, "window": rq, "column": feat,
                          "blocks": len(w),
                          "complete": (x >= 0.999).mean().to_numpy(),
                          "empty": (x <= 0).mean().to_numpy(),
                          "mean_cov": x.mean().to_numpy()})
        rows.append(r)
covt = pd.concat(rows, ignore_index=True)
covt = covt.merge(fam.rename_axis("column").reset_index(), on="column", how="left")
covt.to_csv(out.dir / "1f_view_coverage_by_column.csv", index=False)
summ = covt.groupby(["view", "window"]).agg(
    features=("column", "size"),
    complete_ge_99=("complete", lambda s: int((s >= 0.99).sum())),
    complete_90_99=("complete", lambda s: int(((s >= 0.9) & (s < 0.99)).sum())),
    complete_lt_90=("complete", lambda s: int((s < 0.9).sum())),
    empty_gt_50=("empty", lambda s: int((s > 0.5).sum())))
out.p("Feature columns by share of blocks with complete coverage:")
out.table(summ, "1f_view_coverage_summary")
r1a = covt[covt["window"] == "RQ1a"]
dead = r1a[r1a["empty"] >= 0.99]
out.p(f"\nRQ1a window: {dead['column'].nunique()} feature columns are empty (series ended / not started, e.g. "
      "DE-AT-LU borders, CH ENTSO-E gas) -> drop them from RQ1a models:")
out.p("  " + ", ".join(sorted(dead["column"].unique())))
dead[["view", "column", "family"]].to_csv(out.dir / "1f_empty_columns_RQ1a.csv", index=False)
structural = r1a["column"].map(lambda c: bool(V.STRUCTURAL_NAN.search(c)))
out.p(f"\nRQ1a window: {r1a.loc[structural, 'column'].nunique()} activation-price / masked TRE columns have "
      "structural NaN (no activation) - not gaps, excluded below.")
weak = r1a[(r1a["complete"] < 0.9) & (r1a["empty"] < 0.99) & ~structural].sort_values("complete")
out.p(f"RQ1a window: {weak['column'].nunique()} further feature columns complete in < 90 % of blocks (any view); worst:")
out.table(weak[["view", "column", "family", "complete", "empty", "mean_cov"]].round(3),
          "1f_weak_columns_RQ1a", show=25, index=False)
fs = r1a[r1a["empty"] < 0.99].groupby(["family", "view"])["complete"].mean().unstack()
fig, ax = plt.subplots(figsize=(5, 0.28 * len(fs) + 1.2))
heat(ax, fs)
ax.set_title("RQ1a window (2021 ->): mean share of blocks with\ncomplete data, by column family (empty columns excluded)")
out.fig(fig, "1f_view_coverage_family_RQ1a")

# ── 1g  targets ─────────────────────────────────────────────────────────────────
out.h("1g  Targets: CH blocks per series and year, and blocks without a price")
tg = pd.concat([load_expost(v, None, ["market", "product", "direction", "procurement", "block_start_utc",
                                       "block_end_utc", "partial_in_sample", PRICE]) for v in ("fcr", "afrr", "mfrr")])
tg["year"] = tg["t"].dt.year
n = tg.pivot_table(index="series", columns="year", values=PRICE, aggfunc="size", fill_value=0)
na = tg[tg[PRICE].isna()].pivot_table(index="series", columns="year", values="t", aggfunc="size", fill_value=0)
out.p("Blocks per series x year:")
out.table(n, "1g_target_blocks_series_year", show=20)
out.p(f"\nBlocks with {PRICE} = NaN (nothing procured):")
out.table(na, "1g_target_missing_price_series_year", show=20)

out.close()
