"""Modelling views, step 5.1: aggregate the 15-min master onto each CH target block.

For every Swissgrid auction block (after targets.combine_tenders) each master column
is aggregated over the delivery window  block_start_utc <= ts_utc < block_end_utc
with its aggregation_rule from master/data_dictionary.csv. This is the ex-post
(delivery-window) feature set; the ex-ante views (5.3) are built on top of it.

Outputs (only with --write), in Clean/Data/views/:
  {fcr,afrr,mfrr}_expost.parquet            one row per block: keys, targets, regimes, features
  {fcr,afrr,mfrr}_expost_coverage.parquet   same keys; per feature the share of non-NaN quarter-hours
  _dictionary.csv                            one row per column per view (role, rules, source)
  build_views_report.txt

Masks (5.4): TRE activation values are set to NULL where flag_tre_extra_activation_*
(dictionary column `flags`) before aggregating. Gap imputation and RQ windows are applied
at load time (Clean/views.py). Ex-ante rules: build_views_exante.py.

Run from Master_Thesis with .venv active:
    python Clean/build_views.py            # build + checks, writes nothing
    python Clean/build_views.py --write
"""
import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CLEAN = Path(__file__).resolve().parent
sys.path.insert(0, str(CLEAN))
import targets  # noqa: E402

MASTER = CLEAN / "Data" / "master"
MT = MASTER / "master_15min.parquet"
AB = MASTER / "auction_blocks.parquet"
DD = MASTER / "data_dictionary.csv"
OUT = CLEAN / "Data" / "views"

TARGET_MARKET = "ch_swissgrid"
VIEWS = {"FCR": "fcr_expost", "aFRR": "afrr_expost", "mFRR": "mfrr_expost"}
ZERO_INFO = ["at_gen_oil_mw", "de_ch_countertrade_mw", "at_ch_countertrade_mw",
             "it_nord_outage_prod_forced_mw"]
TIME_KEYS = ["ts_utc", "ts_local"]
IV = ["block_start_utc", "block_end_utc"]
BLOCK_KEY = list(targets.BLOCK_KEY)
SPOT_COLS = {"ch_load_actual_mw": "mean", "ch_gen_total_mw": "mean",
             "de_lu_load_actual_mw": "mean", "ch_imb_vol_mwh": "sum"}

_report: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _report.append(s)


def h(t):
    log(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


# ------------------------------------------------------------------ plan
def agg_kind(rule) -> str:
    r = str(rule).strip().lower()
    if r.startswith("volume-weighted"):
        return "vwmean"
    if r.startswith("mean"):
        return "mean"
    if r in ("sum", "max", "min", "any", "mode"):
        return r
    raise ValueError(f"unknown aggregation_rule: {rule!r}")


def build_plan(ab_cols: set[str]) -> tuple[dict, dict, dict, dict]:
    """-> plan {col: kind}, weights {price_col: weight_col}, excluded {col: reason},
    masks {col: flag column} (TRE extra activations, 5.4)"""
    dd = pd.read_csv(DD)
    ddm = dd[dd["table"] == "master_15min"].set_index("column")
    plan, weights, excluded = {}, {}, {}
    masks = {c: f for c, fl in ddm["flags"].dropna().items() for f in fl.split(";")
             if f.startswith("flag_tre_extra_activation")}
    for c in pq.read_schema(MT).names:
        if c in TIME_KEYS:
            excluded[c] = "time key"
        elif "_xchk_" in c:
            excluded[c] = "cross-check column (5.7)"
        elif c in ZERO_INFO:
            excluded[c] = "zero information (always 0)"
        elif c.startswith("regime_") and c in ab_cols:
            excluded[c] = "regime taken from auction_blocks"
        elif c.startswith("src_"):
            plan[c] = "mode"
        else:
            if c not in ddm.index:
                raise KeyError(f"{c} missing from data_dictionary")
            rule = ddm.at[c, "aggregation_rule"]
            if pd.isna(rule):
                raise ValueError(f"{c} has no aggregation_rule")
            plan[c] = agg_kind(rule)
            if plan[c] == "vwmean":
                w = c.replace("_price_eur_mwh", "_mw")
                if w == c or w not in ddm.index:
                    raise KeyError(f"no volume column for {c} (looked for {w})")
                weights[c] = w
    masks = {c: f for c, f in masks.items() if c in plan}
    return plan, weights, excluded, masks


def col_ref(c: str, masks: dict) -> str:
    q = f'm."{c}"'
    return f'(CASE WHEN m."{masks[c]}" THEN NULL ELSE {q} END)' if c in masks else q


def sql_expr(c: str, kind: str, w: str | None, masks: dict) -> str:
    q = col_ref(c, masks)
    if kind == "vwmean":
        wq = col_ref(w, masks)
        return (f"sum({q} * {wq}) / nullif(sum(CASE WHEN {q} IS NOT NULL AND {wq} IS NOT NULL "
                f"THEN {wq} END), 0)")
    return {"mean": f"avg({q})", "sum": f"sum({q})", "max": f"max({q})", "min": f"min({q})",
            "any": f"bool_or(CAST({q} AS BOOLEAN))", "mode": f"mode({q})"}[kind]


# ------------------------------------------------------------------ aggregation
def aggregate(iv: pd.DataFrame, plan: dict, weights: dict,
              masks: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("iv", iv)
    aggs = [f'{sql_expr(c, k, weights.get(c), masks)} AS "{c}"' for c, k in plan.items()]
    covs = [f'count({col_ref(c, masks)})::DOUBLE / count(*) AS "cov__{c}"' for c in plan]
    sql = f"""
        SELECT i.block_start_utc, i.block_end_utc, count(*) AS n_qh,
               {', '.join(aggs)}, {', '.join(covs)}
        FROM iv i JOIN read_parquet('{MT}') m
          ON m.ts_utc >= i.block_start_utc AND m.ts_utc < i.block_end_utc
        GROUP BY 1, 2"""
    res = con.execute(sql).df()
    for c in IV:
        res[c] = pd.to_datetime(res[c], utc=True).astype("datetime64[ns, UTC]")
    cov_cols = [c for c in res.columns if c.startswith("cov__")]
    feat = res.drop(columns=cov_cols)
    cov = res[IV + ["n_qh"] + cov_cols].rename(columns=lambda c: c.removeprefix("cov__"))
    return feat, cov


# ------------------------------------------------------------------ checks
def spot_check(view: pd.DataFrame, name: str, n: int = 5, seed: int = 0) -> None:
    cols = [c for c in SPOT_COLS if c in view.columns]
    # master stores floats as float32; recompute in float64 like DuckDB's avg/sum
    m = pd.read_parquet(MT, columns=["ts_utc"] + cols).set_index("ts_utc").astype("float64")
    sample = view[~view["partial_in_sample"]].sample(min(n, len(view)), random_state=seed)
    bad = 0
    for _, r in sample.iterrows():
        sl = m[(m.index >= r.block_start_utc) & (m.index < r.block_end_utc)]
        for c in cols:
            ref = sl[c].mean() if SPOT_COLS[c] == "mean" else sl[c].sum(min_count=1)
            if not np.isclose(ref, r[c], equal_nan=True, rtol=1e-6, atol=1e-6):
                bad += 1
                log(f"  !! {name} {r.block_start_utc} {c}: view={r[c]} pandas={ref}")
    log(f"  spot check {name}: {len(sample)} blocks x {len(cols)} cols, mismatches={bad}")
    if bad:
        raise AssertionError(f"spot check failed for {name}")


def role(c: str, ab_cols: set[str], plan: dict) -> str:
    if c in BLOCK_KEY or c in ("block_end_utc", "block_start_local", "duration_h",
                               "partial_in_sample"):
        return "key"
    if c.startswith("regime_"):
        return "regime"
    if c.startswith("flag_"):
        return "flag"
    if c in ("n_qh", "n_tenders", "auction_id", "currency"):
        return "meta"
    if "_xchk_" in c:
        return "xchk"
    if c in ab_cols:
        return "target"
    if c in plan:
        return "feature"
    return "other"


# ------------------------------------------------------------------ main
def main(write: bool) -> None:
    ab_raw = pd.read_parquet(AB)
    ab = targets.combine_tenders(ab_raw)
    for c in IV:
        ab[c] = ab[c].astype("datetime64[ns, UTC]")
    ab_cols = set(ab.columns)
    ch = ab[ab["market"] == TARGET_MARKET].copy()

    h("1. Targets")
    log(f"auction_blocks {len(ab_raw):,} -> combine_tenders {len(ab):,}; {TARGET_MARKET}: {len(ch):,}")
    t = ch.assign(year=ch.block_start_utc.dt.year).pivot_table(
        index=["product", "procurement"], columns="year", values="direction",
        aggfunc="size", fill_value=0)
    log(t.to_string())

    h("2. Aggregation plan")
    plan, weights, excluded, masks = build_plan(ab_cols)
    log(pd.Series(plan).value_counts().to_string())
    log(f"\nvolume weights: {weights}")
    log(f"\nTRE masks (5.4): {len(masks)} columns -> NULL where flagged: "
        f"{sorted(set(masks.values()))}")
    log(f"\nexcluded ({len(excluded)}):")
    for c, r in excluded.items():
        if c not in TIME_KEYS:
            log(f"  {c:45s} {r}")
    dd = pd.read_csv(DD)
    wx = dd[(dd.table == "master_15min") & dd.source.str.contains("MeteoSwiss", na=False)]
    log("\nweather rules (check against 5.2: mean for levels, sum for amounts / degree-hours):")
    log(wx[["column", "unit", "aggregation_rule"]].to_string(index=False))

    h("3. Aggregate")
    iv = ch[IV].drop_duplicates().reset_index(drop=True)
    feat, cov = aggregate(iv, plan, weights, masks)
    log(f"distinct delivery windows: {len(iv):,}; with master rows: {len(feat):,}")

    clash = (set(feat.columns) - set(IV)) & ab_cols
    if clash:
        raise ValueError(f"feature names clash with auction_blocks: {sorted(clash)}")

    h("4. Views")
    out, dict_rows = {}, []
    ddi = dd.set_index(["table", "column"])
    for prod, name in VIEWS.items():
        tgt = ch[ch["product"] == prod]
        dropped = [c for c in tgt.columns if tgt[c].isna().all()]
        tgt = tgt.drop(columns=dropped)
        view = tgt.merge(feat, on=IV, how="left", validate="many_to_one")
        view["n_qh"] = view["n_qh"].fillna(0).astype(int)
        keys = [c for c in BLOCK_KEY if c in view.columns]
        for c in IV + ["partial_in_sample", "duration_h"]:
            if c not in keys:
                keys.append(c)
        cv = tgt[keys].merge(cov, on=IV, how="left", validate="many_to_one")

        # checks
        if view.duplicated(keys).any():
            raise AssertionError(f"{name}: duplicate block keys")
        exp = (view["duration_h"] * 4).round().astype(int)
        mism = view[(view["n_qh"] != exp) & ~view["partial_in_sample"]]
        if len(mism):
            log(mism[keys + ["n_qh"]].head(10).to_string())
            raise AssertionError(f"{name}: {len(mism)} full blocks with n_qh != duration_h*4")
        log(f"\n{name}: {len(view):,} blocks x {view.shape[1]} cols "
            f"({view.block_start_utc.min():%Y-%m-%d} -> {view.block_start_utc.max():%Y-%m-%d}); "
            f"currency {sorted(view['currency'].dropna().unique())}; "
            f"partial blocks {int(view['partial_in_sample'].sum())}")
        log(f"  dropped all-NaN auction_blocks columns: {dropped}")
        fcov = cv[list(plan)].mean()
        log(f"  feature coverage (mean share of non-NaN qh): median {fcov.median():.3f}, "
            f"features with 0 coverage: {int((fcov == 0).sum())}")
        spot_check(view, name)

        for c in view.columns:
            r = role(c, ab_cols, plan)
            src_tab = "master_15min" if r == "feature" else "auction_blocks"
            meta = ddi.loc[(src_tab, c)] if (src_tab, c) in ddi.index else None
            dict_rows.append({
                "table": name, "column": c, "role": r,
                "aggregation": plan.get(c, "") + (f" (weight {weights[c]})" if c in weights else ""),
                "unit": None if meta is None else meta["unit"],
                "source": None if meta is None else meta["source"],
                "availability_rule": None if meta is None else meta["availability_rule"],
                "notes": None if meta is None else meta["notes"],
            })
        out[name] = (view, cv)

    if not write:
        log("\nDry run: nothing written. Rerun with --write.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (view, cv) in out.items():
        view.to_parquet(OUT / f"{name}.parquet", index=False)
        cv.to_parquet(OUT / f"{name}_coverage.parquet", index=False)
    pd.DataFrame(dict_rows).to_csv(OUT / "_dictionary.csv", index=False)
    log(f"\nWritten to {OUT.relative_to(CLEAN.parent)}")
    (OUT / "build_views_report.txt").write_text("\n".join(_report) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
