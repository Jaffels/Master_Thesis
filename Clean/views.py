"""Load a modelling view with the RQ window (5.5) and the per-view gap handling (5.4).

Nothing is stored: windows and imputation are applied when a view is read, so the files
in Clean/Data/views/ stay NaN-faithful (Decision 5).

    import sys; sys.path.insert(0, "Clean")
    from views import load, feature_columns
    df = load("mfrr", kind="exante", rq="RQ2", impute=True)
    X = df[feature_columns(df, "mfrr", kind="exante")]

Windows (WINDOWS below; on the block's local delivery start, end = data cut-off):
  baseline  2021-01-01 ->        extended  2015-01-01 ->
  RQ1a      2021-01-01 ->        RQ1b      2015-01-01 ->   (structural breaks)
  RQ2       2015-01-01 ->, column `split`: train < 2021-01-01 <= test
  RQ3       2016-03-31 ->   (CH imbalance prices start; spike threshold is set in the
                             RQ3 target definition, relative to a rolling level)
Blocks cut by the sample edges (`partial_in_sample`) are dropped unless keep_partial=True.

Blocks without procurement (decided 3 Oct 2026): a bool column `procured` (awarded_mw > 0)
is added on load. Daily mFRR blocks where Swissgrid bought nothing have no price (5,474
blocks, 83-91 % of the 2024 daily mFRR blocks). procured_only=True (default) drops them for
price models; procured_only=False keeps them (e.g. for a procured yes / no model).

Gap handling (impute=True, for linear / RQ1b models; tree models: keep the default False):
  - only feature columns (not keys, targets, regimes, calendar, gate closure), and not
    columns whose NaN is structural (activation prices without activation, masked TRE
    activations: STRUCTURAL_NAN)
  - per series (product, direction, procurement, start hour) in time order
  - only interior NaN runs covering at most MAX_GAP (one day) from the first missing
    block start to the last missing block end -> linear interpolation in time
  - leading / trailing NaN (series not started / ended) and longer gaps stay NaN
  - a bool column <col>_imputed is added for every column with at least one imputed value
Weekly blocks (168 h) are never imputed by this rule.

Check from Master_Thesis with .venv active:
    python Clean/views.py            # row counts per view x RQ window, imputation summary
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path(__file__).resolve().parent
VIEWS_DIR = CLEAN / "Data" / "views"
TZ = "Europe/Zurich"
MAX_GAP = pd.Timedelta("24h")
RQ2_TEST_START = "2021-01-01"

WINDOWS = {
    "baseline": "2021-01-01",
    "extended": "2015-01-01",
    "RQ1a": "2021-01-01",
    "RQ1b": "2015-01-01",
    "RQ2": "2015-01-01",
    "RQ3": "2016-03-31",
}

# roles (views/_dictionary.csv, views/_exante_dictionary.csv) that are never features
NON_FEATURE = {"key", "meta", "target", "xchk", "gate_closure", "other"}
NOT_IMPUTED = NON_FEATURE | {"regime", "source", "calendar", "flag"}
SERIES = ["product", "direction", "procurement", "_slot"]
# NaN by construction, not a gap: activation prices without activation (and TRE values
# masked in 5.4) -> never imputed, also not their lookbacks
STRUCTURAL_NAN = re.compile(r"_act(_all)?_price|^ch_(pv)?tre_.*_act_")


def dictionary(view: str, kind: str) -> pd.DataFrame:
    f = "_dictionary.csv" if kind == "expost" else "_exante_dictionary.csv"
    d = pd.read_csv(VIEWS_DIR / f)
    return d[d["table"] == f"{view}_{kind}"].set_index("column")


def feature_columns(df: pd.DataFrame, view: str, kind: str = "exante",
                    perfect_forecast: bool = False) -> list[str]:
    """Feature columns present in df. perfect_forecast=False (default) excludes the
    __pf upper-bound columns; True uses __pf instead of the matching __dw columns."""
    d = dictionary(view, kind)
    cols = [c for c in df.columns if c in d.index and d.at[c, "role"] not in NON_FEATURE]
    if kind == "exante":
        if perfect_forecast:
            pf_base = {c.removesuffix("__pf") for c in cols if c.endswith("__pf")}
            cols = [c for c in cols if not (c.endswith("__dw") and c.removesuffix("__dw") in pf_base)]
        else:
            cols = [c for c in cols if d.at[c, "role"] != "perfect_forecast"]
    imputed = [c for c in df.columns if c.endswith("_imputed")]
    return cols + imputed


def apply_window(df: pd.DataFrame, rq: str | None, keep_partial: bool = False) -> pd.DataFrame:
    if not keep_partial and "partial_in_sample" in df:
        df = df[~df["partial_in_sample"].astype(bool)]
    if rq is None:
        return df
    if rq not in WINDOWS:
        raise KeyError(f"unknown window {rq!r}; choose from {sorted(WINDOWS)}")
    start = pd.Timestamp(WINDOWS[rq], tz=TZ).tz_convert("UTC")
    df = df[df["block_start_utc"] >= start].copy()
    if rq == "RQ2":
        test = pd.Timestamp(RQ2_TEST_START, tz=TZ).tz_convert("UTC")
        df["split"] = np.where(df["block_start_utc"] >= test, "test", "train")
    return df


def impute_single_days(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Linear interpolation (in time) of interior NaN runs spanning <= MAX_GAP, per series."""
    df = df.copy()
    loc = df["block_start_utc"].dt.tz_convert(TZ)
    df["_slot"] = np.where(df["procurement"] == "4h", loc.dt.hour, 0)
    df = df.sort_values(SERIES + ["block_start_utc"])
    counts, new = {}, {}
    for _, g in df.groupby(SERIES, sort=False, observed=True):
        t = g["block_start_utc"].dt.as_unit("ns").astype("int64").to_numpy().astype("float64")
        t_end = g["block_end_utc"].dt.as_unit("ns").astype("int64").to_numpy()
        for c in cols:
            x = g[c].to_numpy(dtype="float64")
            nan = np.isnan(x)
            if not nan.any() or nan.all():
                continue
            # NaN runs
            d = np.diff(np.concatenate([[0], nan.astype(np.int8), [0]]))
            starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)   # [s, e)
            fill = np.zeros(len(x), dtype=bool)
            for s, e in zip(starts, ends):
                if s == 0 or e == len(x):
                    continue                                              # leading / trailing
                span = t_end[e - 1] - int(t[s])
                if span <= MAX_GAP.value:
                    fill[s:e] = True
            if not fill.any():
                continue
            ok = ~nan
            vals = x.copy()
            vals[fill] = np.interp(t[fill], t[ok], x[ok])
            new.setdefault(c, []).append(pd.Series(vals, index=g.index))
            new.setdefault(f"{c}_imputed", []).append(pd.Series(fill, index=g.index))
            counts[c] = counts.get(c, 0) + int(fill.sum())
    flags = {}
    for c, parts in new.items():
        s = pd.concat(parts)
        if c.endswith("_imputed"):
            flags[c] = s.astype(bool).reindex(df.index, fill_value=False)
        else:
            df.loc[s.index, c] = s.astype(df[c].dtype)
    df = pd.concat([df.drop(columns="_slot"), pd.DataFrame(flags, index=df.index)],
                   axis=1).sort_index()
    return df, pd.Series(counts, dtype="int64").sort_values(ascending=False)


def load(view: str, kind: str = "exante", rq: str | None = None, impute: bool = False,
         columns: list[str] | None = None, keep_partial: bool = False,
         return_imputation_counts: bool = False, procured_only: bool = True):
    """view: 'fcr' | 'afrr' | 'mfrr'; kind: 'exante' | 'expost'."""
    need = None if columns is None else list(dict.fromkeys(
        [*columns, "awarded_mw", "block_start_utc", "partial_in_sample"]))
    df = pd.read_parquet(VIEWS_DIR / f"{view}_{kind}.parquet", columns=need)
    df["procured"] = df["awarded_mw"].fillna(0) > 0
    if procured_only:
        df = df[df["procured"]]
    df = apply_window(df, rq, keep_partial)
    counts = pd.Series(dtype="int64")
    if impute:
        d = dictionary(view, kind)
        cols = [c for c in df.columns if c in d.index and d.at[c, "role"] not in NOT_IMPUTED
                and pd.api.types.is_float_dtype(df[c])
                and not STRUCTURAL_NAN.search(c.split("__")[0])]
        df, counts = impute_single_days(df, cols)
    return (df, counts) if return_imputation_counts else df


def _check() -> None:
    rows = []
    for v in ["fcr", "afrr", "mfrr"]:
        for kind in ["expost", "exante"]:
            p = VIEWS_DIR / f"{v}_{kind}.parquet"
            if not p.exists():
                print(f"missing {p.name}")
                continue
            full = pd.read_parquet(p, columns=["block_start_utc", "procurement", "partial_in_sample"])
            for rq in WINDOWS:
                w = apply_window(full, rq)
                rows.append({"view": f"{v}_{kind}", "window": rq, "blocks": len(w),
                             "from": f"{w.block_start_utc.min():%Y-%m-%d}" if len(w) else "",
                             **({"test_blocks": int((w["split"] == "test").sum())} if rq == "RQ2" else {})})
    print(pd.DataFrame(rows).pivot_table(index="view", columns="window", values="blocks").to_string())
    print("\nImputation (RQ1b window), top columns by imputed blocks:")
    for v in ["fcr", "afrr", "mfrr"]:
        for kind in ["expost", "exante"]:
            if not (VIEWS_DIR / f"{v}_{kind}.parquet").exists():
                continue
            df, cnt = load(v, kind, rq="RQ1b", impute=True, return_imputation_counts=True)
            n_imp = sum(c.endswith("_imputed") for c in df.columns)
            print(f"  {v}_{kind}: {len(df):,} blocks; {len(cnt)} columns imputed, "
                  f"{int(cnt.sum()):,} values; {n_imp} _imputed flags")
            if len(cnt):
                print("    " + cnt.head(8).to_string().replace("\n", "\n    "))
            # sanity: imputed values only where the original was NaN, never leading/trailing
            raw = apply_window(pd.read_parquet(VIEWS_DIR / f"{v}_{kind}.parquet",
                                               columns=list(cnt.index[:20]) + [
                                                   "block_start_utc", "partial_in_sample"]), "RQ1b")
            for c in cnt.index[:20]:
                flag = df[f"{c}_imputed"].reindex(raw.index, fill_value=False)
                if raw.loc[flag.astype(bool), c].notna().any():
                    raise AssertionError(f"{v}_{kind} {c}: imputed over an existing value")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    _check()
    sys.exit(0)
