"""
frequency_merge.py — cross-check the TSO archive against Energy-Charts on the
overlap (May-Jun 2022) and build one merged frequency-feature series.

Inputs (both produced by existing scripts, identical feature columns):
    Frequency/Data/production/raw_1s/<YYYY>/freq_tso_<YYYY-MM>.parquet     (TSO, 1-s)
    Frequency/Data/production/frequency_tso_{15min,4h}.parquet
    EnergyCharts/Data/frequency/raw/<YYYY>/freq_<a>_<b>.parquet            (Energy-Charts, 1-s)
    EnergyCharts/Data/frequency/frequency_{15min,4h}.parquet

Step 1  compare  per overlap day at 1 s: common seconds, correlation, mean difference
                 (Energy-Charts minus TSO, mHz), MAE, 99th percentile |diff|, best clock
                 lag in -10..+10 s; at 15 min: correlation of mean/std deviation.
                 -> Frequency/Data/compare/compare_daily.csv, compare_report.txt
Step 2  merge    one row per interval from the sources in --sources priority order
                 (default tso,energycharts,zenodo,zenodo2021 — Zenodo only fills gaps); a source
                 whose interval is < 90 % covered loses to one with >= 90 %. Adds
                 `source` and `coverage` (= n_seconds / interval length; DST blocks are
                 3 h / 5 h). Trimmed to --start/--end (default end 2026-08-31 = cut-off).
                 Missing intervals -> frequency_missing_15min.csv, frequency_missing_days.csv
                 -> Frequency/Data/production/frequency_merged_15min.parquet
                    Frequency/Data/production/frequency_merged_4h.parquet

Run from the thesis root (venv active):
    python Frequency/frequency_merge.py                    # compare + merge
    python Frequency/frequency_merge.py --compare-only
    python Frequency/frequency_merge.py --merge-only --sources energycharts,tso,zenodo

Writes only below Frequency/Data/ (never deletes).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TSO = Path("Frequency") / "Data" / "production"
EC = Path("EnergyCharts") / "Data" / "frequency"
CMP = Path("Frequency") / "Data" / "compare"
LOCAL_TZ = "Europe/Zurich"
MAX_LAG_S = 10


def log(msg: str = "", buf: list[str] | None = None) -> None:
    print(msg, flush=True)
    if buf is not None:
        buf.append(msg)


def read_1s(files: list[Path]) -> pd.Series:
    if not files:
        return pd.Series(dtype="float64")
    s = pd.concat([pd.read_parquet(p).iloc[:, 0] for p in files]).astype("float64")
    s = s.dropna()                                   # Energy-Charts has NaN seconds
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


# ----------------------------------------------------------------------- compare

def compare() -> None:
    buf: list[str] = []
    CMP.mkdir(parents=True, exist_ok=True)
    tso = read_1s(sorted((TSO / "raw_1s").glob("*/freq_tso_*.parquet")))
    if tso.empty:
        log("COMPARE: no TSO 1-s data — run frequency_tso_pull.py first")
        return
    lo, hi = tso.index.min(), tso.index.max()
    ec_files = []
    for p in sorted(EC.glob("raw/*/freq_*.parquet")):
        a, b = p.stem.split("_")[1:3]
        if pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=2) >= lo and pd.Timestamp(a, tz="UTC") - pd.Timedelta(days=1) <= hi:
            ec_files.append(p)
    ec = read_1s(ec_files)
    common = tso.index.intersection(ec.index)
    log("=" * 78, buf)
    log(f"TSO archive vs Energy-Charts — {datetime.now():%Y-%m-%d %H:%M}", buf)
    log("=" * 78, buf)
    log(f"TSO 1-s span:  {lo} -> {hi} ({len(tso):,} s)", buf)
    log(f"Energy-Charts files in window: {len(ec_files)}, {len(ec):,} s", buf)
    log(f"Common seconds: {len(common):,}", buf)
    if len(common) == 0:
        log("No overlap — nothing to compare.", buf)
        (CMP / "compare_report.txt").write_text("\n".join(buf) + "\n", encoding="utf-8")
        return

    t, e = tso.loc[common], ec.loc[common]
    d_mhz = (e - t) * 1000
    day = common.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d")
    rows = []
    for dd in sorted(set(day)):
        m = day == dd
        if m.sum() < 3600:            # e.g. the lone 00:00:00 second of the day after the archive ends
            continue
        tt, ee = t[m], e[m]
        # clock lag: correlation of 1-s changes (insensitive to a constant offset)
        dt = tt.diff()
        lags = {}
        for lag in range(-MAX_LAG_S, MAX_LAG_S + 1):
            de = ec.reindex(tt.index + pd.Timedelta(seconds=lag)).diff().values
            ok = ~(np.isnan(de) | np.isnan(dt.values))
            if ok.sum() > 1000:
                lags[lag] = float(np.corrcoef(dt.values[ok], de[ok])[0, 1])
        best = max(lags, key=lags.get) if lags else np.nan
        sh = ec.reindex(tt.index + pd.Timedelta(seconds=int(best))).values if lags else np.full(len(tt), np.nan)
        ok = ~np.isnan(sh)
        mae_best = float(np.mean(np.abs(sh[ok] - tt.values[ok])) * 1000) if ok.any() else np.nan
        rows.append(dict(
            day=dd, common_s=int(m.sum()),
            corr=float(np.corrcoef(tt, ee)[0, 1]) if m.sum() > 2 else np.nan,
            mean_diff_mhz=float(d_mhz[m].mean()), mae_mhz=float(d_mhz[m].abs().mean()),
            p99_abs_diff_mhz=float(d_mhz[m].abs().quantile(0.99)),
            best_lag_s=best, mae_at_best_lag_mhz=mae_best, diff_corr_at_best_lag=lags.get(best, np.nan),
            diff_corr_at_lag0=lags.get(0, np.nan)))
    daily = pd.DataFrame(rows)
    daily.to_csv(CMP / "compare_daily.csv", index=False)

    log("\n1-s level (Energy-Charts minus TSO):", buf)
    log(f"  correlation          {np.corrcoef(t, e)[0, 1]:.5f}", buf)
    log(f"  mean difference      {d_mhz.mean():+.3f} mHz", buf)
    log(f"  MAE                  {d_mhz.abs().mean():.3f} mHz", buf)
    log(f"  99th pct |diff|      {d_mhz.abs().quantile(0.99):.3f} mHz", buf)
    log(f"  best lag per day     {daily['best_lag_s'].value_counts().to_dict()}  "
        f"(s; negative = TSO timestamps trail Energy-Charts)", buf)
    log(f"  MAE at best lag      {daily['mae_at_best_lag_mhz'].mean():.3f} mHz (mean of days)", buf)
    log(f"  days compared        {len(daily)}  ({daily['day'].min()} -> {daily['day'].max()})", buf)
    worst = daily.sort_values("mae_mhz", ascending=False).head(5)
    log("  worst days by MAE:", buf)
    for _, r in worst.iterrows():
        log(f"    {r['day']}: MAE {r['mae_mhz']:.2f} mHz, corr {r['corr']:.4f}, "
            f"{r['common_s']:,} s, best lag {r['best_lag_s']}", buf)

    q_t = _read_agg(TSO / "frequency_tso_15min.parquet")
    q_e = _read_agg(EC / "frequency_15min.parquet")
    if q_t is not None and q_e is not None:
        idx = q_t.index.intersection(q_e.index)
        idx = idx[(q_t.loc[idx, "n_seconds"] > 800) & (q_e.loc[idx, "n_seconds"] > 800)]
        log(f"\n15-min level ({len(idx):,} intervals with > 800 s in both):", buf)
        for c in ("mean_df_mhz", "std_df_mhz", "mean_abs_df_mhz", "share_outside_50mhz",
                  "fcr_up_equiv_h", "fcr_down_equiv_h"):
            if c in q_t and c in q_e and len(idx) > 2:
                a, b = q_t.loc[idx, c], q_e.loc[idx, c]
                log(f"  {c:<22} corr {np.corrcoef(a, b)[0, 1]:.4f}   mean diff {float((b - a).mean()):+.5f}", buf)
    log("\nRule of thumb: corr > 0.99 and |mean diff| < 1 mHz -> the two series can be spliced.", buf)
    (CMP / "compare_report.txt").write_text("\n".join(buf) + "\n", encoding="utf-8")
    print(f"\nWritten: {CMP / 'compare_report.txt'}, {CMP / 'compare_daily.csv'}")


def _read_agg(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df[~df.index.duplicated(keep="first")].sort_index()


# ------------------------------------------------------------------------- merge

def expected_seconds(idx: pd.DatetimeIndex, res: str) -> np.ndarray:
    hours = 0.25 if res == "15min" else 4
    end = (idx.tz_localize(None) + pd.Timedelta(hours=hours)).tz_localize(
        LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
    sec = (end - idx).total_seconds().values
    if res == "15min":
        return np.full(len(idx), 900.0)
    return np.where(np.isnan(sec), hours * 3600, sec)


SOURCES = {
    "tso": lambda res: TSO / f"frequency_tso_{res}.parquet",
    "energycharts": lambda res: EC / f"frequency_{res}.parquet",
    "zenodo": lambda res: TSO / "zenodo" / f"frequency_zenodo_{res}.parquet",
    "zenodo2021": lambda res: TSO / "zenodo_5105820" / f"frequency_zenodo_{res}.parquet",
}


def merge(order: list[str], start: str | None, end: str | None,
          zenodo_until: str | None = "2022-06-30") -> None:
    """Priority = position in `order`; a source whose interval is < 90 % covered loses
    to any source with >= 90 % coverage. Trimmed to [start, end] (local dates, inclusive).
    Also writes the missing 15-min intervals and a per-day gap summary."""
    for res in ("15min", "4h"):
        parts = []
        for name in order:
            df = _read_agg(SOURCES[name](res))
            if df is not None and name.startswith("zenodo") and zenodo_until:
                # after Jun 2022 Zenodo comes from other files (SG HoBA / TransnetBW website)
                # whose 15-min features deviate from Energy-Charts (std +10 %, max |df| +26 %,
                # share > 50 mHz +9 %; zenodo_verify.txt 2026-10-01) -> not used as filler
                df = df[df.index < (pd.Timestamp(zenodo_until) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)]
            if df is not None:
                parts.append(df.assign(source=name))
        if not parts:
            log(f"MERGE {res}: no input")
            continue
        both = pd.concat(parts)
        if start:
            both = both[both.index >= pd.Timestamp(start).tz_localize(LOCAL_TZ)]
        if end:
            both = both[both.index < (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)]
        both["coverage"] = both["n_seconds"] / expected_seconds(both.index, res)
        prio = both["source"].map({n: i for i, n in enumerate(order)})
        both["_rank"] = 10 * (both["coverage"] < 0.9).astype(int) + prio
        both = both.sort_values("_rank", kind="stable")
        merged = both[~both.index.duplicated(keep="first")].drop(columns="_rank").sort_index()
        out = TSO / f"frequency_merged_{res}.parquet"
        TSO.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp)
        tmp.replace(out)
        log(f"MERGE {res}: {len(merged):,} rows, {merged.index.min()} -> {merged.index.max()}  -> {out}")
        src = merged.groupby(merged.index.strftime("%Y"))["source"].value_counts().unstack(fill_value=0)
        log(f"  rows per year and source:\n{src.to_string()}")
        if res != "15min":
            continue
        low = merged[merged["coverage"] < 0.9]
        log(f"  intervals with coverage < 90 %: {len(low):,} ({len(low) / len(merged):.2%})")
        lo = pd.Timestamp(start).tz_localize(LOCAL_TZ) if start else merged.index.min()
        hi = ((pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ) - pd.Timedelta(minutes=15)
              if end else merged.index.max())
        full = pd.date_range(lo, hi, freq="15min")
        miss = full.difference(merged.index)
        log(f"  missing 15-min intervals in {lo:%Y-%m-%d} -> {hi:%Y-%m-%d}: {len(miss):,} "
            f"({len(miss) / len(full):.2%})")
        gaps = pd.DataFrame({"interval_start_local": miss})
        gaps.to_csv(TSO / "frequency_missing_15min.csv", index=False)
        if len(miss):
            day = miss.strftime("%Y-%m-%d")
            per_day = pd.Series(1, index=miss).groupby(day).sum().rename("missing_15min")
            per_day.index.name = "date_local"
            per_day.to_csv(TSO / "frequency_missing_days.csv")
            m = per_day.groupby(per_day.index.str[:7]).sum()
            log("  missing per month (top 12): " + ", ".join(
                f"{k}: {v}" for k, v in m.sort_values(ascending=False).head(12).items()))
            log(f"  -> {TSO / 'frequency_missing_15min.csv'}, {TSO / 'frequency_missing_days.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--sources", default="tso,energycharts,zenodo,zenodo2021",
                    help="priority order, comma-separated (missing inputs are skipped)")
    ap.add_argument("--start", default=None, help="first local date kept (default: first data)")
    ap.add_argument("--end", default="2026-08-31", help="last local date kept (data cut-off)")
    ap.add_argument("--zenodo-until", default="2022-06-30",
                    help="last local date Zenodo may fill (its later data fails the 15-min check); "
                         "'' = no limit")
    args = ap.parse_args()
    if not args.merge_only:
        compare()
    if not args.compare_only:
        order = [x.strip() for x in args.sources.split(",") if x.strip()]
        bad = [x for x in order if x not in SOURCES]
        if bad:
            ap.error(f"unknown source(s): {bad}; choose from {list(SOURCES)}")
        merge(order, args.start, args.end, args.zenodo_until or None)


if __name__ == "__main__":
    main()
