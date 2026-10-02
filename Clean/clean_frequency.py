"""Clean layer: Continental-Europe grid frequency (to-do 3.4, 2 Oct 2026).

Run from Master_Thesis with .venv active:
    python Clean/clean_frequency.py            # build, check, print summary
    python Clean/clean_frequency.py --write    # also write Clean/Data/frequency/

Input   Frequency/Data/production/frequency_merged_15min.parquet (1-s data from the
        TSO archive until 29 Jun 2022, Energy-Charts from 30 Jun 2022, Zenodo gap
        fillers; aggregated to 15 min by frequency_merge.py). Missing quarter-hours
        are absent from that file; here they become NaN rows on the full grid.
Output  Clean/Data/frequency/frequency.parquet + frequency_dictionary.csv

Rules
- Robust features only (finding F2): mean / mean-absolute deviation, shares outside
  ±10/50/100 mHz, FCR activation proxies. Peak features (max_abs_df, std_df,
  min/max_df) are source- and cleaning-sensitive and are left out.
- flag_ce_freq_implausible: 15-min mean |df| > 200 mHz. These are frozen 48.000 Hz
  placeholder stretches in the TSO archive (7 Apr - 24 Jun 2015, 19-20 Oct 2015,
  24-28 Jan 2017, 3 short ones in 2016) and isolated glitches; all frequency
  features are set to NaN there. The largest real event in the sample (system
  split 8 Jan 2021) has a 15-min mean of -65 mHz.
- flag_ce_freq_suspect: 100-200 mHz, values kept.
- flag_ce_freq_low_coverage: coverage < 0.9 (fewer than 90 % of seconds), kept.
- src_ce_freq: source of each quarter-hour (tso / energycharts / zenodo / zenodo2021).
- 4-h features for block products are aggregated from these 15-min values in the
  views (not taken from frequency_merged_4h.parquet), so flags apply consistently.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import config as C  # noqa: E402

IMPLAUSIBLE_MHZ, SUSPECT_MHZ, MIN_COVERAGE = 200.0, 100.0, 0.9
SRC = "TSO archive (netztransparenz, TransnetBW) / Energy-Charts (Fraunhofer ISE) / Zenodo, via frequency_merge.py"
AVAIL = "ex post (measured); for RQ2 use lagged values only"
MAP = {  # source column -> (clean name, unit, aggregation, note)
    "mean_df_mhz":          ("ce_freq_mean_df_mhz", "mHz", "mean", "mean deviation from 50 Hz"),
    "mean_abs_df_mhz":      ("ce_freq_mean_abs_df_mhz", "mHz", "mean", "mean absolute deviation from 50 Hz"),
    "share_outside_10mhz":  ("ce_freq_out_10mhz_share", "share", "mean", "share of seconds with |df| > 10 mHz"),
    "share_outside_50mhz":  ("ce_freq_out_50mhz_share", "share", "mean", "share of seconds with |df| > 50 mHz"),
    "share_outside_100mhz": ("ce_freq_out_100mhz_share", "share", "mean", "share of seconds with |df| > 100 mHz"),
    "fcr_up_equiv_h":       ("ce_fcr_up_equiv_h", "h", "sum", "FCR upward activation proxy: sum of max(-df,0)/200 mHz over seconds, in full-activation hours"),
    "fcr_down_equiv_h":     ("ce_fcr_down_equiv_h", "h", "sum", "FCR downward activation proxy, in full-activation hours"),
    "coverage":             ("ce_freq_coverage_share", "share", "min", "share of seconds with data"),
}


def build():
    f = pd.read_parquet(C.ROOT / "Frequency" / "Data" / "production" / "frequency_merged_15min.parquet")
    f.index = K.to_utc(f.index)
    f = f[~f.index.duplicated(keep="last")]
    g = K.grid().index
    f = f.reindex(g)
    out = pd.DataFrame(index=g)
    dic = []
    for src, (name, unit, agg, note) in MAP.items():
        out[name] = f[src].astype("float64")
        dic.append(dict(column=name, source=SRC, source_series=f"frequency_merged_15min.{src}", unit=unit,
                        resolution_native="15min (from 1-s data)", aggregation_rule=agg,
                        availability_rule=AVAIL, notes=note))
    absdf = f["mean_abs_df_mhz"]
    implaus = (absdf > IMPLAUSIBLE_MHZ).fillna(False)
    out.loc[implaus, list(out.columns)] = float("nan")
    out["src_ce_freq"] = f["source"].astype("string").where(~implaus)
    out["flag_ce_freq_implausible"] = implaus.astype(bool)
    out["flag_ce_freq_suspect"] = ((absdf > SUSPECT_MHZ) & (absdf <= IMPLAUSIBLE_MHZ)).fillna(False).astype(bool)
    out["flag_ce_freq_low_coverage"] = ((f["coverage"] < MIN_COVERAGE) & ~implaus).fillna(False).astype(bool)
    for col, note in [("src_ce_freq", "source of the 1-s data for this quarter-hour"),
                      ("flag_ce_freq_implausible", f"mean |df| > {IMPLAUSIBLE_MHZ:.0f} mHz (frozen 48 Hz stretches / glitches); all features set to NaN"),
                      ("flag_ce_freq_suspect", f"mean |df| {SUSPECT_MHZ:.0f}-{IMPLAUSIBLE_MHZ:.0f} mHz; values kept"),
                      ("flag_ce_freq_low_coverage", f"coverage < {MIN_COVERAGE}; values kept")]:
        dic.append(dict(column=col, source="derived", source_series="frequency_merged_15min", unit="-",
                        resolution_native="15min", aggregation_rule="any" if col.startswith("flag") else "mode",
                        availability_rule="-", notes=note))
    return f, out, dic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    f, out, dic = build()
    assert not K.check_names(out.columns), K.check_names(out.columns)
    assert len(out) == 409_052
    num = [c for c in out.columns if c.startswith("ce_")]
    s = out[num].describe().T[["count", "min", "50%", "max"]]
    s["nan_share"] = out[num].isna().mean().round(4)
    print(s.round(3).to_string())
    fl = {c: int(out[c].sum()) for c in out.columns if c.startswith("flag_")}
    imp = out.index[out.flag_ce_freq_implausible].tz_convert(K.TZ)
    print(f"\nflags: {fl}")
    print("implausible by year-month:", pd.Series(1, index=imp).groupby(imp.strftime('%Y-%m')).size().to_dict())
    print("source:", out.src_ce_freq.value_counts(dropna=False).to_dict())
    if a.write:
        p = K.write_clean(out, "frequency", "frequency", dic)
        print(f"written: {p} + frequency_dictionary.csv")


if __name__ == "__main__":
    main()
