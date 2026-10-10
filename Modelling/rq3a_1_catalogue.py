"""RQ3a step 1 (todo 9.3): spike / episode catalogue, calendar patterns, hour-boundary test and
ex-post driver profiles.

Sides and targets (todo 5.8, D1): short = spike_short_q990 (main), long = spike_long_q010 (main),
robustness = spike_short_z30 (short only, no episodes). Unit = quarter-hour; calendar tests are
repeated at DAY level (days with >= 1 spike QH) because spikes cluster in time (RQ3 working note 4.2).

Parts
  A  catalogue: per year / imbalance-resolution regime / pricing regime / threshold: spike QH,
     share, episodes, median + max length, peak price, mean excess over threshold
  B  calendar: hour, weekday, month; day-level two-proportion z-tests for weekend, holiday, bridge day,
     Christmas period, DST switch day, school holiday (>= 50 % of CH), per side
  C  trigger vs duration (working note 4.3): minute-of-hour of spike QH and of episode ends,
     signed imbalance volume per minute-of-hour; direction check (sign of ch_imb_vol_mwh)
  D  ex-post driver profile: standardised mean difference spike vs non-spike for forecast errors
     (same QH, actual - DA forecast), activation, frequency, flows, DA prices, weather (perfect
     forecast). Each variable is first demeaned within hour-of-day x month (removes the solar-hour
     / season confounding). Windows: all (2016-03-31 ->) and qh (1 Jun 2022 ->, QH prices).

Run from Master_Thesis with .venv active:
    python Modelling/rq3a_1_catalogue.py            # -> Modelling/Output/rq3a_1/
Read-only on Clean/Data; writes only Modelling/Output/rq3a_1/.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Clean"))
import views as V  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = ROOT / "Modelling" / "Output" / "rq3a_1"
MASTER = ROOT / "Clean" / "Data" / "master" / "master_15min.parquet"
TZ = V.TZ
QH_START = pd.Timestamp("2022-06-01")           # first day of QH prices (regime_imb_resolution == qh)
ZONES = ["ch", "de_lu", "fr", "it_nord", "at"]

SIDES = {"short": ("spike_short_q990", "tgt_price_short_eur_mwh", "thr_short_q990"),
         "long": ("spike_long_q010", "tgt_price_long_eur_mwh", "thr_long_q010")}
THRESH = ["spike_short_q975", "spike_short_q990", "spike_short_q995", "spike_short_z30",
          "spike_long_q025", "spike_long_q010", "spike_long_q005"]

lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


def fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.3f}")


def ztest(k1, n1, k0, n0) -> tuple[float, float]:
    """Two-proportion z-test (pooled). Returns (z, two-sided p)."""
    if min(n1, n0) < 5:
        return float("nan"), float("nan")
    p = (k1 + k0) / (n1 + n0)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    if se == 0:
        return float("nan"), float("nan")
    z = (k1 / n1 - k0 / n0) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


# ------------------------------------------------------------------ load
def load() -> pd.DataFrame:
    view_cols = ["ts_utc", "ts_local", "tgt_price_short_eur_mwh", "tgt_price_long_eur_mwh", "tgt_imb_vol_mwh",
                 "tgt_system_imbalance_mw", "thr_short_q990", "thr_long_q010", *THRESH,
                 "episode_short_id", "episode_short_len_qh", "episode_long_id", "episode_long_len_qh",
                 "cal_hour", "cal_dow", "cal_month", "cal_year", "cal_holiday", "cal_bridge_day",
                 "cal_xmas_period", "cal_dst_switch_day", "regime_imb_resolution", "regime_imb_pricing"]
    pf = ["ch_temp_lw_degc__pf", "ch_ghi_lw_wm2__pf", "ch_ghi_ramp_lw_wm2__pf", "ch_wind_lw_ms__pf",
          "ch_precip_hydro_mm__pf", "ch_snow_hydro_cm__pf", "ch_melt_dh_hydro_kh__pf"]
    df = V.load_rq3(columns=view_cols + pf, school_holidays=True)
    import pyarrow.parquet as pq
    have = set(pq.read_schema(MASTER).names)
    act = ["ch_afrr_up_act_mw", "ch_afrr_down_act_mw", "ch_mfrr_up_act_all_mw", "ch_mfrr_down_act_all_mw",
           "ch_load_actual_mw", "ce_freq_mean_abs_df_mhz", "ch_de_flow_phys_mw", "de_ch_flow_phys_mw",
           "ch_fr_flow_phys_mw", "fr_ch_flow_phys_mw", "ch_it_nord_flow_phys_mw", "it_nord_ch_flow_phys_mw",
           "ch_at_flow_phys_mw", "at_ch_flow_phys_mw", *[f"{z}_price_da_eur_mwh" for z in ZONES]]
    pairs = {f"err_{z}_load_mw": (f"{z}_load_actual_mw", f"{z}_load_da_fc_mw") for z in ZONES}
    pairs.update({f"err_{z}_{g}_mw": (f"{z}_gen_{g}_mw", f"{z}_gen_{g}_da_fc_mw") for z in ZONES
                  for g in ("solar", "wind_on")})
    need = sorted({c for c in act if c in have} | {c for p in pairs.values() for c in p if c in have})
    m = pd.read_parquet(MASTER, columns=["ts_utc"] + need)
    n = len(df)
    df = df.merge(m, on="ts_utc", how="left", validate="one_to_one")
    assert len(df) == n
    for name, (a, f) in pairs.items():
        if a in df and f in df:
            df[name] = df[a].astype("float64") - df[f].astype("float64")
    df["ch_net_export_mw"] = (df.filter(regex=r"^ch_.*_flow_phys_mw$").sum(axis=1, min_count=1)
                              - df.filter(regex=r"^[a-z_]+_ch_flow_phys_mw$").sum(axis=1, min_count=1))
    loc = pd.to_datetime(df["ts_utc"]).dt.tz_localize("UTC") if pd.to_datetime(df["ts_utc"]).dt.tz is None \
        else pd.to_datetime(df["ts_utc"])
    loc = loc.dt.tz_convert(TZ)
    df["loc"] = loc
    df["date"] = loc.dt.tz_localize(None).dt.normalize()
    df["year"] = loc.dt.year
    df["minute"] = loc.dt.minute
    df["weekend"] = (loc.dt.dayofweek >= 5).astype(int)
    df["school_hol"] = (df["cal_school_holiday_share"].fillna(0) >= 0.5).astype(int)
    df["res"] = df["regime_imb_resolution"].astype(str)
    df["pricing"] = df["regime_imb_pricing"].astype(str)
    for _, (s, _, _) in SIDES.items():
        df[s] = df[s].fillna(0).astype(int)
    df["spike_short_z30"] = df["spike_short_z30"].fillna(0).astype(int)
    return df


# ------------------------------------------------------------------ A catalogue
def episode_table(df: pd.DataFrame, side: str) -> pd.DataFrame:
    spike, price, thr = SIDES[side]
    idc = f"episode_{side}_id"
    e = df[df[idc].notna()]
    g = e.groupby(idc)
    sgn = 1 if side == "short" else -1
    t = pd.DataFrame({
        "start": g["loc"].min(), "end": g["loc"].max(),
        "len_qh": g.size(),
        "peak": g[price].max() if side == "short" else g[price].min(),
        "excess_mean": g.apply(lambda x: (sgn * (x[price] - x[thr])).mean()),
        "res": g["res"].first(), "pricing": g["pricing"].first(),
    })
    t["year"] = t["start"].dt.year
    t["end_minute"] = t["end"].dt.minute
    t["start_minute"] = t["start"].dt.minute
    t["side"] = side
    return t.reset_index(drop=True)


def part_a(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    log("=" * 90)
    log("A. Catalogue (quarter-hours flagged, episodes, severity)")
    log("=" * 90)
    eps = {s: episode_table(df, s) for s in SIDES}
    pd.concat(eps.values()).to_csv(OUT / "episodes.csv", index=False)
    rows = []
    for by in ("year", "res", "pricing"):
        for side, (spike, price, thr) in SIDES.items():
            qh = df.groupby(by)[spike].agg(["sum", "mean"]).rename(columns={"sum": "spike_qh", "mean": "share"})
            e = eps[side].groupby(by).agg(episodes=("len_qh", "size"), len_median=("len_qh", "median"),
                                          len_max=("len_qh", "max"), peak=("peak", "max" if side == "short" else "min"),
                                          excess_mean=("excess_mean", "mean"))
            t = qh.join(e)
            t.insert(0, "side", side)
            t.insert(0, "by", by)
            t.index.name = "group"
            rows.append(t.reset_index())
    cat = pd.concat(rows)
    cat.to_csv(OUT / "catalogue.csv", index=False)
    for by in ("year", "res", "pricing"):
        log(f"\n-- by {by}")
        log(fmt(cat[cat["by"] == by].drop(columns="by").set_index(["side", "group"])))
    thr = pd.DataFrame({c: df.groupby("year")[c].mean() for c in THRESH})
    thr.to_csv(OUT / "threshold_share_by_year.csv")
    log("\n-- share of QH flagged, all thresholds (by year)")
    log(fmt(thr))
    # jaccard between sides' main + robustness (short)
    a, b = df["spike_short_q990"].astype(bool), df["spike_short_z30"].astype(bool)
    log(f"\nJaccard short q990 vs z30: {(a & b).sum() / max((a | b).sum(), 1):.3f}")
    return eps


# ------------------------------------------------------------------ B calendar
def part_b(df: pd.DataFrame) -> None:
    log("\n" + "=" * 90)
    log("B. Calendar patterns")
    log("=" * 90)
    wins = {"all": df, "qh": df[df["date"] >= QH_START]}
    res = []
    for wname, w in wins.items():
        for side, (spike, _, _) in SIDES.items():
            # hour / weekday / month: share of QH flagged vs base
            for var in ("cal_hour", "cal_dow", "cal_month"):
                t = w.groupby(var)[spike].mean()
                t = pd.DataFrame({"window": wname, "side": side, "var": var, "level": t.index, "share": t.values,
                                  "ratio_to_base": t.values / max(w[spike].mean(), 1e-12)})
                res.append(t)
            # day level: day has >= 1 spike QH
            d = w.groupby("date").agg(spike_day=(spike, "max"), weekend=("weekend", "max"), holiday=("cal_holiday", "max"),
                                      bridge=("cal_bridge_day", "max"), xmas=("cal_xmas_period", "max"),
                                      dst=("cal_dst_switch_day", "max"), school=("school_hol", "max"))
            for flag in ("weekend", "holiday", "bridge", "xmas", "dst", "school"):
                on, off = d[d[flag] == 1], d[d[flag] == 0]
                z, p = ztest(on["spike_day"].sum(), len(on), off["spike_day"].sum(), len(off))
                res.append(pd.DataFrame([{"window": wname, "side": side, "var": "day_flag", "level": flag,
                                          "n_days_on": len(on), "share_on": on["spike_day"].mean() if len(on) else np.nan,
                                          "share_off": off["spike_day"].mean(), "z": z, "p": p}]))
    cal = pd.concat(res)
    cal.to_csv(OUT / "calendar.csv", index=False)
    for wname in wins:
        log(f"\n-- day-level flag tests, window {wname} (share of days with >= 1 spike QH)")
        t = cal[(cal["window"] == wname) & (cal["var"] == "day_flag")].drop(columns=["window", "var"])
        log(fmt(t.set_index(["side", "level"])))
        for side in SIDES:
            t = cal[(cal["window"] == wname) & (cal["side"] == side) & (cal["var"] == "cal_hour")]
            top = t.sort_values("ratio_to_base", ascending=False).head(5)
            log(f"   {side}: hours with highest spike share: " + ", ".join(f"{int(r.level):02d}h x{r.ratio_to_base:.1f}" for r in top.itertuples()))


# ------------------------------------------------------------------ C trigger vs duration
def part_c(df: pd.DataFrame, eps: dict[str, pd.DataFrame]) -> None:
    log("\n" + "=" * 90)
    log("C. Trigger vs duration (hour boundary) and direction check; QH regime only")
    log("=" * 90)
    q = df[df["date"] >= QH_START]
    rows = []
    for side, (spike, _, _) in SIDES.items():
        s = q[q[spike] == 1]
        pos = s["minute"].value_counts(normalize=True).reindex([0, 15, 30, 45]).fillna(0)
        vol = (-1 if side == "short" else 1) * q.loc[q[spike] == 1].groupby("minute")["tgt_imb_vol_mwh"].mean()
        e = eps[side]
        e = e[e["start"].dt.tz_localize(None) >= QH_START]
        e_end = e["end_minute"].value_counts(normalize=True).reindex([0, 15, 30, 45]).fillna(0)
        e_start = e["start_minute"].value_counts(normalize=True).reindex([0, 15, 30, 45]).fillna(0)
        for mnt in (0, 15, 30, 45):
            rows.append({"side": side, "minute": mnt, "share_spike_qh": pos[mnt], "share_episode_starts": e_start[mnt],
                         "share_episode_ends": e_end[mnt], "mean_signed_imb_vol_mwh": vol.get(mnt, np.nan)})
        # test: episodes (len >= 2) ending at :45 vs 25 % under uniform ends
        e2 = e[e["len_qh"] >= 2]
        k, n = int((e2["end_minute"] == 45).sum()), len(e2)
        z = (k / n - 0.25) / math.sqrt(0.25 * 0.75 / n) if n >= 10 else float("nan")
        log(f"{side}: episodes (len>=2, QH regime) n={n}; ending at :45 = {k / max(n, 1):.3f} (uniform 0.25), z = {z:.2f}")
        # direction check: spike side should match the sign of the system imbalance volume
        v = q.loc[q[spike] == 1, "tgt_imb_vol_mwh"].dropna()
        ok = (v < 0).mean() if side == "short" else (v > 0).mean()
        log(f"{side}: share of spike QH whose ch_imb_vol_mwh has the expected sign = {ok:.3f} (n={len(v)})")
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "hour_boundary.csv", index=False)
    log("\n-- minute-of-hour of spike QH / episode starts / ends and mean signed imbalance volume (+ = in spike direction)")
    log(fmt(t.set_index(["side", "minute"])))


# ------------------------------------------------------------------ D profiles
def part_d(df: pd.DataFrame) -> None:
    log("\n" + "=" * 90)
    log("D. Ex-post driver profile: standardised mean difference spike vs no-spike")
    log("   (variables demeaned within year x month x hour-of-day; sd = sd of the demeaned no-spike values)")
    log("=" * 90)
    cand = [c for c in df.columns if c.startswith("err_")
            or c in ("ch_afrr_up_act_mw", "ch_afrr_down_act_mw", "ch_mfrr_up_act_all_mw", "ch_mfrr_down_act_all_mw",
                     "ch_load_actual_mw", "ce_freq_mean_abs_df_mhz", "ch_net_export_mw", "tgt_imb_vol_mwh",
                     *[f"{z}_price_da_eur_mwh" for z in ZONES])
            or c.endswith("__pf") or c == "cal_school_holiday_share"]
    out = []
    for wname, w in {"all": df, "qh": df[df["date"] >= QH_START]}.items():
        grp = [w["cal_year"], w["cal_month"], w["cal_hour"]]
        X = w[cand].astype("float64")
        Xd = X - X.groupby(grp).transform("mean")
        for side, (spike, _, _) in SIDES.items():
            m1 = w[spike] == 1
            for c in cand:
                x1, x0 = Xd.loc[m1, c].dropna(), Xd.loc[~m1, c].dropna()
                if len(x1) < 50 or len(x0) < 1000 or x0.std() == 0:
                    continue
                d = (x1.mean() - x0.mean()) / x0.std()
                se = math.sqrt(1 / len(x1) + d * d / (2 * len(x0)) + 1 / len(x0)) if len(x0) else float("nan")
                out.append({"window": wname, "side": side, "variable": c, "n_spike": len(x1), "std_diff": d, "se": se})
    prof = pd.DataFrame(out)
    prof["abs_d"] = prof["std_diff"].abs()
    prof.to_csv(OUT / "driver_profile.csv", index=False)
    for wname in ("qh", "all"):
        for side in SIDES:
            t = prof[(prof["window"] == wname) & (prof["side"] == side)].sort_values("abs_d", ascending=False).head(12)
            log(f"\n-- {side}, window {wname}: 12 largest standardised differences")
            log(fmt(t.set_index("variable")[["n_spike", "std_diff", "se"]]))
    # long vs short contrast: variables where signs differ
    pv = prof[prof["window"] == "qh"].pivot(index="variable", columns="side", values="std_diff").dropna()
    pv["diff_long_minus_short"] = pv["long"] - pv["short"]
    pv.reindex(pv["diff_long_minus_short"].abs().sort_values(ascending=False).index).head(15).to_csv(OUT / "driver_contrast_qh.csv")
    log("\n-- long vs short contrast (qh window): 15 largest |std_diff_long - std_diff_short|")
    log(fmt(pv.reindex(pv["diff_long_minus_short"].abs().sort_values(ascending=False).index).head(15)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    log(f"RQ3a step 1: {len(df):,} quarter-hours, {df['loc'].min()} -> {df['loc'].max()}")
    eps = part_a(df)
    part_b(df)
    part_c(df, eps)
    part_d(df)
    (OUT / "report_rq3a_1.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
