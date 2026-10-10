"""RQ3c (todo 9.4): how did the 3 March 2025 aFRR-energy price cap (ElCom, 1,000 EUR/MWh on mandatory
bids) change the frequency, duration and severity of imbalance-price spikes?

Design (decided 3 Oct / 8 Oct 2026, D3):
  * FIXED thresholds (a rolling threshold adapts after the cap and hides the effect): short = q99 of the
    short price, long = q01 of the long price. Two threshold sources, both held constant over pre and post:
      pre12m   = the 12 months before the cap (the plan; in-sample for the pre window, so pre share = 1 % by construction)
      prior12m = the 12 months before that (out of sample for pre and post; the cleaner comparison)
  * Windows: pre 3 Mar 2024 - 2 Mar 2025, post 3 Mar 2025 - 2 Mar 2026 (12 + 12 months, same seasons).
    Post variants: 'all' and 'dual' (post rows with dual pricing only: excludes Jan-Feb 2026 single price,
    control for the pricing change; Jul-Dec 2025 AEP was informational only).
  * Placebo: the same procedure with the cap date one year earlier (3 Mar 2024): any 'effect' there is drift.
  * Metrics per side: share of QH above threshold, episodes per week (consecutive exceedance QH), mean
    episode length (QH), mean excess over threshold (EUR/MWh), p99 (short) / p01 (long) of the price, extreme
    value. Ratios post / pre with 95 % CI from a weekly block bootstrap (1,000 draws).
  * Mechanism check: monthly share of QH with aFRR-up activation price > 1,000 EUR/MWh (and aFRR-down < -1,000).

Run from Master_Thesis with .venv active:
    python Modelling/rq3c_1_pricecap.py             # -> Modelling/Output/rq3c_1/
Read-only on Clean/Data; writes only Modelling/Output/rq3c_1/.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Clean"))
import views as V  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = ROOT / "Modelling" / "Output" / "rq3c_1"
MASTER = ROOT / "Clean" / "Data" / "master" / "master_15min.parquet"
TZ = V.TZ
CAP = pd.Timestamp("2025-03-03")        # REGIME_DATES['afrr_energy_cap'] (ElCom, 3 Mar 2025, KW 10)
B = 1000
SEED = 20261010
PRICE = {"short": "tgt_price_short_eur_mwh", "long": "tgt_price_long_eur_mwh"}
lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


def load() -> pd.DataFrame:
    df = V.load_rq3(columns=["ts_utc", "tgt_price_short_eur_mwh", "tgt_price_long_eur_mwh", "regime_imb_pricing"])
    m = pd.read_parquet(MASTER, columns=["ts_utc", "ch_afrr_up_act_price_eur_mwh", "ch_afrr_down_act_price_eur_mwh"])
    n = len(df)
    df = df.merge(m, on="ts_utc", how="left", validate="one_to_one")
    assert len(df) == n
    t = pd.to_datetime(df["ts_utc"])
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t
    df["date"] = t.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    df["pricing"] = df["regime_imb_pricing"].astype(str)
    return df.sort_values("ts_utc").reset_index(drop=True)


def window(df, start, end):
    return df[(df["date"] >= start) & (df["date"] <= end)]


def thresholds(df, start, end):
    w = window(df, start, end)
    return {"short": w[PRICE["short"]].quantile(0.99), "long": w[PRICE["long"]].quantile(0.01)}


def weekly(w: pd.DataFrame, side: str, thr: float, t0: pd.Timestamp):
    """Weekly aggregates + raw price arrays. Exceedance flag on the QH grid; episodes = consecutive flags."""
    p = w[PRICE[side]].to_numpy(float)
    ok = ~np.isnan(p)
    sgn = 1.0 if side == "short" else -1.0
    flag = ok & (sgn * (p - thr) > 0)
    excess = np.where(flag, sgn * (p - thr), 0.0)
    prev = np.r_[False, flag[:-1]]
    start = flag & ~prev
    wk = ((w["date"] - t0).dt.days // 7).to_numpy()
    ep_id = np.cumsum(start) * flag
    # episode length attributed to the week of its start
    lens = pd.Series(flag.astype(int)).groupby(ep_id).transform("sum").to_numpy()
    d = pd.DataFrame({"wk": wk, "n": ok.astype(int), "exc": flag.astype(int), "excess": excess,
                      "ep": start.astype(int), "len": np.where(start, lens, 0)})
    wkly = d.groupby("wk").sum()
    raw = [p[(wk == k) & ok] for k in wkly.index]
    return wkly, raw, p[ok]


def stats(wkly: pd.DataFrame, raw_all: np.ndarray, side: str, idx: np.ndarray | None = None, raw=None) -> dict:
    a = wkly if idx is None else wkly.iloc[idx]
    n, exc, ep = a["n"].sum(), a["exc"].sum(), a["ep"].sum()
    q = 99 if side == "short" else 1
    vals = raw_all if idx is None else np.concatenate([raw[i] for i in idx])
    return {"share_qh": exc / n, "episodes_per_week": ep / len(a), "mean_len_qh": a["len"].sum() / max(ep, 1),
            "mean_excess": a["excess"].sum() / max(exc, 1), "pctl_price": np.percentile(vals, q),
            "extreme": vals.max() if side == "short" else vals.min()}


def boot(w_pre, raw_pre, all_pre, w_post, raw_post, all_post, side, rng):
    base_pre, base_post = stats(w_pre, all_pre, side), stats(w_post, all_post, side)
    reps = []
    for _ in range(B):
        i = rng.integers(0, len(w_pre), len(w_pre))
        j = rng.integers(0, len(w_post), len(w_post))
        s0 = stats(w_pre, all_pre, side, i, raw_pre)
        s1 = stats(w_post, all_post, side, j, raw_post)
        reps.append({k: (s1[k] / s0[k] if k in ("share_qh", "episodes_per_week", "mean_len_qh", "mean_excess")
                         else s1[k] - s0[k]) for k in s0 if k != "extreme"})
    r = pd.DataFrame(reps)
    out = []
    for k in r.columns:
        est = base_post[k] / base_pre[k] if k != "pctl_price" else base_post[k] - base_pre[k]
        out.append({"metric": k, "pre": base_pre[k], "post": base_post[k],
                    "effect": est, "effect_type": "ratio post/pre" if k != "pctl_price" else "difference",
                    "ci_lo": r[k].quantile(0.025), "ci_hi": r[k].quantile(0.975)})
    out.append({"metric": "extreme", "pre": base_pre["extreme"], "post": base_post["extreme"]})
    return pd.DataFrame(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    log(f"RQ3c: {len(df):,} quarter-hours; cap date {CAP.date()}")
    rng = np.random.default_rng(SEED)
    one_y = pd.DateOffset(years=1)
    rows = []
    for design, cap in (("real", CAP), ("placebo", CAP - one_y)):
        pre0, pre1 = cap - one_y, cap - pd.Timedelta(days=1)
        post0, post1 = cap, cap + one_y - pd.Timedelta(days=1)
        sources = {"pre12m": (pre0, pre1), "prior12m": (pre0 - one_y, pre0 - pd.Timedelta(days=1))}
        variants = ["all", "dual"] if design == "real" else ["all"]
        for src, (t0, t1) in sources.items():
            thr = thresholds(df, t0, t1)
            log(f"\n=== {design}: cap {cap.date()}, thresholds from {src} ({t0.date()} - {t1.date()}): "
                f"short q99 = {thr['short']:.1f}, long q01 = {thr['long']:.1f} EUR/MWh")
            for var in variants:
                wpre = window(df, pre0, pre1)
                wpost = window(df, post0, post1)
                if var == "dual":
                    wpost = wpost[wpost["pricing"] == "dual"]
                log(f"  -- post variant {var}: pre {len(wpre):,} QH, post {len(wpost):,} QH")
                for side in ("short", "long"):
                    wk_pre, raw_pre, all_pre = weekly(wpre, side, thr[side], pre0)
                    wk_post, raw_post, all_post = weekly(wpost, side, thr[side], pre0)
                    t = boot(wk_pre, raw_pre, all_pre, wk_post, raw_post, all_post, side, rng)
                    t.insert(0, "side", side)
                    t.insert(0, "variant", var)
                    t.insert(0, "thr_source", src)
                    t.insert(0, "design", design)
                    rows.append(t)
                    log(f"     {side}:")
                    log(t.drop(columns=["design", "thr_source", "variant", "side"]).set_index("metric")
                        .to_string(float_format=lambda x: f"{x:,.3f}"))
    res = pd.concat(rows)
    res.to_csv(OUT / "pricecap_effects.csv", index=False)

    # mechanism: monthly aFRR activation price beyond +-1,000
    d = df[df["date"] >= pd.Timestamp("2023-01-01")].copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    mech = d.groupby("month").apply(lambda g: pd.Series({
        "afrr_up_gt_1000_qh": int((g["ch_afrr_up_act_price_eur_mwh"] > 1000).sum()),
        "afrr_down_lt_m1000_qh": int((g["ch_afrr_down_act_price_eur_mwh"] < -1000).sum()),
        "short_gt_1000_qh": int((g[PRICE["short"]] > 1000).sum()),
        "long_lt_m1000_qh": int((g[PRICE["long"]] < -1000).sum()),
        "afrr_up_price_max": g["ch_afrr_up_act_price_eur_mwh"].max()}))
    mech.to_csv(OUT / "mechanism_monthly.csv")
    log("\n=== Mechanism: monthly count of QH beyond +-1,000 EUR/MWh (cap 3 Mar 2025)")
    log(mech.loc["2024-01":].to_string(float_format=lambda x: f"{x:,.0f}"))
    (OUT / "report_rq3c_1.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
