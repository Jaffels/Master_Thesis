"""RQ4 (todo 9.6): revenue simulation of forecast-based bidding in the Swiss pay-as-bid capacity markets
(aFRR and mFRR, 4h blocks). Uses the strict ex-ante RQ2 forecasts (Modelling/Output/rq2_2_delta/preds_*.parquet).

Design (decided 10 Oct 2026):
  * Products: aFRR up/down 4h and mFRR up/down 4h (pay-as-bid). FCR is marginal-priced -> not simulated.
  * Bidder: price-taker with 1 MW, one bid price b per block (CHF per MW and hour of delivery).
    Accepted if b <= c, where c = price_bid_max = highest accepted bid of the block (ex post).
    Pay-as-bid: an accepted bid earns b (not c). Revenue per block = b * accepted (per MW and hour; x4 for the block).
    No market impact (1 MW), so the result is an upper-bound style, not a full market model.
  * Strategies (all use only information known at gate closure):
      oracle       b = c                       (best possible bid for a price-taker; upper bound)
      naive        b = same slot yesterday     (best RQ2 baseline)
      prev_auction b = previous auction result
      roll_med     b = median of the last 7 same-slot results
      fixed_1d/7d/30d  b = median of all results of the last 1 / 7 / 30 days (known with 24 h lag)
      lgbm         b = LightGBM forecast (RQ2, strict ex-ante)
    plus OFFSET-ADJUSTED variants (suffix _off) for naive, roll_med and lgbm: b = k * forecast, where k is
    re-chosen at the start of every month from a grid, maximising the revenue the same rule would have made
    over the 90 days before (only blocks known at that time). Bezold et al. (2025): the offset matters as much as the model.
  * Metrics: mean revenue per block, annualised (x 2,190 blocks per year), win rate, share of oracle revenue,
    value loss vs oracle, MAE of the bid against y (RQ2 target) and against c; per year; paired differences
    vs naive and vs naive_off with a moving-block bootstrap (block = 7 days, 2,000 draws, 95 % CI).
  * Link forecast error -> revenue: slope of (revenue gain vs naive) / (MAE reduction vs naive) per series.

Run from Master_Thesis with .venv active:
    python Modelling/rq4_1_revenue.py                      # main run, exante forecasts -> Modelling/Output/rq4_1/
    python Modelling/rq4_1_revenue.py --set perfect        # upper bound with perfect-forecast features -> rq4_1_perfect/
Read-only on Clean/Data and Modelling/Output/rq2_*; writes only into its output folder (--out to change).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "Clean" / "Data" / "master" / "auction_blocks.parquet"
TZ = "Europe/Zurich"
SERIES = ["aFRR|up|4h", "aFRR|down|4h", "mFRR|up|4h", "mFRR|down|4h"]
TARGET = {"aFRR": "price_bid_vwap", "mFRR": "price_settle_ch"}   # = RQ2 targets
K_GRID = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0])
LOOKBACK_D = 90
MIN_PAST = 150            # blocks needed to pick k, otherwise k = 1
BLOCKS_PER_YEAR = 6 * 365
BOOT_B = 2000
BOOT_BLOCK = 42           # 7 days of 4h blocks
STRATS = ["oracle", "naive", "prev_auction", "roll_med", "fixed_1d", "fixed_7d", "fixed_30d", "lgbm",
          "naive_off", "roll_med_off", "lgbm_off"]


def revenue(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Pay-as-bid revenue of a 1 MW bid b against the highest accepted bid c (per MW and hour)."""
    b = np.maximum(b, 0.0)
    return np.where(b <= c + 1e-9, b, 0.0)


def load(pred_set: str) -> pd.DataFrame:
    p = pd.read_parquet(ROOT / "Modelling" / "Output" / "rq2_2_delta" / f"preds_{pred_set}.parquet")
    p = p[p["series"].isin(SERIES)].copy()
    ab = pd.read_parquet(MASTER)
    ab = ab[(ab["market"] == "ch_swissgrid") & (ab["product"].isin(["aFRR", "mFRR"])) & (ab["procurement"] == "4h")]
    if ab.duplicated(["product", "direction", "block_start_utc"]).any():
        raise ValueError("duplicate 4h blocks in auction_blocks (tender_series?)")
    keep = ["product", "direction", "block_start_utc", "price_bid_max", "price_bid_vwap", "price_settle_ch",
            "offered_mw", "awarded_mw"]
    d = p.merge(ab[keep], on=["product", "direction", "block_start_utc"], how="left", suffixes=("", "_ab"))
    d = d.rename(columns={"price_bid_max": "c"})
    d["start"] = d["block_start_utc"].dt.tz_convert("UTC")
    # fixed-window bids from the full history of each series (results known 24 h after their slot)
    for s in SERIES:
        prod, direc, _ = s.split("|")
        full = ab[(ab["product"] == prod) & (ab["direction"] == direc)].sort_values("block_start_utc")
        full = full.dropna(subset=[TARGET[prod]])
        av = pd.Series(full[TARGET[prod]].to_numpy(), index=full["block_start_utc"].dt.tz_convert("UTC") + pd.Timedelta("24h"))
        idx = d.index[d["series"] == s]
        t = d.loc[idx, "start"]
        for w in (1, 7, 30):
            r = av.rolling(f"{w}D").median().rename("v").reset_index().rename(columns={"block_start_utc": "t"})
            r.columns = ["t", "v"]
            q = pd.DataFrame({"t": t.to_numpy(), "ix": idx}).sort_values("t")
            m = pd.merge_asof(q, r.sort_values("t"), on="t", direction="backward", tolerance=pd.Timedelta("4h"))
            d.loc[m["ix"].to_numpy(), f"fixed_{w}d"] = m["v"].to_numpy()
    d = d.rename(columns={"b_prev_slot": "naive", "b_prev_auction": "prev_auction", "b_roll_med": "roll_med", "pred": "lgbm"})
    d["oracle"] = d["c"]
    return d.sort_values(["series", "start"]).reset_index(drop=True)


def add_offset_bids(d: pd.DataFrame, base: str) -> pd.Series:
    """b = k * base; k re-chosen at the start of each local month on the 90 days before (known blocks only)."""
    out = pd.Series(np.nan, index=d.index)
    ks = []
    for s, g in d.groupby("series", sort=False):
        loc = g["start"].dt.tz_convert(TZ)
        months = loc.dt.strftime("%Y-%m")
        t_av = g["start"] + pd.Timedelta("24h")           # block t is known from t+24h on
        for mth in months.unique():
            mstart = pd.Timestamp(mth + "-01", tz=TZ).tz_convert("UTC")
            in_m = (months == mth).to_numpy().copy()
            past = np.array(((t_av <= mstart) & (g["start"] >= mstart - pd.Timedelta(days=LOOKBACK_D))).to_numpy()
                        & g[base].notna().to_numpy())
            k = 1.0
            if past.sum() >= MIN_PAST:
                bp, cp = g.loc[past, base].to_numpy(), g.loc[past, "c"].to_numpy()
                k = float(K_GRID[np.argmax([revenue(bp * kk, cp).mean() for kk in K_GRID])])
            out.loc[g.index[in_m]] = g.loc[in_m, base].to_numpy() * k
            ks.append((s, base, str(mth), k, int(past.sum())))
    add_offset_bids.log.extend(ks)
    return out


add_offset_bids.log = []


def moving_block_boot(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(x)
    nb = int(np.ceil(n / BOOT_BLOCK))
    starts = np.arange(max(n - BOOT_BLOCK, 1))
    means = np.empty(BOOT_B)
    for i in range(BOOT_B):
        st = rng.choice(starts, nb)
        means[i] = np.concatenate([x[s:s + BOOT_BLOCK] for s in st])[:n].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="exante", choices=["exante", "perfect", "exante_noweather", "perfect_noweather"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "Modelling" / "Output" / ("rq4_1" if a.set == "exante" else f"rq4_1_{a.set}")
    out.mkdir(parents=True, exist_ok=True)

    d = load(a.set)
    d = d.dropna(subset=["c", "naive", "roll_med", "lgbm"]).reset_index(drop=True)
    for base in ("naive", "roll_med", "lgbm"):
        d[f"{base}_off"] = add_offset_bids(d, base)
    d = d.dropna(subset=["fixed_30d"]).reset_index(drop=True)      # common rows for all strategies
    rows = {}
    for sname in STRATS:
        d[f"rev_{sname}"] = revenue(d[sname].to_numpy(), d["c"].to_numpy())
        d[f"win_{sname}"] = (np.maximum(d[sname], 0) <= d["c"] + 1e-9).astype(float)
    d["year"] = d["start"].dt.tz_convert(TZ).dt.year

    lines = [f"RQ4 revenue simulation  (forecast set: {a.set}; price-taker 1 MW; pay-as-bid; bid vs c = highest accepted bid)",
             f"common rows per series: {d.groupby('series').size().to_dict()}", ""]
    summ, by_year, diffs, slopes = [], [], [], []
    rng = np.random.default_rng(42)
    for s, g in d.groupby("series", sort=False):
        orc = g["rev_oracle"].sum()
        mae_naive = (g["naive"] - g["y"]).abs().mean()
        for sname in STRATS:
            r = g[f"rev_{sname}"]
            summ.append(dict(series=s, strategy=sname, n=len(g), rev_per_block=r.mean(),
                             rev_per_mw_year=r.mean() * BLOCKS_PER_YEAR * 4, win_rate=g[f"win_{sname}"].mean(),
                             share_of_oracle=r.sum() / orc, value_loss_per_block=g["rev_oracle"].mean() - r.mean(),
                             mae_vs_y=(g[sname] - g["y"]).abs().mean(), mae_vs_c=(g[sname] - g["c"]).abs().mean(),
                             mean_bid=g[sname].mean()))
            for yr, gy in g.groupby("year"):
                by_year.append(dict(series=s, year=yr, strategy=sname, n=len(gy), rev_per_block=gy[f"rev_{sname}"].mean(),
                                    share_of_oracle=gy[f"rev_{sname}"].sum() / gy["rev_oracle"].sum(), win_rate=gy[f"win_{sname}"].mean()))
        for sname in ["lgbm", "lgbm_off", "roll_med_off", "naive_off", "fixed_7d", "roll_med"]:
            for ref in ["naive", "naive_off"]:
                if sname == ref:
                    continue
                x = (g[f"rev_{sname}"] - g[f"rev_{ref}"]).to_numpy()
                lo, hi = moving_block_boot(x, rng)
                diffs.append(dict(series=s, strategy=sname, vs=ref, mean_diff_per_block=x.mean(), ci_lo=lo, ci_hi=hi,
                                  rel_diff_pct=100 * x.mean() / g[f"rev_{ref}"].mean(), significant=(lo > 0) or (hi < 0)))
        for sname in ["lgbm", "lgbm_off"]:
            dmae = mae_naive - (g["lgbm"] - g["y"]).abs().mean()
            drev = (g[f"rev_{sname}"] - g["rev_naive"]).mean() * BLOCKS_PER_YEAR * 4
            slopes.append(dict(series=s, strategy=sname, mae_reduction_vs_naive=dmae, gain_per_mw_year=drev,
                               gain_per_mw_year_per_unit_mae=drev / dmae if abs(dmae) > 1e-9 else np.nan))
    summ, by_year, diffs, slopes = map(pd.DataFrame, (summ, by_year, diffs, slopes))
    offs = pd.DataFrame(add_offset_bids.log, columns=["series", "base", "month", "k", "n_past"])
    summ.to_csv(out / "revenue_summary.csv", index=False)
    by_year.to_csv(out / "revenue_by_year.csv", index=False)
    diffs.to_csv(out / "revenue_diff_bootstrap.csv", index=False)
    slopes.to_csv(out / "mae_to_revenue.csv", index=False)
    offs.to_csv(out / "offsets_k.csv", index=False)
    d[["series", "start", "c", "y"] + STRATS + [f"rev_{x}" for x in STRATS]].to_parquet(out / "blocks.parquet")

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", lambda v: f"{v:,.3f}")
    for s in SERIES:
        t = summ[summ.series == s].set_index("strategy")[["rev_per_block", "rev_per_mw_year", "win_rate", "share_of_oracle", "mae_vs_y", "mae_vs_c", "mean_bid"]]
        lines += [f"== {s}  (n = {int(summ[summ.series == s].n.iloc[0]):,})", t.to_string(), ""]
        dd = diffs[diffs.series == s][["strategy", "vs", "mean_diff_per_block", "ci_lo", "ci_hi", "rel_diff_pct", "significant"]]
        lines += ["paired revenue difference per block (moving-block bootstrap, 95 % CI):", dd.to_string(index=False), ""]
        lines += ["offset k used (share of months): " + str(offs[(offs.series == s)].groupby("base")["k"].agg(["mean", "min", "max"]).round(2).to_dict("index")), ""]
    lines += ["== MAE reduction -> annual revenue (CHF per MW and year, hours x 4 h blocks)", slopes.to_string(index=False), ""]
    lines += ["== share of oracle revenue by year (selected strategies)",
              by_year[by_year.strategy.isin(["naive", "naive_off", "lgbm", "lgbm_off"])].pivot_table(
                  index=["series", "year"], columns="strategy", values="share_of_oracle").to_string(), ""]
    (out / "report_rq4_1.txt").write_text("\n".join(lines))
    print("\n".join(lines[:60]))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
