"""Capacity targets from master/auction_blocks.parquet (decided 2 Oct 2026).

Swissgrid weekly aFRR / mFRR blocks can have several tenders for the same delivery
week: tender_series 0 = main tender, 1-5 = additional tenders (_S1.._S5). Decision:
**combine the tenders of one block, weighted by awarded volume**, so every
(market, product, direction, procurement, block) has exactly one target row.

    from targets import combine_tenders
    ab = combine_tenders(pd.read_parquet(C.MASTER_DIR / "auction_blocks.parquet"))

Rules per combined block
- price_bid_vwap, price_settle_ch      weighted by awarded_mw (tenders with no award get weight 0);
                                       NaN if nothing was awarded in any tender. Weighting the
                                       per-tender VWAP by awarded MW = VWAP over all accepted bids.
- price_bid_min / price_bid_max / price_settle_max   min / max / max over the tenders
- n_bids, n_accepted, offered_mw, awarded_mw, awarded_ch_mw, cost, n_settle_prices   sums
- n_tenders (new), auction_id = ids joined with "+", tender_series dropped
- all other columns (block times, currency, regimes, cross-checks, regelleistung columns)
  are identical within a block and taken from the main tender
Blocks with a single tender (all regelleistung.net rows, all daily / 4h blocks) are unchanged.
The raw per-tender rows stay in auction_blocks.parquet.

Run  python Clean/targets.py   for a summary (reads only, writes nothing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

BLOCK_KEY = ["market", "product", "direction", "procurement", "block_start_utc", "block_end_utc"]
WEIGHTED = ["price_bid_vwap", "price_settle_ch"]
SUMS = ["n_bids", "n_accepted", "offered_mw", "awarded_mw", "awarded_ch_mw", "cost", "n_settle_prices"]
MINS = ["price_bid_min"]
MAXS = ["price_bid_max", "price_settle_max"]


def combine_tenders(ab: pd.DataFrame) -> pd.DataFrame:
    ab = ab.copy()
    ab["_tender"] = ab["tender_series"].fillna(0).astype(int)
    n = ab.groupby(BLOCK_KEY, sort=False)["_tender"].transform("size")
    single = ab[n == 1].drop(columns=["_tender", "tender_series"]).assign(n_tenders=1)
    multi = ab[n > 1].sort_values(BLOCK_KEY + ["_tender"])
    if multi.empty:
        return single
    g = multi.groupby(BLOCK_KEY, sort=False)
    base = g.first()                         # main tender values for all other columns
    w = multi["awarded_mw"].fillna(0.0)
    agg = {}
    for c in WEIGHTED:
        ok = multi[c].notna() & (w > 0)
        num = (multi[c].where(ok, 0.0) * w.where(ok, 0.0)).groupby([multi[k] for k in BLOCK_KEY], sort=False).sum()
        den = w.where(ok, 0.0).groupby([multi[k] for k in BLOCK_KEY], sort=False).sum()
        agg[c] = (num / den).where(den > 0)
    for c in SUMS:
        agg[c] = g[c].sum(min_count=1)
    for c in MINS:
        agg[c] = g[c].min()
    for c in MAXS:
        agg[c] = g[c].max()
    agg["auction_id"] = g["auction_id"].agg("+".join)
    agg["n_tenders"] = g.size()
    comb = base.drop(columns=["_tender", "tender_series"])
    for c, s in agg.items():
        comb[c] = s
    comb = comb.reset_index()
    out = pd.concat([single, comb[single.columns]], ignore_index=True)
    out = out.sort_values(BLOCK_KEY[:5]).reset_index(drop=True)
    if out.duplicated(BLOCK_KEY).any():
        raise ValueError("combine_tenders: duplicate blocks after combining")
    return out


if __name__ == "__main__":
    ab = pd.read_parquet(C.MASTER_DIR / "auction_blocks.parquet")
    out = combine_tenders(ab)
    print(f"auction_blocks: {len(ab):,} rows -> {len(out):,} blocks after combining tenders")
    m = out[out["n_tenders"] > 1]
    print(f"combined blocks: {len(m):,}")
    print(m.groupby(["product", "direction", "procurement"]).agg(blocks=("n_tenders", "size"),
          max_tenders=("n_tenders", "max"), first=("block_start_local", "min"), last=("block_start_local", "max")).to_string())
    # check: awarded volume and cost are preserved
    for c in ("awarded_mw", "cost"):
        print(f"{c}: total before {ab[c].sum():,.1f}  after {out[c].sum():,.1f}")
