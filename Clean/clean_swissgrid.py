"""Clean layer: Swissgrid (to-do 3.5, 2 Oct 2026).

Run from Master_Thesis with .venv active:
    python Clean/clean_swissgrid.py            # build, check, print summary
    python Clean/clean_swissgrid.py --write    # also write Clean/Data/swissgrid/

Output (Clean/Data/swissgrid/), each with <name>_dictionary.csv
  energy_overview.parquet   2015 -> : system totals, physical cross-border flows,
                            aFRR / mFRR activation and prices (Energy Overview)
  imbalance.parquet         2023 -> : imbalance prices long / short / single (AEP)
  system_balance.parquet    2026 -> : control energy, control-area balance,
                            intraday NTC, commercial net flows, spot spreads
  tre.parquet               2023 -> : mFRR energy bids per quarter-hour (TRE)
  auction_blocks.parquet    long table, one row per capacity-auction delivery block
                            (targets; table design 5.1)

Conventions (table design 4, checks 2.3-2.5)
- Energy Overview kWh per quarter-hour -> average MW (x 4 / 1000).
- Down / import quantities are stored as positive magnitudes, the direction is in
  the name. Net quantities keep their sign: ch_*_sched_net_mw > 0 = export from CH;
  ch_system_imbalance_mw < 0 = control area short (confirmed by the AEP rule).
- Activation prices are NaN in quarter-hours without activation (source: 0).
- mFRR activation from the Energy Overview equals TRE in 2026 (all activations,
  incl. non-balancing ones, check 2.4) and is higher than TRE in 2024-25 (probably
  incl. RR): named ch_mfrr_*_act_all_mw, never use as balancing volume. aFRR from
  the Energy Overview equals the published system balance (2026: 99-100 %).
- TRE: bid-side features per product (offered MW, number of bids, min/max bid
  price) plus activation columns; flag_tre_extra_activation_up/down (2026) mark
  quarter-hours where TRE activation exceeds the published mFRR activation
  (table design 4.7); views mask TRE activation features there.
- Intraday NTC: 99999 MW (placeholder) -> NaN + flag_ntc_id_placeholder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import config as C  # noqa: E402

SG = C.ROOT / "SwissGrid"
SRC = "Swissgrid"
KWH_TO_MW = 4 / 1000


def read_ts(pattern: str, tcol: str = "timestamp") -> pd.DataFrame:
    fs = sorted(SG.glob(pattern))
    if not fs:
        raise FileNotFoundError(pattern)
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d.index = K.to_utc(d.pop(tcol))
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d


def on_grid(d: pd.DataFrame) -> pd.DataFrame:
    return d.reindex(K.grid().index)


def entry(col, series, unit, res, agg, avail, note=""):
    return dict(column=col, source=SRC, source_series=series, unit=unit, resolution_native=res,
                aggregation_rule=agg, availability_rule=avail, notes=note)


# ---------------------------------------------------------------- Energy Overview
def energy_overview():
    e = read_ts("EnergyOverview/Data/qh/*.parquet")
    e = e[e.index >= C.start_utc()]
    g = on_grid(e)
    out, dic = pd.DataFrame(index=g.index), []
    AV = "ex post (Energy Overview, published with a delay)"
    totals = {"end_user_consumption_kwh": ("ch_cons_enduse_mw", "end-user consumption, Swiss control block"),
              "production_kwh": ("ch_prod_mw", "total production, Swiss control block"),
              "consumption_kwh": ("ch_cons_mw", "total consumption, Swiss control block"),
              "net_outflow_tn_kwh": ("ch_tn_net_outflow_mw", "net outflow of the transmission grid"),
              "vertical_feedin_tn_kwh": ("ch_tn_vertical_feedin_mw", "vertical feed-in into the transmission grid"),
              "transit_kwh": ("ch_transit_mw", "transit"),
              "import_kwh": ("ch_import_mw", "import (physical)"),
              "export_kwh": ("ch_export_mw", "export (physical)")}
    for src, (name, note) in totals.items():
        out[name] = g[src] * KWH_TO_MW
        dic.append(entry(name, f"EnergieUebersichtCH.{src}", "MW", "15min", "mean", AV, note + "; kWh per QH -> MW"))
    for nb in ("at", "de", "fr", "it"):
        for a, b in (("ch", nb), (nb, "ch")):
            name = f"{a}_{b}_flow_phys_mw"
            out[name] = g[f"xb_{a}_{b}_kwh"] * KWH_TO_MW
            dic.append(entry(name, f"EnergieUebersichtCH.xb_{a}_{b}_kwh", "MW", "15min", "mean", AV,
                             f"physical cross-border flow {a.upper()}->{b.upper()} (IT = whole border)"))
    acts = {"afrr_pos_energy_kwh": ("ch_afrr_up_act_mw", 1, "activated aFRR up; equals system balance 2026"),
            "afrr_neg_energy_kwh": ("ch_afrr_down_act_mw", -1, "activated aFRR down (magnitude); equals system balance 2026"),
            "mfrr_pos_energy_kwh": ("ch_mfrr_up_act_all_mw", 1, "ALL activated mFRR up incl. non-balancing (2026 = TRE; 2024-25 > TRE, probably incl. RR); not a balancing volume"),
            "mfrr_neg_energy_kwh": ("ch_mfrr_down_act_all_mw", -1, "ALL activated mFRR down (magnitude), see up")}
    for src, (name, sign, note) in acts.items():
        out[name] = g[src] * KWH_TO_MW * sign
        dic.append(entry(name, f"EnergieUebersichtCH.{src}", "MW", "15min", "mean", AV, note))
    prices = {"afrr_pos_price_eur_mwh": ("ch_afrr_up_act_price_eur_mwh", "ch_afrr_up_act_mw"),
              "afrr_neg_price_eur_mwh": ("ch_afrr_down_act_price_eur_mwh", "ch_afrr_down_act_mw"),
              "mfrr_pos_price_eur_mwh": ("ch_mfrr_up_act_price_eur_mwh", "ch_mfrr_up_act_all_mw"),
              "mfrr_neg_price_eur_mwh": ("ch_mfrr_down_act_price_eur_mwh", "ch_mfrr_down_act_all_mw")}
    for src, (name, vol) in prices.items():
        out[name] = g[src].where(out[vol] > 0)
        dic.append(entry(name, f"EnergieUebersichtCH.{src}", "EUR/MWh", "15min", "volume-weighted mean", AV,
                         "average activation price; NaN when nothing was activated (source shows 0); from 2014"))
    vl = read_ts("EnergyOverview/Data/vertical_load_1h/*.parquet")["vertical_load_mw"]
    out["ch_tn_vertical_load_mw"] = K.hourly_to_qh(vl[vl.index >= C.start_utc()], "repeat")
    dic.append(entry("ch_tn_vertical_load_mw", "EnergieUebersichtCH vertical load (hourly sheet)", "MW", "1h", "mean", AV,
                     "hourly vertical grid load; ends 2022 (use 15-min feed-in or ENTSO-E load later)"))
    return out, dic


# ---------------------------------------------------------------- imbalance prices
def imbalance():
    d = on_grid(read_ts("ImbalancePrices/Data/qh/*.parquet"))
    out, dic = pd.DataFrame(index=d.index), []
    AV = "published after delivery (monthly files, mid following month); ENTSO-E near real time"
    for src, name, note in [("long_eur_mwh", "ch_imb_price_long_eur_mwh", "BG-long price; binding until 31 Dec 2025"),
                            ("short_eur_mwh", "ch_imb_price_short_eur_mwh", "BG-short price; binding until 31 Dec 2025"),
                            ("aep_eur_mwh", "ch_imb_price_single_eur_mwh", "single price (AEP); published from Jul 2025, binding from 1 Jan 2026")]:
        out[name] = d[src]
        dic.append(entry(name, f"Swissgrid imbalance prices XML.{src}", "EUR/MWh", "15min", "mean", AV, note + "; from Jan 2023 (ENTSO-E before, joined in build_master)"))
    out["flag_imb_aep_informational"] = (out["ch_imb_price_single_eur_mwh"].notna()
                                         & (out.index < pd.Timestamp("2026-01-01", tz=K.TZ).tz_convert("UTC")))
    dic.append(entry("flag_imb_aep_informational", "derived", "bool", "15min", "any", "-",
                     "single price published but not binding (Jul-Dec 2025)"))
    return out, dic


# ---------------------------------------------------------------- system balance 2026
def system_balance():
    ce = on_grid(read_ts("SystemBalance/Data/control_energy/*.parquet"))
    cab = on_grid(read_ts("SystemBalance/Data/control_area_balance/*.parquet"))
    xb = on_grid(read_ts("SystemBalance/Data/cross_border/*.parquet"))
    out, dic = pd.DataFrame(index=ce.index), []
    AV = "ex post (weekly CSV, 2026 only)"
    for p in ("afrr", "mfrr"):
        for d_src, d in (("pos", "up"), ("neg", "down")):
            sign = 1 if d == "up" else -1
            out[f"ch_{p}_{d}_offered_mw"] = ce[f"{p}_{d_src}_offered_mw"] * sign
            out[f"ch_{p}_{d}_act_mw"] = ce[f"{p}_{d_src}_activated_mw"] * sign
            out[f"ch_{p}_{d}_act_price_eur_mwh"] = ce[f"{p}_{d_src}_price_eur_mwh"]
            out[f"ch_{p}_{d}_cost_eur"] = ce[f"{p}_{d_src}_cost_eur"]
            dic += [entry(f"ch_{p}_{d}_offered_mw", f"control_energy.{p}_{d_src}_offered_mw", "MW", "15min", "mean", AV, "offered (magnitude)"),
                    entry(f"ch_{p}_{d}_act_mw", f"control_energy.{p}_{d_src}_activated_mw", "MW", "15min", "mean", AV,
                          "published balancing activation (magnitude)" + ("; includes PVTRE_sa- (check 2.4)" if (p, d) == ("mfrr", "down") else "")),
                    entry(f"ch_{p}_{d}_act_price_eur_mwh", f"control_energy.{p}_{d_src}_price_eur_mwh", "EUR/MWh", "15min", "mean", AV, "NaN when nothing activated"),
                    entry(f"ch_{p}_{d}_cost_eur", f"control_energy.{p}_{d_src}_cost_eur", "EUR", "15min", "sum", AV,
                          "cost per quarter-hour (source: cumulative weekly kEUR, differenced by the parser); sign as in source")]
    for src, name, sign, note in [("mfrr_sa_pos_mw", "ch_mfrr_sa_up_act_mw", 1, "scheduled mFRR activation up"),
                                  ("mfrr_sa_neg_mw", "ch_mfrr_sa_down_act_mw", -1, "scheduled mFRR activation down (magnitude)"),
                                  ("nrv_pos_import_mw", "ch_igcc_up_import_mw", 1, "imbalance netting (IGCC) import"),
                                  ("nrv_neg_export_mw", "ch_igcc_down_export_mw", -1, "imbalance netting (IGCC) export (magnitude)"),
                                  ("frce_pos_import_mw", "ch_frce_up_import_mw", 1, "FRCE import"),
                                  ("frce_neg_export_mw", "ch_frce_down_export_mw", -1, "FRCE export (magnitude)"),
                                  ("system_imbalance_mw", "ch_system_imbalance_mw", 1, "control-area imbalance; < 0 = short, > 0 = long")]:
        out[name] = cab[src] * sign
        dic.append(entry(name, f"control_area_balance.{src}", "MW", "15min", "mean", "ex post (UTC CSV, 2026 only)", note))
    # (mfrr_da_* are 0 throughout 2026 -> not carried; aep duplicates the imbalance files)
    ph = pd.Series(False, index=xb.index)
    for nb in ("at", "de", "fr", "it"):
        exp = xb[f"ntc_id_ch_{nb}_mw"].astype("float64")
        imp = -xb[f"ntc_id_{nb}_ch_mw"].astype("float64")
        for name, v in ((f"ch_{nb}_ntc_id_mw", exp), (f"{nb}_ch_ntc_id_mw", imp)):
            bad = v.abs() >= 99_999
            ph |= bad
            out[name] = v.where(~bad)
            dic.append(entry(name, f"cross_border.ntc_id ({name[:5].upper()})", "MW", "1h", "mean", "intraday (published hourly)",
                             "intraday NTC; import direction stored positive; 99999 placeholder -> NaN; a few negative values (CH-IT) kept"))
        out[f"ch_{nb}_sched_net_mw"] = xb[f"comm_flow_net_{nb}_mw"]
        dic.append(entry(f"ch_{nb}_sched_net_mw", f"cross_border.comm_flow_net_{nb}_mw", "MW", "15min", "mean", "ex post",
                         "commercial net exchange; > 0 = export from CH (= ENTSO-E 12.1.F total, check 2.3)"))
    out["ch_sched_net_total_mw"] = xb["comm_flow_net_total_mw"]
    dic.append(entry("ch_sched_net_total_mw", "cross_border.comm_flow_net_total_mw", "MW", "15min", "mean", "ex post", "> 0 = net export"))
    for nb, src in (("at", "at"), ("de", "de"), ("fr", "fr"), ("it_nord", "itn")):
        name = f"{nb}_ch_da_spread_eur_mwh"
        out[name] = xb[f"spot_spread_{src}_ch_eur_mwh"]
        dic.append(entry(name, f"cross_border.spot_spread_{src}_ch_eur_mwh", "EUR/MWh", "15min/1h", "mean", "D-1 (day-ahead)",
                         "day-ahead price neighbour minus CH (2026 only; cross-check for the DA price pull)"))
    out["flag_ntc_id_placeholder"] = ph.to_numpy()
    dic.append(entry("flag_ntc_id_placeholder", "derived", "bool", "1h", "any", "-", "some intraday NTC was 99999 (set to NaN)"))
    return out, dic


# ---------------------------------------------------------------- TRE
TRE_PRODUCTS = {"TRE_mFRR_sa+": "ch_tre_sa_up", "TRE_mFRR_sa-": "ch_tre_sa_down",
                "TRE_mFRR_da+": "ch_tre_da_up", "TRE_mFRR_da-": "ch_tre_da_down",
                "PVTRE_sa-": "ch_pvtre_sa_down"}


def tre(sysbal: pd.DataFrame):
    q = pd.concat([pd.read_parquet(f) for f in sorted(SG.glob("TRE/Data/qh/*/*.parquet"))], ignore_index=True)
    q = q[q["product"].isin(TRE_PRODUCTS)]
    q["ts"] = K.to_utc(q["delivery_start"])
    g = K.grid().index
    out, dic = pd.DataFrame(index=g), []
    AV = "bids: before delivery (gate closure); activations: ex post"
    for p, pre in TRE_PRODUCTS.items():
        x = q[q["product"] == p].drop_duplicates("ts", keep="last").set_index("ts").reindex(g)
        has = x["n_bids"].notna()
        for src, suf, unit, agg, note in [("offered_mw", "offered_mw", "MW", "mean", "offered energy-bid volume"),
                                          ("n_bids", "bids_n", "n", "mean", "number of bids"),
                                          ("price_min", "price_min_eur_mwh", "EUR/MWh", "min", "lowest bid price"),
                                          ("price_max", "price_max_eur_mwh", "EUR/MWh", "max", "highest bid price"),
                                          ("activated_mw", "act_mw", "MW", "mean", "activated MW (all purposes; mask with flag_tre_extra_activation_*)"),
                                          ("act_price_marginal", "act_price_marginal_eur_mwh", "EUR/MWh", "max", "highest activated bid price; mask with flag"),
                                          ("act_price_vwap", "act_price_vwap_eur_mwh", "EUR/MWh", "mean", "volume-weighted activated price; mask with flag")]:
            col = f"{pre}_{suf}"
            v = x[src].astype("float64")
            if src == "activated_mw":
                v = v.where(~has | v.notna(), 0.0)
            out[col] = v
            dic.append(entry(col, f"TRE qh view, product {p}.{src}", unit, "15min", agg, AV, note))
    # flags vs published balancing activation (2026)
    up = out[["ch_tre_sa_up_act_mw", "ch_tre_da_up_act_mw"]].sum(axis=1, min_count=1)
    dn = out[["ch_tre_sa_down_act_mw", "ch_tre_da_down_act_mw", "ch_pvtre_sa_down_act_mw"]].sum(axis=1, min_count=1)
    pub_up, pub_dn = sysbal["ch_mfrr_up_act_mw"].fillna(0), sysbal["ch_mfrr_down_act_mw"].fillna(0)
    in26 = sysbal["ch_mfrr_up_act_mw"].notna() | sysbal["ch_mfrr_down_act_mw"].notna()
    out["flag_tre_extra_activation_up"] = (in26 & ((up.fillna(0) - pub_up) >= 1)).to_numpy()
    out["flag_tre_extra_activation_down"] = (in26 & ((dn.fillna(0) - pub_dn) >= 1)).to_numpy()
    for d in ("up", "down"):
        dic.append(entry(f"flag_tre_extra_activation_{d}", "derived (TRE vs control_energy)", "bool", "15min", "any", "-",
                         "TRE activation exceeds published mFRR activation by >= 1 MW (2026 only; table design 4.7)"))
    return out, dic


# ---------------------------------------------------------------- auction blocks
def auction_blocks():
    a = pd.read_parquet(SG / "Auctions" / "Data" / "auctions.parquet")
    a["block_start_utc"], a["block_end_utc"] = K.to_utc(a.delivery_start), K.to_utc(a.delivery_end)
    a = a[(a.block_start_utc < C.end_utc()) & (a.block_end_utc > C.start_utc())].copy()
    a["procurement"] = np.where(a.duration_h >= 160, "week", np.where(a.duration_h >= 23, "day", "4h"))
    a["market"] = "ch_swissgrid"
    a["partial_in_sample"] = (a.block_end_utc > C.end_utc()) | (a.block_start_utc < C.start_utc())
    a["block_start_local"] = a.block_start_utc.dt.tz_convert(K.TZ)
    r = K.regimes(pd.DatetimeIndex(a.block_start_utc))
    for c in r.columns:
        a[c] = r[c].to_numpy()
    ren = {"awarded_mw_ch": "awarded_ch_mw", "bid_min": "price_bid_min", "bid_max": "price_bid_max",
           "settle_max": "price_settle_max", "settle_ch": "price_settle_ch", "bid_vwap": "price_bid_vwap"}
    a = a.rename(columns=ren)
    cols = ["market", "product", "direction", "procurement", "block_start_utc", "block_end_utc", "block_start_local",
            "duration_h", "partial_in_sample", "auction_id", "tender_series", "currency", "n_bids", "n_accepted",
            "offered_mw", "awarded_mw", "awarded_ch_mw", "price_bid_min", "price_bid_max", "price_settle_max",
            "price_settle_ch", "n_settle_prices", "price_bid_vwap", "cost"] + list(r.columns)
    a = a[cols].sort_values(["product", "direction", "procurement", "block_start_utc"]).reset_index(drop=True)
    pr = "per MW and hour of delivery, in `currency` (FCR EUR, aFRR/mFRR CHF)"
    notes = {"market": "ch_swissgrid (regelleistung.net blocks are added in 3.6)",
             "procurement": "week (>=160 h) / day (>=23 h) / 4h by block length",
             "partial_in_sample": "block extends beyond the sample (e.g. week from 31 Aug 2026)",
             "price_bid_min": "lowest accepted bid " + pr, "price_bid_max": "marginal (highest) accepted bid " + pr,
             "price_settle_max": "highest settlement price " + pr, "price_settle_ch": "FCR: Swiss settlement price " + pr + " (= regelleistung CH price / block hours, check 2.6); NaN when no bid was accepted (nothing procured in that block: 5,462 daily mFRR blocks, mostly 2024)",
             "price_bid_vwap": "volume-weighted accepted bid " + pr,
             "awarded_ch_mw": "capacity awarded in Switzerland (FCR: CH share of the cooperation)",
             "cost": "total cost of the block in `currency`"}
    dic = [dict(column=c, source=SRC if not c.startswith("regime_") else "derived (common.regimes)",
                source_series="Auctions/Data/auctions.parquet", notes=notes.get(c, "")) for c in cols]
    return a, dic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    tables = {}
    tables["energy_overview"] = energy_overview()
    tables["imbalance"] = imbalance()
    tables["system_balance"] = system_balance()
    tables["tre"] = tre(tables["system_balance"][0])
    blocks, bdic = auction_blocks()
    for name, (df, dic) in tables.items():
        bad = K.check_names(df.columns)
        assert not bad, (name, bad)
        assert len(df) == 409_052, name
        num = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
        first = df[num].apply(lambda s: s.first_valid_index()).min()
        print(f"\n=== {name}: {len(df.columns)} columns | data from {first} | NaN share (median over columns, after first value) "
              f"{np.nanmedian([df.loc[df[c].first_valid_index():, c].isna().mean() for c in num if df[c].first_valid_index() is not None]):.4f}")
        fl = {c: int(df[c].sum()) for c in df.columns if c.startswith("flag_")}
        if fl:
            print("  flags:", fl)
    print(f"\n=== auction_blocks: {len(blocks)} blocks")
    print(blocks.groupby(["product", "direction", "procurement"], observed=True)
          .agg(n=("auction_id", "size"), first=("block_start_local", "min"), last=("block_start_local", "max"),
               settle_ch_nan=("price_settle_ch", lambda s: round(s.isna().mean(), 3))).to_string())
    if a.write:
        for name, (df, dic) in tables.items():
            print("written:", K.write_clean(df, "swissgrid", name, dic))
        print("written:", K.write_long(blocks, "swissgrid", "auction_blocks", bdic))


if __name__ == "__main__":
    main()
