"""Clean layer: ENTSO-E Transmission (to-do 3.5 Transmission, 2 Oct 2026).

Run from Master_Thesis with .venv active:
    python Clean/clean_transmission.py            # build, check, print summary
    python Clean/clean_transmission.py --write    # also write Clean/Data/transmission/

Output  Clean/Data/transmission/transmission.parquet + _dictionary.csv (15-min grid)
Per directed border  {from}_{to}  with  from/to in ch, de, fr, it_nord, at:
  _ntc_{da,wa,ma,ya}_mw       forecasted transfer capacity (11.1), day/week/month/year ahead
  _flow_phys_mw               physical flow (12.1.G)
  _sched_total_mw / _sched_da_mw   commercial schedules, total (A05) / day-ahead (A01) (12.1.F)
  _countertrade_mw            countertrading volume (13.1.B)
  _expl_alloc_mw / _expl_price_eur_mwh / _expl_offer_mw   explicit daily auctions (11.1.A / 12.1.A)
Per neighbour  ch_{nb}_sched_net_mw = CH->nb - nb->CH (total schedule; > 0 = export from CH)

German border (regime file B1, B2, B2b; check 2.3)
- NTC and commercial schedules were published on the bidding-zone border: DE_AT_LU
  until 30 Sep 2018 (includes the Austrian border: CH->DE_AT_LU 5,200 MW = DE 4,000 +
  AT 1,200), DE_LU from 1 Oct 2018. Pre-split values go to ch_de_at_lu_* / de_at_lu_ch_*
  columns, post-split values to ch_de_* / de_ch_*. Never concatenate them.
  (Day-ahead schedules and week-ahead NTC exist on this border only after / before
  the split respectively.)
- Physical flows and countertrading are on the TransnetBW control area (unchanged at
  the split); explicit auctions on the bidding-zone border but for the CH-DE border
  only -> continuous ch_de_* columns.

Other rules
- Explicit auctions: hourly product of the daily auction (A04_A01). Where a day was
  published under classification sequence 2 (seq2), seq2 is used, else seq1. Missing
  hours stay NaN (a missing day = no published result). CH<->DE offered capacity
  exists only from 2 Jan 2018.
- Countertrading: event documents (15/30/60 min); duplicate stamps merged (max); the
  two direction codes B03_A01 / B03_A02 summed. NaN = no document (most likely no
  countertrade); views may fill 0.
- Implausible values (found 2 Oct 2026): NTC exactly 10000 MW (placeholder) -> NaN +
  flag_ntc_placeholder; total schedule > highest day-ahead NTC ever published for that
  direction + 2000 MW (CH<->FR 2015, up to 21,800 MW) -> NaN + flag_sched_total_implausible;
  > that quarter-hour's NTC + 2000 MW -> flag_sched_total_suspect only (value kept).
- Week- and month-ahead NTC: daily values as daily steps; year-ahead NTC: monthly
  values as monthly steps. Congestion costs (13.1.C) are empty for CH: not carried.
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

SRC = "ENTSO-E Transparency Platform"
NB = {"DE": "de", "FR": "fr", "IT_NORD": "it_nord", "AT": "at"}
SPLIT = pd.Timestamp("2018-10-01", tz=K.TZ).tz_convert("UTC")
SCHED_MARGIN_MW = 2000


def borders():
    for code, a in NB.items():
        yield f"CH-{code}", "ch", a
        yield f"{code}-CH", a, "ch"


def entry(col, series, unit, res, agg, avail, note=""):
    return dict(column=col, source=SRC, source_series=series, unit=unit, resolution_native=res,
                aggregation_rule=agg, availability_rule=avail, notes=note)


def read(ds, var, b, **kw):
    try:
        return K.read_entsoe("Transmission", ds, var, b, **kw)
    except FileNotFoundError:
        return None


def month_steps(s: pd.Series) -> pd.Series:
    """Monthly values -> step function from local month start to next month start."""
    loc = s.index.tz_convert(K.TZ).tz_localize(None)
    start = (loc + pd.Timedelta(days=2)).to_period("M").to_timestamp()      # robust to 23:00 stamps
    b = pd.DataFrame({"v": s.to_numpy(), "block_start_utc": start}).drop_duplicates("block_start_utc", keep="last")
    b["block_end_utc"] = b.block_start_utc + pd.offsets.MonthBegin(1)
    for c in ("block_start_utc", "block_end_utc"):
        b[c] = pd.DatetimeIndex(b[c]).tz_localize(K.TZ).tz_convert("UTC")
    return K.blocks_to_qh(b, ["v"])["v"]


def split_de(s: pd.Series, b: str, f: str, t: str):
    """For CH<->DE bidding-zone products: (pre-split series, post-split series, names)."""
    if "DE" not in b:
        return [(f"{f}_{t}", s, "")]
    pre_name = f"ch_de_at_lu" if f == "ch" else "de_at_lu_ch"
    pre, post = s.where(s.index < SPLIT), s.where(s.index >= SPLIT)
    return [(pre_name, pre, "until 30 Sep 2018: DE_AT_LU bidding-zone border (includes the AT border)"),
            (f"{f}_{t}", post, "from 1 Oct 2018: DE_LU bidding-zone border")]


def build():
    g = K.grid().index
    out, dic = pd.DataFrame(index=g), []
    for b, f, t in borders():
        # --- NTC
        for var, tag, how, avail in (("dayahead", "da", "h", "D-1"), ("weekahead", "wa", "d", "week W-1"),
                                     ("monthahead", "ma", "d", "month M-1"), ("yearahead", "ya", "m", "year Y-1")):
            d = read("ntc", var, b)
            if d is None or d.empty:
                continue
            s = d["ntc"].astype("float64")
            if how == "h":
                q = K.hourly_to_qh(s, "repeat")
            elif how == "d":
                q = K.local_steps(s.to_frame(), 1, ["ntc"])["ntc"]
            else:
                q = month_steps(s)
            for pref, part, note in split_de(q, b, f, t):
                if part.notna().any():
                    col = f"{pref}_ntc_{tag}_mw"
                    out[col] = part
                    dic.append(entry(col, f"11.1 NTC {var}, {b}", "MW", {"h": "1h", "d": "1 value per day", "m": "1 value per month"}[how],
                                     "mean", avail, note))
        # --- physical flows (TransnetBW for DE: continuous)
        d = read("physical_flows", "A11", b)
        if d is not None:
            col = f"{f}_{t}_flow_phys_mw"
            out[col] = K.hourly_to_qh(d.iloc[:, 0].astype("float64"), "repeat")
            dic.append(entry(col, f"12.1.G physical flow, {b}", "MW", "1h / 15min", "mean", "ex post",
                             "DE side = TransnetBW control area" if "DE" in b else ""))
        # --- scheduled exchanges (bidding-zone border for DE)
        for var, tag, avail in (("total_A05", "total", "ex post (final schedule)"), ("dayahead_A01", "da", "D-1 after day-ahead coupling")):
            d = read("scheduled_exchanges", var, b)
            if d is None:
                continue
            q = K.hourly_to_qh(d.iloc[:, 0].astype("float64"), "repeat")
            for pref, part, note in split_de(q, b, f, t):
                if part.notna().any():
                    col = f"{pref}_sched_{tag}_mw"
                    out[col] = part
                    dic.append(entry(col, f"12.1.F scheduled exchanges {var}, {b}", "MW", "1h until 2024, 15min from 2025",
                                     "mean", avail, note))
        # --- countertrading (event documents)
        d = read("countertrading", "A91", b, dedup=False)
        if d is not None and len(d):
            ct = d.groupby(level=0).max().sum(axis=1, min_count=1).astype("float64")
            col = f"{f}_{t}_countertrade_mw"
            out[col] = K.hourly_to_qh(ct, "repeat")
            dic.append(entry(col, f"13.1.B countertrading, {b} (B03_A01 + B03_A02)", "MW", "event (15/30/60 min)", "mean",
                             "ex post", "NaN = no document (most likely no countertrade); views may fill 0"))
        # --- explicit auctions
        a = read("explicit_allocated", "daily_A01", b)
        if a is not None:
            def pick(df, what):
                c1, c2 = f"B05_A04_A01_seq1__{what}", f"B05_A04_A01_seq2__{what}"
                s1 = df[c1] if c1 in df else pd.Series(np.nan, index=df.index)
                return (df[c2].combine_first(s1) if c2 in df else s1).astype("float64")
            for what, suf, unit in (("quantity", "expl_alloc_mw", "MW"), ("price.amount", "expl_price_eur_mwh", "EUR/MWh")):
                col = f"{f}_{t}_{suf}"
                out[col] = K.hourly_to_qh(pick(a, what), "repeat")
                dic.append(entry(col, f"12.1.A explicit allocation (daily auction, hourly product), {b}", unit, "1h", "mean",
                                 "D-1 (daily auction result)", "seq2 used where published, else seq1; missing day = no published result"))
        o = read("explicit_offered", "daily_A01", b)
        if o is not None:
            c1, c2 = "A31_A04_A01_seq1__quantity", "A31_A04_A01_seq2__quantity"
            s = (o[c2].combine_first(o[c1]) if c2 in o else o[c1]).astype("float64")
            col = f"{f}_{t}_expl_offer_mw"
            out[col] = K.hourly_to_qh(s, "repeat")
            dic.append(entry(col, f"11.1.A explicit offered capacity (daily auction), {b}", "MW", "1h", "mean", "D-1",
                             "CH<->DE only from 2 Jan 2018" if "DE" in b else ""))
    # --- implausible values (found 2 Oct 2026)
    ph = pd.Series(False, index=g)
    for c in [c for c in out.columns if "_ntc_" in c]:
        bad = out[c] == 10000
        ph |= bad
        out.loc[bad, c] = np.nan
    out["flag_ntc_placeholder"] = ph.to_numpy()
    dic.append(entry("flag_ntc_placeholder", "derived", "bool", "15min", "any", "-",
                     "some NTC was exactly 10000 MW (placeholder; CH->IT_NORD week-ahead 2025) -> NaN"))
    imp = pd.Series(False, index=g)
    sus = pd.Series(False, index=g)
    for c in [c for c in out.columns if c.endswith("_sched_total_mw")]:
        ntc = out.get(c.replace("_sched_total_mw", "_ntc_da_mw"))
        if ntc is None:
            continue
        hard = (out[c] > ntc.max() + SCHED_MARGIN_MW).fillna(False)      # above any NTC ever published
        soft = (out[c] > ntc + SCHED_MARGIN_MW).fillna(False) & ~hard     # above this quarter-hour's NTC
        imp |= hard
        sus |= soft
        out.loc[hard, c] = np.nan
    out["flag_sched_total_implausible"] = imp.to_numpy()
    out["flag_sched_total_suspect"] = sus.to_numpy()
    dic.append(entry("flag_sched_total_implausible", "derived", "bool", "15min", "any", "-",
                     f"some total schedule > highest day-ahead NTC ever published for that direction + {SCHED_MARGIN_MW} MW "
                     "(CH<->FR 2015, up to 21,800 MW) -> NaN"))
    dic.append(entry("flag_sched_total_suspect", "derived", "bool", "15min", "any", "-",
                     f"some total schedule > that quarter-hour's day-ahead NTC + {SCHED_MARGIN_MW} MW; value kept "
                     "(schedule or NTC may be wrong; mostly CH->FR 2015)"))

    # --- net commercial exchange per neighbour (> 0 = export from CH)
    for nb in ("de", "fr", "it_nord", "at"):
        for pre_ch, pre_nb, note in ((f"ch_{nb}", f"{nb}_ch", ""),) + (((f"ch_de_at_lu", "de_at_lu_ch", "pre-split DE_AT_LU border"),) if nb == "de" else ()):
            a_, b_ = f"{pre_ch}_sched_total_mw", f"{pre_nb}_sched_total_mw"
            if a_ in out and b_ in out:
                col = f"{pre_ch}_sched_net_mw"
                out[col] = out[a_] - out[b_]
                dic.append(entry(col, f"{a_} - {b_}", "MW", "1h / 15min", "mean", "ex post",
                                 "> 0 = export from CH; equals Swissgrid comm_flow_net (check 2.3)" + (f"; {note}" if note else "")))
    return out, dic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    out, dic = build()
    bad = K.check_names(out.columns)
    assert not bad, bad
    assert len(out) == 409_052
    num = out.select_dtypes(include="number")
    fv = num.apply(lambda c: c.first_valid_index())
    lv = num.apply(lambda c: c.last_valid_index())
    s = pd.DataFrame({"first": fv.dt.tz_convert(K.TZ).dt.strftime("%Y-%m-%d"), "last": lv.dt.tz_convert(K.TZ).dt.strftime("%Y-%m-%d"),
                      "median": num.median().round(0), "max": num.max().round(0),
                      "nan_in_range": [round(num.loc[fv[c]:lv[c], c].isna().mean(), 4) if fv[c] is not None else np.nan for c in num.columns]})
    print(s.to_string())
    # checks
    sg = C.DATA_DIR / "swissgrid" / "system_balance.parquet"
    if sg.exists():
        x = pd.read_parquet(sg, columns=["ts_utc", "ch_de_sched_net_mw", "ch_fr_sched_net_mw", "ch_at_sched_net_mw", "ch_it_sched_net_mw"]).set_index("ts_utc")
        for nb, sgc in (("de", "ch_de_sched_net_mw"), ("fr", "ch_fr_sched_net_mw"), ("at", "ch_at_sched_net_mw"), ("it_nord", "ch_it_sched_net_mw")):
            j = pd.concat([out[f"ch_{nb}_sched_net_mw"], x[sgc]], axis=1).dropna()
            print(f"check 2026 net schedule vs Swissgrid, {nb}: identical {((j.iloc[:, 0] - j.iloc[:, 1]).abs() < 1).mean():.3f} (n={len(j)})")
    pre = out["ch_de_at_lu_ntc_da_mw"].dropna()
    print("pre-split CH->DE_AT_LU NTC median:", pre.median(), "| post-split CH->DE median:", out["ch_de_ntc_da_mw"].median(),
          "+ CH->AT:", out["ch_at_ntc_da_mw"][out.index >= SPLIT].median())
    if a.write:
        print("written:", K.write_clean(out, "transmission", "transmission", dic))


if __name__ == "__main__":
    main()
