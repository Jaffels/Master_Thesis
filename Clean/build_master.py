"""Build the combined dataset (master layer) from the clean layer (plan step 3, 2 Oct 2026).

Run from Master_Thesis with .venv active:
    python Clean/build_gap_list.py --write     # first: gap list must be current
    python Clean/build_master.py               # dry run: build + validate + report, writes nothing
    python Clean/build_master.py --write       # also write the files below

Output (with --write), in Clean/Data/master/:
    master_15min.parquet        15-min grid (409,052 rows): ts_utc, ts_local, regimes, all drivers / targets
    auction_blocks.parquet      Swissgrid + regelleistung.net blocks stacked (+ ENTSO-E contracted-reserve cross-check)
    data_dictionary.csv         one row per column of both tables (table design Section 9)
    nan_share_by_year.csv       NaN share per master column and local year
    nan_runs_unexplained.csv    NaN runs >= 1 day inside a column's own span, with / without a gap-list match
    build_master_report.txt     validation report (same text as printed)
Nothing is written if a validation check fails (exit code 1).

Rules (table design Sections 4-5, 8-9)
- Every clean grid table must sit exactly on common.grid(); the master is a column-wise join.
- Source priority where two sources publish the same variable (4.3):
    CH imbalance prices long / short   ENTSO-E until 31 Dec 2022, Swissgrid from 1 Jan 2023;
                                       from 1 Jan 2026 long = short = Swissgrid single price (AEP),
                                       as ENTSO-E publishes it (switch: IMB_2026_LONG_SHORT_FROM_SINGLE).
                                       ENTSO-E kept as *_xchk_entsoe from 2023; src_ch_imb_price.
    CH aFRR activation (MW, price)     Swissgrid Energy Overview, Swissgrid system balance in 2026
                                       (identical in 2026); src_ch_afrr_act. ENTSO-E 17.1.F price kept
                                       as *_xchk_entsoe (integer-rounded in 2025; has a price also
                                       when nothing was activated).
    CH mFRR activation                 system balance (2026) = balancing activation. Energy Overview
                                       mFRR = all activations -> renamed *_act_all_* (never a balancing volume).
    CH total generation                Swissgrid production (Energy Overview, was ch_prod_mw) =
                                       ch_gen_total_mw; ENTSO-E 16.1.B&C total -> *_xchk_entsoe.
    Physical flows, schedules          ENTSO-E primary; Swissgrid Energy Overview flows and 2026
                                       commercial net flows -> *_xchk_swissgrid. Note: the Swissgrid
                                       flows are defined differently (both directions non-zero in 94 %
                                       of QH; net corr. 0.98 with ENTSO-E) -> cross-check only.
    CH actual load                     ENTSO-E 6.1.A; in months where ENTSO-E published the day-ahead
                                       forecast as actual (> 90 % of QH identical: Sep-Nov 2021, 2022)
                                       -> Swissgrid consumption ch_cons_mw (same level, r 0.94-0.96);
                                       flag_ch_load_actual_is_forecast, src_ch_load_actual (3 Oct 2026, EDA 6).
  Swissgrid "it" border columns are renamed to "it_nord" (same CH-IT border as ENTSO-E).
- Secondary series (plan step 3, decided here):
    load_fc_long            IN the master (daily / weekly steps like the NTC horizons; known ex ante)
    contracted_reserves_ch  NOT in the master (block products) -> block means as
                            price_xchk_entsoe / amount_xchk_entsoe_mw in auction_blocks (4.3, 5.1)
    neighbour series        all IN the master; selection happens in the views
    long tables             installed_capacity, outage_events stay in their domain folders
- Regime columns: common.regimes() + regime_ch_gen_reporting (generation) + ch_afrr_platform_fallback
  (outages), placed first after the time key.
- No imputation (Decision 5). TRE activation columns are carried unmasked; views mask them with
  flag_tre_extra_activation_* (4.7).

Validation (fails the build): grid row count per local year (DST-aware), unique / sorted ts_utc
equal to common.grid(), name grammar, every column in the data dictionary, regime columns
complete, unchanged columns keep their clean-layer NaN share, auction blocks: keys complete,
no duplicates, no overlaps per (market, product, direction, procurement, tender_series), every block
overlaps the sample, blocks cut by the sample edges are marked partial_in_sample. Reported only: NaN runs >= 1 day not matched by the gap list.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import config as C  # noqa: E402

TZ = C.TZ_LABEL
QH = pd.Timedelta(C.FREQ)

IMB_2026_LONG_SHORT_FROM_SINGLE = True     # see docstring
RUN_MIN_QH = 96                            # NaN runs shorter than 1 day are not listed

# (domain, table, gap-list source used to explain NaN runs)
GRID_TABLES = [
    ("load", "load", "ENTSO-E"),
    ("load", "load_fc_long", "ENTSO-E"),
    ("generation", "generation", "ENTSO-E"),
    ("transmission", "transmission", "ENTSO-E"),
    ("outages", "outages", None),
    ("balancing", "balancing", "ENTSO-E"),
    ("swissgrid", "energy_overview", "Swissgrid"),
    ("swissgrid", "imbalance", "Swissgrid"),
    ("swissgrid", "system_balance", "Swissgrid"),
    ("swissgrid", "tre", "Swissgrid"),
    ("weather", "weather", None),
    ("frequency", "frequency", "TSO/Energy-Charts/Zenodo"),
]
# Tables joined only once their clean file exists (3 Oct 2026: ENTSO-E day-ahead prices,
# Clean/clean_prices.py). Without the file the master builds exactly as before.
OPTIONAL_TABLES = [
    ("prices", "day_ahead", "ENTSO-E"),
]
GRID_TABLES += [t for t in OPTIONAL_TABLES if (C.DATA_DIR / t[0] / f"{t[1]}.parquet").exists()]
NOT_IN_MASTER = {
    "balancing/contracted_reserves_ch": "block products: cross-check columns in auction_blocks",
    "generation/installed_capacity": "long table (zone x year), stays in Clean/Data/generation/",
    "outages/outage_events": "long table (events with created_utc), stays in Clean/Data/outages/",
}

_XS = "_xchk_swissgrid"
RENAMES = {
    "swissgrid/energy_overview": {
        **{f"{a}_{b}_flow_phys_mw": f"{a}_{b}_flow_phys_mw{_XS}"
           for a, b in [("ch", "de"), ("de", "ch"), ("ch", "fr"), ("fr", "ch"), ("ch", "at"), ("at", "ch")]},
        "ch_it_flow_phys_mw": f"ch_it_nord_flow_phys_mw{_XS}",
        "it_ch_flow_phys_mw": f"it_nord_ch_flow_phys_mw{_XS}",
        "ch_mfrr_up_act_price_eur_mwh": "ch_mfrr_up_act_all_price_eur_mwh",
        "ch_mfrr_down_act_price_eur_mwh": "ch_mfrr_down_act_all_price_eur_mwh",
        "ch_prod_mw": "ch_gen_total_mw",
    },
    "generation/generation": {"ch_gen_total_mw": "ch_gen_total_mw_xchk_entsoe"},
    "swissgrid/system_balance": {
        "ch_de_sched_net_mw": f"ch_de_sched_net_mw{_XS}",
        "ch_fr_sched_net_mw": f"ch_fr_sched_net_mw{_XS}",
        "ch_at_sched_net_mw": f"ch_at_sched_net_mw{_XS}",
        "ch_it_sched_net_mw": f"ch_it_nord_sched_net_mw{_XS}",
        "ch_it_ntc_id_mw": "ch_it_nord_ntc_id_mw",
        "it_ch_ntc_id_mw": "it_nord_ch_ntc_id_mw",
    },
}
RENAME_NOTES = {
    "ch_gen_total_mw": "CH total generation = Swissgrid production (Energy Overview, clean name ch_prod_mw); "
                       "primary over ENTSO-E (60-70 % coverage until 2024)",
    "ch_gen_total_mw_xchk_entsoe": "ENTSO-E 16.1.B&C CH total; incomplete (about 50-70 % of Swissgrid "
                                   "production until 2024, about 90 % from 2025); cross-check only",
    "ch_mfrr_up_act_all_price_eur_mwh": "Energy Overview mFRR price of ALL activations (incl. non-balancing); "
                                        "not the balancing activation price",
    "ch_mfrr_down_act_all_price_eur_mwh": "Energy Overview mFRR price of ALL activations (incl. non-balancing); "
                                          "not the balancing activation price",
}

# combined columns: name -> tables that publish it (handled by combine_*; anything else duplicated = error)
COMBINED = {
    "ch_imb_price_long_eur_mwh": {"balancing/balancing", "swissgrid/imbalance"},
    "ch_imb_price_short_eur_mwh": {"balancing/balancing", "swissgrid/imbalance"},
    "ch_afrr_up_act_price_eur_mwh": {"balancing/balancing", "swissgrid/energy_overview", "swissgrid/system_balance"},
    "ch_afrr_down_act_price_eur_mwh": {"balancing/balancing", "swissgrid/energy_overview", "swissgrid/system_balance"},
    "ch_afrr_up_act_mw": {"swissgrid/energy_overview", "swissgrid/system_balance"},
    "ch_afrr_down_act_mw": {"swissgrid/energy_overview", "swissgrid/system_balance"},
}

# single-source columns that combine() replaces (source switch per month)
PATCHED = {"ch_load_actual_mw"}
LOAD_COPY_SHARE = 0.9          # month counts as "actual = forecast" above this share of identical QH

REGIME_FIRST = ["regime_afrr_dir", "regime_fcr", "regime_afrr_daily", "regime_mfrr_merged", "regime_de_zone",
                "regime_imb_resolution", "regime_imb_pricing", "regime_ch_gen_reporting",
                "ch_afrr_platform_fallback"]

# flag -> regex of the data columns it qualifies (data dictionary column `flags`)
FLAG_LINKS = {
    **{f"flag_{z}_load_actual_suspect": rf"^{z}_load_actual_mw$" for z in ("ch", "de_lu", "fr", "it_nord", "at")},
    "flag_de_lu_load_da_fc_q4_2018_patch": r"^de_lu_load_da_fc_mw$",
    "flag_ch_gen_partial_2015h1": r"^ch_gen_.*(?<!_xchk_swissgrid)$",
    "flag_at_gen_wind_on_id_fc_placeholder": r"^at_gen_wind_on_id_fc_mw$",
    "flag_ntc_placeholder": r"_ntc_(da|wa|ma|ya)_mw$",
    "flag_sched_total_implausible": r"_sched_total_mw$",
    "flag_sched_total_suspect": r"_sched_total_mw$",
    "flag_ntc_id_placeholder": r"_ntc_id_mw$",
    "flag_imb_price_entsoe_error_2025_01": r"^ch_imb_price_(long|short)_eur_mwh_xchk_entsoe$",
    "flag_imb_aep_informational": r"^ch_imb_price_single_eur_mwh$",
    "flag_tre_extra_activation_up": r"^ch_(tre|pvtre)_\w+_up_act",
    "flag_tre_extra_activation_down": r"^ch_(tre|pvtre)_\w+_down_act",
    "flag_ce_freq_implausible": r"^ce_",
    "flag_ce_freq_suspect": r"^ce_",
    "flag_ce_freq_low_coverage": r"^ce_",
    "flag_ch_afrr_fallback_published_late": r"^ch_afrr_(fallback_share|platform_fallback)$",
    "flag_ch_load_actual_is_forecast": r"^ch_load_actual_mw$",
}

DICT_COLS = ["table", "column", "class", "area", "variable", "unit", "source", "source_series",
             "resolution_native", "start_utc", "end_utc", "nan_share", "aggregation_rule",
             "availability_rule", "flags", "notes", "clean_table"]

AB_KEYS = ["market", "product", "direction", "procurement", "tender_series", "block_start_utc", "block_end_utc"]
# tender_series: Swissgrid 0 = main tender, 1-5 = additional tenders for the same week (_S1.., advance
# procurement); regelleistung.net = <NA> (tender 1 only). Checks treat <NA> as its own series.


def loc_utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz=TZ).tz_convert("UTC")


T2023, T2026 = loc_utc("2023-01-01"), loc_utc("2026-01-01")


class Report:
    def __init__(self):
        self.lines, self.failures = [], []

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.lines.append(msg)

    def check(self, ok: bool, msg: str):
        self(("  ok    " if ok else "  FAIL  ") + msg)
        if not ok:
            self.failures.append(msg)


# ---------------------------------------------------------------- reading
def read_grid_table(domain: str, name: str, g: pd.DatetimeIndex, rep: Report) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = C.DATA_DIR / domain / f"{name}.parquet"
    df = pd.read_parquet(path)
    ts = K.to_utc(df.pop("ts_utc"))
    df.index = ts
    df.index.name = "ts_utc"
    df = df.drop(columns=[c for c in ("ts_local",) if c in df.columns])
    rep.check(len(df) == len(g) and df.index.equals(g), f"{domain}/{name}: on the grid ({len(df):,} rows, {df.shape[1]} cols)")
    d = pd.read_csv(C.DATA_DIR / domain / f"{name}_dictionary.csv")
    return df, d


def as_float(s: pd.Series) -> pd.Series:
    return s.astype("float64")


# ---------------------------------------------------------------- source priority
def combine(tables: dict, g: pd.DatetimeIndex, dicts: dict) -> tuple[dict, list[dict]]:
    """Build the combined columns. Returns {column: Series} and their dictionary rows."""
    out, rows = {}, []
    bal, eo = tables["balancing/balancing"], tables["swissgrid/energy_overview"]
    sb, im = tables["swissgrid/system_balance"], tables["swissgrid/imbalance"]
    idx = g
    after23, after26 = np.asarray(idx >= T2023), np.asarray(idx >= T2026)

    # CH imbalance prices
    single = as_float(im["ch_imb_price_single_eur_mwh"])
    src = None
    for side in ("long", "short"):
        c = f"ch_imb_price_{side}_eur_mwh"
        e, s = as_float(bal[c]), as_float(im[c])
        sg = s.where(after23)
        if IMB_2026_LONG_SHORT_FROM_SINGLE:
            sg = sg.where(~after26, single)
        prim = sg.combine_first(e)
        lab = np.where(sg.notna(), np.where(after26 & IMB_2026_LONG_SHORT_FROM_SINGLE, "swissgrid_single", "swissgrid"),
                       np.where(e.notna(), "entsoe", None))
        src = lab if src is None else np.where(pd.isna(src), lab, src)
        out[c] = prim
        out[f"{c}_xchk_entsoe"] = e.where(after23)
        r = dict(dicts["swissgrid/imbalance"][c])
        r.update(source="Swissgrid (2023-) / ENTSO-E (-2022)",
                 source_series=f"{dicts['balancing/balancing'][c]['source_series']} until 2022; "
                               f"{r['source_series']} from 2023"
                               + ("; aep_eur_mwh from 2026" if IMB_2026_LONG_SHORT_FROM_SINGLE else ""),
                 notes=("Swissgrid primary from 1 Jan 2023 (= ENTSO-E in 99.6 %); "
                        + ("from 1 Jan 2026 single price: long = short = AEP (regime_imb_pricing); " if IMB_2026_LONG_SHORT_FROM_SINGLE else "")
                        + "source per row in src_ch_imb_price; missing at source 1 Jul-29 Sep 2016 and 31 Dec 2022"),
                 clean_table="balancing/balancing + swissgrid/imbalance")
        rows.append(r)
        rx = dict(dicts["balancing/balancing"][c])
        rx.update(column=f"{c}_xchk_entsoe", notes="ENTSO-E 17.1.G from 2023 (where Swissgrid is primary); "
                  "2-5 Jan 2025 ENTSO-E error (flag_imb_price_entsoe_error_2025_01); Jan 2026 missing at ENTSO-E",
                  clean_table="balancing/balancing")
        rows.append(rx)
    out["src_ch_imb_price"] = pd.Series(pd.Categorical(src, categories=["entsoe", "swissgrid", "swissgrid_single"]), index=idx)
    rows.append(dict(column="src_ch_imb_price", source="derived", source_series="build_master.py", unit="category",
                     resolution_native="15min", aggregation_rule="mode", availability_rule="-",
                     notes="source of ch_imb_price_long/short_eur_mwh per row", clean_table="-"))

    # CH aFRR activation: Energy Overview, system balance in 2026
    src = None
    for d in ("up", "down"):
        for c in (f"ch_afrr_{d}_act_mw", f"ch_afrr_{d}_act_price_eur_mwh"):
            e, s = as_float(eo[c]), as_float(sb[c])
            s26 = s.where(after26)
            out[c] = s26.combine_first(e)
            r = dict(dicts["swissgrid/energy_overview"][c])
            r.update(source="Swissgrid", source_series=f"{r['source_series']}; system balance "
                     f"{dicts['swissgrid/system_balance'][c]['source_series']} in 2026",
                     notes=(r.get("notes") or "") + "; system balance from 2026 (identical to Energy Overview in "
                     "2026); source per row in src_ch_afrr_act",
                     clean_table="swissgrid/energy_overview + swissgrid/system_balance")
            rows.append(r)
            if c.endswith("_mw"):
                lab = np.where(s26.notna(), "system_balance", np.where(e.notna(), "energy_overview", None))
                src = lab if src is None else np.where(pd.isna(src), lab, src)
            else:
                out[f"{c}_xchk_entsoe"] = as_float(bal[c])
                rx = dict(dicts["balancing/balancing"][c])
                rx.update(column=f"{c}_xchk_entsoe", notes="ENTSO-E 17.1.F; = Swissgrid except 2025 "
                          "(ENTSO-E rounded to whole EUR); also has a price when nothing was activated",
                          clean_table="balancing/balancing")
                rows.append(rx)
    out["src_ch_afrr_act"] = pd.Series(pd.Categorical(src, categories=["energy_overview", "system_balance"]), index=idx)
    rows.append(dict(column="src_ch_afrr_act", source="derived", source_series="build_master.py", unit="category",
                     resolution_native="15min", aggregation_rule="mode", availability_rule="-",
                     notes="source of ch_afrr_{up,down}_act_mw / _act_price_eur_mwh per row (from the up volume)",
                     clean_table="-"))

    # CH actual load: ENTSO-E, Swissgrid consumption in months where ENTSO-E 'actual' = DA forecast
    ld = tables["load/load"]
    a, f, cons = as_float(ld["ch_load_actual_mw"]), as_float(ld["ch_load_da_fc_mw"]), as_float(eo["ch_cons_mw"])
    same = pd.Series(np.isclose(a, f, rtol=0, atol=0.5) & a.notna().to_numpy(), index=idx)
    month = pd.Index(idx.tz_convert(TZ).strftime("%Y-%m"))
    share = same.groupby(month.to_numpy()).mean()
    bad = np.asarray(month.isin(share[share > LOAD_COPY_SHARE].index))
    out["ch_load_actual_mw"] = a.where(~bad, cons)
    out["flag_ch_load_actual_is_forecast"] = pd.Series(bad, index=idx)
    lab = np.where(bad & cons.notna().to_numpy(), "swissgrid_cons", np.where(a.notna(), "entsoe", None))
    out["src_ch_load_actual"] = pd.Series(pd.Categorical(lab, categories=["entsoe", "swissgrid_cons"]), index=idx)
    months = ", ".join(sorted(share[share > LOAD_COPY_SHARE].index))
    r = dict(dicts["load/load"]["ch_load_actual_mw"])
    r.update(source="ENTSO-E / Swissgrid", clean_table="load/load + swissgrid/energy_overview",
             notes=(r["notes"] + "; " if isinstance(r.get("notes"), str) and r["notes"] else "") +
             f"months where ENTSO-E published the DA forecast as actual replaced by Swissgrid ch_cons_mw ({months}); "
             "source per row in src_ch_load_actual")
    rows.append(r)
    rows.append(dict(column="flag_ch_load_actual_is_forecast", source="derived", source_series="build_master.py",
                     unit="bool", resolution_native="15min", aggregation_rule="any", availability_rule="-",
                     notes=f"month in which ENTSO-E CH actual load = DA forecast in > {LOAD_COPY_SHARE:.0%} of QH "
                           f"({months}); ch_load_actual_mw = Swissgrid ch_cons_mw there", clean_table="-"))
    rows.append(dict(column="src_ch_load_actual", source="derived", source_series="build_master.py", unit="category",
                     resolution_native="15min", aggregation_rule="mode", availability_rule="-",
                     notes="source of ch_load_actual_mw per row", clean_table="-"))
    return out, rows


# ---------------------------------------------------------------- dictionary helpers
_NAME = re.compile(rf"^({K._AREA_RE})_(?:({K._AREA_RE})_)?(.+?)_({K._UNIT_RE})$")


def parse_name(c: str) -> tuple[str, str, str, str]:
    """-> class, area, variable, unit"""
    if c in ("ts_utc", "ts_local"):
        return "key", "", "", ""
    if c == "ch_afrr_platform_fallback":
        return "regime", "ch", "afrr_platform_fallback", "0/1"
    for p, cl in (("flag_", "flag"), ("regime_", "regime"), ("src_", "source"), ("eda_", "eda")):
        if c.startswith(p):
            return cl, "", c[len(p):], ""
    base, _, xs = c.partition("_xchk_")
    m = _NAME.match(base)
    if not m:
        return ("xchk" if xs else "data"), "", base, ""
    a1, a2, var, unit = m.groups()
    return ("xchk" if xs else "data"), (f"{a1}->{a2}" if a2 else a1), var, unit


def flags_for(col: str, flag_cols: list[str]) -> str:
    return ";".join(f for f in flag_cols if f in FLAG_LINKS and re.search(FLAG_LINKS[f], col))


# ---------------------------------------------------------------- master table
def build_master(rep: Report) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    g = K.grid()
    gi = g.index
    tables, dicts, origin = {}, {}, {}
    rep("Reading clean grid tables")
    for domain, name, _ in GRID_TABLES:
        key = f"{domain}/{name}"
        df, d = read_grid_table(domain, name, gi, rep)
        ren = RENAMES.get(key, {})
        missing = set(ren) - set(df.columns)
        rep.check(not missing, f"{key}: rename sources present" + (f" (missing {sorted(missing)})" if missing else ""))
        df = df.rename(columns=ren)
        d["column"] = d["column"].replace(ren)
        tables[key] = df
        dicts[key] = {r["column"]: r for r in d.to_dict("records")}
        for c in df.columns:
            origin.setdefault(c, set()).add(key)

    dup = {c: t for c, t in origin.items() if len(t) > 1}
    unexpected = {c: t for c, t in dup.items() if COMBINED.get(c) != t}
    rep.check(not unexpected, "duplicate names only where a source rule exists"
              + (f": {unexpected}" if unexpected else f" ({len(dup)} combined columns)"))

    comb, comb_rows = combine(tables, gi, dicts)

    cols, rows = {}, []
    reg = K.regimes(gi)
    for c in reg.columns:
        cols[c] = reg[c]
        rows.append(dict(column=c, source="derived", source_series="common.REGIME_DATES", unit="category"
                         if reg[c].dtype.name == "category" else "0/1", resolution_native="15min",
                         aggregation_rule="mode", availability_rule="known in advance (market-design date)",
                         notes="break dates: " + ", ".join(f"{k} {v}" for k, v in K.REGIME_DATES.items()),
                         clean_table="common.regimes()"))
    unchanged = []
    for domain, name, _ in GRID_TABLES:
        key = f"{domain}/{name}"
        for c in tables[key].columns:
            if c in PATCHED:
                cols[c] = comb[c].astype("float32")
                continue
            if c in COMBINED:
                if c not in cols:
                    for cc in [k for k in comb if k == c or k.startswith(c + "_xchk") ]:
                        cols[cc] = comb[cc]
                continue
            s = tables[key][c]
            cols[c] = s.astype("float32") if pd.api.types.is_float_dtype(s) else s
            r = dict(dicts[key][c])
            r["clean_table"] = key
            if c in RENAME_NOTES:
                r["notes"] = RENAME_NOTES[c] + ("; " + str(r["notes"]) if isinstance(r.get("notes"), str) and r["notes"] else "")
            elif c.endswith(_XS):
                r["notes"] = "Swissgrid cross-check of the ENTSO-E primary; not used in models" + (
                    "; definition differs (both directions non-zero in 94 % of QH; net corr. 0.98)" if "_flow_phys_" in c else "")
            rows.append(r)
            unchanged.append(c)
    for c in ("src_ch_imb_price", "src_ch_afrr_act", "src_ch_load_actual", "flag_ch_load_actual_is_forecast"):
        cols[c] = comb[c]
    rows += comb_rows
    for k in tables:
        tables[k] = None

    order = [c for c in REGIME_FIRST if c in cols] + [c for c in cols if c not in REGIME_FIRST]
    for c in order:
        if pd.api.types.is_float_dtype(cols[c]) and cols[c].dtype != C.FLOAT_DTYPE:
            cols[c] = cols[c].astype(C.FLOAT_DTYPE)
    m = pd.DataFrame({"ts_local": g["ts_local"], **{c: cols[c] for c in order}}, index=gi)
    m.index.name = "ts_utc"
    rows.insert(0, dict(column="ts_local", source="derived", source_series="ts_utc", unit="Europe/Zurich",
                        resolution_native="15min", notes="local label of ts_utc", clean_table="common.grid()"))
    rows.insert(0, dict(column="ts_utc", source="derived", source_series="common.grid()", unit="UTC",
                        resolution_native="15min", notes="primary key, interval start", clean_table="common.grid()"))
    dd = pd.DataFrame(rows)
    dd["table"] = "master_15min"
    return m, dd, dict(dicts=dicts, unchanged=unchanged)


def finish_dictionary(m: pd.DataFrame, dd: pd.DataFrame) -> pd.DataFrame:
    dd = dd.drop_duplicates("column", keep="first").set_index("column")
    flag_cols = [c for c in m.columns if c.startswith("flag_")]
    starts, ends, nans = {}, {}, {}
    for c in m.columns:
        v = m[c]
        if v.dtype == bool or c.startswith(("flag_", "regime_")) or c == "ch_afrr_platform_fallback":
            nn = m.index
        else:
            nn = m.index[v.notna().to_numpy()]
        starts[c], ends[c] = (nn.min(), nn.max()) if len(nn) else (pd.NaT, pd.NaT)
        nans[c] = round(float(v.isna().mean()), 5)
    dd = dd.reindex(["ts_utc"] + list(m.columns))
    dd.index.name = "column"
    dd = dd.reset_index()
    parsed = dd["column"].map(parse_name)
    dd["class"] = parsed.str[0]
    dd["area"] = parsed.str[1]
    dd["variable"] = parsed.str[2]
    pu = parsed.str[3]
    dd["unit"] = dd["unit"].where(dd["unit"].notna(), pu)
    dd["start_utc"] = dd["column"].map(starts)
    dd["end_utc"] = dd["column"].map(ends)
    dd["nan_share"] = dd["column"].map(nans)
    dd["flags"] = dd["column"].map(lambda c: flags_for(c, flag_cols) if parse_name(c)[0] in ("data", "xchk") else "")
    dd["table"] = "master_15min"
    return dd.reindex(columns=DICT_COLS)


# ---------------------------------------------------------------- auction blocks
def block_means(grid_idx: pd.DatetimeIndex, values: np.ndarray, st: pd.DatetimeIndex, en: pd.DatetimeIndex) -> np.ndarray:
    ok = ~np.isnan(values)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, values, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    i0 = np.searchsorted(grid_idx.asi8, st.asi8, side="left")
    i1 = np.searchsorted(grid_idx.asi8, en.asi8, side="left")
    n = cn[i1] - cn[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[i1] - cs[i0]) / np.maximum(n, 1), np.nan)


def build_auction_blocks(rep: Report) -> tuple[pd.DataFrame, pd.DataFrame]:
    gi = K.grid().index
    sg = pd.read_parquet(C.DATA_DIR / "swissgrid" / "auction_blocks.parquet")
    rl = pd.read_parquet(C.DATA_DIR / "regelleistung" / "auction_blocks_rl.parquet")
    dsg = pd.read_csv(C.DATA_DIR / "swissgrid" / "auction_blocks_dictionary.csv")
    drl = pd.read_csv(C.DATA_DIR / "regelleistung" / "auction_blocks_rl_dictionary.csv")
    for df in (sg, rl):
        for c in df.columns:
            if df[c].dtype.name == "category" or c.startswith("regime_") and df[c].dtype == object:
                df[c] = df[c].astype(str)

    # ENTSO-E contracted reserves as block means (Swissgrid rows only)
    cr = pd.read_parquet(C.DATA_DIR / "balancing" / "contracted_reserves_ch.parquet")
    cr.index = K.to_utc(cr.pop("ts_utc"))
    rep.check(cr.index.equals(gi), "balancing/contracted_reserves_ch: on the grid")
    proc = {"week": "weekly", "day": "daily", "4h": "daily"}
    dirn = {"sym": "symmetric", "up": "up", "down": "down"}
    sg["price_xchk_entsoe"] = np.nan
    sg["amount_xchk_entsoe_mw"] = np.nan
    used = []
    for (p, d, pr), ix in sg.groupby(["product", "direction", "procurement"]).groups.items():
        stem = f"ch_{p.lower()}_{proc[pr]}_{dirn[d]}"
        st, en = K.to_utc(sg.loc[ix, "block_start_utc"]), K.to_utc(sg.loc[ix, "block_end_utc"])
        for col, out in ((f"{stem}_price_eur_mw", "price_xchk_entsoe"), (f"{stem}_amount_mw", "amount_xchk_entsoe_mw")):
            if col in cr.columns:
                sg.loc[ix, out] = block_means(gi, cr[col].to_numpy("float64"), st, en)
                used.append(f"{p}/{d}/{pr} <- {col}")
    rep(f"  ENTSO-E contracted-reserve cross-check mapped for: {len(used)} column links")

    ab = pd.concat([sg, rl], ignore_index=True, sort=False)
    first = AB_KEYS + ["block_start_local", "duration_h", "partial_in_sample", "currency"]
    regs = [c for c in ab.columns if c.startswith("regime_")]
    ab = ab[first + [c for c in ab.columns if c not in first + regs] + regs]
    ab["tender_series"] = ab["tender_series"].astype("Int64")
    ab = ab.sort_values(["market", "product", "direction", "procurement", "block_start_utc", "tender_series"],
                        kind="stable").reset_index(drop=True)

    # dictionary
    d = pd.concat([dsg.assign(_t="swissgrid"), drl.assign(_t="regelleistung")], ignore_index=True)
    rows = []
    for c in ab.columns:
        sub = d[d["column"] == c]
        if c in ("price_xchk_entsoe", "amount_xchk_entsoe_mw"):
            rows.append(dict(column=c, source="ENTSO-E Transparency Platform", source_series="17.1.B&C contracted reserves, CH",
                             unit="EUR/MW (as published)" if c.startswith("price") else "MW",
                             notes="mean of the ENTSO-E step function over the block; Swissgrid rows only; "
                                   "cross-check, not a target (table design 4.3); 4h blocks matched to the ENTSO-E daily series",
                             clean_table="balancing/contracted_reserves_ch"))
            continue
        r = dict(column=c, source=" / ".join(sub["source"].dropna().astype(str).unique()),
                 source_series=" / ".join(sub["source_series"].dropna().astype(str).unique()),
                 notes=" | ".join(sub["notes"].dropna().astype(str).unique()),
                 clean_table=" + ".join(f"{t}/auction_blocks" + ("_rl" if t == "regelleistung" else "") for t in sub["_t"]))
        rows.append(r)
    dd = pd.DataFrame(rows)
    dd["table"] = "auction_blocks"
    dd["class"] = np.where(dd["column"].isin(AB_KEYS), "key",
                           np.where(dd["column"].str.startswith("regime_"), "regime",
                                    np.where(dd["column"].str.startswith("flag_"), "flag",
                                             np.where(dd["column"].str.contains("xchk"), "xchk", "data"))))
    dd["nan_share"] = dd["column"].map(lambda c: round(float(ab[c].isna().mean()), 5))
    dd["start_utc"] = ab["block_start_utc"].min()
    dd["end_utc"] = ab["block_end_utc"].max()
    return ab, dd.reindex(columns=DICT_COLS)


# ---------------------------------------------------------------- validation
def expected_rows_per_year() -> dict[int, int]:
    s0 = pd.Timestamp(C.SAMPLE_START, tz=TZ)
    s1 = C.end_utc()
    out = {}
    for y in range(s0.year, pd.Timestamp(C.CUTOFF).year + 1):
        a = max(pd.Timestamp(f"{y}-01-01", tz=TZ).tz_convert("UTC"), C.start_utc())
        b = min(pd.Timestamp(f"{y + 1}-01-01", tz=TZ).tz_convert("UTC"), s1)
        out[y] = int((b - a) / QH)
    return out


def validate_master(m: pd.DataFrame, dd: pd.DataFrame, info: dict, rep: Report) -> None:
    rep("\nValidation: master_15min")
    g = K.grid()
    exp = expected_rows_per_year()
    got = m["ts_local"].dt.year.value_counts().sort_index().to_dict()
    rep.check(got == exp, f"rows per local year match the DST-aware grid ({len(m):,} rows)")
    if got != exp:
        rep(f"        expected {exp}\n        got      {got}")
    rep.check(len(m) == 409_052, "total rows = 409,052")
    rep.check(m.index.equals(g.index), "ts_utc equals common.grid() (sorted, complete)")
    rep.check(not m.index.duplicated().any(), "no duplicate ts_utc")
    rep.check(m.index.min() >= C.start_utc() and m.index.max() < C.end_utc(), "no rows outside [SAMPLE_START, CUTOFF]")
    bad = K.check_names(m.columns)
    rep.check(not bad, "all column names follow the grammar" + (f": {bad}" if bad else ""))
    miss = sorted(set(m.columns) - set(dd["column"]))
    rep.check(not miss, "every column in the data dictionary" + (f": {miss}" if miss else ""))
    nodesc = dd.loc[dd["source"].isna(), "column"].tolist()
    rep.check(not nodesc, "every dictionary row has a source" + (f": {nodesc}" if nodesc else ""))
    absent = [c for c in REGIME_FIRST if c not in m.columns]
    rep.check(not absent, "all regime columns present" + (f": missing {absent}" if absent else ""))
    nan_reg = [c for c in REGIME_FIRST if c in m.columns and m[c].isna().any()]
    rep.check(not nan_reg, "regime columns without NaN" + (f": {nan_reg}" if nan_reg else ""))
    flag_bad = [c for c in m.columns if c.startswith("flag_") and m[c].dtype != bool]
    rep.check(not flag_bad, "flag columns are boolean" + (f": {flag_bad}" if flag_bad else ""))
    # unchanged columns keep their clean-layer NaN share
    mism = []
    for key, d in info["dicts"].items():
        for c, r in d.items():
            if c in info["unchanged"] and pd.notna(r.get("nan_share")):
                share = float(m[c].isna().mean())
                if abs(share - float(r["nan_share"])) > 2e-5:
                    mism.append(f"{c} ({key}: {r['nan_share']} -> {share:.5f})")
    rep.check(not mism, f"NaN share of {len(info['unchanged'])} copied columns = clean layer" + (f": {mism[:10]}" if mism else ""))
    unlinked = [f for f in m.columns if f.startswith("flag_") and f not in FLAG_LINKS]
    rep(f"  info  flags without a FLAG_LINKS entry (dictionary `flags` stays empty): {unlinked or 'none'}")
    n_cls = dd["class"].value_counts().to_dict()
    mem = m.memory_usage(deep=True).sum() / 1e9
    rep(f"  info  columns by class: {n_cls}; in memory {mem:.2f} GB")


def nan_reports(m: pd.DataFrame, rep: Report) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NaN share per year, and NaN runs >= RUN_MIN_QH inside each column's own span vs the gap list."""
    year = m["ts_local"].dt.year.to_numpy()
    num = [c for c in m.columns if pd.api.types.is_float_dtype(m[c])]
    by_year = pd.DataFrame({c: pd.Series(m[c].isna().to_numpy()).groupby(year).mean() for c in num}).T.round(4)
    by_year.index.name = "column"

    gl = pd.read_parquet(C.MASTER_DIR / "gap_list.parquet")
    gl = gl[gl["kind"] != "head"]
    gs, ge = K.to_utc(gl["start_utc"]).asi8, K.to_utc(gl["end_utc"]).asi8
    src_of = {}
    for domain, name, src in GRID_TABLES:
        src_of[f"{domain}/{name}"] = src
    dd_src = {}
    runs = []
    t = m.index.asi8
    for c in num:
        if c.startswith(("src_",)):
            continue
        na = m[c].isna().to_numpy()
        if na.all() or not na.any():
            continue
        valid = np.flatnonzero(~na)
        lo, hi = valid[0], valid[-1]
        inner = na[lo:hi + 1].astype(np.int8)
        dif = np.diff(np.concatenate([[0], inner, [0]]))
        st, en = np.flatnonzero(dif == 1), np.flatnonzero(dif == -1)
        keep = (en - st) >= RUN_MIN_QH
        for a, b in zip(st[keep] + lo, en[keep] + lo):
            runs.append((c, a, b))
    rows = []
    area_tok = {"ch": "CH", "de_lu": "DE_LU", "de": "DE", "de_at_lu": "DE_AT_LU", "at": "AT", "fr": "FR",
                "it_nord": "IT_NORD", "it": "IT", "de_amprion": "DE", "ce": ""}
    series = gl["series"].astype(str).to_numpy()
    srcs = gl["source"].astype(str).to_numpy()
    for c, a, b in runs:
        t0, t1 = t[a], t[b - 1]
        hit = (gs <= t1) & (ge >= t0)
        _, area, _, _ = parse_name(c)
        a1 = area.split("->")[0] if area else ""
        tok = area_tok.get(a1, "")
        if tok:
            hit &= np.char.find(series.astype("U"), tok) >= 0
        matched = sorted(set(series[hit]))[:5]
        rows.append(dict(column=c, run_start_utc=pd.Timestamp(t0, tz="UTC"), run_end_utc=pd.Timestamp(t1, tz="UTC"),
                         n_qh=b - a, days=round((b - a) / 96, 2), explained=bool(hit.any()),
                         gap_list_series="; ".join(matched)))
    nr = pd.DataFrame(rows, columns=["column", "run_start_utc", "run_end_utc", "n_qh", "days", "explained", "gap_list_series"])
    n_un = int((~nr["explained"]).sum()) if len(nr) else 0
    rep(f"\nNaN runs >= {RUN_MIN_QH} QH inside a column's own span: {len(nr):,} runs in "
        f"{nr['column'].nunique() if len(nr) else 0} columns; {n_un:,} without a gap-list match (area-token heuristic)")
    if n_un:
        top = nr[~nr["explained"]].groupby("column")["days"].agg(["count", "sum"]).sort_values("sum", ascending=False).head(15)
        rep("  largest unexplained (runs, days):")
        for c, r in top.iterrows():
            rep(f"    {c:55s} {int(r['count']):4d} runs  {r['sum']:8.1f} days")
    return by_year, nr


def validate_blocks(ab: pd.DataFrame, rep: Report) -> None:
    rep("\nValidation: auction_blocks")
    nokey = [k for k in AB_KEYS if k != "tender_series"]
    rep.check(ab[nokey].notna().all().all(), f"keys complete ({len(ab):,} blocks)")
    kk = ab[AB_KEYS].assign(tender_series=ab["tender_series"].fillna(-1))
    rep.check(not kk.duplicated().any(), "no duplicate blocks (key incl. tender_series)")
    rep.check(bool((ab["block_end_utc"] > ab["block_start_utc"]).all()), "block_end > block_start")
    over = 0
    detail = []
    for k, x in ab.assign(tender_series=ab["tender_series"].fillna(-1)).groupby(
            ["market", "product", "direction", "procurement", "tender_series"]):
        x = x.sort_values("block_start_utc")
        o = x["block_start_utc"].to_numpy()[1:] < x["block_end_utc"].to_numpy()[:-1]
        if o.any():
            over += int(o.sum())
            detail.append(f"{k}: {int(o.sum())}")
    rep.check(over == 0, "no overlapping blocks per (market, product, direction, procurement, tender_series)" + (f": {detail}" if detail else ""))
    inside = (ab["block_end_utc"] > C.start_utc()) & (ab["block_start_utc"] < C.end_utc())
    rep.check(bool(inside.all()), "every block overlaps the sample (covered by the grid)")
    edge = (ab["block_start_utc"] < C.start_utc()) | (ab["block_end_utc"] > C.end_utc())
    rep.check(bool((ab.loc[edge, "partial_in_sample"]).all()), f"blocks cut by the sample edges marked partial_in_sample ({int(edge.sum())})")
    regs = [c for c in ab.columns if c.startswith("regime_")]
    rep.check(len(regs) == 7 and not ab[regs].isna().any().any(), f"{len(regs)} regime columns without NaN")
    rep(f"  info  blocks per market: {ab['market'].value_counts().to_dict()}; partial_in_sample: {int(ab['partial_in_sample'].sum())}")
    x = ab[ab["price_xchk_entsoe"].notna()]
    if len(x):
        agree = (np.abs(x["price_xchk_entsoe"] - x["price_bid_vwap"]) <= 0.05 * np.abs(x["price_bid_vwap"]).clip(lower=1))
        rep(f"  info  ENTSO-E price cross-check present for {len(x):,} Swissgrid blocks; within 5 % of price_bid_vwap: "
            f"{agree.mean():.1%} (units as published; see dictionary)")


# ---------------------------------------------------------------- writing
def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="write the master files (default: dry run)")
    args = ap.parse_args()
    t0 = time.time()
    rep = Report()
    rep(f"build_master.py  {pd.Timestamp.now(tz=TZ):%Y-%m-%d %H:%M}  sample {C.SAMPLE_START} -> {C.CUTOFF} ({TZ})")
    gl = C.MASTER_DIR / "gap_list.parquet"
    newest = max(p.stat().st_mtime for p in (C.CLEAN_DIR / "build_gap_list.py", *C.DATA_DIR.glob("*/*.parquet"))
                 if p.parent.name != "master")
    rep.check(gl.exists() and gl.stat().st_mtime >= newest,
              "gap list newer than build_gap_list.py and every clean table (else rerun build_gap_list.py --write)")
    rep(f"  info  not in the master: {NOT_IN_MASTER}")

    m, dd, info = build_master(rep)
    dd = finish_dictionary(m, dd)
    validate_master(m, dd, info, rep)
    by_year, runs = nan_reports(m, rep)

    rep("\nBuilding auction_blocks")
    ab, dab = build_auction_blocks(rep)
    validate_blocks(ab, rep)
    data_dict = pd.concat([dd, dab], ignore_index=True)

    rep(f"\nmaster_15min: {m.shape[0]:,} rows x {m.shape[1] + 1} columns (incl. ts_utc); "
        f"auction_blocks: {ab.shape[0]:,} x {ab.shape[1]}; {time.time() - t0:.0f} s")
    if rep.failures:
        rep(f"\n{len(rep.failures)} check(s) FAILED -> nothing written")
        sys.exit(1)
    if not args.write:
        rep("\nDry run: all checks passed. Rerun with --write to save.")
        return
    C.MASTER_DIR.mkdir(parents=True, exist_ok=True)
    atomic_parquet(m.reset_index(), C.MASTER_DIR / "master_15min.parquet")
    atomic_parquet(ab, C.MASTER_DIR / "auction_blocks.parquet")
    data_dict.to_csv(C.MASTER_DIR / "data_dictionary.csv", index=False)
    by_year.to_csv(C.MASTER_DIR / "nan_share_by_year.csv")
    runs.to_csv(C.MASTER_DIR / "nan_runs_unexplained.csv", index=False)
    rep(f"Written to {C.MASTER_DIR}: master_15min.parquet, auction_blocks.parquet, data_dictionary.csv, "
        "nan_share_by_year.csv, nan_runs_unexplained.csv, build_master_report.txt")
    (C.MASTER_DIR / "build_master_report.txt").write_text("\n".join(rep.lines) + "\n")


if __name__ == "__main__":
    main()
