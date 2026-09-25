"""
swissgrid_auctions_parse.py — Swissgrid FCR/aFRR/mFRR capacity auction results → tidy parquet.

Input : Swissgrid/Manual_Download/Tenders/**/  *PRL*SRL*TRL*Ergebnis*.csv  and  *vorgezogene*ergebnis*.csv
Output: Swissgrid/Auctions/Data/
          bids/<product>/<delivery_year>.parquet   one row per published bid
          auctions.parquet                          one row per (auction, direction, delivery block)
          _parse_manifest.csv                       per source file: rows, drops, checks

Run from the thesis root:  python Swissgrid/Auctions/swissgrid_auctions_parse.py

Notes on the source data (from the 2015–2027 profile):
- PRL = FCR (EUR), SRL = aFRR (CHF), TRL = mFRR capacity (CHF). Currency is kept native; no FX here.
- PRL/SRL files contain ACCEPTED bids only; TRL files also contain rejected bids (awarded = 0).
- Direction comes from the ID (TRL+/TRL- until Sep 2025) or the description (SRL+/-, UP/DOWN).
- Delivery period: KWnn = ISO week (Mon–Sun); YY_MM_DD = one day; "HH:MM bis HH:MM" = 4h block.
- Pre-2019 files have no Angebotspreis; there Preis is the bid (pay-as-bid). From 2019 on,
  Angebotspreis = bid and Preis = settlement price (marginal for FCR).
- Year files overlap at the boundary (e.g. PRL_15_KW53 is in both 2015 and 2016). Identical bid
  rows are legitimate within an auction, so dedup is per auction: keep the copy from the file
  with the most rows (tie: file year = delivery year, then later file). Copies with differing
  row counts are counted as conflicts in the manifest.
- Two revised IDs exist: TRL+_18_03_20-Neu (a normal auction, no original) and TRL+_19_KW42_KORR
  (one corrected bid, aggregated into TRL+_19_KW42). Bids keep the suffix in `revision`.
"""
from __future__ import annotations

import argparse
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"
DEFAULT_IN = Path("Swissgrid/Manual_Download/Tenders")
DEFAULT_OUT = Path("Swissgrid/Auctions/Data")

# Source columns in order; pandas renames the repeated "Einheit" to Einheit, Einheit.1, ...
COLMAP = {
    "Ausschreibung": "auction_id",
    "Beschreibung": "description",
    "Angebotenes Volumen": "offered_mw",
    "Zugesprochenes Volumen": "awarded_mw",
    "Leistungspreis": "capacity_price",      # per MW per product period (week / day / 4h)
    "Einheit.2": "capacity_price_unit",
    "Kosten": "cost",
    "Preis": "price_mwh_raw",                 # normalised to per MW per hour ("MWh*")
    "Einheit.4": "price_mwh_unit",
    "Land": "country",
    "Angebotspreis": "bid_price_mwh_raw",     # 2019+ only
    "Teilbarkeit": "divisible",
}
NUMERIC = ["offered_mw", "awarded_mw", "capacity_price", "cost", "price_mwh_raw", "bid_price_mwh_raw"]

ID_RE = re.compile(
    r"^(?P<prod_raw>PRL|SRL|TRL[+-]?)_(?P<yy>\d{2})_"
    r"(?:KW(?P<kw>\d{2})|(?P<mm>\d{2})_(?P<dd>\d{2}))"
    r"(?:_S(?P<series>\d+))?"
    r"(?:[-_](?P<revision>Neu|KORR))?$",       # re-run / corrected auctions (2018, 2019)
    re.IGNORECASE,
)
BLOCK_RE = re.compile(r"(\d{2}):(\d{2})\s*bis\s*(\d{2}):(\d{2})")
FILE_YEAR_RE = re.compile(r"(20\d{2})[-_](?:PRL|vorgezogene)", re.IGNORECASE)
PRODUCT = {"PRL": "FCR", "SRL": "aFRR", "TRL": "mFRR", "TRL+": "mFRR", "TRL-": "mFRR"}


# --------------------------------------------------------------------------- reading
def find_sources(root: Path) -> list[Path]:
    pats = ["*PRL*SRL*TRL*rgebnis*.csv", "*vorgezogene*rgebnis*.csv"]
    files = {p for pat in pats for p in root.rglob(pat)}
    return sorted(files)


def read_one(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", keep_default_na=False)
    df = raw.rename(columns=COLMAP)
    for c in COLMAP.values():
        if c not in df.columns:
            df[c] = pd.NA
    df = df[list(COLMAP.values())].replace("", pd.NA)

    info = {"file": path.name, "rows_read": len(df)}
    for c in NUMERIC:
        s = df[c].astype("string").str.replace("'", "", regex=False).str.strip()
        num = pd.to_numeric(s, errors="coerce")
        info[f"nan_{c}"] = int((num.isna() & s.notna()).sum())   # unparseable, not just missing
        df[c] = num.astype("float64")

    m = FILE_YEAR_RE.search(path.name)
    df["file_year"] = int(m.group(1)) if m else np.nan
    df["source_file"] = path.name
    return df, info


# --------------------------------------------------------------------------- derivations
def delivery_window(auction_id: str, description: str):
    """Return (prod_raw, series, revision, start, end, weekly, flag) for one auction/description."""
    m = ID_RE.match(auction_id or "")
    if not m:
        return (None, None, "", pd.NaT, pd.NaT, None, "id_unparsed")
    g = m.groupdict()
    year = 2000 + int(g["yy"])
    flag = ""
    if g["kw"]:
        day0, n_days, weekly = date.fromisocalendar(year, int(g["kw"]), 1), 7, True
    else:
        day0, n_days, weekly = date(year, int(g["mm"]), int(g["dd"])), 1, False

    base = datetime.combine(day0, datetime.min.time())
    b = BLOCK_RE.search(description or "")
    if b and not weekly:
        h1, m1, h2, m2 = map(int, b.groups())
        start = base + timedelta(hours=h1, minutes=m1)
        end = base + timedelta(hours=h2, minutes=m2)          # "24:00" rolls to next day
    else:
        if b and weekly:
            flag = "weekly_with_block"
        start, end = base, base + timedelta(days=n_days)
    # block starts are 00/04/08/... never inside the 02:00–03:00 DST gap
    start = pd.Timestamp(start).tz_localize(TZ)
    end = pd.Timestamp(end).tz_localize(TZ)
    rev = (g["revision"] or "").upper()
    return (g["prod_raw"].upper(), int(g["series"] or 0), rev, start, end, weekly, flag)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    keys = df[["auction_id", "description"]].drop_duplicates()
    win = pd.DataFrame(
        [delivery_window(a, d) for a, d in keys.itertuples(index=False)],
        columns=["prod_raw", "tender_series", "revision", "delivery_start", "delivery_end", "weekly", "flag"],
        index=keys.index,
    )
    keys = pd.concat([keys, win], axis=1)
    df = df.merge(keys, on=["auction_id", "description"], how="left")

    df["product"] = df["prod_raw"].map(PRODUCT)
    desc = df["description"].fillna("").str.upper()
    up = (df["prod_raw"] == "TRL+") | desc.str.contains(r"\bUP\b|SRL\+", regex=True)
    down = (df["prod_raw"] == "TRL-") | desc.str.contains(r"\bDOWN\b|SRL-", regex=True)
    df["direction"] = np.select([up & ~down, down & ~up], ["up", "down"], default="sym")

    df["currency"] = df["capacity_price_unit"].str.extract(r"^(EUR|CHF)", expand=False)
    df.loc[~df["price_mwh_unit"].fillna("").str.endswith("MWh*"), "flag"] = (
        df["flag"].fillna("") + ";price_unit_anomaly"
    ).str.strip(";")

    dur = (df["delivery_end"] - df["delivery_start"]).dt.total_seconds() / 3600
    df["duration_h"] = dur.astype("float64")                # 3/5h blocks and 167/169h weeks on DST days
    df["delivery_year"] = df["delivery_start"].dt.year

    # Bid vs settlement price, both per MW per hour
    has_bid = df["bid_price_mwh_raw"].notna()
    df["bid_price_mwh"] = df["bid_price_mwh_raw"].where(has_bid, df["price_mwh_raw"])
    df["settle_price_mwh"] = df["price_mwh_raw"]
    df["accepted"] = df["awarded_mw"] > 0
    df["tender_series"] = df["tender_series"].astype("Int64")
    return df.drop(columns=["capacity_price_unit", "price_mwh_unit", "price_mwh_raw", "bid_price_mwh_raw"])


# --------------------------------------------------------------------------- dedup
def dedup_auctions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    key = ["auction_id", "description"]
    src = df.groupby(key + ["source_file"], dropna=False).agg(
        n=("auction_id", "size"),
        file_year=("file_year", "first"),
        delivery_year=("delivery_year", "first"),
    ).reset_index()
    order = {f: i for i, f in enumerate(sorted(df["source_file"].unique()))}
    src = src.assign(
        year_match=(src["file_year"] == src["delivery_year"]).astype(int),
        order=src["source_file"].map(order),
    ).sort_values(key + ["n", "year_match", "order"])
    keep = src.groupby(key, dropna=False).tail(1)[key + ["source_file"]]
    flagged = src.merge(keep, on=key + ["source_file"], how="left", indicator=True)
    dropped = flagged[flagged["_merge"] == "left_only"].groupby("source_file").size()
    conflicts = int((src.groupby(key, dropna=False)["n"].nunique() > 1).sum())
    return df.merge(keep, on=key + ["source_file"], how="inner"), dropped, conflicts


# --------------------------------------------------------------------------- aggregates
def build_auctions(bids: pd.DataFrame) -> pd.DataFrame:
    keys = ["product", "direction", "auction_id", "tender_series", "delivery_start",
            "delivery_end", "duration_h", "weekly", "currency"]
    b = bids.copy()
    # TRL+_19_KW42_KORR corrects one bid of TRL+_19_KW42 (1 bid vs 36): aggregate them together.
    # "-Neu" (TRL+_18_03_20-Neu) has no original and stays a normal auction.
    korr = b["revision"] == "KORR"
    b["auction_id"] = b["auction_id"].astype(str)
    b.loc[korr, "auction_id"] = b.loc[korr, "auction_id"].str.replace(r"[-_]KORR$", "", case=False, regex=True)
    b["w_bid"] = b["bid_price_mwh"] * b["awarded_mw"]
    b["ch"] = b["country"].isna() | (b["country"] == "CH")    # pre-2017 has no country column
    b["awarded_mw_ch"] = b["awarded_mw"].where(b["ch"], 0.0)
    b["settle_ch"] = b["settle_price_mwh"].where(b["ch"] & b["accepted"])

    allb = b.groupby(keys, dropna=False, observed=True).agg(
        n_bids=("auction_id", "size"),
        offered_mw=("offered_mw", "sum"),
    )
    acc = b[b["accepted"]].groupby(keys, dropna=False, observed=True).agg(
        n_accepted=("auction_id", "size"),
        awarded_mw=("awarded_mw", "sum"),
        awarded_mw_ch=("awarded_mw_ch", "sum"),
        w_bid=("w_bid", "sum"),
        bid_min=("bid_price_mwh", "min"),
        bid_max=("bid_price_mwh", "max"),          # marginal accepted bid
        settle_max=("settle_price_mwh", "max"),
        settle_ch=("settle_ch", "mean"),           # FCR: CH clearing price
        n_settle_prices=("settle_price_mwh", "nunique"),
        cost=("cost", "sum"),
    )
    out = allb.join(acc, how="left").reset_index()
    out["bid_vwap"] = out["w_bid"] / out["awarded_mw"]
    return out.drop(columns="w_bid").sort_values(["product", "direction", "delivery_start"])


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    files = find_sources(args.inp)
    if not files:
        raise SystemExit(f"No auction files found under {args.inp.resolve()}")

    frames, infos = [], []
    for p in files:
        df, info = read_one(p)
        frames.append(add_derived(df))              # per file keeps peak memory down
        infos.append(info)
        print(f"read {p.name:<70} {info['rows_read']:>9,} rows")

    bids = pd.concat(frames, ignore_index=True)
    del frames
    bids, dropped, conflicts = dedup_auctions(bids)

    # checks: normalised price should equal capacity_price / duration_h
    # (source prices are rounded to 2 decimals → absolute + relative tolerance)
    implied = bids["capacity_price"] / bids["duration_h"]
    bids["norm_ok"] = (implied - bids["bid_price_mwh"]).abs() <= 0.006 + 0.01 * bids["bid_price_mwh"].abs()

    man = pd.DataFrame(infos).set_index("file")
    grp = bids.groupby("source_file")
    man["rows_kept"] = grp.size()
    man["auctions_kept"] = grp["auction_id"].nunique()
    man["auction_copies_dropped"] = dropped
    man["id_unparsed"] = grp["flag"].apply(lambda s: s.fillna("").str.contains("id_unparsed").sum())
    man["unit_anomalies"] = grp["flag"].apply(lambda s: s.fillna("").str.contains("unit_anomaly").sum())
    man["norm_ok_share"] = grp["norm_ok"].mean().round(4)
    man["delivery_min"] = grp["delivery_start"].min()
    man["delivery_max"] = grp["delivery_end"].max()
    man = man.fillna({"rows_kept": 0, "auctions_kept": 0, "auction_copies_dropped": 0,
                      "id_unparsed": 0, "unit_anomalies": 0})

    # write
    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["product", "direction", "auction_id", "tender_series", "revision", "description",
            "delivery_start", "delivery_end", "duration_h", "weekly", "country", "currency",
            "offered_mw", "awarded_mw", "accepted", "capacity_price", "cost",
            "bid_price_mwh", "settle_price_mwh", "divisible", "flag", "norm_ok", "source_file"]
    bids = bids[cols].copy()
    for c in ["product", "direction", "country", "currency", "divisible", "source_file"]:
        bids[c] = bids[c].astype("category")
    for (prod, yr), g in bids.groupby(["product", bids["delivery_start"].dt.year], observed=True):
        d = args.out / "bids" / str(prod)
        d.mkdir(parents=True, exist_ok=True)
        g.reset_index(drop=True).to_parquet(d / f"{int(yr)}.parquet", index=False)

    auctions = build_auctions(bids[bids["flag"].fillna("").ne("id_unparsed")])
    auctions.to_parquet(args.out / "auctions.parquet", index=False)
    man.to_csv(args.out / "_parse_manifest.csv")

    print(f"\n{len(bids):,} bids, {len(auctions):,} auction-blocks → {args.out}")
    print(f"boundary auctions with conflicting copies: {conflicts}")
    print(f"revised-auction bids (-Neu/_KORR): {int((bids['revision'] != '').sum())}")
    print(man[["rows_read", "rows_kept", "auction_copies_dropped", "id_unparsed",
               "unit_anomalies", "norm_ok_share"]].to_string())


if __name__ == "__main__":
    main()
