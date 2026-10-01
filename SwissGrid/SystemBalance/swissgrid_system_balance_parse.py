#!/usr/bin/env python3
"""
Swissgrid system-balance CSV parser (2026+ quarter-hourly CSVs from swissgrid.ch).

Run from the thesis root:
    python Swissgrid/SystemBalance/swissgrid_system_balance_parse.py
    python Swissgrid/SystemBalance/swissgrid_system_balance_parse.py --years 2026 --force

Needs: pandas, pyarrow

SAFETY: input files are opened read-only. The script writes ONLY under the
output directory (default Swissgrid/SystemBalance/Data/) and never deletes
anything. Existing year files are skipped unless --force is given; with
--force they are replaced atomically (write to temp file, then rename).

Sources (searched recursively under Swissgrid/Manual_Download/):
  1. Ausgleichsenergie-und-Regelenergie-YYYY.csv  -> control_energy/
  2. Grenzfluesse-YYYY.csv                        -> cross_border/
  3. control-area-balance-YYYY.csv                -> control_area_balance/

  Format (verified on the 2026 files downloaded 2026-09-24):
  - ';' separated, '.' decimal, values like ".317922" (no leading zero).
  - Files 1 + 2: UTF-8 with BOM, header "Date Time", LOCAL wall-clock
    labels "dd.mm.yyyy HH:MM" = interval START (checked: aFRR activated MW
    equals file 3, which is in UTC, at every quarter-hour).
  - File 3: plain ASCII, header "Date Time [UTC]" -> UTC labels.
  - Column letters (C:, D:, ...) prefix the headers in files 1 + 2; columns
    are mapped by letter AND checked against a keyword, so a re-ordered
    file fails loudly instead of being mis-mapped.

  Cost columns I/J/K and R/S/T in file 1 are CUMULATIVE WEEKLY totals:
  - unit is kEUR although the header says [EUR]
    (check: diff x 1000 == activated MW x avg price x 0.25 h, exact);
  - the sum restarts every Monday 00:00 local time;
  - an empty cell at the start of a week means "nothing activated yet" (0),
    an empty cell later in the week means "unchanged" (carry forward).
  The parser turns them into per-quarter-hour costs in EUR. The first row
  of a file that does not start on a Monday 00:00 cannot be differenced;
  it is filled with activated MW x price x 0.25 h (flagged in the manifest).

  Activated MW (E/F, N/O) is empty when nothing was activated -> set to 0
  (file 3 has 0 there). Prices of non-activated quarter-hours stay NaN.

  If the same dataset-year exists more than once (e.g. an older partial
  Grenzfluesse-2026.csv in another folder), the copy with the latest last
  timestamp wins; the manifest reports how many overlapping cells differ.

Checks per file (a failing file is reported and skipped, others continue):
  - every expected column present, keyword check per column
  - continuous 15-min UTC grid, no gaps/duplicates (92/100 rows on DST days)
  - control energy: K == I + J and T == R + S (cumulative), differenced
    cost == MW x price x 0.25 h; reported in the manifest

Output (timestamp = interval START, datetime64[us, Europe/Zurich]):
  Data/control_energy/<year>.parquet
  Data/cross_border/<year>.parquet
  Data/control_area_balance/<year>.parquet
  Data/variables.csv        column -> source header, unit, notes
  Data/_parse_manifest.csv  one row per (dataset, year)
  Load all years: pd.read_parquet("Swissgrid/SystemBalance/Data/control_energy/")
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"
QH = pd.Timedelta(minutes=15)

# dataset -> (file regex, timestamp column, labels in UTC?, column spec)
# column spec: (source letter or exact header, keyword that must appear, output name, unit)
CONTROL_ENERGY_COLS = [
    ("C", "aFRR+ Angebotene", "afrr_pos_offered_mw", "MW"),
    ("D", "aFRR- Angebotene", "afrr_neg_offered_mw", "MW"),
    ("E", "aFRR+ Aktivierte Menge [MW]", "afrr_pos_activated_mw", "MW"),
    ("F", "aFRR- Aktivierte Menge [MW]", "afrr_neg_activated_mw", "MW"),
    ("G", "aFRR+ Durchschnittspreis", "afrr_pos_price_eur_mwh", "EUR/MWh"),
    ("H", "aFRR- Durchschnittspreis", "afrr_neg_price_eur_mwh", "EUR/MWh"),
    ("I", "aFRR+ Kosten", "_cum_afrr_pos", "kEUR cumulative weekly"),
    ("J", "aFRR- Kosten", "_cum_afrr_neg", "kEUR cumulative weekly"),
    ("K", "aFRR+/- Kosten", "_cum_afrr_tot", "kEUR cumulative weekly"),
    ("L", "mFRR+ Angebotene", "mfrr_pos_offered_mw", "MW"),
    ("M", "mFRR- Angebotene", "mfrr_neg_offered_mw", "MW"),
    ("N", "mFRR+ Aktivierte Menge [MW]", "mfrr_pos_activated_mw", "MW"),
    ("O", "mFRR- Aktivierte Menge [MW]", "mfrr_neg_activated_mw", "MW"),
    ("P", "mFRR+ Durchschnittspreis", "mfrr_pos_price_eur_mwh", "EUR/MWh"),
    ("Q", "mFRR- Durchschnittspreis", "mfrr_neg_price_eur_mwh", "EUR/MWh"),
    ("R", "mFRR+ Kosten", "_cum_mfrr_pos", "kEUR cumulative weekly"),
    ("S", "mFRR- Kosten", "_cum_mfrr_neg", "kEUR cumulative weekly"),
    ("T", "mFRR+/- Kosten", "_cum_mfrr_tot", "kEUR cumulative weekly"),
]
# cumulative column -> (output cost column, activated MW column, price column)
COST_SPEC = {
    "_cum_afrr_pos": ("afrr_pos_cost_eur", "afrr_pos_activated_mw", "afrr_pos_price_eur_mwh"),
    "_cum_afrr_neg": ("afrr_neg_cost_eur", "afrr_neg_activated_mw", "afrr_neg_price_eur_mwh"),
    "_cum_mfrr_pos": ("mfrr_pos_cost_eur", "mfrr_pos_activated_mw", "mfrr_pos_price_eur_mwh"),
    "_cum_mfrr_neg": ("mfrr_neg_cost_eur", "mfrr_neg_activated_mw", "mfrr_neg_price_eur_mwh"),
}
TOTALS = {"_cum_afrr_tot": ("afrr_total_cost_eur", "_cum_afrr_pos", "_cum_afrr_neg"),
          "_cum_mfrr_tot": ("mfrr_total_cost_eur", "_cum_mfrr_pos", "_cum_mfrr_neg")}

CROSS_BORDER_COLS = [
    ("B", "Intraday NTC Schweiz > Österreich", "ntc_id_ch_at_mw", "MW"),
    ("C", "Intraday NTC Österreich > Schweiz", "ntc_id_at_ch_mw", "MW"),
    ("D", "Intraday NTC Schweiz > Deutschland", "ntc_id_ch_de_mw", "MW"),
    ("E", "Intraday NTC Deutschland > Schweiz", "ntc_id_de_ch_mw", "MW"),
    ("F", "Intraday NTC Schweiz > Frankreich", "ntc_id_ch_fr_mw", "MW"),
    ("G", "Intraday NTC Frankreich > Schweiz", "ntc_id_fr_ch_mw", "MW"),
    ("H", "Intraday NTC Schweiz > Italien", "ntc_id_ch_it_mw", "MW"),
    ("I", "Intraday NTC Italien > Schweiz", "ntc_id_it_ch_mw", "MW"),
    ("J", "Österreich Netto", "comm_flow_net_at_mw", "MW"),
    ("K", "Deutschland Netto", "comm_flow_net_de_mw", "MW"),
    ("L", "Frankreich Netto", "comm_flow_net_fr_mw", "MW"),
    ("M", "Italien Netto", "comm_flow_net_it_mw", "MW"),
    ("N", "Summe alle Grenzen", "comm_flow_net_total_mw", "MW"),
    ("O", "Spot Österreich-Schweiz", "spot_spread_at_ch_eur_mwh", "EUR/MWh"),
    ("P", "Spot Deutschland-Schweiz", "spot_spread_de_ch_eur_mwh", "EUR/MWh"),
    ("Q", "Spot Frankreich-Schweiz", "spot_spread_fr_ch_eur_mwh", "EUR/MWh"),
    ("R", "Spot Italien Nord-Schweiz", "spot_spread_itn_ch_eur_mwh", "EUR/MWh"),
]

CONTROL_AREA_COLS = [
    ("Abgedeckte Bedarf der aFRR+", "aFRR+", "afrr_pos_mw", "MW"),
    ("Abgedeckte Bedarf der aFRR-", "aFRR-", "afrr_neg_mw", "MW"),
    ("Abgedeckte Bedarf der SA mFRR+", "SA mFRR+", "mfrr_sa_pos_mw", "MW"),
    ("Abgedeckte Bedarf der SA mFRR-", "SA mFRR-", "mfrr_sa_neg_mw", "MW"),
    ("Abgedeckte Bedarf der DA mFRR+", "DA mFRR+", "mfrr_da_pos_mw", "MW"),
    ("Abgedeckte Bedarf der DA mFRR-", "DA mFRR-", "mfrr_da_neg_mw", "MW"),
    ("NRV+ (Import)", "NRV+", "nrv_pos_import_mw", "MW"),
    ("NRV- (Export)", "NRV-", "nrv_neg_export_mw", "MW"),
    ("FRCE+ (Import)", "FRCE+", "frce_pos_import_mw", "MW"),
    ("FRCE- (Export)", "FRCE-", "frce_neg_export_mw", "MW"),
    ("Total System Imbalance", "Imbalance", "system_imbalance_mw", "MW"),
    ("AE-Preis", "AE-Preis", "aep_eur_mwh", "EUR/MWh (source ct/kWh x 10)"),
]

DATASETS = {
    "control_energy": (re.compile(r"Ausgleichsenergie-und-Regelenergie-(\d{4})\.csv$", re.I),
                       "Date Time", False, CONTROL_ENERGY_COLS),
    "cross_border": (re.compile(r"Grenzfluesse-(\d{4})\.csv$", re.I),
                     "Date Time", False, CROSS_BORDER_COLS),
    "control_area_balance": (re.compile(r"control-area-balance-(\d{4})\.csv$", re.I),
                             "Date Time [UTC]", True, CONTROL_AREA_COLS),
}

NOTES = {
    "afrr_pos_activated_mw": "empty in source = nothing activated -> 0",
    "afrr_neg_activated_mw": "empty in source = nothing activated -> 0",
    "mfrr_pos_activated_mw": "empty in source = nothing activated -> 0",
    "mfrr_neg_activated_mw": "empty in source = nothing activated -> 0",
    "afrr_pos_cost_eur": "per quarter-hour; differenced from weekly cumulative kEUR (col I)",
    "afrr_neg_cost_eur": "per quarter-hour; differenced from weekly cumulative kEUR (col J)",
    "afrr_total_cost_eur": "per quarter-hour; differenced from weekly cumulative kEUR (col K)",
    "mfrr_pos_cost_eur": "per quarter-hour; differenced from weekly cumulative kEUR (col R)",
    "mfrr_neg_cost_eur": "per quarter-hour; differenced from weekly cumulative kEUR (col S)",
    "mfrr_total_cost_eur": "per quarter-hour; differenced from weekly cumulative kEUR (col T)",
    "aep_eur_mwh": "single imbalance price (BG-AEP); equals ImbalancePrices aep_eur_mwh; "
                   "most recent weeks may be preliminary",
    "system_imbalance_mw": "negative = control area short",
}


# ---------------------------------------------------------------------------
def read_csv(path: Path, ts_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0] != ts_col:
        raise ValueError(f"first column is {df.columns[0]!r}, expected {ts_col!r}")
    return df


def map_columns(df: pd.DataFrame, spec, by_letter: bool) -> tuple[pd.DataFrame, list[str]]:
    out, problems = {}, []
    headers = list(df.columns[1:])
    for key, keyword, name, _unit in spec:
        if by_letter:
            hits = [h for h in headers if h.split(":", 1)[0] == key]
        else:
            hits = [h for h in headers if h == key]
        if len(hits) != 1:
            problems.append(f"missing column {key}")
            continue
        h = hits[0]
        if keyword.lower() not in h.lower():
            problems.append(f"column {key} header {h!r} lacks {keyword!r}")
            continue
        s = df[h].str.strip().replace("", np.nan)
        out[name] = pd.to_numeric(s, errors="coerce")
        bad = s.notna() & out[name].isna()
        if bad.any():
            problems.append(f"column {key}: {int(bad.sum())} non-numeric values")
    extra = len(headers) - len(spec)
    if extra:
        problems.append(f"{extra} unexpected extra columns")
    return pd.DataFrame(out), problems


def build_timestamps(labels: pd.Series, utc: bool) -> pd.DatetimeIndex:
    naive = pd.to_datetime(labels.str.strip(), format="%d.%m.%Y %H:%M")
    if utc:
        idx = pd.DatetimeIndex(naive).tz_localize("UTC")
    else:
        idx = pd.DatetimeIndex(naive).tz_localize(TZ, ambiguous="infer", nonexistent="raise")
    idx = idx.tz_convert(TZ)
    step = idx.tz_convert("UTC").to_series().diff().iloc[1:]
    if idx.has_duplicates or not (step == QH).all():
        bad = step[step != QH]
        raise ValueError(f"15-min grid broken at {len(bad)} place(s), first {bad.index[:3].tolist()}")
    return idx.astype(f"datetime64[us, {TZ}]")


def difference_weekly(cum: pd.Series, idx: pd.DatetimeIndex) -> tuple[pd.Series, bool]:
    """Weekly cumulative kEUR -> per-QH EUR. Returns (series, first_row_undefined)."""
    local = idx.tz_localize(None)
    week = pd.Series((local - pd.to_timedelta(local.weekday, unit="D")).normalize(), index=cum.index)
    first_of_week = week.ne(week.shift())
    c = cum.groupby(week).ffill().fillna(0.0)       # empty at week start = 0, later = unchanged
    d = c.groupby(week).diff()
    d[first_of_week] = c[first_of_week]
    file_starts_midweek = not (local[0].weekday() == 0 and local[0].hour == 0 and local[0].minute == 0)
    if file_starts_midweek:
        d.iloc[0] = np.nan
    return d * 1000.0, file_starts_midweek


def parse_control_energy(df: pd.DataFrame, idx, info: dict) -> pd.DataFrame:
    for c in ("afrr_pos_activated_mw", "afrr_neg_activated_mw",
              "mfrr_pos_activated_mw", "mfrr_neg_activated_mw"):
        df[c] = df[c].fillna(0.0)
    for tot, (_out, a, b) in TOTALS.items():
        diff = (df[tot].fillna(0) - df[a].fillna(0) - df[b].fillna(0)).abs().max()
        info[f"check_{tot[5:]}_eq_sum_max_keur"] = round(float(diff), 6)
    for cum, (out, mw, price) in {**COST_SPEC,
                                  **{k: (v[0], None, None) for k, v in TOTALS.items()}}.items():
        cost, midweek = difference_weekly(df[cum], idx)
        if mw is not None:
            direct = df[mw] * df[price].fillna(0) * 0.25
            if midweek:
                cost.iloc[0] = direct.iloc[0]
            err = (cost - direct).abs()
            info[f"check_{out}_vs_mw_x_price_max_eur"] = round(float(err.max()), 3)
            info[f"check_{out}_share_within_1eur"] = round(float((err <= 1).mean()), 5)
        df[out] = cost
        info["first_row_filled_from_mw_x_price"] = midweek
    # totals' first row = pos + neg
    if info.get("first_row_filled_from_mw_x_price"):
        df.loc[df.index[0], "afrr_total_cost_eur"] = df["afrr_pos_cost_eur"].iloc[0] + df["afrr_neg_cost_eur"].iloc[0]
        df.loc[df.index[0], "mfrr_total_cost_eur"] = df["mfrr_pos_cost_eur"].iloc[0] + df["mfrr_neg_cost_eur"].iloc[0]
    return df.drop(columns=[c for c in df.columns if c.startswith("_cum_")])


def parse_file(dataset: str, path: Path) -> tuple[pd.DataFrame, dict]:
    _rx, ts_col, utc, spec = DATASETS[dataset]
    raw = read_csv(path, ts_col)
    vals, problems = map_columns(raw, spec, by_letter=(dataset != "control_area_balance"))
    if problems:
        raise ValueError("; ".join(problems))
    idx = build_timestamps(raw[ts_col], utc)
    vals.index = range(len(vals))
    info: dict = {}
    if dataset == "control_energy":
        vals = parse_control_energy(vals, idx, info)
    if dataset == "control_area_balance":
        vals["aep_eur_mwh"] = vals["aep_eur_mwh"] * 10.0
    vals.insert(0, "timestamp", idx)
    info.update(rows=len(vals), first=str(idx[0]), last=str(idx[-1]),
                nan_cells=int(vals.drop(columns="timestamp").isna().sum().sum()))
    return vals, info


def overlap_diff(a: pd.DataFrame, b: pd.DataFrame) -> int:
    m = a.merge(b, on="timestamp", suffixes=("_a", "_b"))
    n = 0
    for c in a.columns:
        if c == "timestamp":
            continue
        x, y = m[c + "_a"], m[c + "_b"]
        n += int((~(np.isclose(x, y, equal_nan=True))).sum())
    return n


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet.tmp", dir=path.parent)
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)   # only our own temp file
        raise


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv.tmp", dir=path.parent)
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def discover(input_dir: Path) -> dict[tuple[str, int], list[Path]]:
    found: dict[tuple[str, int], list[Path]] = {}
    for p in sorted(input_dir.rglob("*.csv")):
        for ds, (rx, *_rest) in DATASETS.items():
            m = rx.search(p.name)
            if m:
                found.setdefault((ds, int(m.group(1))), []).append(p)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", default="Swissgrid/Manual_Download")
    ap.add_argument("--output", default="Swissgrid/SystemBalance/Data")
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--force", action="store_true", help="re-parse existing years")
    args = ap.parse_args(argv)

    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.is_dir():
        print(f"input folder not found: {in_dir} (run from the thesis root)", file=sys.stderr)
        return 2
    found = discover(in_dir)
    if not found:
        print("no matching CSV files found", file=sys.stderr)
        return 1

    manifest_path = out_dir / "_parse_manifest.csv"
    old = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    rows, n_err = [], 0
    for (ds, year), paths in sorted(found.items()):
        if args.years and year not in args.years:
            continue
        target = out_dir / ds / f"{year}.parquet"
        if target.exists() and not args.force:
            print(f"[skip] {ds} {year} (exists; --force to re-parse)")
            continue
        parsed = []
        for p in paths:
            try:
                df, info = parse_file(ds, p)
                parsed.append((p, df, info))
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"[ERROR] {ds} {year} {p}: {e}")
                rows.append(dict(dataset=ds, year=year, source=str(p), status="error", error=str(e)))
        if not parsed:
            continue
        parsed.sort(key=lambda t: t[1]["timestamp"].iloc[-1])
        src, df, info = parsed[-1]
        others = [(p, overlap_diff(d, df)) for p, d, _ in parsed[:-1]]
        atomic_parquet(df, target)
        row = dict(dataset=ds, year=year, source=str(src), status="ok", **info,
                   superseded_copies="; ".join(f"{p} ({n} differing cells in overlap)" for p, n in others),
                   parsed_at=dt.datetime.now().isoformat(timespec="seconds"))
        rows.append(row)
        extra = f", {len(others)} older copy ignored" if others else ""
        print(f"[ok] {ds} {year}: {info['rows']} rows {info['first']} -> {info['last']}{extra}")

    if rows:
        new = pd.DataFrame(rows)
        if not old.empty:
            keep = ~old.set_index(["dataset", "year"]).index.isin(new.set_index(["dataset", "year"]).index)
            new = pd.concat([old[keep], new], ignore_index=True)
        atomic_csv(new.sort_values(["dataset", "year"]), manifest_path)

    var_rows = []
    for ds, (_rx, _ts, _utc, spec) in DATASETS.items():
        for key, keyword, name, unit in spec:
            if name.startswith("_cum_"):
                continue
            var_rows.append(dict(dataset=ds, column=name, source=key, unit=unit, note=NOTES.get(name, "")))
        if ds == "control_energy":
            for out in ["afrr_pos_cost_eur", "afrr_neg_cost_eur", "afrr_total_cost_eur",
                        "mfrr_pos_cost_eur", "mfrr_neg_cost_eur", "mfrr_total_cost_eur"]:
                var_rows.append(dict(dataset=ds, column=out, source="I/J/K, R/S/T (differenced)",
                                     unit="EUR per quarter-hour", note=NOTES.get(out, "")))
    atomic_csv(pd.DataFrame(var_rows), out_dir / "variables.csv")
    print(f"done: {n_err} error(s). Manifest: {manifest_path}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
