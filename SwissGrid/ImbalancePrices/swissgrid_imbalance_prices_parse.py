#!/usr/bin/env python3
"""
Swissgrid imbalance-price parser (swissgrid-prices-for-{balance,imbalance}-energy_YYMMDD-YYMMDD.xml/.xlsx).

Run from the thesis root:
    python Swissgrid/ImbalancePrices/swissgrid_imbalance_prices_parse.py
    python Swissgrid/ImbalancePrices/swissgrid_imbalance_prices_parse.py --years 2026 --force

Needs: pandas, pyarrow, openpyxl

SAFETY: input files are opened read-only. The script writes ONLY under the
output directory (default Swissgrid/ImbalancePrices/Data/) and never deletes
anything. Existing year files are skipped unless --force is given; with
--force they are replaced atomically (write to temp file, then rename).

Source (verified on all 47 monthly files 2023-01 .. 2026-08):
  One XML + one XLSX per delivery month. File name changes from
  "...balance-energy..." to "...imbalance-energy..." in 2026; the separator
  before the period is "_" or "-". Files may sit in sub-folders.
  XML: <template><printedAt/><timeSeriesGroup>...<timeSeries>
         <Report_TS_Titel_en>BG-long</...><unit>ct/kWh</unit>
         <timeSeriesData><DATAVALUE><TIME>2023-01-01T00:00:00+01:00</TIME><VALUE>-2.48</VALUE>...
  - TIME = interval START with an explicit UTC offset -> no DST guessing.
    (The official XSD uses "Report-TS-Titel" with hyphens; the real files use
    underscores, so the XSD is not usable for validation. Both are accepted.)
  - Series: BG-long / BG-short (2023-01 .. 2025-12), BG-AEP single imbalance
    price (2025-07 .. today; the only series from 2026-01).
  - Unit ct/kWh -> EUR/MWh = x10.
  XLSX (sheet DATA) carries the same values with LOCAL wall-clock labels
  (ambiguous on the October DST day). It is used only as a cross-check, by
  row position.

Checks per month (a failing month is reported and skipped, others continue):
  - exact continuous 15-min grid from month start to next month start
    (92/100 rows on DST days), no gaps / duplicates
  - every series has the same timestamps, unit == ct/kWh, known series name
  - XLSX values equal the XML values (rounded to 0.01 ct/kWh)
  - if two files cover the same month, the one with the later printedAt wins

Output (timestamp = interval START, datetime64[us, Europe/Zurich]):
  Data/qh/<year>.parquet      timestamp, long_eur_mwh, short_eur_mwh, aep_eur_mwh
                              (NaN where a series was not published)
                              Load all years: pd.read_parquet("Swissgrid/ImbalancePrices/Data/qh/")
  Data/_parse_manifest.csv    one row per source month
  Data/_entsoe_check.csv      per month: agreement with ENTSO-E 17.1.G (CH)
  Data/combined_ch.parquet    ENTSO-E for years before the first Swissgrid
                              year (2021-2022) + Swissgrid from 2023, column
                              `source` = entsoe | swissgrid. Rebuilt every run.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"
FILE_RE = re.compile(r"swissgrid-prices-for-(?:im)?balance-energy[-_](\d{6})-(\d{6})\.(xml|xlsx)$", re.I)
SERIES = {"bg-long": "long", "bg-short": "short", "bg-aep": "aep"}
COLS = [f"{s}_eur_mwh" for s in SERIES.values()]
UNIT = "ct/kwh"
CT_KWH_TO_EUR_MWH = 10.0
XLSX_TOL_CT = 0.005 + 1e-9        # XLSX shows 2 decimals
ENTSOE_TOL_EUR = 0.05 + 1e-9      # ENTSO-E publishes 0.1 EUR/MWh


# ---------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(el, suffixes) -> str | None:
    for c in el:
        if any(_local(c.tag).replace("-", "_").endswith(s) for s in suffixes):
            return (c.text or "").strip()
    return None


def parse_xml(path: Path):
    """Return (DataFrame[timestamp + value cols], printed_at, series list)."""
    root = ET.parse(path).getroot()
    printed = _child_text(root, ("printedAt",))
    frames, names = [], []
    for ts in root.iter():
        if _local(ts.tag) != "timeSeries":
            continue
        name = _child_text(ts, ("TS_Titel_en",))
        unit = _child_text(ts, ("unit",))
        if name is None or name.lower() not in SERIES:
            raise ValueError(f"unknown series name {name!r}")
        if (unit or "").lower().replace(" ", "") != UNIT:
            raise ValueError(f"series {name}: unexpected unit {unit!r}")
        times, vals = [], []
        for dv in ts.iter():
            if _local(dv.tag) != "DATAVALUE":
                continue
            times.append(_child_text(dv, ("TIME",)))
            vals.append(_child_text(dv, ("VALUE",)))
        t = pd.to_datetime(times, utc=True, format="ISO8601")
        v = pd.to_numeric(pd.Series(vals).replace("", np.nan), errors="raise").to_numpy(float)
        s = pd.Series(v, index=t, name=SERIES[name.lower()])
        if s.index.has_duplicates:
            raise ValueError(f"series {name}: {s.index.duplicated().sum()} duplicate timestamps")
        frames.append(s)
        names.append(name)
    if not frames:
        raise ValueError("no timeSeries found")
    idx0 = frames[0].index
    for s in frames[1:]:
        if not s.index.sort_values().equals(idx0.sort_values()):
            raise ValueError(f"series {s.name} has different timestamps than {frames[0].name}")
    df = pd.concat(frames, axis=1).sort_index()
    return df, printed, names


def expected_grid(start_yymmdd: str, end_yymmdd: str) -> pd.DatetimeIndex:
    a = pd.Timestamp(f"20{start_yymmdd}").tz_localize(TZ)
    b = pd.Timestamp(f"20{end_yymmdd}").tz_localize(TZ)
    return pd.date_range(a.tz_convert("UTC"), b.tz_convert("UTC"), freq="15min", inclusive="left")


def xlsx_check(path: Path, xml_df: pd.DataFrame) -> tuple[str, float | None]:
    """Compare XLSX values with XML by row position. Returns (status, max_abs_diff_ct)."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    hdr = None
    for i in range(min(len(raw), 30)):
        if any(isinstance(x, str) and x.strip().lower() in SERIES for x in raw.iloc[i]):
            hdr = i
            break
    if hdr is None:
        return "no_header", None
    names = {j: SERIES[str(x).strip().lower()] for j, x in raw.iloc[hdr].items()
             if isinstance(x, str) and x.strip().lower() in SERIES}
    body = raw.iloc[hdr + 1:]
    ts = pd.to_datetime(body.iloc[:, 0].astype(str), format="%d.%m.%Y %H:%M:%S", errors="coerce")
    body = body[ts.notna().to_numpy()]  # drops the disclaimer footer
    if len(body) != len(xml_df):
        return f"row_count {len(body)} vs xml {len(xml_df)}", None
    if set(names.values()) != set(xml_df.columns):
        return f"series {sorted(names.values())} vs xml {sorted(xml_df.columns)}", None
    worst = 0.0
    for j, s in names.items():
        a = pd.to_numeric(body.iloc[:, j], errors="coerce").to_numpy(float)
        b = xml_df[s].to_numpy(float)
        if (np.isnan(a) != np.isnan(b)).any():
            return f"{s}: NaN pattern differs", None
        d = np.nanmax(np.abs(a - np.round(b, 2))) if np.isfinite(a).any() else 0.0
        worst = max(worst, float(d))
    return ("ok" if worst <= XLSX_TOL_CT else "value_mismatch"), worst


def atomic_parquet(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)  # only our own temp file


def discover(in_dir: Path):
    """{(start, end): {'xml': [paths], 'xlsx': [paths]}}"""
    out: dict = {}
    for p in sorted(in_dir.rglob("*")):
        m = FILE_RE.search(p.name)
        if m and p.is_file():
            out.setdefault((m.group(1), m.group(2)), {"xml": [], "xlsx": []})[m.group(3).lower()].append(p)
    return dict(sorted(out.items()))


def to_zurich(df_utc: pd.DataFrame) -> pd.DataFrame:
    out = df_utc.reindex(columns=[c.replace("_eur_mwh", "") for c in COLS]) * CT_KWH_TO_EUR_MWH
    out = out.round(6)
    out.columns = COLS
    out.index = out.index.tz_convert(TZ).astype(f"datetime64[us, {TZ}]")
    out.index.name = "timestamp"
    return out.reset_index()


# ---------------------------------------------------------------------------
def entsoe_check(sg: pd.DataFrame, entsoe_dir: Path) -> pd.DataFrame:
    rows = []
    years = sorted(sg["timestamp"].dt.year.unique())
    parts = [pd.read_parquet(entsoe_dir / f"{y}.parquet") for y in years if (entsoe_dir / f"{y}.parquet").exists()]
    if not parts:
        return pd.DataFrame()
    e = pd.concat(parts)
    e.index = pd.DatetimeIndex(e.index).tz_convert(TZ)
    s = sg.set_index("timestamp")
    s.index = pd.DatetimeIndex(s.index).tz_convert(TZ)
    j = s.join(e[["Long", "Short"]], how="left")
    for per, g in j.groupby(j.index.tz_convert(TZ).strftime("%Y-%m")):
        for sg_col, e_col in (("long_eur_mwh", "Long"), ("short_eur_mwh", "Short"),
                              ("aep_eur_mwh", "Long"), ("aep_eur_mwh", "Short")):
            if g[sg_col].isna().all():
                continue
            # BG-AEP is compared only where ENTSO-E carries it (months without long/short)
            if sg_col == "aep_eur_mwh" and g["long_eur_mwh"].notna().any():
                continue
            both = g[[sg_col, e_col]].dropna()
            d = (both[sg_col] - both[e_col]).abs()
            rows.append({
                "month": per, "swissgrid": sg_col, "entsoe": e_col, "n_swissgrid": int(g[sg_col].notna().sum()),
                "n_overlap": len(both), "n_missing_in_entsoe": int(g[sg_col].notna().sum() - len(both)),
                "share_equal": round(float((d <= ENTSOE_TOL_EUR).mean()), 4) if len(both) else np.nan,
                "max_abs_diff": round(float(d.max()), 3) if len(both) else np.nan,
                "days_with_diff": ";".join(sorted({f"{t:%d}" for t in d.index[d > ENTSOE_TOL_EUR]})),
            })
    return pd.DataFrame(rows)


def build_combined(sg: pd.DataFrame, entsoe_dir: Path, target: Path) -> str:
    first = int(sg["timestamp"].dt.year.min())
    parts = []
    for p in sorted(entsoe_dir.glob("*.parquet")):
        if p.stem.isdigit() and int(p.stem) < first:
            e = pd.read_parquet(p)
            e.index = pd.DatetimeIndex(e.index).tz_convert(TZ).astype(f"datetime64[us, {TZ}]")
            e = e.rename(columns={"Long": "long_eur_mwh", "Short": "short_eur_mwh"})
            e["aep_eur_mwh"] = np.nan
            e.index.name = "timestamp"
            parts.append(e.reset_index()[["timestamp"] + COLS].assign(source="entsoe"))
    parts.append(sg.assign(source="swissgrid"))
    c = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    if c["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps in combined series")
    atomic_parquet(c, target)
    step = c["timestamp"].diff()
    gaps = c.loc[step > pd.Timedelta("15min"), "timestamp"]
    for t in gaps:
        print(f"    combined: gap before {t} ({step[gaps.index[gaps == t][0]]})")
    by = c.groupby("source")["timestamp"].agg(["min", "max", "size"])
    return "; ".join(f"{k}: {r['min']:%Y-%m-%d} .. {r['max']:%Y-%m-%d} ({r['size']} rows)" for k, r in by.iterrows())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="thesis root (default: cwd)")
    ap.add_argument("--input-dir", default="Swissgrid/Manual_Download/Prices_for_imbalance_energy")
    ap.add_argument("--output-dir", default="Swissgrid/ImbalancePrices/Data")
    ap.add_argument("--entsoe-dir", default="Entsoe/Balancing/Data/production/imbalance_prices/all/CH",
                    help="ENTSO-E 17.1.G CH yearly parquet (cross-check + pre-2023 patch)")
    ap.add_argument("--years", help="comma list / ranges, e.g. 2023,2025-2026")
    ap.add_argument("--force", action="store_true", help="re-parse years that already have output")
    ap.add_argument("--no-xlsx-check", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    rp = lambda p: (Path(p) if Path(p).is_absolute() else root / p)
    in_dir, out_dir, e_dir = rp(a.input_dir), rp(a.output_dir), rp(a.entsoe_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"input dir not found: {in_dir}")
    years = None
    if a.years:
        years = set()
        for part in a.years.split(","):
            lo, _, hi = part.partition("-")
            years.update(range(int(lo), int(hi or lo) + 1))

    periods = discover(in_dir)
    print(f"Input : {in_dir}\nOutput: {out_dir}\nMonths: {len(periods)}")
    man_path = out_dir / "_parse_manifest.csv"
    manifest = pd.read_csv(man_path) if man_path.exists() else pd.DataFrame()

    by_year: dict[int, list] = {}
    for (s, e) in periods:
        by_year.setdefault(2000 + int(s[:2]), []).append((s, e))

    new_rows, n_err = [], 0
    for y, plist in sorted(by_year.items()):
        if years and y not in years:
            continue
        target = out_dir / "qh" / f"{y}.parquet"
        if target.exists() and not a.force:
            print(f"  {y} exists, skip (use --force)")
            continue
        month_frames = []
        for (s, e) in plist:
            files = periods[(s, e)]
            rec = {"year": y, "period": f"{s}-{e}", "parsed_at": pd.Timestamp.now().isoformat(timespec="seconds")}
            try:
                if not files["xml"]:
                    raise ValueError("no XML file for this month (XLSX-only parsing not supported: DST-ambiguous)")
                parsed = []
                for p in files["xml"]:
                    df, printed, names = parse_xml(p)
                    parsed.append((pd.Timestamp(printed) if printed else pd.Timestamp.min.tz_localize("UTC"), p, df, names))
                parsed.sort(key=lambda t: t[0])
                printed, xml_path, df, names = parsed[-1]
                grid = expected_grid(s, e)
                if not df.index.equals(grid):
                    miss, extra = grid.difference(df.index), df.index.difference(grid)
                    raise ValueError(f"grid mismatch: {len(df)} rows vs {len(grid)} expected, "
                                     f"{len(miss)} missing (first {miss[:1].tolist()}), {len(extra)} extra")
                xstat, xdiff = ("skipped", None)
                if not a.no_xlsx_check:
                    stem = xml_path.with_suffix(".xlsx")
                    xl = stem if stem.exists() else (files["xlsx"][0] if files["xlsx"] else None)
                    xstat, xdiff = xlsx_check(xl, df) if xl else ("missing", None)
                month_frames.append(to_zurich(df))
                rec.update(
                    status="ok", file=str(xml_path.relative_to(root)) if xml_path.is_relative_to(root) else str(xml_path),
                    printed_at=str(printed), series=";".join(names), rows=len(df), expected_rows=len(grid),
                    first_start=str(df.index[0].tz_convert(TZ)), last_start=str(df.index[-1].tz_convert(TZ)),
                    n_nan=int(df.isna().sum().sum()), n_xml_copies=len(files["xml"]),
                    xlsx_check=xstat, xlsx_max_abs_diff_ct=xdiff,
                    min_eur_mwh=float(np.nanmin(df.to_numpy()) * 10), max_eur_mwh=float(np.nanmax(df.to_numpy()) * 10),
                )
                print(f"  {s}-{e} ok  {len(df):5d} rows  {','.join(names):24s} xlsx={xstat}")
            except Exception as ex:  # one bad month must not stop the run
                n_err += 1
                rec.update(status="error", error=f"{type(ex).__name__}: {ex}")
                print(f"  {s}-{e} ERROR {rec['error']}", file=sys.stderr)
            new_rows.append(rec)
        if month_frames:
            ydf = pd.concat(month_frames, ignore_index=True).sort_values("timestamp")
            if ydf["timestamp"].duplicated().any():
                raise SystemExit(f"{y}: overlapping months produce duplicate timestamps")
            atomic_parquet(ydf, target)
            print(f"  {y} -> {target.relative_to(root) if target.is_relative_to(root) else target} ({len(ydf)} rows)")

    if new_rows:
        new = pd.DataFrame(new_rows)
        if not manifest.empty:
            manifest = manifest[~manifest["period"].isin(new["period"])]
        manifest = pd.concat([manifest, new], ignore_index=True).sort_values("period")
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(man_path, index=False)

    # cross-check + combined series over everything that is on disk
    qh_dir = out_dir / "qh"
    if qh_dir.is_dir() and any(qh_dir.glob("*.parquet")):
        sg = pd.read_parquet(qh_dir)
        sg["timestamp"] = sg["timestamp"].dt.tz_convert(TZ)
        sg = sg.sort_values("timestamp").reset_index(drop=True)
        if e_dir.is_dir():
            chk = entsoe_check(sg, e_dir)
            chk.to_csv(out_dir / "_entsoe_check.csv", index=False)
            if not chk.empty:
                main_pairs = chk[(chk.swissgrid.str[:4] == chk.entsoe.str.lower().str[:4]) |
                                 (chk.swissgrid.str.startswith("aep") & (chk.entsoe == "Long"))]
                bad = main_pairs[(main_pairs.share_equal < 0.99) | main_pairs.share_equal.isna()]
                print(f"ENTSO-E check: {len(main_pairs)} month-series compared, "
                      f"{len(bad)} below 99% agreement or absent in ENTSO-E -> {out_dir / '_entsoe_check.csv'}")
                for r in bad.itertuples():
                    print(f"    {r.month} {r.swissgrid:14s} share_equal={r.share_equal}  "
                          f"missing_in_entsoe={r.n_missing_in_entsoe}  days={r.days_with_diff}")
            print("Combined:", build_combined(sg, e_dir, out_dir / "combined_ch.parquet"))
        else:
            print(f"ENTSO-E dir not found ({e_dir}) - cross-check and combined series skipped")
    print(f"Done: {len(new_rows)} month(s) processed, {n_err} error(s). Manifest: {man_path}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
