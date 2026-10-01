"""
frequency_tso_probe.py — probe for the TSO 1-second grid-frequency archive
(Continental Europe, measured by TransnetBW; archive now hosted by the German
TSOs' joint platform netztransparenz.de / SG HoBA).

Why: Energy-Charts /frequency (already pulled, EnergyCharts/Data/frequency/) only
starts on 1 May 2022. The TSO archive covers Feb 2012 -> Jun 2022 as yearly zip
files, so it fills 2021-01 -> 2022-04 (and 2015-2020 if the sample is extended)
and overlaps Energy-Charts in May-Jun 2022 for a cross-check.

This module also owns the download and parsing logic used by
frequency_tso_pull.py (same pattern as Regelleistung/).

What the probe does
  1. Downloads ONE yearly zip (default 2021) to Frequency/Data/raw/ (kept, reused
     by the pull) — or uses a zip you placed there by hand.
  2. Lists the zip content (nested zips are opened too) and prints the first raw
     lines of the first inner file.
  3. Sniffs the format (delimiter, decimal mark, date format, Hz vs mHz) and parses
     a sample file: rows, time step, duplicates, gaps, value range.
  4. Checks the DST days (last Sunday of March / October) to decide whether the
     timestamps are local time (CET/CEST) or fixed CET/UTC.
  5. Optional --zenodo: lists the files and licence of the Jülich/QMUL
     "Pre-Processed Power Grid Frequency Time Series (2020-2023)" record
     (DOI 10.5281/zenodo.15784548) — reference only, not downloaded.

Run from the thesis root (venv active):
    python Frequency/frequency_tso_probe.py                 # probes 2021
    python Frequency/frequency_tso_probe.py --year 2022     # archive end (Jun 2022?)
    python Frequency/frequency_tso_probe.py --zenodo

Writes only below Frequency/Data/ (never deletes):
    raw/<YYYY>_Frequenz.zip           downloaded archive (reused by the pull)
    _probe/probe_report_<YYYY>.txt    human-readable report (also printed)
"""

from __future__ import annotations

import argparse
import io
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path("Frequency") / "Data"
RAW = DATA / "raw"
PROBE_DIR = DATA / "_probe"
SOURCE_TZ = "Europe/Berlin"      # same rules as Europe/Zurich
LOCAL_TZ = "Europe/Zurich"
TIMEOUT_S = 120
MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (Macintosh) thesis-data-pull (research use)"

# Archive URL candidates, tried in order (the site has used both spellings).
URL_TEMPLATES = [
    "https://www.netztransparenz.de/xspproxy/api/staticfiles/ntp-relaunch/dokumente/"
    "regelenergie/sek%C3%BCndliche%20frequenz/{year}_frequenz.zip",
    "https://www.netztransparenz.de/xspproxy/api/staticfiles/ntp-relaunch/dokumente/"
    "regelenergie/sek%C3%BCndliche%20frequenz/{year}_Frequenz.zip",
]
PAGE_URL = "https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Sek%C3%BCndliche-Daten"
ZENODO_API = "https://zenodo.org/api/records/15784548"

FILL_LIMIT_S = 3        # archive stores a value every 1-4 s -> forward-fill gaps <= 3 s


def log(msg: str = "", buf: list[str] | None = None) -> None:
    print(msg, flush=True)
    if buf is not None:
        buf.append(msg)


# --------------------------------------------------------------------- download

def zip_path(year: int) -> Path:
    return RAW / f"{year}_Frequenz.zip"


def find_local_zip(year: int) -> Path | None:
    """A zip for this year already in raw/ (downloaded or placed by hand, any case)."""
    if not RAW.exists():
        return None
    for p in sorted(RAW.iterdir()):
        if p.suffix.lower() == ".zip" and p.name.lower().startswith(f"{year}_frequenz"):
            return p
    return None


def download_year(year: int, force: bool = False) -> tuple[Path | None, str]:
    """Download the yearly zip (streamed, atomic). Returns (path, message)."""
    existing = find_local_zip(year)
    if existing and not force:
        if zipfile.is_zipfile(existing):
            return existing, f"using existing {existing}"
        return None, f"{existing} exists but is not a valid zip — delete or replace it by hand"
    RAW.mkdir(parents=True, exist_ok=True)
    target = zip_path(year)
    tmp = target.with_suffix(".zip.part")
    errors = []
    for tmpl in URL_TEMPLATES:
        url = tmpl.format(year=year)
        for attempt in range(MAX_RETRIES + 1):
            try:
                with requests.get(url, stream=True, timeout=TIMEOUT_S,
                                  headers={"User-Agent": USER_AGENT}) as r:
                    if r.status_code in (429,) or r.status_code >= 500:
                        raise requests.HTTPError(f"HTTP {r.status_code}")
                    if r.status_code != 200:
                        errors.append(f"{url} -> HTTP {r.status_code}")
                        break
                    n = 0
                    with tmp.open("wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            n += len(chunk)
                if not zipfile.is_zipfile(tmp):
                    errors.append(f"{url} -> 200 but not a zip ({n:,} bytes; HTML page?)")
                    tmp.unlink(missing_ok=True)
                    break
                tmp.replace(target)
                return target, f"downloaded {url} ({n / 1e6:.1f} MB)"
            except (requests.RequestException, OSError) as e:
                if attempt < MAX_RETRIES:
                    time.sleep(20 * (attempt + 1))
                    continue
                errors.append(f"{url} -> {type(e).__name__}: {e}")
    tmp.unlink(missing_ok=True)
    return None, ("download failed: " + " | ".join(errors) +
                  f"\n  -> download {year}_Frequenz.zip by hand from {PAGE_URL}"
                  f"\n     and save it as {target}; then re-run.")


# ---------------------------------------------------------------- zip handling

def iter_members(zf: zipfile.ZipFile, prefix: str = ""):
    """Yield (name, opener) for every data file, opening nested zips.
    opener() returns a fresh binary file object (streamed, not read into memory)."""
    for info in sorted(zf.infolist(), key=lambda i: i.filename):
        if info.is_dir() or info.filename.startswith("__MACOSX") or info.filename.endswith(".DS_Store"):
            continue
        name = prefix + info.filename
        if info.filename.lower().endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(zf.read(info)))
            yield from iter_members(inner, prefix=name + "!")
        else:
            yield name, (lambda zf=zf, info=info: zf.open(info))


def list_members(zf: zipfile.ZipFile, prefix: str = "") -> list[tuple[str, int]]:
    out = []
    for info in sorted(zf.infolist(), key=lambda i: i.filename):
        if info.is_dir():
            continue
        out.append((prefix + info.filename, info.file_size))
        if info.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(zf.read(info))) as inner:
                out.extend(list_members(inner, prefix + info.filename + "!"))
    return out


# --------------------------------------------------------------------- parsing

@dataclass
class Fmt:
    encoding: str
    sep: str
    decimal: str
    skiprows: int
    split_datetime: bool      # date and time in separate columns
    date_format: str | None   # strftime format of the combined "date time" string
    n_cols: int


DATE_PATTERNS = [
    (re.compile(r"^\d{2}\.\d{2}\.\d{4}$"), "%d.%m.%Y"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "%Y/%m/%d"),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%d/%m/%Y"),
]


def decode(data: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace"), "latin-1"


def _date_fmt(token: str) -> str | None:
    for rx, f in DATE_PATTERNS:
        if rx.match(token):
            return f
    return None


def sniff(data: bytes) -> Fmt:
    """Detect the layout from the first lines of a file."""
    text, enc = decode(data[:20000])
    lines = text.splitlines()
    skip = 0
    while skip < len(lines) and not lines[skip].strip().lstrip('"').strip()[:1].isdigit():
        skip += 1
    if skip >= len(lines):
        raise ValueError("no data line starting with a digit in the first 20 kB")
    sample = [l for l in lines[skip:skip + 50] if l.strip()]
    first = sample[0].strip()
    if first.startswith('"'):
        first = first.replace('"', "")

    sep = max([";", "\t", ","], key=lambda s: first.count(s))
    if first.count(sep) == 0:
        sep = r"\s+"
        parts = first.split()
    else:
        parts = first.split(sep)
    parts = [p.strip().strip('"') for p in parts]

    # date + time together ("2021-01-01 00:00:00") or split ("01.01.2021", "00:00:00")
    if " " in parts[0] or "T" in parts[0]:
        d, t = re.split(r"[ T]", parts[0], maxsplit=1)
        split_dt, rest = False, parts[1:]
    else:
        d, t = parts[0], parts[1]
        split_dt, rest = True, parts[2:]
    dfmt = _date_fmt(d)
    tfmt = "%H:%M:%S.%f" if "." in t else "%H:%M:%S"
    date_format = f"{dfmt} {tfmt}" if dfmt else None

    # value column: decimal comma if sep != "," and value contains ","; or sep == ","
    # and the row has one field too many (e.g. "01.01.2021,00:00:00,50,012")
    val = sep.join(rest) if sep != r"\s+" else " ".join(rest)
    decimal = "."
    if sep != "," and "," in val:
        decimal = ","
    elif sep == "," and len(rest) == 2 and all(x.isdigit() for x in rest):
        decimal = ","     # value split across two fields -> rejoined in parse_member
    n_cols = len(parts)
    return Fmt(enc, sep, decimal, skip, split_dt, date_format, n_cols)


def head_bytes(opener, n: int = 20000) -> bytes:
    with opener() as fh:
        return fh.read(n)


CHUNK_ROWS = 2_000_000


def parse_member(opener, fmt: Fmt | None = None) -> tuple[pd.DataFrame, Fmt]:
    """Parse one inner file (streamed in chunks) -> DataFrame(ts_naive, f) in file order."""
    fmt = fmt or sniff(head_bytes(opener))
    parts = []
    with opener() as fh:
        reader = pd.read_csv(fh, sep=fmt.sep, header=None, skiprows=fmt.skiprows, dtype=str,
                             encoding=fmt.encoding, skipinitialspace=True,
                             engine="c" if fmt.sep != r"\s+" else "python",
                             skip_blank_lines=True, on_bad_lines="skip", chunksize=CHUNK_ROWS)
        for df in reader:
            parts.append(_parse_chunk(df, fmt))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ts_naive", "f"])
    return out, fmt


def _parse_chunk(df: pd.DataFrame, fmt: Fmt) -> pd.DataFrame:
    if fmt.split_datetime:
        ts_str = df.iloc[:, 0].str.strip() + " " + df.iloc[:, 1].str.strip()
        vals = df.iloc[:, 2:]
    else:
        ts_str = df.iloc[:, 0].str.strip().str.replace("T", " ", regex=False)
        vals = df.iloc[:, 1:]
    if fmt.sep == "," and fmt.decimal == "," and vals.shape[1] >= 2:
        v = vals.iloc[:, 0].str.strip() + "." + vals.iloc[:, 1].str.strip()
    else:
        v = vals.iloc[:, 0].str.strip()
        if fmt.decimal == ",":
            v = v.str.replace(",", ".", regex=False)
    ts = None
    if fmt.date_format:
        ts = pd.to_datetime(ts_str, format=fmt.date_format, errors="coerce")
        if ts.isna().mean() > 0.01:      # mixed formats inside the file
            ts = None
    if ts is None:
        ts = pd.to_datetime(ts_str, format="mixed", dayfirst=True, errors="coerce")
    f = pd.to_numeric(v, errors="coerce").astype("float64")
    out = pd.DataFrame({"ts_naive": ts.values, "f": f.values})
    return out.dropna(subset=["ts_naive"])


def unit_to_hz(f: pd.Series) -> tuple[pd.Series, str]:
    """Values in Hz (~50) or as deviation in mHz (~0) -> Hz."""
    med = float(np.nanmedian(f.values)) if len(f) else np.nan
    if 45 < med < 55:
        return f, "Hz"
    if abs(med) < 1000:
        return 50.0 + f / 1000.0, "deviation in mHz -> converted to Hz"
    raise ValueError(f"unexpected value level (median {med})")


def last_sunday(year: int, month: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() + 1) % 7)


def detect_tz_mode(ts_naive: pd.Series) -> tuple[str, str]:
    """'local' if the spring-forward hour is missing and/or the fall-back hour is
    repeated; 'fixed' if the spring hour has data and no fall-back hour repeats;
    'unknown' if no DST day is in the data."""
    t = pd.DatetimeIndex(ts_naive)
    years = sorted(set(t.year))
    notes, votes, spring_votes = [], [], []
    for y in years:
        mar, octo = last_sunday(y, 3), last_sunday(y, 10)
        s0 = pd.Timestamp(mar) + pd.Timedelta(hours=2)
        sp = t[(t >= s0 - pd.Timedelta(hours=1)) & (t < s0 + pd.Timedelta(hours=2))]
        if len(sp) > 3600:
            in_gap = ((sp >= s0) & (sp < s0 + pd.Timedelta(hours=1))).sum()
            spring_votes.append("local" if in_gap == 0 else "fixed")
            notes.append(f"{mar}: {in_gap} values in 02:00-02:59 (0 = local time)")
        f0 = pd.Timestamp(octo) + pd.Timedelta(hours=2)
        fb = t[(t >= f0) & (t < f0 + pd.Timedelta(hours=1))]
        if len(fb) > 600:
            dup = pd.Index(fb.floor("s")).duplicated().sum()
            votes.append("local" if dup > 1000 else "fixed")
            notes.append(f"{octo}: {dup} repeated seconds in 02:00-02:59 (>0 = local time)")
    # the spring gap is the clearer signal (a local-time archive may also store the
    # repeated October hour only once), so it decides when present
    votes = spring_votes or votes
    if not votes:
        return "unknown", "no DST day in this data"
    mode = max(sorted(set(votes)), key=votes.count)
    if len(set(votes)) > 1:
        notes.append("CONFLICTING signals — check and set --tz explicitly")
    return mode, "; ".join(notes)


def fallback_first_pass(ts: pd.DatetimeIndex) -> np.ndarray:
    """For the repeated hour 02:00-02:59 on fall-back days: True for rows of the first
    pass (CEST), False for the second pass (CET). The second pass starts where the
    clock jumps back in file order. Rows outside that hour: True (ignored by pandas)."""
    amb = np.ones(len(ts), dtype=bool)
    vals = ts.values
    for y in sorted(set(ts.year)):
        h0 = np.datetime64(pd.Timestamp(last_sunday(y, 10)) + pd.Timedelta(hours=2))
        h1 = h0 + np.timedelta64(1, "h")
        pos = np.flatnonzero((vals >= h0) & (vals < h1))
        if len(pos) < 2:
            continue
        back = np.flatnonzero(np.diff(vals[pos]) < -np.timedelta64(30, "m"))
        if len(back):
            amb[pos[back[0] + 1:]] = False
    return amb


def to_utc_1s(df: pd.DataFrame, tz_mode: str) -> tuple[pd.Series, dict]:
    """Naive timestamps (file order) -> regular 1-s UTC series of frequency in Hz.

    local : Europe/Berlin wall clock; in the repeated October hour the first pass is
            CEST and the second CET (file order); spring-forward seconds are dropped.
    fixed : CET all year (UTC+1).   utc : already UTC.
    Sub-second values are averaged per second; gaps <= FILL_LIMIT_S s are
    forward-filled (archive stores a value every 1-4 s)."""
    ts = pd.DatetimeIndex(df["ts_naive"])
    f = df["f"].astype("float64").values
    stats = {"rows_raw": len(ts)}
    if tz_mode == "local":
        amb = fallback_first_pass(ts)
        loc = ts.tz_localize(SOURCE_TZ, ambiguous=amb, nonexistent="NaT")
        utc = loc.tz_convert("UTC")
    elif tz_mode == "fixed":
        utc = (ts - pd.Timedelta(hours=1)).tz_localize("UTC")
    elif tz_mode == "utc":
        utc = ts.tz_localize("UTC")
    else:
        raise ValueError(tz_mode)
    s = pd.Series(f, index=utc)
    s = s[s.index.notna()]
    stats["rows_nonexistent_dropped"] = stats["rows_raw"] - len(s)
    s = s[(s > 45) & (s < 55)]
    stats["rows_out_of_range_dropped"] = stats["rows_raw"] - stats["rows_nonexistent_dropped"] - len(s)
    s = s.groupby(s.index.floor("s")).mean().sort_index()
    stats["seconds_with_value"] = len(s)
    if s.empty:
        stats["seconds_after_fill"] = 0
        return s.rename("frequency_hz"), stats
    grid = pd.date_range(s.index.min(), s.index.max(), freq="s", tz="UTC")
    r = s.reindex(grid)
    isna = r.isna()
    run_id = (~isna).cumsum()                         # NaN runs share the id of the value before
    run_len = isna.groupby(run_id).transform("sum")   # length of the NaN run each row is in
    fill_ok = isna & (run_len <= FILL_LIMIT_S)        # fill short gaps only, leave long gaps empty
    s = r.where(~fill_ok, r.ffill()).dropna()
    stats["seconds_after_fill"] = len(s)
    s.index.name = "timestamp_utc"
    return s.rename("frequency_hz"), stats


def parse_zip(path: Path, tz_mode: str = "auto", verbose=print) -> tuple[pd.Series, dict]:
    """Whole yearly zip -> 1-s UTC series + parse info."""
    frames, fmts, names = [], {}, []
    with zipfile.ZipFile(path) as zf:
        for name, opener in iter_members(zf):
            try:
                d, fmt = parse_member(opener)
            except Exception as e:     # noqa: BLE001 — report and continue
                verbose(f"    ! {name}: not parsed ({type(e).__name__}: {e})")
                continue
            frames.append(d)
            names.append(name)
            fmts[name] = fmt
            verbose(f"    {name}: {len(d):,} rows, sep={fmt.sep!r}, decimal={fmt.decimal!r}, "
                    f"date_format={fmt.date_format!r}")
    if not frames:
        raise ValueError("no parsable file in zip")
    df = pd.concat(frames, ignore_index=True)
    df["f"], unit = unit_to_hz(df["f"])
    detected, tz_note = detect_tz_mode(df["ts_naive"])
    mode = tz_mode if tz_mode != "auto" else (detected if detected != "unknown" else "local")
    s, stats = to_utc_1s(df, mode)
    info = dict(files=len(names), unit=unit, tz_detected=detected, tz_used=mode,
                tz_note=tz_note, **stats)
    return s, info


# ------------------------------------------------------------------------ probe

def probe(year: int, force: bool) -> None:
    buf: list[str] = []
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 78, buf)
    log(f"TSO frequency archive probe — year {year} — {datetime.now():%Y-%m-%d %H:%M}", buf)
    log("=" * 78, buf)
    path, msg = download_year(year, force=force)
    log(f"1. Download: {msg}", buf)
    if not path:
        _write(buf, year)
        return
    log(f"   size {path.stat().st_size / 1e6:.1f} MB", buf)

    with zipfile.ZipFile(path) as zf:
        members = list_members(zf)
        log(f"\n2. Content: {len(members)} files", buf)
        for n, sz in members[:30]:
            log(f"   {sz / 1e6:9.1f} MB  {n}", buf)
        if len(members) > 30:
            log(f"   ... ({len(members) - 30} more)", buf)
        first_name, first_open = next(iter_members(zf))
        first_head = head_bytes(first_open)
        fmt = sniff(first_head)
        d, _ = parse_member(first_open, fmt)
    text, enc = decode(first_head[:3000])
    log(f"\n   First lines of {first_name} (encoding {enc}):", buf)
    for line in text.splitlines()[:8]:
        log(f"   | {line}", buf)

    log(f"\n3. Format: {fmt}", buf)
    d["f"], unit = unit_to_hz(d["f"])
    steps = pd.Series(pd.DatetimeIndex(d["ts_naive"])).diff().dt.total_seconds().dropna()
    log(f"   sample file: {len(d):,} rows, {d['ts_naive'].min()} -> {d['ts_naive'].max()}, unit {unit}", buf)
    if len(steps):
        log("   time step (s): " + ", ".join(f"{k:g}: {v:.2%}" for k, v in
                                           steps.value_counts(normalize=True).head(5).items()), buf)
    log(f"   value range {d['f'].min():.4f} – {d['f'].max():.4f} Hz, NaN {d['f'].isna().mean():.3%}", buf)

    log("\n4. Full-year parse (all files, DST check, 1-s grid) ...", buf)
    t0 = time.time()
    s, info = parse_zip(path, verbose=lambda m: log(m, buf))
    log(f"   parsed in {time.time() - t0:.0f} s", buf)
    for k, v in info.items():
        log(f"   {k}: {v}", buf)
    if not s.empty:
        loc = s.index.tz_convert(LOCAL_TZ)
        log(f"   span (local): {loc.min()} -> {loc.max()}", buf)
        per_month = s.groupby(loc.strftime("%Y-%m")).size()
        log("   seconds per month (coverage vs. calendar):", buf)
        for m, n in per_month.items():
            a = pd.Timestamp(m + "-01").tz_localize(LOCAL_TZ)
            b = a + pd.offsets.MonthBegin(1)
            log(f"     {m}: {n:>10,}  ({n / (b - a).total_seconds():.2%})", buf)
    _write(buf, year)


def probe_zenodo() -> None:
    buf: list[str] = []
    log("\n5. Zenodo record 15784548 (reference only)", buf)
    try:
        r = requests.get(ZENODO_API, timeout=60, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        rec = r.json()
        meta = rec.get("metadata", {})
        log(f"   title:   {meta.get('title')}", buf)
        log(f"   version: {meta.get('version')}  published {meta.get('publication_date')}", buf)
        lic = meta.get("license") or meta.get("rights")
        log(f"   licence: {lic}", buf)
        for f in rec.get("files", []):
            log(f"   {f.get('size', 0) / 1e9:6.2f} GB  {f.get('key')}", buf)
    except Exception as e:   # noqa: BLE001
        log(f"   failed: {type(e).__name__}: {e}", buf)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    (PROBE_DIR / "probe_zenodo.txt").write_text("\n".join(buf) + "\n", encoding="utf-8")


def _write(buf: list[str], year: int) -> None:
    p = PROBE_DIR / f"probe_report_{year}.txt"
    p.write_text("\n".join(buf) + "\n", encoding="utf-8")
    print(f"\nReport written to {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--force", action="store_true", help="re-download even if the zip exists")
    ap.add_argument("--zenodo", action="store_true", help="only list the Zenodo record")
    args = ap.parse_args()
    if args.zenodo:
        probe_zenodo()
        return
    probe(args.year, args.force)


if __name__ == "__main__":
    main()
