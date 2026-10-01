"""
frequency_zenodo.py — gap filler from the Jülich/QMUL "Pre-Processed Power Grid
Frequency Time Series (2020-2023)", Zenodo DOI 10.5281/zenodo.15784548
(licence CC BY 4.0 + MIT; Continental Europe data from TransnetBW until Jun 2022 and
SG HoBA / netztransparenz from Jun 2022).

Why: two gap types remain in the merged series —
  * March 2022: no 202203 file in the TSO archive zip;
  * Energy-Charts gaps Jul 2022 -> 2023 (Sep/Oct/Nov 2022, Jun 2023, ...).
Zenodo covers both periods from the TSO sources. It is used ONLY where the TSO archive
and Energy-Charts have no (or < 90 %) data — see frequency_merge.py --sources.

Steps
  --list    download Data_cleansed.zip (~1 GB, resumable), DESCRIPTION.md, LICENSE.md,
            scripts.zip; list the archive, show first lines of each Continental Europe
            file and the download URLs used in the authors' scripts. No parsing.
  (default) parse the Continental Europe year files -> 1-s UTC -> per local month
              production/zenodo/raw_1s/<YYYY>/freq_zenodo_<YYYY-MM>.parquet
              production/zenodo/agg/freq_zenodo_agg_<YYYY-MM>.parquet
            then combine -> production/zenodo/frequency_zenodo_{15min,4h}.parquet
            and VERIFY the clock against the TSO archive (and Energy-Charts): MAE at
            shifts of -1 h / 0 / +1 h and -5..+5 s. A best shift of +-1 h means the
            timezone assumption is wrong -> re-run with --tz.

Timestamps: the record says naive local time, CET for Continental Europe. --tz auto
detects it on the DST days (spring 02:00-02:59 present -> fixed CET).

Run from the thesis root (venv active):
    python Frequency/frequency_zenodo.py --list
    python Frequency/frequency_zenodo.py                    # all CE years found (2020-2023)
    python Frequency/frequency_zenodo.py --years 2022 2023
    python Frequency/frequency_zenodo.py --verify-only

Writes only below Frequency/Data/ (atomic writes, never deletes):
    raw/zenodo/..., production/zenodo/..., _probe/zenodo_list.txt, compare/zenodo_verify.txt
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EnergyCharts"))
import frequency_tso_probe as P                                    # noqa: E402
from energycharts_pull import frequency_features                   # noqa: E402

RECORD = "https://zenodo.org/records/15784548/files/{name}?download=1"
FILES = ["Data_cleansed.zip", "DESCRIPTION.md", "LICENSE.md", "scripts.zip"]
RAW = P.DATA / "raw" / "zenodo"
OUT = P.DATA / "production" / "zenodo"
TSO_RAW = P.DATA / "production" / "raw_1s"
EC_RAW = Path("EnergyCharts") / "Data" / "frequency" / "raw"
LOCAL_TZ = P.LOCAL_TZ
NORDIC_RX = re.compile(r"nordic|fingrid|finland|helsinki|(^|[^a-z])fi([^a-z]|$)", re.I)
SKIP_EXT = (".md", ".py", ".json", ".ipynb", ".pdf", ".png", ".txt.md", ".yml", ".yaml")
YEAR_RX = re.compile(r"(20[12]\d)")


def log(msg: str = "", buf: list[str] | None = None) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    if buf is not None:
        buf.append(msg)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


# --------------------------------------------------------------------- download

def download(name: str) -> Path:
    """Streamed, resumable (HTTP Range) download into raw/zenodo/."""
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / name
    if target.exists():
        if not name.endswith(".zip") or zipfile.is_zipfile(target):
            return target
    part = target.with_suffix(target.suffix + ".part")
    url = RECORD.format(name=name)
    for attempt in range(6):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": P.USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                if r.status_code == 416:          # already complete
                    break
                if r.status_code not in (200, 206):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                if r.status_code == 200 and have:
                    have = 0                      # server ignored Range -> restart
                total = have + int(r.headers.get("Content-Length", 0))
                mode = "ab" if have else "wb"
                t0, n = time.time(), have
                with part.open(mode) as f:
                    for chunk in r.iter_content(4 << 20):
                        f.write(chunk)
                        n += len(chunk)
                        if total > 50e6 and time.time() - t0 > 10:
                            print(f"    {name}: {n / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)
                            t0 = time.time()
            break
        except (requests.RequestException, OSError) as e:
            log(f"  {name}: {type(e).__name__}: {e} — retry {attempt + 1}/5 (resumes)")
            time.sleep(15 * (attempt + 1))
    else:
        raise RuntimeError(f"download of {name} failed; re-run to resume ({part})")
    part.replace(target)
    if name.endswith(".zip") and not zipfile.is_zipfile(target):
        raise RuntimeError(f"{target} is not a valid zip")
    log(f"  downloaded {name} ({target.stat().st_size / 1e6:,.1f} MB)")
    return target


def ce_members(zf: zipfile.ZipFile) -> list[tuple[str, int, object]]:
    """(name, year, opener) for Continental Europe data files."""
    out = []
    for name, opener in P.iter_members(zf):
        fname = name.split("!")[-1].split("/")[-1].lower()
        if fname.endswith(SKIP_EXT):
            continue
        if NORDIC_RX.search(name):
            continue
        ym = YEAR_RX.findall(name)
        if not ym:
            continue
        out.append((name, int(ym[-1]), opener))
    return out


# ------------------------------------------------------------------------- list

def do_list() -> None:
    buf: list[str] = []
    for f in FILES:
        download(f)
    log("=" * 70, buf)
    log(f"Zenodo 15784548 — content — {datetime.now():%Y-%m-%d %H:%M}", buf)
    log("=" * 70, buf)
    for f in ("DESCRIPTION.md", "LICENSE.md"):
        log(f"\n--- {f} ---", buf)
        for line in (RAW / f).read_text(encoding="utf-8", errors="replace").splitlines()[:80]:
            log(f"  {line}", buf)
    with zipfile.ZipFile(RAW / "Data_cleansed.zip") as zf:
        log("\n--- Data_cleansed.zip ---", buf)
        for n, sz in P.list_members(zf):
            log(f"  {sz / 1e6:9.1f} MB  {n}", buf)
        log("\n--- Continental Europe files (selected) ---", buf)
        for name, year, opener in ce_members(zf):
            head = P.head_bytes(opener, 600)
            text, _ = P.decode(head)
            log(f"  {year}  {name}", buf)
            for line in text.splitlines()[:4]:
                log(f"      | {line}", buf)
    with zipfile.ZipFile(RAW / "scripts.zip") as zf:
        log("\n--- URLs in scripts.zip ---", buf)
        urls = set()
        for name, opener in P.iter_members(zf):
            with opener() as fh:
                txt = fh.read().decode("utf-8", errors="replace")
            for u in re.findall(r"https?://[^\s'\"\)>]+", txt):
                urls.add((name, u))
        for name, u in sorted(urls):
            log(f"  {name}: {u}", buf)
    p = P.PROBE_DIR / "zenodo_list.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(buf) + "\n", encoding="utf-8")
    print(f"\nWritten: {p}")


# ------------------------------------------------------------------------ build

def build(years: list[int] | None, tz_mode: str) -> None:
    zpath = download("Data_cleansed.zip")
    with zipfile.ZipFile(zpath) as zf:
        members = ce_members(zf)
        if years:
            members = [m for m in members if m[1] in years]
        if not members:
            log("No Continental Europe files selected — run --list and check the names.")
            return
        by_year: dict[int, list] = {}
        for name, year, opener in members:
            by_year.setdefault(year, []).append((name, opener))
        for year in sorted(by_year):
            t0 = time.time()
            # main file first (TransnetBW/<year>.zip), then the others only for timestamps
            # it does not have (2022 has three overlapping files with misleading names)
            files = sorted(by_year[year], key=lambda x: _file_rank(x[0], year))
            log(f"BUILD {year}: {', '.join(n for n, _ in files)}")
            # each file gets its own timestamp convention: the Zenodo files are not all
            # in the same one (verify showed whole days 1 h off), so it is calibrated per
            # file against the TSO archive / Energy-Charts unless --tz is given
            series = []
            for name, opener in files:
                d, fmt = parse_allow_nan(opener)
                d = d.dropna(subset=["f"])
                if d.empty:
                    log(f"  {name}: no valid values")
                    continue
                d["f"], unit = P.unit_to_hz(d["f"])
                if tz_mode != "auto":
                    mode, why = tz_mode, "forced with --tz"
                else:
                    mode, why = calibrate_tz(d)
                f_s, stats = P.to_utc_1s(d, mode)
                log(f"  {name}: {len(d):,} valid rows ({d['ts_naive'].min()} -> {d['ts_naive'].max()}), "
                    f"unit {unit}, tz {mode} [{why}], {stats['seconds_after_fill']:,} s")
                series.append(f_s)
            if not series:
                continue
            s = pd.concat(series)
            n_before = len(s)
            s = s[~s.index.duplicated(keep="first")].sort_index()     # main file wins
            log(f"  combined: {len(s):,} s ({n_before - len(s):,} duplicate seconds from later files dropped)")
            loc = s.index.tz_convert(LOCAL_TZ)
            months = loc.strftime("%Y-%m")
            for m in sorted(set(months)):
                if not m.startswith(str(year)):
                    continue
                part = s[months == m]
                a = pd.Timestamp(m + "-01").tz_localize(LOCAL_TZ)
                cov = len(part) / ((a + pd.offsets.MonthBegin(1)) - a).total_seconds()
                atomic_parquet(part.astype("float32").to_frame(),
                               OUT / "raw_1s" / m[:4] / f"freq_zenodo_{m}.parquet")
                q = frequency_features(part, "15min").assign(resolution="15min")
                h = frequency_features(part, "4h").assign(resolution="4h")
                atomic_parquet(pd.concat([q, h]), OUT / "agg" / f"freq_zenodo_agg_{m}.parquet")
                log(f"  {m}: {len(part):>10,} s  coverage {cov:.2%}")
            log(f"BUILD {year}: done in {time.time() - t0:.0f} s")
    combine()


CAL_DAYS = 8


def _reference_day(day: pd.Timestamp) -> pd.Series:
    """TSO archive (preferred) or Energy-Charts 1-s values around one local day."""
    m = day.strftime("%Y-%m")
    tso = sorted(TSO_RAW.glob(f"*/freq_tso_{m}.parquet"))
    files = tso if tso else _ec_files_for(day)
    if not files:
        return pd.Series(dtype="float64")
    parts = [_one_day(files, day + pd.Timedelta(hours=h)) for h in (-2, 0, 2)]
    r = pd.concat(parts)
    return r[~r.index.duplicated(keep="first")].sort_index()


def calibrate_tz(d: pd.DataFrame) -> tuple[str, str]:
    """Pick local / fixed (CET) / utc by matching sample days against the reference.
    Score = median over days of the MAE at the best lag in -5..+5 s."""
    ts = pd.DatetimeIndex(d["ts_naive"])
    dates = pd.Series(1, index=ts).groupby(ts.normalize()).size()
    dates = dates[dates > 80000].index
    if len(dates) == 0:
        return "fixed", "no full day to calibrate; default fixed CET"
    pick = [dates[i] for i in np.linspace(0, len(dates) - 1, min(len(dates), 3 * CAL_DAYS)).astype(int)]
    scores: dict[str, list[float]] = {"local": [], "fixed": [], "utc": []}
    used = 0
    for day in pick:
        if used >= CAL_DAYS:
            break
        dl = day.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent="shift_forward")
        ref = _reference_day(dl)
        if len(ref) < 20000:
            continue
        sub = d[(d["ts_naive"] >= day - pd.Timedelta(hours=3)) & (d["ts_naive"] < day + pd.Timedelta(hours=27))]
        day_scores = {}
        for mode in scores:
            z, _ = P.to_utc_1s(sub, mode)
            z = z[(z.index >= dl.tz_convert("UTC")) & (z.index < (dl + pd.Timedelta(days=1)).tz_convert("UTC"))]
            best = np.inf
            for lag in range(-5, 6):
                r = ref.reindex(z.index + pd.Timedelta(seconds=lag)).values
                ok = ~np.isnan(r)
                if ok.sum() > 10000:
                    best = min(best, float(np.mean(np.abs(r[ok] - z.values[ok])) * 1000))
            day_scores[mode] = best
        if all(np.isfinite(v) for v in day_scores.values()):
            for mode, v in day_scores.items():
                scores[mode].append(v)
            used += 1
    if used == 0:
        detected, _ = P.detect_tz_mode(d["ts_naive"])
        mode = detected if detected != "unknown" else "fixed"
        return mode, "no reference overlap; DST check"
    med = {m: float(np.median(v)) for m, v in scores.items()}
    mode = min(med, key=med.get)
    txt = ", ".join(f"{m} {v:.2f}" for m, v in sorted(med.items(), key=lambda x: x[1]))
    return mode, f"calibrated on {used} days, median MAE mHz: {txt}"


def _file_rank(name: str, year: int) -> tuple:
    n = name.replace("\\", "/")
    return (0 if f"TransnetBW/{year}.zip" in n else 1 if "TransnetBW/" in n else 2, n)


def parse_allow_nan(opener):
    """Like P.parse_member, but NaN rows are expected (Zenodo marks gaps with NaN)."""
    fmt = P.sniff(P.head_bytes(opener))
    parts = []
    with opener() as fh:
        reader = pd.read_csv(fh, sep=fmt.sep, header=None, skiprows=fmt.skiprows, dtype=str,
                             encoding=fmt.encoding, skipinitialspace=True,
                             engine="c" if fmt.sep != r"\s+" else "python",
                             skip_blank_lines=True, on_bad_lines="skip", chunksize=P.CHUNK_ROWS)
        for df in reader:
            parts.append(P._parse_chunk(df, fmt))
    return pd.concat(parts, ignore_index=True), fmt


def combine() -> None:
    files = sorted((OUT / "agg").glob("freq_zenodo_agg_*.parquet"))
    if not files:
        return
    allagg = pd.concat([pd.read_parquet(p) for p in files])
    for res in ("15min", "4h"):
        part = allagg[allagg["resolution"] == res].drop(columns="resolution")
        part = part[~part.index.duplicated(keep="first")].sort_index()
        atomic_parquet(part, OUT / f"frequency_zenodo_{res}.parquet")
        log(f"COMBINE: frequency_zenodo_{res}.parquet: {len(part):,} rows, "
            f"{part.index.min()} -> {part.index.max()}")


# ----------------------------------------------------------------------- verify

def _one_day(files: list[Path], day: pd.Timestamp) -> pd.Series:
    """1-s values of one local day from a list of parquet files (UTC index)."""
    a = day.tz_convert("UTC")
    b = (day + pd.Timedelta(days=1)).tz_convert("UTC")
    parts = []
    for p in files:
        s = pd.read_parquet(p).iloc[:, 0].astype("float64").dropna()
        parts.append(s[(s.index >= a) & (s.index < b)])
    if not parts:
        return pd.Series(dtype="float64")
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep="first")].sort_index()


def _best_day(path: Path) -> pd.Timestamp:
    """Local day with the most Zenodo seconds in this month file (closest to the 15th on ties)."""
    idx = pd.read_parquet(path).index.tz_convert(LOCAL_TZ)
    days = pd.Series(1, index=idx).groupby(idx.strftime("%Y-%m-%d")).sum()
    days = days[days == days.max()]
    best = min(days.index, key=lambda d: abs(int(d[-2:]) - 15))
    return pd.Timestamp(best).tz_localize(LOCAL_TZ)


def _ec_files_for(day: pd.Timestamp) -> list[Path]:
    out = []
    d = day.tz_localize(None).normalize()
    for p in EC_RAW.glob(f"{d.year}/freq_*.parquet"):
        a, b = (pd.Timestamp(x) for x in p.stem.split("_")[1:3])
        if a - pd.Timedelta(days=1) <= d <= b + pd.Timedelta(days=1):
            out.append(p)
    return out


def verify() -> None:
    """Clock check per sampled day (best-covered day of up to 10 overlapping months;
    month files only, so memory stays small). Shifts tested: -2 h..+2 h and -5..+5 s."""
    buf: list[str] = []
    zfiles = {p.stem.split("_")[-1]: p for p in (OUT / "raw_1s").glob("*/freq_zenodo_*.parquet")}
    if not zfiles:
        log("VERIFY: no Zenodo data built yet")
        return
    log("=" * 70, buf)
    log(f"Zenodo vs TSO archive / Energy-Charts — {datetime.now():%Y-%m-%d %H:%M}", buf)
    log("=" * 70, buf)
    log(f"Zenodo months: {min(zfiles)} -> {max(zfiles)} ({len(zfiles)} files)", buf)
    tso_files = {p.stem.split("_")[-1]: p for p in TSO_RAW.glob("*/freq_tso_*.parquet")}
    ec_months = {p.parent.name for p in EC_RAW.glob("*/freq_*.parquet")}
    refs = {
        "TSO archive": sorted(set(zfiles) & set(tso_files)),
        "Energy-Charts": sorted(m for m in zfiles if m[:4] in ec_months and m >= "2022-05"),
    }
    for label, months in refs.items():
        if not months:
            log(f"\n{label}: no overlapping months", buf)
            continue
        pick = [months[i] for i in np.linspace(0, len(months) - 1, min(10, len(months))).astype(int)]
        log(f"\n{label}: {len(months)} overlapping months, one full day sampled from {', '.join(pick)}", buf)
        n_ok = 0
        for m in pick:
            day = _best_day(zfiles[m])
            zz = _one_day([zfiles[m]], day)
            rfiles = [tso_files[m]] if label == "TSO archive" else _ec_files_for(day)
            # wider window so that +-1 h shifts still find reference values
            ref = pd.concat([_one_day(rfiles, day + pd.Timedelta(hours=h)) for h in (-2, 0, 2)])
            ref = ref[~ref.index.duplicated(keep="first")].sort_index()
            res = {}
            for shift in [-7200, -3600, 3600, 7200] + list(range(-5, 6)):
                r = ref.reindex(zz.index + pd.Timedelta(seconds=shift))
                ok = r.notna().values
                if ok.sum() > 1000:
                    res[shift] = float(np.mean(np.abs(r.values[ok] - zz.values[ok])) * 1000)
            if not res:
                log(f"  {day:%Y-%m-%d}: no reference values", buf)
                continue
            best = min(res, key=res.get)
            n_ok += abs(best) <= 5
            flag = "" if abs(best) <= 5 else "   <-- CLOCK OFF"
            log(f"  {day:%Y-%m-%d}: best shift {best:+6d} s, MAE {res[best]:6.3f} mHz "
                f"(at 0 s: {res.get(0, float('nan')):6.3f}){flag}", buf)
        log(f"  -> {n_ok}/{len(pick)} sampled days within +-5 s"
            + ("" if n_ok == len(pick) else "  — CHECK: rebuild the affected years with --tz"), buf)
    p = P.DATA / "compare" / "zenodo_verify.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(buf) + "\n", encoding="utf-8")
    print(f"\nWritten: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="download + list content only")
    ap.add_argument("--years", type=int, nargs="+", help="CE years to build (default: all found)")
    ap.add_argument("--tz", choices=["auto", "local", "fixed", "utc"], default="auto")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if args.list:
        do_list()
        return
    if not args.verify_only:
        build(args.years, args.tz)
    verify()


if __name__ == "__main__":
    main()
