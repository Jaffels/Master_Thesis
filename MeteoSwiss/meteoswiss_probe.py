"""
MeteoSwiss probe — download station/parameter/inventory metadata and select
the stations for the two weighted Swiss series (load and hydro).

Selection rules (settings in meteoswiss_common.py):
  Eligible station = SwissMetNet station, not on a crest, every required
  hourly parameter measured since <= PULL_START and still reported within
  the last 30 days.
  LOAD : nearest eligible station (height <= 1000 m) to the main load centre
         of each Swissgrid Energy Overview consumption group.
  HYDRO: all eligible stations between 1000 and 2600 m in the main hydro
         cantons (VS, GR, TI, GL, OW/NW/UR).

Outputs (MeteoSwiss/Data/):
  meta/ogd-smn_meta_*.csv     raw metadata as downloaded
  meta/snow_parameter.txt     hourly snow-depth parameter, if one exists
  station_candidates.csv      every eligible station with distance/height per
                              group — review this to judge the selection
  station_selection.csv       selected stations + within-group weights
  probe_report.txt            human-readable summary

Usage (from the thesis root, venv active):
    python MeteoSwiss/meteoswiss_probe.py
    python MeteoSwiss/meteoswiss_probe.py --offline   # reuse saved metadata
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import meteoswiss_common as C  # noqa: E402

META_FILES = {
    "stations": "ogd-smn_meta_stations.csv",
    "parameters": "ogd-smn_meta_parameters.csv",
    "inventory": "ogd-smn_meta_datainventory.csv",
}


def load_meta(offline: bool) -> dict[str, pd.DataFrame]:
    C.META_DIR.mkdir(parents=True, exist_ok=True)
    assets = None
    out = {}
    for key, fname in META_FILES.items():
        path = C.META_DIR / fname
        if not offline:
            if assets is None:
                assets = C.http_get(f"{C.BASE}/collections/{C.COLLECTION}").json()["assets"]
            content = C.http_get(assets[fname]["href"]).content
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(content)
            tmp.replace(path)
        out[key] = C.read_semicolon_csv(path.read_bytes())
        print(f"  {fname}: {len(out[key]):,} rows; columns: {list(out[key].columns)}")
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def detect_snow_param(params: pd.DataFrame) -> str | None:
    code = C.find_col(params, "shortname")
    text_cols = [c for c in params.columns if "description" in c.lower()]
    gran = next((c for c in params.columns if "granularity" in c.lower()), None)
    text = params[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
    mask = text.str.contains("snow depth|schneehöhe|schneehoehe", regex=True)
    if gran is not None:
        mask &= params[gran].astype(str).str.upper().eq("H")
    else:
        mask &= params[code].astype(str).str.endswith(("h0", "hs0"))
    hits = params.loc[mask, code].astype(str).tolist()
    return hits[0] if hits else None


def station_table(meta, needed_params: list[str]) -> pd.DataFrame:
    st, inv = meta["stations"], meta["inventory"]
    s = pd.DataFrame({
        "station_abbr": st[C.find_col(st, "station_abbr")].astype(str),
        "station_name": st[C.find_col(st, "station_name")],
        "canton": st[C.find_col(st, "canton")].astype(str),
        "height_m": pd.to_numeric(st[C.find_col(st, "height", exclude=("barometer",))], errors="coerce"),
        "lat": pd.to_numeric(st[C.find_col(st, "wgs84", "lat")], errors="coerce"),
        "lon": pd.to_numeric(st[C.find_col(st, "wgs84", "lon")], errors="coerce"),
        "exposition": st[C.find_col(st, "exposition", "en")].astype(str).str.lower(),
    })

    i_st = C.find_col(inv, "station_abbr")
    i_par = C.find_col(inv, "parameter")
    i_since = C.find_col(inv, "since")
    i_till = C.find_col(inv, "till")
    inv = inv[[i_st, i_par, i_since, i_till]].copy()
    inv.columns = ["station_abbr", "param", "since", "till"]
    for c in ("since", "till"):
        inv[c] = pd.to_datetime(inv[c], dayfirst=True, errors="coerce")

    start = pd.Timestamp(C.PULL_START_LOCAL)
    recent = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    for p in needed_params:
        sub = inv[inv["param"] == p].set_index("station_abbr")
        s[f"{p}_since"] = s["station_abbr"].map(sub["since"])
        s[f"{p}_till"] = s["station_abbr"].map(sub["till"])
        s[f"has_{p}"] = (s[f"{p}_since"] <= start) & (s[f"{p}_till"].isna() | (s[f"{p}_till"] >= recent))
    s["crest"] = s["exposition"].str.contains("crest")
    s["excluded"] = s["station_abbr"].isin(C.EXCLUDE_STATIONS)
    return s


def select(stations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cands = [], []
    ok = ~stations["crest"] & ~stations["excluded"]

    # ---- LOAD
    elig_load = stations[ok & stations["height_m"].le(C.LOAD_MAX_HEIGHT_M)
                         & np.logical_and.reduce([stations[f"has_{p}"] for p in C.REQUIRED["load"]])]
    for group, anchors in C.LOAD_ANCHORS.items():
        override = C.OVERRIDES.get(f"load:{group}")
        picked = []
        for name, lat, lon in anchors:
            d = haversine_km(lat, lon, elig_load["lat"], elig_load["lon"])
            ranked = elig_load.assign(distance_km=d.round(1)).sort_values("distance_km")
            for rank, (_, r) in enumerate(ranked.head(3).iterrows(), 1):
                cands.append({"role": "load", "group": group, "anchor": name, "rank": rank,
                              **r[["station_abbr", "station_name", "canton", "height_m", "distance_km"]]})
            if not override and len(ranked):
                best = ranked.iloc[0]
                picked.append((name, best["station_abbr"], best["distance_km"]))
        if override:
            picked = [("override", a, np.nan) for a in override]
        n = len(picked)
        for name, abbr, dist in picked:
            rows.append({"role": "load", "group": group, "anchor": name, "station_abbr": abbr,
                         "distance_km": dist, "weight_in_group": 1.0 / n,
                         "far_flag": bool(dist > C.LOAD_MAX_DISTANCE_KM) if pd.notna(dist) else False})

    # ---- HYDRO
    lo, hi = C.HYDRO_HEIGHT_RANGE_M
    elig_hydro = stations[ok & stations["height_m"].between(lo, hi)
                          & np.logical_and.reduce([stations[f"has_{p}"] for p in C.REQUIRED["hydro"]])]
    for group, cantons in C.HYDRO_GROUPS.items():
        override = C.OVERRIDES.get(f"hydro:{group}")
        sub = elig_hydro[elig_hydro["canton"].isin(cantons)].sort_values("height_m")
        for _, r in sub.iterrows():
            cands.append({"role": "hydro", "group": group, "anchor": "/".join(cantons), "rank": np.nan,
                          **r[["station_abbr", "station_name", "canton", "height_m"]], "distance_km": np.nan})
        abbrs = override or sub["station_abbr"].tolist()
        for a in abbrs:
            rows.append({"role": "hydro", "group": group, "anchor": "/".join(cantons), "station_abbr": a,
                         "distance_km": np.nan, "weight_in_group": 1.0 / len(abbrs), "far_flag": False})

    sel = pd.DataFrame(rows)
    # One station can serve two anchors of the same group -> merge weights.
    sel = (sel.groupby(["role", "group", "station_abbr"], as_index=False)
              .agg(anchor=("anchor", " + ".join), distance_km=("distance_km", "max"),
                   weight_in_group=("weight_in_group", "sum"), far_flag=("far_flag", "max")))
    info = stations.set_index("station_abbr")[["station_name", "canton", "height_m", "lat", "lon"]]
    sel = sel.join(info, on="station_abbr")
    return sel, pd.DataFrame(cands)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="reuse metadata saved in Data/meta")
    args = ap.parse_args()

    print("Metadata:")
    meta = load_meta(args.offline)

    snow = detect_snow_param(meta["parameters"])
    (C.META_DIR / "snow_parameter.txt").write_text(snow or "")
    print(f"Hourly snow-depth parameter: {snow or 'none found'}")

    needed = sorted(set(sum(C.REQUIRED.values(), [])))
    stations = station_table(meta, needed)
    sel, cands = select(stations)

    C.DATA.mkdir(parents=True, exist_ok=True)
    cands.to_csv(C.CANDIDATES_CSV, index=False)
    sel.to_csv(C.SELECTION_CSV, index=False)

    lines = [f"MeteoSwiss probe — {pd.Timestamp.now():%Y-%m-%d %H:%M}",
             f"Stations in metadata: {len(stations)}; snow parameter: {snow or 'none'}",
             f"Eligible (not crest, required params since {C.PULL_START_LOCAL}, still active):"]
    for role, req in C.REQUIRED.items():
        m = ~stations["crest"] & np.logical_and.reduce([stations[f"has_{p}"] for p in req])
        lines.append(f"  {role}: {int(m.sum())} stations have {req}")
    missing_groups = [g for g in C.LOAD_ANCHORS if g not in set(sel.loc[sel.role == "load", "group"])]
    missing_groups += [g for g in C.HYDRO_GROUPS if g not in set(sel.loc[sel.role == "hydro", "group"])]
    lines.append(f"Groups without a station: {missing_groups or 'none'}")
    far = sel[sel["far_flag"]]
    lines.append(f"Load stations > {C.LOAD_MAX_DISTANCE_KM} km from their anchor: "
                 + (", ".join(f"{r.group}:{r.station_abbr} ({r.distance_km} km)" for r in far.itertuples()) or "none"))
    lines.append("")
    lines.append(sel.to_string(index=False))
    report = "\n".join(lines)
    (C.DATA / "probe_report.txt").write_text(report)
    print("\n" + report)
    print(f"\nWrote {C.SELECTION_CSV.relative_to(C.ROOT.parent)} ({sel.station_abbr.nunique()} stations)")


if __name__ == "__main__":
    main()
