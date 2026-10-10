"""
Swiss school holidays per canton and a load-weighted national share per day
(RQ3 decision D5, 8 Oct 2026: calendar mechanism for oversupply / undersupply spikes).

Source 2020 -> 2027: OpenHolidays API (openholidaysapi.org, /SchoolHolidays, all 26
cantons). Compulsory school (group code ending in "-VS", Volksschule) is used where
a canton publishes several school types; otherwise all entries of the canton.
2016 -> 2019 are NOT in the API. They are back-cast per canton and holiday name from
the 2020-2027 pattern: Swiss cantons fix most holidays by ISO week, Easter-linked ones
by their distance to Easter Sunday. For each canton x holiday name the rule with the
smaller spread over 2020-2027 is used (start = modal ISO week + weekday, or modal
offset to Easter; length = modal length). A leave-one-year-out test on 2020-2026
reports how many holiday days the rule reproduces. Back-cast days are flagged.

Weights: annual consumption share of the Swissgrid Energy Overview groups (the same
shares as the load-weighted weather series); within a group of several cantons the
share is split equally (simplification, e.g. SH/ZH, GE/VD, BL/BS).

Writes (only with --write):
  Calendar/Data/raw/openholidays_<year>.json        API answers (cached)
  Clean/Data/calendar/school_holidays.parquet       one row per local day 2015-2027:
      date, ch_school_holiday_share (0-1, load-weighted), n_cantons_on_holiday,
      flag_school_holiday_backcast, sh_<canton> (0/1 per canton)
  Clean/Data/calendar/school_holidays_dictionary.csv, school_holidays_report.txt
Load: views.load(..., school_holidays=True) / views.load_rq3(school_holidays=True)

Run from Master_Thesis with .venv active (27 requests, < 1 min):
    python Calendar/school_holidays.py            # pull + back-cast + checks, no files
    python Calendar/school_holidays.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Calendar" / "Data" / "raw"
OUT = ROOT / "Clean" / "Data" / "calendar"
EO_QH = ROOT / "SwissGrid" / "EnergyOverview" / "Data" / "qh"
URL = "https://openholidaysapi.org/SchoolHolidays"
API_FROM, API_TO = 2020, 2027
FIRST, LAST = 2015, 2027
CANTONS = ["AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", "JU", "LU", "NE", "NW",
           "OW", "SG", "SH", "SO", "SZ", "TG", "TI", "UR", "VD", "VS", "ZG", "ZH"]
GROUPS = {"sh_zh": ["SH", "ZH"], "be_ju": ["BE", "JU"], "ge_vd": ["GE", "VD"], "ag": ["AG"],
          "vs": ["VS"], "sg": ["SG"], "lu": ["LU"], "bl_bs": ["BL", "BS"], "ti": ["TI"],
          "fr": ["FR"], "gr": ["GR"], "gl": ["GL"], "tg": ["TG"], "sz_zg": ["SZ", "ZG"],
          "so": ["SO"], "ow_nw_ur": ["OW", "NW", "UR"], "ne": ["NE"], "ai_ar": ["AI", "AR"]}


def easter(y: int) -> date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, month, day)


def pull(year: int, force: bool) -> list[dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"openholidays_{year}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    params = {"countryIsoCode": "CH", "languageIsoCode": "DE",
              "validFrom": f"{year}-01-01", "validTo": f"{year}-12-31"}
    for i in range(4):
        try:
            r = requests.get(URL, params=params, timeout=60, headers={"accept": "application/json"})
            r.raise_for_status()
            data = r.json()
            f.write_text(json.dumps(data))
            return data
        except (requests.RequestException, ValueError):
            if i == 3:
                raise
            time.sleep(5 * (i + 1))
    return []


def entries(data: list[dict]) -> pd.DataFrame:
    rows = []
    for e in data:
        name = next((n["text"] for n in e.get("name", []) if n.get("language") == "DE"),
                    e.get("name", [{}])[0].get("text", ""))
        groups = [g.get("code", "") for g in e.get("groups", []) or []]
        for s in e.get("subdivisions", []) or []:
            code = s.get("code", "")
            if code.count("-") > 2:           # municipality entries: skip
                continue
            rows.append({"canton": code.split("-")[1], "name": name,
                         "region": code if code.count("-") == 2 else None,
                         "start": pd.Timestamp(e["startDate"]), "end": pd.Timestamp(e["endDate"]),
                         "vs": any(g.endswith("-VS") for g in groups), "has_group": bool(groups)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # canton-wide entries where a canton has them; regional entries only for cantons that
    # publish by region (e.g. GR) -> a day counts if most regions are on holiday (to_days)
    keep = []
    for c, g in df.groupby("canton"):
        g = g[g["region"].isna()] if g["region"].isna().any() else g
        keep.append(g[g["vs"] | ~g["has_group"]] if g["vs"].any() else g)
    return pd.concat(keep).drop_duplicates(["canton", "region", "name", "start", "end"])


def holiday_year(row) -> int:
    # Christmas holidays starting in late December belong to the following school year's
    # calendar year of their end; key them by start year for the rules.
    return row["start"].year


def fit_rules(api: pd.DataFrame) -> pd.DataFrame:
    """canton x name -> rule (iso / easter), anchor, length."""
    out = []
    for (c, n), g in api.groupby(["canton", "name"]):
        g = g.sort_values("start").drop_duplicates("start")
        if len(g) < 3:
            continue
        iso = g["start"].dt.isocalendar()
        iso_key = list(zip(iso["week"].astype(int), g["start"].dt.weekday))
        eo = [(s.date() - easter(s.year)).days for s in g["start"]]
        length = (g["end"] - g["start"]).dt.days
        iso_mode = max(set(iso_key), key=iso_key.count)
        eo_mode = max(set(eo), key=eo.count)
        iso_hits, eo_hits = iso_key.count(iso_mode), eo.count(eo_mode)
        rule = "easter" if eo_hits > iso_hits else "iso"
        out.append({"canton": c, "name": n, "rule": rule,
                    "iso_week": iso_mode[0], "iso_weekday": iso_mode[1], "easter_offset": eo_mode,
                    "length_days": int(length.mode().iloc[0]),
                    "years": len(g), "hits": max(iso_hits, eo_hits)})
    return pd.DataFrame(out)


def apply_rules(rules: pd.DataFrame, years) -> pd.DataFrame:
    rows = []
    for r in rules.itertuples():
        for y in years:
            if r.rule == "easter":
                start = pd.Timestamp(easter(y) + timedelta(days=int(r.easter_offset)))
            else:
                try:
                    start = pd.Timestamp(date.fromisocalendar(y, int(r.iso_week), int(r.iso_weekday) + 1))
                except ValueError:       # week 53 in a 52-week year
                    start = pd.Timestamp(date.fromisocalendar(y, 52, int(r.iso_weekday) + 1))
            rows.append({"canton": r.canton, "name": r.name, "start": start,
                         "end": start + pd.Timedelta(days=int(r.length_days))})
    return pd.DataFrame(rows)


def to_days(df: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(f"{FIRST}-01-01", f"{LAST}-12-31", freq="D")
    m = pd.DataFrame(0, index=days, columns=CANTONS, dtype="int8")
    reg = df["region"].notna() if "region" in df else pd.Series(False, index=df.index)
    for r in df[~reg].itertuples():
        m.loc[r.start:r.end, r.canton] = 1
    for c, g in df[reg].groupby("canton"):        # regional cantons: majority of regions
        n = g["region"].nunique()
        cnt = pd.Series(0, index=days)
        for reg_code, gr in g.groupby("region"):
            on = pd.Series(0, index=days)
            for r in gr.itertuples():
                on.loc[r.start:r.end] = 1
            cnt += on
        m[c] = (cnt / n >= 0.5).astype("int8")
    return m


def loyo(api: pd.DataFrame) -> pd.DataFrame:
    """leave-one-year-out: fit on the other API years, predict the held-out year."""
    res = []
    years = sorted(set(api["start"].dt.year) & set(range(API_FROM, API_TO)))
    truth = to_days(api)
    for y in years:
        rules = fit_rules(api[api["start"].dt.year != y])
        pred = to_days(apply_rules(rules, [y]))
        t = truth.loc[str(y)]
        p = pred.loc[str(y)]
        res.append({"year": y, "holiday_days_true": int(t.values.sum()),
                    "recall": round((t.values & p.values).sum() / max(t.values.sum(), 1), 3),
                    "precision": round((t.values & p.values).sum() / max(p.values.sum(), 1), 3)})
    return pd.DataFrame(res)


def weights() -> pd.DataFrame:
    """year x canton weight (sums to 1), from Energy Overview consumption groups."""
    rows = {}
    for f in sorted(EO_QH.glob("*.parquet")):
        cols = [f"cons_{g}_kwh" for g in GROUPS]
        d = pd.read_parquet(f, columns=["timestamp"] + cols)
        y = int(d["timestamp"].dt.year.mode()[0])
        s = d[cols].sum(min_count=1)
        if s.isna().all():
            continue
        s = s / s.sum()
        w = {}
        for g, cs in GROUPS.items():
            for c in cs:
                w[c] = s[f"cons_{g}_kwh"] / len(cs)
        rows[y] = w
    w = pd.DataFrame(rows).T.sort_index()
    return w.reindex(range(FIRST, LAST + 1)).ffill().bfill()


def main() -> int:
    ap = argparse.ArgumentParser(description="Swiss school holidays (D5)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download the API years")
    a = ap.parse_args()

    api = pd.concat([entries(pull(y, a.force)) for y in range(API_FROM, API_TO + 1)])
    api = api.drop_duplicates(["canton", "region", "name", "start", "end"])
    missing = sorted(set(CANTONS) - set(api["canton"]))
    rules = fit_rules(api)
    back = apply_rules(rules, range(FIRST - 1, API_FROM))
    days_api, days_back = to_days(api), to_days(back)
    api_from = api["start"].min()
    m = days_api.copy()
    pre = m.index < api_from
    m.loc[pre] = days_back.loc[pre]
    w = weights()
    yr = m.index.year
    W = w.loc[yr].set_index(m.index)[CANTONS]
    share = (m[CANTONS] * W).sum(axis=1)
    out = pd.DataFrame({"date": m.index.date,
                        "ch_school_holiday_share": share.round(4).to_numpy(),
                        "n_cantons_on_holiday": m[CANTONS].sum(axis=1).to_numpy(),
                        "flag_school_holiday_backcast": pre})
    for c in CANTONS:
        out[f"sh_{c.lower()}"] = m[c].to_numpy()

    test = loyo(api)
    lines = ["Swiss school holidays", "",
             f"API entries: {len(api)} ({api['start'].min().date()} -> {api['end'].max().date()}), "
             f"cantons with data: {api['canton'].nunique()}/26" + (f", missing: {missing}" if missing else ""),
             f"rules fitted (canton x holiday): {len(rules)}; easter-linked: {(rules['rule'] == 'easter').sum()}",
             f"back-cast days before {api_from.date()}", "",
             "leave-one-year-out test of the back-cast rules (holiday days, all cantons):",
             test.to_string(index=False), "",
             "load-weighted share of Switzerland on school holiday, mean by month (2016-2026):",
             out.assign(m=pd.to_datetime(out["date"]).dt.month,
                        y=pd.to_datetime(out["date"]).dt.year)
                .query("y >= 2016 and y <= 2026").groupby("m")["ch_school_holiday_share"].mean()
                .round(2).to_string()]
    print("\n".join(lines))
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = OUT / "school_holidays.tmp"
        out.to_parquet(tmp, index=False)
        tmp.replace(OUT / "school_holidays.parquet")
        pd.DataFrame([
            dict(column="ch_school_holiday_share", unit="share", source="OpenHolidays API 2020-; back-cast 2015-2019",
                 rule="sum of canton school-holiday flags weighted by annual consumption share (Energy Overview groups, split equally within a group)",
                 availability="known years ahead (calendar) -> ex ante"),
            dict(column="n_cantons_on_holiday", unit="count", source="as above", rule="cantons on holiday", availability="ex ante"),
            dict(column="flag_school_holiday_backcast", unit="bool", source="", rule="day before the first API year: back-cast by ISO week / Easter rule", availability=""),
            dict(column="sh_<canton>", unit="0/1", source="as above", rule="compulsory-school holiday in the canton", availability="ex ante"),
        ]).to_csv(OUT / "school_holidays_dictionary.csv", index=False)
        (OUT / "school_holidays_report.txt").write_text("\n".join(lines) + "\n")
        print(f"\nwritten: {OUT.relative_to(ROOT)}/school_holidays.parquet + dictionary + report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
