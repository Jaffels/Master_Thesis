# profile_swissgrid.py
from pathlib import Path
import hashlib, pandas as pd

ROOT = Path("Swissgrid/Manual_Download")
out = open("swissgrid_profile.txt", "w", encoding="utf-8")
w = lambda *a: print(*a, file=out)

# 1) Auction results: ID patterns, units, pricing rule per product/year
for p in sorted((ROOT / "Tenders/Auction_Results_2015-2025").glob("*.csv")):
    df = pd.read_csv(p, sep=";", dtype=str, encoding="utf-8-sig")
    df.columns = [f"{c}__{i}" if c.startswith("Einheit") else c for i, c in enumerate(df.columns)]
    df["prod"] = df["Ausschreibung"].str.split("_").str[0]
    w(f"\n===== {p.name}  rows={len(df)}  cols={list(df.columns)}")
    for prod, g in df.groupby("prod"):
        ids = g["Ausschreibung"].unique()
        unit_cols = [c for c in g.columns if c.startswith("Einheit")]
        w(f"-- {prod}: {len(ids)} auctions, {len(g)} rows")
        w("   ids:", list(ids[:4]), "...", list(ids[-3:]))
        w("   Beschreibung:", list(g["Beschreibung"].unique()[:12]))
        w("   units:", g[unit_cols].drop_duplicates().head(5).values.tolist())
        w("   Land:", g.get("Land", pd.Series(dtype=str)).value_counts().to_dict())
        w("   rows awarded==0:", (pd.to_numeric(g["Zugesprochenes Volumen"], errors="coerce") == 0).sum())
        n_preis = g.groupby("Ausschreibung")["Preis"].nunique()
        w(f"   auctions with single Preis (marginal?): {(n_preis == 1).mean():.0%}")

# 2) Energy Overview: dedupe copies, then structure of 2019/2020/2025
groups = {}
for p in ROOT.rglob("EnergieUebersichtCH-*"):
    groups.setdefault(p.name, []).append(hashlib.md5(p.read_bytes()).hexdigest())
w("\n===== EnergieUebersicht copies identical:",
  all(len(set(v)) == 1 for v in groups.values()))
for name in ["EnergieUebersichtCH-2019.xls", "EnergieUebersichtCH-2020.xlsx", "EnergieUebersichtCH-2025.xlsx"]:
    p = ROOT / "Balancing" / name
    xl = pd.ExcelFile(p)
    w(f"\n===== {name} sheets={xl.sheet_names}")
    for sh in xl.sheet_names:
        d = xl.parse(sh, header=None, nrows=4)
        w(f"--- {sh}\n{d.T.head(80).to_string(max_colwidth=60)}")

# 3) One imbalance-price file in each format
p = next((ROOT / "Prices_for_imbalance_energy").rglob("*250101-250201.xlsx"))
w(f"\n===== {p.name}\n", pd.read_excel(p, header=None, nrows=12).to_string())
w(open(p.with_suffix(".xml"), encoding="utf-8", errors="replace").read()[:3000])
out.close()