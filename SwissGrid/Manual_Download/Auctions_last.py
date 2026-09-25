import pandas as pd, re
from pathlib import Path
ID_RE = re.compile(r"^(PRL|SRL|TRL[+-]?)_\d{2}_(?:KW\d{2}|\d{2}_\d{2})(?:_S\d+)?$")
for f in ["2018-PRL-SRL-TRL-Ergebnis.csv", "2019_PRL_SRL_TRL_Ergebnis.csv"]:
    p = next(Path("Swissgrid/Manual_Download/Tenders").rglob(f))
    df = pd.read_csv(p, sep=";", dtype=str, encoding="utf-8-sig")
    bad = df[~df["Ausschreibung"].str.match(ID_RE)]
    print(f, bad.groupby(["Ausschreibung", "Beschreibung"]).size().to_string(), "\n")