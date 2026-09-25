# inventory_manual_download.py
from pathlib import Path
import pandas as pd

ROOT = Path("Swissgrid/Manual_Download")   # adjust to your folder
OUT = Path("manual_download_inventory.txt")
N = 8  # preview rows

def preview_csv(p):
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(p, encoding=enc) as f:
                head = "".join(next(f, "") for _ in range(N + 5))
            return f"[encoding={enc}]\n{head}"
        except UnicodeDecodeError:
            continue
    return "[could not decode]"

def preview_excel(p):
    out = []
    xl = pd.ExcelFile(p)
    for sh in xl.sheet_names:
        df = xl.parse(sh, header=None, nrows=N + 5)
        full = xl.parse(sh, header=None, usecols=[0])
        out.append(f"--- sheet '{sh}' (~{len(full)} rows, {df.shape[1]} cols)\n"
                   f"{df.to_string(max_colwidth=30)}")
    return "\n".join(out)

with OUT.open("w", encoding="utf-8") as o:
    files = sorted(p for p in ROOT.rglob("*") if p.is_file())
    o.write(f"{len(files)} files under {ROOT}\n\n")
    for p in files:
        o.write(f"{p.relative_to(ROOT)}  ({p.stat().st_size/1e6:.2f} MB)\n")
    for p in files:
        o.write(f"\n\n===== {p.relative_to(ROOT)} =====\n")
        try:
            suf = p.suffix.lower()
            if suf in (".csv", ".txt"):
                o.write(preview_csv(p))
            elif suf in (".xlsx", ".xls", ".xlsm"):
                o.write(preview_excel(p))
            else:
                o.write("[skipped: unsupported type]")
        except Exception as e:
            o.write(f"[error: {e!r}]")
print(f"wrote {OUT}")