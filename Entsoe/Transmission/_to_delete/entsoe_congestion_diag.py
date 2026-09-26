"""
Diagnostic for the 13.1.B / 13.1.C (A91/A92) 400s.

Hits the live ENTSO-E API directly (bypassing entsoe-py so the raw
Acknowledgement <Reason> is visible) with several parameter sets on the CH-AT
border — the one that showed one direction empty and one direction 400 — plus
the DE border. For each it prints HTTP status, the TimeSeries count, and the
exact reason code+text. That tells us whether the fix is a required businessType
(B03 counter-trade / B04 congestion-costs / A46 system-operator), a direction
restriction, or that CH genuinely has no such data.

Run from the thesis root (reads the same .env as the probe):
    .venv/bin/python Entsoe/Transmission/entsoe_congestion_diag.py
"""
from __future__ import annotations
import os, re, sys
from pathlib import Path
import requests

BASE = "https://web-api.tp.entsoe.eu/api"
PERIOD = {"periodStart": "202401010000", "periodEnd": "202402010000"}  # Jan 2024

C = {
    "CH": "10YCH-SWISSGRIDZ",
    "AT": "10YAT-APG------L",
    "DE_LU": "10Y1001A1001A82H",
    "DE_TRANSNET": "10YDE-ENBW-----N",
}


def get_key() -> str:
    k = os.getenv("ENTSOE_API_KEY")
    if k:
        return k
    here = Path(__file__).resolve().parent
    for env in [p / ".env" for p in [here, *here.parents][:6]]:
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("ENTSOE_API_KEY") and "=" in line:
                    return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def run(label: str, params: dict, key: str) -> None:
    p = {**params, **PERIOD, "securityToken": key}
    try:
        r = requests.get(BASE, params=p, timeout=60)
    except Exception as e:
        print(f"{label:44s} REQUEST-FAILED {type(e).__name__}: {e}")
        return
    n_ts = r.text.count("<TimeSeries>")
    codes = re.findall(r"<code>(.*?)</code>", r.text)
    texts = re.findall(r"<text>(.*?)</text>", r.text)
    reason = "; ".join(f"{c}:{t}" for c, t in zip(codes, texts)) or "(no Reason tag)"
    print(f"{label:44s} HTTP {r.status_code}  TS={n_ts:<3d} {reason[:110]}")


def main() -> int:
    key = get_key()
    if not key:
        print("ENTSOE_API_KEY not found (env or .env)")
        return 1

    # documentType, then which businessType (None = omit)
    for dt, name, bts in [
        ("A91", "COUNTERTRADING", [None, "B03", "A46"]),
        ("A92", "CONGESTION_COSTS", [None, "B04", "A46"]),
    ]:
        print(f"\n=== {name}  (documentType={dt}) ===")
        for bt in bts:
            for f, t in [("CH", "AT"), ("AT", "CH"),
                         ("CH", "DE_TRANSNET"), ("CH", "DE_LU")]:
                # in_Domain = to, out_Domain = from  (entsoe-py convention)
                params = {"documentType": dt, "in_Domain": C[t], "out_Domain": C[f]}
                if bt:
                    params["businessType"] = bt
                tag = f"in={t:11s} out={f:11s} bt={bt or '-':4s}"
                run(tag, params, key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
