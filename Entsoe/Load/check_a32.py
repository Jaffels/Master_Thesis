"""One-off diagnostic: is month-ahead (6.1.D / A32) genuinely absent for a
zone/year, or present-but-dropped by the parser?

Run from Entsoe/Load/ (same venv/key as the pull):
    ../../.venv/bin/python check_a32.py CH 2021
    ../../.venv/bin/python check_a32.py AT 2023
    ../../.venv/bin/python check_a32.py FR 2021      # a known-populated control

For each it prints three things:
  1. whether the keep-all parser patch is actually loaded in this process;
  2. what TP's RAW A65/A32 document contains for that zone+year — the number of
     TimeSeries, Points, and the distinct businessType codes (or NoMatchingData);
  3. what the patched client method returns (what the pull actually sees).

Reading the result:
  * RAW says NoMatchingDataError, or 0 TimeSeries / 0 Points  -> TP has no
    month-ahead forecast for that zone+year. 'empty' is correct; the data does
    not exist and cannot be recovered. (Expected for TSOs that started
    publishing 6.1.D later than 2021.)
  * RAW shows TimeSeries WITH Points but the client method returns 0 rows ->
    a real parser/patch problem; the printed businessTypes say what to map.
"""

import sys
from pathlib import Path

import bs4
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entsoe_load_probe as probe          # applies keep-all + margin patches
import entsoe.parsers as parsers
from entsoe import EntsoePandasClient
from entsoe.mappings import lookup_area

zone = sys.argv[1] if len(sys.argv) > 1 else "CH"
year = int(sys.argv[2]) if len(sys.argv) > 2 else 2021

# 1) is the parser patch actually loaded in THIS process?
pl = parsers.parse_loads
print(f"parse_loads -> {pl.__module__}.{pl.__qualname__}")
print(f"keep-all patch loaded? {pl.__module__ == 'entsoe_load_probe'}\n")

key = probe.load_api_key()
if not key:
    print("no ENTSOE_API_KEY — export it or put it in .env")
    sys.exit(1)

client = EntsoePandasClient(api_key=key)
area = lookup_area(zone)
tz = probe.TZ
start = pd.Timestamp(f"{year}-01-01", tz=tz)
end = pd.Timestamp(f"{year + 1}-01-01", tz=tz)

# 2) RAW document straight from TP — the ground truth
params = {"documentType": "A65", "processType": "A32",
          "outBiddingZone_Domain": area.code}
try:
    text = client._base_request(params=params, start=start, end=end).text
    soup = bs4.BeautifulSoup(text, "html.parser")
    ts = soup.find_all("timeseries")
    bts = sorted({t.find("businesstype").text
                  for t in ts if t.find("businesstype")})
    pts = sum(len(t.find_all("point")) for t in ts)
    print(f"RAW A32 {zone} {year}: xml={len(text)}B  timeseries={len(ts)}  "
          f"points={pts}  businessTypes={bts or '(none)'}")
except Exception as e:
    print(f"RAW A32 {zone} {year}: {type(e).__name__}: {str(e)[:160]}")

# 3) through the patched client method — exactly what the pull consumes
try:
    df = client.query_load_forecast(zone, start=start, end=end,
                                    process_type="A32")
    print(f"query_load_forecast -> rows={len(df)}  cols={list(df.columns)}")
except Exception as e:
    print(f"query_load_forecast -> {type(e).__name__}: {str(e)[:160]}")
