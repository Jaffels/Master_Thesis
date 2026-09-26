_26 September 2026_

## The big picture

My thesis needs one big table: a row for every 15 minutes from 2021 to 2026, and a column
for everything that might explain reserve power prices (electricity use, solar and wind,
water in the reservoirs, cross-border capacity, outages, imbalance prices, and the reserve
prices themselves). The data comes from two places: **Swissgrid** (the Swiss grid operator)
and **ENTSO-E** (the European platform where all grid operators publish their data).

Before joining the two sources, I wanted each one to be clean on its own. So today had two
parts: in the morning, the **imbalance prices** from Swissgrid; in the afternoon, **fixing the
open problems in the ENTSO-E data**.

---

## Part 1 (morning): imbalance prices

### The goal

A company that buys or sells electricity has to tell Swissgrid in advance how much it will
use or deliver. If reality doesn't match that plan, Swissgrid has to fix the difference with
reserve power, and the company pays (or receives) the **imbalance price** for the difference.
These prices tell me how tight or relaxed the system was in each 15-minute slot, so they are
an important input for explaining reserve power prices.

Swissgrid publishes one file per month (an XML file and an Excel copy), from January 2023 to
August 2026. For earlier years, I already have the same prices from ENTSO-E.

### What we did

1. **Looked inside all the files first.** All 44 months, not just samples. Findings:
   - The XML files write the time **with the time zone attached** (e.g. `+01:00` in winter,
     `+02:00` in summer). That means the night the clocks go back in October is never
     confusing, so I used the XML as the main source.
   - The Excel files only show the local clock time, which is ambiguous on that October
     night. I used them only as a **second copy to check against**.
   - The official file description from Swissgrid (the `.xsd` file) doesn't actually match
     the real files (it spells the tag names differently), so the script checks the files
     itself instead.
   - The order of the price columns changes from month to month, so the script reads them
     **by name, never by position**.

2. **Built the script.** `swissgrid_imbalance_prices_parse.py` reads every monthly file and
   turns it into one clean table per year, with prices converted from ct/kWh to **EUR/MWh**
   (×10). For every month it checks:
   - that there is exactly one value for every 15 minutes (no gaps, no doubles),
   - that the unit is what we expect,
   - that the Excel copy shows exactly the same numbers as the XML.

   If anything is off, that month is reported and the rest carry on.

3. **Ran it on all 44 months.** Everything came through: no errors, no empty values, the
   right number of 15-minute slots every month, and the Excel copies match the XML exactly.

4. **Compared with ENTSO-E.** For almost every month the two sources give **exactly the same
   prices**, which also confirms that the unit and currency are right.

5. **Made one long series from 2021 to 2026.** The script also builds a combined file:
   ENTSO-E for 2021–2022, Swissgrid from 2023, with a column saying where each value comes
   from.

**Result:** 15-minute imbalance prices for Switzerland, 2021–2026, clean and checked.

### What we learned

- **Switzerland changed how imbalance prices work.** Until 2025 there were **two prices**:
  one for companies that had too much electricity ("long") and one for those that had too
  little ("short"). Swissgrid started publishing a **single price** (called AEP) in July 2025,
  alongside the old two. From January 2026, only the single price exists. This is a big
  change in the rules, like the reserve market changes found earlier, and it belongs in
  Chapter 3. I still need to find Swissgrid's official announcement about it.
- **ENTSO-E has a few mistakes, and Swissgrid is the better source:**
  - On 2–5 January 2025, ENTSO-E shows one price where there should be two.
  - January 2026 is missing completely from my ENTSO-E download.
  - In May–August 2026, a small share of values differ. Swissgrid publishes its files a few
    weeks after the month ends, so its numbers are probably the corrected ones.
- **One day is missing: 31 December 2022.** In the afternoon I checked this properly (see
  Part 2). ENTSO-E simply never published that day.
- **Prices can be extreme.** The lowest price in 2026 was about −6,900 EUR/MWh. I'll need to
  handle such outliers carefully when I explore the data.

---

## Part 2 (afternoon): cleaning up the ENTSO-E data

I went through the list of open ENTSO-E problems one by one. The downloads that need the
internet ran on my Mac; the rest was done directly on the files. **Nothing was deleted.**
Every script that was changed has a backup copy (ending in `.bak_20260926`), and files I
don't want to use were moved into separate folders.

### 1. The Switzerland–Germany border capacity was missing

**What it is:** the "NTC" is the maximum amount of electricity that can be traded across a
border. It matters because it limits how much Switzerland can import or export.

**The problem:** for the German border, all four versions (day-, week-, month- and
year-ahead) were empty.

**Why:** Germany has two different codes on ENTSO-E: one for the grid company next to
Switzerland (TransnetBW) and one for the German market as a whole. Electricity flows are
published under the first code, but the border capacity only under the second. My download
used the wrong one and got "no data" every time.

**Fix:** I told the script which code to use for each type of data and downloaded the
capacity again. **Result:** complete hourly data from 2021 to 2026 in both directions. Only
the week-ahead version really doesn't exist.

**Side finding:** Switzerland → Germany was a flat 4,000 MW for years (lower in 2026), while
Germany → Switzerland is much smaller (around 1,000 MW) and dropped between 2021 and 2023.

### 2. Some Swiss reserve prices were missing on ENTSO-E

**The problem:** a few years of Swiss reserve prices were missing, and the newer daily
auctions for secondary reserve (aFRR, which started in autumn 2025) weren't there at all.

**Why:** the download script only fetched what a test run in **June 2025** had found. In June
2025 those daily auctions didn't exist yet, so they were never requested.

**Fix:** I combined the results of all three test runs, so anything found in any of them is
now downloaded. **Result:** the daily aFRR prices are there from mid-October 2025. The other
missing years are really missing on ENTSO-E.

**Important decision:** ENTSO-E has too many holes in the reserve prices. For example, the
Swiss primary-reserve (FCR) prices are missing for six months in 2021–22, and one old series
shows the same price (6.26) for almost two years, which can't be real. The **Swissgrid auction
files are complete**, so they are my **main source for reserve prices**. ENTSO-E is only
used as a cross-check. That check works: for the daily aFRR auctions, ENTSO-E and Swissgrid
agree in most time blocks.

### 3. The missing day: 31 December 2022

I asked ENTSO-E for just that day, and also looked it up myself on the ENTSO-E website: it is
**empty there too**. So the day was never published. I'll leave it empty and ask Swissgrid for
it. Bonus: I downloaded five days from the website by hand, and they match my automatic
download **exactly**. That's a nice proof that my download works correctly.

### 4. The download "logbooks" were out of date

Each download keeps a list (a "manifest") of what it saved. For two areas, the list only
showed a small part of what was actually on disk (e.g. 37 entries for 228 files). I wrote a
small tool that rebuilds the list from the actual files. I tested it on an area whose list
was already correct, and it matched perfectly.

### 5–8. Housekeeping

- **Downloads can no longer freeze.** Earlier, one download hung for 13 hours on a single
  request. Now every request gives up after 60 seconds and tries again.
- **Downloads are faster:** the waiting time between requests went from 2 seconds to
  0.3 seconds (still well below ENTSO-E's limit).
- **Unusable files moved aside:** the German and Italian bid data can never be complete (too
  much data per day for ENTSO-E's limits), so it now sits in a separate `_set_aside` folder
  with a note explaining why.
- **An old test script** was moved into a `_to_delete` folder. I can delete it whenever I
  want.

---

## What is EDA, and why am I doing it?

**EDA (Exploratory Data Analysis)** means looking at the data carefully before building any
models: charts, averages, gaps, outliers, patterns over the day, week and year, and how the
different variables move together. It tells me whether the data is trustworthy, which
variables are useful, where the market rules changed (important for research question 1b),
and what exactly I should predict. A big part of it goes straight into Chapter 3. It was
planned for weeks 7–12, so it's overdue, and it can only start once all the data sits in one
table.

---

## Where things stand

- **ENTSO-E:** the real data problems are solved or confirmed as gaps at the source. Two small
  checks remain.
- **Swissgrid:** auctions, energy overview and imbalance prices are done. A few smaller files
  and the email are still open.
- **Known gaps to mention in Chapter 3:** 31 Dec 2022 imbalance prices; January 2026 on ENTSO-E;
  some Swiss reserve prices on ENTSO-E (covered by Swissgrid).

## What comes next

1. **Email Swissgrid now** (`sdl-ausschreibung@swissgrid.ch`) about the response-time data for
   research question 3. Also ask for older imbalance prices, including 31 December 2022. This
   is overdue, and it's the only lead left for question 3.
2. **Two small ENTSO-E checks:** how many days the per-power-plant data lost around the
   clock changes, and whether the "fall-back" records are useful for question 3.
3. **Finish Swissgrid:** look at the daily secondary-control file (for question 3), read the
   2026 files (control energy, cross-border), find the announcement about the single imbalance
   price, check a gap of about 30 days in the 2024 mFRR auctions, and later the monthly
   activation files.
4. **Clean each source into the same format:** same time zone, clear column names, known gaps
   marked, unreliable series removed.
5. **Build the big table:** everything on one 15-minute timeline, plus versions per 4-hour
   block, day and week to match the auctions.
6. **Start EDA.**
7. **Talk to my supervisor** about the schedule (week 17 of 29, data work still running).
