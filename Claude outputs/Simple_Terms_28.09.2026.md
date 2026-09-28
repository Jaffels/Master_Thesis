_28 September 2026_

## The big picture

My thesis needs one big table: a row for every 15 minutes from 2021 to 2026, with a column
for everything that might explain reserve power prices. Yesterday almost all ENTSO-E checks
were finished. Today I closed the **last two open questions**, made the checks **repeatable
with one script**, and decided **when (and whether) to download data again**.

Nothing in the data or the download scripts was changed today. Everything was read-only.

---

## 1. The clock change costs almost nothing

**What it is:** twice a year the clocks change. On those days the download of the
per-power-plant data sometimes fails, and I wanted to know how much data that really costs.

**What I did:** yesterday I only checked whole days. Today I checked **hour by hour** on all
55 clock-change days from 2021 to 2026, for all five countries.

**Result:** only **7 hours are missing in total**. Switzerland loses 1 hour (26 October 2025),
Austria 1 hour, France 5 hours, Germany and Northern Italy none. On those days, the power
plants report just as completely as on normal days.

**So:** the problem is negligible. Question closed.

## 2. A few more gaps in the neighbours' data

The hour-by-hour check also found gaps that yesterday's day-by-day check missed:

- **Northern Italy:** four more days with no data at all (12 May 2022, 18 June 2025,
  26 January and 9 July 2026), plus 2 March 2025 almost empty.
- **France:** a few scattered missing hours each year from 2021 to 2024.

**Switzerland is not affected.** These gaps only matter if I use the neighbours'
per-power-plant data. I'll mark them in the cleaned data.

**Lesson:** a download can say "OK" for a year even when single hours or days are missing.
Only a detailed check finds them.

## 3. The "fall-back" records: definitely no help for question 3

**What I checked:** every field of the 831 records again, one by one.

**Result:** every record says the same thing: Swissgrid lost its connection to the European
reserve platform on a given day. There is **no information about providers, amounts or
reaction times**. So the lead for question 3 is **definitely closed**. Only the email to
Swissgrid is left.

## 4. Two traps in the fall-back records

I still want to use these records to mark the "without the European platform" periods in my
big table. Two things need care:

- **Published years later.** All records for 2022 to 2024 were only put online on
  **15 April 2025**. Since then, each day is published the next day. So before April 2025,
  nobody could have known this information in real time. In a forecasting model I can
  use it to describe **which period** we're in, but not as information available "on the day".
- **The exact times are fuzzy.** Every record starts at midnight. On days when the
  connection was lost only for part of the day, I know **how long** it lasted, but not
  exactly **when**. So I'll mark this **per day**.

I'll ask Swissgrid about both points in the email.

## 5. A script that repeats these checks

**What it is:** `check_data_gaps.py` (saved in `Entsoe/Generation/`). It repeats today's two
checks with one command and prints a short table. It **only reads** the data. With
`--write` it also saves the results as tables in a new folder `Entsoe/_checks/`.

```
python Entsoe/Generation/check_data_gaps.py generation
python Entsoe/Generation/check_data_gaps.py fallback
python Entsoe/Generation/check_data_gaps.py all --write
```

**Why it's useful:** I can re-run it after every new download, use its output to mark the
gaps in the cleaned data, and cite its numbers in Chapter 3.

**When to run it:**

- after downloading generation or outage data again;
- right before cleaning the data (with `--write`);
- right before finishing Chapter 3.

## 6. No big re-download — the data stops at 31 August 2026

**The question:** should I download all the ENTSO-E data again around 20 October?

**The answer: no.** The data up to 31 August 2026 is complete and checked. A full download
would take more than a day and wouldn't fix anything. It's better to **fix the end of my
data at 31 August 2026**, so every number in the thesis stays the same and can be
reproduced. If my supervisor and I decide to add later months, I'll do **one** full
download just before modelling.

The only exception: the **Northern Italy data after 18 July 2026**. Around **20 October** I'll
look on the ENTSO-E website whether it's there now. If yes, I download just that one file
again (the steps are in the workflow file). If not, I note it as missing at the source and
stop checking.

---

## What we learned

- **Check at the right level of detail.** Day-level checks said "fine" for Northern Italy;
  hour-level checks found four more missing days.
- **When information was published matters, not just what it says.** A record written years
  later can't be used as if it had been known at the time.
- **A fixed end date for the data** keeps the thesis reproducible and saves days of
  re-downloading.

## Where things stand

- **ENTSO-E:** all checks are done. Left: move the French bids aside, and look at the
  Northern Italy data around 20 October.
- **Swissgrid:** auctions, energy overview and imbalance prices are done. The smaller files
  and the email are still open.
- **Question 3:** still only depends on the Swissgrid email.
- **New gaps to mention in Chapter 3:** four more days in the Northern Italian power-plant
  data, and scattered hours in the French data (neighbours only).

## What comes next

1. **Email Swissgrid now** (`sdl-ausschreibung@swissgrid.ch`). It's overdue and it's the only
   lead left for question 3. Also ask for older imbalance prices (including 31 December
   2022), why Swiss secondary reserve runs without the European platform, why the old
   records were only published in April 2025, and whether exact times exist.
2. **Move the French bids aside** and check that the download plan only shows Swiss bids.
3. **Talk to my supervisor** about the schedule and about fixing the data end at
   31 August 2026.
4. **Finish Swissgrid:** the daily secondary-control file, the 2026 files, the announcement
   about the single imbalance price, the gap in the 2024 mFRR auctions, and later the
   monthly activation files.
5. **Around 20 October:** check the Northern Italy data on the ENTSO-E website.
6. **Run the gap script with `--write`**, then **clean each source into the same format**,
   **build the big table** and **start EDA**.
