_29 September 2026_

## The big picture

My Swiss reserve prices don't live in isolation: Switzerland buys its primary reserve (FCR)
together with Germany and other countries, and German secondary and tertiary reserve (aFRR,
mFRR) prices are the obvious benchmark next door. ENTSO-E has almost no German reserve data
before 2021, so today I got it from the source instead: **regelleistung.net**, the German
grid operators' joint website for reserve auctions.

Result: **German (and FCR-cooperation) auction prices for every 4-hour block from July 2018
to August 2026, with 0 errors.**

---

## 1. Where the data comes from

**What it is:** regelleistung.net has a "data center" page with a download button for every
day's auction results. Behind that button is a simple web address that returns an Excel file
for one day and one product. No account or key is needed.

**What I did:** two scripts in a new folder `Regelleistung/`, built the same way as the
ENTSO-E ones:

- **Test script** (`regelleistung_probe.py`): asks for two sample days per year and prints
  which years have data and which columns the files contain.
- **Download script** (`regelleistung_pull.py`): downloads every day, keeps each original Excel
  file, and then stacks them into one clean table per product and year.

My Claude workspace can't reach the website, so everything ran on my Mac.

## 2. The test: the history starts later than hoped

The test showed the data center only has the **daily auctions**:

- **aFRR and mFRR from 12 July 2018** (the day Germany switched to daily 4-hour auctions).
- **FCR from 1 July 2019** (one product per day until June 2020, then six 4-hour blocks).

The older weekly German auctions (2015 – mid-2018) **are not there**. So the download starts in
July 2018 instead of 2015.

## 3. The download: complete, 0 errors

- **8,952 requests** in about **75 minutes**: 8,565 with data, 387 empty, **0 errors**.
- **All 387 empty days are before the market started** (365 days of FCR before July 2019,
  11 days of aFRR and mFRR before 12 July 2018). After those dates, **not a single day is
  missing**.
- The script worked out the exact start and end time of every block (e.g. "POS_00_04" =
  positive reserve, midnight to 4 am) for **100 % of the rows**.
- The start looked worrying (hundreds of "empty" in a row), but that was just the year before
  FCR daily auctions existed — exactly what the test had predicted.

## 4. A bonus: the Swiss FCR price is in there too

The FCR files list **every country in the joint auction, including Switzerland**. So I have a
second, independent source for the Swiss FCR price for every block since July 2019 (for
example 15 July 2019: Switzerland 167.04, Germany 217.66 EUR/MW). I can use it to double-check
my Swissgrid auction data.

## 5. Three traps to handle when cleaning

1. **Sometimes there is a second auction.** On a few days per year (26 days in 2024, a handful
   in the other years) FCR has two auctions for the same blocks. The second one has strange
   prices: sometimes 0, once 3,382 EUR/MW (28–29 October 2025), sometimes empty. For the
   normal price I use **only the first auction** and just mark the days with a second one.
2. **The price unit changes on 8 December 2021.** Before that date, aFRR/mFRR prices are per MW
   for the **whole 4-hour block**; from that date they are per MW **per hour**. To compare
   them I have to **divide the old prices by 4**. The script keeps them in separate columns,
   so nothing gets mixed by mistake.
3. **The column names change on 7 September 2022.** FCR columns go from short codes
   ("DE", "CH") to full names ("GERMANY", "SWITZERLAND"). At the same time a column called
   "import/export" becomes "deficit/surplus" — I still need to check whether plus and minus
   mean the same thing before and after.

---

## What we learned

- **Test first, then download.** The test showed in two minutes that there is nothing before
  2018, which saved a useless download and explained the "empty" days right away.
- **"Empty" isn't always bad.** Empty days before a market exists are expected; what matters
  is that there are no gaps after the start.
- **Units and names change quietly.** The same column can mean "per block" one day and "per
  hour" the next. Keeping the original names in the table makes these changes visible.
- **Other countries' data can check my own.** The German website gives me a second copy of the
  Swiss FCR price.

## Where things stand

- **ENTSO-E (2015–2026):** downloaded and checked. Left: move the French bids aside, and look
  at the Northern Italy data around 20 October.
- **Swissgrid:** auctions, energy overview and imbalance prices are done. The smaller files
  and the email are still open.
- **regelleistung.net (new):** German/cooperation auction prices July 2018 – August 2026,
  complete, 0 errors.
- **Question 3:** still only depends on the Swissgrid email.

## What comes next

1. **Email Swissgrid now** (`sdl-ausschreibung@swissgrid.ch`) — still overdue and the only
   lead for question 3 (see the list of questions in the workflow file).
2. **Check the plus/minus meaning** of the FCR import/export vs deficit/surplus columns.
3. **Compare the Swiss FCR price** from regelleistung.net with the Swissgrid auction data,
   block by block.
4. **Move the French bids aside** and check that the download plan only shows Swiss bids.
5. **Decide** on the schedule, fixing the data end at 31 August 2026, and whether the sample
   starts in 2015 or 2021.
6. **Finish Swissgrid:** the daily secondary-control file, the 2026 files, the single
   imbalance price announcement, the 2024 mFRR gap, and later the monthly activation files.
7. **Around 20 October:** check the Northern Italy data on the ENTSO-E website.
8. **Extend the gap script to 2015–2020**, then **clean each source into the same format,
   build the big table** and **start EDA**.
