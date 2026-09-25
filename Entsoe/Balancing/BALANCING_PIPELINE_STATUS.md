# ENTSO-E Outages Data Pipeline — Status

_Last updated: 2026-09-24 (Swissgrid auction ingestion added — §6, §8, §10)_

Status of the ENTSO-E **Outages-domain** acquisition layer for the thesis
(Swiss ancillary-service price forecasting). Scope: IFs IN 7.2 (fall-backs,
imbalance netting), IF mFRR 3.11, IF aFRR 3.10, TR 10.1.A&B (transmission
infrastructure unavailability), TR 10.1.C (offshore grid), and 15.1.A&B&C&D
(generation and production unit unavailability). Sibling to the Load,
Generation, Balancing and Transmission pipelines; same two-script pattern.

---

## 1. Summary

The ENTSO-E Outages acquisition layer is **built and complete.** Two scripts,
structurally identical to the Load/Generation/Transmission pairs:

- `entsoe_outages_probe.py` — probes which `(dataset, variant, area)` series
  return data and on which area code; owns all shared request machinery,
  throttling, pagination/overflow logic, and the generic lxml parser.
- `entsoe_outages_pull.py` — multi-year production pull
  (**2021-01 → 2026-09**), driven by the probe's coverage report, importing
  the probe as a library.

Seven article groups, iterated over the relevant area/border grids. Final pull
result: **335 ok, 227 empty, 0 error** across 594 series-years
(99 series × 6 years).

**Outages are structurally different from Load/Generation/Transmission:**

1. **Event documents, not time series.** Each outage is a document with its own
   `doc_mrid`, `revision`, and `unavail_start/end` window. The output schema is
   one row per availability Point — a wide, event-keyed frame, not a regular
   DatetimeIndex grid.
2. **Overlap selection.** The API returns documents whose unavailability period
   *overlaps* the query window, so a long outage appears in multiple year files.
   Deduplication on `(doc_mrid, revision)` is required downstream before any
   analysis.
3. **entsoe-py's outage parsers are not used.** The library's parsers crash
   (KeyError) on unknown EIC codes, drop `Reason.code/text` and transmission
   asset lists, and silently stop at offset 4800. All Outages methods in the
   probe use a custom lxml-based parser that handles every document type (A77,
   A78, A79, A80, A53 fall-backs) from a single code path.
4. **Fall-backs paginate at 100 docs/page, not 200.** Per the API specification
   for the A53 fall-back endpoint; the probe exposes this as `PAGE_SIZE_FALLBACK`.

---

## 2. Components

| File | Role |
|------|------|
| `entsoe_outages_probe.py` | Coverage probe + all shared request machinery (throttling, retry, offset pagination, window halving, lxml parser, `tidy`/`profile`, key loading) + five client method wrappers. |
| `entsoe_outages_pull.py` | Production pull. Imports the probe as a library. One parquet per `(dataset, variant, area, year)`, resumable, with a `_pull_manifest.csv`. |

> As with the other pipelines, `pull.py` contains no fetch or parse logic —
> it calls `probe.invoke(...)`, `probe.tidy(...)`, etc. Every request-layer
> change lives in `probe.py`.

### Request-layer specifics (all in `probe.py`)

- **Five client wrappers** added to `EntsoePandasClient` at import time:
  `query_outages_generation_units` (A80), `query_outages_production_units` (A77),
  `query_outages_transmission` (A78), `query_outages_offshore_grid` (A79),
  `query_outages_fallbacks` (A53). None of these are native entsoe-py methods.
- **Monthly windowing → offset pagination → overflow halving.** Each wrapper
  calls `fetch_outage_documents`, which iterates monthly sub-windows, pages
  through offsets until the page is short, and halves any window whose document
  count hits the API ceiling. Window halving recurses to a 6-hour floor.
- **Fall-backs** (A53) require both `processType` and `businessType` as
  mandatory parameters. Four `businessType` variants are fetched per process
  type: `C47` (disconnection), `A53` (planned), `A54` (unplanned), `A83`
  (auction cancellation).
- **Sparse/event-driven series** (transmission, offshore, fall-backs) are pulled
  for their full grid regardless of probe status — one quiet probe month is not
  a verdict for rare events.
- **Dense series** (generation / production unit outages) are pulled for
  probe-status-ok areas only (same rule as Load/Generation).

---

## 3. What the pipeline provides

### 3.1 Articles covered

| Article | Description | docType | Area grid | Notes |
|---------|-------------|---------|-----------|-------|
| 15.1.A | Planned unavailability of generation units | A80 / bt A53 | CH, DE_LU, FR, IT_NORD, AT | ≥100 MW units |
| 15.1.B | Forced unavailability of generation units | A80 / bt A54 | same | |
| 15.1.C | Planned unavailability of production units | A77 / bt A53 | same | |
| 15.1.D | Forced unavailability of production units | A77 / bt A54 | same | |
| 10.1.A | Planned unavailability in transmission grid | A78 / bt A53 | CH↔DE/FR/IT_NORD/AT + CH internal | per directed border |
| 10.1.B | Forced unavailability in transmission grid | A78 / bt A54 | same | |
| 10.1.C | Unavailability of offshore grid infrastructure | A79 | DE_LU, DE_TENNET, DE_50HZ, DE_AMPRION | DE only, per user instruction |
| IF aFRR 3.10 | aFRR fall-backs | A53 / pt A51 | CH, DE, FR, IT, AT (CTA level) | 4 businessType variants each |
| IF mFRR 3.11 | mFRR fall-backs | A53 / pt A47 | same | 4 businessType variants each |
| IFs IN 7.2 | Imbalance netting fall-backs | A53 / pt A63 | same | 4 businessType variants each |

### 3.2 Output schema

One parquet per `(dataset, variant, area, year)`. Each file is a flat,
event-keyed DataFrame (RangeIndex), one row per availability Point, with
columns:

| Column group | Columns |
|---|---|
| Core identity | `doc_mrid`, `revision`, `created`, `docstatus` |
| Outage window | `businesstype`, `unavail_start`, `unavail_end` |
| Time point | `period`, `start`, `end`, `resolution`, `position`, `quantity` |
| Document fields | `doc.*` — reason code/text, document type, etc. |
| TimeSeries fields | `ts.*` — biddingzone EIC, production resource EIC/name/psr/nominalP, asset EICs for transmission |

String columns (EICs, mRIDs, names) remain strings; `revision`/`position` are
`Int64`; `quantity` and `nominalP` are float; datetime columns are
`datetime64[us, Europe/Zurich]`.

### 3.3 Coverage summary

| Dataset | Variant | CH | DE_LU | FR | IT_NORD | AT |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `gen_unit_outages` | planned_A53 | ok | ok | ok | ok | ok |
| `gen_unit_outages` | forced_A54 | ok | ok | ok | ok | ok |
| `prod_unit_outages` | planned_A53 | ok | ok | ok | ok | ok |
| `prod_unit_outages` | forced_A54 | ok | ok | ok | ok | ok |

Transmission (10.1.A&B), offshore (10.1.C), and fall-backs (3.10/3.11/7.2) are
event-driven and checked via the manifest rather than reported per-area above.
Many fall-back `(variant, area, year)` combinations are genuinely empty — the
API returns "No matching data" for most businessType/processType/area combos in
most years. Non-empty fall-back years are recorded as ok in the manifest.

**Overall pull result: 335 ok, 227 empty, 0 error.**

---

## 4. Performance notes

- **DE_LU generation/production unit outages are the densest series** (~18,000
  documents/year; ~18–20 minutes per year at a 2-second request interval). The
  pull triggers repeated window-halving for April and September in most years.
  Total wall-clock time for the full pull was approximately 9 hours.
- **Fall-back series (360 series-years across 60 series)** are almost entirely
  empty. Most of the 227 empty series-years are fall-back combinations.
- **One transient 599 (gateway connect timeout)** on `fallback_mfrr` /
  `unplanned_A54` / DE_LU / 2023 at 20:15 during the initial run. Nothing was
  written; the manifest recorded it as error. Resolved cleanly on the next run
  (09:00 the following morning) with 0 errors.

---

## 5. Known issues

None open. One issue was encountered and resolved during the pull:

**O-1 — Transient 599 gateway timeout (ENTSO-E infrastructure).**  
`uu-gateway-router/connectTimeout` on one fall-back request. The probe's
`call_with_retry` correctly identified it as transient (HTTP 5xx) and would
have retried, but the 3-attempt ceiling was exhausted before the gateway
recovered. Recorded as error (no file written); retried and resolved cleanly on
the next run. No code change required.

---

## 6. Relationship to the wider ENTSO-E pipeline

| Domain | Status doc | Status |
|--------|-----------|--------|
| **Load** (6.1.A/B/C/D/E, 8.1) | `LOAD_PIPELINE_STATUS.md` | Built + complete |
| **Balancing** | `PIPELINE_STATUS.md` | ~97% pulled |
| **Generation** (14.1.A/B/C/D, 16.1.A/B&C/D) | `GENERATION_PIPELINE_STATUS.md` | Built + complete |
| **Transmission** (11.1, 12.1.F/G, 13.1.B/C) | `TRANSMISSION_PIPELINE_STATUS.md` | Built + complete (56 ok / 13 empty / 0 error) |
| **Outages** (IFs IN 7.2, IF mFRR 3.11, IF aFRR 3.10, TR 10.1.A/B/C, 15.1.A/B/C/D) | **This document** | **Built + complete (335 ok / 227 empty / 0 error)** |

### Non-ENTSO-E tracks

| Track | Status | Priority |
|-------|--------|----------|
| **Swissgrid — capacity auctions** (FCR/aFRR/mFRR, 2015–2027) | **Built + complete** (5,366,493 bids / 69,421 auction-blocks, 0 unparsed) — see §10 | Done |
| **Swissgrid — Energy Overview** (15-min system data) | Not built — files profiled, copies to clean first | **High — next task** |
| **Swissgrid — imbalance prices** (2023-01 → 2026-08) | Downloaded, not parsed | Medium |
| **Swissgrid — control energy / cross-border / control area balance** (2026 only) | Downloaded, not parsed | Medium |
| **Swissgrid — TRE mFRR energy bids** (2023-01 → 2026-09, ~3.5 GB) | Downloaded, not parsed | Low |
| **RQ3** response-time data | Still unsourced. Only lead: Swissgrid's second-by-second aFRR file (system-level, current day only) | **High / at-risk** |
| **JAO** | NTC partially covered by TR 11.1; dedicated JAO source not addressed | Medium |
| **EDA** (Phase 3) | Blocked on Energy Overview + integration | — |

## 7. Thesis relevance

- **RQ1 (price drivers):** Planned unavailability of generation and production
  units (15.1.A&B&C&D) is one of the most informative exogenous regressors
  identified in the literature — Kraft et al. (2020) explicitly select
  `planned unavailable capacity` as a key predictor for FCR prices. The
  DE_LU series is particularly relevant given Switzerland's integration into
  the joint FCR Cooperation (Austrian, Belgian, French, German, Dutch, and
  Swiss TSOs).
- **RQ1b (structural breaks):** Transmission outage volumes (10.1.A&B) provide
  a secondary indicator of grid stress and redispatch activity around the
  2019–2020 reform period.
- **Fall-backs (IF aFRR 3.10, IF mFRR 3.11, IFs IN 7.2):** Directly relevant
  to RQ3 (response-time characterisation) if any non-empty data exists. The
  high rate of empty combinations suggests this source may not provide the
  activation-level granularity needed for RQ3 — Swissgrid's own activation
  records remain the primary target.
- **Offshore grid (10.1.C):** Captured for context only; not a direct driver of
  Swiss ancillary service prices. Relevant only insofar as German offshore
  outages affect DE grid conditions and thereby German spot/reserve prices.

---

## 8. Next steps

### Outages (unchanged)
1. **Inspect the manifest** for the fall-back ok rows — which
   `(processType, businessType, area, year)` combinations actually have data,
   and whether the volumes are meaningful for RQ3.
   ```bash
   python -c "
   import pandas as pd
   m = pd.read_csv('Entsoe/Outages/Data/production/_pull_manifest.csv')
   ok = m[m.status=='ok'][['dataset','variant','area','year','rows','docs']]
   print(ok[ok.dataset.str.startswith('fallback')].sort_values('rows', ascending=False))
   "
   ```
2. **Deduplicate** on `(doc_mrid, revision)` before any cross-year analysis.
3. **Feature engineering** (downstream): planned unavailable capacity (MW) per
   area per hour from 15.1.A+C; net transmission outage capacity per border from 10.1.A+B.

### Swissgrid (in priority order)
1. **Clean the Energy Overview copies.** The four folders (`Balancing/`,
   `Energy_in_the_grid/`, `Production_and_consumption/`, `Transmission/`) hold the
   same `EnergieUebersichtCH-YYYY` files but are not byte-identical. Keep `Balancing/`,
   delete the other three, re-download 2026 into `Balancing/`.
2. **Build the Energy Overview parser** (`Swissgrid/EnergyOverview/`). Must handle:
   - `.xls` (2009–2019) and `.xlsx` (2020–2026); needs `xlrd` + `openpyxl`.
   - Sheet change: 2025 adds `Datetime`, drops `Zeitreihen1h00`.
   - **Timestamp convention change:** 2019/2020 label the interval *end*
     (first value 00:15, datetime cells); 2025 labels the interval *start*
     (first value 00:00, `dd.mm.yyyy HH:MM` text). Normalise to interval start.
   - Wide layout (variables as rows, timestamps as columns) → transpose.
   - Units: energy in kWh per 15 min; control-energy prices in EUR/MWh.
   - DST: 92/100 quarter-hours on switch days.
   - Contains no imbalance prices (those come from the separate files).
3. **Build the imbalance-price parser.** Monthly XML (use the official XSD) or XLSX,
   2023-01 → 2026-08, BG-long / BG-short in **ct/kWh** (×10 → EUR/MWh).
   File name changes from `...balance-energy...` to `...imbalance-energy...` in 2026.
   Pre-2023: use ENTSO-E imbalance prices (Balancing pipeline).
4. **Parse the 2026-only CSVs** (`Ausgleichsenergie-und-Regelenergie`, `Grenzfluesse`,
   `control-area-balance`). Semicolon-separated, `dd.mm.yyyy HH:MM`, 15-min.
   Cost columns I/J/K and R/S/T are **cumulative weekly totals in kEUR** — difference
   them before use. Intraday NTC in `Grenzfluesse` is not available from ENTSO-E.
5. **Inspect `secondary-daily_2026-09-24.csv`** for RQ3. If useful, set up daily
   collection (cron/launchd — deferred for now).
6. **Email `sdl-ausschreibung@swissgrid.ch`**: historical second-by-second aFRR
   archive; anonymised provider-level activation / response-time data for research;
   pre-2023 imbalance prices and pre-2026 control-energy / cross-border files.
7. **TRE parser** (lower priority): latin-1, ~100 MB/month, chunked read,
   partition by month.
8. **Update the thesis scope** (RQ1b): structural-break data now exists — see §10.4.
9. **Integrate** all sources into the EDA layer (Phase 3).

## 9. File / output layout

```
Entsoe/Outages/
├── entsoe_outages_probe.py         # probe + shared request machinery + 5 client wrappers
├── entsoe_outages_pull.py          # production pull (imports probe)
└── Data/
    ├── _coverage_<range>.csv                              # probe verdicts
    ├── _probe_state.json                                  # resumable probe state
    ├── <series>__<variant>__<area>__<range>.parquet       # probe samples
    └── production/
        ├── _pull_manifest.csv                             # rows / docs / span / file per series-year
        └── <dataset>/<variant>/<area>/
            ├── <year>.parquet                             # ok years
            ├── <year>.empty                               # genuinely empty year (skip on re-pull)
            └── _empty.parquet                             # series empty across all years
```

---

## 10. Swissgrid ingestion — work done 2026-09-24

### 10.1 Scripts

| Script | Location | Purpose | Output |
|--------|----------|---------|--------|
| `inventory_manual_download.py` | thesis root | Lists every file in `Manual_Download` with size and first rows | `manual_download_inventory.txt` |
| `profile_swissgrid.py` | thesis root | Auction ID patterns / units / pricing per product-year; Energy Overview sheet structure; imbalance-price file layout | `swissgrid_profile.txt` |
| `swissgrid_auctions_parse.py` | `Swissgrid/Auctions/` | Production parser for all capacity auction files | see 10.2 |

Run: `python Swissgrid/Auctions/swissgrid_auctions_parse.py` from the thesis root
(needs `pandas`, `pyarrow`). Reads everything under `Swissgrid/Manual_Download/Tenders/`.

### 10.2 Auction parser output

```
Swissgrid/Auctions/Data/
├── bids/<FCR|aFRR|mFRR>/<delivery_year>.parquet   # one row per published bid
├── auctions.parquet                                # one row per auction × direction × delivery block
└── _parse_manifest.csv                             # per source file: rows, drops, checks
```

- **Bids:** `product`, `direction` (up/down/sym), `auction_id`, `tender_series`,
  `revision`, `delivery_start/end` (`datetime64[us, Europe/Zurich]`), `duration_h`,
  `country`, `currency`, `offered_mw`, `awarded_mw`, `accepted`, `capacity_price`,
  `cost`, `bid_price_mwh`, `settle_price_mwh`, `divisible`, `flag`, `norm_ok`.
- **Auctions:** `n_bids`, `n_accepted`, `offered_mw`, `awarded_mw`, `awarded_mw_ch`,
  `bid_min`, `bid_max` (marginal accepted bid), `bid_vwap`, `settle_max`,
  `settle_ch` (FCR: CH clearing price), `n_settle_prices`, `cost`.

**Final run:** 5,366,493 bids, 69,421 auction-blocks, 0 unparsed IDs, 0 conflicting
boundary copies, `norm_ok_share` = 1.0 for every file.

### 10.3 Parsing decisions

- **Products:** PRL = FCR (EUR), SRL = aFRR (CHF), TRL = mFRR (CHF). Currency kept native.
- **Published bids:** FCR and aFRR files contain accepted bids only; mFRR files also
  contain rejected bids (full bid curve for mFRR only).
- **Prices:** all normalised to per MW per hour. Pre-2019 `Preis` is the bid
  (pay-as-bid); from 2019 `Angebotspreis` = bid, `Preis` = settlement price.
  Check: `capacity_price / duration_h == bid_price_mwh` holds in 100 % of rows.
- **Delivery windows:** `KWnn` = ISO week (Mon–Sun), `YY_MM_DD` = day,
  `HH:MM bis HH:MM` = 4 h block. DST days give 3 h/5 h blocks and 167 h/169 h weeks.
- **Direction:** from the ID (`TRL+`/`TRL-`) or description (`SRL+/-`, `UP`/`DOWN`);
  aFRR before 2018 and all FCR = `sym`.
- **Year-boundary duplicates:** same auction in two year files (e.g. `PRL_15_KW53`).
  Kept the copy with most rows (all copies turned out identical).
- **Advance-procurement file** (`20230906_regelleistung_2024_vorgezogene_beschaffung_ergebnis.csv`)
  is fully contained in the 2024 file → contributes 0 rows.
- **Revised auctions:** `TRL+_18_03_20-Neu` (70 bids) has no original → normal auction.
  `TRL+_19_KW42_KORR` (1 bid) corrects one bid of `TRL+_19_KW42` (36 bids) → aggregated
  into the original. Small risk of 5 MW double-count in that week.
- **Data glitch:** one 2015 mFRR row has price unit `CH` instead of `CHF/MWh*` (flagged).

### 10.4 Findings relevant to the thesis

- **RQ1b is no longer at risk.** Auction data covers 2015–2026, so all reforms are observable:
  - FCR: weekly → daily (Jul 2019) → 4 h blocks (2020); pay-as-bid → marginal (Jul 2019).
  - aFRR: symmetric → split up/down (2018); daily 4 h blocks added 30 Sep 2025
    (weekly and daily coexist in 2026).
  - mFRR: `TRL+`/`TRL-` merged into `TRL` from week 40 2025.
  The preliminary study lists only the 2019–2020 FCR reforms; the aFRR/mFRR breaks are new.
- **FCR files include all cooperation countries** (AT, BE, CZ, DE, DK, FR, NL, SI, CH).
  Use `settle_ch` / `awarded_mw_ch` for Swiss-specific series.
- **RQ3:** no response-time data in any downloaded file. TRE shows only whether a bid
  was activated per 15 min.

### 10.5 Files downloaded 2026-09-24

- Auction results 2026 and 2027 (`2026-PRL-SRL-TRL-Ergebnis.csv`, `2027-PRL-SRL-TRL-Ergebnis.csv`)
- TRE 06.2026 (republished), 08.2026 (complete), 09.2026 (to date)
- Imbalance prices 2026-04 → 2026-08 (XML/XLSX) + `swissgrid-prices-for-balance-energy.xsd`
- Refreshed `Ausgleichsenergie-und-Regelenergie-2026.csv`, `Grenzfluesse-2026.csv`;
  new `control-area-balance-2026.csv`
- `Swissgrid/Manual_Download/Secondary_control_energy/secondary-daily_2026-09-24.csv`

Not available online: pre-2023 imbalance prices; pre-2026 control-energy and
cross-border CSVs (request from `sdl-ausschreibung@swissgrid.ch`).
