Sep 26, 2026 · @Johan

Data acquisition layer for the Master Thesis (Swiss ancillary-service price forecasting). All ENTSO-E pipelines cover 2021-01 to 2026-09; Swissgrid auctions cover 2015-2027.

## Pipeline status snapshot

| Domain | Scripts | Coverage | Pull result | Last updated |
| --- | --- | --- | --- | --- |
| **Balancing** (17.1.B/C/F/G/H, 12.3.E/F) | probe + pull | CH complete; DE/FR/IT context | 204 files in production (IT/DE bids set aside); CH aFRR daily (Oct 2025→) and FCR weekly (2021–22) added; manifest rebuilt from disk | 2026-09-26 |
| **Generation** (14.1.A/B/C/D, 16.1.A/B&C/D) | probe + pull | CH + 4 neighbours | Complete; 1–2 DST days lost on per-unit (negligible) | 2026-09-22 |
| **Load** (6.1.A/B/C/D/E, 8.1) | probe + pull | CH + 4 neighbours | Complete; CH longer horizons start \~2023 (genuine) | 2026-09-21 |
| **Transmission** (11.1, 12.1.F/G, 13.1.B/C) | probe + pull | 8 directed CH borders | CH↔DE NTC re-pulled on `DE_LU` (day/month/year-ahead ok; week-ahead genuinely empty); border codes pinned | 2026-09-26 |
| **Outages** (15.1.A–D, 10.1.A–C, IF fall-backs) | probe + pull | CH + 4 neighbours | 335 ok / 227 empty / 0 error | 2026-09-24 |
| **Swissgrid — Auctions** (FCR/aFRR/mFRR 2015–2027) | swissgrid\_auctions\_parse.py | 2015–2027 all products | 5,366,493 bids / 69,421 auction-blocks / 0 unparsed | 2026-09-24 |
| **Swissgrid — Energy Overview** | swissgrid\_energy\_overview\_parse.py | 2009–2026 complete | 18/18 files ok, 0 errors, 0 NaN; fixed 64-column schema | 2026-09-26 |
| **Swissgrid — Imbalance prices** | swissgrid\_imbalance\_prices\_parse.py | 2023-01 → 2026-08 (+ ENTSO-E 2021–2022 in combined file) | 44/44 months ok, 0 errors, 0 NaN; XLSX = XML in all 44; ENTSO-E match ≥99% except 2–5 Jan 2025; Jan 2026 absent in ENTSO-E; 31 Dec 2022 confirmed missing at source | 2026-09-26 |
| **Swissgrid — TRE / CSVs** | not built | 2023-01 → 2026-09 (downloaded) | Not parsed | — |
| **Combined dataset (ENTSO-E + Swissgrid)** | not built | 2021-01 → 2026-08 | Planned: clean layer → 15-min master → modelling views | — |
| **RQ3 response-time data** | — | No confirmed source | Unsourced — at risk | — |
| **JAO (NTC)** | — | Partially covered by TR 11.1 | Not addressed | — |

---

## ENTSO-E Balancing pipeline

Two scripts: `entsoe_balancing_probe.py` (coverage probe + all shared request machinery) and `entsoe_balancing_pull.py` (production pull, imports probe as library). The pull ran \~19 h. **216 / 222 series-years written**; the 6 missing are `aggregated_bids A47_mFRR FR` (2021–2026) — non-core — stalled on a hung network request. Switzerland is completely and cleanly covered.

**Update 2026-09-26:** FR bids 2021–2026 are now on disk; IT/DE bids moved to `Data/_set_aside/` (D6); CH aFRR daily and CH FCR weekly contracted reserves added via `_coverage_union.csv`; manifest rebuilt from disk (234 entries). See *ENTSO-E data-integrity round* below.

### What's on disk

| Series | ENTSO-E | Status | Notes |
| --- | --- | --- | --- |
| Imbalance prices | 17.1.G | OK | 15-min |
| Imbalance volumes | 17.1.H | OK | 15-min |
| Activated balancing energy prices | 17.1.F | OK | Long format (duplicate timestamps legitimate) |
| Aggregated bids — aFRR / mFRR | 12.3.E | OK (CH, FR) | CH bids sparse → no cap issues; IT/DE set aside 2026-09-26 (D6) |
| Contracted reserve price + volume — FCR | 17.1.B&C | OK | Daily; + weekly pre-reform |
| Contracted reserve price + volume — aFRR | 17.1.B&C | OK | Weekly (A02); daily (A01) from 14 Oct 2025 (added 2026-09-26) |
| Contracted reserve price + volume — mFRR | 17.1.B&C | Gaps | Daily + weekly; daily prices 2023–24 and weekly 2024 empty at source |
| Contracted reserve price + volume — FCR weekly | 17.1.B&C | Do not use | Price constant 6.26 Jan 2021–Sep 2022 → stale legacy series |
| Procured balancing capacity — FCR / mFRR | 12.3.F | Unstable | Varies across years (D8); rely on contracted reserves instead |

### Reform structure confirmed in data

CH FCR was daily + weekly in 2021 → daily-only by 2025. CH mFRR volume/price split across agreement types in 2021, consolidated by 2025. FCR daily/weekly overlap during reform must be reconciled at cleaning time, not concatenated. The raw pull keeps every (product × agreement-type) as a separate series.

### Key data issues

- **D2 — Area level mixed:** `imbalance_volumes DE → DE_AMPRION` = Amprion control area only (\~¼ of Germany). `IT_NORD` series = Italian North bidding zone, not Italy. DE contracted reserve prices are fine (uniform procurement).
- **D5 — Forward-fill artefact:** Weekly/daily capacity blocks are ffilled to 15-min — these are step functions, not 15-min observations. Resample accordingly.
- **D6 — Dense foreign aggregated bids (IT, DE) irreducibly incomplete:** Single day exceeds the 100-TimeSeries cap. Recommend dropping IT/DE bids, keeping CH. **Done 2026-09-26** (`EXCLUDE_SERIES` in pull; files in `Data/_set_aside/`).
- **D8 — `procured_balancing_capacity` (12.3.F) unstable across years.** CH FCR/mFRR present in 2021, empty in 2025.

### Key pipeline issues

- **P4 — No request timeout (FIXED 2026-09-26 — 60 s `REQUEST_TIMEOUT_S` in all 5 pipelines):** `entsoe-py` issues HTTP requests with no timeout. A stalled connection hangs forever. This is what hung the run at \[217/222\] for \~13 h. Fix: wrap requests with a hard \~60 s socket timeout.
- **P5 — Throttle interval too conservative (FIXED 2026-09-26 — default `--interval 0.3` in all 5 pulls):** At `--interval 2.0` the pull ran \~19 h. Lever: `--interval 0.3` (\~200 req/min, half the 400/min ceiling) cuts runtime dramatically.
- **P6 — Rate limiting is NOT the cause of errors:** ENTSO-E's limit is 400 req/min; exceeding it returns HTTP 429 + a 10-min ban. We never hit 429. The 400/599 errors are data-volume/server issues.

### Next steps

1. ~~Accept 216/222 and add timeout fix (P4)~~ — **DONE 2026-09-26.**
2. ~~Verify via `_pull_manifest.csv`~~ — **DONE 2026-09-26** (manifest rebuilt from disk; CH covers 2021-01 → 2026-08 except the gaps listed above).
3. ~~Drop IT/DE from the bids pull; `--interval 0.3` default~~ — **DONE 2026-09-26.**
4. German benchmark capacity prices → **regelleistung.net** (ENTSO-E DE coverage late/patchy).

---

## ENTSO-E Generation pipeline

Two scripts: `entsoe_generation_probe.py` and `entsoe_generation_pull.py`. Eight article-variant combinations across five areas (CH + DE\_LU / FR / IT\_NORD / AT), up to 228 series-years. Pull is complete. No monkeypatching needed — unlike Load and Balancing, every generation article has a native `entsoe-py` method.

`query_generation_per_plant` (16.1.A) is `@day_limited` — entsoe-py fans out to \~365 daily sub-requests per year. The pull implements its own **per-day fallback loop** that catches both 400 HTTP errors and parser errors per day, losing only the bad day(s) instead of the whole year.

### Coverage

| Article | Description | CH | DE\_LU | FR | IT\_NORD | AT |
| --- | --- | --- | --- | --- | --- | --- |
| 14.1.A | Installed capacity per type | OK | OK | OK | OK | OK |
| 14.1.B | Installed capacity per unit | varies by TSO | varies | varies | varies | varies |
| 14.1.C | Day-ahead aggregated generation forecast | OK | OK | OK | OK | OK |
| 14.1.D | Wind & solar forecast (day-ahead) | OK | OK | OK | OK | OK |
| 14.1.D | Wind & solar forecast (intraday) | sparse | OK | OK | sparse | sparse |
| 16.1.A | Actual generation per unit | OK (1–2 DST days lost/yr) | OK | OK (parser fallback) | OK | OK |
| 16.1.B&C | Actual generation per type | OK | OK | OK | OK | OK |
| 16.1.D | Water reservoirs & hydro storage | OK | OK | OK | OK | OK |

All-NaN fuel columns (e.g. offshore wind for CH) are preserved in parquet — do not drop them; the schema must be stable across areas.

### Key issues

- **D-G1 — DST-transition 400s on 16.1.A:** entsoe-py constructs malformed period parameters on CET fall-back days. Per-day fallback catches these; 1–2 days lost per year. No fix without patching entsoe-py.
- **D-G2 — entsoe-py parser crash on FR 16.1.A:** Some FR generation units lack the `<name>` XML tag → `AttributeError`. Per-day fallback recovers the vast majority of days.
- **P-G1 — 16.1.A is slow:** A single area-year takes several minutes. Use `--only per_unit` to run separately.
- **P-G2 — No request timeout (shared with Balancing P4 and Load P-L2).** FIXED 2026-09-26.

### Thesis relevance

- **RQ1:** VRES generation (16.1.B&C, 14.1.D) and hydro reservoir filling rate (16.1.D) are key exogenous drivers. The water reservoir series captures Switzerland's dual role as energy producer and reserve provider.
- **RQ2:** Day-ahead generation forecast (14.1.C), wind & solar forecast (14.1.D), and installed capacity (14.1.A) are candidate forecasting features.

### Next steps

1. ~~Verify pull via `_pull_manifest.csv`~~ — **DONE 2026-09-26**: manifest was stale (37 entries); rebuilt from disk → all 228 files listed.
2. Check per-day fallback results for 16.1.A: confirm skipped days are negligible.
3. Feature engineering (downstream): VRES share = wind+solar / total generation; hydro availability = reservoir filling rate.

---

## ENTSO-E Load pipeline

Two scripts: `entsoe_load_probe.py` and `entsoe_load_pull.py`. Six articles across five areas (CH + DE\_LU / FR / IT\_NORD / AT), 150 series-years over the range. Pull is built and validated.

Load is materially simpler than Balancing: endpoints are cap-free and `@month_limited` — no 100-TimeSeries cap, no adaptive chunking, no offset-pagination. Every series answered on its clean national/bidding-zone code (no control-area slices), so nothing needs area-pinning. Two custom additions in `probe.py`: a permissive forecast parser (keeps non-A60/A61 TimeSeries that the stock parser silently drops) and an 8.1 margin wrapper (no native entsoe-py method).

### Coverage

| Series | CH | DE\_LU | FR | IT\_NORD | AT |
| --- | --- | --- | --- | --- | --- |
| Actual load (6.1.A) | OK — hourly | OK — 15-min | OK | OK | OK |
| Day-ahead forecast (6.1.B) | OK | OK | OK | OK | OK |
| Week-ahead forecast (6.1.C) | OK (from \~2023) | OK | OK | OK | OK |
| Month-ahead forecast (6.1.D) | OK (from \~2023) | OK | OK | OK | OK |
| Year-ahead forecast (6.1.E) | OK (from \~2023) | OK | OK | OK | OK |
| Forecast margin (8.1) | Empty (genuine) | Empty | Empty | Empty | Empty |

CH week/month/year-ahead forecasts start \~2023; AT from \~2025. This is genuine non-publication at source. Actual load and day-ahead forecast are complete for all zones back to 2021. CH load is **hourly**; neighbours are **15-min** — resample/align before joining.

### Key issues

- **D-L1 — Longer forecast horizons staggered by TSO.** CH from \~2023, AT from \~2025. Not recoverable. Build features on actual load and day-ahead forecast (both complete).
- **D-L2 — Forecast column schema varies within a series.** A horizon comes back as `Min/Max Forecasted Load` or `Forecasted Load` depending on publication type. Coalesce at cleaning time.
- **D-L3 — `forecast_margin` (8.1) empty for all zones.** Recommend dropping it.
- **D-L5 — CH vs neighbour resolution mismatch.** CH hourly vs 15-min neighbours.

### Next steps

1. Run/confirm production pull and verify via `_pull_manifest.csv`.
2. Decide on `forecast_margin` (8.1) — likely drop it.
3. Integrate with Balancing and (once built) Swissgrid + Generation data for EDA.

---

## ENTSO-E Transmission pipeline

Two scripts: `entsoe_transmission_probe.py` and `entsoe_transmission_pull.py`. Nine article-variant combinations over 8 directed CH borders (CH↔DE/FR/IT\_NORD/AT). **Final pull: 56 ok / 13 genuinely empty / 0 error.** Pipeline is complete.

**Correction 2026-09-26:** CH↔DE NTC had been saved as `_empty.parquet` for all horizons — it was queried on `DE_TRANSNET`, but NTC is only published on the `DE_LU` border. Codes are now pinned and NTC CH-DE/DE-CH re-pulled.

Two structural differences from Load/Generation: (1) every product is directional or area-scoped, not single-area; (2) two articles (13.1.B countertrading, 13.1.C congestion costs) have no native entsoe-py method — added as custom wrappers on `_base_request`.

**DE neighbour resolves to two different EICs depending on product:** `DE_TRANSNET` for physical flows and scheduled exchanges; `DE_LU` for NTC. Pin these in `PINNED_BORDER_CODE` before any re-pull.

### Coverage

| Dataset | Variant | CH borders | Notes |
| --- | --- | --- | --- |
| NTC (11.1) | day/week/month/year-ahead | All 4 borders OK (DE on `DE_LU` since 2026-09-26) | DE week-ahead genuinely empty (also on `DE_LU`); CH-FR year-ahead only 2026 |
| Scheduled exchanges (12.1.F) | total A05 + day-ahead A01 | OK | Resolves to `DE_TRANSNET` |
| Physical flows (12.1.G) | A11 | OK | 15-min for DE, hourly for FR/IT\_NORD/AT |
| Countertrading (13.1.B) | A91 | OK | Custom wrapper; 100-instance cap → sub-hour overflow splitting |
| Congestion costs (13.1.C) | A92 | Empty (genuine) | Confirmed via diagnostic — no CH data at any resolution |

### All bugs found and fixed (logged for future reference)

- **D-T2 — A91 100-instance cap:** A month exceeds the API cap; handled via `window_on_overflow` decorator halving to 1-hour floor.
- **D-T3 — Duplicate labels on countertrading:** Half-open `[start, end)` boundary trim + per-column dedup before assembly.
- **D-T5 — False February retries:** Retry classifier substring-matched `"502"` in error message; `"202502…"` (February) contains `"502"`. Fixed by classifying from actual HTTP status code.
- **D-T6 — Error bodies truncated before useful content:** ENTSO-E puts `<Reason><code>/<text>` near the end of acknowledgement XML. Fixed by extracting via regex before head-slicing.

### Next steps

1. ~~Pin resolved border codes~~ — **DONE 2026-09-26.**
2. ~~Delete `entsoe_congestion_diag.py`~~ — moved to `Transmission/_to_delete/` 2026-09-26; delete that folder manually.
3. Feature engineering (downstream): NTC utilization = scheduled exchange ÷ NTC per border/hour; countertrading volume as congestion-activity proxy.
4. Integrate with Load, Generation, and Swissgrid data for EDA.

---

## ENTSO-E Outages pipeline

Two scripts: `entsoe_outages_probe.py` and `entsoe_outages_pull.py`. Seven article groups, 594 series-years (99 series × 6 years). **Final pull: 335 ok / 227 empty / 0 error.** Pipeline is complete.

Outages are structurally different from the other domains: (1) event documents with their own `doc_mrid`/`revision`, not regular time series; (2) overlap selection — a long outage appears in multiple year files, so **deduplication on `(doc_mrid, revision)` is required downstream**; (3) entsoe-py's outage parsers are not used (they crash on unknown EIC codes, drop `Reason.code/text`, and silently stop at offset 4800) — all parsing uses a custom lxml-based parser; (4) fall-backs paginate at 100 docs/page, not 200.

Five client wrappers added at import time: `query_outages_generation_units` (A80), `query_outages_production_units` (A77), `query_outages_transmission` (A78), `query_outages_offshore_grid` (A79), `query_outages_fallbacks` (A53).

### Articles covered

| Article | Description | Area grid |
| --- | --- | --- |
| 15.1.A | Planned unavailability of generation units (≥100 MW) | CH, DE\_LU, FR, IT\_NORD, AT |
| 15.1.B | Forced unavailability of generation units | same |
| 15.1.C | Planned unavailability of production units | same |
| 15.1.D | Forced unavailability of production units | same |
| 10.1.A | Planned unavailability in transmission grid | CH↔DE/FR/IT\_NORD/AT + CH internal |
| 10.1.B | Forced unavailability in transmission grid | same |
| 10.1.C | Unavailability of offshore grid infrastructure | DE only |
| IF aFRR 3.10 | aFRR fall-backs | CH, DE, FR, IT, AT (CTA level) |
| IF mFRR 3.11 | mFRR fall-backs | same |
| IFs IN 7.2 | Imbalance netting fall-backs | same |

### Performance

DE\_LU generation/production unit outages are the densest series (\~18,000 docs/year; \~18–20 min/year at 2 s interval). Full pull took \~9 hours. Fall-back series (360 series-years) are almost entirely empty — most of the 227 empty entries are fall-back combinations.

### Thesis relevance

- **RQ1:** Planned unavailability of generation/production units (15.1.A&B&C&D) is one of the most informative exogenous regressors identified in Kraft et al. (2020) for FCR prices. DE\_LU series particularly relevant given Switzerland's participation in the joint FCR Cooperation.
- **RQ1b:** Transmission outage volumes (10.1.A&B) are a secondary indicator of grid stress around the 2019–2020 reform period.
- **RQ3:** IF aFRR/mFRR fall-backs may be relevant, but the high empty rate suggests limited activation-level granularity — Swissgrid's own activation records remain the primary RQ3 target.

### Next steps

1. Inspect manifest for non-empty fall-back rows: which `(processType, businessType, area, year)` combinations have data, and are volumes meaningful for RQ3?
2. Deduplicate on `(doc_mrid, revision)` before any cross-year analysis.
3. Feature engineering (downstream): planned unavailable capacity (MW) per area per hour from 15.1.A+C; net transmission outage capacity per border from 10.1.A+B.

---

## Swissgrid ingestion — work done 2026-09-24

### Scripts built

| Script | Location | Purpose | Output |
| --- | --- | --- | --- |
| `inventory_manual_download.py` | thesis root | Lists every file in `Manual_Download` with size and first rows | `manual_download_inventory.txt` |
| `profile_swissgrid.py` | thesis root | Auction ID patterns / units / pricing per product-year; Energy Overview sheet structure; imbalance-price file layout | `swissgrid_profile.txt` |
| `swissgrid_auctions_parse.py` | `Swissgrid/Auctions/` | Production parser for all capacity auction files | see below |

Run: `python Swissgrid/Auctions/swissgrid_auctions_parse.py` from the thesis root (needs `pandas`, `pyarrow`). Reads everything under `Swissgrid/Manual_Download/Tenders/`.

### Auction parser output

```
Swissgrid/Auctions/Data/
├── bids/<FCR|aFRR|mFRR>/<delivery_year>.parquet   # one row per published bid
├── auctions.parquet                                # one row per auction × direction × delivery block
└── _parse_manifest.csv                             # per source file: rows, drops, checks
```

**Bids columns:** `product`, `direction` (up/down/sym), `auction_id`, `tender_series`, `revision`, `delivery_start/end` (datetime64\[us, Europe/Zurich\]), `duration_h`, `country`, `currency`, `offered_mw`, `awarded_mw`, `accepted`, `capacity_price`, `cost`, `bid_price_mwh`, `settle_price_mwh`, `divisible`, `flag`, `norm_ok`.

**Auctions columns:** `n_bids`, `n_accepted`, `offered_mw`, `awarded_mw`, `awarded_mw_ch`, `bid_min`, `bid_max` (marginal accepted bid), `bid_vwap`, `settle_max`, `settle_ch` (FCR: CH clearing price), `n_settle_prices`, `cost`.

**Final run:** 5,366,493 bids / 69,421 auction-blocks / 0 unparsed IDs / 0 conflicting boundary copies / `norm_ok_share` = 1.0 for every file.

### Parsing decisions

- **Products:** PRL = FCR (EUR), SRL = aFRR (CHF), TRL = mFRR (CHF). Currency kept native.
- **Published bids:** FCR and aFRR files contain accepted bids only; mFRR files also contain rejected bids (full bid curve).
- **Prices:** all normalised to per MW per hour. Pre-2019 `Preis` = bid (pay-as-bid); from 2019 `Angebotspreis` = bid, `Preis` = settlement price. Check: `capacity_price / duration_h == bid_price_mwh` holds in 100% of rows.
- **Delivery windows:** `KWnn` = ISO week (Mon–Sun), `YY_MM_DD` = day, `HH:MM bis HH:MM` = 4 h block. DST days give 3 h/5 h blocks and 167 h/169 h weeks.
- **Direction:** from ID (`TRL+`/`TRL-`) or description (`SRL+/-`, `UP`/`DOWN`); aFRR before 2018 and all FCR = `sym`.
- **Year-boundary duplicates:** same auction in two year files (e.g. `PRL_15_KW53`). Kept the copy with most rows (all copies turned out identical).
- **Revised auctions:** `TRL+_18_03_20-Neu` (70 bids) has no original → treated as normal. `TRL+_19_KW42_KORR` (1 bid) corrects one bid of `TRL+_19_KW42` (36 bids) → aggregated into the original. Small risk of 5 MW double-count in that week.
- **Data glitch:** one 2015 mFRR row has price unit `CH` instead of `CHF/MWh*` (flagged).
- **Advance-procurement file** (`20230906_regelleistung_2024_vorgezogene_beschaffung_ergebnis.csv`) is fully contained in the 2024 file → contributes 0 rows.

### Findings relevant to the thesis

**RQ1b is no longer at risk.** Auction data covers 2015–2026; all reforms are now observable:

- FCR: weekly → daily (Jul 2019) → 4 h blocks (2020); pay-as-bid → marginal (Jul 2019).
- aFRR: symmetric → split up/down (2018); daily 4 h blocks added 30 Sep 2025 (weekly and daily coexist in 2026).
- mFRR: `TRL+`/`TRL-` merged into `TRL` from week 40 2025.

The preliminary study lists only the 2019–2020 FCR reforms; the aFRR/mFRR breaks are new findings. FCR files include all cooperation countries (AT, BE, CZ, DE, DK, FR, NL, SI, CH) — use `settle_ch` / `awarded_mw_ch` for Swiss-specific series.

**RQ3:** No response-time data in any downloaded file. TRE shows only whether a bid was activated per 15 min.

### Files downloaded 2026-09-24

- Auction results 2026 and 2027 (`2026-PRL-SRL-TRL-Ergebnis.csv`, `2027-PRL-SRL-TRL-Ergebnis.csv`)
- TRE 06.2026 (republished), 08.2026 (complete), 09.2026 (to date)
- Imbalance prices 2026-04 → 2026-08 (XML/XLSX) + `swissgrid-prices-for-balance-energy.xsd`
- Refreshed `Ausgleichsenergie-und-Regelenergie-2026.csv`, `Grenzfluesse-2026.csv`; new `control-area-balance-2026.csv`
- `Swissgrid/Manual_Download/Secondary_control_energy/secondary-daily_2026-09-24.csv`

Not yet available online: pre-2023 imbalance prices; pre-2026 control-energy and cross-border CSVs → request from `sdl-ausschreibung@swissgrid.ch`.

### Remaining Swissgrid tasks (in priority order)

1. ~~**Clean Energy Overview copies.**~~ — **DONE 2026-09-26 (manual).** Deleted `Energy_in_the_grid/`, `Production_and_consumption/`, `Transmission/`. Re-downloaded `EnergieUebersichtCH-2026.xlsx` into `Balancing/` and re-ran `--years 2026 --force`. Energy Overview pipeline fully complete.
2. ~~**Build Energy Overview parser**~~ — **DONE 2026-09-25**, see section below.
3. ~~**Build imbalance-price parser.**~~ — **DONE 2026-09-26**, see section below.
4. **Parse 2026-only CSVs** (`Ausgleichsenergie-und-Regelenergie`, `Grenzfluesse`, `control-area-balance`). Semicolon-separated, `dd.mm.yyyy HH:MM`, 15-min. Cost columns I/J/K and R/S/T are cumulative weekly totals in kEUR — difference them before use. Intraday NTC in `Grenzfluesse` not available from ENTSO-E.
5. **Inspect `secondary-daily_2026-09-24.csv`** for RQ3. If useful, set up daily collection (cron/launchd — deferred for now).
6. **Email `sdl-ausschreibung@swissgrid.ch`**: historical second-by-second aFRR archive; anonymised provider-level activation / response-time data for research; pre-2023 imbalance prices and pre-2026 control-energy / cross-border files.
7. **TRE parser** (lower priority): latin-1 encoding, \~100 MB/month, chunked read, partition by month.
8. **Update thesis scope** (RQ1b): structural-break data now exists — document aFRR/mFRR breaks in Chapter 3.

---

## Swissgrid Energy Overview parser — work done 2026-09-25

### Script

| Script | Location | Purpose |
| --- | --- | --- |
| `swissgrid_energy_overview_parse.py` | `Swissgrid/EnergyOverview/` | Parses all `EnergieUebersichtCH-YYYY.xls/.xlsx` files (2009–2026) into clean 15-min and hourly parquet files |

Run from the thesis root (needs `pandas`, `pyarrow`, `openpyxl`, `xlrd`):

```
python Swissgrid/EnergyOverview/swissgrid_energy_overview_parse.py            # new years only
python Swissgrid/EnergyOverview/swissgrid_energy_overview_parse.py --force    # re-parse everything
python Swissgrid/EnergyOverview/swissgrid_energy_overview_parse.py --years 2026 --force
```

Reads only `Swissgrid/Manual_Download/Balancing/` (read-only), writes only to `Swissgrid/EnergyOverview/Data/`, never deletes. Existing years are skipped unless `--force`; outputs are replaced atomically.

### Output

```
Swissgrid/EnergyOverview/Data/
├── qh/<year>.parquet                 # 15-min, fixed 64-column schema
├── vertical_load_1h/<year>.parquet   # hourly vertical grid load (MW), 2009–2022 only
├── variables.csv                     # column → German label, English label, unit
└── _parse_manifest.csv               # per file × sheet: status, rows, convention, checks
```

Load all years in one call: `pd.read_parquet("Swissgrid/EnergyOverview/Data/qh/")`.

**Columns (64 + `timestamp`):** system totals (`end_user_consumption_kwh`, `production_kwh`, `consumption_kwh`, `net_outflow_tn_kwh`, `vertical_feedin_tn_kwh`), control energy (`afrr_pos/neg_energy_kwh`, `mfrr_pos/neg_energy_kwh`), cross-border flows (`xb_ch_de_kwh`, `xb_de_ch_kwh`, … 8 directed borders), `transit/import/export_kwh`, control-energy prices (`afrr_pos/neg_price_eur_mwh`, `mfrr_pos/neg_price_eur_mwh`), production/consumption for 18 canton groups (`prod_vs_kwh`, `cons_ge_vd_kwh`, …) plus cross-canton and foreign-territory totals. Secondary control = aFRR, tertiary = mFRR.

**Timestamps:** `timestamp` = interval **start**, `datetime64[us, Europe/Zurich]` (same convention as all other pipelines).
**Units:** as published — energy in kWh per interval (×4/1000 → average MW), prices in EUR/MWh.

### Final run (all 18 files)

| Years | Rows | Source columns | Label convention | Notes |
| --- | --- | --- | --- | --- |
| 2009–2013 | full year | 20 | interval end | no prices, no cantons (NaN) |
| 2014 | full year | 24 | interval end | prices added; unit written differently, values verified as EUR/MWh |
| 2015–2020 | full year | 64 | interval end | 1 DST label quirk per year (handled) |
| 2021–2024 | full year | 64 | interval end | clean labels |
| 2025 | full year | 64 | interval **start** | text timestamps |
| 2026 | partial year (to ~15 Sep 2026) | 64 | interval start | Re-downloaded and re-parsed 2026-09-26 |

0 errors, 0 unmapped headers, 0 NaN in source columns, every complete year has exactly 35,040 / 35,136 rows.

### Parsing decisions

- **Layout is long, not wide.** Timestamps in rows, variables in columns in every file. The "wide → transpose" note from 2026-09-24 was wrong (artefact of how `profile_swissgrid.py` displayed the sheets). No transpose needed.
- **Timestamps rebuilt from row position.** The parser builds a continuous UTC 15-min index from the first timestamp and checks every source label against it. Mismatches are only allowed on DST days; a gap or duplicate anywhere else fails that file (other files continue). Needed because 2009–2020 files label the last summer-time interval on the October switch day as 03:00 instead of 02:00, so the labels can't be converted directly.
- **Fixed schema.** Column names come from a fixed list of 64, never from the unit text. Every year gets all 64 columns, NaN where a variable wasn't published yet. Unit differences are flagged in the manifest (`unit_mismatches`), not silently renamed.
- **2014 price check:** yearly medians of aFRR prices 2014/2015/2016 = 44.4/49.3/42.0 (pos) and 28.6/32.3/26.8 (neg) EUR/MWh → same scale, no conversion needed.

### Findings relevant to the thesis (Chapter 3 data description)

- **Coverage grows over time:** 20 variables 2009–2013, 24 in 2014 (control-energy prices added), 64 from 2015 (canton data added). Any model using canton or price features starts in 2014/2015 at the earliest.
- **Timestamp convention changes in 2025:** interval end up to 2024, interval start from 2025. Normalised to interval start.
- **Hourly vertical load ends in 2022** (the note from 2026-09-24 said 2025). From 2023 use the 15-min vertical feed-in or ENTSO-E load instead.
- The Energy Overview contains **no imbalance prices** — those come from the separate imbalance-price files (next task).

### Remaining for this domain

None — pipeline complete as of 2026-09-26.

---

## Swissgrid imbalance-price parser — work done 2026-09-26

### Script

| Script | Location | Purpose |
| --- | --- | --- |
| `swissgrid_imbalance_prices_parse.py` | `Swissgrid/ImbalancePrices/` | Parses the monthly imbalance-price XML files (2023-01 → 2026-08) into 15-min parquet, cross-checks against the XLSX copies and ENTSO-E 17.1.G, and builds a 2021–2026 combined series |

Run from the thesis root (needs `pandas`, `pyarrow`, `openpyxl`):

```
python Swissgrid/ImbalancePrices/swissgrid_imbalance_prices_parse.py                        # new years only
python Swissgrid/ImbalancePrices/swissgrid_imbalance_prices_parse.py --years 2026 --force    # after adding a new month
python Swissgrid/ImbalancePrices/swissgrid_imbalance_prices_parse.py --no-xlsx-check         # skip the XLSX cross-check
```

Reads `Swissgrid/Manual_Download/Prices_for_imbalance_energy/` recursively (read-only), writes only to `Swissgrid/ImbalancePrices/Data/`, never deletes. Same skip/`--force`/atomic-write behaviour as the Energy Overview parser.

### Output

```
Swissgrid/ImbalancePrices/Data/
├── qh/<year>.parquet        # timestamp, long_eur_mwh, short_eur_mwh, aep_eur_mwh
├── combined_ch.parquet      # ENTSO-E 2021–2022 + Swissgrid 2023+, column `source`
├── _parse_manifest.csv      # per month: printedAt, series, rows vs expected, XLSX check, min/max
└── _entsoe_check.csv        # per month × series: overlap, share equal (±0.05 EUR/MWh), max diff, differing days
```

**Timestamps:** interval **start**, `datetime64[us, Europe/Zurich]`. **Units:** EUR/MWh (source ct/kWh × 10). The exact match with ENTSO-E confirms the unit and currency.

### Final run

| Year | Rows | Series | Notes |
| --- | --- | --- | --- |
| 2023 | 35,040 | long, short | complete |
| 2024 | 35,136 | long, short | complete (leap year) |
| 2025 | 35,040 | long, short (+ AEP from Jul) | complete |
| 2026 | 23,324 | AEP only | Jan–Aug |

44/44 months ok, 0 errors, 0 NaN, every month on an exact 15-min grid (92/100 rows on DST days handled through the UTC offsets). XLSX values = XML values in all 44 months.

### Parsing decisions

- **XML is the primary source.** Every timestamp carries an explicit UTC offset, so the October DST hour is unambiguous. The XLSX has local wall-clock labels only (ambiguous on DST days) and is used only as a row-by-row cross-check.
- **The official XSD doesn't match the files:** the XSD uses `Report-TS-Titel_*` (hyphens), the files use `Report_TS_Titel_*` (underscores). The parser accepts both and validates structure itself (series names, unit, grid) instead of against the XSD.
- **Series order varies between files** (long/short swapped) → mapped by name, never by position.
- **Unknown series name or unit ≠ ct/kWh** → that month fails loudly instead of being silently dropped.
- **Revisions:** if two files cover the same month, the one with the later `printedAt` wins (none in the current download).

### Findings relevant to the thesis

- **Pricing regime change — single imbalance price (BG-AEP).** Swissgrid published BG-AEP alongside BG-long/BG-short from **Jul 2025**; from **Jan 2026** only BG-AEP is published. ENTSO-E shows the same: from 2026 `Long` = `Short` = AEP. This is a structural break for any imbalance-price feature (dual → single pricing) — document in Chapter 3 next to the aFRR/mFRR reforms. (To check: official Swissgrid announcement and whether Jul–Dec 2025 AEP was informational only.)
- **ENTSO-E 17.1.G matches Swissgrid exactly** (±0.05 EUR/MWh) for 2023-01 → 2026-04, except:
  - **2–5 Jan 2025:** ENTSO-E has `Long` = `Short` (a single price) for these 4 days, while Swissgrid has distinct long/short values. Probably an ENTSO-E publication error → Swissgrid is the correct source here.
  - **Jan 2026:** missing entirely from the ENTSO-E pull → Swissgrid fills it.
  - **May–Aug 2026:** 0.2–1 % of intervals differ (up to 724 EUR/MWh), plus 7 intervals missing in ENTSO-E. Swissgrid files are printed mid-following-month, so these are probably settlement revisions not reflected on ENTSO-E → treat Swissgrid as final.
  - **1–2 Nov 2025:** 13 intervals differ (small).
- **ENTSO-E 2022 lacks 31 Dec 2022** → `combined_ch.parquet` has a one-day gap before 2023-01-01. **Confirmed missing at source 2026-09-26** (targeted API query and manual check on the Transparency Platform). Left as NaN; requested from Swissgrid.
- **Extreme values:** 2026 min −6,884.8 EUR/MWh (single price). Heavy tails → consider robust scaling / winsorising in EDA.

### Remaining for this domain

1. Re-run with `--years 2026 --force` after each new monthly download.
2. ~~Re-pull 31 Dec 2022~~ — not possible, gap is at source. Ask Swissgrid (email item).
3. Decide how to model the dual → single price break (e.g. use AEP from 2026, long/short before, regime dummy).

---

## ENTSO-E data-integrity round — work done 2026-09-26 (afternoon)

Goal: close the open ENTSO-E issues **before** combining ENTSO-E and Swissgrid into one dataset. All changes are non-destructive: every edited script has a `*.bak_20260926` copy next to it, and nothing was deleted (files moved to `_set_aside/` or `_to_delete/`). API pulls run on the Mac (`.venv`), because the Claude workspace cannot reach the ENTSO-E API.

### What was fixed

| # | Issue | Cause | Action | Result |
| --- | --- | --- | --- | --- |
| 1 | CH↔DE NTC empty for all horizons | Pull queried NTC on `DE_TRANSNET`; NTC is only published on the `DE_LU` border. The coverage CSV had been overwritten by a later `--only` probe run, so the pull fell back to the first candidate code | Filled `PINNED_BORDER_CODE` (NTC → `DE_LU`; flows/schedules/countertrading → `DE_TRANSNET`; countertrading IT border → `IT`); re-pulled `--only ntc` | Day-ahead 49,655 hourly rows per direction (no gaps, duplicates or NaN); month-ahead 2,068; year-ahead 67; week-ahead genuinely empty on `DE_LU` too |
| 2 | CH contracted-reserve gaps | The Balancing pull uses only the newest coverage CSV (June 2025 probe), so series absent in June 2025 were never pulled | Built `Data/_coverage_union.csv` (series ok in any of the 3 probe windows: 51 vs 40); re-pulled `--only contracted_reserve` | CH aFRR daily added (14 Oct 2025 → Aug 2026); CH mFRR daily prices 2023–24 and weekly 2024 confirmed empty at source |
| 3 | CH imbalance prices missing 31 Dec 2022 | Not published by ENTSO-E | `patch_ch_imbalance_20221231.py` (targeted query) + manual check on the Transparency Platform | Confirmed missing at source. Left as NaN, flagged later; request from Swissgrid |
| 4 | Stale manifests | Balancing listed 42 entries vs 216 files; Generation 37 vs 228 | New `Entsoe/rebuild_manifest_from_disk.py` (reads parquet metadata only); validated on Load: 144/144 files identical | Balancing 234 entries, Generation 228; old manifests kept as `.bak_20260926` |
| 5 | P4 — no request timeout | entsoe-py default | `REQUEST_TIMEOUT_S = 60` in all 5 probes, passed to the client in every probe and pull | Fixed in all domains; timeouts are retried |
| 6 | P5 slow throttle; D6 dense IT/DE bids | — | Default `--interval 0.3` in all 5 pulls; `EXCLUDE_SERIES` in the Balancing pull | Checked with `--plan`: only CH + FR bids remain |
| 7 | Partial IT/DE bid files | D6 | Moved to `Entsoe/Balancing/Data/_set_aside/` with a README | Done |
| 8 | `entsoe_congestion_diag.py` | Diagnostic finished | Moved to `Entsoe/Transmission/_to_delete/` | Delete the folder manually |

### Findings relevant to the thesis

- **Reserve-price source decision:** ENTSO-E contracted reserves have too many holes, so **Swissgrid auctions are the primary source for reserve prices and volumes (the targets)**. ENTSO-E contracted reserves are a cross-check only.
- **Cross-check that works:** ENTSO-E CH aFRR daily contracted price = volume-weighted average accepted bid. It matches Swissgrid `bid_vwap` in 83–86% of 4h blocks (correlation ≈ 0.96–0.98). ENTSO-E misses 30 Sep – 13 Oct 2025.
- **ENTSO-E CH FCR weekly price** is constant at 6.26 from Jan 2021 to Sep 2022, a stale legacy series. Exclude it.
- **ENTSO-E CH FCR daily price** is NaN from Nov 2021 to Apr 2022. Swissgrid covers this period.
- **CH↔DE NTC:** CH→DE day-ahead is a flat 4,000 MW in 2021–2025 (2026 median 3,059). DE→CH median is about 1,000 MW (max 2,200) and declines from 2021 to 2023. Some 0-MW hours need a look in EDA.
- **Pull validation:** a manual Transparency Platform export (CH imbalance, 27–31 Dec 2021) matches the API pull exactly (480/480 rows). This can be cited as validation.
- **Swissgrid mFRR 2024:** about 2,005 4h blocks per direction vs about 2,190 expected (roughly 30 days missing). Check this in the Swissgrid round.
- **CH imbalance volumes (17.1.H):** 4 quarter-hours missing in 2024 and 102 in 2025 (Sep–Oct). Fill or flag in the clean layer.
- **Timezones are mixed in the raw files:** DE series in Europe/Berlin, IT per-unit in Europe/Rome, water reservoirs in UTC. Normalise in the clean layer.

### Commands (run from `Master_Thesis`, venv active)

```
python Entsoe/Transmission/entsoe_transmission_pull.py --only ntc --interval 0.3
python Entsoe/Balancing/entsoe_balancing_pull.py --only contracted_reserve --interval 0.3
python Entsoe/Balancing/patch_ch_imbalance_20221231.py          # dry run; add --write to apply
python Entsoe/rebuild_manifest_from_disk.py <Balancing|Generation|Load> [--write]
```

---

## Plan going forward — from raw data to one dataset

Order: **fix open data issues → clean layer → combined dataset → modelling views → EDA.**

1. **Close the remaining data issues.** ENTSO-E checks 9–10 and the Swissgrid list (see open items).
2. **Clean layer, per domain (not started).**
   - One time convention: UTC internally, Europe/Zurich interval-start labels.
   - Clear column names, e.g. `ch_load_actual_mw`, `de_lu_solar_da_fc_mw`.
   - Coalesce forecast column variants (D-L2).
   - Treat daily/weekly block products as step functions (D5).
   - Deduplicate outages on `(doc_mrid, revision)` and convert them to MW unavailable per interval.
   - Drop unreliable series: 8.1 margin, 12.3.F, CH FCR weekly, IT/DE bids.
   - Flag known gaps: 31 Dec 2022, ENTSO-E Jan 2026, imbalance-volume gaps.
3. **Combined dataset.** One 15-min master table (2021-01 → 2026-08), rows = time, columns = variables. CH plus neighbours as prefixed columns. Swissgrid takes priority where both sources overlap.
4. **Modelling views.** Aggregate to each target's grain: 4h blocks and days (FCR, aFRR daily), weeks (weekly products).
5. **EDA (Phase 3).** Data quality, distributions, seasonality, structural breaks (RQ1b), relationships between drivers and prices, target definition. Most of it feeds Chapter 3.

**Design choices still to confirm** (recommended options in bold):

- Resolution: **15-min master + aggregated views**, hourly only, or 4h blocks only.
- Scope: **CH + neighbours**, or CH only.
- Build: **clean layer, then join**, or a single builder script.

---

## Open items and next steps

### ENTSO-E status

Items 1–8 of the data-integrity list are **DONE** (see above). Remaining:

1. **\[Generation\] Check 16.1.A per-day fallback** (item 9). Confirm the skipped DST days are negligible. Reads disk only.
2. **\[Outages\] Check fall-back rows for RQ3** (item 10). CH aFRR fall-backs have about 830 rows. Reads disk only.

### Swissgrid (next focus)

3. **Email `sdl-ausschreibung@swissgrid.ch` (overdue; the only RQ3 lead).** Request:
   - the historical second-by-second aFRR archive;
   - anonymised provider-level activation / response-time data;
   - pre-2023 imbalance prices, **incl. 31 Dec 2022**;
   - pre-2026 control-energy and cross-border files.
4. **Inspect `secondary-daily_2026-09-24.csv`.** Is it relevant for RQ3?
5. **Parse the 2026-only CSVs** (`Ausgleichsenergie-und-Regelenergie`, `Grenzfluesse`, `control-area-balance`). Weekly cumulative cost columns must be differenced.
6. **Confirm the switch to a single imbalance price (AEP)** with Swissgrid's official announcement.
7. **Check the Swissgrid mFRR 2024 gap** (about 30 days of 4h blocks missing).
8. **TRE parser** (lower priority).
9. **Imbalance prices:** re-run `--years 2026 --force` after each new monthly download.

### Then

10. **Clean layer → combined dataset → views → EDA** (see plan above).
11. **\[JAO\]** Assess whether a dedicated pull is needed (NTC is now complete for all 4 CH borders via ENTSO-E 11.1).

### Thesis scope updates

- **RQ1b is no longer at risk.** Auction data for 2015–2026 covers all reform transitions. Update Chapter 3 to include the aFRR/mFRR structural breaks (2018 direction split; 2025 daily blocks; 2025 mFRR merger) and the switch from dual to single imbalance pricing (2025/26).
- **RQ3 remains at risk.** There is still no confirmed access path. If the Swissgrid email and the `secondary-daily` inspection yield nothing, trigger the contingency and drop RQ3. The original deadline (~25 Jul 2026) has passed; send the email now and agree a new date with the supervisor.
- **Schedule:** late September 2026 is week 17 of 29. EDA was planned for weeks 7–12 and model development for weeks 13–19, but data acquisition is still running. Review the remaining phases with the supervisor.
- **Chapter 3 — data description:**
  - Energy Overview coverage changes (20 → 24 → 64 variables), the 2025 timestamp change, and the end of hourly vertical load in 2022.
  - Swissgrid auctions as the primary reserve-price source; ENTSO-E as cross-check (aFRR validation above).
  - **Known source gaps:** 31 Dec 2022 imbalance prices; ENTSO-E Jan 2026 imbalance prices; ENTSO-E CH FCR daily prices Nov 2021 – Apr 2022; ENTSO-E CH mFRR daily prices 2023–24; CH↔DE week-ahead NTC not published.

### Blocked

- **EDA (Phase 3):** blocked on the clean layer and the combined 15-min dataset.

---

## File layout reference

All pipelines follow the same two-script pattern (probe + pull) and the same output structure. One parquet per `(dataset, variant, area/target, year)`, resumable via `_pull_manifest.csv`.

```
Master_Thesis/
├── Entsoe/
│   ├── rebuild_manifest_from_disk.py        # rebuilds a manifest from the parquet files
│   ├── Balancing/
│   │   ├── entsoe_balancing_probe.py
│   │   ├── entsoe_balancing_pull.py
│   │   ├── patch_ch_imbalance_20221231.py   # one-off (confirmed: no data at source)
│   │   └── Data/
│   │       ├── _coverage_<range>.csv
│   │       ├── _coverage_union.csv          # union of all probe windows (drives the pull)
│   │       ├── _set_aside/                  # IT/DE aggregated bids (D6) + README
│   │       └── production/
│   │           ├── _pull_manifest.csv
│   │           └── <dataset>/<variant>/<area>/<year>.parquet
│   ├── Generation/     # same structure
│   ├── Load/           # same structure
│   ├── Transmission/   # target = directed border (e.g. CH-DE); _to_delete/ holds the old diag script
│   └── Outages/        # year.parquet | year.empty | _empty.parquet
│
└── Swissgrid/
    ├── Auctions/
    │   ├── swissgrid_auctions_parse.py
    │   └── Data/
    │       ├── bids/<FCR|aFRR|mFRR>/<delivery_year>.parquet
    │       ├── auctions.parquet
    │       └── _parse_manifest.csv
    ├── ImbalancePrices/
    │   ├── swissgrid_imbalance_prices_parse.py
    │   └── Data/
    │       ├── qh/<year>.parquet
    │       ├── combined_ch.parquet
    │       ├── _parse_manifest.csv
    │       └── _entsoe_check.csv
    ├── EnergyOverview/
    │   ├── swissgrid_energy_overview_parse.py
    │   └── Data/
    │       ├── qh/<year>.parquet
    │       ├── vertical_load_1h/<year>.parquet
    │       ├── variables.csv
    │       └── _parse_manifest.csv
    ├── Manual_Download/
    │   ├── Tenders/          # auction CSV files (input to parser)
    │   ├── TRE/              # monthly activation files
    │   ├── Secondary_control_energy/
    │   └── ...               # imbalance prices, CSVs, XSD
    └── ...
```

**Common pipeline conventions:**

- `probe.py` owns all request-layer logic; `pull.py` imports probe as a library and contains no fetch logic.
- `_pull_manifest.csv`: filter `rows > 0` to see what actually landed; `rows = 0` with no file = genuine empty (API returned no data).
- Parquet naming: `<dataset>/<variant>/<area>/<year>.parquet` for all ENTSO-E domains.
- Outages additionally use `<year>.empty` (skip on re-pull) and `_empty.parquet` (series empty across all years).
- All datetime columns: `datetime64[us, Europe/Zurich]` — except in the raw ENTSO-E files: DE series use Europe/Berlin, IT per-unit Europe/Rome, water reservoirs UTC (normalised in the clean layer).
- Every probe/pull has a 60 s request timeout (`REQUEST_TIMEOUT_S`) and every pull defaults to `--interval 0.3`.
- Edited scripts from 2026-09-26 have a `*.bak_20260926` backup next to them.
