Sep 29, 2026 · @Johan

Data acquisition layer for the Master Thesis (Swiss ancillary-service price forecasting). All ENTSO-E pipelines cover 2021-01 to 2026-09, **plus a 2015-01 → 2020-12 extension pulled on 2026-09-28 into separate `production_pre2021/` folders** (see *Pre-2021 extension*); Swissgrid auctions cover 2015-2027; **regelleistung.net (German / FCR-cooperation capacity auctions) added 2026-09-29: aFRR/mFRR from 12 Jul 2018, FCR from 1 Jul 2019, to 31 Aug 2026** (see *regelleistung.net pipeline*).

## Pipeline status snapshot

| Domain                                             | Scripts                                | Coverage                                                 | Pull result                                                                                                                                                                         | Last updated |
| -------------------------------------------------- | :------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| **Balancing** (17.1.B/C/F/G/H, 12.3.E/F)           | probe + pull                           | CH complete; DE/FR/IT context (no foreign bids)          | 204 files, manifest re-verified (234 entries = 204 files + 30 empty records); FR mFRR bids decided → set aside (expect 198 files, CH bids only); retry classifier fixed (P7)        | 2026-09-27   |
| **Generation** (14.1.A/B/C/D, 16.1.A/B&C/D)        | probe + pull                           | CH + 4 neighbours                                        | Manifest re-verified 228/228. CH/DE_LU complete; AT 1 h. DST loss checked hour by hour: 7 h in total (CH 1 h) → negligible. Source gaps: FR per-unit 9 + 16–26 Feb 2026 + scattered hours 2021–24; IT_NORD per-unit 5 full days 2022–26 + ends 18 Jul 2026 (check ~20 Oct); DE_LU reservoirs not published. Gap check: `check_data_gaps.py` | 2026-09-28   |
| **Load** (6.1.A/B/C/D/E, 8.1)                      | probe + pull                           | CH + 4 neighbours                                        | Complete; CH longer horizons start \~2023 (genuine); retry classifier fixed (P7)                                                                                                    | 2026-09-27   |
| **Transmission** (11.1, 12.1.F/G, 13.1.B/C)        | probe + pull                           | 8 directed CH borders                                    | CH↔DE NTC re-pulled on `DE_LU` (day/month/year-ahead ok; week-ahead genuinely empty); border codes pinned; diag script deleted                                                      | 2026-09-27   |
| **Outages** (15.1.A–D, 10.1.A–C, IF fall-backs)    | probe + pull                           | CH + 4 neighbours                                        | 335 ok / 227 empty / 0 error; fall-backs re-checked field by field: no RQ3 value confirmed; CH aFRR platform fall-back almost continuous 2025–26 (new structural break) — 2022–24 history only published 15 Apr 2025, all documents start at 00:00; retry classifier fixed (P7) | 2026-09-28   |
| **Swissgrid — Auctions** (FCR/aFRR/mFRR 2015–2027) | swissgrid\_auctions\_parse.py          | 2015–2027 all products                                   | 5,366,493 bids / 69,421 auction-blocks / 0 unparsed                                                                                                                                 | 2026-09-24   |
| **Swissgrid — Energy Overview**                    | swissgrid\_energy\_overview\_parse.py  | 2009–2026 complete                                       | 18/18 files ok, 0 errors, 0 NaN; fixed 64-column schema                                                                                                                             | 2026-09-26   |
| **Swissgrid — Imbalance prices**                   | swissgrid\_imbalance\_prices\_parse.py | 2023-01 → 2026-08 (+ ENTSO-E 2021–2022 in combined file) | 44/44 months ok, 0 errors, 0 NaN; XLSX = XML in all 44; ENTSO-E match ≥99% except 2–5 Jan 2025; Jan 2026 absent in ENTSO-E; 31 Dec 2022 confirmed missing at source                 | 2026-09-26   |
| **Swissgrid — TRE / CSVs**                         | not built                              | 2023-01 → 2026-09 (downloaded)                           | Not parsed                                                                                                                                                                          | —            |
| **ENTSO-E 2015–2020 extension** (all 5 domains)   | probes + pulls (`--coverage-pre`), `pre_split.py`, `build_pre2021_coverage.py` | 2015-01 → 2020-12, CH + 4 neighbours (DE: `DE`/`DE_AT_LU` before 1 Oct 2018) | 0 errors; Load 168/168, Gen 195/195, Bal 66/78, Trans 60 ok / 9 empty, Outages 170 ok / 424 empty; separate `production_pre2021/` trees; Q4 2018 DE load-forecast patch applied (35,040/35,040) | 2026-09-28 |
| **Combined dataset (ENTSO-E + Swissgrid)**         | not built                              | 2021-01 → 2026-08                                        | Planned: clean layer → 15-min master → modelling views                                                                                                                              | —            |
| **RQ3 response-time data**                         | —                                      | No confirmed source                                      | Unsourced — at risk. ENTSO-E fall-back lead closed 2026-09-27 (re-confirmed 2026-09-28: no provider, quantity or timing fields); Swissgrid email is the only lead                  | 2026-09-28   |
| **Regelleistung.net — capacity results** (FCR/aFRR/mFRR) | probe + pull (`Regelleistung/`) | aFRR/mFRR 12 Jul 2018 → 31 Aug 2026; FCR 1 Jul 2019 → 31 Aug 2026 | 8,952 requests: 8,565 ok / 387 empty (all before the market start) / 0 error; 27 parquet year files, delivery times parsed 100%; no missing days after the start dates | 2026-09-29   |
| **JAO (NTC)**                                      | —                                      | Partially covered by TR 11.1                             | Not addressed                                                                                                                                                                       | —            |

---

## ENTSO-E Balancing pipeline

Two scripts: `entsoe_balancing_probe.py` (coverage probe + all shared request machinery) and `entsoe_balancing_pull.py` (production pull, imports probe as library). The pull ran \~19 h. **216 / 222 series-years written**; the 6 missing are `aggregated_bids A47_mFRR FR` (2021–2026) — non-core — stalled on a hung network request. Switzerland is completely and cleanly covered.

**Update 2026-09-26:** FR bids 2021–2026 are now on disk; IT/DE bids moved to `Data/_set_aside/` (D6); CH aFRR daily and CH FCR weekly contracted reserves added via `_coverage_union.csv`; manifest rebuilt from disk (234 entries). See *ENTSO-E data-integrity round* below.

**Update 2026-09-27:** Manifest re-verified (rebuild is idempotent: 234 → 234). Request timeout (P4) re-verified in all 5 pipelines. Retry classifier fixed (P7). FR mFRR aggregated bids: decided to set aside (D9) and add to `EXCLUDE_SERIES` → only CH bids remain. See *ENTSO-E verification round* below.

### What's on disk

| Series | ENTSO-E | Status | Notes |
| --- | --- | --- | --- |
| Imbalance prices | 17.1.G | OK | 15-min |
| Imbalance volumes | 17.1.H | OK | 15-min |
| Activated balancing energy prices | 17.1.F | OK | Long format (duplicate timestamps legitimate) |
| Aggregated bids — aFRR / mFRR | 12.3.E | OK (CH only) | CH aFRR complete; CH mFRR sparse in 2021 (105 days) and 2022 (197 days), complete from 2023 → use Swissgrid bids; IT/DE set aside 2026-09-26 (D6); FR set aside 2026-09-27 (D9) |
| Contracted reserve price + volume — FCR | 17.1.B&C | OK | Daily; + weekly pre-reform |
| Contracted reserve price + volume — aFRR | 17.1.B&C | OK | Weekly (A02); daily (A01) from 14 Oct 2025 (added 2026-09-26) |
| Contracted reserve price + volume — mFRR | 17.1.B&C | Gaps | Daily + weekly; daily prices 2023–24 and weekly 2024 empty at source |
| Contracted reserve price + volume — FCR weekly | 17.1.B&C | Do not use (2020–22) | Price constant 6.26 Jan 2021–Sep 2022 → stale legacy series. 2015–2019 values are real (pre-2021 pull, median 22 → 8 EUR/MW/h) |
| Procured balancing capacity — FCR / mFRR | 12.3.F | Unstable | Varies across years (D8); rely on contracted reserves instead |

### Reform structure confirmed in data

CH FCR was daily + weekly in 2021 → daily-only by 2025. CH mFRR volume/price split across agreement types in 2021, consolidated by 2025. FCR daily/weekly overlap during reform must be reconciled at cleaning time, not concatenated. The raw pull keeps every (product × agreement-type) as a separate series.

### Key data issues

- **D2 — Area level mixed:** `imbalance_volumes DE → DE_AMPRION` = Amprion control area only (\~¼ of Germany). `IT_NORD` series = Italian North bidding zone, not Italy. DE contracted reserve prices are fine (uniform procurement).
- **D5 — Forward-fill artefact:** Weekly/daily capacity blocks are ffilled to 15-min — these are step functions, not 15-min observations. Resample accordingly.
- **D6 — Dense foreign aggregated bids (IT, DE) irreducibly incomplete:** Single day exceeds the 100-TimeSeries cap. Recommend dropping IT/DE bids, keeping CH. **Done 2026-09-26** (`EXCLUDE_SERIES` in pull; files in `Data/_set_aside/`).
- **D8 — `procured_balancing_capacity` (12.3.F) unstable across years.** CH FCR/mFRR present in 2021, empty in 2025.
- **D9 — FR mFRR aggregated bids patchy:** 193–299 days per year (2021 only from May). Cap truncation and non-publication can't be told apart. Not used in modelling → set aside and added to `EXCLUDE_SERIES` (2026-09-27).

### Key pipeline issues

- **P4 — No request timeout (FIXED 2026-09-26 — 60 s `REQUEST_TIMEOUT_S` in all 5 pipelines):** `entsoe-py` issues HTTP requests with no timeout. A stalled connection hangs forever. This is what hung the run at \[217/222\] for \~13 h. Fix: wrap requests with a hard \~60 s socket timeout.
- **P5 — Throttle interval too conservative (FIXED 2026-09-26 — default `--interval 0.3` in all 5 pulls):** At `--interval 2.0` the pull ran \~19 h. Lever: `--interval 0.3` (\~200 req/min, half the 400/min ceiling) cuts runtime dramatically.
- **P6 — Rate limiting is NOT the cause of errors:** ENTSO-E's limit is 400 req/min; exceeding it returns HTTP 429 + a 10-min ban. We never hit 429. The 400/599 errors are data-volume/server issues.
- **P7 — Retry classifier matched status codes as text (FIXED 2026-09-27):** Balancing, Generation, Load and Outages retried when `"429"` or `"50x"` appeared anywhere in the error message. The message includes the request URL, so dates like `202502…`, `202504…` or `…0429…` made genuine 400 errors retry with backoff (wasted time only, no data affected). Now classified from the HTTP status code, the same logic as the Transmission fix (D-T5); text matching only when there is no response (timeouts, dropped connections). Offline test 5/5 ok. Backups: `*.py.bak_classifier`.

### Next steps

1. ~~Accept 216/222 and add timeout fix (P4)~~ — **DONE 2026-09-26.**
2. ~~Verify via `_pull_manifest.csv`~~ — **DONE 2026-09-26** (manifest rebuilt from disk; CH covers 2021-01 → 2026-08 except the gaps listed above).
3. ~~Drop IT/DE from the bids pull; `--interval 0.3` default~~ — **DONE 2026-09-26.**
4. ~~German benchmark capacity prices → **regelleistung.net**~~ — **DONE 2026-09-29** (see *regelleistung.net pipeline*).
5. ~~Decide on FR bids~~ — **DECIDED 2026-09-27:** set aside (D9) + `EXCLUDE_SERIES`; rebuild manifest (expect 198 files). Mark done once `--only aggregated_bids --plan` shows CH only.

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
| 16.1.A | Actual generation per unit | OK (1 h missing 2021–26: 26 Oct 2025) | OK (0 h missing) | OK; 13/36/18/5 h missing 2021–24; 2026 missing 9 Feb + 16–26 Feb | OK; full days missing 12 May 2022, 18 Jun 2025, 26 Jan + 9 Jul 2026; 2 Mar 2025 23 h; 2026 ends 18 Jul | OK (1 h missing: 26 Oct 2025) |
| 16.1.B&C | Actual generation per type | OK | OK | OK | OK | OK |
| 16.1.D | Water reservoirs & hydro storage | OK | Empty (genuine, `NoMatchingDataError`) | OK | OK | OK |

All-NaN fuel columns (e.g. offshore wind for CH) are preserved in parquet — do not drop them; the schema must be stable across areas.

### Key issues

- **D-G1 — DST-transition 400s on 16.1.A:** entsoe-py constructs malformed period parameters on CET fall-back days. Per-day fallback catches these; 1–2 days lost per year. No fix without patching entsoe-py. Day-level check (2026-09-27): no whole days lost for CH/AT/DE_LU. **Hour-level check (2026-09-28): 7 h lost on all 55 DST days (2021–26, 5 areas) — CH 1 h, AT 1 h (both 26 Oct 2025), DE_LU 0, FR 5, IT_NORD 0. Share of non-empty unit values on DST days is the same as on normal days → negligible, closed.**
- **D-G2 — entsoe-py parser crash on FR 16.1.A:** Some FR generation units lack the `<name>` XML tag → `AttributeError`. Per-day fallback recovers the vast majority of days.
- **P-G1 — 16.1.A is slow:** A single area-year takes several minutes. Use `--only per_unit` to run separately.
- **P-G2 — No request timeout (shared with Balancing P4 and Load P-L2).** FIXED 2026-09-26.
- **D-G3 — Source gaps confirmed by re-pull (2026-09-27):** FR 2026 per-unit is missing 9 Feb and 16–26 Feb (the per-day loop skips the same 12 days again). IT_NORD 2026 per-unit ends 18 Jul 2026 (the year-wide call succeeds and returns an identical file) — probably publication lag, re-pull later. `water_reservoirs` DE_LU is never published (probe: `NoMatchingDataError`).
- **D-G4 — Non-DST hour gaps in neighbour per-unit data (found 2026-09-28, hour-level check):** IT_NORD has more missing days than the day-level check showed: 12 May 2022, 18 Jun 2025, 26 Jan 2026 and 9 Jul 2026 (full days), 2 Mar 2025 (23 h) and three 2-h gaps in 2025. FR has scattered missing hours in 2021–24 (13/36/18/5 h; largest 23 May 2022, 14 h). CH is not affected. Flag in the clean layer; only relevant if neighbour per-unit data is used.
- **Note:** a pull reports "OK" per year even when single hours or days are missing. The hour-level check (`check_data_gaps.py`) is what finds them.

### Thesis relevance

- **RQ1:** VRES generation (16.1.B&C, 14.1.D) and hydro reservoir filling rate (16.1.D) are key exogenous drivers. The water reservoir series captures Switzerland's dual role as energy producer and reserve provider.
- **RQ2:** Day-ahead generation forecast (14.1.C), wind & solar forecast (14.1.D), and installed capacity (14.1.A) are candidate forecasting features.

### Next steps

1. ~~Verify pull via `_pull_manifest.csv`~~ — **DONE 2026-09-26**: manifest was stale (37 entries); rebuilt from disk → all 228 files listed. Re-verified 2026-09-27 (228/228).
2. ~~Check per-day fallback for 16.1.A~~ — **DONE 2026-09-27**: CH/AT/DE_LU complete 2021–2026; IT_NORD 1 day missing in 2022 and 2025; FR 2026 and IT_NORD 2026 gaps confirmed at source (D-G3).
3. Feature engineering (downstream): VRES share = wind+solar / total generation; hydro availability = reservoir filling rate.
4. Re-pull IT_NORD 2026 per-unit later, in case the gap after 18 Jul is publication lag. **~20 Oct 2026:** first check the Transparency Platform; re-pull only if data exists (step-by-step procedure in *Gap-check round* below). If still missing, log as a source gap and stop checking.
5. ~~Hour-level DST check for 16.1.A~~ — **DONE 2026-09-28**: 7 h lost in total, negligible (D-G1).

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
2. ~~Delete `entsoe_congestion_diag.py`~~ — **DONE 2026-09-27** (deleted; no references remain).
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
- **RQ1b / RQ2 — CH aFRR platform fall-back (new, 2026-09-27):** The aFRR fall-back documents show CH aFRR running in fall-back mode (connection to the European platform lost) for about 99% of 2025 and all of 2026 to date. First full-day fall-back 9 Feb 2024; continuous without gaps since 21 May 2026. This is a structural break for aFRR prices → build a `ch_afrr_platform_fallback` flag in the clean layer.
- **RQ3:** ~~IF aFRR/mFRR fall-backs~~ — checked 2026-09-27: no RQ3 value. The documents are TSO-level platform connection losses (reason B13, "Real time connection lost"), with no provider, activation or response-time data. CH mFRR fall-backs are empty. Swissgrid's own activation records remain the only RQ3 target. **Re-confirmed 2026-09-28 field by field:** every document has Swissgrid as sender and receiver, reason B13 only, revision 1, and no `Point`/quantity data in the XML (the parser keeps Points when present). Nothing about providers, activation or timing.
- **Caveats for the fall-back flag (found 2026-09-28):**
  - **Published after the fact:** all 2022–2024 documents were created on 15 Apr 2025 (median lag 998 / 618 / 201 days for 2022 / 2023 / 2024); from Apr 2025 each day is published the next day. For forecasting (RQ2) the flag was not known in real time before Apr 2025 → use it as a regime indicator, not as a same-day feature.
  - **Coarse intraday timing:** all 831 documents start at local 00:00, 796 last exactly 24 h. On partial days the duration is known but probably not the true start time → build the flag per day, or treat intraday timing as approximate.

### Next steps

1. ~~Inspect fall-back rows for RQ3~~ — **DONE 2026-09-27**: no RQ3 value; aFRR platform fall-back is a new structural break (see thesis relevance and *ENTSO-E verification round*).
2. Deduplicate on `(doc_mrid, revision)` before any cross-year analysis — **and across the business-type folders** (`planned_A53`, `unplanned_A54`, `disconnection_C47`, …): the API ignores the business-type filter for fall-backs, so every folder holds the same documents (aFRR: 3,324 rows → 831 unique; imbalance netting: 52 → 13).
3. Feature engineering (downstream): planned unavailable capacity (MW) per area per hour from 15.1.A+C; net transmission outage capacity per border from 10.1.A+B.
4. ~~Re-check fall-back rows for RQ3~~ — **DONE 2026-09-28**: confirmed no RQ3 value; publication-lag and intraday-timing caveats added (see thesis relevance).
5. After any Outages re-pull: run `check_data_gaps.py fallback` and compare with the baseline (see *Gap-check round*).

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
| 8 | `entsoe_congestion_diag.py` | Diagnostic finished | Moved to `Entsoe/Transmission/_to_delete/`, then deleted | Deleted 2026-09-27 |

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

## ENTSO-E verification round — work done 2026-09-27

Goal: verify on the Mac that the 2026-09-26 fixes are really in place, and close the two remaining disk-only checks. Everything was run by hand from `Master_Thesis` with `.venv` active. Nothing was deleted except the old diagnostic script; every edited script has a backup.

### What was checked or fixed

| # | Item | Finding | Action | Result |
| --- | --- | --- | --- | --- |
| 9 | Balancing / Generation manifests | Already rebuilt on 2026-09-26; the 18/37 figures came from the old `.bak_20260926` files | Re-ran the rebuild (dry run + `--write`) | Identical: Balancing 234 entries (204 files + 30 empty records), Generation 228/228. Rebuild is idempotent |
| 10 | `entsoe_congestion_diag.py` | `_to_delete/` no longer existed; `find` shows no copy anywhere | — | Deleted; no references remain |
| 11 | P4 request timeout | All 5 probes and 5 pulls create the client with `timeout=REQUEST_TIMEOUT_S`; no direct HTTP calls; entsoe-py 0.8.1 supports `timeout`. `Load/check_a32.py` (one-off diagnostic) has no timeout — harmless | None needed | Confirmed |
| 12 | P7 retry classifier | Same bug class as D-T5 in Balancing, Generation, Load, Outages (status codes matched as text, URL dates matched) | Status-code classification copied from Transmission; backups `*.py.bak_classifier` | All 5 pipelines consistent; compile ok; offline test 5/5 ok |
| 13 | P5 interval / D6 exclusion | `--interval` default 0.3 in all 5 pulls; `EXCLUDE_SERIES` drops DE/IT bids; 12 IT/DE files in `_set_aside/` | None needed | Confirmed |
| 14 | FR mFRR aggregated bids (D9) | 193–299 days per year; not used in modelling | Decided: move to `_set_aside/`, add `("aggregated_bids", "FR")` to `EXCLUDE_SERIES` (backup `.bak_fr_exclude`), README note, rebuild manifest | Decided; verify: `--plan` shows 2 CH series, 198 files |
| 15 | 16.1.A per-day fallback | CH/AT/DE_LU: 0 missing days. IT_NORD: 1 day in 2022, 1 in 2025. FR 2026: 12 days. IT_NORD 2026: ends 18 Jul | Re-pulled FR and IT_NORD 2026 with a filtered coverage CSV (`/tmp/coverage_perunit_fr_itnord.csv`), `--force`, after a backup | Identical files → gaps are at source (D-G3) |
| 16 | Outages fall-backs (RQ3) | TSO-level "Real time connection lost" documents; no provider or activation data; all business-type folders identical | — | RQ3 lead closed; new aFRR structural break found |
| 17 | Generation `water_reservoirs` | 24 series instead of 30: DE_LU missing | Checked probe coverage: `NoMatchingDataError` | Genuine non-publication |

### Finding: CH aFRR platform fall-back

Fall-back hours per year after deduplication (CH, aFRR):

| Year | Documents | Fall-back hours | Share of year |
| --- | --- | --- | --- |
| 2022 | 62 | 1,378 | \~16% (from June) |
| 2023 | 6 | 61 | <1% |
| 2024 | 157 | 3,716 | \~42% |
| 2025 | 363 | 8,660 | \~99% |
| 2026 (to 31 Aug) | 243 | 5,821 | \~100% |

First full-day fall-back 9 Feb 2024; from 2025 almost every day (breaks of 1–3 days only: late Feb, early Apr, early Nov 2025; late Mar 2026); continuous since 21 May 2026. Imbalance netting fall-backs are rare and short (3–4 events per year, minutes each, \~5 h in total). CH mFRR fall-backs: no data.

**To do:** confirm the reason with Swissgrid (announcements about the European aFRR platform, and the email), and model it as a flag, not a single break date.

### Other findings

- **CH mFRR aggregated bids** are sparse in 2021 (105 days) and 2022 (197 days), complete from 2023. The Swissgrid auction files (full mFRR bid curve) are the primary source anyway.
- **IT_NORD 2026 per-unit generation** may be publication lag: re-pull later and check on the Transparency Platform (Actual Generation per Generation Unit, IT-North, any date after 18 Jul).

### Commands (run from `Master_Thesis`, venv active)

```
python Entsoe/rebuild_manifest_from_disk.py Balancing [--write]
python Entsoe/Balancing/entsoe_balancing_pull.py --only aggregated_bids --plan
# targeted re-pull: filtered copy of the coverage CSV, so other areas are not overwritten
python Entsoe/Generation/entsoe_generation_pull.py --coverage /tmp/coverage_perunit_fr_itnord.csv \
  --start 2026-01-01 --end 2026-09-01 --force [--plan]
```

---

## Gap-check round — work done 2026-09-28

Goal: close the two remaining checks from 2026-09-27 (DST loss on 16.1.A; RQ3 value of the fall-back rows) and make them repeatable. Read-only: no data or pipeline script was changed.

### What was checked

| # | Item | Finding | Result |
| --- | --- | --- | --- |
| 18 | 16.1.A hours lost around DST switches | Checked hour by hour (UTC hour buckets) on all 55 DST days 2021–26, 5 areas: 7 h missing in total — CH 1 h and AT 1 h (26 Oct 2025), FR 5 h (31 Oct 2021 3 h, 27 Mar 2022 1 h, 26 Mar 2023 1 h), DE_LU/IT_NORD 0. Non-empty unit share on DST days = normal days | **Negligible → closed** (D-G1) |
| 19 | 16.1.A hour gaps outside DST days | IT_NORD: 4 more full days + 2 Mar 2025 (23 h) not caught by the day-level check; FR: scattered hours 2021–24 | New issue D-G4 (neighbour data only) |
| 20 | CH aFRR fall-back rows for RQ3 | 3,324 rows → 831 documents; all fields constant except dates; no quantity/Point data | **No RQ3 value — confirmed** |
| 21 | Fall-back flag usability | 2022–24 history published 15 Apr 2025; all documents start 00:00 | Caveats added (Outages → thesis relevance) |

### New script: `Entsoe/Generation/check_data_gaps.py`

Repeatable, read-only gap check for both items. `--write` saves CSVs to `Entsoe/_checks/` (nothing else is written).

```
python Entsoe/Generation/check_data_gaps.py generation        # per-unit hour gaps + DST hours
python Entsoe/Generation/check_data_gaps.py fallback          # CH aFRR fall-back summary + sanity checks
python Entsoe/Generation/check_data_gaps.py all --write       # both + CSVs in Entsoe/_checks/
```

Checks whole hours only (missing single quarter-hours inside an hour are not reported). A file that ends early shows in `last_ts`, not as missing hours.

**Baseline (2026-09-28) to compare against:**

- DST hours missing: CH 1, AT 1, DE_LU 0, FR 5, IT_NORD 0.
- 2026 missing hours: FR 288 (12 full days); IT_NORD 48 (2 full days), `last_ts` 2026-07-18.
- Fall-back share of period: 2022 0.17, 2023 0.01, 2024 0.42, 2025 0.99, 2026 1.00; checks: reason `['B13']`, start hours `[0]`, revisions `[1]`, quantity `False`.

**When to run:**

1. After any Generation re-pull → `generation` (next: IT_NORD 2026, ~20 Oct).
2. After any Outages re-pull → `fallback` (the four check lines must stay at baseline; otherwise revisit RQ3 / flag design).
3. Before building the clean layer → `all --write` (gap list for flags, fall-back documents for the flag).
4. Before finalising Chapter 3 → `all` (confirm cited numbers).
5. Only if the sample is extended past Oct 2026 → after the full refresh, `all` (covers the 25 Oct 2026 DST switch).

### IT_NORD 2026 re-pull procedure (~20 Oct 2026)

First check the Transparency Platform: [Actual Generation per Generation Unit](https://transparency.entsoe.eu/generation/r2/actualGenerationPerGenerationUnit/show) → Area Italy / IT-North → a date after 18 Jul 2026 (e.g. 15 Aug and 1 Sep). Values shown → re-pull. Empty / N/A → source gap, log it and stop checking.

The pull has no area filter, so a filtered coverage CSV limits it to IT_NORD per-unit:

```
# 1. back up the current file
cp Entsoe/Generation/Data/production/actual_generation_unit/per_unit/IT_NORD/2026.parquet Entsoe/Generation/Data/production/actual_generation_unit/per_unit/IT_NORD/2026.parquet.bak
# 2. coverage list with only IT_NORD per-unit
python -c "import pandas as pd; c=pd.read_csv('Entsoe/Generation/Data/_coverage_20240601_20240701.csv'); c[(c.dataset=='actual_generation_unit')&(c.area=='IT_NORD')].to_csv('/tmp/coverage_itnord.csv', index=False)"
# 3. dry run — must list only IT_NORD per-unit 2026
python Entsoe/Generation/entsoe_generation_pull.py --coverage /tmp/coverage_itnord.csv --start 2026-01-01 --end 2026-09-01 --force --plan
# 4. real run (same command without --plan)
python Entsoe/Generation/entsoe_generation_pull.py --coverage /tmp/coverage_itnord.csv --start 2026-01-01 --end 2026-09-01 --force
# 5. check: IT_NORD 2026 last_ts later than 2026-07-18
python Entsoe/Generation/check_data_gaps.py generation
```

### Decision (recommended, to confirm with supervisor): no full ENTSO-E re-pull — fixed data cut-off 31 Aug 2026

- All ENTSO-E data is complete and verified to 31 Aug 2026; a full re-pull takes more than a day (Outages ~9 h, Balancing ~19 h at the old interval) and fixes no known gap.
- A fixed cut-off makes every cited number reproducible and matches the combined dataset (2021-01 → 2026-08).
- ENTSO-E revisions of recent months are covered by Swissgrid, which takes priority where both exist.
- Only targeted re-pulls (IT_NORD 2026 per-unit). If the sample is extended (e.g. autumn 2026), do **one** full refresh just before modelling.

---

## Pre-2021 extension (2015-01 → 2020-12) — work done 2026-09-28

Goal: match the Swissgrid auction history (2015+), so the RQ1b reforms (aFRR up/down split 2018, FCR daily/marginal pricing 2019) have drivers, not only prices. All pulls ran on the Mac; **0 errors**. The 2021+ data, manifests and coverage files were not touched.

### Structural change: German bidding-zone split (1 Oct 2018)

Before 1 Oct 2018 DE, AT and LU formed one bidding zone (`DE_AT_LU`); from 1 Oct 2018 it is `DE_LU` + `AT`. German series therefore answer on a different code before and after the split:

| Domain | Before 1 Oct 2018 | From 1 Oct 2018 |
| --- | --- | --- |
| Load, Generation | `DE` (Germany, country level) | `DE_LU` |
| Outages (15.1, 10.1 CH↔DE borders) | `DE_AT_LU` (includes Austrian units) | `DE_LU` |
| NTC CH↔DE (11.1) | `DE_AT_LU` border | `DE_LU` border |
| Flows, schedules, countertrading CH↔DE | `DE_TRANSNET` (unchanged) | `DE_TRANSNET` |

The first probe run (before the fix) silently fell back to `DE_AMPRION` (one TSO, about ¼ of Germany) for Jan 2015 — the reason for the code change below.

### Code changes (backups `*.bak_pre2021` next to every edited script)

- **All 5 probes:** Germany-wide codes (`DE`, `DE_AT_LU`) are tried **before** the single-TSO codes. Transmission probe: date-aware NTC pin (`PINNED_BORDER_CODE_PRE_SPLIT`, `pinned_border_code()`): `DE_AT_LU` for windows ending on or before 1 Oct 2018.
- **New `Entsoe/pre_split.py`:** cuts a year at the split date and joins the two parts into the usual single year file.
- **Load, Generation, Outages, Transmission pulls:** new options `--coverage-pre <csv>` and `--split` (default 2018-10-01). Without `--coverage-pre` the behaviour is unchanged (checked: 2021+ plans still 150 / 228 / 99 / 69 series). Outages: documents crossing the split are deduplicated within the year file. Transmission: if one half of 2018 errors, the whole year is marked error (not saved).
- **New `Entsoe/build_pre2021_coverage.py`:** turns the three probe windows into the coverage files that drive the pull (`_probe_pre2021/_coverage_pre_split.csv`, `_coverage_post_split.csv`; Balancing `_coverage_pre2021.csv`). Rules: pre = ok in Jan 2015 or Jun 2017; post = ok in Jun 2019 + pre series empty in Jun 2019 (codes mapped to `DE_LU`); a zonal DE series that resolved to a single TSO is dropped before the split and reset to `DE_LU` after. **Exclusions:** per-unit generation (16.1.A) outside CH (parser crash on DE/IT_NORD, slowest dataset); all DE balancing (empty on every code before 2021 → regelleistung.net).
- **New `Entsoe/Load/patch_de_loadforecast_2018q4.py`:** see finding 2 below.

### Probe results (Jan 2015, Jun 2017, Jun 2019; written to `<Domain>/Data/_probe_pre2021/`)

- **Available from 2015:** load (actual + day-, week-, year-ahead), generation per type, installed capacity, wind/solar day-ahead, NTC (day/month/year-ahead), physical flows, total schedules (A05), generation-unit outages, planned transmission outages, CH FCR weekly.
- **Not available before 2021:** aggregated bids (12.3.E), procured capacity (12.3.F), all fall-backs, congestion costs; day-ahead schedules (A01) only from 2019; DE and IT balancing; CH/AT month-ahead load forecast; CH wind/solar intraday.
- **CH week- and year-ahead load forecasts exist 2015–2020.** Together with the log entry "CH longer horizons start ~2023" this means they were published early, stopped before 2021 and came back ~2023.

### Pull results (output: `<Domain>/Data/production_pre2021/`, log `Entsoe/pull_pre2021.log`)

| Domain | Result | Runtime |
| --- | --- | --- |
| Load | 168/168 series-years non-empty | 34 min |
| Balancing | 78 series-years, 66 non-empty (CH mFRR only from 2019) | 67 min |
| Transmission | 60 ok, 9 genuinely empty (CH-FR year-ahead NTC, countertrading DE→CH and AT→CH, all congestion costs), 0 error | 3 h 53 min |
| Generation | 195/195 series-years non-empty | 70 min |
| Outages | 170 ok, 424 empty (mostly fall-backs), 0 error | 3 h 04 min |

**CH start dates (first timestamp):** actual load, load forecasts, generation per type, wind/solar forecast, FCR weekly price + volume, generation-unit outages, planned transmission outages: Jan 2015 · imbalance volumes 26 Mar 2015 · per-unit generation 29 Mar 2015 · reservoirs 22 Feb 2015 · activated balancing energy prices 1 May 2015 · generation forecast 1 Jul 2015 · aFRR weekly price + volume 28 Sep 2015 · production-unit outages Oct 2015 · **imbalance prices 31 Mar 2016** · mFRR contracted volume 2019 only. Forced transmission outages exist for 2015–2020 (sparse; probe months happened to be empty).

**Checks done:** CH load complete every year (8,760/8,784 h); DE/AT/IT_NORD load complete; FR load 1–19 h missing per year. Generation, NTC and flows join cleanly at 1 Oct 2018 (no gaps, no duplicates). **CH FCR weekly price is real 2015–2019** (median 22 → 8 EUR/MW/h; ~100 distinct values per year); the stale constant 6.26 only appears from 2020 → the "Do not use" verdict applies from 2020 on, not to 2015–2019. Luxembourg level shift in DE load (DE → DE_LU) not visible in monthly means → negligible.

### Findings

1. **CH↔DE NTC before the split is a different border.** CH→DE_AT_LU = 5,200 MW until Sep 2018 = exactly CH→DE 4,000 + CH→AT 1,200 after the split. DE_AT_LU→CH = 2,400 vs DE→CH 1,400 + AT→CH 675 (not additive). → Treat pre-split CH↔DE NTC as its own regime (flag or separate feature); do not concatenate with post-split values.
2. **DE_LU day-ahead load forecast missing 57 days in Q4 2018** (3,368 quarter-hours, first months of the new zone). Patch script `patch_de_loadforecast_2018q4.py` re-queries DE_LU, then `DE`; fills only missing timestamps, level check (≤3 % median difference), backup `2018.parquet.bak_q4patch`, provenance list `2018_q4patch_filled.csv`. **Dry run 2026-09-28:** DE_LU re-query still has none of the missing rows; `DE` fills all 3,368 (median ratio DE/DE_LU on overlap = 0.9915, i.e. Luxembourg ≈ 0.85 %) → **applied with `--write` (verified: 35,040 rows, 0 missing, 0 duplicates, 0 NaN)**. Filled rows are Germany without Luxembourg — keep the provenance flag. DE_LU actual load misses 4 hours in Oct 2018 (1, 8, 28, 30 Oct) — source gap, flag.
3. **Pre-split DE outage files include Austrian units** (`DE_AT_LU`) → overlap with the AT files. Deduplicate on unit / `doc_mrid` in the clean layer.

### Commands (run from `Master_Thesis`, venv active)

```
python Entsoe/build_pre2021_coverage.py                 # rebuild coverage files from the probes
P=Entsoe/Load/Data/_probe_pre2021                        # same pattern for Generation / Transmission / Outages
python Entsoe/Load/entsoe_load_pull.py --start 2015-01-01 --end 2021-01-01 \
  --coverage $P/_coverage_post_split.csv --coverage-pre $P/_coverage_pre_split.csv \
  --out Entsoe/Load/Data/production_pre2021 [--plan]
python Entsoe/Balancing/entsoe_balancing_pull.py --start 2015-01-01 --end 2021-01-01 \
  --coverage Entsoe/Balancing/Data/_probe_pre2021/_coverage_pre2021.csv --out Entsoe/Balancing/Data/production_pre2021
python Entsoe/Load/patch_de_loadforecast_2018q4.py [--write]
```

Why a separate output folder: the Transmission and Outages pulls decide "empty series" (`_empty.parquet`) and write series-level manifest rows from the years of the current run only. Writing 2015–2020 into the existing trees would mark series with 2021+ data as empty and overwrite verified manifests. The clean layer reads both trees.

---

## regelleistung.net pipeline — work done 2026-09-29

Goal: German (and FCR-cooperation) capacity-auction prices as benchmark / driver series. ENTSO-E has no DE balancing data before 2021 and is patchy after (Balancing next step 4). All requests ran on the Mac (the Claude workspace cannot reach regelleistung.net).

### Source

Public download API behind the regelleistung.net Datacenter (no key, no registration):

```
GET https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/<report>
    ?date=YYYY-MM-DD&exportFormat=xlsx&market=CAPACITY|ENERGY&productTypes=FCR|aFRR|mFRR
<report> = resultsoverview (tender results) | anonymousresults (+ countryCodeA2=DE) | demands
```

One request = one delivery day. A day without data returns HTTP 200 with a header-only workbook.

### Scripts (`Regelleistung/`, same probe + pull pattern as ENTSO-E)

| Script | Purpose |
| --- | --- |
| `regelleistung_probe.py` | Owns all request logic (session, 60 s timeout, retry on 429/5xx/timeouts by status code — lesson P7, xlsx sniffing). Probe mode: 2 sample days per year and task → `Data/_probe/probe_<run>.csv` + raw samples |
| `regelleistung_pull.py` | Stage 1 download: one raw file per task-day (`.xlsx`, or `.empty` marker), resumable, append-only `_download_log.csv`. Stage 2 build: parquet per task-year + `_build_manifest.csv` |

Default scope: `results:CAPACITY` for FCR, aFRR, mFRR, 2018-07-01 → 2026-08-31 (data cut-off). Anonymous bid lists, demands and the ENERGY market are available with `--only` (not pulled).

### Output

```
Regelleistung/Data/
├── _probe/probe_<run>.csv, samples/          # probe 2026-09-29
├── raw/results_CAPACITY/<FCR|aFRR|mFRR>/<year>/<YYYY-MM-DD>.xlsx | .empty
├── raw/_download_log.csv                     # one row per request
└── production/
    ├── results_CAPACITY/<FCR|aFRR|mFRR>/<year>.parquet
    └── _build_manifest.csv                   # per task-year: files, rows, schema hash, first/last delivery
```

**Columns:** `query_date` + all source columns in snake_case (e.g. `GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]` → `germany_marginal_capacity_price_eur_mw_h`), numbers converted, plus `delivery_start` / `delivery_end` (interval start/end, `datetime64[us, Europe/Zurich]`, from `DATE_FROM` + product code such as `POS_00_04`, `NEGPOS_20_24`). Each year keeps its own column set (schemas change, see below).

### Result

| Product | Empty days | First data | Rows per day | Years |
| --- | --- | --- | --- | --- |
| FCR | 365 (1 Jul 2018 – 30 Jun 2019) | 1 Jul 2019 | 1 (daily `NEGPOS_00_24`) until Jun 2020, then 6 (4h blocks); 12 on second-auction days | 2019–2026 |
| aFRR | 11 (1–11 Jul 2018) | 12 Jul 2018 | 12 (6 blocks × POS/NEG) every day | 2018–2026 |
| mFRR | 11 (1–11 Jul 2018) | 12 Jul 2018 | 12 every day | 2018–2026 |

8,952 requests: 8,565 ok, 387 empty, 0 errors; ~75 min at `--interval 0.5`. The empty days are all before the market start (12 Jul 2018 = German switch to daily 4h aFRR/mFRR auctions; FCR daily auctions from 1 Jul 2019). No missing day after the start dates; delivery times parsed for 100% of rows. The earlier weekly German tenders (2015 – mid-2018) are **not** in the Datacenter.

### Findings (to handle in the clean layer)

- **R1 — FCR second auction (`tender_number = 2`):** on 2–26 days per year (2020: 8, 2021: 5, 2022: 6, 2023: 12, 2024: 26, 2025: 2, 2026: 0) FCR has 12 rows: a second auction for the same blocks. Its prices are extreme or empty (0 EUR/MW on several days 2020–22, 3,382 EUR/MW on 28–29 Oct 2025, empty on 1 Jan 2024). Benchmark price = `tender_number == 1`; use tender 2 only as a flag.
- **R2 — aFRR/mFRR price unit change on 8 Dec 2021:** up to 7 Dec 2021 capacity prices are EUR/MW per 4h block (`*_eur_mw`), from 8 Dec 2021 EUR/MW per hour (`*_eur_mw_h`). Divide the old values by 4 before joining. Energy-price columns (`*_energy_price_eur_mwh`) only until Dec 2021 (separate energy market since Nov 2020).
- **R3 — FCR column names change on 7 Sep 2022:** `de_…`, `ch_…`, `at_…` up to 6 Sep 2022 → `germany_…`, `switzerland_…`, `austria_…` from 7 Sep 2022 (2022 file has 54 columns). At the same time `import(-)_export(+)` → `deficit(-)_surplus(+)` — **check whether the sign flips.** Countries are added over time (e.g. Denmark, Czech Republic) → schema hash changes in 2023, 2025, 2026.
- **R4 — Swiss FCR price is included** (`ch_` / `switzerland_settlementcapacity_price_eur_mw`) for every block from Jul 2019 → direct cross-check against Swissgrid `settle_ch` (e.g. 15 Jul 2019: CH 167.04 vs DE 217.66 EUR/MW for the daily product).
- **R5 — FCR 2019-07 → 2020-06 is one daily product (`NEGPOS_00_24`)**; 4h blocks from 1 Jul 2020 — matches the Swissgrid FCR reform timeline.

### Commands (run from `Master_Thesis`, venv active)

```
python Regelleistung/regelleistung_probe.py [--only all --years 2016 2020 2024]
python Regelleistung/regelleistung_pull.py --plan
python Regelleistung/regelleistung_pull.py                  # download + build (resumable)
python Regelleistung/regelleistung_pull.py --retry-errors   # only days whose last status was error
python Regelleistung/regelleistung_pull.py --build-only     # rebuild parquet from raw files
```

---

## Plan going forward — from raw data to one dataset

Order: **fix open data issues → clean layer → combined dataset → modelling views → EDA.**

1. **Close the remaining data issues.** ENTSO-E checks are done (2026-09-27/28), apart from verifying the FR-bids set-aside and the IT_NORD 2026 check (~20 Oct). Remaining: the Swissgrid list (see open items). Data cut-off 31 Aug 2026 (see *Gap-check round*).
2. **Clean layer, per domain (not started).**
   - One time convention: UTC internally, Europe/Zurich interval-start labels.
   - Clear column names, e.g. `ch_load_actual_mw`, `de_lu_solar_da_fc_mw`.
   - Coalesce forecast column variants (D-L2).
   - Treat daily/weekly block products as step functions (D5).
   - Deduplicate outages on `(doc_mrid, revision)` and convert them to MW unavailable per interval.
   - Deduplicate fall-back documents across business-type folders; build a `ch_afrr_platform_fallback` flag (per day, or per 15 min with approximate intraday timing), and mark that pre-Apr 2025 values were not known in real time.
   - Drop unreliable series: 8.1 margin, 12.3.F, CH FCR weekly **from 2020** (2015–2019 values are real), IT/DE/FR bids.
   - **Pre-2021 tree (`production_pre2021/`):** read alongside `production/`; flag pre-split CH↔DE NTC as its own regime; deduplicate `DE_AT_LU` outages against AT; flag rows filled by the Q4 2018 patch (`2018_q4patch_filled.csv`) and the 4 missing DE load hours in Oct 2018.
   - **regelleistung.net:** keep FCR `tender_number == 1` (flag second auctions, R1); convert aFRR/mFRR per-block prices before 8 Dec 2021 to per hour (÷ 4, R2); map FCR `de_`/`ch_` to `germany_`/`switzerland_` names and check the import/export vs deficit/surplus sign (R3).
   - Flag known gaps: 31 Dec 2022, ENTSO-E Jan 2026, imbalance-volume gaps, FR per-unit Feb 2026 + scattered hours 2021–24, IT_NORD per-unit full days (D-G4) and after 18 Jul 2026. Take the gap list from `check_data_gaps.py all --write`.
3. **Combined dataset.** One 15-min master table (2021-01 → 2026-08, **or 2015-01 → 2026-08 if the extension is confirmed with the supervisor**), rows = time, columns = variables. CH plus neighbours as prefixed columns. Swissgrid takes priority where both sources overlap.
4. **Modelling views.** Aggregate to each target's grain: 4h blocks and days (FCR, aFRR daily), weeks (weekly products).
5. **EDA (Phase 3).** Data quality, distributions, seasonality, structural breaks (RQ1b), relationships between drivers and prices, target definition. Most of it feeds Chapter 3.

**Design choices still to confirm** (recommended options in bold):

- Resolution: **15-min master + aggregated views**, hourly only, or 4h blocks only.
- Scope: **CH + neighbours**, or CH only.
- Build: **clean layer, then join**, or a single builder script.

---

## Open items and next steps

### ENTSO-E status

Data-integrity items 1–8 (2026-09-26), 9–17 (2026-09-27) and 18–21 (2026-09-28) are **DONE**. Remaining:

1. **\[Balancing\] Verify the FR-bids set-aside:** `--only aggregated_bids --plan` shows 2 CH series; manifest rebuilt with 198 files.
2. **\[Generation\] ~20 Oct 2026: check IT_NORD per-unit after 18 Jul on the Transparency Platform.** Data there → re-pull procedure (*Gap-check round*) + `check_data_gaps.py generation`. Not there → log as source gap, done.
3. **\[All\] Confirm the data cut-off (31 Aug 2026)** with the supervisor; no full re-pull unless the sample is extended.
4. **\[All\] Confirm the sample start (2015 vs 2021)** with the supervisor — the 2015–2020 ENTSO-E data is now on disk (*Pre-2021 extension*).
5. ~~**\[Load\] Apply the Q4 2018 patch**~~ — **DONE 2026-09-28**: 3,368 rows from `DE`; 2018 file 35,040 rows, 0 missing, 0 duplicates, 0 NaN; backup `2018.parquet.bak_q4patch`, provenance `2018_q4patch_filled.csv`.
6. **\[Generation\] Extend `check_data_gaps.py` to 2015–2020** (`production_pre2021/`) before the clean layer.

### Swissgrid (next focus)

7. **Email `sdl-ausschreibung@swissgrid.ch` (overdue; the only RQ3 lead).** Request:
   - the historical second-by-second aFRR archive;
   - anonymised provider-level activation / response-time data;
   - pre-2023 imbalance prices, **incl. 31 Dec 2022**;
   - pre-2026 control-energy and cross-border files;
   - why CH aFRR has run in platform fall-back almost continuously since 2024/25, and whether there is an official announcement;
   - **new (2026-09-28):** why the 2022–24 fall-back history was only published on 15 Apr 2025, and whether exact intraday start/end times of the fall-back periods are available (ENTSO-E documents all start at 00:00).
8. **Inspect `secondary-daily_2026-09-24.csv`.** Is it relevant for RQ3?
9. **Parse the 2026-only CSVs** (`Ausgleichsenergie-und-Regelenergie`, `Grenzfluesse`, `control-area-balance`). Weekly cumulative cost columns must be differenced.
10. **Confirm the switch to a single imbalance price (AEP)** with Swissgrid's official announcement.
11. **Check the Swissgrid mFRR 2024 gap** (about 30 days of 4h blocks missing).
12. **TRE parser** (lower priority).
13. **Imbalance prices:** re-run `--years 2026 --force` after each new monthly download.

### Regelleistung.net

14a. **Check the FCR sign convention** (`import(-)_export(+)` until 6 Sep 2022 vs `deficit(-)_surplus(+)` from 7 Sep 2022) before using the balance columns.
14b. **Cross-check CH FCR price** (regelleistung.net `switzerland_settlement…`) against Swissgrid `settle_ch` per 4h block.
14c. Optional: anonymous bid lists / demands / energy market (`--only anonymous`, `demands`, `ENERGY`) — only if needed for RQ1/RQ2.

### Then

14. **Run `check_data_gaps.py all --write`**, then **clean layer → combined dataset → views → EDA** (see plan above).
15. **\[JAO\]** Assess whether a dedicated pull is needed (NTC is now complete for all 4 CH borders via ENTSO-E 11.1).
16. **Before finalising Chapter 3:** run `check_data_gaps.py all` to confirm the cited gap numbers.

### Thesis scope updates

- **RQ1b is no longer at risk.** Auction data for 2015–2026 covers all reform transitions. Update Chapter 3 to include the aFRR/mFRR structural breaks (2018 direction split; 2025 daily blocks; 2025 mFRR merger) and the switch from dual to single imbalance pricing (2025/26). Add the CH aFRR platform fall-back (first full-day 9 Feb 2024; almost continuous 2025–26) as a further break — noting that the 2022–24 history was only published in Apr 2025.
- **RQ3 remains at risk.** There is still no confirmed access path. If the Swissgrid email and the `secondary-daily` inspection yield nothing, trigger the contingency and drop RQ3. The original deadline (~25 Jul 2026) has passed; send the email now and agree a new date with the supervisor. The ENTSO-E fall-back lead is closed (2026-09-27, re-confirmed 2026-09-28): no response-time information.
- **Schedule:** late September 2026 is week 17 of 29. EDA was planned for weeks 7–12 and model development for weeks 13–19, but data acquisition is still running. Review the remaining phases with the supervisor.
- **Chapter 3 — data description:**
  - **regelleistung.net:** German/cooperation capacity prices from 12 Jul 2018 (aFRR/mFRR) and 1 Jul 2019 (FCR); weekly tenders before that not available; aFRR/mFRR unit change 8 Dec 2021; FCR second auctions.
  - Energy Overview coverage changes (20 → 24 → 64 variables), the 2025 timestamp change, and the end of hourly vertical load in 2022.
  - Swissgrid auctions as the primary reserve-price source; ENTSO-E as cross-check (aFRR validation above).
  - DST handling: per-unit generation loses 7 h in total on 55 DST days (CH 1 h) — negligible.
  - Data cut-off 31 Aug 2026 (if confirmed with the supervisor).
  - **German bidding-zone split (1 Oct 2018):** DE_AT_LU → DE_LU + AT; codes used per period, and that pre-split CH↔DE NTC = CH↔(DE+AT) (5,200 = 4,000 + 1,200 MW).
  - **Pre-2021 CH coverage:** start dates per series (see *Pre-2021 extension*); CH imbalance prices from 31 Mar 2016; CH longer-horizon load forecasts published 2015–2020, absent 2021–22, back from ~2023.
  - **Known source gaps:** 31 Dec 2022 imbalance prices; ENTSO-E Jan 2026 imbalance prices; ENTSO-E CH FCR daily prices Nov 2021 – Apr 2022; ENTSO-E CH mFRR daily prices 2023–24; CH↔DE week-ahead NTC not published. FR per-unit generation 9 + 16–26 Feb 2026 (+ scattered hours 2021–24); IT_NORD per-unit generation after 18 Jul 2026 and 4 further full days (12 May 2022, 18 Jun 2025, 26 Jan + 9 Jul 2026); DE_LU reservoir filling not published; CH mFRR aggregated bids sparse 2021–22.

### Blocked

- **EDA (Phase 3):** blocked on the clean layer and the combined 15-min dataset.

---

## File layout reference

All pipelines follow the same two-script pattern (probe + pull) and the same output structure. One parquet per `(dataset, variant, area/target, year)`, resumable via `_pull_manifest.csv`.

```
Master_Thesis/
├── Entsoe/
│   ├── rebuild_manifest_from_disk.py        # rebuilds a manifest from the parquet files
│   ├── pre_split.py                         # DE bidding-zone split helper (--coverage-pre), 2026-09-28
│   ├── build_pre2021_coverage.py            # probe windows -> pre/post-split coverage CSVs
│   ├── pull_pre2021.log                     # log of the 2015–2020 pull
│   ├── Balancing/
│   │   ├── entsoe_balancing_probe.py
│   │   ├── entsoe_balancing_pull.py
│   │   ├── patch_ch_imbalance_20221231.py   # one-off (confirmed: no data at source)
│   │   └── Data/
│   │       ├── _coverage_<range>.csv
│   │       ├── _coverage_union.csv          # union of all probe windows (drives the pull)
│   │       ├── _probe_pre2021/              # 2015/2017/2019 probe windows + pre/post-split coverage
│   │       ├── production_pre2021/          # 2015–2020 pull (same layout as production/)
│   │       ├── _set_aside/                  # IT/DE (D6) and FR (D9) aggregated bids + README
│   │       └── production/
│   │           ├── _pull_manifest.csv
│   │           └── <dataset>/<variant>/<area>/<year>.parquet
│   ├── _checks/                             # CSVs from check_data_gaps.py --write
│   ├── Generation/     # same structure + check_data_gaps.py (read-only gap check: per-unit + fall-backs)
│   ├── Load/           # same structure
│   ├── Transmission/   # target = directed border (e.g. CH-DE)
│   └── Outages/        # year.parquet | year.empty | _empty.parquet
│
├── Regelleistung/
│   ├── regelleistung_probe.py               # request layer + coverage/schema probe
│   ├── regelleistung_pull.py                # download (raw xlsx/.empty) + parquet build
│   └── Data/
│       ├── _probe/
│       ├── raw/<report>_<market>/<product>/<year>/<date>.xlsx|.empty (+ _download_log.csv)
│       └── production/<report>_<market>/<product>/<year>.parquet (+ _build_manifest.csv)
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
- Retry logic classifies errors by HTTP status code (429 and 5xx are retried; text matching only for timeouts and dropped connections) in all 5 probes.
- Gap check (2026-09-28): `python Entsoe/Generation/check_data_gaps.py <generation|fallback|all> [--write]` — read-only; compare with the baseline in *Gap-check round*.
- Backups from 2026-09-28: `*.bak_pre2021` (all 5 probes; Load, Generation, Outages, Transmission pulls).
- Backups from 2026-09-27: `*.py.bak_classifier` (Balancing, Generation, Load, Outages probes) and `entsoe_balancing_pull.py.bak_fr_exclude`.
