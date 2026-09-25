# ENTSO-E Outages Data Pipeline — Status

_Last updated: 2026-09-24_

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
| **Swissgrid** ingestion (FCR/aFRR/mFRR auction results, activation data) | Not built | **High — primary price data** |
| **RQ3** response-time data | Unsourced — no confirmed access path to Swissgrid activation logs | **High / at-risk** |
| **JAO** | NTC partially covered by TR 11.1; dedicated JAO source not addressed | Medium |
| **EDA** (Phase 3) | Blocked on integration of all sources | — |

---

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
2. **Deduplicate** on `(doc_mrid, revision)` before any cross-year analysis —
   a single outage spanning a year boundary is present in both year files.
3. **Build the Swissgrid ingestion layer** — FCR/aFRR/mFRR auction results and
   energy overview files are the primary price data for RQ1/RQ2 and the primary
   candidate for RQ3. This is the highest-priority outstanding task.
4. **Re-evaluate RQ3** once Swissgrid data access is confirmed. If activation
   response-time records are not available via Swissgrid, the non-empty
   fall-back records here are the next candidate but likely insufficient on
   their own.
5. **Integrate** into the EDA layer (Phase 3) alongside Load, Generation,
   Balancing and Transmission. Key join: `unavail_start/end` windows onto the
   15-min price/generation time index via an interval join.
6. **Feature engineering** (downstream): planned unavailable capacity (MW) per
   area per hour, aggregated from 15.1.A+C; net transmission outage capacity
   per border from 10.1.A+B.

---

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
