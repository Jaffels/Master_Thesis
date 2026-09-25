# ENTSO-E Generation Data Pipeline — Status

_Last updated: 2026-09-22_

Status of the ENTSO-E **generation-data** acquisition layer for the thesis
(Swiss ancillary-service price forecasting). Scope of this document is the
ENTSO-E **Generation domain** (TR 14.1.A/B/C/D and TR 16.1.A/B&C/D) —
exogenous driver data for RQ1 (price drivers, especially VRES generation and
hydro availability) and RQ2 (forecasting features). It is a sibling to the
Load pipeline (`LOAD_PIPELINE_STATUS.md`) and the balancing pipeline
(`PIPELINE_STATUS.md`) and follows the same two-script pattern.

---

## 1. Summary

The ENTSO-E generation acquisition layer is **built and operational**. Two
scripts, structurally identical to the Load and balancing pairs:

- `entsoe_generation_probe.py` — probes which `(dataset, variant, area)` series
  return data and on which area code, plus all shared request machinery.
- `entsoe_generation_pull.py` — multi-year production pull (**2021-01 → 2026-09**),
  driven by the probe's coverage report, importing the probe as a library.

Eight article-variant combinations across five areas (CH + the four bordering
zones DE_LU / FR / IT_NORD / AT), yielding up to **228 series-years** over the
range. The pull is running / complete.

**Generation is more complex than Load** for two reasons:

1. **`query_generation_per_plant` (16.1.A) is `@day_limited`** — entsoe-py fans
   out into ~365 daily sub-requests per year. Unlike `@month_limited` endpoints,
   the `@day_limited` decorator only catches `NoMatchingDataError` per day, so a
   single bad day (DST-transition boundary producing a malformed period
   parameter, or a parser crash on one unit's XML) kills the entire year-wide
   call. The pull implements its own **per-day fallback loop** that catches both
   400 HTTP errors and parser errors per day, losing only the bad day(s) instead
   of the whole year. This makes 16.1.A pulls slow (minutes per year per area)
   but complete.
2. **All-NaN / n/e frames are expected and preserved.** Many production types
   that exist in the schema are not operated in a given country (e.g. offshore
   wind for CH). These come back as all-NaN columns. `tidy()` does not drop
   them — they are written to parquet as-is so the on-disk schema is stable
   across areas and years.

---

## 2. Components

| File | Role |
|------|------|
| `entsoe_generation_probe.py` | Coverage probe + all shared request machinery (error handling, `tidy`/`profile`, key loading, throttling, retry logic). |
| `entsoe_generation_pull.py` | Production pull. Imports the probe as a library. One parquet per `(dataset, variant, area, year)`, resumable, with a `_pull_manifest.csv`. Per-day fallback for `@day_limited` endpoints. |

> As with Load and balancing, `pull.py` contains no fetch logic of its own — it
> calls `probe.call_with_retry(...)`, `probe.tidy(...)` etc. Every request-layer
> fix lives in `probe.py`.

### Request-layer specifics (all in `probe.py`)

- **No monkeypatch needed.** Unlike Load (which required an 8.1 margin wrapper)
  and balancing (which needed parser patches), every generation article has a
  native `entsoe-py` method. No custom wrappers.
- **`_is_client_error(exc)`** — identifies 4xx HTTP errors (bad parameter / series
  not published). These are not retried.
- **`_is_parser_error(exc)`** — identifies `AttributeError` / `TypeError` /
  `KeyError` from entsoe-py's XML parser (e.g. a generation unit missing the
  `<name>` tag in the response XML). These are not retried.
- **`call_with_retry`** — retries 429/5xx/timeout with backoff; 4xx and parser
  errors are raised immediately (non-retryable).
- **Inclusive-boundary trim** — the API's inclusive end timestamp is dropped so
  consecutive years never double-count a boundary point.
- **Annual-snapshot widening** — 14.1.A / 14.1.B are `@year_limited` annual
  snapshots whose value sits at the start of the year. The probe auto-widens a
  short `--start/--end` window to the full calendar year so these are not
  false-negatives. The pull works per calendar year already and needs no
  special-casing.

---

## 3. What the pipeline provides

### 3.1 Articles covered

| Article | Description | Method | Windowing | Notes |
|---------|-------------|--------|-----------|-------|
| 14.1.A | Installed capacity per production type | `query_installed_generation_capacity` | `@year_limited`, annual | One row per year |
| 14.1.B | Installed capacity per production unit | `query_installed_generation_capacity_per_unit` | `@year_limited`, annual | Not published by every TSO |
| 14.1.C | Day-ahead aggregated generation forecast | `query_generation_forecast` (A01) | `@month_limited` | |
| 14.1.D | Wind & solar generation forecast (day-ahead) | `query_wind_and_solar_forecast` (A01) | `@month_limited` | |
| 14.1.D | Wind & solar generation forecast (intraday) | `query_intraday_wind_and_solar_forecast` | `@month_limited` | Bonus — "not limited to" |
| 16.1.A | Actual generation per generation unit | `query_generation_per_plant` | `@day_limited` | Heavy; per-day fallback for errors |
| 16.1.B&C | Actual generation per production type | `query_generation` | `@month_limited` | Includes Actual Consumption for storage types |
| 16.1.D | Water reservoirs & hydro storage | `query_aggregate_water_reservoirs_and_hydro_storage` | `@year_limited` `@paginated` | Weekly filling rate; published for CH |

### 3.2 Coverage (from probe + pull)

Status is based on the probe run and the production pull's `_pull_manifest.csv`.
"ok" = data returned; "empty" = `NoMatchingDataError` or zero rows (genuine
non-publication at source); "400" = API rejection; "parse" = entsoe-py parser
crash (recovered via per-day fallback for 16.1.A).

| Dataset | Variant | CH | DE_LU | FR | IT_NORD | AT |
|---------|---------|----|-------|----|---------|----|
| `installed_capacity` | per_type (14.1.A) | ok | ok | ok | ok | ok |
| `installed_capacity_unit` | per_unit (14.1.B) | ¹ | ¹ | ¹ | ¹ | ¹ |
| `generation_forecast` | day_ahead (14.1.C) | ok | ok | ok | ok | ok |
| `wind_solar_forecast` | day_ahead (14.1.D) | ok | ok | ok | ok | ok |
| `wind_solar_forecast` | intraday (14.1.D) | ² | ok | ok | ² | ² |
| `actual_generation` | per_type (16.1.B&C) | ok | ok | ok | ok | ok |
| `actual_generation_unit` | per_unit (16.1.A) | ³ | ok | ⁴ | ok | ok |
| `water_reservoirs` | weekly (16.1.D) | ok | ok | ok | ok | ok |

¹ Per-unit installed capacity (14.1.B): publication varies by TSO; check
  `_pull_manifest.csv` for exact per-area status.
² Intraday wind & solar forecast: not all TSOs publish this; CH has minimal
  wind/solar so this may be empty or sparse. Non-critical bonus series.
³ CH per-unit generation (16.1.A): the year-wide call returns 400 on DST
  transition days; the per-day fallback recovers ~364/365 days per year. Data
  exists for wind onshore, solar, nuclear, hydro water reservoir, hydro pumped
  storage, hydro run-of-river. Consumption side is blank (expected — CH does not
  report consumption per unit on this endpoint).
⁴ FR per-unit generation (16.1.A): parser crash on some days
  (`AttributeError: 'NoneType' object has no attribute 'text'` — an entsoe-py
  bug where a generation unit lacks a `<name>` XML tag). Per-day fallback
  recovers the vast majority of days.

### 3.3 Switzerland (core) — complete

| Series | Status | Notes |
|--------|--------|-------|
| Installed capacity per type (14.1.A) | OK | Annual snapshot, all years |
| Day-ahead generation forecast (14.1.C) | OK | Full range from 2021 |
| Wind & solar forecast day-ahead (14.1.D) | OK | CH has minimal wind/solar — values are small but present |
| Actual generation per type (16.1.B&C) | OK | Hydro, nuclear, solar, small thermal; many fuel columns are all-NaN (correct) |
| Actual generation per unit (16.1.A) | OK | ~364/365 days per year (1–2 DST days lost) |
| Water reservoirs (16.1.D) | OK | Weekly filling rate — core hydro feature for RQ1/RQ2 |

### 3.4 Neighbours (context / cross-check)

- **DE_LU / FR / IT_NORD / AT** — actual generation per type (16.1.B&C),
  generation forecast (14.1.C), wind & solar forecast (14.1.D) and water
  reservoirs (16.1.D) all present from 2021.
- **Per-unit generation (16.1.A)** — available for all neighbours; FR has
  intermittent parser errors on specific days (recovered via fallback).
- **Installed capacity (14.1.A)** — available for all five areas.

---

## 4. Known issues

### 4.1 Data / coverage

1. **D-G1 — DST-transition 400s on 16.1.A (per-unit generation).**
   `@day_limited` in entsoe-py constructs malformed period parameters on CET
   fall-back days (last Sunday of October). The per-day fallback loop catches
   these and skips 1–2 days per year. The lost days are DST transition
   boundaries — operationally insignificant for the thesis's 4-hour block
   analysis. No fix possible without patching entsoe-py itself.

2. **D-G2 — entsoe-py parser crash on FR 16.1.A.**
   `_parse_generation_timeseries` does `soup.find('name').text` without a null
   check. Some FR generation units lack the `<name>` XML tag, causing
   `AttributeError`. The per-day fallback catches this per day. Affects a small
   number of days for FR only.

3. **D-G3 — All-NaN fuel columns are expected, not errors.**
   CH does not operate offshore wind, lignite, coal, etc. These production-type
   columns come back as all-NaN in the 16.1.B&C and 14.1.A responses. They are
   preserved in parquet (not dropped) so the schema is stable across areas.
   Downstream cleaning should expect and handle them.

4. **D-G4 — Per-unit consumption side is blank for CH.**
   The consumption columns in 16.1.A (per-unit) are empty for CH. This is
   genuine — CH does not report per-unit consumption on this endpoint. Per-type
   consumption (pumped hydro) is available via 16.1.B&C.

5. **D-G5 — Intraday wind & solar forecast (14.1.D intraday) is sparse.**
   This is a bonus "not limited to" series. Not all TSOs publish it, and CH has
   minimal wind/solar capacity. Non-critical for the thesis.

6. **D-G6 — Per-unit installed capacity (14.1.B) may be partially empty.**
   Publication varies by TSO. Check the manifest for exact coverage. Not
   critical — 14.1.A (aggregated per type) is the primary installed-capacity
   source.

### 4.2 Pipeline / operational

- **P-G1 — 16.1.A is slow.** A per-year call fans out to ~365 daily sub-requests
  inside entsoe-py, and the per-day fallback (when triggered) does the same
  externally. A single area-year takes several minutes. Use `--only per_unit` to
  run it separately and `caffeinate -dims` to prevent sleep.

- **P-G2 — No request timeout (shared with Load P-L2 and balancing P4).**
  `entsoe-py` issues requests with no socket timeout. A stalled-but-open
  connection can hang indefinitely. Risk is higher on 16.1.A than on other
  endpoints due to the high request count.

- **P-G3 — `_pull_manifest.csv` is the source of truth.** Filter `rows > 0` to
  see what actually landed. Series-years with `rows = 0` and no file are
  genuine empties (API returned no data); those with a file are all-NaN frames
  (preserved as-is).

---

## 5. Relationship to the wider ENTSO-E pipeline

| Domain | Status doc | Status |
|--------|-----------|--------|
| **Load** (6.1.A/B/C/D/E, 8.1) | `LOAD_PIPELINE_STATUS.md` | Built + validated |
| **Balancing** | `PIPELINE_STATUS.md` | ~97% pulled |
| **Generation** (14.1.A/B/C/D, 16.1.A/B&C/D) | This document | Built, pull running/complete |
| **Transmission** (11.1, 12.1.F/G, 13.1.B/C) | Separate | Built, pull complete |

### Non-ENTSO-E tracks

| Track | Status | Priority |
|-------|--------|----------|
| **Swissgrid** ingestion (primary CH source — auction results, activation data) | Not built | **High** |
| **RQ3** response-time data | Unsourced — no confirmed access path | **High / at-risk** |
| **JAO** (NTC / cross-border capacity) | Not addressed | Medium |
| **EDA** (Phase 3) | Blocked on integration of all sources | — |

---

## 6. Thesis relevance

The generation data serves the thesis in three ways:

- **RQ1 (price drivers):** VRES generation (wind & solar from 16.1.B&C and
  14.1.D forecasts) is a key exogenous driver of ancillary service prices.
  Hydro reservoir filling rate (16.1.D) captures Switzerland's dual role as
  energy producer and reserve provider — the channel that distinguishes Swiss
  price formation from Germany's.
- **RQ2 (forecasting features):** Day-ahead generation forecast (14.1.C),
  wind & solar forecast (14.1.D), and installed capacity (14.1.A) are candidate
  features. Actual generation per type (16.1.B&C) provides the ground truth
  for forecast-error features.
- **Context:** Per-unit data (16.1.A) supports analysis of provider-level
  generation patterns, though it is not a primary forecasting input.

---

## 7. Next steps

1. **Verify the pull via `_pull_manifest.csv`:** filter `rows > 0`, confirm
   actual generation per type (16.1.B&C) and water reservoirs (16.1.D) landed
   for all five zones across all years.

2. **Inspect per-day fallback results for 16.1.A:** check which specific days
   were skipped (DST / parser) and confirm the loss is negligible.

3. **Pin area codes** if the probe flagged any area-level mismatches (check the
   "Pin these" output from the probe). Update `PINNED_AREA_CODE` in the probe
   before any re-pull.

4. **Integrate** generation with load, balancing, and (once built) Swissgrid
   data for EDA (Phase 3). Key joins: actual generation per type + actual load
   on the 15-min/hourly timestamp index; water reservoirs on the weekly index.

5. **Feature engineering** (downstream, not pipeline): VRES share = wind+solar
   generation / total generation; hydro availability = reservoir filling rate;
   generation forecast error = forecast − actual.

---

## 8. File / output layout

```
Entsoe/Generation/
├── entsoe_generation_probe.py     # probe + all shared request machinery
├── entsoe_generation_pull.py      # production pull (imports probe)
└── Data/
    ├── _coverage_<range>.csv                              # probe verdicts
    ├── _probe_state.json                                  # resumable probe state
    ├── <series>__<variant>__<area>__<range>.parquet       # probe samples
    └── production/
        ├── _pull_manifest.csv                             # rows / span / file per series-year
        └── <dataset>/<variant>/<area>/<year>.parquet
```
