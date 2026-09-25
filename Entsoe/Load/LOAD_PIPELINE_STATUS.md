# ENTSO-E Load Data Pipeline — Status

_Last updated: 2026-09-21_

Status of the ENTSO-E **load-data** acquisition layer for the thesis (Swiss
ancillary-service price forecasting). Scope of this document is the ENTSO-E
**Load domain** (TR 6.1.A/B/C/D/E and 8.1) — exogenous driver data for RQ1
(price drivers) and RQ2 (forecasting). It is a sibling to the balancing pipeline
(`PIPELINE_STATUS.md`) and follows the same two-script pattern. Swissgrid (the
primary CH source), balancing, JAO and RQ3 response-time data are tracked in
their own documents.

---

## 1. Summary

The ENTSO-E load acquisition layer is **built and validated**. Two scripts,
structurally identical to the balancing pair:

- `entsoe_load_probe.py` — probes which `(dataset, variant, area)` series return
  data and on which area code, plus all shared request machinery.
- `entsoe_load_pull.py` — multi-year production pull (**2021-01 → 2026-09**),
  driven by the probe's coverage report, importing the probe as a library.

Six articles across five areas (CH + the four bordering zones DE_LU / FR /
IT_NORD / AT). **25 of the 30 series return data**; the 5 empty ones are all
`forecast_margin` (8.1), which TP does not surface for these zones on the
standard interface. That is **150 series-years** over the range.

**Load is materially simpler than balancing:** the endpoints are cap-free and
`@month_limited` in `entsoe-py`, so there is **no 100-TimeSeries cap, no adaptive
chunking, and no offset-pagination** — a full year is one call that the library
splits into months internally. Every series also answered on its **clean
national / bidding-zone code** (no control-area slices), so — unlike balancing
(Issue D2) — **nothing needs area-pinning**.

The thesis core is fully covered: **actual load (6.1.A)** and **day-ahead
forecast (6.1.B)** are complete for all five zones back to 2021. The only gaps
are in the **longer forecast horizons** (week/month/year-ahead) for CH and AT in
early years — and those gaps are **genuine non-publication at source**, verified
against TP's own UI, not a pipeline fault.

---

## 2. Components

| File | Role |
|------|------|
| `entsoe_load_probe.py` | Coverage probe + all shared request machinery (error handling, parser patch, 8.1 margin wrapper, key loading, tidy/profile). |
| `entsoe_load_pull.py` | Production pull. Imports the probe as a library. One parquet per `(dataset, variant, area, year)`, resumable, with a `_pull_manifest.csv`. |
| `check_a32.py` | One-off diagnostic: distinguishes genuine no-data from a parser drop for a given zone/year forecast horizon. |
| `verify_ch_load.py` | One-off: reproduces TP's weekly Actual Min/Max from the pulled hourly actual load, to confirm the realised data is present. |

> As with balancing, `pull.py` contains no fetch logic of its own — it calls
> `probe.call_with_retry(...)`, `probe.tidy(...)` etc. Every request-layer fix
> lives in `probe.py`.

### Request-layer specifics (all in `probe.py`)

- **Permissive forecast parser (`_patch_load_forecast_keep_all`)** — `entsoe-py`'s
  stock `parse_loads` keeps **only** the A60/A61 (min/max) TimeSeries for the
  week/month/year-ahead horizons and silently drops anything else. Where a zone
  publishes a horizon as a **single value** under a different businessType, the
  stock parser returns an empty frame. The patch keeps A60/A61 as
  `Min/Max Forecasted Load` **and** any other series as a plain `Forecasted Load`
  column, so single-value publications survive. Day-ahead (A01) and actual (A16)
  paths are untouched.
- **8.1 margin wrapper (`query_load_forecast_margin`)** — 8.1 has no `entsoe-py`
  method; a thin `documentType=A70` raw-request wrapper is added, mirroring the
  balancing probe's parser patches. Empty responses surface as `NoMatchingDataError`.
- **Transient errors** — `call_with_retry` retries 429/5xx/599/timeout with
  backoff; "no data" (`NoMatchingDataError`) is treated as an answer, not a failure.
- **Inclusive-boundary trim** — the API's inclusive end timestamp is dropped so
  consecutive years never double-count a boundary point.

---

## 3. What the pipeline provides

### 3.1 Coverage (probe, Aug 2026 sample)

| Series | ENTSO-E | CH | DE_LU | FR | IT_NORD | AT |
|--------|---------|----|-------|----|---------|----|
| `actual_load` | 6.1.A | ok | ok | ok | ok | ok |
| `load_forecast` day-ahead | 6.1.B | ok | ok | ok | ok | ok |
| `load_forecast` week-ahead | 6.1.C | ok | ok | ok | ok | ok |
| `load_forecast` month-ahead | 6.1.D | ok | ⚠ transient¹ | ok | ok | ok |
| `load_forecast` year-ahead | 6.1.E | ok | ok | ok | ok | ok |
| `forecast_margin` | 8.1 | empty | empty | empty | empty | empty |

¹ DE_LU month-ahead hit a one-off HTTP 599 (gateway timeout) during the probe,
not a data gap; it comes through on `--retry-errors` / in the pull. The coverage
sample reflects **Aug 2026 only** — historical forecast-horizon availability
differs (see 3.3).

### 3.2 Switzerland (core) — complete and clean

| Series | Status | Notes |
|--------|--------|-------|
| Actual total load (6.1.A) | OK | **Hourly** (CH publishes load hourly, not 15-min) |
| Day-ahead forecast (6.1.B) | OK | Full range from 2021 |
| Week/Month/Year-ahead forecast (6.1.C/D/E) | Partial | **Published from ~2023**; absent 2021–2022 (genuine, see D-L1) |

### 3.3 Neighbours (context / cross-check)

- **DE_LU / FR / IT_NORD** — actual load, day-ahead, and week/month/year-ahead
  forecasts all present **from 2021**.
- **AT** — actual load and day-ahead present from 2021; **week/month/year-ahead
  forecasts absent before ~2025** (genuine non-publication).
- Resolution: **CH is hourly; DE_LU / FR / IT_NORD / AT are 15-min.**

### 3.4 Forecast-horizon availability (the one real caveat)

The week/month/year-ahead forecast horizons have **staggered publication start
dates per TSO** — CH from ~2023, AT from ~2025, DE/FR/IT from 2021. This was
verified at the raw API level (`NoMatchingDataError` on the exact
`BZN|10YCH-SWISSGRIDZ` domain the TP web UI uses; a `FR 2021` control returns
data normally). The `_pull_manifest.csv` (`rows > 0`) is the source of truth for
exact per-series coverage. **Actual load and day-ahead forecast are unaffected —
both complete for all zones back to 2021.**

---

## 4. Known issues

### 4.1 Data / coverage

1. **D-L1 — Longer forecast horizons are staggered by TSO, at source.**
   CH week/month/year-ahead forecasts start ~2023; AT ~2025. Not recoverable —
   never published. These are **secondary features**; the models lean on actual
   load and day-ahead forecast, which are complete. Don't build a feature that
   assumes CH month-ahead before 2023.
2. **D-L2 — Forecast column schema varies within a series.** A horizon comes back
   as `Min/Max Forecasted Load` where TP publishes a range and as
   `Forecasted Load` where TP publishes a single value, so a series' per-year
   files may carry either or both columns. This is the raw-pull / reconcile-later
   choice — coalesce at cleaning time, don't assume a fixed schema.
3. **D-L3 — `forecast_margin` (8.1) is empty for all zones.** 8.1 is an *annual*
   year-ahead series; the point for a year is stamped at the year start, so the
   one-month probe window never overlaps it — and it may also simply be
   unpublished for CH/neighbours. It is therefore **excluded from the pull**
   (coverage = empty). It is not in the thesis driver set and is the lowest-value
   of the six. **Recommendation: drop it**, or re-probe over a year-spanning
   window (below) to confirm before deciding.
4. **D-L4 — "6.1.A&B" and "6.1.A&C&D&E" are TP *overlay views*, not API series.**
   They pair actual load with the forecasts; 6.1.A appears in both. We pull each
   article once as a raw series, so the overlays are downstream joins on the
   timestamp index. The **Actual Min/Max** columns visible in TP's
   week/month/year views are a UI aggregation of realised load (6.1.A), which we
   hold at **full hourly resolution** — richer than the bucketed min/max.
   Per-bucket min/max can be derived if a 1:1 match to the TP table is wanted.
5. **D-L5 — CH vs neighbour resolution mismatch.** CH load is hourly; neighbours
   are 15-min. Resample/align before joining CH to the others.

### 4.2 Pipeline / operational

- **P-L1 — No cap / no pagination (advantage).** Load endpoints are cap-free and
  `@month_limited`, so none of the balancing pull's chunking / offset-pagination
  / dense-window 400/599 machinery is needed. The pull is correspondingly light
  and fast.
- **P-L2 — No request timeout (latent, shared with balancing P4).** `entsoe-py`
  issues requests with no socket timeout, so a stalled-but-open connection can
  hang indefinitely. Load volume is small (few TimeSeries, cap-free), so the risk
  is much lower than on the dense balancing bids, but the latent issue is the
  same. Fix opportunistically with a ~60 s socket timeout treated as skip-with-warning.
- **P-L3 — Coverage driven by one probed month (Aug 2026).** Same caveat as
  balancing D1: a horizon that existed only in another window wouldn't appear in
  the coverage. Mitigated here because the manifest's `rows > 0` is checked
  per-year across the full pull, so real historical gaps show up regardless.
- **P-L4 — Throttle interval.** Default `--interval 2.0`. Load is light, so
  runtime is modest; lower to `--interval 0.3` (~200 req/min, half the 400/min
  ceiling) if a re-pull needs to be fast.

---

## 5. Relationship to the wider pipeline

| Track | This doc | Status |
|-------|----------|--------|
| ENTSO-E **Load** (6.1.A/B/C/D/E, 8.1) | ✅ this | built + validated |
| ENTSO-E **Balancing** | `PIPELINE_STATUS.md` | ~97% pulled |
| **Swissgrid** ingestion (primary CH source) | separate | not built — **High** |
| **RQ3** response-time data | separate | unsourced — **High / at-risk** |
| **JAO** (NTC / cross-border) | separate | not addressed — Medium |
| **EDA** (Phase 3) | — | blocked on integration |

---

## 6. Recommended next steps

1. **Run / confirm the production pull** across the full range:
   ```bash
   cd ~/Master_Thesis/Entsoe/Load
   ../../.venv/bin/python entsoe_load_pull.py \
     --coverage Data/_coverage_20260801_20260901.csv \
     --start 2021-01-01 --end 2026-09-01
   ```
   Use `--plan` first to see the series × years grid without calling the API.
2. **Verify via `_pull_manifest.csv`** (filter `rows > 0`): confirm actual load +
   day-ahead landed for all five zones across all years, and note where the
   longer horizons begin per zone (documents D-L1 precisely for Chapter 3).
3. **Clear the DE_LU month-ahead probe blip** (optional, cosmetic on coverage):
   `entsoe_load_probe.py --retry-errors`.
4. **Decide on `forecast_margin` (8.1):** drop it, or test properly with a
   year-spanning re-probe before deciding —
   `entsoe_load_probe.py --only forecast_margin --start 2025-01-01 --end 2026-01-01`
   (writes a *separate* coverage CSV; pass that file's path to `--coverage` if you
   then pull it, so the main pull isn't redirected).
5. **(Optional) Precompute per-bucket Actual Min/Max** to mirror TP's
   week/month/year overlay tables, if a 1:1 presentation match is wanted. The
   hourly `actual_load` series already contains the underlying data.
6. **Integrate** load with balancing (and, once built, Swissgrid + ENTSO-E
   price/generation) for EDA (Phase 3).

---

## 7. File / output layout

```
Entsoe/Load/
├── entsoe_load_probe.py     # probe + all shared request machinery
├── entsoe_load_pull.py      # production pull (imports probe)
├── check_a32.py             # diagnostic: genuine-empty vs parser-drop
├── verify_ch_load.py        # diagnostic: reproduce TP Actual Min/Max
└── Data/
    ├── _coverage_<range>.csv                          # probe verdicts (drives the pull)
    ├── _probe_state.json                              # resumable probe state
    ├── <series>__<variant>__<area>__<range>.parquet   # probe samples
    └── production/
        ├── _pull_manifest.csv                         # rows / span / file per series-year
        └── <dataset>/<variant>/<area>/<year>.parquet
```
