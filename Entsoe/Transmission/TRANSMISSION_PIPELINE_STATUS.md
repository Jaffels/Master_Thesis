# ENTSO-E Transmission Data Pipeline — Status

_Last updated: 2026-09-23_

Status of the ENTSO-E **Transmission-domain** acquisition layer for the thesis
(Swiss ancillary-service price forecasting). Scope of this document is TR 11.1,
12.1.F, 12.1.G, 13.1.B and 13.1.C — cross-border capacity, schedule, flow and
congestion-management data. It is a sibling to the Load, balancing, and
Generation pipelines and follows the same two-script pattern.

---

## 1. Summary

The ENTSO-E Transmission acquisition layer is **built, debugged, and complete**.
Two scripts, structurally identical to the Load/Generation pairs:

- `entsoe_transmission_probe.py` — probes which `(dataset, variant, target)`
  series return data and on which area/border code, plus all shared request
  machinery, including two custom endpoint wrappers entsoe-py doesn't ship.
- `entsoe_transmission_pull.py` — multi-year production pull
  (**2021-01 → 2026-09**), driven by the probe's coverage report, importing the
  probe as a library.

Nine article-variant combinations, mostly iterated over 8 directed CH borders
(CH↔DE/FR/IT_NORD/AT), one over 5 single areas. Final pull result:
**56 series-years ok, 13 genuinely empty, 0 error.**

**Transmission is structurally different from Load/Generation in two ways:**

1. **Every product is directional or area-scoped, not single-area.** Most
   articles are cross-border flows (`country_code_from` / `country_code_to`),
   so the probe/pull iterate directed borders (`CH-DE`, `DE-CH`, …) instead of
   single areas. One article (13.1.C) is the opposite — a single-area figure
   that the API rejects if given a border pair at all.
2. **Two articles (13.1.B, 13.1.C) have no entsoe-py method.** They were added
   as wrappers on top of entsoe-py's own raw request machinery, and needed two
   rounds of live-API debugging to get the request shape and pagination right
   (see §4).

---

## 2. Components

| File | Role |
|------|------|
| `entsoe_transmission_probe.py` | Coverage probe + all shared request machinery (error handling, `tidy`/`profile`, key loading, throttling, retry logic) + the two A91/A92 wrappers. |
| `entsoe_transmission_pull.py` | Production pull. Imports the probe as a library. One parquet per `(dataset, variant, target, year)`, resumable, with a `_pull_manifest.csv`. |
| `entsoe_congestion_diag.py` | One-off diagnostic (not part of the pipeline) used to extract the exact `<Reason>` codes from ENTSO-E's 400 responses while debugging the A91/A92 wrappers. Can be deleted. |

> As with Load/Generation, `pull.py` contains no fetch logic of its own — it
> calls `probe.call_with_retry(...)`, `probe.tidy(...)`, `probe.targets_for(...)`
> etc. Every request-layer fix lives in `probe.py`.

### Request-layer specifics (all in `probe.py`)

- **`DE` neighbour resolves to two different EICs depending on product.**
  `CTA|DE(TransnetBW)` (`DE_TRANSNET`) is the control area that physically
  borders Switzerland; `DE_LU` is the German-Luxembourg bidding zone. Physical
  flows and scheduled exchanges resolve to `DE_TRANSNET`; NTC resolves to
  `DE_LU`. The probe tries `["DE_TRANSNET", "DE_LU", "DE_AMPRION", "DE_TENNET",
  "DE_50HZ"]` in order and records whichever one actually answers — **pin the
  resolved code per product in `PINNED_BORDER_CODE`** rather than relying on
  fallthrough order for the production pull.
- **`query_countertrading` (A91) and `query_congestion_costs` (A92) are custom
  wrappers** — entsoe-py ships neither. Built on `_base_request` (for the
  `NoMatchingDataError` empty/error split) plus the library's generic
  timeseries parser. See §4 for the request-shape bugs found and fixed.
- **`scope` field on `Task`** distinguishes border-scope products (`from !=
  to`, most articles) from area-scope products (`from == to`, 13.1.C only).
  `targets_for(task)` returns directed borders or single areas accordingly —
  a diagonal-only combo list for area scope, so a fallback code is never
  paired against a *different* fallback code on the other side.
- **`call_with_retry`** — retries 429/5xx/timeout with backoff, classified by
  the response's actual **HTTP status code**, not substring-matching the error
  message. (An earlier version substring-matched `"502"`/`"503"` etc. in the
  message text, which spuriously flagged any **February** request — the period
  string `202502…` contains `"502"` — as transient and retried it twice before
  failing. Fixed.)
- **`describe_error`** — extracts the ENTSO-E acknowledgement's `<Reason
  code=…><text>` instead of head-truncating the raw XML body, so the actual
  rejection reason (not just `<mRID>…`) is visible in logs and the manifest.
- **Inclusive-boundary trim** — the API's inclusive end timestamp is dropped
  (`[start, end)`) so consecutive years, and sub-windows produced by overflow
  splitting, never double-count a boundary point.

---

## 3. What the pipeline provides

### 3.1 Articles covered

| Article | Description | Method | Scope | Windowing | Notes |
|---------|-------------|--------|-------|-----------|-------|
| 11.1 | Forecasted transfer capacity (NTC), day/week/month/year-ahead | `query_net_transfer_capacity_{horizon}` | border | native, `@year_limited` | 4 horizon variants |
| 12.1.F | Scheduled commercial exchanges — total (A05) + day-ahead (A01) | `query_scheduled_exchanges` | border | native | 2 variants |
| 12.1.G | Cross-border physical flows | `query_crossborder_flows` | border | native | |
| 13.1.B | Countertrading | `query_countertrading` (custom wrapper, A91) | border | `@month_limited` + overflow-split to 1h | No native method; 100-instance API cap |
| 13.1.C | Costs of congestion management | `query_congestion_costs` (custom wrapper, A92) | **area** | `@month_limited` + overflow-split to 1h | No native method; per-area, not per-border |

### 3.2 Coverage (from probe + final pull manifest)

| Dataset | Variant | CH-DE / DE | CH-FR | CH-IT_NORD | CH-AT | Notes |
|---------|---------|:---:|:---:|:---:|:---:|-------|
| `ntc` | dayahead | ok | ok | ok | ok | all 4 horizons ok all borders |
| `ntc` | weekahead | ok¹ | ok | ok | ok | ¹ DE weekahead genuinely empty (not published at that horizon) |
| `ntc` | monthahead | ok | ok | ok | ok | |
| `ntc` | yearahead | ok | ok | ok | ok | |
| `scheduled_exchanges` | total_A05 | ok | ok | ok | ok | resolves to `DE_TRANSNET` |
| `scheduled_exchanges` | dayahead_A01 | ok | ok | ok | ok | resolves to `DE_TRANSNET` |
| `physical_flows` | A11 | ok | ok | ok | ok | resolves to `DE_TRANSNET` |
| `countertrading` | A91 | ok | ok | ok | ok | complete after overflow-split + retry fixes |
| `congestion_costs` | A92 | empty² | empty² | empty² | empty² | ² genuinely empty for all 5 areas incl. CH |

Final tally across the full 2021–2026 pull: **56 ok, 13 empty (saved as
`_empty.parquet`), 0 error.**

### 3.3 Switzerland borders (core) — complete

| Series | Status | Notes |
|--------|--------|-------|
| NTC, all 4 horizons | OK | 8 borders × 4 horizons, full range |
| Scheduled exchanges, total + day-ahead | OK | 15-min resolution, `DE_TRANSNET` for the German leg |
| Physical flows | OK | 15-min for DE, hourly for FR/IT_NORD/AT |
| Countertrading | OK | Recovered via sub-hour overflow splitting on dense days (e.g. 2025-02-28) |
| Congestion costs | Empty (genuine) | Confirmed via live-API diagnostic: CH has no A92 data at any resolution tried |

### 3.4 Neighbours (context / cross-check)

- **DE / FR / IT_NORD / AT** — NTC, scheduled exchanges, physical flows and
  countertrading all present from 2021, both directions.
- **Congestion costs** — genuinely empty for all 5 areas (CH, DE_LU, FR,
  IT_NORD, AT), not just CH.

---

## 4. Known issues (resolved)

Unlike the Generation doc, none of these are open — they're logged here because
each one changed the wrapper's request shape or the probe's error-classification
logic, and a future re-pull or a similar wrapper for another domain should not
rediscover them.

1. **D-T1 — German bidding-zone vs. control-area split (`CTA|DE(TransnetBW)`).**
   Physical/commercial series and capacity series publish under different EICs
   for the same physical border. Fixed by trying `DE_TRANSNET` before `DE_LU`
   and recording which one resolves per product.

2. **D-T2 — A91/A92 have no entsoe-py method.** Added as wrappers on
   `_base_request` + the generic parser. Two follow-on bugs surfaced only once
   live requests were made (not visible from the library source alone):
   - **A91 100-instance cap.** Countertrading publishes one TimeSeries per MTU;
     a month exceeds the API's 100-instance response limit. entsoe-py's own
     `PaginationError` mapping does **not** recognize ENTSO-E's actual wording
     (`"exceeds the allowed maximum"`), so the built-in pagination decorators
     never fire and it surfaces as a bare `HTTPError`. Fixed with a
     `window_on_overflow` decorator that halves the request period on that
     specific error text, down to a 1-hour floor (one very dense day —
     2025-02-28 — needed the full sub-hour split).
   - **A92 "Area EICs shall be the same".** Congestion costs is a per-area
     figure, not a border product; the API rejects any request where
     `in_Domain != out_Domain`. Fixed by giving `Task` a `scope` field and
     iterating single areas (diagonal-only candidate codes) for this article.

3. **D-T3 — `"cannot reindex on an axis with duplicate labels"` on countertrading.**
   Two compounding causes: (a) the wrapper's boundary trim was inclusive, so
   `window_on_overflow`'s sub-window stitching double-counted the pivot
   timestamp; (b) a document with multiple `businessType` columns of differing
   coverage, one containing a repeated MTU, crashed on `DataFrame` assembly.
   Fixed with a half-open `[start, end)` trim and per-column
   `duplicated(keep="last")` dedup before assembly.

4. **D-T4 — `RuntimeError: No active exception to reraise`.** A bare `raise`
   statement in `window_on_overflow` sat outside its `except` block at the
   1-hour floor, so Python had nothing to re-raise. Fixed by capturing the
   overflow exception explicitly and re-raising it by name.

5. **D-T5 — False "transient" retries on any February request.** The retry
   classifier substring-matched `"502"`/`"503"`/etc. in the exception message;
   the period parameter `202502…` (February) contains `"502"` as a substring,
   so every February 400 was retried twice before correctly failing. Fixed by
   classifying transience from the response's actual HTTP status code.

6. **D-T6 — Error bodies were truncated before the useful part.** ENTSO-E's
   acknowledgement XML puts `<Reason><code>/<text>` near the *end* of the
   document, after `<mRID>`/dates/participants; a head-truncated body preview
   always showed the useless preamble. Fixed by extracting `<code>`/`<text>`
   explicitly via regex before falling back to a head slice.

All six were found and fixed through live-API round-trips (not solvable from
static code review alone), each verified against synthetic XML fixtures before
being applied to the real pull.

### 4.1 Non-issues confirmed, not bugs

- **Congestion costs (13.1.C) genuinely empty for CH.** Confirmed via a direct
  diagnostic script that tried `businessType` variants (`B03`, `B04`, `A46`)
  against multiple areas — all returned a clean "No matching data found", never
  a rejection. This matches the thesis's own expectation that congestion-related
  series may be unavailable for the Swiss market.
- **NTC week-ahead empty for the DE border only.** A genuine non-publication at
  that horizon for that border, not a code-resolution failure (day/month/year
  horizons all resolve fine on the same border).

---

## 5. Relationship to the wider ENTSO-E pipeline

| Domain | Status doc | Status |
|--------|-----------|--------|
| **Load** (6.1.A/B/C/D/E, 8.1) | `LOAD_PIPELINE_STATUS.md` | Built + validated |
| **Balancing** | `PIPELINE_STATUS.md` | ~97% pulled |
| **Generation** (14.1.A/B/C/D, 16.1.A/B&C/D) | `GENERATION_PIPELINE_STATUS.md` | Built, pull running/complete |
| **Transmission** (11.1, 12.1.F/G, 13.1.B/C) | This document | **Built, pull complete (56 ok / 13 empty / 0 error)** |
| **Outages** (IFs IN 7.2, IF mFRR 3.11, IF aFRR 3.10, TR 10.1.A/B/C, 15.1.A/B/C/D) | Not yet written | Scripts exist at `Entsoe/Outages/`; status doc not yet produced |

### Non-ENTSO-E tracks

| Track | Status | Priority |
|-------|--------|----------|
| **Swissgrid** ingestion (primary CH source — auction results, activation data) | Not built | **High** |
| **RQ3** response-time data | Unsourced — no confirmed access path | **High / at-risk** |
| **JAO** (NTC / cross-border capacity) | Partially covered by TR 11.1 above; dedicated JAO source not addressed | Medium |
| **EDA** (Phase 3) | Blocked on integration of all sources | — |

---

## 6. Thesis relevance

The Transmission data serves the thesis in two ways:

- **RQ1 (price drivers):** cross-border flows, scheduled exchanges and NTC on
  the CH-DE/FR/IT_NORD/AT borders are candidate exogenous regressors — the
  thesis background specifically names interconnector flows and cross-border
  price signals as drivers of Swiss ancillary-service pricing, and NTC
  utilization (scheduled exchange ÷ NTC) is a natural derived feature.
- **RQ1b (structural breaks / market design):** countertrading volumes are a
  secondary indicator of congestion-driven redispatch activity around the
  2019–2020 reform period, complementing the price-formation analysis.
- **Congestion costs (13.1.C)** were explored as a possible driver but are
  confirmed genuinely unavailable for CH at the ENTSO-E level — this rules the
  series out rather than leaving it as an open question, and matches the
  thesis's own anticipation of this gap.

---

## 7. Next steps

1. **Pin the resolved border codes** in `PINNED_BORDER_CODE` (physical
   flows/scheduled exchanges → `DE_TRANSNET`, NTC → `DE_LU`) before any future
   re-pull, so a fallback ordering change in entsoe-py or the API can't silently
   swap the level used.
2. **Integrate** Transmission series with Load, Generation, and (once built)
   Swissgrid data for EDA (Phase 3). Key joins: physical flows / scheduled
   exchanges on the 15-min timestamp index; NTC on its own (coarser) horizon
   index.
3. **Feature engineering** (downstream, not pipeline): NTC utilization rate
   (scheduled exchange ÷ NTC by border/hour); net physical flow direction and
   magnitude per border; countertrading volume as a congestion-activity proxy.
4. **Delete `entsoe_congestion_diag.py`** — its job (finding the A91/A92
   request shape) is done and its findings are folded into the wrappers.
5. **Write an Outages status doc**, mirroring this structure, once that
   domain's pull is confirmed complete.

---

## 8. File / output layout

```
Entsoe/Transmission/
├── entsoe_transmission_probe.py   # probe + shared request machinery + A91/A92 wrappers
├── entsoe_transmission_pull.py    # production pull (imports probe)
└── Data/
    ├── _coverage_<range>.csv                              # probe verdicts
    ├── _probe_state.json                                  # resumable probe state
    ├── <series>__<variant>__<target>__<range>.parquet     # probe samples
    └── production/
        ├── _pull_manifest.csv                             # rows / span / file per series-year
        └── <dataset>/<variant>/<target>/<year>.parquet     # or _empty.parquet if genuinely empty
```
