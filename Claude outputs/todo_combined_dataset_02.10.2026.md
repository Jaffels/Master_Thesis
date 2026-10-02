# To-do: from raw data to one dataset → EDA

Created 1 Oct 2026, updated 2 Oct 2026 (sections 1–4 done: master table built). Order: **decisions → checks → clean layer → master table → views → EDA.**
All commands run from `Master_Thesis` with `.venv` active.
Reference files: `table_design_02.10.2026.md` (how the dataset is built) and `regime_and_reporting_changes_02.10.2026.md` (every break and data problem found).

---

## 1. Decisions (do first, they shape the build) — DONE 2 Oct 2026

1. ~~**Sample start: 2015 or 2021.**~~ → Master table **Jan 2015 → 31 Aug 2026**; the window per research question is a filter at the analysis stage (baseline 2021→, extended 2015→).
2. ~~**Data cut-off: 31 Aug 2026.**~~ → Confirmed; set once in `Clean/config.py`.
3. ~~**Table design.**~~ → 15-min master table for 15-min / hourly data + long tables for auction blocks, bids, events; CH + neighbours as prefixed columns; one clean script per domain, then one join script.
4. ~~**Storage format.**~~ → Parquet files, DuckDB as query layer (`Clean/db.py`; DuckDB installed).
5. ~~**Explicit-auction gaps 2015–2017.**~~ → NaN in the master table; gap handling chosen per view (keep NaN / fill single days + `_imputed` flag / start the feature later).

---

## 2. Quick checks (feed the cleaning rules) — DONE 2 Oct 2026

1. ~~**[ENTSO-E Balancing] FR-bids set-aside.**~~ → Correct: only 2 CH series; manifest = 198 files on disk.
2. ~~**[Swissgrid] mFRR 2024 gap.**~~ → Source gap: 31 days without daily mFRR auctions; same in Swissgrid's archive of 30 Dec 2025.
3. ~~**[Swissgrid] Sign of `comm_flow_net_*`.**~~ → > 0 = export from CH. Found: ENTSO-E CH↔DE schedules were pulled on the wrong level (TransnetBW) → re-pulled on `DE_LU` / `DE_AT_LU`, now = Swissgrid (99.6 %).
4. ~~**[Swissgrid] Extra TRE activations.**~~ → 976 up / 298 down quarter-hours in 2026 (probably redispatch). Flag them; TRE used for bid-side features only, never as balancing volume.
5. ~~**[regelleistung.net] FCR sign convention.**~~ → No sign flip; only the names changed on 7 Sep 2022.
6. ~~**[regelleistung.net] CH FCR price vs Swissgrid `settle_ch`.**~~ → Identical in 100 % of 13,830 blocks (after ÷ block hours).
7. ~~**[ENTSO-E] Extend `check_data_gaps.py` to 2015–2020.**~~ → Done (`--tree`); CH per-unit generation starts 26 Jun 2015.
8. ~~**[All] Gap list.**~~ → New script `Clean/build_gap_list.py` (177 series) → `Clean/Data/master/gap_list.parquet`. Confirmed source gaps added: CH imbalance prices 1 Jul – 29 Sep 2016 (ENTSO-E), aFRR 20–24 h blocks Jul/Aug 2026 (Swissgrid).

---

## 3. Clean layer (one script per domain) — DONE 2 Oct 2026

Shared rules for every domain (as built):
- Time: UTC internally (`ts_utc`, interval start), Europe/Zurich labels; complete 15-min grid of 409,052 rows.
- Column names `{area}_{variable}[_{qualifier}]_{unit}`, checked by `common.check_names()`.
- Missing values stay NaN and are listed in the gap list; **flags only for values that exist but are special** (patched, suspect, placeholder, implausible → NaN + flag). Never forward-filled across a gap.
- Down / import quantities as positive magnitudes; net quantities keep their sign.
- Shared helpers in `Clean/common.py` (`grid`, `read_entsoe`, `hourly_to_qh`, `blocks_to_qh`, `local_steps`, `regimes`, `write_clean`, `write_long`).

1. ~~**ENTSO-E Load**~~ → `clean_load.py`. 8.1 forecast margin not used (one value per year, ends 2020). Suspect-value flags (CH, AT, IT_NORD); Q4 2018 patch flag.
2. ~~**ENTSO-E Generation**~~ → `clean_generation.py`. **CH reporting breaks** (Jan–Jun 2015 NaN; jumps 1 Jan 2020 / 2024 / 2025) → `regime_ch_gen_reporting`; CH total: use Swissgrid `ch_prod_mw`. AT intraday wind placeholder → NaN.
3. ~~**ENTSO-E Balancing**~~ → `clean_balancing.py`. CH imbalance prices (= Swissgrid 99.6 % from 2023), neighbours, volumes, CH activation prices, CH bids; contracted reserves as cross-check only.
4. ~~**ENTSO-E pre-2021 tree**~~ → handled inside each ENTSO-E script (`read_entsoe` reads both trees; pre-split CH↔DE NTC and schedules in `ch_de_at_lu_*` columns; Austrian units removed from pre-split DE outages; Q4 2018 patch flag).
5. ~~**ENTSO-E Transmission**~~ → `clean_transmission.py`. Missing explicit-auction hours = NaN + gap list. NTC placeholder (10,000 MW) and impossible CH↔FR schedules (2015) → NaN + flag; doubtful schedules flagged only.
6. ~~**ENTSO-E Outages**~~ → `clean_outages.py`. Cancelled / withdrawn documents dropped; latest-created document wins per unit; aFRR fall-back as daily share + `ch_afrr_platform_fallback` + published-late flag; `outage_events` table keeps creation times for ex-ante views.
7. ~~**Swissgrid auctions**~~ → `auction_blocks.parquet` (primary reserve-price source). Kept as a block table (decision 3), not as step functions on the grid.
8. ~~**Swissgrid imbalance prices, Energy Overview, system balance, TRE**~~ → `clean_swissgrid.py`. Energy Overview mFRR = all activations (`act_all`); TRE missing days in the gap list.
9. ~~**regelleistung.net**~~ → `clean_regelleistung.py`. Prices in EUR/MW/h (÷ actual block hours); no sign flip; second-auction and CH-balance flags.
10. ~~**MeteoSwiss**~~ → `clean_meteoswiss.py`. Levels repeated, hourly amounts ÷ 4.
11. ~~**Grid frequency**~~ → `clean_frequency.py`. Robust features only; **8,095 quarter-hours frozen at 48.000 Hz** (TSO archive, mainly 7 Apr – 24 Jun 2015) → NaN + flag; 4-h features aggregated from 15 min in the views.

All clean tables are written to `Clean/Data/<domain>/` (with `_dictionary.csv` each).

---

## 4. Combined dataset (15-min master table) — DONE 2 Oct 2026

1. ~~Write `Clean/build_master.py`: join all clean tables onto the 409,052-row grid (2015-01-01 → 2026-08-31).~~
2. ~~Source priority: Swissgrid before ENTSO-E where both exist (imbalance prices from 2023, system balance 2026, CH total production); the other source as `_xchk_<source>` column only where useful.~~
3. ~~Add the regime columns (`common.regimes()`) + `ch_afrr_platform_fallback` + `regime_ch_gen_reporting`.~~
4. ~~Stack `swissgrid/auction_blocks` + `regelleistung/auction_blocks_rl` into one `master/auction_blocks.parquet`.~~
5. ~~Decide which secondary series go into the master: long-horizon load forecasts (`load_fc_long`), CH contracted reserves (cross-check), neighbour series.~~
6. ~~Validate: row count per year (DST-aware), no duplicate timestamps, names follow the grammar, NaN share per column vs gap list, regime columns without NaN.~~
7. ~~Write `master/data_dictionary.csv` from the per-domain dictionaries (column, source, unit, resolution, start/end, aggregation and availability rule, notes).~~
8. ~~Before: rerun `python Clean/build_gap_list.py --write` (new confirmed reasons added after the last run).~~
9. ~~**Swissgrid weekly blocks with several tenders.**~~ → Combine the tenders of one block **weighted by awarded volume** (`Clean/targets.py`, `combine_tenders()`), used by the views.

**Result (`python Clean/build_master.py --write`, all checks passed):**
- `master/master_15min.parquet`: **409,052 rows × 451 columns** (400 data, 17 cross-check, 20 flags, 9 regime, 3 source, 2 time key); 0.71 GB in memory, 185 MB on disk. Rows per year match the DST-aware grid; regime columns complete.
- `master/auction_blocks.parquet`: **153,792 blocks** (Swissgrid 68,562, de_regelleistung 71,352, fcr_coop 13,878); key includes `tender_series`; 8 blocks cut by the sample edges marked `partial_in_sample`.
- `master/data_dictionary.csv` (both tables), `nan_share_by_year.csv`, `nan_runs_unexplained.csv`, `build_master_report.txt`.
- Source rules as built: CH imbalance prices ENTSO-E → 2022, Swissgrid 2023 →, **2026 long = short = single price (AEP)** (`src_ch_imb_price`); CH aFRR activation from the Energy Overview, system balance in 2026 (`src_ch_afrr_act`), ENTSO-E as `_xchk_entsoe`; Energy Overview mFRR renamed `*_act_all_*`; `ch_gen_total_mw` = Swissgrid production, ENTSO-E as `_xchk_entsoe`; flows and net schedules ENTSO-E, Swissgrid as `_xchk_swissgrid`; Swissgrid `it` border → `it_nord`.
- Secondary series: `load_fc_long` **in** the master; CH contracted reserves **not** (block means as `price_xchk_entsoe` / `amount_xchk_entsoe_mw` in `auction_blocks`); all neighbour series in; `installed_capacity`, `outage_events` stay in their domain folders.
- NaN runs ≥ 1 day without a gap-list match: 351 runs in 22 columns, **none for CH** (mostly IT_NORD pumping, FR coal) → report only.
- Zero-information columns (always 0): `at_gen_oil_mw`, `de_ch_countertrade_mw`, `at_ch_countertrade_mw`, `it_nord_outage_prod_forced_mw` → drop in the views.

---

## 5. Modelling views — NEXT (`Clean/build_views.py`)

1. Aggregate to each target's grain: 4h blocks and days (FCR, aFRR daily), weeks (weekly products), by joining drivers onto `auction_blocks` (`block_start_utc ≤ ts < block_end_utc`).
2. Weather: mean for temperature/radiation, sum for precipitation and degree-hours.
3. Information-time rule (RQ2, RQ3b): use each column's availability rule from the data dictionary; lag weather to D-1 or label it as a perfect-forecast upper bound; outages from `outage_events.created_utc`; aFRR fall-back not known in real time before Apr 2025.
4. Gap handling per view (decision 5); mask TRE activation features with `flag_tre_extra_activation_*`; drop or keep `*_suspect` values per view.
5. Windows per research question as filters (e.g. RQ1b 2015→, RQ2 test 2021→, RQ3 from 31 Mar 2016 with a relative spike threshold).
6. Read `auction_blocks` through `targets.combine_tenders()` (one row per block; 488 weekly blocks combined, volume and cost preserved).
7. Read `aggregation_rule` / `availability_rule` from `master/data_dictionary.csv`; use no `_xchk_*` columns as features; drop the four zero-information columns.

---

## 6. EDA (Phase 3)

1. Data quality: coverage, gaps, flags per column (gap list + dictionary).
2. Distributions and seasonality of the reserve prices.
3. Structural breaks (RQ1b): aFRR up/down split 11 Jun 2018, FCR daily 1 Jul 2019 and 4h 1 Jul 2020, CH↔DE bidding-zone split 1 Oct 2018, imbalance-price resolution 1 Jun 2022, 2025 daily aFRR blocks (30 Sep), mFRR merger (29 Sep 2025), single imbalance price 1 Jan 2026, CH aFRR platform fall-back (first full day 9 Feb 2024, almost continuous 2025–26). Keep reporting breaks (CH ENTSO-E generation) apart from market breaks.
4. Relationships between drivers and prices.
5. Target definition. Most of this feeds Chapter 3.
6. Check the CH load series: ENTSO-E CH day-ahead forecast fits actual load worse than the neighbours (corr 0.84) → compare with Swissgrid load.

---

## 7. Open decisions and open questions

1. **Day-ahead prices:** add the free ENTSO-E 12.1.D pull (CH + 4 neighbours, 2015→)? RQ1a has no spot-price driver until EPEX data arrive.
2. **EPEX:** which countries; start of history and hourly → 15-min change of the CH products.
3. **Frequency spring 2015 (optional):** build Zenodo 5105820 for 2015 and prefer it over frozen TSO stretches in `frequency_merge.py`.
4. **Confirm:** 2026 CH imbalance long / short = single price (switch `IMB_2026_LONG_SHORT_FROM_SINGLE` in `build_master.py`; alternative: NaN and use `ch_imb_price_single_eur_mwh` only).
5. **Regime file, open points:** cause of the 1 Jun 2022 imbalance-price resolution change; March 2025 price cap (RQ3c) not found in the data; reason for the aFRR platform fall-back; whether the pre-split CH↔DE schedule includes Austria.

---

## 8. Not blocking: do alongside or later

1. **[ENTSO-E Generation] ~20 Oct 2026:** check IT_NORD per-unit after 18 Jul on the Transparency Platform. Data there → re-pull procedure (*Gap-check round*) + `python Entsoe/Generation/check_data_gaps.py generation`. Not there → log as source gap.
2. **[Swissgrid] Optional data request** to `sdl-ausschreibung@swissgrid.ch`: pre-2023 imbalance prices (incl. 31 Dec 2022 and 1 Jul – 29 Sep 2016), pre-2026 control-energy and cross-border files, missing TRE days, the extra TRE activations (purpose), the 31 days without daily mFRR auctions in 2024, the missing aFRR 20–24 h blocks (Jul/Aug 2026), and intraday start/end times of aFRR fall-back periods.
3. **[MeteoSwiss] Archived weather forecasts:** check whether the Open-Meteo historical forecast API covers 2021+ well enough for an ex-ante weather test (run on the Mac).
4. **[MeteoSwiss] RQ2 weather ablation:** model with vs. without the weather block ("no gain beyond ENTSO-E forecasts" is a reportable result).
5. **[Frequency] Licence:** read the netztransparenz terms of use; ask TransnetBW before publishing figures from the raw archive; cite Zenodo 5105820 without redistributing it.
6. **[Frequency] Glitch filter** on 1-s TSO data, only if `max_abs_df` / `std_df` are needed.
7. **[regelleistung.net] Optional:** anonymous bid lists / demands / energy market, only if needed for RQ1/RQ2.
8. **Housekeeping:** archive the `JAO/` folder; archive `Frequency/Data/raw/` zips (~2.5 GB); delete `~/sg_check`.
9. **Refreshes after new downloads:** imbalance prices `--years 2026 --force`; TRE `--months YYYY-MM --force`; system balance `--force`; MeteoSwiss `python MeteoSwiss/meteoswiss_pull.py --force` then `python MeteoSwiss/meteoswiss_build.py`; then rerun the affected `Clean/clean_*.py --write` and `Clean/build_gap_list.py --write`.
10. **Before finalising Chapter 3:** re-run `python Entsoe/Generation/check_data_gaps.py all --write`, `python Clean/build_gap_list.py --write` and `python Clean/build_master.py --write` to confirm the cited numbers.
11. **Thesis draft:** resolve the "open decision" notes (sample start, cut-off; 1.5.3, 3.1); update Table 1 / Section 3.4 (aFRR split 11 Jun 2018, imbalance-price resolution 1 Jun 2022, CH↔DE regimes, CH ENTSO-E reporting breaks, frozen frequency stretches); data appendix: one rejected mFRR bid (TRL_26_01_02, 20–24 h) corrected in Swissgrid's 2026 file of 1 Oct 2026, analysis uses the earlier download.
12. **After any clean-layer rerun:** `python Clean/build_gap_list.py --write`, then `python Clean/build_master.py --write` (the build refuses to run if the gap list is older than any clean table).
13. **Git:** commit `Clean/targets.py` (untracked on 2 Oct 2026).
