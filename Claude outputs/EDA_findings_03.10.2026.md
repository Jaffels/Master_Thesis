# EDA (Phase 3) – first pass, 3 Oct 2026

Scripts: `EDA/eda_common.py` + `EDA/eda_1_quality.py` … `eda_6_load.py` (todo items 6.1–6.6). They read only the master and the ex-post views. Each one writes PNG figures, CSV tables and `report_<step>.txt` to `EDA/Output/<step>/`. Only pandas / numpy / matplotlib are used. All numbers below come from the ex-post views of 3 Oct (CH Swissgrid blocks, tenders combined, partial blocks dropped). The ex-ante rerun (section 5) does not affect them.

## 1 Data quality
- 158 of 417 master data columns miss > 1 % in at least one full year of their life. The first and last year of a series are excluded from this count.
- RQ1a window: 14 feature columns are empty → drop them from RQ1a models. These are all DE-AT-LU border columns, `ch_gen_gas_mw` and the PV-TRE activation prices. 22 further columns are structural NaN (activation prices).
- Gap list: 351 NaN runs ≥ 1 day are not explained by the gap list (656 column-days). Most are `it_nord_gen_hydro_ps_cons_mw` (437 days) → this links to open point 5.4 (FR / IT_NORD pumped storage: NaN = 0?).
- Flags: `flag_ch_afrr_fallback_published_late` 42 % of 2024 and 30 % of 2025; `flag_imb_aep_informational` 50 % of 2025.
- **mFRR 4h without procurement:** 83 % (up) / 91 % (down) of the 2024 blocks have no price. Other years: 0–19 %, with no slot pattern → **the target needs a rule** (drop the blocks, or a two-part model: procured yes / no, then the price).

## 2 Prices
- Very skewed: p99 / median = 32 (mFRR up 4h) and 68 (mFRR down 4h); FCR 4h 4.9; aFRR weekly 13–17. → model log prices.
- Level regimes: 2021–23 high (aFRR down week median 30 CHF in 2022 vs 5–10 otherwise), falling from mid-2023.
- Seasonality, each price divided by its own yearly median:
  - weekly aFRR / mFRR: March–April high (up to 2.7×), July–August low (0.45–0.7×)
  - mFRR up 4h: 08 h and 16 h slots high, 00 h low (0.28×), weekends low (0.44–0.54×)
  - aFRR down 4h: 12 h slot 3.3× (solar)
  - FCR 4h: flat
- Persistence (log, 4h): same slot the day before ρ = 0.85–0.95; the previous block is weaker for aFRR down 4h (0.53). Weekly series: lag-1 ρ ≥ 0.93 → strong naive baselines for RQ2.

## 3 Structural breaks (±26 weeks, placebo = same dates one year earlier)
- **afrr_split:** sym (median 19.5) → up + down together 13.4, each side ~5.0 (×0.26). The placebo also falls (×0.33) → mostly seasonal; the break effect is small.
- **de_zone_split:** aFRR down week ×2.7 and mFRR down week ×2.0 (placebo ×1.45), mFRR up 4h ×2.05 (placebo ×0.79) → a real break for the down products and mFRR up 4h.
- **fcr_daily:** level ×1.01 (placebo ×1.35). **fcr_4h:** ×1.21. Both FCR breaks mainly change the volatility, not the level (CV FCR week 0.15–0.25 → 4h 0.6–0.8).
- **mfrr_merged (29 Sep 2025):** mFRR down 4h ×0.39 and mFRR down week ×0.51 (placebo ×1.11); mFRR up 4h ×1.23.
- **afrr_daily / afrr_fallback:** intervals too wide; no clear level change.
- **Imbalance price:** hours with 4 equal quarter-hour prices fall from 0.76 to 0.03 on 1 Jun 2022 (confirms the resolution change). **New finding:** a second step in mid-2018, when the share rises from ~0.45 to ~0.78 → add it to the regime file (open point 7.5). From 1 Jan 2026, short – long = 0.
- **aFRR fall-back share vs aFRR weekly prices:** Spearman −0.4 to −0.6. This is mostly the post-2023 decline (level effect).
- **Reporting break (CH ENTSO-E):** ENTSO-E total / Swissgrid production is 0.53–0.71 until 2024 and ~0.9 from 2025. Solar actual coverage jumps in 2020, the solar DA forecast in 2024, run-of-river in 2025 (~×7). Do not compare these columns across regimes.

## 4 Drivers (ex-post, Spearman, RQ1a window where available)
- Level correlations of the weekly series are dominated by the shared 2021–23 trend: fr_gen_oil, expl_alloc and similar columns have |ρ| ≈ 0.6 but ≈ 0 within the month. → report the within-month (month × slot) correlations as the driver evidence.
- Within month × slot:
  - FCR 4h: DE coal / gas / wind ~±0.25
  - mFRR up 4h: load DE / AT / CH and IT_NORD generation +0.39–0.40
  - mFRR down 4h: FR load −0.30
  - aFRR down 4h: DE / AT pumping consumption, TRE down offered, mFRR down offered ~±0.5–0.6. Many of these are ex-post market outcomes (offered volumes), not ex-ante drivers.
- Cross-market (weekly levels): CH FCR 4h vs FCR-coop ρ = 0.96 (it is the same price). CH mFRR down vs DE aFRR down ~0.6. CH aFRR weekly vs DE only 0.1–0.3. Week-on-week changes: only FCR co-moves (0.9); everything else ≈ 0.

## 5 Target definition
- aFRR and mFRR are pay-as-bid: `price_settle_ch` = VWAP (ρ 1.00).
- Marginal / VWAP: aFRR week 1.16–1.23 (up to 1.36 in 2026), aFRR 4h 1.02–1.06, mFRR 1.15–1.26. VWAP and marginal move together in levels (ρ 0.96–0.99). Week-to-week changes agree less for the weekly products (0.64–0.80).
- FCR: CH bids are near 0 (min / VWAP = 0; marginal / VWAP ~10). The CH price is the cooperation clearing price → **FCR target = `price_settle_ch`, not the bid VWAP.**
- **aFRR 4h is very thin:** median 1–3 bids per block and 30–50 MW awarded. The weekly product had ~250–400 MW and 10–24 bids.
- For aFRR / FCR, offered = awarded in the source (only accepted bids reported) → there is no competition measure. mFRR offered / awarded is 2–8.

## 6 CH load
- **Finding:** in Sep–Nov 2021 and all of 2022, the ENTSO-E CH "actual" load **is the day-ahead forecast** (identical in 100 % of 2022 quarter-hours). That explains r = 1.00 in 2022. The other years have r 0.70–0.90 (neighbours: 0.94–0.99).
- Swissgrid `ch_cons_mw` has the same level as the ENTSO-E actual (ratio 0.95–1.02) and r 0.94–0.96 until 2022. `ch_cons_enduse_mw` is about 11–20 % lower. `ch_tn_vertical_feedin_mw` is not a load proxy (r < 0.3).
- 2021→: the CH forecast agrees on the daily level (r 0.90) but much less on the daily shape (r within day 0.73; neighbours 0.96–0.99).
- → **Proposal:** use `ch_cons_mw` as the CH load in the views. Flag or mask `ch_load_actual_mw` for 2021-09 – 2022-12. The ENTSO-E DA forecast stays as the ex-ante input, with a note on its poor intraday shape.

## To decide / follow up
1. The mFRR 4h no-procurement rule (target definition, Chapter 3).
2. New flag `flag_ch_load_actual_is_forecast` (2021-09 – 2022-12), or switch the CH load to Swissgrid `ch_cons_mw`.
3. Regime file: the 2018 imbalance-price resolution step.
4. The aFRR target: VWAP is defensible. Marginal ≈ VWAP × 1.2 with the same dynamics. Keep the marginal price as a robustness target.
