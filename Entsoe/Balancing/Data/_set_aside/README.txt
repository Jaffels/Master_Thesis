Set aside 2026-09-26 (not deleted).

aggregated_bids/*/DE and aggregated_bids/*/IT: ENTSO-E 12.3.E aggregated
balancing-energy bids for Germany and Italy. Irreducibly incomplete (issue D6):
a single day exceeds the API's 100-TimeSeries cap, so days are silently
truncated. Not used in the thesis; CH (and FR) bids are complete and stay in
production/. The Balancing pull now skips these via EXCLUDE_SERIES in
entsoe_balancing_pull.py. Move a folder back into production/ to restore it.
