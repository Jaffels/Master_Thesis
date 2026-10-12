REVENUE DIAGNOSTICS (ml_diagnostics.revenue_diagnostics)
  revenue_monthly.csv     revenue per block, win rate and mean bid per series x strategy x month
  bid_vs_clearing.csv     per series x strategy: how far bids are from the highest accepted bid c (median ratio,
                          share of bids lost because too high, revenue left on the table when accepted)
  plots/<series>/         cumulative_revenue.png, share_of_oracle_by_year.png, win_rate_monthly.png,
                          bid_ratio_hist.png (bid / c), offset_k.png (chosen offset over time)
