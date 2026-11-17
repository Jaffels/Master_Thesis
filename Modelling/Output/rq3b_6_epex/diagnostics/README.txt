CLASSIFIER DIAGNOSTICS (ml_diagnostics.classifier_diagnostics)
  capture_rate.csv        share of all event hours that fall into the top 1 / 2 / 5 / 10 / 20 % of hours ranked by
                          predicted risk ("if Swissgrid / a trader watched only the riskiest x % of hours, how many
                          spikes would they see?"), plus precision and lift in that top group
  metrics_by_year.csv     Brier, Brier skill vs the reference model, PR-AUC, ROC-AUC, event rate per model and year
  calibration_bins.csv    20 quantile bins of predicted probability: mean prediction vs observed rate
  monthly_rates.csv       observed event rate vs mean predicted probability per month
  plots/<group>/          reliability.png, pr_curve.png, capture_rate.png, monthly_rates.png, brier_by_year.png
