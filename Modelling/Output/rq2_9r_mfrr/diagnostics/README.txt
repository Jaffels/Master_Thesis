FORECAST DIAGNOSTICS (ml_diagnostics.forecast_diagnostics)
Errors are prediction - actual (positive = over-forecast), EUR/MWh, test period only.
  summary_by_model.csv     per series x model: n, MAE, RMSE, bias, median error, error quantiles, share over-forecast,
                           R2 and correlation with the actual, share of total |error| caused by the worst 1 % / 5 % of blocks
  fold_metrics.csv         per series x refit fold x model: test window, n_train, MAE, bias, skill vs the baseline
                           -> does accuracy drift with time since the last refit or across regimes?
  error_breakdown.csv      MAE / bias per year, quarter, delivery slot, weekday and every regime_* column present
  worst_errors.csv         the 25 largest errors per series (best model), with every model's forecast next to them
  quantile_calibration.csv observed share of actuals below each predicted quantile (should equal the level), per model
                           and year; plus pinball loss per level
  importance_*.csv         feature importance per model (mean / std over folds, how often in the top 20)
  plots/<series>/          timeseries.png (daily mean actual vs forecasts), zoom_last_8w.png (block level, with the
                           80 % / 90 % interval of the first quantile model), rolling_mae.png (90-day rolling MAE),
                           mae_by_year.png, mae_by_slot.png (4h only), scatter.png (forecast vs actual, asinh axes),
                           error_hist.png, quantile_calibration.png, importance_<model>.png
