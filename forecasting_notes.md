# Synthetic Demand Forecasting Guide
This is fictional sample material for demonstration.

## Forecast evaluation
Facility-level forecasts are evaluated using weighted absolute percentage error and mean absolute error. Small-volume facilities require separate error analysis because intermittent demand can distort percentage-based metrics.

## Model governance
Champion models are selected after comparing validation metrics and operational stability. A candidate should not replace a production model solely because of a one-day performance change. Retraining requires a documented evaluation and approval step.

## Monitoring
Track data completeness, feature distributions, prediction error, and model version. Investigate sustained drift before recalibration or retraining.
