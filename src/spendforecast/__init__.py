"""spendforecast - predict LLM token spend overruns before the invoice arrives.

The package is organised around three questions:

1. *What will we spend?* :mod:`spendforecast.forecasters` and
   :mod:`spendforecast.backtest` forecast the next period and measure how
   accurate the forecast is with a rolling-origin backtest.
2. *When do we blow the budget?* :mod:`spendforecast.breach` turns a forecast
   into a breach date and a safe-to-spend allowance.
3. *Is today normal?* :mod:`spendforecast.anomaly` finds spikes, sustained
   level shifts and sign errors using robust statistics.

Everything runs offline on numpy alone and is fully deterministic.
"""

from spendforecast.anomaly import (
    AnomalyDetector,
    AnomalyResult,
    LeakDetector,
    NegativeSpendDetector,
    SpikeDetector,
    median_absolute_deviation,
    robust_z_scores,
    seasonal_decomposition,
)
from spendforecast.backtest import (
    BacktestFold,
    BacktestResult,
    rolling_origin_backtest,
    select_best_forecaster,
)
from spendforecast.breach import BreachPrediction, BreachPredictor
from spendforecast.forecasters import (
    AutoForecaster,
    BaseForecaster,
    CrostonForecast,
    Forecaster,
    HoltLinearForecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalHoltWinters,
)
from spendforecast.series import UsagePoint, UsageSeries, bucket_start, normalize_period

__version__ = "0.1.0"

__all__ = [
    "AnomalyDetector",
    "AnomalyResult",
    "AutoForecaster",
    "BacktestFold",
    "BacktestResult",
    "BaseForecaster",
    "BreachPrediction",
    "BreachPredictor",
    "CrostonForecast",
    "Forecaster",
    "HoltLinearForecaster",
    "LeakDetector",
    "LinearTrendForecaster",
    "MovingAverageForecaster",
    "NaiveForecaster",
    "NegativeSpendDetector",
    "SeasonalHoltWinters",
    "SpikeDetector",
    "UsagePoint",
    "UsageSeries",
    "bucket_start",
    "median_absolute_deviation",
    "normalize_period",
    "robust_z_scores",
    "rolling_origin_backtest",
    "seasonal_decomposition",
    "select_best_forecaster",
]
