"""Rolling-origin backtesting for time-series forecasters.

The evaluation protocol is the standard one from the forecasting literature:
train on ``[0, t)``, predict the next ``horizon`` values, compare with what
actually happened, then slide the origin forward by ``step`` and repeat.  The
model never sees a single future observation, which is exactly how a forecast
would be used in production.

The metrics reported are the ones an architect actually needs:

``mae``
    Mean absolute error in dollars.
``mape``
    Mean absolute percentage error, scale free.
``bias``
    Signed mean error.  A positive bias means the model systematically
    over-forecasts; a negative bias means it under-forecasts.  This is the
    metric that tells you whether a budget will actually be blown.
``mase``
    Mean absolute scaled error: the MAE divided by the in-sample MAE of the
    naive one-step-ahead forecast, so a value below 1.0 beats naive.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from spendforecast.series import FloatArray, UsageSeries, as_values

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from spendforecast.forecasters import Forecaster

ForecasterFactory = Callable[[], "Forecaster"]


@dataclass(slots=True, frozen=True)
class BacktestFold:
    """A single rolling-origin evaluation fold.

    Attributes:
        origin: Index of the first predicted observation.
        train_size: Number of observations used for training.
        forecast: Predicted values for the fold horizon.
        actual: The values that really happened.
        errors: ``forecast - actual`` for each step.
    """

    origin: int
    train_size: int
    forecast: tuple[float, ...]
    actual: tuple[float, ...]
    errors: tuple[float, ...]

    @property
    def horizon(self) -> int:
        """Number of steps predicted in this fold."""
        return len(self.forecast)


@dataclass(slots=True)
class BacktestResult:
    """The outcome of a rolling-origin backtest.

    Attributes:
        folds: The individual folds, oldest origin first.
        mae: Mean absolute error.
        mape: Mean absolute percentage error (``nan`` when no actual was
            non-zero).
        bias: Signed mean error; positive means over-forecasting.
        mase: Mean absolute scaled error against the naive one-step forecast.
        scale: The naive in-sample MAE used as the MASE denominator.
    """

    folds: list[BacktestFold] = field(default_factory=list)
    mae: float = float("nan")
    mape: float = float("nan")
    bias: float = float("nan")
    mase: float = float("nan")
    scale: float = float("nan")

    @property
    def n_folds(self) -> int:
        """Number of evaluated folds."""
        return len(self.folds)

    @property
    def forecasts(self) -> list[float]:
        """Every predicted value, flattened across folds."""
        return [value for fold in self.folds for value in fold.forecast]

    @property
    def actuals(self) -> list[float]:
        """Every actual value, flattened across folds."""
        return [value for fold in self.folds for value in fold.actual]

    @property
    def errors(self) -> list[float]:
        """Every error, flattened across folds."""
        return [value for fold in self.folds for value in fold.errors]

    @property
    def total_actual(self) -> float:
        """Sum of all actual values across the folds."""
        return float(sum(self.actuals))

    @property
    def total_forecast(self) -> float:
        """Sum of all predicted values across the folds."""
        return float(sum(self.forecasts))

    def __repr__(self) -> str:
        return (
            f"BacktestResult(folds={self.n_folds}, mae={self.mae:.4f}, "
            f"mape={self.mape:.2f}%, bias={self.bias:.4f}, mase={self.mase:.4f})"
        )


def naive_scale(values: FloatArray) -> float:
    """In-sample MAE of the one-step-ahead naive forecast.

    This is the denominator of the MASE scale factor: the error a trivial
    "tomorrow equals today" predictor would have made on the same data.

    Args:
        values: The observed values.

    Returns:
        The mean absolute first difference, or ``nan`` for a series shorter
        than two observations.
    """
    if values.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(values))))


def _metrics(
    errors: NDArray[np.float64], actuals: NDArray[np.float64], scale: float
) -> tuple[float, float, float, float]:
    """Compute ``(mae, mape, bias, mase)`` from raw error arrays."""
    if errors.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    non_zero = actuals != 0.0
    if np.any(non_zero):
        mape = float(np.mean(np.abs(errors[non_zero] / actuals[non_zero])) * 100.0)
    else:
        mape = float("nan")
    mase = mae / scale if scale and not np.isnan(scale) and scale > 0.0 else float("nan")
    return mae, mape, bias, mase


def rolling_origin_backtest(
    series: UsageSeries | Sequence[float],
    forecaster_factory: ForecasterFactory,
    *,
    min_train: int = 10,
    horizon: int = 1,
    step: int = 1,
) -> BacktestResult:
    """Evaluate a forecaster with a rolling-origin (expanding window) backtest.

    For each origin ``t`` the forecaster is trained on the observations
    ``[0, t)`` and asked for ``horizon`` future values, which are compared
    against the observations ``[t, t + horizon)``.  The training window always
    *expands* and always ends before the forecast origin, so the model never
    sees data from the future.

    Args:
        series: The observed values (a series or a plain sequence).
        forecaster_factory: Zero-argument callable producing a fresh, unfitted
            forecaster for each fold.
        min_train: Minimum number of training observations in the first fold.
        horizon: Number of steps predicted per fold.
        step: Distance between consecutive forecast origins.

    Returns:
        A :class:`BacktestResult` with the folds and the aggregate metrics.

    Raises:
        ValueError: If ``min_train``, ``horizon`` or ``step`` are not positive,
            or if the series is too short to produce a single fold.
    """
    if min_train < 1:
        raise ValueError(f"min_train must be positive, got {min_train}")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if step < 1:
        raise ValueError(f"step must be positive, got {step}")

    values = as_values(series)
    n = values.size
    if n == 0:
        raise ValueError("cannot backtest an empty series")
    if min_train >= n:
        raise ValueError(f"min_train ({min_train}) must be smaller than the series length ({n})")
    if min_train + horizon > n:
        raise ValueError(
            f"min_train ({min_train}) + horizon ({horizon}) exceeds the series length ({n})"
        )

    scale = naive_scale(values)
    folds: list[BacktestFold] = []
    all_errors: list[float] = []
    all_actuals: list[float] = []

    origin = min_train
    while origin + horizon <= n:
        train = values[:origin]
        model = forecaster_factory()
        model.fit(train)
        predicted = np.asarray(model.forecast(horizon), dtype=np.float64)
        actual = values[origin : origin + horizon]
        errors = predicted - actual
        folds.append(
            BacktestFold(
                origin=origin,
                train_size=origin,
                forecast=tuple(float(value) for value in predicted),
                actual=tuple(float(value) for value in actual),
                errors=tuple(float(value) for value in errors),
            )
        )
        all_errors.extend(float(value) for value in errors)
        all_actuals.extend(float(value) for value in actual)
        origin += step

    error_array = np.asarray(all_errors, dtype=np.float64)
    actual_array = np.asarray(all_actuals, dtype=np.float64)
    mae, mape, bias, mase = _metrics(error_array, actual_array, scale)
    return BacktestResult(folds=folds, mae=mae, mape=mape, bias=bias, mase=mase, scale=scale)


def select_best_forecaster(
    series: UsageSeries | Sequence[float],
    forecasters: dict[str, ForecasterFactory],
    *,
    min_train: int = 10,
    horizon: int = 1,
    step: int = 1,
    metric: str = "mae",
) -> tuple[str, dict[str, BacktestResult]]:
    """Backtest every candidate model and return the winner plus the table.

    Args:
        series: The observed values.
        forecasters: Mapping of model name to a zero-argument factory.
        min_train: Minimum number of training observations per fold.
        horizon: Number of steps predicted per fold.
        step: Distance between consecutive forecast origins.
        metric: Metric used to rank the models: ``"mae"``, ``"mape"`` or
            ``"mase"``.

    Returns:
        ``(winner_name, results)`` where ``results`` maps every model name to
        its :class:`BacktestResult`.

    Raises:
        ValueError: If ``forecasters`` is empty or ``metric`` is unknown.
    """
    if not forecasters:
        raise ValueError("at least one forecaster is required")
    if metric not in {"mae", "mape", "mase"}:
        raise ValueError(f"metric must be one of 'mae', 'mape', 'mase'; got {metric!r}")

    results: dict[str, BacktestResult] = {}
    for name, factory in forecasters.items():
        result = rolling_origin_backtest(
            series, factory, min_train=min_train, horizon=horizon, step=step
        )
        if result.n_folds > 0:
            results[name] = result
    if not results:
        raise ValueError("no candidate model could be backtested on this series")

    def score(result: BacktestResult) -> float:
        value = getattr(result, metric)
        return float("inf") if np.isnan(value) else value

    winner = min(results, key=lambda name: score(results[name]))
    return winner, results
