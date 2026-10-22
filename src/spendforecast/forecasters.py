"""Forecasting models for spend series.

Every forecaster implements the same tiny contract:

``fit(series)``
    Learn from a :class:`~spendforecast.series.UsageSeries` (or a plain
    sequence of numbers) and return ``self``.
``forecast(horizon)``
    Return ``horizon`` future values as a ``list[float]``.
``fitted_values``
    The in-sample one-step-ahead predictions, useful for residual diagnostics.

The models are implemented from scratch on top of numpy: Holt's linear trend
(double exponential smoothing), additive Holt-Winters (triple exponential
smoothing), ordinary least squares and Croston's method for intermittent
demand.  No forecasting library is wrapped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from spendforecast.backtest import BacktestResult, rolling_origin_backtest
from spendforecast.series import FloatArray, UsageSeries, as_values

ForecasterFactory = Callable[[], "Forecaster"]


@runtime_checkable
class Forecaster(Protocol):
    """The contract shared by every forecasting model in this package."""

    def fit(self, series: UsageSeries | Sequence[float]) -> Forecaster:
        """Learn the model from ``series`` and return ``self``."""
        ...

    def forecast(self, horizon: int) -> list[float]:
        """Predict ``horizon`` future values."""
        ...

    @property
    def fitted_values(self) -> FloatArray:
        """In-sample one-step-ahead predictions."""
        ...


class BaseForecaster(ABC):
    """Shared plumbing for the forecasting models.

    Subclasses implement :meth:`_fit` and :meth:`_forecast`; this base class
    handles input validation, storing the training values and exposing the
    in-sample fitted values.
    """

    def __init__(self) -> None:
        self._values: FloatArray = np.zeros(0, dtype=np.float64)
        self._fitted: FloatArray = np.zeros(0, dtype=np.float64)
        self._is_fitted = False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fitted={self._is_fitted})"

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._is_fitted

    @property
    def fitted_values(self) -> FloatArray:
        """In-sample one-step-ahead predictions from the last :meth:`fit` call.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before use")
        return self._fitted

    def fit(self, series: UsageSeries | Sequence[float]) -> BaseForecaster:
        """Learn the model from ``series``.

        Args:
            series: A :class:`~spendforecast.series.UsageSeries` or a sequence
                of cost values.

        Returns:
            ``self``, so that ``Forecaster().fit(series)`` chains.

        Raises:
            ValueError: If the series is empty.
        """
        values = as_values(series)
        if values.size == 0:
            raise ValueError("cannot fit a forecaster on an empty series")
        self._values = values
        self._fit(values)
        self._is_fitted = True
        return self

    def forecast(self, horizon: int) -> list[float]:
        """Predict ``horizon`` future values.

        Args:
            horizon: Number of steps to predict; must not be negative.

        Returns:
            The predicted values, oldest first.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If ``horizon`` is negative.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before forecasting")
        if horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {horizon}")
        if horizon == 0:
            return []
        return [float(value) for value in self._forecast(horizon)]

    @abstractmethod
    def _fit(self, values: FloatArray) -> None:
        """Model specific fitting logic operating on raw values."""

    @abstractmethod
    def _forecast(self, horizon: int) -> FloatArray:
        """Model specific forecasting logic."""


class NaiveForecaster(BaseForecaster):
    """Naive forecaster: repeat the last observation (or the last season).

    The seasonal variant repeats the most recent full season, which is a
    surprisingly strong baseline whenever demand has a weekly rhythm.
    """

    def __init__(self, *, seasonal_period: int | None = None) -> None:
        """Create the forecaster.

        Args:
            seasonal_period: Length of the season to repeat. ``None`` uses the
                plain last-value naive forecast.

        Raises:
            ValueError: If ``seasonal_period`` is not a positive integer.
        """
        super().__init__()
        if seasonal_period is not None and seasonal_period < 1:
            raise ValueError(f"seasonal_period must be positive, got {seasonal_period}")
        self._seasonal_period = seasonal_period

    def _fit(self, values: FloatArray) -> None:
        n = values.size
        self._fitted = np.empty(n, dtype=np.float64)
        if self._seasonal_period is None or n < 2:
            self._fitted[0] = values[0]
            if n > 1:
                self._fitted[1:] = values[:-1]
            return
        m = self._seasonal_period
        self._fitted[:m] = values[0]
        self._fitted[m:] = values[:-m]

    def _forecast(self, horizon: int) -> FloatArray:
        values = self._values
        n = values.size
        if self._seasonal_period is None or n < self._seasonal_period:
            return np.full(horizon, values[-1], dtype=np.float64)
        m = self._seasonal_period
        base = n - m
        return np.array([values[base + (step % m)] for step in range(horizon)], dtype=np.float64)


class MovingAverageForecaster(BaseForecaster):
    """Forecast the mean of the most recent ``window`` observations."""

    def __init__(self, window: int = 7) -> None:
        """Create the forecaster.

        Args:
            window: Number of trailing observations to average.

        Raises:
            ValueError: If ``window`` is not positive.
        """
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be positive, got {window}")
        self._window = window

    def _fit(self, values: FloatArray) -> None:
        n = values.size
        window = min(self._window, n)
        self._fitted = np.empty(n, dtype=np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(values)))
        for index in range(n):
            start = max(0, index - window)
            self._fitted[index] = (cumulative[index] - cumulative[start]) / max(1, index - start)

    def _forecast(self, horizon: int) -> FloatArray:
        tail = self._values[-min(self._window, self._values.size) :]
        return np.full(horizon, float(tail.mean()), dtype=np.float64)


class HoltLinearForecaster(BaseForecaster):
    """Holt's linear trend method (double exponential smoothing).

    The recursions are::

        l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
        b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
        y_hat_{t+h} = l_t + h * b_t

    ``l`` is the level (a smoothed version of the current value) and ``b`` is
    the trend (a smoothed version of the first difference).
    """

    def __init__(self, *, alpha: float = 0.3, beta: float = 0.1) -> None:
        """Create the forecaster.

        Args:
            alpha: Level smoothing constant in ``(0, 1]``.
            beta: Trend smoothing constant in ``[0, 1]``.

        Raises:
            ValueError: If the smoothing constants are out of range.
        """
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self._alpha = alpha
        self._beta = beta
        self._level = 0.0
        self._trend = 0.0

    @property
    def level(self) -> float:
        """The smoothed level of the last fit."""
        return self._level

    @property
    def trend(self) -> float:
        """The smoothed trend of the last fit."""
        return self._trend

    def _fit(self, values: FloatArray) -> None:
        alpha = self._alpha
        beta = self._beta
        n = values.size
        level = float(values[0])
        trend = float(values[1] - values[0]) if n > 1 else 0.0
        fitted = np.empty(n, dtype=np.float64)
        fitted[0] = level
        for index in range(1, n):
            fitted[index] = level + trend
            previous_level = level
            level = alpha * float(values[index]) + (1.0 - alpha) * (level + trend)
            trend = beta * (level - previous_level) + (1.0 - beta) * trend
        self._level = level
        self._trend = trend
        self._fitted = fitted

    def _forecast(self, horizon: int) -> FloatArray:
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        return self._level + steps * self._trend


class SeasonalHoltWinters(BaseForecaster):
    """Additive Holt-Winters (triple exponential smoothing) with season ``m``.

    The recursions are::

        l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
        b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
        s_t = gamma * (y_t - l_t) + (1 - gamma) * s_{t-m}
        y_hat_{t+h} = l_t + h * b_t + s_{t+h-1}

    The seasonal indices are seeded from the first season (de-meaned) and are
    then updated in place, so ``s_{t-m}`` is always the most recent value for
    the seasonal phase of ``t``.
    """

    def __init__(
        self, *, period: int = 7, alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.2
    ) -> None:
        """Create the forecaster.

        Args:
            period: Length of the season (7 for weekly seasonality of daily data).
            alpha: Level smoothing constant in ``(0, 1]``.
            beta: Trend smoothing constant in ``[0, 1]``.
            gamma: Seasonal smoothing constant in ``[0, 1]``.

        Raises:
            ValueError: If the period or the smoothing constants are out of range.
        """
        super().__init__()
        if period < 2:
            raise ValueError(f"period must be at least 2, got {period}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self._period = period
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._level = 0.0
        self._trend = 0.0
        self._seasonal: FloatArray = np.zeros(0, dtype=np.float64)

    @property
    def period(self) -> int:
        """Length of the season used by the model."""
        return self._period

    @property
    def level(self) -> float:
        """The smoothed level of the last fit."""
        return self._level

    @property
    def trend(self) -> float:
        """The smoothed trend of the last fit."""
        return self._trend

    @property
    def seasonal_indices(self) -> FloatArray:
        """The current seasonal index for each phase of the season."""
        return self._seasonal

    def _fit(self, values: FloatArray) -> None:
        alpha = self._alpha
        beta = self._beta
        gamma = self._gamma
        n = values.size
        m = self._period

        seasonal = np.zeros(m, dtype=np.float64)
        seed = values[: min(m, n)]
        season_mean = float(seed.mean())
        for phase in range(m):
            if phase < n:
                seasonal[phase] = float(values[phase]) - season_mean

        if n > m:
            level = season_mean
            trend = (float(values[m : 2 * m].mean()) - season_mean) / m
        else:
            level = season_mean
            trend = 0.0

        fitted = np.empty(n, dtype=np.float64)
        for index in range(n):
            phase = index % m
            used = float(seasonal[phase])
            fitted[index] = level + trend + used
            previous_level = level
            level = alpha * (float(values[index]) - used) + (1.0 - alpha) * (level + trend)
            trend = beta * (level - previous_level) + (1.0 - beta) * trend
            if index >= m:
                seasonal[phase] = gamma * (float(values[index]) - level) + (1.0 - gamma) * used

        self._level = level
        self._trend = trend
        self._seasonal = seasonal
        self._fitted = fitted

    def _forecast(self, horizon: int) -> FloatArray:
        n = self._values.size
        m = self._period
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        phases = (n - 1 + np.arange(1, horizon + 1)) % m
        return self._level + steps * self._trend + self._seasonal[phases]


class LinearTrendForecaster(BaseForecaster):
    """Ordinary least squares regression of cost on time.

    The model is ``y_hat_t = slope * t + intercept`` fitted with
    :func:`numpy.linalg.lstsq`, which is the numerically stable way to solve
    the normal equations.
    """

    def __init__(self) -> None:
        """Create the forecaster."""
        super().__init__()
        self._slope = 0.0
        self._intercept = 0.0

    @property
    def slope(self) -> float:
        """Fitted slope (cost change per step)."""
        return self._slope

    @property
    def intercept(self) -> float:
        """Fitted intercept (cost at t = 0)."""
        return self._intercept

    def _fit(self, values: FloatArray) -> None:
        n = values.size
        design = np.column_stack((np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64)))
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        self._slope = float(coefficients[0])
        self._intercept = float(coefficients[1])
        self._fitted = design @ coefficients

    def _forecast(self, horizon: int) -> FloatArray:
        n = self._values.size
        steps = n - 1 + np.arange(1, horizon + 1, dtype=np.float64)
        return self._slope * steps + self._intercept


class CrostonForecast(BaseForecaster):
    """Croston's method for intermittent (spiky) demand.

    LLM spend is bursty: most periods are quiet and a few are huge.  Croston's
    method smooths the *size* of a non-zero observation and the *interval*
    between non-zero observations separately::

        z_t = alpha * y_t + (1 - alpha) * z_{t-1}     (when y_t > 0)
        p_t = alpha * q_t + (1 - alpha) * p_{t-1}     (when y_t > 0)
        y_hat_{t+h} = z_t / p_t

    where ``q_t`` is the number of periods elapsed since the previous non-zero
    observation.  The forecast is a constant rate per period, which is the
    honest representation of intermittent demand.
    """

    def __init__(self, *, alpha: float = 0.2) -> None:
        """Create the forecaster.

        Args:
            alpha: Smoothing constant in ``(0, 1]``.

        Raises:
            ValueError: If ``alpha`` is out of range.
        """
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._size = 0.0
        self._interval = 0.0
        self._rate = 0.0

    @property
    def rate(self) -> float:
        """The smoothed demand rate (cost per period) from the last fit."""
        return self._rate

    @property
    def size(self) -> float:
        """The smoothed size of a non-zero observation."""
        return self._size

    @property
    def interval(self) -> float:
        """The smoothed interval between non-zero observations."""
        return self._interval

    def _fit(self, values: FloatArray) -> None:
        alpha = self._alpha
        size = 0.0
        interval = 0.0
        last_index: int | None = None
        fitted = np.zeros(values.size, dtype=np.float64)
        for index in range(values.size):
            observation = float(values[index])
            if observation > 0.0:
                gap = 1.0 if last_index is None else float(index - last_index)
                if size == 0.0:
                    size = observation
                    interval = gap
                else:
                    size = alpha * observation + (1.0 - alpha) * size
                    interval = alpha * gap + (1.0 - alpha) * interval
                last_index = index
            fitted[index] = size / interval if interval > 0.0 else 0.0
        self._size = size
        self._interval = interval
        self._rate = size / interval if interval > 0.0 else 0.0
        self._fitted = fitted

    def _forecast(self, horizon: int) -> FloatArray:
        return np.full(horizon, self._rate, dtype=np.float64)


class AutoForecaster(BaseForecaster):
    """Pick the best model for a series with a rolling-origin backtest.

    The candidate models are evaluated exactly the way they would be used in
    production: repeatedly training on the past and predicting the future.
    The winner is the model with the lowest mean absolute error.
    """

    DEFAULT_CANDIDATES: tuple[str, ...] = (
        "naive",
        "moving_average",
        "holt",
        "holt_winters",
        "trend",
        "croston",
    )

    def __init__(
        self,
        candidates: Mapping[str, ForecasterFactory] | None = None,
        *,
        min_train: int | None = None,
        horizon: int | None = None,
        step: int | None = None,
    ) -> None:
        """Create the auto-forecaster.

        Args:
            candidates: Optional mapping of name to factory. Factories are
                called with no arguments to build a fresh model per fold.
            min_train: Minimum number of training observations per fold.
            horizon: Number of steps predicted per fold.
            step: Distance between consecutive forecast origins.
        """
        super().__init__()
        self._candidates = dict(candidates) if candidates is not None else None
        self._min_train = min_train
        self._horizon = horizon
        self._step = step
        self._winner: str | None = None
        self._winner_model: Forecaster | None = None
        self._results: dict[str, BacktestResult] = {}
        self._comparison: list[tuple[str, BacktestResult]] = []

    @property
    def winner(self) -> str | None:
        """Name of the winning model, or ``None`` before fitting."""
        return self._winner

    @property
    def comparison(self) -> list[tuple[str, BacktestResult]]:
        """The full comparison table: ``(name, backtest result)`` pairs."""
        return list(self._comparison)

    @property
    def backtests(self) -> dict[str, BacktestResult]:
        """Backtest result per candidate model."""
        return dict(self._results)

    @property
    def mae(self) -> float:
        """Mean absolute error of the winning model (``nan`` before fitting)."""
        return float("nan") if self._winner is None else self._results[self._winner].mae

    @property
    def mape(self) -> float:
        """Mean absolute percentage error of the winning model."""
        return float("nan") if self._winner is None else self._results[self._winner].mape

    @property
    def mase(self) -> float:
        """Mean absolute scaled error of the winning model."""
        return float("nan") if self._winner is None else self._results[self._winner].mase

    def _default_candidates(self) -> dict[str, ForecasterFactory]:
        """The default model zoo evaluated by :class:`AutoForecaster`."""
        return {
            "naive": lambda: NaiveForecaster(),
            "moving_average": lambda: MovingAverageForecaster(window=7),
            "holt": lambda: HoltLinearForecaster(),
            "holt_winters": lambda: SeasonalHoltWinters(period=7),
            "trend": lambda: LinearTrendForecaster(),
            "croston": lambda: CrostonForecast(),
        }

    def _fit(self, values: FloatArray) -> None:
        n = values.size
        min_train = self._min_train if self._min_train is not None else max(4, n // 2)
        horizon = self._horizon if self._horizon is not None else max(1, (n - min_train) // 2)
        step = self._step if self._step is not None else horizon
        min_train = max(1, min(min_train, max(1, n - horizon)))
        horizon = max(1, min(horizon, n - min_train))
        step = max(1, min(step, max(1, n - min_train)))

        candidates = (
            self._candidates if self._candidates is not None else self._default_candidates()
        )
        results: dict[str, BacktestResult] = {}
        for name, factory in candidates.items():
            try:
                result = rolling_origin_backtest(
                    values, factory, min_train=min_train, horizon=horizon, step=step
                )
            except (ValueError, RuntimeError):
                continue
            if result.n_folds > 0:
                results[name] = result
        if not results:
            raise ValueError("no candidate model could be backtested on this series")

        self._results = results
        self._comparison = sorted(results.items(), key=lambda item: item[1].mae)
        self._winner = self._comparison[0][0]
        model = candidates[self._winner]()
        model.fit(values)
        self._winner_model = model
        self._fitted = np.asarray(model.fitted_values, dtype=np.float64)

    def _forecast(self, horizon: int) -> FloatArray:
        if self._winner_model is None:  # pragma: no cover - guarded by fit
            raise RuntimeError("AutoForecaster must be fitted before forecasting")
        return np.asarray(self._winner_model.forecast(horizon), dtype=np.float64)
