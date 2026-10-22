"""Tests for the forecasters and the rolling-origin backtest."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from spendforecast import (
    AutoForecaster,
    CrostonForecast,
    HoltLinearForecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalHoltWinters,
    UsageSeries,
    rolling_origin_backtest,
    select_best_forecaster,
)


def make_series(values: list[float]) -> UsageSeries:
    """Build a daily series starting 2026-01-01 from ``values``."""
    base = datetime(2026, 1, 1)
    return UsageSeries.from_values([base + timedelta(days=i) for i in range(len(values))], values)


def test_naive_forecast_repeats_last_value() -> None:
    forecaster = NaiveForecaster().fit(make_series([1.0, 2.0, 3.0]))
    assert forecaster.forecast(3) == pytest.approx([3.0, 3.0, 3.0])


def test_seasonal_naive_repeats_last_season() -> None:
    forecaster = NaiveForecaster(seasonal_period=3).fit(make_series([1.0, 2.0, 3.0, 4.0]))
    forecast = forecaster.forecast(3)
    assert forecast == pytest.approx([2.0, 3.0, 4.0])


def test_moving_average_uses_the_window() -> None:
    forecaster = MovingAverageForecaster(window=3).fit(make_series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert forecaster.forecast(1)[0] == pytest.approx(4.0)


def test_holt_linear_trend_continues_the_slope() -> None:
    values = [float(i) for i in range(20)]
    forecaster = HoltLinearForecaster(alpha=0.9, beta=0.9).fit(make_series(values))
    # The series is y_t = t, so the forecast must continue the line exactly.
    assert forecaster.forecast(3) == pytest.approx([20.0, 21.0, 22.0])
    assert forecaster.level == pytest.approx(19.0)
    assert forecaster.trend == pytest.approx(1.0)


def test_holt_recursion_matches_hand_computed_step() -> None:
    # l_t = a*y_t + (1-a)(l_{t-1} + b_{t-1}); b_t = b(l_t - l_{t-1}) + (1-b)b_{t-1}
    # The implementation stores the *fitted* value level + trend before the
    # update, so the last fitted value is the level of the penultimate step
    # plus the trend at that point.
    alpha, beta = 0.5, 0.5
    values = [10.0, 12.0]
    level, trend = values[0], values[1] - values[0]
    for value in values[1:]:
        previous_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
    forecaster = HoltLinearForecaster(alpha=alpha, beta=beta).fit(make_series(values))
    assert forecaster.level == pytest.approx(level, abs=1e-9)
    assert forecaster.trend == pytest.approx(trend, abs=1e-9)


def test_linear_trend_recovers_slope_and_intercept() -> None:
    values = [3.0 + 2.0 * i for i in range(15)]
    forecaster = LinearTrendForecaster().fit(make_series(values))
    forecast = forecaster.forecast(3)
    # Continuation of y = 3 + 2t at t = 15, 16, 17.
    assert forecast[0] == pytest.approx(33.0, abs=1.0)
    assert forecast[1] == pytest.approx(35.0, abs=1.0)
    assert forecast[2] == pytest.approx(37.0, abs=1.0)


def test_croston_forecast_is_non_negative() -> None:
    values = [0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 7.0]
    forecaster = CrostonForecast(alpha=0.2).fit(make_series(values))
    forecast = forecaster.forecast(4)
    assert all(value >= 0.0 for value in forecast)


def test_croston_total_is_close_to_actual_total() -> None:
    values = [0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 7.0]
    forecaster = CrostonForecast(alpha=0.3).fit(make_series(values))
    forecast = forecaster.forecast(8)
    assert sum(forecast) == pytest.approx(sum(values), rel=0.5)


def test_croston_rate_and_size_are_positive() -> None:
    values = [0.0, 5.0, 0.0, 0.0, 6.0]
    forecaster = CrostonForecast(alpha=0.3).fit(make_series(values))
    assert forecaster.rate > 0.0
    assert forecaster.size > 0.0


def test_seasonal_holt_winters_beats_naive_on_seasonal_data() -> None:
    """Holt-Winters must win on noisy seasonal data.

    The comparison is made with a rolling-origin backtest rather than against
    the final season, because a seasonal-naive forecast trivially reproduces
    the last observed season and would always "win" on that basis.
    """
    rng = np.random.default_rng(11)
    seasonal = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0]
    values = [100.0 + seasonal[i % 7] + float(rng.normal(0, 6.0)) for i in range(60)]
    series = make_series(values)

    hw = rolling_origin_backtest(
        series,
        lambda: SeasonalHoltWinters(period=7, alpha=0.4, beta=0.0, gamma=0.5),
        min_train=28,
        horizon=7,
    )
    naive = rolling_origin_backtest(
        series, lambda: NaiveForecaster(seasonal_period=7), min_train=28, horizon=7
    )
    assert hw.mae < naive.mae


def test_forecast_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        NaiveForecaster().forecast(1)


def test_forecast_of_negative_horizon_raises() -> None:
    forecaster = NaiveForecaster().fit(make_series([1.0, 2.0]))
    with pytest.raises(ValueError, match="horizon"):
        forecaster.forecast(-1)


def test_empty_series_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        NaiveForecaster().fit(UsageSeries([]))


def test_backtest_train_never_sees_the_future() -> None:
    values = [float(i) for i in range(30)]
    result = rolling_origin_backtest(
        make_series(values), lambda: NaiveForecaster(), min_train=10, horizon=1
    )
    for fold in result.folds:
        # The training window ends exactly at the origin: it never contains
        # the observation being predicted.
        assert fold.train_size == fold.origin
        assert fold.origin < len(values)


def test_backtest_metrics_are_correct_for_naive() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result = rolling_origin_backtest(
        make_series(values), lambda: NaiveForecaster(), min_train=3, horizon=1
    )
    # Naive on a rising series always under-forecasts by exactly 1.0.
    assert result.mae == pytest.approx(1.0)
    assert result.mape > 0.0


def test_backtest_fold_count() -> None:
    values = [float(i) for i in range(20)]
    result = rolling_origin_backtest(
        make_series(values), lambda: NaiveForecaster(), min_train=10, horizon=1, step=1
    )
    assert result.n_folds == 10


def test_backtest_rejects_bad_parameters() -> None:
    series = make_series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="min_train"):
        rolling_origin_backtest(series, lambda: NaiveForecaster(), min_train=0)
    with pytest.raises(ValueError, match="horizon"):
        rolling_origin_backtest(series, lambda: NaiveForecaster(), min_train=1, horizon=0)
    with pytest.raises(ValueError, match="step"):
        rolling_origin_backtest(series, lambda: NaiveForecaster(), min_train=1, step=0)


def test_backtest_rejects_short_series() -> None:
    with pytest.raises(ValueError):
        rolling_origin_backtest(make_series([1.0]), lambda: NaiveForecaster(), min_train=5)


def test_select_best_forecaster_returns_a_winner() -> None:
    values = [3.0 + 2.0 * i + float(np.random.default_rng(3).normal(0, 0.1)) for i in range(40)]
    winner, results = select_best_forecaster(
        make_series(values),
        {
            "naive": lambda: NaiveForecaster(),
            "linear": lambda: LinearTrendForecaster(),
        },
        min_train=15,
    )
    assert winner in {"naive", "linear"}
    assert set(results) == {"naive", "linear"}
    assert all(np.isfinite(result.mae) for result in results.values())


def test_select_best_forecaster_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="metric"):
        select_best_forecaster(
            make_series([1.0, 2.0, 3.0]), {"naive": lambda: NaiveForecaster()}, metric="nope"
        )


def test_select_best_forecaster_requires_candidates() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_best_forecaster(make_series([1.0, 2.0, 3.0]), {})


def test_auto_forecaster_picks_a_sensible_winner() -> None:
    values = [float(i) for i in range(40)]
    auto = AutoForecaster(min_train=15).fit(make_series(values))
    assert auto.winner is not None
    assert len(auto.comparison) >= 2
    assert np.isfinite(auto.mae)
    assert auto.mase >= 0.0
