"""Tests for budget breach prediction and safe-to-spend allowances."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from spendforecast import BreachPredictor, LinearTrendForecaster, NaiveForecaster, UsageSeries


def make_series(values: list[float], start: datetime | None = None) -> UsageSeries:
    """Build a daily series from ``values``."""
    base = start or datetime(2026, 1, 1)
    return UsageSeries.from_values([base + timedelta(days=i) for i in range(len(values))], values)


def test_no_breach_when_spend_is_well_under_budget() -> None:
    predictor = BreachPredictor(budget=10_000.0, period="month")
    prediction = predictor.predict(make_series([1.0] * 20))
    assert prediction.will_breach is False
    assert prediction.breach_date is None
    assert prediction.headroom > 0.0
    assert prediction.overspend == 0.0


def test_breach_is_detected_when_spend_exceeds_budget() -> None:
    # 50 dollars a day against a 600 dollar budget for the month.
    predictor = BreachPredictor(budget=600.0, period="month")
    prediction = predictor.predict(make_series([50.0] * 20))
    assert prediction.will_breach is True
    assert prediction.breach_date is not None


def test_breach_date_is_in_the_future() -> None:
    predictor = BreachPredictor(budget=600.0, period="month")
    prediction = predictor.predict(make_series([50.0] * 20))
    assert prediction.breach_date is not None
    assert prediction.breach_date >= prediction.as_of
    assert prediction.days_to_breach is not None
    assert prediction.days_to_breach >= 0


def test_projected_total_exceeds_budget_on_breach() -> None:
    predictor = BreachPredictor(budget=600.0, period="month")
    prediction = predictor.predict(make_series([50.0] * 20))
    assert prediction.projected_total > prediction.budget
    assert prediction.headroom < 0.0
    assert prediction.overspend == pytest.approx(prediction.projected_total - prediction.budget)


def test_spent_to_date_is_reported() -> None:
    predictor = BreachPredictor(budget=10_000.0, period="month")
    prediction = predictor.predict(make_series([2.0] * 10))
    assert prediction.spent_to_date == pytest.approx(20.0)


def test_confidence_is_a_probability() -> None:
    predictor = BreachPredictor(budget=10_000.0, period="month")
    prediction = predictor.predict(make_series([2.0] * 20))
    assert 0.0 <= prediction.confidence <= 1.0


def test_confidence_band_brackets_the_projection() -> None:
    predictor = BreachPredictor(budget=10_000.0, period="month")
    prediction = predictor.predict(make_series([2.0] * 20))
    assert prediction.confidence_low <= prediction.projected_total <= prediction.confidence_high


def test_explicit_forecaster_is_used() -> None:
    predictor = BreachPredictor(budget=10_000.0, period="month")
    prediction = predictor.predict(make_series([2.0] * 20), forecaster=NaiveForecaster())
    assert prediction.model != ""


def test_error_fraction_override_widens_the_band() -> None:
    predictor = BreachPredictor(budget=10_000.0, period="month")
    narrow = predictor.predict(make_series([2.0] * 20), error_fraction=0.01)
    wide = predictor.predict(make_series([2.0] * 20), error_fraction=0.50)
    assert (wide.confidence_high - wide.confidence_low) >= (
        narrow.confidence_high - narrow.confidence_low
    )


def test_negative_error_fraction_is_rejected() -> None:
    predictor = BreachPredictor(budget=100.0)
    with pytest.raises(ValueError, match="error_fraction"):
        predictor.predict(make_series([1.0] * 5), error_fraction=-0.1)


def test_empty_series_is_rejected() -> None:
    predictor = BreachPredictor(budget=100.0)
    with pytest.raises(ValueError, match="empty"):
        predictor.predict(UsageSeries([]))


def test_future_data_is_ignored() -> None:
    """Observations after ``now`` must not influence the prediction."""
    start = datetime(2026, 1, 1)
    values = [2.0] * 10 + [999.0] * 10
    predictor = BreachPredictor(budget=10_000.0, period="month")

    clean = predictor.predict(make_series(values, start), now=start + timedelta(days=9))
    short = predictor.predict(make_series([2.0] * 10, start), now=start + timedelta(days=9))
    assert clean.projected_total == pytest.approx(short.projected_total)


def test_safe_to_spend_is_positive_when_under_budget() -> None:
    predictor = BreachPredictor(budget=10_000.0, period="month")
    allowance = predictor.safe_to_spend(make_series([2.0] * 10))
    assert allowance > 0.0


def test_safe_to_spend_is_zero_when_already_over_target() -> None:
    predictor = BreachPredictor(budget=10.0, period="month")
    allowance = predictor.safe_to_spend(make_series([100.0] * 10), target=10.0)
    assert allowance == 0.0


def test_safe_to_spend_uses_the_pessimistic_band() -> None:
    """The allowance must be computed from the upper (pessimistic) edge."""
    predictor = BreachPredictor(budget=1_000.0, period="month")
    series = make_series([10.0] * 10)
    prediction = predictor.predict(series)
    allowance = predictor.safe_to_spend(series)
    expected = (1_000.0 - prediction.confidence_high) / prediction.days_remaining
    assert allowance == pytest.approx(max(0.0, expected))


def test_safe_to_spend_rejects_negative_target() -> None:
    predictor = BreachPredictor(budget=100.0)
    with pytest.raises(ValueError, match="target"):
        predictor.safe_to_spend(make_series([1.0] * 5), target=-1.0)


def test_breach_prediction_repr_is_readable() -> None:
    predictor = BreachPredictor(budget=100.0, period="month")
    prediction = predictor.predict(make_series([1.0] * 5))
    assert "will_breach" in repr(prediction)


def test_prediction_uses_linear_trend_when_given() -> None:
    predictor = BreachPredictor(budget=500.0, period="month")
    prediction = predictor.predict(
        make_series([10.0 + 2.0 * i for i in range(20)]),
        forecaster=LinearTrendForecaster(),
    )
    assert prediction.projected_total > 0.0
