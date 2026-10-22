"""Budget breach prediction and safe-to-spend allowances.

The question this module answers is the one an architect asks at the start of
a billing period: *if we keep behaving like this, when do we blow the budget,
and how much can we spend per day and still land inside it?*

The approach is deliberately simple and auditable:

1. Forecast the remaining spend of the period from the observed history.
2. Add the spend already committed to get a projected period total.
3. Walk the forecast forward cumulatively until it crosses the budget; that
   crossing point is the breach date.
4. Turn the backtest error of the forecaster into a confidence band around the
   projection, and use the pessimistic edge of that band for the safe-to-spend
   number.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from spendforecast.backtest import BacktestResult, rolling_origin_backtest
from spendforecast.forecasters import AutoForecaster, Forecaster
from spendforecast.series import UsageSeries, as_values

_SECONDS_PER_DAY = 86400.0

BUDGET_PERIODS = ("day", "week", "month")


def _period_bounds(moment: datetime, period: str) -> tuple[datetime, datetime]:
    """Return the ``(start, end)`` of the budget period containing ``moment``.

    The end is exclusive: it is the start of the next period.
    """
    if period == "day":
        start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)
    if period == "week":
        start = moment - timedelta(days=moment.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=7)
    if period == "month":
        start = moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            return start, start.replace(year=start.year + 1, month=1)
        return start, start.replace(month=start.month + 1)
    raise ValueError(
        f"unsupported budget period {period!r}; expected one of {list(BUDGET_PERIODS)}"
    )


@dataclass(slots=True, frozen=True)
class BreachPrediction:
    """The outcome of a budget breach prediction.

    Attributes:
        will_breach: Whether the projected total exceeds the budget.
        breach_date: When the budget is expected to be exceeded, or ``None``.
        projected_total: Spend projected for the whole period.
        budget: The budget that was tested.
        headroom: ``budget - projected_total``; negative means overspend.
        confidence: Share of backtest folds whose actual stayed inside the
            error band, in ``[0, 1]``.
        days_remaining: Days left in the budget period.
        spent_to_date: Spend already recorded inside the budget period.
        forecast_total: Spend forecast for the remaining part of the period.
        mape: Mean absolute percentage error of the backtest, in percent.
        confidence_low: Lower edge of the projection band.
        confidence_high: Upper edge of the projection band.
        period_start: Start of the budget period (inclusive).
        period_end: End of the budget period (exclusive).
        as_of: The reference moment the prediction was made for.
        model: Name of the model that produced the forecast.
        notes: Human readable diagnostics.
    """

    will_breach: bool
    breach_date: datetime | None
    projected_total: float
    budget: float
    headroom: float
    confidence: float
    days_remaining: int
    spent_to_date: float
    forecast_total: float
    mape: float
    confidence_low: float
    confidence_high: float
    period_start: datetime
    period_end: datetime
    as_of: datetime
    model: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def overspend(self) -> float:
        """How much the projection exceeds the budget (``0`` when inside)."""
        return max(0.0, -self.headroom)

    @property
    def days_to_breach(self) -> int | None:
        """Days between the reference moment and the breach, if any."""
        if self.breach_date is None:
            return None
        return (self.breach_date.date() - self.as_of.date()).days

    def __repr__(self) -> str:
        breach = self.breach_date.isoformat() if self.breach_date else "n/a"
        return (
            f"BreachPrediction(will_breach={self.will_breach}, breach_date={breach}, "
            f"projected_total={self.projected_total:.2f}, budget={self.budget:.2f}, "
            f"headroom={self.headroom:.2f}, confidence={self.confidence:.2f})"
        )


class BreachPredictor:
    """Predict when a budget will be exceeded and how much may still be spent.

    Attributes:
        budget: The budget for one budget period, in US dollars.
        period: The budget period: ``"day"``, ``"week"`` or ``"month"``.
    """

    def __init__(self, budget: float, period: str = "month") -> None:
        """Create the predictor.

        Args:
            budget: Budget for one period, in US dollars.
            period: The budget period: ``"day"``, ``"week"`` or ``"month"``.

        Raises:
            ValueError: If the budget is negative or the period is unknown.
        """
        if budget < 0.0:
            raise ValueError(f"budget must be non-negative, got {budget}")
        if period not in BUDGET_PERIODS:
            raise ValueError(
                f"unsupported budget period {period!r}; expected one of {list(BUDGET_PERIODS)}"
            )
        self._budget = float(budget)
        self._period = period

    @property
    def budget(self) -> float:
        """The configured budget."""
        return self._budget

    @property
    def period(self) -> str:
        """The configured budget period."""
        return self._period

    @staticmethod
    def _step_seconds(series: UsageSeries) -> float:
        """Median spacing between observations, in seconds."""
        stamps = series.timestamps
        if len(stamps) < 2:
            return _SECONDS_PER_DAY
        deltas = [(b - a).total_seconds() for a, b in itertools.pairwise(stamps)]
        return float(np.median(deltas)) or _SECONDS_PER_DAY

    @staticmethod
    def _build_forecaster(forecaster: Forecaster | None) -> tuple[Forecaster, str]:
        """Return the forecaster to use plus a human readable model name."""
        if forecaster is not None:
            return forecaster, type(forecaster).__name__
        return AutoForecaster(), "auto"

    @staticmethod
    def _backtest(
        series: UsageSeries, forecaster: Forecaster | None, steps: int
    ) -> BacktestResult | None:
        """Backtest the forecaster to obtain its error band."""
        n = len(series)
        if n < 4:
            return None
        min_train = max(2, n // 2)
        horizon = max(1, min(steps, max(1, n - min_train)))
        if min_train + horizon > n:
            return None

        def factory() -> Forecaster:
            if forecaster is not None:
                return type(forecaster)()  # type: ignore[call-arg]
            return AutoForecaster(min_train=min_train, horizon=horizon, step=horizon)

        try:
            return rolling_origin_backtest(
                series.values, factory, min_train=min_train, horizon=horizon, step=horizon
            )
        except (ValueError, RuntimeError):  # pragma: no cover - defensive
            return None

    def predict(
        self,
        series: UsageSeries | Sequence[float],
        *,
        now: datetime | None = None,
        forecaster: Forecaster | None = None,
        error_fraction: float | None = None,
    ) -> BreachPrediction:
        """Predict whether and when the budget will be exceeded.

        Args:
            series: Observed spend for the period. When a
                :class:`~spendforecast.series.UsageSeries` is given, points
                after ``now`` are ignored so that no future data leaks into the
                forecast.
            now: The reference moment. Defaults to the last observation.
            forecaster: Optional explicit model. When omitted, an
                :class:`~spendforecast.forecasters.AutoForecaster` selects the
                best model with a rolling-origin backtest.
            error_fraction: Optional override for the relative error band
                around the projection. Defaults to the band derived from the
                backtest.

        Returns:
            A :class:`BreachPrediction`.

        Raises:
            ValueError: If the series is empty or ``error_fraction`` is
                negative.
        """
        if error_fraction is not None and error_fraction < 0.0:
            raise ValueError(f"error_fraction must be non-negative, got {error_fraction}")
        history, moment = self._prepare_history(series, now)

        period_start, period_end = _period_bounds(moment, self._period)
        days_remaining = max(0, (period_end.date() - moment.date()).days)

        step_seconds = self._step_seconds(history)
        days_per_step = max(step_seconds / _SECONDS_PER_DAY, 1.0)
        steps_remaining = int(np.ceil(days_remaining / days_per_step)) if days_remaining else 0

        current = [point for point in history if period_start <= point.timestamp < period_end]
        spent_to_date = float(sum(point.cost_usd for point in current))

        model, model_name = self._build_forecaster(forecaster)
        fitted = model.fit(history.values)
        predictions = np.asarray(fitted.forecast(steps_remaining), dtype=np.float64)
        forecast_total = float(predictions.sum()) if predictions.size else 0.0
        projected_total = spent_to_date + forecast_total

        backtest = self._backtest(history, forecaster, steps_remaining)
        if error_fraction is not None:
            band_fraction = error_fraction
            confidence = self._coverage(backtest, band_fraction) if backtest else 0.0
        elif backtest is not None:
            band_fraction = self._band_fraction(backtest)
            confidence = self._coverage(backtest, band_fraction)
        else:
            band_fraction = 0.5
            confidence = 0.0

        band = max(abs(projected_total) * band_fraction, 0.0)
        confidence_low = projected_total - band
        confidence_high = projected_total + band

        headroom = self._budget - projected_total
        will_breach = projected_total > self._budget
        breach_date = self._breach_date(
            predictions, spent_to_date, self._budget, moment, step_seconds
        )

        notes: list[str] = []
        if backtest is not None and not np.isnan(backtest.mape):
            notes.append(
                f"backtest MAPE {backtest.mape:.1f}% over {backtest.n_folds} folds; "
                f"error band {band_fraction * 100:.1f}%"
            )
        if days_remaining == 0:
            notes.append("no days remaining in the budget period")

        return BreachPrediction(
            will_breach=will_breach,
            breach_date=breach_date,
            projected_total=projected_total,
            budget=self._budget,
            headroom=headroom,
            confidence=confidence,
            days_remaining=days_remaining,
            spent_to_date=spent_to_date,
            forecast_total=forecast_total,
            mape=float(backtest.mape) if backtest is not None else float("nan"),
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            period_start=period_start,
            period_end=period_end,
            as_of=moment,
            model=model_name,
            notes=notes,
        )

    @staticmethod
    def _prepare_history(
        series: UsageSeries | Sequence[float], now: datetime | None
    ) -> tuple[UsageSeries, datetime]:
        """Normalise the input into a history series plus a reference moment."""
        if isinstance(series, UsageSeries):
            if not len(series):
                raise ValueError("cannot predict a breach from an empty series")
            moment = now if now is not None else series.end
            assert moment is not None
            truncated = [point for point in series if point.timestamp <= moment]
            history = UsageSeries(truncated) if truncated else UsageSeries(series.points[:1])
            return history, moment
        values = as_values(series)
        if values.size == 0:
            raise ValueError("cannot predict a breach from an empty series")
        history = UsageSeries.from_values(
            [_synthetic_stamp(index) for index in range(values.size)], values
        )
        moment = now if now is not None else history.end
        assert moment is not None
        return history, moment

    @staticmethod
    def _band_fraction(backtest: BacktestResult) -> float:
        """Derive the relative error band around a projection from a backtest.

        The band is the larger of the mean absolute percentage error and the
        absolute bias of the backtest.  The bias is included because a model
        that is systematically optimistic is more dangerous than one that is
        merely imprecise, and a pure MAPE band would hide it.
        """
        candidates = [value for value in (backtest.mape, abs(backtest.bias)) if not np.isnan(value)]
        if not candidates:
            return 0.5
        mape_fraction = backtest.mape / 100.0 if not np.isnan(backtest.mape) else 0.0
        scale = float(np.mean(np.abs(backtest.actuals))) if backtest.actuals else 0.0
        bias_fraction = abs(backtest.bias) / scale if scale > 0.0 else 0.0
        return float(max(mape_fraction, bias_fraction, 0.0))

    @staticmethod
    def _coverage(backtest: BacktestResult, band_fraction: float) -> float:
        """Fraction of backtest folds whose actual stayed inside the band."""
        if backtest.n_folds == 0:
            return 0.0
        inside = 0
        for fold in backtest.folds:
            forecast_total = float(sum(fold.forecast))
            actual_total = float(sum(fold.actual))
            band = abs(forecast_total) * band_fraction
            if abs(actual_total - forecast_total) <= band + 1e-9:
                inside += 1
        return inside / backtest.n_folds

    @staticmethod
    def _breach_date(
        predictions: np.ndarray,
        spent_to_date: float,
        budget: float,
        moment: datetime,
        step_seconds: float,
    ) -> datetime | None:
        """Find the first forecast step at which the cumulative spend crosses the budget."""
        if predictions.size == 0:
            return None
        running = spent_to_date
        for index, value in enumerate(predictions):
            running += float(value)
            if running > budget:
                return moment + timedelta(seconds=step_seconds * (index + 1))
        return None

    def safe_to_spend(
        self,
        series: UsageSeries | Sequence[float],
        *,
        now: datetime | None = None,
        target: float | None = None,
        forecaster: Forecaster | None = None,
        error_fraction: float | None = None,
    ) -> float:
        """How much may be spent per remaining day and still land under the target.

        The calculation uses the pessimistic edge of the forecast error band, so
        the returned allowance is a number you can commit to rather than an
        optimistic average.

        Args:
            series: Observed spend for the period.
            now: The reference moment. Defaults to the last observation.
            target: The spend target for the period. Defaults to the budget.
            forecaster: Optional explicit model.
            error_fraction: Optional override for the relative error band.

        Returns:
            The daily allowance in US dollars; ``0.0`` when the target is
            already exceeded or no days remain.
        """
        goal = self._budget if target is None else float(target)
        if goal < 0.0:
            raise ValueError(f"target must be non-negative, got {goal}")
        prediction = self.predict(
            series, now=now, forecaster=forecaster, error_fraction=error_fraction
        )
        if prediction.days_remaining <= 0:
            return 0.0
        allowance = goal - prediction.confidence_high
        if allowance <= 0.0:
            return 0.0
        return allowance / prediction.days_remaining


def _synthetic_stamp(index: int) -> datetime:
    """A deterministic daily timestamp used when a bare numeric series is given."""
    return datetime(2026, 1, 1) + timedelta(days=index)
