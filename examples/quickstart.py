#!/usr/bin/env python3
"""End-to-end demo of spendforecast.

Run with::

    uv run python examples/quickstart.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from spendforecast import (
    AnomalyDetector,
    AutoForecaster,
    BreachPredictor,
    LinearTrendForecaster,
    NaiveForecaster,
    SeasonalHoltWinters,
    UsageSeries,
    rolling_origin_backtest,
)


def build_daily_spend(days: int = 45) -> UsageSeries:
    """A daily spend series with weekly seasonality, a trend and some noise."""
    rng = np.random.default_rng(2026)
    weekly = [1.0, 1.15, 1.05, 1.1, 1.25, 0.8, 0.65]
    base = datetime(2026, 1, 1)
    values = [
        40.0 + 0.8 * i + 18.0 * weekly[i % 7] + float(rng.normal(0, 3.0)) for i in range(days)
    ]
    return UsageSeries.from_values(
        [base + timedelta(days=i) for i in range(days)], values, model="gpt-4o"
    )


def main() -> None:
    series = build_daily_spend()
    print(f"Daily spend series: {len(series)} days, total ${series.total_cost:,.2f}")
    print(f"Range: {series.start:%Y-%m-%d} -> {series.end:%Y-%m-%d}\n")

    print("=" * 68)
    print("1. Backtest: which forecaster explains this series best?")
    print("=" * 68)
    candidates = {
        "naive": lambda: NaiveForecaster(),
        "holt-linear": lambda: LinearTrendForecaster(),
        "holt-winters": lambda: SeasonalHoltWinters(period=7),
    }
    results = {
        name: rolling_origin_backtest(series, factory, min_train=21, horizon=7)
        for name, factory in candidates.items()
    }
    print(f"{'model':<16}{'MAE':>10}{'MAPE':>10}{'MASE':>10}{'bias':>10}")
    print("-" * 56)
    for name, result in results.items():
        print(
            f"{name:<16}{result.mae:>10.2f}{result.mape:>9.1f}%"
            f"{result.mase:>10.3f}{result.bias:>10.2f}"
        )
    best = min(results, key=lambda n: results[n].mae)
    print(f"\nWinner: {best} (MAE ${results[best].mae:.2f}/day)")

    print()
    print("=" * 68)
    print("2. Will this month's budget be blown?")
    print("=" * 68)
    predictor = BreachPredictor(budget=1500.0, period="month")
    prediction = predictor.predict(series)
    print(f"Spent to date      : ${prediction.spent_to_date:,.2f}")
    print(f"Forecast remainder : ${prediction.forecast_total:,.2f}")
    print(f"Projected total    : ${prediction.projected_total:,.2f}")
    print(f"Budget             : ${prediction.budget:,.2f}")
    print(f"Headroom           : ${prediction.headroom:,.2f}")
    print(f"Will breach        : {prediction.will_breach}")
    if prediction.breach_date is not None:
        print(
            f"Breach date        : {prediction.breach_date:%Y-%m-%d} "
            f"({prediction.days_to_breach} days away)"
        )
    print(
        f"Confidence         : {prediction.confidence:.0%} "
        f"(band ${prediction.confidence_low:,.0f} - ${prediction.confidence_high:,.0f})"
    )
    print(f"Model used         : {prediction.model}")
    allowance = predictor.safe_to_spend(series)
    print(f"Safe to spend/day  : ${allowance:,.2f}")

    print()
    print("=" * 68)
    print("3. Is today normal? (spike, leak and sign-error detection)")
    print("=" * 68)
    values = list(series.values)
    values[20] *= 4.5  # a runaway request
    values.extend([90.0] * 6)  # a retry loop that never stopped
    values.append(-12.0)  # a refund / sign error
    dirty = UsageSeries.from_values(
        [series.start + timedelta(days=i) for i in range(len(values))], values
    )
    detector = AnomalyDetector(period=7, spike_threshold=3.0, leak_threshold=3.0, min_run=3)
    for result in detector.detect(dirty):
        stamp = result.timestamp.strftime("%Y-%m-%d") if result.timestamp else "n/a"
        print(
            f"  {stamp}  {result.kind:<12} severity={result.severity:<7} "
            f"value={result.value:>9.2f}  expected={result.expected:>9.2f}  "
            f"score={result.score:>7.2f}"
        )

    print()
    print("=" * 68)
    print("4. Auto-selected model for the forecast")
    print("=" * 68)
    auto = AutoForecaster(min_train=21).fit(series)
    print(f"Selected: {auto.winner}  (MAE ${auto.mae:.2f}, MAPE {auto.mape:.1f}%)")
    for name, result in auto.comparison:
        print(f"  {name:<16} MAE ${result.mae:>8.2f}   MASE {result.mase:>6.3f}")


if __name__ == "__main__":
    main()
