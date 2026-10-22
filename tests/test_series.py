"""Tests for the usage series: bucketing, resampling and splitting."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from spendforecast import UsagePoint, UsageSeries, bucket_start, normalize_period


def test_normalize_period_aliases() -> None:
    assert normalize_period("h") == "hour"
    assert normalize_period("D") == "day"
    assert normalize_period("weekly") == "week"


def test_normalize_period_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported period"):
        normalize_period("fortnight")


def test_bucket_start_hour() -> None:
    stamp = datetime(2026, 3, 1, 13, 47, 12)
    assert bucket_start(stamp, "hour") == datetime(2026, 3, 1, 13, 0, 0)


def test_bucket_start_day() -> None:
    stamp = datetime(2026, 3, 1, 13, 47, 12)
    assert bucket_start(stamp, "day") == datetime(2026, 3, 1, 0, 0, 0)


def test_bucket_start_week_is_monday() -> None:
    # 2026-03-01 is a Sunday, so the week starts on Monday 2026-02-23.
    stamp = datetime(2026, 3, 1, 13, 47, 12)
    assert bucket_start(stamp, "week") == datetime(2026, 2, 23, 0, 0, 0)


def test_series_is_sorted_on_construction() -> None:
    base = datetime(2026, 1, 1)
    points = [
        UsagePoint(timestamp=base + timedelta(days=2), cost_usd=3.0),
        UsagePoint(timestamp=base, cost_usd=1.0),
        UsagePoint(timestamp=base + timedelta(days=1), cost_usd=2.0),
    ]
    series = UsageSeries(points)
    assert list(series.values) == [1.0, 2.0, 3.0]


def test_usage_point_total_tokens() -> None:
    point = UsagePoint(timestamp=datetime(2026, 1, 1), input_tokens=10, output_tokens=5)
    assert point.total_tokens == 15


def test_from_values_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        UsageSeries.from_values([datetime(2026, 1, 1)], [1.0, 2.0])


def test_from_values_builds_series() -> None:
    base = datetime(2026, 1, 1)
    series = UsageSeries.from_values([base, base + timedelta(days=1)], [1.5, 2.5])
    assert len(series) == 2
    assert series.total_cost == pytest.approx(4.0)


def test_resample_daily_aggregates() -> None:
    base = datetime(2026, 1, 1, 9, 0)
    points = [
        UsagePoint(timestamp=base, cost_usd=1.0, input_tokens=10),
        UsagePoint(timestamp=base + timedelta(hours=2), cost_usd=2.0, input_tokens=20),
        UsagePoint(timestamp=base + timedelta(days=1, hours=3), cost_usd=4.0, input_tokens=40),
    ]
    daily = UsageSeries(points).resample("day")
    assert len(daily) == 2
    assert daily.values[0] == pytest.approx(3.0)
    assert daily.values[1] == pytest.approx(4.0)


def test_resample_hourly_buckets_are_hourly() -> None:
    base = datetime(2026, 1, 1, 9, 30)
    points = [
        UsagePoint(timestamp=base, cost_usd=1.0),
        UsagePoint(timestamp=base + timedelta(minutes=45), cost_usd=2.0),
        UsagePoint(timestamp=base + timedelta(hours=1), cost_usd=4.0),
    ]
    hourly = UsageSeries(points).resample("hour")
    # 09:30 and 10:15 share the 09:00 bucket; 10:30 opens the 10:00 bucket.
    assert len(hourly) == 2
    assert hourly.values[0] == pytest.approx(1.0)
    assert hourly.values[1] == pytest.approx(6.0)


def test_resample_zero_fills_gaps() -> None:
    base = datetime(2026, 1, 1)
    points = [
        UsagePoint(timestamp=base, cost_usd=1.0),
        UsagePoint(timestamp=base + timedelta(days=3), cost_usd=5.0),
    ]
    daily = UsageSeries(points).resample("day")
    # Three empty days between the two observations must still be present.
    assert len(daily) == 4
    assert daily.values[1] == 0.0
    assert daily.values[2] == 0.0
    assert daily.values[3] == pytest.approx(5.0)


def test_resample_marks_mixed_dimensions() -> None:
    base = datetime(2026, 1, 1)
    points = [
        UsagePoint(timestamp=base, cost_usd=1.0, model="a", team_id="x"),
        UsagePoint(timestamp=base, cost_usd=2.0, model="b", team_id="y"),
    ]
    daily = UsageSeries(points).resample("day")
    assert daily.points[0].model == "mixed"
    assert daily.points[0].team_id == "mixed"


def test_rolling_mean() -> None:
    base = datetime(2026, 1, 1)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    series = UsageSeries.from_values([base + timedelta(days=i) for i in range(5)], values)
    rolling = series.rolling_mean(2)
    assert len(rolling) == 5
    assert rolling[0] == pytest.approx(1.0)
    assert rolling[1] == pytest.approx(1.5)
    assert rolling[-1] == pytest.approx(4.5)


def test_cumulative_is_cumsum() -> None:
    base = datetime(2026, 1, 1)
    series = UsageSeries.from_values(
        [base + timedelta(days=i) for i in range(4)], [1.0, 2.0, 3.0, 4.0]
    )
    assert list(series.cumulative()) == pytest.approx([1.0, 3.0, 6.0, 10.0])


def test_split_partitions_the_series() -> None:
    base = datetime(2026, 1, 1)
    series = UsageSeries.from_values(
        [base + timedelta(days=i) for i in range(10)], [float(i) for i in range(10)]
    )
    train, test = series.split(0.8)
    assert len(train) == 8
    assert len(test) == 2
    assert train.total_cost == pytest.approx(sum(range(8)))
    assert test.total_cost == pytest.approx(8.0 + 9.0)


def test_split_rejects_bad_ratio() -> None:
    base = datetime(2026, 1, 1)
    series = UsageSeries.from_values([base], [1.0])
    with pytest.raises(ValueError, match="ratio"):
        series.split(1.5)


def test_totals_are_consistent() -> None:
    base = datetime(2026, 1, 1)
    points = [
        UsagePoint(timestamp=base, input_tokens=100, output_tokens=50, cost_usd=1.25),
        UsagePoint(
            timestamp=base + timedelta(days=1), input_tokens=200, output_tokens=25, cost_usd=2.75
        ),
    ]
    series = UsageSeries(points)
    assert series.total_cost == pytest.approx(4.0)
    assert series.total_tokens == 375
    assert series.input_tokens[-1] == 200


def test_empty_series_handles_properties() -> None:
    series = UsageSeries([])
    assert len(series) == 0
    assert series.start is None
    assert series.end is None
    assert series.total_cost == 0.0
