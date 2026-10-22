"""Tests for spike, level-shift and negative-spend anomaly detection."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from spendforecast import (
    AnomalyDetector,
    LeakDetector,
    NegativeSpendDetector,
    SpikeDetector,
    UsageSeries,
    median_absolute_deviation,
    robust_z_scores,
    seasonal_decomposition,
)


def make_series(values: list[float]) -> UsageSeries:
    """Build a daily series from ``values``."""
    base = datetime(2026, 1, 1)
    return UsageSeries.from_values([base + timedelta(days=i) for i in range(len(values))], values)


def test_median_absolute_deviation_of_constant_series_is_zero() -> None:
    assert median_absolute_deviation(np.ones(10)) == 0.0


def test_median_absolute_deviation_matches_hand_computation() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    # median = 3, deviations = [2, 1, 0, 1, 97], median deviation = 1
    assert median_absolute_deviation(values) == pytest.approx(1.0)


def test_robust_z_scores_are_zero_for_the_median() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scores = robust_z_scores(values)
    assert abs(scores[2]) == pytest.approx(0.0, abs=1e-9)


def test_robust_z_score_of_a_spike_is_large() -> None:
    # A median/MAD score uses 0.6745 as the consistency constant, so a
    # five-point series with one huge outlier has a very large score.
    values = np.array([1.0, 1.0, 1.0, 1.0, 50.0])
    scores = robust_z_scores(values)
    assert abs(scores[-1]) > 4.0


def test_robust_detector_is_not_fooled_by_the_outlier_itself() -> None:
    """A mean/std z-score would be masked by the outlier; MAD must not be."""
    values = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 500.0])
    scores = robust_z_scores(values)
    mean = values.mean()
    std = values.std()
    naive_z = abs((values[-1] - mean) / std)
    assert abs(scores[-1]) > naive_z


def test_seasonal_decomposition_reconstructs_the_series() -> None:
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 4
    trend, seasonal, residual = seasonal_decomposition(np.array(values), period=7)
    reconstructed = trend + seasonal + residual
    assert np.allclose(reconstructed, np.array(values), atol=1e-6)


def test_seasonal_component_captures_the_period() -> None:
    values = np.array([10.0, 20.0, 10.0, 20.0] * 7)
    _, seasonal, _ = seasonal_decomposition(values, period=4)
    # The two distinct seasonal levels must differ.
    assert np.nanmax(seasonal) - np.nanmin(seasonal) > 1.0


def test_spike_detector_flags_a_planted_spike() -> None:
    # A seasonal baseline with one huge outlier on top of it.
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 3
    values[10] = 300.0
    results = SpikeDetector(period=7, threshold=3.0).detect(make_series(values))
    assert len(results) == 1
    assert results[0].kind == "spike"
    assert results[0].index == 10


def test_spike_detector_is_quiet_on_a_clean_series() -> None:
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 3
    results = SpikeDetector(period=7, threshold=3.0).detect(make_series(values))
    assert results == []


def test_leak_detector_flags_a_sustained_level_shift() -> None:
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 4 + [60.0] * 10
    results = LeakDetector(period=7, threshold=3.0, min_run=3).detect(make_series(values))
    assert results
    assert any(result.kind == "level_shift" for result in results)


def test_leak_detector_does_not_flag_a_single_spike() -> None:
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 3
    values[12] = 300.0
    results = LeakDetector(period=7, threshold=3.0, min_run=3).detect(make_series(values))
    assert not any(result.kind == "level_shift" for result in results)


def test_leak_detector_is_quiet_on_a_clean_seasonal_series() -> None:
    """A clean repeating seasonal pattern must not be reported as a leak."""
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 4
    results = LeakDetector(period=7, threshold=3.0, min_run=3).detect(make_series(values))
    assert results == []


def test_leak_detector_reports_a_sustained_shift_exactly_once() -> None:
    """The shift is reported once, at the period where the level jumps."""
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 4 + [60.0] * 10
    results = LeakDetector(period=7, threshold=3.0, min_run=3).detect(make_series(values))
    shifts = [result for result in results if result.kind == "level_shift"]
    assert len(shifts) == 1
    assert shifts[0].index == 28
    assert shifts[0].score > 0.0
    assert shifts[0].severity in {"low", "medium", "high"}


def test_leak_detector_reports_a_shift_in_the_middle_of_the_series() -> None:
    """A shift that is followed by a return to normal is still detected."""
    values = (
        [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 3
        + [60.0] * 7
        + [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 3
    )
    results = LeakDetector(period=7, threshold=3.0, min_run=3).detect(make_series(values))
    shifts = [result for result in results if result.kind == "level_shift"]
    assert len(shifts) == 1
    assert shifts[0].index == 21


def test_leak_detector_ignores_a_shift_that_is_too_short() -> None:
    """A run shorter than ``min_run`` is not a sustained leak."""
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 4 + [60.0, 60.0]
    results = LeakDetector(period=7, threshold=3.0, min_run=3).detect(make_series(values))
    assert not any(result.kind == "level_shift" for result in results)


def test_leak_detector_ignores_a_shift_at_the_very_start() -> None:
    """Without a trailing history an elevated start is indistinguishable
    from a series that simply begins high, so it is not reported."""
    values = [60.0] * 7 + [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 4
    results = LeakDetector(period=7, threshold=3.0, min_run=3).detect(make_series(values))
    assert not any(result.kind == "level_shift" for result in results)


def test_leak_detector_handles_a_series_shorter_than_the_window() -> None:
    """A series too short to hold a run is simply not analysed."""
    results = LeakDetector(period=7, threshold=3.0, min_run=3).detect(make_series([1.0, 2.0]))
    assert results == []


def test_leak_detector_does_not_report_a_downward_shift() -> None:
    """Only leaks -- spend that went up -- are reported."""
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 4 + [3.0] * 10
    results = LeakDetector(period=7, threshold=3.0, min_run=3).detect(make_series(values))
    assert not any(result.kind == "level_shift" for result in results)


def test_leak_detector_is_not_fooled_by_a_spike_inside_its_history() -> None:
    """A spike in the trailing history must not raise the reference level."""
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 4
    values[10] = 300.0
    results = LeakDetector(period=7, threshold=3.0, min_run=3).detect(make_series(values))
    assert not any(result.kind == "level_shift" for result in results)


def test_negative_spend_detector_flags_negative_values() -> None:
    results = NegativeSpendDetector().detect(make_series([1.0, -2.0, 3.0]))
    assert len(results) == 1
    assert results[0].kind == "negative"


def test_anomaly_result_has_severity() -> None:
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 3
    values[10] = 300.0
    results = SpikeDetector(period=7, threshold=3.0).detect(make_series(values))
    assert results[0].severity in {"low", "medium", "high"}


def test_anomaly_result_repr_is_readable() -> None:
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 3
    values[10] = 300.0
    results = SpikeDetector(period=7, threshold=3.0).detect(make_series(values))
    assert "spike" in repr(results[0])


def test_combined_detector_finds_multiple_kinds() -> None:
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 3
    values[5] = 300.0
    values.append(-5.0)
    detector = AnomalyDetector(period=7, spike_threshold=3.0, leak_threshold=3.0, min_run=3)
    results = detector.detect(make_series(values))
    kinds = {result.kind for result in results}
    assert "spike" in kinds
    assert "negative" in kinds


def test_combined_detector_is_quiet_on_clean_data() -> None:
    clean = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 4
    detector = AnomalyDetector(period=7, spike_threshold=3.0, leak_threshold=3.0, min_run=3)
    assert detector.detect(make_series(clean)) == []
    assert detector.is_anomalous(make_series(clean)) is False


def test_detector_exposes_sub_detectors() -> None:
    detector = AnomalyDetector()
    assert detector.spike is not None
    assert detector.leak is not None
    assert detector.negative is not None


def test_anomaly_results_carry_timestamps() -> None:
    """The detectors must attach the observation timestamp to each result."""
    base = datetime(2026, 1, 1)
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 3
    values[10] = 300.0
    series = UsageSeries.from_values([base + timedelta(days=i) for i in range(len(values))], values)
    results = SpikeDetector(period=7, threshold=3.0).detect(series)
    assert results
    assert results[0].timestamp == base + timedelta(days=10)


def test_detectors_accept_a_plain_sequence_of_floats() -> None:
    """A bare list of numbers must work as well as a UsageSeries."""
    values = [10.0, 14.0, 8.0, 12.0, 16.0, 9.0, 11.0] * 3
    values[10] = 300.0
    results = SpikeDetector(period=7, threshold=3.0).detect(values)
    assert len(results) == 1
    assert results[0].index == 10
