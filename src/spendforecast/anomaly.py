"""Anomaly detection for LLM spend series.

Three detectors are combined here, because production LLM spend has three
distinct failure modes:

:class:`SpikeDetector`
    A single period far above the seasonal baseline: a bad prompt, a traffic
    spike, a mis-configured batch job.
:class:`LeakDetector`
    A *sustained* level shift: a retry loop started burning money, a leaked
    credential, a runaway agent. This is the most expensive bug class in
    production and it is invisible to a spike detector, because no individual
    period looks unusual.
:class:`NegativeSpendDetector`
    Negative cost, which means refunds or a sign error in the pipeline.

The statistical core is a **robust** z-score based on the median and the
median absolute deviation (MAD)::

    mad = median(|x - median(x)|)
    robust_z = 0.6745 * (x - median(x)) / mad

The 0.6745 factor makes the robust z comparable to a standard z-score for
normally distributed data.  A mean/std z-score is *not* used: a single huge
outlier inflates the standard deviation enough to hide itself, which is
precisely the case we need to catch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from spendforecast.series import UsageSeries, as_timestamps, as_values

MAD_SCALE = 0.6745

# Number of refinement passes used by :func:`seasonal_decomposition`.
_DECOMPOSITION_ITERATIONS = 3

# Largest number of observations for which the robust slope compares every
# pair of observations.  Beyond that the quadratic cost is not worth it.
_MAX_SLOPE_PAIRS = 512

ANOMALY_KINDS = ("spike", "level_shift", "negative")


def median_absolute_deviation(values: NDArray[np.float64]) -> float:
    """Return the median absolute deviation of ``values``.

    Args:
        values: The observations.

    Returns:
        ``median(|x - median(x)|)``; ``0.0`` for an empty input.
    """
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def robust_z_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute robust (median/MAD based) z-scores.

    The score is ``0.6745 * (x - median) / MAD`` so that it is directly
    comparable with a standard z-score: for normally distributed data the two
    agree closely.

    A plain MAD score is badly behaved when the residuals are *almost*
    constant, which is exactly what a healthy, strongly seasonal series looks
    like: the MAD collapses towards zero and every tiny residual is inflated
    into a huge score, so a perfectly clean series is reported as full of
    anomalies.  To avoid that, the MAD is floored at a small fraction of the
    series' own scale (its standard deviation), and a series whose spread is
    entirely zero scores zero everywhere.

    Args:
        values: The observations.

    Returns:
        One score per observation.
    """
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    median = float(np.median(values))
    mad = median_absolute_deviation(values)
    if mad > 0.0:
        # Floor the scale so that near-constant residuals are not amplified
        # into false positives.
        scale_floor = 0.05 * float(np.std(values))
        effective_mad = max(mad, scale_floor)
        if effective_mad > 0.0:
            return MAD_SCALE * (values - median) / effective_mad
    spread = float(np.mean(np.abs(values - median)))
    if spread > 0.0:
        return (values - median) / spread
    return np.zeros_like(values)


def seasonal_decomposition(
    values: NDArray[np.float64], period: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Additive seasonal decomposition of ``values``.

    The decomposition is iterative, in the spirit of STL:

    1. Estimate the trend with a centred moving average over one season.
    2. Detrend the series and average the deviations within each seasonal
       phase to get a seasonal profile.
    3. Subtract the seasonal profile, re-estimate the trend, and repeat.

    Iterating matters.  A centred moving average over a season borrows from
    the neighbouring phases, so a single pass leaves a systematic, phase
    dependent offset in the residual -- for a perfectly clean, strongly
    seasonal series that offset is *larger* than any real anomaly, so the
    series is reported as full of anomalies.  Two or more passes drive the
    offset to zero and leave a residual that genuinely reflects the noise.

    The seasonal profile is centred (its phase means sum to zero) so that the
    seasonal component cannot absorb part of the trend.

    Args:
        values: The observations.
        period: Length of the season.

    Returns:
        ``(trend, seasonal, residual)`` arrays of the same length as
        ``values``.

    Raises:
        ValueError: If ``period`` is smaller than 2.
    """
    if period < 2:
        raise ValueError(f"period must be at least 2, got {period}")
    n = values.size
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()

    seasonal = np.zeros(n, dtype=np.float64)
    trend = _centred_moving_average(values, period)

    for _ in range(_DECOMPOSITION_ITERATIONS):
        detrended = values - trend
        profile = np.zeros(period, dtype=np.float64)
        for phase in range(period):
            indices = np.arange(phase, n, period)
            if indices.size:
                profile[phase] = float(np.mean(detrended[indices]))
        profile -= float(np.mean(profile))

        seasonal = np.zeros(n, dtype=np.float64)
        for phase in range(period):
            seasonal[phase::period] = profile[phase]

        trend = _centred_moving_average(values - seasonal, period)

    residual = values - trend - seasonal
    return trend, seasonal, residual


def _centred_moving_average(values: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Centred moving average with partial windows at the edges.

    The edge windows are truncated to the data that actually exists, which
    makes the estimate at the very start and end of the series depend on very
    few observations.  A single spike near an edge therefore propagates
    backwards into the trend and manufactures anomalies that are not there.

    The edge estimate is therefore computed as the *mean of the window with
    its extremes trimmed* (a trimmed mean), which keeps most of the
    smoothing benefit while stopping one extreme value from moving the
    estimate.
    """
    n = values.size
    trend = np.zeros(n, dtype=np.float64)
    half = period // 2
    for index in range(n):
        start = max(0, index - half)
        stop = min(n, index + half + 1)
        window = values[start:stop]
        truncated = start == 0 or stop == n
        if truncated and window.size >= 3:
            ordered = np.sort(window)
            trend[index] = float(ordered[1:-1].mean())
        elif truncated and window.size > 0:
            trend[index] = float(window.mean())
        else:
            trend[index] = float(window.mean())
    return trend


@dataclass(slots=True, frozen=True)
class AnomalyResult:
    """A single detected anomaly.

    Attributes:
        timestamp: When the anomalous period starts.
        value: The observed value.
        expected: The expected value from the baseline.
        score: The robust z-score of the residual (signed).
        kind: One of ``"spike"``, ``"level_shift"`` or ``"negative"``.
        severity: ``"low"``, ``"medium"`` or ``"high"``.
        index: Position of the period in the series.
    """

    timestamp: datetime | None
    value: float
    expected: float
    score: float
    kind: str
    severity: str
    index: int = -1

    def __repr__(self) -> str:
        stamp = self.timestamp.isoformat() if self.timestamp else "n/a"
        return (
            f"AnomalyResult({self.kind}, {stamp}, value={self.value:.2f}, "
            f"expected={self.expected:.2f}, score={self.score:.2f}, severity={self.severity})"
        )


def _severity(score: float) -> str:
    """Map an absolute robust z-score to a severity label."""
    magnitude = abs(score)
    if magnitude >= 6.0:
        return "high"
    if magnitude >= 4.0:
        return "medium"
    return "low"


def _robust_slope(times: NDArray[np.float64], values: NDArray[np.float64]) -> float:
    """Return the robust slope of ``values`` against ``times``.

    The slope is the median of the slopes between all pairs of observations,
    which is the Theil-Sen estimator.  Unlike a least squares fit it is not
    tilted by a single outlier, which matters here: a spike in the trailing
    history must not be able to invent a trend -- or hide one.

    Args:
        times: The positions of the observations.
        values: The observations.

    Returns:
        The slope per unit of ``times``; ``0.0`` when it cannot be estimated.
    """
    n = times.size
    if n < 2:
        return 0.0
    if n > _MAX_SLOPE_PAIRS:
        # Too many pairs to compare exhaustively; the slope between the
        # first and the last observation is a good enough stand-in and keeps
        # the cost of the estimate linear.
        gap = times[-1] - times[0]
        if gap == 0.0:
            return 0.0
        return float((values[-1] - values[0]) / gap)
    left, right = np.triu_indices(n, k=1)
    gaps = times[right] - times[left]
    usable = gaps != 0.0
    if not bool(usable.any()):
        return 0.0
    slopes = (values[right] - values[left])[usable] / gaps[usable]
    return float(np.median(slopes))


class SpikeDetector:
    """Detect a single period far above the seasonal baseline.

    The baseline is the seasonal decomposition's ``trend + seasonal``; a
    period is flagged when its residual exceeds ``threshold`` robust z-scores.
    """

    def __init__(self, *, period: int = 7, threshold: float = 3.5) -> None:
        """Create the detector.

        Args:
            period: Length of the season used for the baseline.
            threshold: Robust z-score above which a period is a spike.

        Raises:
            ValueError: If ``period`` is smaller than 2 or the threshold is not
                positive.
        """
        if period < 2:
            raise ValueError(f"period must be at least 2, got {period}")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self._period = period
        self._threshold = threshold

    def detect(self, series: UsageSeries | Sequence[float]) -> list[AnomalyResult]:
        """Return every spike found in ``series``.

        Args:
            series: The observed spend.

        Returns:
            The detected spikes, ordered by time.
        """
        values = as_values(series)
        if values.size < 3:
            return []
        trend, seasonal, residual = seasonal_decomposition(values, self._period)
        scores = robust_z_scores(residual)
        stamps = self._timestamps(series, values.size)
        results: list[AnomalyResult] = []
        for index in range(values.size):
            if scores[index] > self._threshold:
                results.append(
                    AnomalyResult(
                        timestamp=stamps[index] if index < len(stamps) else None,
                        value=float(values[index]),
                        expected=float(trend[index] + seasonal[index]),
                        score=float(scores[index]),
                        kind="spike",
                        severity=_severity(scores[index]),
                        index=index,
                    )
                )
        return results

    @staticmethod
    def _timestamps(series: UsageSeries | Sequence[float], size: int) -> tuple[datetime, ...]:
        """Timestamps of the observations, when the input carries them."""
        if isinstance(series, UsageSeries):
            return series.timestamps
        try:
            return as_timestamps(series)
        except (AttributeError, TypeError):
            return ()


class LeakDetector:
    """Detect sustained level shifts above a *causal* baseline.

    A leak is a *run* of consecutive periods whose spend is materially higher
    than the level the series was at before the run started -- not a single
    outlier.  This is the shape of the most expensive bug in production LLM
    spend: a retry loop that keeps paying for the same request every minute.

    The baseline is estimated **causally**: the reference level, the local
    trend and the seasonal profile are taken only from the observations that
    come *strictly before* the candidate run.  A global baseline cannot work
    here, because a leak that runs to the end of the series drags the global
    trend upwards and thereby hides the very shift it is meant to reveal.

    Three properties follow from the causal baseline:

    * A leak is a *step*: the mean of the run minus the mean of the
      expected values must be clearly positive.
    * The significance of the step is judged against the spread of the
      *seasonally adjusted* history, so that the normal week-to-week swing
      is not counted as noise.  A robust (median based) spread is used, so
      that a single spike inside the history cannot inflate the denominator
      and mask the step.
    * A leak is *sustained*: the run has to last at least ``min_run``
      periods, and every period of it has to sit above the level that the
      trailing history predicts for that period.  A single spike fails that
      test -- the periods around it sit at the baseline rather than above
      it -- so a spike is not reported twice; that is the
      :class:`SpikeDetector`'s job.

    Because the baseline is causal, a shift in the very first periods of a
    series is not reported: without a trailing history there is no way to
    tell an elevated start apart from a series that simply begins high.
    """

    # A phase median only resists a single outlier when the phase has at
    # least this many samples.  With fewer, the smallest sample of the phase
    # is used instead, so that a spike inside the history cannot raise the
    # reference level of its own phase.
    _MIN_SAMPLES_PER_PHASE = 3

    # Fraction of the level used as the noise scale when the history has no
    # measurable spread at all.
    _MIN_RELATIVE_SPREAD = 0.05

    # A period belongs to the shifted part of a run when it is at least this
    # many noise scales above the baseline.  It keeps a period that merely
    # grazes the baseline from being reported as the start of the shift.
    _ONSET_NOISE_SCALES = 2.0

    # The shifted part of a run also has to reach this fraction of the
    # typical elevation of the run.
    _ONSET_FRACTION = 0.25

    def __init__(self, *, period: int = 7, threshold: float = 3.0, min_run: int = 3) -> None:
        """Create the detector.

        Args:
            period: Length of the season used for the baseline.
            threshold: Robust z-score the run's average deviation must exceed.
            min_run: Minimum number of consecutive periods in a run.

        Raises:
            ValueError: If the parameters are out of range.
        """
        if period < 2:
            raise ValueError(f"period must be at least 2, got {period}")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if min_run < 1:
            raise ValueError(f"min_run must be positive, got {min_run}")
        self._period = period
        self._threshold = threshold
        self._min_run = min_run
        # How far back the trailing history reaches.
        self._history = max(4 * period, 8 * min_run)
        # How much history is needed before the baseline can be trusted: three
        # observations per seasonal phase, so that a phase median is not
        # decided by a single outlier.
        self._min_history = max(3 * period, min_run)

    def detect(self, series: UsageSeries | Sequence[float]) -> list[AnomalyResult]:
        """Return every sustained level shift found in ``series``.

        A candidate run is only judged against a baseline that was estimated
        from the observations preceding it.  A shift in the first few periods
        of a series therefore cannot be separated from the reference level
        and is not reported.

        Args:
            series: The observed spend.

        Returns:
            One :class:`AnomalyResult` per detected run, ordered by time.
        """
        values = as_values(series)
        n = values.size
        if n < self._min_run:
            return []
        stamps = self._timestamps(series, n)
        results: list[AnomalyResult] = []
        # The runs that have already been reported.  They are excluded from
        # the trailing history of later candidates, so that one leak cannot
        # inflate the baseline of the next one.
        leaked: list[tuple[int, int]] = []
        covered_until = -1

        for start in range(self._min_history, n):
            if start <= covered_until:
                # Already inside a run that has been dealt with; keep the
                # earliest detection.
                continue
            history, indices = self._trailing_history(values, start, leaked)
            if history.size < self._min_history:
                # Not enough trailing data to know what "normal" looks like.
                continue

            level, profile, spread = self._trailing_baseline(history, indices)
            expected = self._expected(history, level, profile, indices, n)
            above = values > expected

            # The first stretch of periods at or after ``start`` that stays
            # above the trailing baseline.  A single spike is such a stretch
            # as well, but it is too short to be a leak.
            first = start
            while first < n and not above[first]:
                first += 1
            if first >= n:
                break
            last = first
            while last + 1 < n and above[last + 1]:
                last += 1

            # Drop the periods that only graze the baseline, so that the
            # detection is pinned on the period the level actually jumped at.
            onset = self._onset(values, expected, first, last, spread)
            if last - onset + 1 < self._min_run:
                # Not a sustained rise: the series drops straight back to the
                # baseline, which is what a single spike looks like.
                covered_until = last
                continue

            deviations = values[onset : last + 1] - expected[onset : last + 1]
            score = float(np.mean(deviations) / spread)
            if score <= self._threshold:
                covered_until = last
                continue

            # A leak is a *step up*: the level of the run has to be above the
            # level the trailing history was at.  Without this check a
            # history that contains a level change of its own -- a series
            # that simply starts high, say -- can leave the expectation
            # below the run and be mistaken for a leak.
            run_level = float(
                np.median(
                    values[onset : last + 1] - profile[np.arange(onset, last + 1) % self._period]
                )
            )
            if run_level <= level:
                covered_until = last
                continue

            leaked.append((onset, last))
            covered_until = last
            results.append(
                AnomalyResult(
                    timestamp=stamps[onset] if onset < len(stamps) else None,
                    value=float(np.mean(values[onset : last + 1])),
                    expected=float(np.mean(expected[onset : last + 1])),
                    score=score,
                    kind="level_shift",
                    severity=_severity(score),
                    index=onset,
                )
            )
        return results

    def _trailing_history(
        self, values: NDArray[np.float64], start: int, leaked: list[tuple[int, int]]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Return the observations that precede ``start`` and their indices.

        The periods of the runs that have already been reported are left out,
        so that an earlier leak cannot contaminate the baseline of a later
        one.

        Args:
            values: The whole series.
            start: The first period of the candidate run.
            leaked: The runs that have already been reported.

        Returns:
            The trailing observations and their absolute indices.
        """
        indices = np.arange(max(0, start - self._history), start, dtype=np.int64)
        if leaked:
            keep = np.ones(indices.size, dtype=bool)
            for low, high in leaked:
                keep &= (indices < low) | (indices > high)
            indices = indices[keep]
        return values[indices], indices

    def _trailing_baseline(
        self, history: NDArray[np.float64], indices: NDArray[np.int64]
    ) -> tuple[float, NDArray[np.float64], float]:
        """Estimate the level, the seasonal profile and the noise scale.

        The estimate uses *only* the trailing history, so a leak at the end of
        the series cannot inflate its own reference level.

        Args:
            history: The observations that precede the candidate run.
            indices: Their absolute indices in the series.

        Returns:
            ``(level, profile, spread)``: the level of the history, the
            seasonal profile (one value per phase) and the robust spread of
            the seasonally adjusted history.
        """
        phases = indices % self._period

        # The level is the median of the history, and the profile is the
        # median deviation of each phase from it.  Medians rather than means
        # throughout, so that a single spike inside the history cannot raise
        # the reference level of its own phase.
        level = float(np.median(history))
        profile = np.zeros(self._period, dtype=np.float64)
        seen = np.zeros(self._period, dtype=bool)
        for phase in range(self._period):
            samples = history[phases == phase]
            if samples.size == 0:
                continue
            seen[phase] = True
            if samples.size >= self._MIN_SAMPLES_PER_PHASE:
                profile[phase] = float(np.median(samples) - level)
            else:
                # Too few samples for a median to resist an outlier, so the
                # smallest sample of the phase is used instead.
                profile[phase] = float(np.min(samples) - level)
        profile = np.where(seen, profile, 0.0)

        # The noise scale is the spread of the *seasonally adjusted* history.
        # A moving average over one season removes the seasonal swing and
        # what is left is the noise of the series.  A robust (median based)
        # spread is used, so that a spike inside the history cannot inflate
        # the denominator and mask the step.
        residuals = history - _centred_moving_average(history, self._period)
        spread = median_absolute_deviation(residuals)
        if spread <= 0.0:
            spread = float(np.std(residuals))
        if spread <= 0.0:
            spread = float(np.mean(np.abs(residuals)))
        if spread <= 0.0:
            # A noiseless history has no spread to speak of; fall back to a
            # small fraction of the level, so that the score stays a ratio
            # instead of dividing by zero.
            spread = self._MIN_RELATIVE_SPREAD * abs(level)
        if spread <= 0.0:
            spread = 1.0
        return level, profile, spread

    def _expected(
        self,
        history: NDArray[np.float64],
        level: float,
        profile: NDArray[np.float64],
        indices: NDArray[np.int64],
        n: int,
    ) -> NDArray[np.float64]:
        """Return the value every period is expected to have.

        The expectation continues the local trend of the trailing history and
        adds the seasonal profile on top of it.  A series that is simply
        growing is therefore not mistaken for a leak.

        Args:
            history: The observations that precede the candidate run.
            level: The trailing level estimate.
            profile: The trailing seasonal profile.
            indices: The absolute indices of the trailing history.
            n: The number of observations.

        Returns:
            The expected value of every period of the series.
        """
        phases = np.arange(n) % self._period
        if indices.size < 2:
            return level + profile[phases]

        # The local trend of the history, measured robustly so that a spike
        # cannot tilt it: the median of the slopes between all pairs of
        # seasonally adjusted observations.
        adjusted = history - profile[indices % self._period]
        times = indices.astype(np.float64) - float(indices[0])
        slope = _robust_slope(times, adjusted)
        intercept = float(np.median(adjusted - slope * times))
        positions = np.arange(n, dtype=np.float64) - float(indices[0])
        return intercept + slope * positions + profile[phases]

    def _onset(
        self,
        values: NDArray[np.float64],
        expected: NDArray[np.float64],
        first: int,
        last: int,
        spread: float,
    ) -> int:
        """Return the period the elevated run shifted at.

        The periods at the front of the run that only graze the baseline are
        dropped: they are the noise that happened to land above it, not the
        start of the shift.  The shifted periods are the ones that are
        clearly above the baseline, and the first of them is where the shift
        began.

        Args:
            values: The whole series.
            expected: The expected value of every period.
            first: The first period of the run.
            last: The last period of the run.
            spread: The noise scale of the trailing history.

        Returns:
            The index the shift began at.
        """
        deviations = values[first : last + 1] - expected[first : last + 1]
        typical = float(np.median(deviations))
        threshold = max(self._ONSET_NOISE_SCALES * spread, self._ONSET_FRACTION * typical)
        for offset in range(deviations.size):
            if bool(np.all(deviations[offset:] >= threshold)):
                return first + offset
        return first

    @staticmethod
    def _timestamps(series: UsageSeries | Sequence[float], size: int) -> tuple[datetime, ...]:
        """Timestamps of the observations, when the input carries them."""
        if isinstance(series, UsageSeries):
            return series.timestamps
        try:
            return as_timestamps(series)
        except (AttributeError, TypeError):
            return ()


class NegativeSpendDetector:
    """Detect negative spend, which indicates refunds or a sign error."""

    def __init__(self) -> None:
        """Create the detector."""
        self._threshold = 0.0

    def detect(self, series: UsageSeries | Sequence[float]) -> list[AnomalyResult]:
        """Return every negative observation found in ``series``.

        Args:
            series: The observed spend.

        Returns:
            The detected negative observations, ordered by time.
        """
        values = as_values(series)
        stamps = self._timestamps(series, values.size)
        results: list[AnomalyResult] = []
        for index in range(values.size):
            if values[index] < 0.0:
                results.append(
                    AnomalyResult(
                        timestamp=stamps[index] if index < len(stamps) else None,
                        value=float(values[index]),
                        expected=0.0,
                        score=float("nan"),
                        kind="negative",
                        severity="high",
                        index=index,
                    )
                )
        return results

    @staticmethod
    def _timestamps(series: UsageSeries | Sequence[float], size: int) -> tuple[datetime, ...]:
        """Timestamps of the observations, when the input carries them."""
        if isinstance(series, UsageSeries):
            return series.timestamps
        try:
            return as_timestamps(series)
        except (AttributeError, TypeError):
            return ()


class AnomalyDetector:
    """Combine the spike, leak and negative-spend detectors.

    Attributes:
        spike: The spike detector.
        leak: The level-shift detector.
        negative: The negative-spend detector.
    """

    def __init__(
        self,
        *,
        period: int = 7,
        spike_threshold: float = 3.5,
        leak_threshold: float = 3.0,
        min_run: int = 3,
    ) -> None:
        """Create the detector.

        Args:
            period: Length of the season used for the baselines.
            spike_threshold: Robust z-score above which a period is a spike.
            leak_threshold: Robust z-score a sustained run must exceed.
            min_run: Minimum number of consecutive periods for a level shift.

        Raises:
            ValueError: If any parameter is out of range.
        """
        self._spike = SpikeDetector(period=period, threshold=spike_threshold)
        self._leak = LeakDetector(period=period, threshold=leak_threshold, min_run=min_run)
        self._negative = NegativeSpendDetector()

    @property
    def spike(self) -> SpikeDetector:
        """The configured spike detector."""
        return self._spike

    @property
    def leak(self) -> LeakDetector:
        """The configured level-shift detector."""
        return self._leak

    @property
    def negative(self) -> NegativeSpendDetector:
        """The configured negative-spend detector."""
        return self._negative

    def detect(self, series: UsageSeries | Sequence[float]) -> list[AnomalyResult]:
        """Return every anomaly found in ``series``.

        The detectors run independently and their results are merged.  A period
        that is both a spike and part of a level shift is reported as a spike
        (the more specific diagnosis) and the level shift is still reported,
        because the sustained problem is the one that costs money.

        Args:
            series: The observed spend.

        Returns:
            The detected anomalies, ordered by time and then by kind.
        """
        results: list[AnomalyResult] = []
        results.extend(self._spike.detect(series))
        results.extend(self._leak.detect(series))
        results.extend(self._negative.detect(series))
        results.sort(key=lambda item: (item.index, item.kind))
        return results

    def is_anomalous(self, series: UsageSeries | Sequence[float]) -> bool:
        """Whether any anomaly was detected.

        Args:
            series: The observed spend.

        Returns:
            ``True`` when at least one anomaly was found.
        """
        return bool(self.detect(series))
