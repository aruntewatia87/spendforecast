"""Time-indexed usage series: resampling, rolling statistics and train/test splits.

A :class:`UsageSeries` is an ordered collection of :class:`UsagePoint` observations.
The important detail it enforces is **calendar-correct bucketing**: hourly buckets
start on the hour, daily buckets start at midnight and weekly buckets start on
Monday. Buckets with no data are zero-filled rather than skipped, because a
missing bucket is a real property of spend (nothing was spent) and silently
dropping it biases every downstream forecast upwards.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_PERIOD_ALIASES: dict[str, str] = {
    "h": "hour",
    "hour": "hour",
    "hourly": "hour",
    "d": "day",
    "day": "day",
    "daily": "day",
    "w": "week",
    "week": "week",
    "weekly": "week",
}

MIXED = "mixed"


def normalize_period(period: str) -> str:
    """Map a period alias (``"h"``, ``"D"``, ``"weekly"`` ...) to a canonical name.

    Args:
        period: The alias to normalise.

    Returns:
        One of ``"hour"``, ``"day"`` or ``"week"``.

    Raises:
        ValueError: If the alias is not recognised.
    """
    key = period.strip().lower()
    try:
        return _PERIOD_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"unsupported period {period!r}; expected one of {sorted(_PERIOD_ALIASES)}"
        ) from None


def bucket_start(timestamp: datetime, period: str) -> datetime:
    """Return the start of the calendar bucket that ``timestamp`` falls into.

    Args:
        timestamp: The observation timestamp.
        period: Canonical period name (``"hour"``, ``"day"`` or ``"week"``).

    Returns:
        The midnight (or top-of-hour / Monday) timestamp opening the bucket.
    """
    if period == "hour":
        return timestamp.replace(minute=0, second=0, microsecond=0)
    if period == "day":
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        monday = timestamp - timedelta(days=timestamp.weekday())
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"unsupported period {period!r}")


def _next_bucket(start: datetime, period: str) -> datetime:
    """Return the start of the bucket following the one opened at ``start``."""
    if period == "hour":
        return start + timedelta(hours=1)
    if period == "day":
        return start + timedelta(days=1)
    if period == "week":
        return start + timedelta(days=7)
    raise ValueError(f"unsupported period {period!r}")


@dataclass(slots=True, frozen=True)
class UsagePoint:
    """A single observation of LLM usage at one point in time.

    Attributes:
        timestamp: When the usage was observed.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens produced.
        cost_usd: Monetary cost of the observation in US dollars.
        model: The model that served the request.
        team_id: The team that incurred the spend.
        request_count: Number of requests represented by this point.
    """

    timestamp: datetime
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    team_id: str = ""
    request_count: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input plus output) for this observation."""
        return self.input_tokens + self.output_tokens


def _aggregate(timestamp: datetime, group: Sequence[UsagePoint]) -> UsagePoint:
    """Collapse a group of points that share a bucket into a single point."""
    if not group:
        return UsagePoint(timestamp=timestamp)
    models = {point.model for point in group if point.model}
    teams = {point.team_id for point in group if point.team_id}
    return UsagePoint(
        timestamp=timestamp,
        input_tokens=sum(point.input_tokens for point in group),
        output_tokens=sum(point.output_tokens for point in group),
        cost_usd=float(sum(point.cost_usd for point in group)),
        model=next(iter(models)) if len(models) == 1 else MIXED,
        team_id=next(iter(teams)) if len(teams) == 1 else MIXED,
        request_count=sum(point.request_count for point in group),
    )


class UsageSeries:
    """An ordered, time-indexed collection of :class:`UsagePoint` observations.

    The series is sorted by timestamp on construction, so downstream maths can
    rely on chronological ordering. Instances are treated as immutable views;
    every transformation returns a new series.
    """

    def __init__(self, points: Iterable[UsagePoint]) -> None:
        """Create a series from ``points``; the points are sorted by timestamp."""
        self._points: tuple[UsagePoint, ...] = tuple(
            sorted(points, key=lambda point: point.timestamp)
        )
        self._values_cache: FloatArray | None = None

    @classmethod
    def from_values(
        cls,
        timestamps: Sequence[datetime],
        values: Sequence[float],
        *,
        model: str = "",
        team_id: str = "",
    ) -> UsageSeries:
        """Build a series from parallel timestamp and cost sequences.

        Args:
            timestamps: Observation timestamps.
            values: Cost in US dollars for each timestamp.
            model: Optional model label applied to every point.
            team_id: Optional team label applied to every point.

        Returns:
            The assembled :class:`UsageSeries`.

        Raises:
            ValueError: If the two sequences have different lengths.
        """
        if len(timestamps) != len(values):
            raise ValueError("timestamps and values must have the same length")
        return cls(
            UsagePoint(timestamp=stamp, cost_usd=float(cost), model=model, team_id=team_id)
            for stamp, cost in zip(timestamps, values, strict=True)
        )

    def __len__(self) -> int:
        """Number of observations in the series."""
        return len(self._points)

    def __iter__(self):
        """Iterate over the points in chronological order."""
        return iter(self._points)

    def __repr__(self) -> str:
        return (
            f"UsageSeries(points={len(self._points)}, start={self.start}, end={self.end}, "
            f"total_cost={self.total_cost:.2f})"
        )

    @property
    def points(self) -> tuple[UsagePoint, ...]:
        """The raw points, in chronological order."""
        return self._points

    @property
    def timestamps(self) -> tuple[datetime, ...]:
        """Timestamps of every observation, in chronological order."""
        return tuple(point.timestamp for point in self._points)

    @property
    def values(self) -> FloatArray:
        """Cost in US dollars for every observation, in chronological order."""
        if self._values_cache is None:
            self._values_cache = np.array(
                [point.cost_usd for point in self._points], dtype=np.float64
            )
        return self._values_cache

    @property
    def input_tokens(self) -> FloatArray:
        """Input tokens for every observation."""
        return np.array([point.input_tokens for point in self._points], dtype=np.float64)

    @property
    def output_tokens(self) -> FloatArray:
        """Output tokens for every observation."""
        return np.array([point.output_tokens for point in self._points], dtype=np.float64)

    @property
    def start(self) -> datetime | None:
        """Timestamp of the first observation, or ``None`` for an empty series."""
        return self._points[0].timestamp if self._points else None

    @property
    def end(self) -> datetime | None:
        """Timestamp of the last observation, or ``None`` for an empty series."""
        return self._points[-1].timestamp if self._points else None

    @property
    def total_cost(self) -> float:
        """Total cost in US dollars across the whole series."""
        return float(self.values.sum()) if self._points else 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input plus output) across the whole series."""
        return sum(point.total_tokens for point in self._points)

    def resample(self, period: str = "day") -> UsageSeries:
        """Aggregate the series into calendar buckets of ``period``.

        Buckets are half-open calendar periods: ``"hour"`` starts on the hour,
        ``"day"`` at midnight and ``"week"`` on Monday. Every bucket between the
        first and the last observation is emitted, and buckets without data are
        zero-filled instead of being skipped.

        Args:
            period: ``"hour"``, ``"day"`` or ``"week"`` (aliases such as ``"h"``
                and ``"D"`` are accepted).

        Returns:
            A new series with one point per bucket.

        Raises:
            ValueError: If ``period`` is not a supported alias.
        """
        canonical = normalize_period(period)
        if not self._points:
            return UsageSeries(())

        buckets: dict[datetime, list[UsagePoint]] = {}
        for point in self._points:
            buckets.setdefault(bucket_start(point.timestamp, canonical), []).append(point)

        aggregated: list[UsagePoint] = []
        cursor = min(buckets)
        last = max(buckets)
        while cursor <= last:
            aggregated.append(_aggregate(cursor, buckets.get(cursor, [])))
            cursor = _next_bucket(cursor, canonical)
        return UsageSeries(aggregated)

    def rolling_mean(self, window: int) -> FloatArray:
        """Trailing rolling mean of the cost values.

        The value at index ``i`` averages the ``window`` observations ending at
        (and including) ``i``. The first ``window - 1`` entries use the shorter
        partial window that is available.

        Args:
            window: Number of observations in the window.

        Returns:
            An array with one rolling mean per observation.

        Raises:
            ValueError: If ``window`` is not positive.
        """
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        values = self.values
        n = values.size
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(values)))
        ends = np.arange(1, n + 1)
        starts = np.maximum(0, ends - window)
        return (cumulative[ends] - cumulative[starts]) / (ends - starts)

    def cumulative(self) -> FloatArray:
        """Running total of the cost values."""
        values = self.values
        if values.size == 0:
            return np.zeros(0, dtype=np.float64)
        return np.cumsum(values)

    def split(self, train_ratio: float) -> tuple[UsageSeries, UsageSeries]:
        """Split the series into a training and a hold-out part.

        The split is positional: the first ``round(len * train_ratio)``
        observations form the training set. A chronological split is required by
        time-series models, and this guarantees it.

        Args:
            train_ratio: Fraction of the observations used for training.

        Returns:
            ``(train, test)``; the test part is empty when the ratio rounds to
            the full length.

        Raises:
            ValueError: If ``train_ratio`` is outside ``(0, 1)``.
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        n = len(self._points)
        cut = min(max(round(n * train_ratio), 1), n)
        return UsageSeries(self._points[:cut]), UsageSeries(self._points[cut:])


def as_values(series: UsageSeries | Sequence[float]) -> FloatArray:
    """Coerce a series, a list of points or a plain sequence into a float array.

    Args:
        series: A :class:`UsageSeries`, an iterable of :class:`UsagePoint` or
            any 1-D sequence of numbers.

    Returns:
        The cost values as a ``float64`` array.

    Raises:
        ValueError: If the input cannot be interpreted as a 1-D numeric sequence.
    """
    if isinstance(series, UsageSeries):
        return series.values
    items = list(series)
    if items and all(isinstance(item, UsagePoint) for item in items):
        return np.array([item.cost_usd for item in items], dtype=np.float64)
    try:
        array = np.asarray(items, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot interpret {type(series).__name__} as a numeric series") from exc
    if array.ndim != 1:
        raise ValueError("series values must be one-dimensional")
    return array


def as_timestamps(series: UsageSeries | Sequence[UsagePoint]) -> tuple[datetime, ...]:
    """Extract the timestamps of a series or a list of points.

    Args:
        series: A :class:`UsageSeries` or an iterable of :class:`UsagePoint`.

    Returns:
        The timestamps in the order they were supplied.
    """
    if isinstance(series, UsageSeries):
        return series.timestamps
    return tuple(point.timestamp for point in series)
