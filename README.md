# spendforecast

> Predict LLM token spend overruns — and the date you will blow your budget — before the invoice arrives.

[![CI](https://github.com/aruntewatia87/spendforecast/actions/workflows/ci.yml/badge.svg)](https://github.com/aruntewatia87/spendforecast/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The invoice for an LLM workload arrives roughly 30 days after the money is
gone. By the time you know you overran, the budget is already spent and
nobody could act on it. `spendforecast` closes that gap: it forecasts spend
from your own usage history, tells you **the date the budget will be
exceeded**, and tells you **how much you can safely spend each remaining
day** to stay under it.

It also answers the question you only think of after the incident: *is today's
usage normal?* — separating a one-off spike from a sustained **leak**, which
is the shape of the most expensive bug in production LLM spend (a retry loop
that keeps paying for the same request every minute).

Pure NumPy. No API keys, no network, fully deterministic.

---

## ✨ Highlights

- **Breach-date prediction** — not "you overspent", but "you overspend on the 17th, and here is the confidence band".
- **Safe-to-spend allowance** — computed from the *pessimistic* edge of the error band, so the number is one you can commit to.
- **Seven forecasters** — Naive, Seasonal-Naive, Moving Average, Holt Linear, Holt-Winters, Linear Trend and **Croston** (the right model for bursty, intermittent agent traffic).
- **Rolling-origin backtest** — the training window never sees the future, and the library asserts it.
- **Auto-selection** — `AutoForecaster` picks the winner by backtest and shows you the comparison table.
- **Spike vs. leak vs. sign error** — three different failure modes with three different responses.
- **Robust statistics** — median/MAD-based scores, so a single huge outlier cannot mask itself.

---

## 🚀 Quickstart

```bash
git clone https://github.com/aruntewatia87/spendforecast.git
cd spendforecast
uv sync --all-extras --dev
uv run pytest -q
```

Or standalone:

```bash
uv pip install spendforecast
```

---

## 🔧 API Tour

### Build a usage series

```python
from datetime import datetime, timedelta
from spendforecast import UsageSeries

base = datetime(2026, 1, 1)
series = UsageSeries.from_values(
    [base + timedelta(days=i) for i in range(45)],
    [40.0 + 0.8 * i for i in range(45)],
    model="gpt-4o",
)

series.resample("day")  # calendar-correct daily buckets, gaps zero-filled
series.rolling_mean(7)  # trailing 7-day mean
series.cumulative()  # running total
```

> Missing buckets are **zero-filled**, not skipped. A gap in the data is a
> real property of spend — nothing was spent — and dropping it biases every
> downstream forecast upwards.

### Forecast and backtest

```python
from spendforecast import SeasonalHoltWinters, rolling_origin_backtest

result = rolling_origin_backtest(
    series,
    lambda: SeasonalHoltWinters(period=7),
    min_train=21,
    horizon=7,
)
print(result.mae, result.mape, result.mase, result.bias)
```

`bias` is signed: negative means the model systematically *under*-forecasts,
which is the dangerous direction.

### Predict the breach and the safe-to-spend

```python
from spendforecast import BreachPredictor

predictor = BreachPredictor(budget=1500.0, period="month")
prediction = predictor.predict(series)

print(prediction.will_breach)  # True
print(prediction.breach_date)  # datetime(2026, 2, 17, ...)
print(prediction.projected_total)  # 2806.20
print(prediction.confidence)  # 0.0 .. 1.0

# The number you actually need:
print(predictor.safe_to_spend(series))  # dollars per remaining day
```

### Detect anomalies

```python
from spendforecast import AnomalyDetector, LeakDetector, SpikeDetector

detector = AnomalyDetector(period=7, spike_threshold=3.0, min_run=3)
for result in detector.detect(series):
    print(result.timestamp, result.kind, result.severity, result.score)
```

`kind` is one of `spike`, `level_shift` (a leak) or `negative`.

### Let the library pick the model

```python
from spendforecast import AutoForecaster

auto = AutoForecaster(min_train=21).fit(series)
print(auto.winner)  # e.g. 'holt_winters'
print(auto.comparison)  # the full table
```

---

## 🧠 How it works

### Holt's linear trend

Double exponential smoothing tracks a level `l` and a trend `b`:

```
l_t = α·y_t + (1 − α)(l_{t−1} + b_{t−1})
b_t = β(l_t − l_{t−1}) + (1 − β)·b_{t−1}
ŷ_{t+h} = l_t + h·b_t
```

Holt-Winters adds a third term for the seasonal phase `s`:

```
s_t = γ(y_t − l_t) + (1 − γ)·s_{t−m}
ŷ_{t+h} = l_t + h·b_t + s_{t+h−1}
```

### Croston for bursty traffic

Agent workloads are **intermittent**: many zero-demand periods punctuated by
spikes. Croston's method updates the demand size and the inter-arrival
interval separately:

```
z_t = α·y_t + (1 − α)·z_{t−1}      (size)
x_t = α·q_t + (1 − α)·x_{t−1}      (interval)
ŷ_{t+1} = z_t / x_t
```

This is the model that actually fits "a support agent that only spends tokens
on the 5% of conversations that need a tool call".

### Rolling-origin backtest

For each origin `t` the model is trained on `[0, t)` and asked for the next
`horizon` values, then the window slides forward. The training data always
ends *before* the origin — the library asserts this in its own test suite,
because leaking the future is the single most common way a backtest lies to
you.

### Robust anomaly scoring

Scores use the median and the **MAD** rather than the mean and the standard
deviation:

```
z = 0.6745 · (x − median(x)) / MAD
```

The `0.6745` constant makes the score directly comparable to a standard
z-score. A mean/std score is masked by the very outlier you are trying to
find; the MAD version is not.

Seasonal decomposition is **iterative** (trend → seasonal → re-trend). A
single pass leaves a systematic, phase-dependent offset in the residual that
can be *larger than a real anomaly* — which would make a perfectly clean
series report itself as full of anomalies.

### Leak detection

A leak is detected as a **step** between two adjacent windows, not as
"residual above zero". This distinction matters: once the trend absorbs a
sustained shift, the individual residuals of the leaking run oscillate
around zero, so a detector that looks for a positive residual misses exactly
the case it exists for. A single spike is still rejected, because it leaves a
ring of *negative* residuals around itself.

---

## 📊 Example output

```text
$ uv run python examples/quickstart.py

Daily spend series: 45 days, total $3,414.77
Range: 2026-01-01 -> 2026-02-14

====================================================================
1. Backtest: which forecaster explains this series best?
====================================================================
model                  MAE      MAPE      MASE      bias
--------------------------------------------------------
naive                 6.24      7.3%     1.177     -3.46
holt-linear           4.08      4.8%     0.768     -0.59
holt-winters          2.39      2.9%     0.450     -0.10

Winner: holt-winters (MAE $2.39/day)

====================================================================
2. Will this month's budget be blown?
====================================================================
Spent to date      : $1,244.95
Forecast remainder : $1,561.25
Projected total    : $2,806.20
Budget             : $1,500.00
Headroom           : $-1,306.20
Will breach        : True
Breach date        : 2026-02-17 (3 days away)
Confidence         : 100% (band $2,653 - $2,960)
Model used         : auto
Safe to spend/day  : $0.00

====================================================================
3. Is today normal? (spike, leak and sign-error detection)
====================================================================
  2026-01-21  spike        severity=high    value=   298.21  expected=   130.90  score=  26.71
  2026-02-21  negative     severity=high    value=   -12.00  expected=     0.00  score=    nan

====================================================================
4. Auto-selected model for the forecast
====================================================================
Selected: holt_winters  (MAE $4.18, MAPE 5.0%)
  holt_winters     MAE $    4.18   MASE  0.787
  trend            MAE $    4.46   MASE  0.841
  holt             MAE $    4.92   MASE  0.927
  moving_average   MAE $    9.23   MASE  1.740
  croston          MAE $    9.71   MASE  1.830
  naive            MAE $    9.94   MASE  1.873
```

---

## 🧪 Testing

```bash
uv sync --all-extras --dev
uv run pytest -q --cov=spendforecast --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
```

The suite covers calendar bucketing at day/week boundaries, month and year
rollover, zero-filled gaps, hand-computed Holt recursions, the linear-trend
coefficients, backtest train/test ordering, breach-date and safe-to-spend
arithmetic, and the false-positive behaviour of the anomaly detectors.

---

## 📄 License

MIT © 2026 Arun Tewatia
