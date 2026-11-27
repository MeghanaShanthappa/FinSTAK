from bisect import bisect_left
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from pathlib import Path
import statistics

from stak_graph.extraction.sources import utc_timestamp


def moment(value):
    return datetime.fromisoformat(utc_timestamp(value).replace("Z", "+00:00"))


def boolean(value):
    if str(value).lower() not in ("true", "false"):
        raise ValueError("is_synthetic must be true or false")
    return str(value).lower() == "true"


@dataclass(frozen=True)
class Session:
    session_date: str
    open_at: str
    close_at: str
    prediction_at: str
    is_synthetic: bool = False

    def __post_init__(self):
        date.fromisoformat(self.session_date)
        for key in ("open_at", "close_at", "prediction_at"):
            object.__setattr__(self, key, utc_timestamp(getattr(self, key)))
        if not moment(self.open_at) < moment(self.close_at) <= moment(self.prediction_at):
            raise ValueError("Session requires open_at < close_at <= prediction_at")


class TradingCalendar:
    def __init__(self, sessions):
        self.sessions = sorted(sessions, key=lambda item: item.session_date)
        if not self.sessions or len({s.session_date for s in self.sessions}) != len(self.sessions):
            raise ValueError("Calendar must contain unique, nonempty sessions")
        self.times = [moment(s.prediction_at) for s in self.sessions]
        if any(left >= right for left, right in zip(self.times, self.times[1:])):
            raise ValueError("Prediction cutoffs must increase with session dates")
        self.index = {s.session_date: i for i, s in enumerate(self.sessions)}
        self.coverage_start = datetime.combine(date.fromisoformat(self.sessions[0].session_date), datetime.min.time(), timezone.utc)

    @classmethod
    def from_csv(cls, path):
        with Path(path).open(newline="", encoding="utf-8") as source:
            return cls([Session(row["session_date"], row["open_at"], row["close_at"],
                                row.get("prediction_at") or row["close_at"], boolean(row["is_synthetic"]))
                        for row in csv.DictReader(source)])

    def first_eligible(self, known_at):
        known = moment(known_at)
        if known < self.coverage_start:
            raise ValueError("Event predates calendar coverage; extend the calendar rather than resetting event age")
        index = bisect_left(self.times, known)
        return index if index < len(self.sessions) else None


@dataclass(frozen=True)
class Bar:
    company_id: str
    session_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    available_at: str
    is_synthetic: bool

    def __post_init__(self):
        date.fromisoformat(self.session_date)
        object.__setattr__(self, "available_at", utc_timestamp(self.available_at))
        values = [self.open, self.high, self.low, self.close, self.volume]
        if not self.company_id or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid company ID or non-finite OHLCV")
        if min(values[:4]) <= 0 or self.volume < 0:
            raise ValueError("Prices must be positive and volume nonnegative")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close) or self.high < self.low:
            raise ValueError("OHLC high/low bounds are inconsistent")


def read_bars(path, calendar):
    bars, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            bar = Bar(row["company_id"], row["session_date"], *(float(row[key]) for key in ("open", "high", "low", "close", "volume")),
                      row["available_at"], boolean(row["is_synthetic"]))
            key = (bar.company_id, bar.session_date)
            if key in seen:
                raise ValueError("Duplicate company/session bar; revisions require a separate point-in-time policy")
            seen.add(key)
            if bar.session_date not in calendar.index:
                raise ValueError("Price session is missing from the explicit trading calendar")
            session = calendar.sessions[calendar.index[bar.session_date]]
            if not moment(session.close_at) <= moment(bar.available_at) <= moment(session.prediction_at):
                raise ValueError("Bar availability must fall between the market close and declared prediction cutoff")
            bars.append(bar)
    return sorted(bars, key=lambda item: (item.company_id, item.session_date))


FEATURE_NAMES = (
    "log_return", "opening_gap", "candle_body", "high_low_range", "upper_wick", "lower_wick",
    "log_volume_change", "realized_volatility_5", "ma_distance_5", "return_missing",
    "volume_change_missing", "volatility_missing", "ma_observation_fraction", "session_gap",
)


def candle_features(history, calendar):
    """Every feature reads only this company's observed prefix, never a future bar."""
    result = []
    returns = []
    for i, bar in enumerate(history):
        index = calendar.index[bar.session_date]
        previous = history[i - 1] if i else None
        gap = index - calendar.index[previous.session_date] if previous else 0
        consecutive = previous is not None and gap == 1
        denominator = previous.close if consecutive else bar.open
        ret = math.log(bar.close / previous.close) if consecutive else 0.0
        returns.append((index, ret, consecutive))
        recent = [value for session, value, valid in returns if session >= index - 4 and valid]
        closes = [past.close for past in history[:i + 1] if calendar.index[past.session_date] >= index - 4]
        result.append([
            ret, bar.open / denominator - 1 if consecutive else 0.0,
            (bar.close - bar.open) / denominator, (bar.high - bar.low) / denominator,
            (bar.high - max(bar.open, bar.close)) / denominator,
            (min(bar.open, bar.close) - bar.low) / denominator,
            math.log1p(bar.volume) - math.log1p(previous.volume) if consecutive else 0.0,
            statistics.pstdev(recent) if len(recent) >= 2 else 0.0,
            bar.close / statistics.mean(closes) - 1,
            float(not consecutive), float(not consecutive), float(len(recent) < 2), len(closes) / 5, float(gap),
        ])
    return result
