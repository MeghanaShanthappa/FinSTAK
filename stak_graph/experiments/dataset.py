"""Causal inputs, next-open execution targets, and purged chronological splits."""

from datetime import datetime
import math
from pathlib import Path

import numpy as np

from stak_graph.graph.market import FEATURE_NAMES

MARKET_FEATURES = ("spy_log_return", "spy_return_5", "spy_return_20", "spy_volatility_5")


def market_features(benchmark, calendar):
    values = np.zeros((len(calendar.sessions), 4), dtype=np.float64)
    closes = {calendar.index[b.session_date]: b.close for b in benchmark}
    returns = {}
    for index in sorted(closes):
        if index - 1 in closes:
            returns[index] = math.log(closes[index] / closes[index - 1])
        values[index, 0] = returns.get(index, 0.0)
        for column, lookback in ((1, 5), (2, 20)):
            if index - lookback in closes:
                values[index, column] = math.log(closes[index] / closes[index - lookback])
        recent = [returns[j] for j in range(index - 4, index + 1) if j in returns]
        values[index, 3] = float(np.std(recent)) if len(recent) >= 2 else 0.0
    return values


def target_at(company, session_index, prices, benchmark, horizon=5):
    """After-close decision -> following open -> fifth following close.

    The benchmark uses identical entry/exit times. Volatility is the RMS of
    the five realized holding-period log-return increments (not their sum).
    """
    first, last = session_index + 1, session_index + horizon
    indices = range(first, last + 1)
    if horizon < 1 or any((company, j) not in prices or j not in benchmark for j in indices):
        raise ValueError("Target requires the complete future holding period")
    start, finish = prices[company, first].open, prices[company, last].close
    excess = math.log(finish / start) - math.log(benchmark[last].close / benchmark[first].open)
    increments = [math.log(prices[company, first].close / start)]
    increments += [math.log(prices[company, j].close / prices[company, j - 1].close) for j in range(first + 1, last + 1)]
    rms = math.sqrt(sum(value * value for value in increments) / horizon)
    return excess, float(excess > 0), math.log(max(rms, 1e-8)), last


def split_of(prediction_date, label_end_date, config):
    if prediction_date < config["sample_start"] or prediction_date > config["sample_end"]:
        return "outside"
    if prediction_date <= config["train_end"]:
        return "train" if label_end_date <= config["train_end"] else "purged"
    if prediction_date <= config["validation_end"]:
        return "validation" if label_end_date <= config["validation_end"] else "purged"
    return "test"


def build_samples(snapshot, bars, benchmark_bars, calendar, config):
    days = snapshot["nodes"]["stock_day"]
    prices = {(b.company_id, calendar.index[b.session_date]): b for b in bars}
    benchmark = {calendar.index[b.session_date]: b for b in benchmark_bars}
    market = market_features(benchmark_bars, calendar)
    features = np.array([day["features"] + market[day["session_index"]].tolist() for day in days], dtype=np.float32)
    lookup = {(day["company_id"], day["session_index"]): i for i, day in enumerate(days)}
    windows, targets, metadata = [], [], []
    for day in days:
        company, index = day["company_id"], day["session_index"]
        start = index - config["lookback_sessions"] + 1
        if start < 0 or any((company, j) not in lookup for j in range(start, index + 1)):
            continue
        if not config["sample_start"] <= day["session_date"] <= config["sample_end"]:
            continue
        excess, direction, log_rms, last = target_at(company, index, prices, benchmark, config["horizon_sessions"])
        label_end = calendar.sessions[last].session_date
        split = split_of(day["session_date"], label_end, config)
        windows.append([lookup[company, j] for j in range(start, index + 1)])
        targets.append([excess, direction, log_rms])
        metadata.append({"day_id": day["id"], "company_id": company, "session_date": day["session_date"],
                         "session_index": index, "prediction_at": day["timestamp"],
                         "entry_at": calendar.sessions[index + 1].open_at,
                         "label_end_at": calendar.sessions[last].close_at, "label_end_date": label_end, "split": split})
    windows = np.array(windows, dtype=np.int64)
    targets = np.array(targets, dtype=np.float32)
    masks = {name: np.array([row["split"] == name for row in metadata]) for name in ("train", "validation", "test", "purged")}
    if any(not masks[name].any() for name in ("train", "validation", "test")):
        raise ValueError("Experiment requires nonempty train, validation, and test periods")
    training_rows = np.unique(windows[masks["train"]])
    mean = features[training_rows].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features[training_rows].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    normalized = ((features - mean) / scale).clip(-10, 10)
    return_scale = float(targets[masks["train"], 0].std())
    log_vol_mean = float(targets[masks["train"], 2].mean())
    log_vol_scale = float(targets[masks["train"], 2].std())
    if return_scale <= 0 or log_vol_scale <= 0:
        raise ValueError("Training targets have zero variance")
    scaler = {"feature_names": [*FEATURE_NAMES, *MARKET_FEATURES], "mean": mean.tolist(), "scale": scale.tolist(),
              "fit_rows": len(training_rows), "fit_latest_date": max(days[i]["session_date"] for i in training_rows),
              "return_scale": return_scale, "log_vol_mean": log_vol_mean, "log_vol_scale": log_vol_scale,
              "clip_standard_deviations": 10}
    return {"features": normalized, "unscaled_features": features, "windows": windows, "targets": targets,
            "metadata": metadata, "masks": masks, "scaler": scaler}


def correlation_edges(snapshot, window=60, neighbors=3, minimum=0.2):
    """Positive Pearson neighbors from the previous `window` completed returns.

    Today's close is already available at the common prediction cutoff. No
    future date, global correlation fit, or future company state is read.
    """
    if window < 5 or neighbors < 1 or not 0 <= minimum < 1:
        raise ValueError("Invalid rolling-correlation configuration")
    days = snapshot["nodes"]["stock_day"]
    companies = sorted({day["company_id"] for day in days})
    company_index = {company: i for i, company in enumerate(companies)}
    n = max(day["session_index"] for day in days) + 1
    returns = np.full((n, len(companies)), np.nan, dtype=np.float64)
    lookup = {}
    for i, day in enumerate(days):
        j, c = day["session_index"], company_index[day["company_id"]]
        lookup[j, c] = i
        if not day["features"][9]:
            returns[j, c] = day["features"][0]
    source, target, weights = [], [], []
    for j in range(window, n):
        recent = returns[j - window + 1:j + 1]
        if not np.isfinite(recent).all():
            continue
        corr = np.corrcoef(recent, rowvar=False)
        for c in range(len(companies)):
            order = sorted(((-float(corr[c, other]), other) for other in range(len(companies))
                            if other != c and np.isfinite(corr[c, other]) and corr[c, other] >= minimum))[:neighbors]
            for negative_weight, other in order:
                if (j, c) in lookup and (j, other) in lookup:
                    source.append(lookup[j, other])
                    target.append(lookup[j, c])
                    weights.append(-negative_weight)
    return np.array(source, dtype=np.int64), np.array(target, dtype=np.int64), np.array(weights, dtype=np.float32)


def causal_replacement(events, seed=913):
    """Negative control: replace event content with already-known past content."""
    from stak_graph.extraction.schema import digest
    ordered = sorted(range(len(events)), key=lambda i: (events[i]["known_at"], events[i]["id"]))
    replacement = list(range(len(events)))
    history = []
    for i in ordered:
        if history:
            replacement[i] = history[int(digest([seed, events[i]["id"]]), 16) % len(history)]
        history.append(i)
    return np.array(replacement, dtype=np.int64)
