"""Metrics and paired calendar-block uncertainty without an IID stock-day assumption."""

import math
import numpy as np


def average_rank(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return result


def auc(labels, probabilities):
    labels = np.asarray(labels).astype(bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    ranks = average_rank(probabilities)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def rank_ic_by_date(truth, score, dates):
    dates = np.asarray(dates)
    result = {}
    for day in sorted(set(dates)):
        mask = dates == day
        left, right = average_rank(np.asarray(truth)[mask]), average_rank(np.asarray(score)[mask])
        if len(left) >= 3 and left.std() > 0 and right.std() > 0:
            result[day] = float(np.corrcoef(left, right)[0, 1])
    return result


def evaluate(targets, predictions, dates):
    truth, labels, log_vol = targets.T
    score, probability, log_vol_prediction = predictions.T
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite predictions")
    predicted = probability >= 0.5
    positive = labels > 0.5
    tp = int((predicted & positive).sum()); tn = int((~predicted & ~positive).sum())
    fp = int((predicted & ~positive).sum()); fn = int((~predicted & positive).sum())
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    ics = rank_ic_by_date(truth, score, dates)
    clipped = probability.clip(1e-7, 1 - 1e-7)
    return {"samples": len(labels), "sessions": len(set(dates)), "positive_fraction": float(labels.mean()),
            "direction_accuracy": float((predicted == positive).mean()),
            "balanced_accuracy": float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
            "mcc": float((tp * tn - fp * fn) / denominator) if denominator else 0.0,
            "roc_auc": auc(labels, probability), "brier_score": float(((probability - labels) ** 2).mean()),
            "binary_log_loss": float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean()),
            "excess_return_rmse": float(np.sqrt(((score - truth) ** 2).mean())),
            "mean_daily_rank_ic": float(np.mean(list(ics.values()))) if ics else None,
            "rank_ic_sessions": len(ics), "log_volatility_rmse": float(np.sqrt(((log_vol_prediction - log_vol) ** 2).mean()))}


def block_interval(values, block=20, replicates=2000, seed=1729):
    """Moving calendar blocks, with all stocks from a date kept together."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or block < 1 or replicates < 1:
        raise ValueError("Invalid block-bootstrap input")
    rng = np.random.default_rng(seed)
    width = min(block, len(values))
    blocks = int(np.ceil(len(values) / width))
    starts = rng.integers(0, len(values) - width + 1, size=(replicates, blocks))
    indices = (starts[:, :, None] + np.arange(width)).reshape(replicates, -1)[:, :len(values)]
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": float(values.mean()), "ci95": [float(low), float(high)],
            "method": "paired moving calendar-block bootstrap", "block_sessions": width,
            "replicates": replicates, "seed": seed, "sessions": len(values)}


def paired_comparison(targets, candidate, baseline, dates, block=20, replicates=2000):
    dates = np.asarray(dates)
    labels = targets[:, 1] > 0.5
    difference = ((candidate[:, 1] >= 0.5) == labels).astype(float) - ((baseline[:, 1] >= 0.5) == labels).astype(float)
    ordered = sorted(set(dates))
    daily_accuracy = [difference[dates == date].mean() for date in ordered]
    candidate_ic = rank_ic_by_date(targets[:, 0], candidate[:, 0], dates)
    baseline_ic = rank_ic_by_date(targets[:, 0], baseline[:, 0], dates)
    common = sorted(set(candidate_ic) & set(baseline_ic))
    log_vol_difference = ((baseline[:, 2] - targets[:, 2]) ** 2) - ((candidate[:, 2] - targets[:, 2]) ** 2)
    daily_log_vol = [log_vol_difference[dates == date].mean() for date in ordered]
    return {
        "direction_accuracy_gain": block_interval(daily_accuracy, block, replicates),
        "daily_rank_ic_gain": block_interval([candidate_ic[d] - baseline_ic[d] for d in common], block, replicates) if common else None,
        "log_volatility_mse_reduction": block_interval(daily_log_vol, block, replicates),
        "interpretation": "Positive favors candidate; exploratory intervals are not corrected for multiple comparisons.",
    }
