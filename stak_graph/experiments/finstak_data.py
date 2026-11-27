"""Causal rolling price normalization and bounded stock-local histories."""
import numpy as np
import torch


def rolling_ohlcv(snapshot, bars, window=90):
    """One shared price mean/scale across OHLC; separate log-volume scale.

    Windows include today, which is observed at the existing end-of-day cutoff.
    Before 90 observations an expanding prefix is used, with its size returned.
    """
    if window < 2:
        raise ValueError('Rolling window must be at least two')
    lookup = {(bar.company_id, bar.session_date): bar for bar in bars}
    days = snapshot['nodes']['stock_day']
    result = np.zeros((len(days), 5), dtype=np.float32)
    counts = np.zeros(len(days), dtype=np.int64)
    groups = {}
    for i, day in enumerate(days):
        groups.setdefault(day['company_id'], []).append(i)
    for indices in groups.values():
        indices.sort(key=lambda i: days[i]['session_index'])
        observed = []
        for i in indices:
            day = days[i]
            bar = lookup[day['company_id'], day['session_date']]
            observed.append((day['session_index'], [bar.open, bar.high, bar.low, bar.close], np.log1p(bar.volume)))
            observed = [x for x in observed if x[0] > day['session_index'] - window]
            prices = np.asarray([x[1] for x in observed], dtype=np.float64)
            volume = np.asarray([x[2] for x in observed])
            # Relative floors handle a flat stock without explosive division.
            price_scale = max(float(prices.std()), abs(float(prices.mean())) * 1e-6, 1e-8)
            result[i, :4] = (prices[-1] - prices.mean()) / price_scale
            result[i, 4] = (volume[-1] - volume.mean()) / max(float(volume.std()), 1e-6)
            counts[i] = len(observed)
    return result.clip(-10, 10), counts


def attach_history(data, snapshot, lookback=30):
    days = snapshot['nodes']['stock_day']
    lookup = {(day['company_id'], day['session_index']): i for i, day in enumerate(days)}
    rows, masks = [], []
    for i, day in enumerate(days):
        history = [lookup.get((day['company_id'], s), -1)
                   for s in range(day['session_index'] - lookback + 1, day['session_index'] + 1)]
        rows.append([i if j < 0 else j for j in history])
        masks.append([j >= 0 for j in history])
    device = data['prices'].device
    data['local_windows'] = torch.tensor(rows, device=device)
    data['local_mask'] = torch.tensor(masks, device=device)
    return data


def tolerance_accuracy(truth, prediction, anchors, tolerances=(.005, .01, .02)):
    truth, prediction, anchors = map(np.asarray, (truth, prediction, anchors))
    if (truth.ndim != 2 or truth.shape[1] != 4 or prediction.shape != truth.shape
            or anchors.shape != (len(truth),) or not len(truth)
            or not all(np.isfinite(v).all() for v in (truth, prediction, anchors))
            or (anchors <= 0).any()):
        raise ValueError('Expected matching finite OHLC arrays and positive observed-close anchors')
    error = np.abs(prediction - truth) / anchors[:, None]
    result = {}
    for tolerance in tolerances:
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Tolerance must be finite and positive')
        hit = error <= tolerance
        result[f'{100 * tolerance:g}%'] = {
            'per_feature_percent': dict(zip(('open', 'high', 'low', 'close'), (100 * hit.mean(0)).tolist())),
            'pooled_fields_percent': float(100 * hit.mean()),
            'whole_candle_percent': float(100 * hit.all(1).mean())}
    return result
