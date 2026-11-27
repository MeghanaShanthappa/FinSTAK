"""Next-trading-session OHLC regression. No market-relative targets or inputs."""
import math
import numpy as np

from stak_graph.graph.market import FEATURE_NAMES
from .dataset import split_of

TARGETS = ('open', 'high', 'low', 'close')


def next_target(company, session_index, prices):
    current, following = prices[company, session_index], prices[company, session_index + 1]
    values = np.array([getattr(following, name) for name in TARGETS], dtype=np.float64)
    return values, current.close


def build_samples(snapshot, bars, calendar, config):
    days = snapshot['nodes']['stock_day']
    prices = {(bar.company_id, calendar.index[bar.session_date]): bar for bar in bars}
    lookup = {(day['company_id'], day['session_index']): index for index, day in enumerate(days)}
    features = np.array([day['features'] for day in days], dtype=np.float32)
    windows, targets, anchors, previous, metadata = [], [], [], [], []
    for day in days:
        company, index = day['company_id'], day['session_index']
        first = index - config['lookback_sessions'] + 1
        if first < 0 or not config['sample_start'] <= day['session_date'] <= config['sample_end']:
            continue
        if any((company, session) not in lookup for session in range(first, index + 1)):
            continue
        if (company, index + 1) not in prices:
            raise ValueError('Next-session target unavailable; incomplete samples cannot silently disappear')
        values, anchor = next_target(company, index, prices)
        target_date = calendar.sessions[index + 1].session_date
        windows.append([lookup[company, session] for session in range(first, index + 1)])
        targets.append(values); anchors.append(anchor)
        previous.append([getattr(prices[company, index], name) for name in TARGETS])
        metadata.append({'company_id': company, 'session_date': day['session_date'],
                         'prediction_at': day['timestamp'], 'target_date': target_date,
                         'target_open_at': calendar.sessions[index + 1].open_at,
                         'label_end_at': calendar.sessions[index + 1].close_at,
                         'split': split_of(day['session_date'], target_date, config)})
    windows = np.asarray(windows, dtype=np.int64)
    targets, anchors = np.asarray(targets), np.asarray(anchors)
    log_targets = np.log(targets / anchors[:, None])
    masks = {name: np.array([row['split'] == name for row in metadata])
             for name in ('train', 'validation', 'test', 'purged')}
    if any(not masks[name].any() for name in ('train', 'validation', 'test')):
        raise ValueError('All chronological splits must be nonempty')
    fit_rows = np.unique(windows[masks['train']])
    feature_mean, feature_scale = features[fit_rows].mean(0), features[fit_rows].std(0)
    feature_scale[feature_scale < 1e-8] = 1
    target_mean, target_scale = log_targets[masks['train']].mean(0), log_targets[masks['train']].std(0)
    if (target_scale < 1e-8).any():
        raise ValueError('Training price changes have zero variance')
    upper = log_targets[:, 1] - np.maximum(log_targets[:, 0], log_targets[:, 3])
    lower = np.minimum(log_targets[:, 0], log_targets[:, 3]) - log_targets[:, 2]
    geometry_scales = [float(upper[masks['train']].mean()), float(lower[masks['train']].mean())]
    if min(geometry_scales) <= 0:
        raise ValueError('Training candle ranges must have positive mean width')
    scaler = {'feature_names': list(FEATURE_NAMES), 'mean': feature_mean.tolist(), 'scale': feature_scale.tolist(),
              'fit_latest_date': max(days[index]['session_date'] for index in fit_rows),
              'target_names': list(TARGETS), 'target_mean': target_mean.tolist(), 'target_scale': target_scale.tolist(),
              'geometry_scales': geometry_scales, 'anchor': 'observed close on prediction day',
              'market_relative': False, 'horizon_sessions': 1}
    return {'features': ((features - feature_mean) / feature_scale).clip(-10, 10),
            'windows': windows, 'targets': targets, 'anchors': anchors, 'log_targets': log_targets,
            'previous_ohlc': np.asarray(previous), 'metadata': metadata, 'masks': masks, 'scaler': scaler}


def metrics(truth, prediction, anchors, companies):
    if truth.shape != prediction.shape or truth.shape[1] != 4 or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError('Expected finite positive OHLC forecasts with matching labels')
    if (prediction[:, 1] < np.maximum(prediction[:, 0], prediction[:, 3]) - 1e-10).any() or (prediction[:, 2] > np.minimum(prediction[:, 0], prediction[:, 3]) + 1e-10).any():
        raise ValueError('Forecast candle geometry is invalid')
    error = prediction - truth
    relative = error / anchors[:, None]
    per_feature = {}
    for index, name in enumerate(TARGETS):
        per_feature[name] = {'MAE_price_units': float(np.abs(error[:, index]).mean()),
                             'RMSE_price_units': float(np.sqrt((error[:, index] ** 2).mean())),
                             'MAPE_percent': float((np.abs(error[:, index]) / truth[:, index]).mean() * 100),
                             'MAE_percent_of_today_close': float(np.abs(relative[:, index]).mean() * 100),
                             'RMSE_percent_of_today_close': float(np.sqrt((relative[:, index] ** 2).mean()) * 100)}
    return {'samples': len(truth), 'per_feature': per_feature,
            'mean_OHLC_MAE_percent_of_today_close': float(np.abs(relative).mean() * 100),
            'mean_OHLC_RMSE_percent_of_today_close': float(np.sqrt((relative ** 2).mean()) * 100),
            'next_close_direction_accuracy': float(((prediction[:, 3] > anchors) == (truth[:, 3] > anchors)).mean()),
            'per_stock_MAE_percent_of_today_close': {company: float(np.abs(relative[np.asarray(companies) == company]).mean() * 100)
                                                    for company in sorted(set(companies))}}
