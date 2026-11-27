"""Official Chronos-2 native forecasts, separate from the supervised encoder head.

Forecast stock/SPY open and close for the next five sessions; compute the exact
same next-open-to-fifth-close excess-return direction from forecast medians.
No test labels, future covariates, or cross-task learning enter inference.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from stak_graph.models.benchmark import EXTERNAL
from stak_graph.graph.market import TradingCalendar, read_bars
from .model_sweep import build, save, render
from .metrics import evaluate, block_interval, paired_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((args.output / 'plan.json').read_text())
    data, study, snapshot = build(args, plan['config'])
    checkpoint = json.loads((EXTERNAL / 'chronos_checkpoint.json').read_text())
    sys.path.insert(0, str(EXTERNAL / 'chronos/src'))
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(checkpoint['model'], revision=checkpoint['revision'], device_map='cuda',
                                               cache_dir='/tmp/stak-graph-models', local_files_only=True, dtype=torch.float32)
    prepared = args.source / 'dataset'
    calendar = TradingCalendar.from_csv(prepared / 'calendar.csv')
    prices = {(bar.company_id, bar.session_date): bar for bar in read_bars(prepared / 'prices.csv', calendar)}
    spy = {bar.session_date: bar for bar in read_bars(prepared / 'benchmark.csv', calendar)}
    levels = np.array([[prices[node['company_id'], node['session_date']].open,
                        prices[node['company_id'], node['session_date']].close,
                        spy[node['session_date']].open, spy[node['session_date']].close]
                       for node in snapshot['nodes']['stock_day']], dtype=np.float32)
    fitted = np.load(args.output / 'chronos_features.npz')
    event_projection = (data['events'].cpu().numpy() - fitted['pca_mean']) @ fitted['pca_components'].T
    projected = torch.as_tensor(event_projection, device='cuda')
    event = projected.new_zeros((len(levels), projected.shape[1]))
    event.index_add_(0, data['edge_day'], projected[data['edge_event']] * data['edge_weight'][:, None])
    count = projected.new_zeros((len(levels), 1))
    count.index_add_(0, data['edge_day'], torch.ones((len(data['edge_day']), 1), device='cuda'))
    event /= count.clamp_min(1).sqrt()
    correlation = torch.zeros_like(data['prices'])
    correlation.index_add_(0, data['corr_target'], data['prices'][data['corr_source']] * data['corr_weight'][:, None])
    covariates = torch.cat((data['prices'], event, correlation), -1).cpu().numpy()
    test = np.flatnonzero(study['masks']['test'])
    score, volatility, forecast_batches = [], [], []
    for offset in range(0, len(test), 32):
        selected = test[offset:offset + 32]
        inputs = []
        for index in selected:
            window = study['windows'][index]
            inputs.append({'target': levels[window].T,
                           'past_covariates': {f'past_{j}': covariates[window, j] for j in range(covariates.shape[1])}})
        quantiles, median = pipeline.predict_quantiles(inputs, prediction_length=5, quantile_levels=[.5],
                                                       batch_size=256, context_length=30, cross_learning=False)
        forecasts = torch.stack(median).cpu().numpy().astype(np.float64)
        if not np.isfinite(forecasts).all() or (forecasts <= 0).any():
            raise ValueError('Native price forecast is nonpositive or nonfinite; no silent clipping permitted')
        forecast_batches.append(forecasts)
        score.extend(np.log(forecasts[:, 1, -1] / forecasts[:, 0, 0]) - np.log(forecasts[:, 3, -1] / forecasts[:, 2, 0]))
        increments = np.concatenate((np.log(forecasts[:, 1, :1] / forecasts[:, 0, :1]),
                                     np.diff(np.log(forecasts[:, 1, :]), axis=1)), axis=1)
        volatility.extend(np.log(np.maximum(np.sqrt((increments ** 2).mean(1)), 1e-8)))
        if offset % 512 == 0:
            print(json.dumps({'stage': 'chronos_native_forecast', 'completed': offset, 'total': len(test)}), flush=True)
    score = np.array(score)
    # Only a monotonic score mapping for shared metric code, not a calibrated
    # probability estimate. Direction is determined solely by the score sign.
    probability = 1 / (1 + np.exp(-(score / study['scaler']['return_scale']).clip(-40, 40)))
    predictions = np.stack((score, probability, volatility), axis=-1)
    saved = np.load(args.output / 'test_predictions.npz')
    targets, dates = saved['targets'], saved['dates']
    np.testing.assert_array_equal(targets, study['targets'][study['masks']['test']])
    result = evaluate(targets, predictions, dates)
    correct = (score >= 0) == (targets[:, 1] > .5)
    daily = [correct[dates == day].mean() for day in sorted(set(dates))]
    summary = json.loads((args.output / 'summary.json').read_text())
    summary['models']['chronos2_native_forecast'] = {'ensemble': result, 'seed_metrics': [],
        'seed_mean_accuracy': result['direction_accuracy'], 'seed_sd_accuracy': 0.,
        'repeated_seeds_applicable': False, 'accuracy_interval': block_interval(daily)}
    summary['paired_comparisons']['chronos2_native_forecast'] = paired_comparison(targets, predictions, saved['gat_gru'], dates)
    summary['Chronos2_native_status'] = 'Official pretrained median forecasts with historical event/price/correlation covariates; no fine-tuning. Later pretraining may overlap these historical prices. Probability mapping is uncalibrated.'
    protocol = {'checkpoint': checkpoint, 'target_series': ['stock_open', 'stock_close', 'SPY_open', 'SPY_close'],
                'prediction_horizon': 5, 'context_sessions': 30, 'covariates': 'past only: 18 price features + 8 train-fit event PCs + 18 rolling-correlation channels',
                'cross_learning': False, 'future_covariates': None, 'supervised_training': False,
                'selection_or_calibration_on_test': False, 'pretraining_overlap_with_2016_2019_prices': 'unknown',
                'direction_rule': 'positive log(predicted_stock_close_day5 / predicted_stock_open_day1) minus the corresponding SPY log ratio',
                'inputs_are_point_in_time': 'source-availability policy and dated SEC evidence; original headline dates remain unverified',
                'same_target_as_supervised_models': True, 'same_architecture_training_budget_as_scratch_models': False}
    save(args.output / 'chronos_native_protocol.json', protocol)
    np.savez_compressed(args.output / 'chronos_native_predictions.npz', targets=targets, dates=dates,
                        predictions=predictions, price_forecasts=np.concatenate(forecast_batches))
    save(args.output / 'summary.json', summary)
    render(args.output, summary)
    print(json.dumps({'chronos2_native_direction_accuracy': result['direction_accuracy']}), flush=True)


if __name__ == '__main__':
    main()
