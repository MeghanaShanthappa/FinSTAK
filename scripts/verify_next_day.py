"""Independently verify the corrected next-session, four-price experiment."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from stak_graph.experiments.next_day_sweep import prepare
from stak_graph.experiments.model_sweep import no_events, save
from stak_graph.experiments.train import predict
from stak_graph.experiments.data import sha256_file
from stak_graph.models.next_day import NextDayModel, FrozenHead, price_predictions
from stak_graph.models.benchmark import EXTERNAL

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--expanded', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
plan = json.loads((args.output / 'plan.json').read_text())
summary = json.loads((args.output / 'summary.json').read_text())
assert plan['config']['SPY'] is False and plan['config']['targets'] == ['open', 'high', 'low', 'close']
for path, expected in {**plan['source_hashes'], **plan['implementation_hashes']}.items():
    assert sha256_file(path) == expected, path
study, data, snapshot, bars, calendar = prepare(args, plan['config'])
saved = np.load(args.output / 'test_predictions.npz', allow_pickle=False)
mask = study['masks']['test']
np.testing.assert_array_equal(saved['targets'], study['targets'][mask])
np.testing.assert_array_equal(saved['anchors'], study['anchors'][mask])
assert data['prices'].shape[1] == 14
assert study['scaler']['fit_latest_date'] <= plan['config']['train_end']
metadata = [row for row in study['metadata'] if row['split'] == 'test']
np.testing.assert_array_equal(saved['dates'], [row['session_date'] for row in metadata])
frame = pd.read_csv(args.source / 'dataset/prices.csv').set_index(['company_id', 'session_date'])
independent = np.array([frame.loc[(row['company_id'], row['target_date']), ['open', 'high', 'low', 'close']].to_numpy(dtype=float)
                        for row in metadata])
np.testing.assert_allclose(saved['targets'], independent, atol=1e-10, rtol=1e-12)
for row in metadata:
    day = calendar.index[row['session_date']]
    assert row['target_date'] == calendar.sessions[day + 1].session_date
    assert row['prediction_at'] < row['target_open_at']
for name, result in summary['models'].items():
    values = saved[name]
    error = values - saved['targets']
    relative = error / saved['anchors'][:, None]
    metric = result['metrics']
    np.testing.assert_allclose(np.abs(relative).mean() * 100, metric['mean_OHLC_MAE_percent_of_today_close'], atol=1e-12)
    for index, key in enumerate(['open', 'high', 'low', 'close']):
        np.testing.assert_allclose(np.abs(error[:, index]).mean(), metric['per_feature'][key]['MAE_price_units'], atol=1e-12)
        np.testing.assert_allclose(np.sqrt((error[:, index] ** 2).mean()), metric['per_feature'][key]['RMSE_price_units'], atol=1e-12)
    assert ((values[:, 3] > saved['anchors']) == (saved['targets'][:, 3] > saved['anchors'])).mean() == metric['next_close_direction_accuracy']
    assert (values[:, 1] >= np.maximum(values[:, 0], values[:, 3]) - 1e-10).all()
    assert (values[:, 2] <= np.minimum(values[:, 0], values[:, 3]) + 1e-10).all()
windows = torch.as_tensor(study['windows'], device='cuda')
test = torch.as_tensor(np.flatnonzero(mask), device='cuda')
first = min(row['session_date'] for row in metadata)
early = torch.as_tensor([i for i, row in enumerate(study['metadata']) if row['split'] == 'test' and row['session_date'] == first], device='cuda')
future_days = torch.tensor([node['session_date'] > first for node in snapshot['nodes']['stock_day']], device='cuda')
future_events = torch.tensor([node['known_at'][:10] > first for node in snapshot['nodes']['event']], device='cuda')
features = torch.as_tensor(np.load(args.output / 'chronos_features.npz')['features'], device='cuda')
checked = []
for name in summary['models']:
    if name in ('chronos2_native', 'last_OHLC', 'last_close'):
        continue
    selected = no_events(data) if name == 'gat_no_events_gru' else data
    if name == 'chronos2_frozen_head':
        model = FrozenHead(features.shape[1], study['scaler']).cuda()
    else:
        graph, temporal = name.rsplit('_', 1)
        model = NextDayModel('gat' if graph == 'gat_no_events' else graph, temporal, study['scaler']).cuda()
    for seed in plan['config']['seeds']:
        checkpoint_path = args.output / f'checkpoints/{name}_seed{seed}.pt'
        history = json.loads((args.output / f'histories/{name}_seed{seed}.json').read_text())
        assert sha256_file(checkpoint_path) == history['checkpoint_sha256']
    checkpoint = torch.load(args.output / f'checkpoints/{name}_seed7.pt', map_location='cpu', weights_only=True)
    assert checkpoint['target_names'] == ['open', 'high', 'low', 'close'] and checkpoint['horizon_sessions'] == 1
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    with torch.no_grad():
        raw = (torch.cat([model(features[group]) for group in test.split(256)]).cpu().numpy() if name == 'chronos2_frozen_head'
               else predict(model, selected, windows, test, batch_size=256))
    reconstructed = price_predictions(raw, saved['anchors'])
    np.testing.assert_allclose(reconstructed, saved[f'{name}_seed7'], atol=1e-6, rtol=1e-7)
    ensemble = np.mean([saved[f'{name}_seed{seed}'] for seed in plan['config']['seeds']], axis=0)
    np.testing.assert_allclose(ensemble, saved[name], atol=0, rtol=0)
    if name != 'chronos2_frozen_head':
        changed = dict(selected)
        changed['prices'] = selected['prices'].clone(); changed['events'] = selected['events'].clone()
        changed['prices'][future_days] += 1000
        changed['events'][future_events] -= 1000
        np.testing.assert_allclose(predict(model, selected, windows, early, 256), predict(model, changed, windows, early, 256), atol=1e-6, rtol=0)
    del model
    checked.append(name)
    print('verified', name, flush=True)
native = np.load(args.output / 'chronos_native_raw.npz')
raw, projected = native['raw_prices'], native['raw_prices'].copy()
projected[:, 1] = np.maximum.reduce((raw[:, 0], raw[:, 1], raw[:, 3]))
projected[:, 2] = np.minimum.reduce((raw[:, 0], raw[:, 2], raw[:, 3]))
np.testing.assert_array_equal(projected, saved['chronos2_native'])
np.testing.assert_array_equal(saved['last_OHLC'], study['previous_ohlc'][mask])
np.testing.assert_array_equal(saved['last_close'], np.repeat(saved['anchors'][:, None], 4, axis=1))
from stak_graph.experiments.next_day_sweep import chronos_pipeline
pipeline, _ = chronos_pipeline()
torch.manual_seed(2183)
first_input = {'target': torch.rand(4, 30) + 100.,
               'past_covariates': {str(index): torch.randn(30) for index in range(36)}}
unrelated_input = {'target': torch.rand(4, 30) * 1000.,
                   'past_covariates': {str(index): torch.randn(30) * 1000 for index in range(36)}}
alone = pipeline.predict_quantiles([first_input], prediction_length=1, quantile_levels=[.5], batch_size=256,
                                    context_length=30, cross_learning=False)[1][0]
with_other = pipeline.predict_quantiles([first_input, unrelated_input], prediction_length=1, quantile_levels=[.5], batch_size=256,
                                         context_length=30, cross_learning=False)[1][0]
torch.testing.assert_close(alone, with_other, atol=1e-4, rtol=1e-6)
history = torch.randn(1, 36, 30)
unrelated_history = torch.randn(1, 36, 30) * 1000
single = pipeline.embed(history, batch_size=256, context_length=30)[0][0]
multiple = pipeline.embed(torch.cat((history, unrelated_history)), batch_size=256, context_length=30)[0][0]
torch.testing.assert_close(single, multiple, atol=1e-4, rtol=1e-5)
save(args.output / 'verification.json', {'passed': True, 'models_reconstructed': checked,
     'next_session_OHLC_labels_independently_joined_to_prices': True, 'SPY_used': False,
     'four_price_outputs_and_one_session_horizon_verified': True, 'future_price_and_event_invariance': True,
     'source_implementation_and_checkpoint_hashes_verified': True, 'per_feature_errors_and_direction_recomputed': True,
     'native_price_projection_and_persistence_reconstructed': True,
     'unrelated_tasks_cannot_change_Chronos_native_forecast_or_frozen_encoding': True})
