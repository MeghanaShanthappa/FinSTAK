"""Independently reconstruct sweep predictions and test causal graph isolation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from stak_graph.models.benchmark import ForecastAdapter
from .model_sweep import build, no_events, save
from .train import predict, restore_predictions
from .data import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((args.output / 'plan.json').read_text())
    summary = json.loads((args.output / 'summary.json').read_text())
    for path, expected in plan['source_hashes'].items():
        if sha256_file(path) != expected:
            raise ValueError('Changed dataset source: ' + path)
    for path, expected in plan['implementation_hashes'].items():
        if sha256_file(path) != expected:
            raise ValueError('Changed implementation during training: ' + path)
    data, study, snapshot = build(args, plan['config'])
    arrays = np.load(args.output / 'test_predictions.npz', allow_pickle=False)
    np.testing.assert_array_equal(arrays['targets'], study['targets'][study['masks']['test']])
    test = torch.as_tensor(np.flatnonzero(study['masks']['test']), device='cuda')
    windows = torch.as_tensor(study['windows'], device='cuda')
    test_metadata = [row for row in study['metadata'] if row['split'] == 'test']
    np.testing.assert_array_equal(arrays['dates'], [row['session_date'] for row in test_metadata])
    checked = []
    for name, result in summary['models'].items():
        prediction = arrays[name]
        independent_accuracy = float(((prediction[:, 1] >= .5) == (arrays['targets'][:, 1] > .5)).mean())
        if independent_accuracy != result['ensemble']['direction_accuracy']:
            raise ValueError('Accuracy reconstruction failed: ' + name)
        frame = pd.DataFrame({'day': arrays['dates'], 'actual': arrays['targets'][:, 0], 'predicted': prediction[:, 0]})
        rank_ics = []
        for _, group in frame.groupby('day'):
            left, right = group['actual'].rank(), group['predicted'].rank()
            if left.std() > 0 and right.std() > 0:
                rank_ics.append(float(left.corr(right)))
        np.testing.assert_allclose(np.mean(rank_ics), result['ensemble']['mean_daily_rank_ic'], atol=1e-12, rtol=0)
        for seed in plan['config']['seeds']:
            history = json.loads((args.output / f'histories/{name}_seed{seed}.json').read_text())
            checkpoint = args.output / f'checkpoints/{name}_seed{seed}.pt'
            if sha256_file(checkpoint) != history['summary']['checkpoint_sha256']:
                raise ValueError('Changed checkpoint: ' + name)
            ensemble = np.mean([arrays[f'{name}_seed{s}'] for s in plan['config']['seeds']], axis=0)
            np.testing.assert_allclose(ensemble, prediction, atol=0, rtol=0)
        if name == 'chronos2_frozen_head':
            features = torch.as_tensor(np.load(args.output / 'chronos_features.npz')['features'], device='cuda')
            model = torch.nn.Sequential(torch.nn.LayerNorm(features.shape[1]), torch.nn.Linear(features.shape[1], 32),
                                        torch.nn.GELU(), torch.nn.Dropout(.1), torch.nn.Linear(32, 3)).cuda()
            checkpoint = torch.load(args.output / f'checkpoints/{name}_seed7.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['state_dict']); model.eval()
            with torch.no_grad():
                raw = torch.cat([model(features[indices]) for indices in test.split(256)]).cpu().numpy()
        else:
            graph, temporal = name.rsplit('_', 1)
            selected = no_events(data) if graph == 'gat_no_events' else data
            model = ForecastAdapter('gat' if graph == 'gat_no_events' else graph, temporal).cuda()
            checkpoint = torch.load(args.output / f'checkpoints/{name}_seed7.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['state_dict']); model.eval()
            raw = predict(model, selected, windows, test, batch_size=256)
            # Perturb all information strictly later than the earliest test
            # cutoff while retaining graph structure and earlier observations.
            first = min(row['session_date'] for row in test_metadata)
            early = torch.as_tensor([i for i, row in enumerate(study['metadata'])
                                     if row['split'] == 'test' and row['session_date'] == first], device='cuda')
            future_days = torch.tensor([node['session_date'] > first for node in snapshot['nodes']['stock_day']], device='cuda')
            future_events = torch.tensor([node['known_at'][:10] > first for node in snapshot['nodes']['event']], device='cuda')
            changed = dict(selected)
            changed['prices'] = selected['prices'].clone()
            changed['events'] = selected['events'].clone()
            changed['prices'][future_days] += 1000
            changed['events'][future_events] -= 1000
            before = predict(model, selected, windows, early, batch_size=256)
            after = predict(model, changed, windows, early, batch_size=256)
            np.testing.assert_allclose(before, after, atol=1e-6, rtol=0)
        restored = restore_predictions(raw, study['scaler'])
        np.testing.assert_allclose(restored, arrays[f'{name}_seed7'], atol=1e-7, rtol=1e-6)
        checked.append(name)
        print('verified', name, flush=True)
        del model
    save(args.output / 'verification.json', {'passed': True, 'models_reconstructed': checked,
         'test_labels_order_accuracy_and_rank_ic_reconstructed': True, 'checkpoint_hashes_verified': True,
         'dataset_and_training_implementation_unchanged': True,
         'future_price_and_event_perturbation_passed_for_all_trained_graph_adapters': True,
         'chronos_frozen_head_reconstructed': True})


if __name__ == '__main__':
    main()
