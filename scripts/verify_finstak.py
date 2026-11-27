"""Reconstruct FinSTAK checkpoints, labels and metrics from recorded sources."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from stak_graph.experiments.finstak_run import prepare
from stak_graph.experiments.next_day_data import metrics, TARGETS
from stak_graph.experiments.finstak_data import tolerance_accuracy
from stak_graph.experiments.train import predict
from stak_graph.experiments.data import sha256_file
from stak_graph.experiments.model_sweep import save
from stak_graph.graph.market import TradingCalendar, read_bars
from stak_graph.models.finstak import FinSTAK
from stak_graph.models.next_day import price_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((args.output / 'plan.json').read_text())
    for group in ('source_hashes', 'implementation_hashes'):
        for path, checksum in plan[group].items():
            if sha256_file(Path(path)) != checksum:
                raise ValueError('Changed input or implementation: ' + path)
    if not json.loads((args.output / 'validation_selection_complete.json').read_text())['complete']:
        raise ValueError('Checkpoint selection not completed')
    config = plan['config']
    study, data, _, _ = prepare(args, config)
    saved = np.load(args.output / 'test_predictions.npz')
    mask = study['masks']['test']
    np.testing.assert_array_equal(saved['targets'], study['targets'][mask])
    np.testing.assert_array_equal(saved['anchors'], study['anchors'][mask])
    # Independent direct join against the next exchange session in the CSV.
    root = args.source / 'dataset'
    calendar = TradingCalendar.from_csv(root / 'calendar.csv')
    bars = {(b.company_id, b.session_date): b for b in read_bars(root / 'prices.csv', calendar)}
    for i, (company, day) in enumerate(zip(saved['companies'], saved['dates'])):
        next_day = calendar.sessions[calendar.index[str(day)] + 1].session_date
        bar = bars[str(company), next_day]
        np.testing.assert_array_equal(saved['targets'][i], [getattr(bar, name) for name in TARGETS])
        if saved['anchors'][i] != bars[str(company), str(day)].close:
            raise ValueError('Incorrect observed close anchor')
    windows = torch.as_tensor(study['windows'], device=args.device)
    test = torch.as_tensor(np.flatnonzero(mask), device=args.device)
    summary = json.loads((args.output / 'summary.json').read_text())
    checks = {}
    for mode in config['modes']:
        individual = []
        for seed in config['seeds']:
            checkpoint = args.output / f'checkpoints/{mode}_seed{seed}.pt'
            history = json.loads((args.output / f'histories/{mode}_seed{seed}.json').read_text())
            if sha256_file(checkpoint) != history['checkpoint_sha256']:
                raise ValueError('Changed checkpoint')
            model = FinSTAK(study['scaler'], companies=len(data['company_ids']), mode=mode).to(args.device)
            state = torch.load(checkpoint, weights_only=True, map_location=args.device)
            if state['config'] != config or state['scaler'] != study['scaler']:
                raise ValueError('Checkpoint protocol mismatch')
            selected = min(history['epochs'], key=lambda item: item['validation_loss'])
            # Training uses a 1e-5 improvement threshold, so very small differences
            # may intentionally retain an earlier near-equal checkpoint.
            chosen = next(row for row in history['epochs'] if row['epoch'] == state['selected_epoch'])
            if chosen['validation_loss'] > selected['validation_loss'] + 1e-5:
                raise ValueError('Checkpoint not selected by validation')
            model.load_state_dict(state['state_dict'])
            prices = price_predictions(predict(model, data, windows, test), saved['anchors'])
            np.testing.assert_allclose(prices, saved[f'{mode}_seed{seed}'], rtol=1e-7, atol=1e-7)
            individual.append(prices)
            del model
        ensemble = np.mean(individual, axis=0)
        np.testing.assert_allclose(ensemble, saved[mode], rtol=1e-7, atol=1e-7)
        actual_metrics = metrics(saved['targets'], saved[mode], saved['anchors'], saved['companies'])
        if actual_metrics != summary['models'][mode]['metrics']:
            raise ValueError('Stored metric mismatch')
        hits = tolerance_accuracy(saved['targets'], saved[mode], saved['anchors'])
        if hits != summary['models'][mode]['tolerance_accuracy']:
            raise ValueError('Stored tolerance accuracy mismatch')
        checks[mode] = {'reconstructed_seeds': config['seeds'], 'metrics_recomputed': True,
                        'tolerance_accuracy_recomputed': True}
    np.testing.assert_array_equal(saved['last_close'], np.repeat(saved['anchors'][:, None], 4, axis=1))
    np.testing.assert_array_equal(saved['last_OHLC'], study['previous_ohlc'][mask])
    audit = summary['causality_audit']
    if set(audit) != set(config['modes']) or any(v['future_perturbation_max_abs_logratio_change'] > 1e-7 for v in audit.values()):
        raise ValueError('Causality audit missing or failed')
    result = {'passed': True, 'samples_independently_joined': len(saved['targets']),
              'target_names': list(TARGETS), 'horizon_sessions': 1, 'input_dimension': data['prices'].shape[1],
              'source_and_implementation_hashes_match': True, 'models': checks,
              'causality_audit': audit, 'persistence_reconstructed': True}
    save(args.output / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
