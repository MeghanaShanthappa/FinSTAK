"""Replay a selected historical ensemble, without feeding labels to inference."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--stage', choices=('data_stage', 'architecture_stage'), required=True)
    parser.add_argument('--fold', choices=('2021', '2022', '2023'), required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    frozen = args.study / 'source_snapshot'
    if not (frozen / 'manifest.json').exists():
        raise ValueError('Frozen implementation is required')
    sys.path.insert(0, str(frozen))
    import numpy as np
    import torch
    from stak_graph.experiments.scale_panel import PanelBatches
    from stak_graph.experiments.scale_run import FOLDS
    from stak_graph.models.scale_finstak import ScaleFinSTAK
    from stak_graph.experiments.data import sha256_file
    from stak_graph.experiments.scale_io import write_json
    torch.set_num_threads(4)
    folder = args.study / args.stage / args.fold
    lock = json.loads((folder / 'validation_selection_complete.json').read_text())
    if not lock['complete'] or lock['test_used_for_checkpoint_selection']:
        raise ValueError('Complete validation selection before replaying forecasts')
    plan = json.loads((args.study / args.stage / 'plan.json').read_text())
    if args.variant not in plan['variants']:
        raise ValueError('Variant is not part of the frozen stage')
    for name, expected in plan['code_hashes'].items():
        if sha256_file(frozen / name) != expected:
            raise ValueError('Frozen implementation checksum mismatch: ' + name)
    for name, key in [('panel.npz', 'panel_sha256'), ('event_arrays.npz', 'event_arrays_sha256'),
                      ('event_arrays_finbert.npz', 'finbert_arrays_sha256')]:
        if sha256_file(args.study / 'packed' / name) != plan[key]:
            raise ValueError('Research input checksum mismatch: ' + name)
    panel = PanelBatches(args.study / 'packed', *FOLDS[args.fold], device=args.device)
    panel.select_encoder(args.variant)
    matches = np.flatnonzero(panel.dates == args.date)
    if len(matches) != 1 or matches[0] not in panel.splits['test']:
        raise ValueError('Choose a prediction date inside the saved historical test fold')
    t = int(matches[0])
    data = panel.batch([t])
    data.pop('target')
    data.pop('volume_target')
    anchor = panel.raw[t, :, 3]
    prices, volumes, hashes = [], [], {}
    for seed in plan['seeds']:
        path = folder / f'{args.variant}_seed{seed}.pt'
        history = json.loads((folder / f'{args.variant}_seed{seed}_history.json').read_text())
        if sha256_file(path) != history['checkpoint_sha256']:
            raise ValueError('Selected checkpoint checksum mismatch: ' + str(seed))
        saved = torch.load(path, weights_only=True, map_location=args.device)
        model = ScaleFinSTAK(panel.scaler, panel.c, panel.events.shape[1], args.variant,
                            category_count=panel.category_count).to(args.device).eval()
        model.load_state_dict(saved['state_dict'])
        with torch.inference_mode():
            price, volume = model(data)
        prices.append(anchor[:, None] * np.exp(price.cpu().numpy().astype(np.float64)))
        if volume is not None:
            volumes.append(volume.cpu().numpy())
        hashes[str(seed)] = sha256_file(path)
    predicted = np.mean(prices, axis=0)
    rows = []
    for i, company in enumerate(panel.companies):
        row = {'company': str(company), **dict(zip(('open', 'high', 'low', 'close'), predicted[i].tolist()))}
        if volumes:
            # Explicitly expose the uncorrected log-space ensemble forecast.
            row['volume_naive'] = float(max(0., np.expm1(panel.volume_mean[t, i] + panel.volume_scale[t, i] * np.mean(volumes, axis=0)[i])))
        rows.append(row)
    write_json(args.output, {'mode': 'historical checkpoint replay', 'prediction_date': args.date,
               'target_date': str(panel.dates[t + 1]), 'stage': args.stage, 'variant': args.variant,
               'training_label_cutoff': panel.scaler['fit_latest_date'], 'checkpoint_sha256': hashes,
               'future_labels_fed_to_model': False, 'price_basis': 'Retrospectively adjusted research OHLC',
               'forecasts': rows})
    print(json.dumps({'forecasts': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
