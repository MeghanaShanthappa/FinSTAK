"""Verify frozen TimesFM representations and reconstruct every encoder model."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stak_graph.experiments.finstak_run import prepare
from stak_graph.experiments.timesfm_run import score, HYBRIDS
from stak_graph.experiments.timesfm_features import context_rows
from stak_graph.experiments.train import predict
from stak_graph.experiments.data import sha256_file
from stak_graph.experiments.model_sweep import save
from stak_graph.graph.market import TradingCalendar, read_bars
from stak_graph.models.timesfm_encoder import TimesFMEncoderFinSTAK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    plan = json.loads((args.output / 'plan.json').read_text())
    for group in ('source_hashes', 'implementation_hashes'):
        for filename, expected in plan[group].items():
            assert sha256_file(Path(filename)) == expected, filename
    study, data, snapshot, _ = prepare(args, plan['config'])
    saved = np.load(args.output / 'test_predictions.npz', allow_pickle=False)
    mask = study['masks']['test']
    np.testing.assert_array_equal(saved['targets'], study['targets'][mask])
    np.testing.assert_array_equal(saved['anchors'], study['anchors'][mask])
    frozen = np.load(args.output / 'frozen_encoder.npz', allow_pickle=False)
    np.testing.assert_array_equal(frozen['fit_indices'], np.flatnonzero(study['masks']['train']))
    np.testing.assert_allclose(frozen['mean'], frozen['raw_embeddings'][frozen['fit_indices']].mean(0, dtype='float64'), atol=1e-12)
    reduced = ((frozen['raw_embeddings'].astype('float64') - frozen['mean']) @ frozen['basis'] / frozen['scale']).clip(-10, 10).astype('float32')
    np.testing.assert_allclose(reduced, frozen['reduced'], atol=1e-5, rtol=1e-5)
    data['temporal_embeddings'] = torch.zeros((len(data['prices']), 32), device=args.device)
    data['temporal_embeddings'][torch.as_tensor(study['windows'][:, -1], device=args.device)] = torch.as_tensor(reduced, device=args.device)
    windows = torch.as_tensor(study['windows'], device=args.device)
    test = torch.as_tensor(np.flatnonzero(mask), device=args.device)
    summary = json.loads((args.output / 'summary.json').read_text())
    assert json.loads((args.output / 'selection_complete.json').read_text())['complete']
    for name, result in summary['models'].items():
        assert score(study, mask, saved[name]) == result, name
    for mode in HYBRIDS:
        individual = []
        for seed in plan['config']['seeds']:
            name = f'encoder_{mode}_seed{seed}'
            path = args.output / 'checkpoints' / (name + '.pt')
            history = json.loads((args.output / 'histories' / (name + '.json')).read_text())
            assert sha256_file(path) == history['checkpoint_sha256']
            state = torch.load(path, weights_only=True, map_location=args.device)
            assert state['config'] == plan['config']
            assert state['scaler'] == study['scaler']
            chosen = next(e for e in history['epochs'] if e['epoch'] == state['selected_epoch'])
            assert chosen['validation_loss'] <= min(e['validation_loss'] for e in history['epochs']) + 1e-5
            model = TimesFMEncoderFinSTAK(study['scaler'], len(data['company_ids']), mode).to(args.device)
            model.load_state_dict(state['state_dict'])
            value = saved['anchors'][:, None] * np.exp(predict(model, data, windows, test).astype('float64'))
            np.testing.assert_allclose(value, saved[name], rtol=1e-7, atol=1e-7)
            individual.append(value)
            del model
        np.testing.assert_allclose(np.mean(individual, 0), saved['encoder_' + mode], rtol=1e-7, atol=1e-7)
    # Re-extract a fixed sample from each chronological split from raw OHLC.
    calendar = TradingCalendar.from_csv(args.source / 'dataset/calendar.csv')
    bars = {(b.company_id, b.session_date): b for b in read_bars(args.source / 'dataset/prices.csv', calendar)}
    days = snapshot['nodes']['stock_day']
    prices = np.array([[getattr(bars[d['company_id'], d['session_date']], f) for f in ('open', 'high', 'low', 'close')] for d in days])
    examples = np.array([np.flatnonzero(study['masks'][s])[j] for s in ('train', 'validation', 'test') for j in (0, -1)])
    rows = context_rows(days, study['windows'][examples, -1], plan['context'], allow_short=True)
    values = np.zeros((len(rows), 4, plan['context']), dtype='float32')
    missing = np.ones_like(values, dtype=bool)
    for i, row in enumerate(rows):
        values[i, :, -len(row):] = np.log(prices[row] / study['anchors'][examples[i]]).T
        missing[i, :, -len(row):] = False
    from timesfm3 import TimesFM3Forecaster, ModelConfig
    foundation = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch', revision=plan['checkpoint_revision'], cache_dir=str(args.cache), device=args.device))
    with torch.inference_mode():
        _, aux = foundation.model.decode(torch.as_tensor(values, device=args.device), horizon=64,
                        target_mask=torch.as_tensor(missing, device=args.device), return_aux_outputs=True)
        actual = aux['__call__:transformer_output'][:, :, plan['context'] // 32 - 1].mean(1).cpu().numpy()
    np.testing.assert_allclose(actual, frozen['raw_embeddings'][examples], rtol=1e-4, atol=1e-4)
    assert all(v <= 1e-7 for v in summary['future_input_audit'].values())
    result = {'passed': True, 'encoder_checkpoints_reconstructed': 12, 'all_metrics_recomputed': True,
              'raw_foundation_embeddings_reextracted': len(examples), 'train_only_PCA_recomputed': True,
              'source_and_code_hashes_match': True, 'future_input_audit': summary['future_input_audit']}
    save(args.output / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
