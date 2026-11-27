"""Frozen node-level transformer encoder comparison, specified before scoring."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .finstak_run import prepare, fit
from .timesfm_run import score, render, HYBRIDS
from .timesfm_features import context_rows
from .train import predict
from .data import sha256_file
from .model_sweep import save
from .metrics import block_interval
from stak_graph.graph.market import TradingCalendar, read_bars
from stak_graph.models.timesfm_encoder import TimesFMEncoderFinSTAK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--foundation-run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--context', type=int, choices=(32, 64, 128))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ('checkpoints', 'histories'):
        (args.output / folder).mkdir()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    foundation_plan = json.loads((args.foundation_run / 'plan.json').read_text())
    selected = json.loads((args.foundation_run / 'native_validation_selection.json').read_text())
    context = args.context or selected['selected_context']['price']
    config = foundation_plan['config']
    plan = {'config': config, 'context': context, 'modes': HYBRIDS,
            'foundation_plan_sha256': sha256_file(args.foundation_run / 'plan.json'),
            'source_hashes': foundation_plan['data_hashes'],
            'implementation_hashes': {str(p): sha256_file(p) for p in Path(__file__).parents[1].rglob('*.py')},
            'representation': 'Mean across four OHLC variates at the final context patch of TimesFM3 transformer_output; frozen backbone, train-only whitened PCA to 32 dimensions',
            'selection': 'Fixed context candidate; aggregate context selection uses validation MAE only' if args.context else 'Context from native price validation; checkpoints from validation only; no test selection',
            'primary_comparison': 'encoder_graph vs encoder_local',
            'checkpoint_revision': foundation_plan['checkpoint_revision']}
    save(args.output / 'plan.json', plan)
    study, data, snapshot, _ = prepare(args, config)
    days = snapshot['nodes']['stock_day']
    calendar = TradingCalendar.from_csv(args.source / 'dataset/calendar.csv')
    bars = read_bars(args.source / 'dataset/prices.csv', calendar)
    lookup = {(b.company_id, b.session_date): b for b in bars}
    prices = np.array([[getattr(lookup[d['company_id'], d['session_date']], f) for f in ('open', 'high', 'low', 'close')] for d in days])
    rows = context_rows(days, study['windows'][:, -1], context, allow_short=True)
    from timesfm3 import TimesFM3Forecaster, ModelConfig
    foundation = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
                  revision=plan['checkpoint_revision'], cache_dir=str(args.cache), device='cuda', per_core_batch_size=16))
    embeddings = []
    start = time.monotonic()
    for offset in range(0, len(rows), 16):
        group = rows[offset:offset + 16]
        values = np.zeros((len(group), 4, context), dtype='float32')
        missing = np.ones_like(values, dtype=bool)
        for i, row in enumerate(group):
            values[i, :, -len(row):] = np.log(prices[row] / study['anchors'][offset + i]).T
            missing[i, :, -len(row):] = False
        with torch.inference_mode():
            _, aux = foundation.model.decode(torch.as_tensor(values, device='cuda'), horizon=64,
                          target_mask=torch.as_tensor(missing, device='cuda'), return_aux_outputs=True)
            hidden = aux['__call__:transformer_output']
            if hidden.shape[:2] != (len(group), 4) or hidden.shape[2] < context // 32:
                raise ValueError('Unexpected foundation hidden-state layout')
            embeddings.append(hidden[:, :, context // 32 - 1].mean(1).cpu().numpy())
        if offset % 320 == 0:
            print(json.dumps({'phase': 'frozen_encoder', 'completed': offset + len(group), 'total': len(rows), 'seconds': time.monotonic() - start}), flush=True)
    embeddings = np.concatenate(embeddings)
    del foundation
    torch.cuda.empty_cache()
    if not np.isfinite(embeddings).all():
        raise ValueError('Nonfinite encoder state')
    # GPU covariance eigendecomposition avoids an expensive full sample SVD.
    fit_values = torch.as_tensor(embeddings[study['masks']['train']], dtype=torch.float64, device='cuda')
    mean = fit_values.mean(0)
    centered = fit_values - mean
    eigenvalues, eigenvectors = torch.linalg.eigh(centered.T @ centered / len(centered))
    basis = eigenvectors[:, -32:]
    scales = eigenvalues[-32:].sqrt().clamp_min(1e-6)
    save(args.output / 'representation_audit.json', {
        'frozen_model_revision': plan['checkpoint_revision'],
        'original_dimension': int(embeddings.shape[1]), 'retained_dimension': 32,
        'training_samples': int(study['masks']['train'].sum()),
        'training_variance_retained': float(eigenvalues[-32:].sum() / eigenvalues.clamp_min(0).sum()),
        'fit_latest_date': max(r['session_date'] for r, selected in zip(study['metadata'], study['masks']['train']) if selected),
        'extraction_seconds': time.monotonic() - start,
        'short_context_examples': sum(len(row) < context for row in rows),
        'padding_policy': 'Leading zero values are masked; no repeated or future observations'})
    reduced = ((torch.as_tensor(embeddings, device='cuda', dtype=torch.float64) - mean) @ basis / scales).clamp(-10, 10).float()
    data['temporal_embeddings'] = torch.zeros((len(days), 32), device='cuda')
    data['temporal_embeddings'][torch.as_tensor(study['windows'][:, -1], device='cuda')] = reduced
    np.savez_compressed(args.output / 'frozen_encoder.npz', raw_embeddings=embeddings,
                        reduced=reduced.cpu().numpy(), mean=mean.cpu().numpy(), basis=basis.cpu().numpy(),
                        scale=scales.cpu().numpy(), fit_indices=np.flatnonzero(study['masks']['train']))
    del fit_values, centered, reduced
    torch.cuda.empty_cache()
    for mode in HYBRIDS:
        for seed in config['seeds']:
            torch.manual_seed(seed)
            model = TimesFMEncoderFinSTAK(study['scaler'], len(data['company_ids']), mode).to('cuda')
            fit(model, 'encoder_' + mode, seed, study, data, config, args.output)
            del model
    save(args.output / 'selection_complete.json', {'complete': True, 'test_used_for_selection': False})
    test_mask = study['masks']['test']
    test = torch.as_tensor(np.flatnonzero(test_mask), device='cuda')
    windows = torch.as_tensor(study['windows'], device='cuda')
    previous = np.load(args.foundation_run / 'test_predictions.npz', allow_pickle=False)
    predictions = {name: previous[name] for name in previous.files if name not in ('targets', 'anchors', 'dates', 'companies')}
    truth, anchors, dates, companies = [previous[name] for name in ('targets', 'anchors', 'dates', 'companies')]
    np.testing.assert_array_equal(truth, study['targets'][test_mask])
    audits, validation_models, validation_predictions = {}, {}, {}
    validation_mask = study['masks']['validation']
    validation = torch.as_tensor(np.flatnonzero(validation_mask), device='cuda')
    for mode in HYBRIDS:
        individual, development = [], []
        for seed in config['seeds']:
            model = TimesFMEncoderFinSTAK(study['scaler'], len(data['company_ids']), mode).to('cuda')
            model.load_state_dict(torch.load(args.output / 'checkpoints' / f'encoder_{mode}_seed{seed}.pt', weights_only=True, map_location='cuda')['state_dict'])
            development.append(study['anchors'][validation_mask, None] * np.exp(predict(model, data, windows, validation).astype('float64')))
            prediction = anchors[:, None] * np.exp(predict(model, data, windows, test).astype('float64'))
            individual.append(prediction)
            predictions[f'encoder_{mode}_seed{seed}'] = prediction
            if seed == config['seeds'][0]:
                future = torch.tensor([d['session_date'] > '2018-12-31' for d in days], device='cuda')
                changed = dict(data)
                for key in ('prices', 'temporal_embeddings'):
                    changed[key] = data[key].clone(); changed[key][future] += 100
                future_event = torch.tensor([e['known_at'][:10] > '2018-12-31' for e in snapshot['nodes']['event']], device='cuda')
                changed['events'] = data['events'].clone(); changed['events'][future_event] += 100
                past = torch.as_tensor(np.flatnonzero(study['masks']['validation'])[::31], device='cuda')
                delta = float(np.abs(predict(model, data, windows, past) - predict(model, changed, windows, past)).max())
                if delta > 1e-7:
                    raise ValueError('Future data altered earlier encoder forecasts')
                audits[mode] = delta
            del model
        predictions['encoder_' + mode] = np.mean(individual, 0)
        validation_predictions['encoder_' + mode] = np.mean(development, 0)
        validation_models['encoder_' + mode] = score(study, validation_mask, validation_predictions['encoder_' + mode])
    result = {name: score(study, test_mask, value) for name, value in predictions.items() if '_seed' not in name}
    comparisons = {}
    for baseline in ('encoder_local', 'encoder_direct', 'encoder_no_decay', 'hybrid_graph', 'original_no_decay'):
        difference = (np.abs(predictions[baseline] - truth) - np.abs(predictions['encoder_graph'] - truth)).mean(1) / anchors * 100
        comparisons['encoder_graph_vs_' + baseline] = block_interval([difference[dates == day].mean() for day in sorted(set(dates))])
    original_summary = json.loads((args.foundation_run / 'summary.json').read_text())
    limits = [s for s in original_summary['limitations'] if not s.startswith('Hybrid uses frozen forecast')]
    limits += ['Encoder route replaces GRU with frozen transformer states, train-only PCA and a learned projection. No foundation weight fine-tuning.',
               'The final context token is pooled across four variates; no actual future target values enter extraction.']
    summary = {'models': result, 'paired_MAE_reduction_percentage_points': comparisons,
               'selected_context': context, 'split_counts': original_summary['split_counts'],
               'future_input_audit': audits, 'validation_models': validation_models, 'limitations': limits}
    np.savez_compressed(args.output / 'validation_predictions.npz', targets=study['targets'][validation_mask],
                        anchors=study['anchors'][validation_mask], **validation_predictions)
    np.savez_compressed(args.output / 'test_predictions.npz', targets=truth, anchors=anchors, dates=dates, companies=companies, **predictions)
    save(args.output / 'summary.json', summary)
    render(args.output, summary)
    print(json.dumps({k: v['metrics']['mean_OHLC_MAE_percent_of_today_close'] for k, v in result.items()}), flush=True)


if __name__ == '__main__':
    main()
