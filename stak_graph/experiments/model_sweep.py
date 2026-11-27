"""Expanded-data graph/temporal comparison with validation-only selection.

All models use identical labels, chronological splits, historical windows and
source observations. Upstream forecasting models have a supervised three-output
readout; Chronos-2 is a separately identified frozen encoder plus trained head.
This is an exploratory architecture comparison, not source-paper reproduction.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from stak_graph.io import read_jsonl
from stak_graph.extraction.registry import CompanyRegistry
from stak_graph.embeddings.pipeline import load_vectors
from stak_graph.graph.builder import GraphBuilder, EventPolicy
from stak_graph.graph.market import TradingCalendar, read_bars
from stak_graph.models.benchmark import ForecastAdapter, graph_edges, EXTERNAL
from .data import sha256_file
from .dataset import build_samples
from .train import make_tensor_data, standardized_targets, restore_predictions, multitask_loss, predict
from .metrics import evaluate, block_interval, paired_comparison


VARIANTS = [('typed', 'gru'), ('gat', 'gru'), ('sagn', 'gru'), ('gat_no_events', 'gru'),
            ('gat', 'tft'), ('gat', 'xlstm'), ('gat', 'patchtst'), ('gat', 'itransformer'),
            ('gat', 'timemixer'), ('gat', 'timemachine')]


def save(path, value):
    path = Path(path)
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def build(args, config):
    prepared = args.source / 'dataset'
    calendar = TradingCalendar.from_csv(prepared / 'calendar.csv')
    bars = read_bars(prepared / 'prices.csv', calendar)
    benchmark = read_bars(prepared / 'benchmark.csv', calendar)
    registry = CompanyRegistry.from_csv(prepared / 'companies.csv')
    events = list(read_jsonl(args.expanded / 'events.jsonl'))
    cutoff = max(session.prediction_at for session in calendar.sessions if session.session_date <= config['sample_end'])
    snapshot = GraphBuilder(calendar, bars, registry, events, [],
                            EventPolicy(half_life_sessions=5, ttl_sessions=20)).build(cutoff, len(calendar.sessions))
    study = build_samples(snapshot, bars, benchmark, calendar, config)
    vectors, manifest = load_vectors(args.expanded / 'embeddings')
    if manifest['encoder'].get('simulated') or snapshot['contains_synthetic_data']:
        raise ValueError('Real source evidence and semantic embeddings required')
    data = make_tensor_data(snapshot, vectors, manifest['dimension'], study, 'cuda', config)
    graph_edges(data, snapshot)
    return data, study, snapshot


def no_events(data):
    result = dict(data)
    # Event nodes remain isolated with self loops; no path reaches a StockDay.
    selected = data['graph_relation'] != 0
    for name in ('graph_source', 'graph_target', 'graph_weight', 'graph_relation'):
        result[name] = data[name][selected]
    return result


def fit(model, name, seed, data, study, config, output, frozen_features=None):
    checkpoint = output / 'checkpoints' / f'{name}_seed{seed}.pt'
    metadata_path = output / 'histories' / f'{name}_seed{seed}.json'
    if checkpoint.exists() and metadata_path.exists():
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(payload['state_dict'])
        return model, json.loads(metadata_path.read_text())['summary']
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    windows = torch.as_tensor(study['windows'], device='cuda')
    targets = standardized_targets(study, 'cuda')
    training = torch.as_tensor(np.flatnonzero(study['masks']['train']), device='cuda')
    validation = torch.as_tensor(np.flatnonzero(study['masks']['validation']), device='cuda')
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, state = float('inf'), 0, None
    history, start = [], time.monotonic()
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = training[torch.randperm(len(training), generator=generator).cuda()]
        losses = []
        for indices in order.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(frozen_features[indices]) if frozen_features is not None else model(data, windows[indices])
            loss = multitask_loss(prediction, targets[indices])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            if frozen_features is None:
                values = torch.as_tensor(predict(model, data, windows, validation, batch_size=256), device='cuda')
            else:
                values = torch.cat([model(frozen_features[indices]) for indices in validation.split(256)])
            validation_loss = float(multitask_loss(values, targets[validation]))
        history.append({'epoch': epoch, 'training_loss': float(np.mean(losses)), 'validation_loss': validation_loss})
        if validation_loss < best - 1e-5:
            best, best_epoch = validation_loss, epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({'variant': name, 'seed': seed, **history[-1], 'best_epoch': best_epoch}), flush=True)
        if epoch - best_epoch >= config['early_stopping_patience']:
            break
    if state is None:
        raise ValueError('No finite selected checkpoint')
    model.load_state_dict(state)
    torch.save({'state_dict': state, 'variant': name, 'seed': seed, 'config': config,
                'selected_epoch': best_epoch}, checkpoint)
    summary = {'variant': name, 'seed': seed, 'validation_loss': best, 'selected_epoch': best_epoch,
               'epochs_run': len(history), 'elapsed_seconds': time.monotonic() - start,
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'checkpoint_sha256': sha256_file(checkpoint)}
    save(metadata_path, {'summary': summary, 'epochs': history})
    return model, summary


def chronos_features(data, study, snapshot, output):
    """Frozen official encoder; PCA/scalers fit on training observations only."""
    path = output / 'chronos_features.npz'
    if path.exists():
        return torch.as_tensor(np.load(path)['features'], device='cuda')
    sys.path.insert(0, str(EXTERNAL / 'chronos/src'))
    from chronos import Chronos2Pipeline
    checkpoint = json.loads((EXTERNAL / 'chronos_checkpoint.json').read_text())
    pipeline = Chronos2Pipeline.from_pretrained(checkpoint['model'], revision=checkpoint['revision'],
                                               device_map='cuda', cache_dir='/tmp/stak-graph-models',
                                               local_files_only=True, dtype=torch.float32)
    event_values = data['events'].cpu().numpy()
    training_events = np.array([event['known_at'][:10] <= '2017-12-31' for event in snapshot['nodes']['event']])
    mean = event_values[training_events].mean(0)
    _, _, components = np.linalg.svd(event_values[training_events] - mean, full_matrices=False)
    component_count = min(8, len(components))
    projected = torch.as_tensor((event_values - mean) @ components[:component_count].T, device='cuda')
    messages = projected.new_zeros((len(data['prices']), component_count))
    messages.index_add_(0, data['edge_day'], projected[data['edge_event']] * data['edge_weight'][:, None])
    counts = projected.new_zeros((len(data['prices']), 1))
    counts.index_add_(0, data['edge_day'], torch.ones((len(data['edge_day']), 1), device='cuda'))
    messages /= counts.clamp_min(1).sqrt()
    correlation = torch.zeros_like(data['prices'])
    correlation.index_add_(0, data['corr_target'], data['prices'][data['corr_source']] * data['corr_weight'][:, None])
    features = torch.cat((data['prices'], messages, correlation), -1)
    windows = torch.as_tensor(study['windows'], device='cuda')
    encoded = []
    for offset in range(0, len(windows), 32):
        inputs = features[windows[offset:offset + 32]].transpose(1, 2).cpu()
        embeddings, _ = pipeline.embed(inputs, batch_size=256, context_length=30)
        # Pool each time-series task independently: no cross-sample attention.
        pooled = torch.stack([torch.cat((item[:, -2].mean(0), item[:, :-1].mean((0, 1)))) for item in embeddings])
        moments = torch.cat((inputs.mean(-1), inputs.std(-1)), -1)
        encoded.append(torch.cat((pooled, moments), -1).float().numpy())
        if offset % 512 == 0:
            print(json.dumps({'stage': 'chronos_frozen_encoding', 'completed': offset, 'total': len(windows)}), flush=True)
    values = np.concatenate(encoded)
    training = study['masks']['train']
    feature_mean = values[training].mean(0)
    feature_scale = values[training].std(0)
    feature_scale[feature_scale < 1e-6] = 1
    normalized = ((values - feature_mean) / feature_scale).clip(-10, 10).astype(np.float32)
    np.savez_compressed(path, features=normalized, mean=feature_mean, scale=feature_scale, pca_mean=mean, pca_components=components[:component_count])
    save(output / 'chronos_protocol.json', {'checkpoint': checkpoint, 'mode': 'frozen pretrained encoder + supervised multitask head',
        'graph': 'fixed causal event aggregation and rolling correlation covariates; no learned GAT',
        'feature_dimensions': features.shape[1], 'head_input_dimension': normalized.shape[1],
        'pca_fit_cutoff': '2017-12-31', 'head_scaler_fit_split': 'train',
        'pretraining_overlap_with_historical_prices': 'unknown; no uncontaminated zero-shot claim',
        'not_a_zero_shot_forecast': True})
    del pipeline
    torch.cuda.empty_cache()
    return torch.as_tensor(normalized, device='cuda')


def render(output, summary):
    rows = []
    for name, result in summary['models'].items():
        metric = result['ensemble']
        ci = result['accuracy_interval']['ci95']
        seed_text = (f'{result["seed_mean_accuracy"] * 100:.2f} ± {result["seed_sd_accuracy"] * 100:.2f}'
                     if result.get('repeated_seeds_applicable', True) else '— (pretrained)')
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{metric["direction_accuracy"] * 100:.2f}%</td>'
                    f'<td>{seed_text}</td>'
                    f'<td>{ci[0] * 100:.2f}–{ci[1] * 100:.2f}</td><td>{metric["balanced_accuracy"]:.4f}</td>'
                    f'<td>{metric["roc_auc"]:.4f}</td></tr>')
    report = '''<!doctype html><meta charset="utf-8"><title>STAK-Graph expanded-source comparison</title>
    <style>body{font:16px system-ui;max-width:1120px;margin:48px auto;padding:0 20px;color:#172635}
    table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}
    th{background:#edf3f8}p{line-height:1.6}.note{background:#fff4d7;padding:16px}</style>
    <h1>STAK-Graph: expanded-source model comparison</h1>
    <p>16 US stocks; training 2016–2017, validation 2018, exploratory test 2019.
    Direction means positive five-session stock return minus SPY, measured from next-session open.
    Each supervised model uses three fixed seeds (7, 19, 41), validation loss for checkpoint selection,
    and a 0.5 decision threshold. No architecture is selected using test accuracy.</p>
    <p class="note">2019 has been examined in previous experiments. These scores are exploratory,
    and do not establish a paper-worthy improvement. Confidence intervals resample calendar blocks,
    keeping stocks from the same date together. Intervals are not adjusted for multiple comparisons.</p>
    <table><thead><tr><th>Model adaptation</th><th>Ensemble accuracy</th><th>Seed mean ± SD (%)</th>
    <th>95% interval (%)</th><th>Balanced accuracy</th><th>ROC AUC</th></tr></thead><tbody>'''
    report += ''.join(rows) + '</tbody></table><h2>Protocols and data</h2><pre>'
    report += html.escape(json.dumps({key: value for key, value in summary.items() if key != 'models'}, indent=2)) + '</pre>'
    (output / 'results.html').write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('checkpoints', 'histories'):
        (args.output / name).mkdir(exist_ok=True)
    config = json.loads((args.source / 'dataset/config.json').read_text())
    config.update({'hidden_dimension': 32, 'seeds': [7, 19, 41], 'max_epochs': 40,
                   'early_stopping_patience': 8, 'batch_size': 256, 'learning_rate': .001,
                   'weight_decay': .001, 'experiment_id': 'expanded_public_sources_model_sweep_v1'})
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data, study, snapshot = build(args, config)
    path = args.output / 'plan.json'
    plan = {'config': config, 'variants': [f'{graph}_{temporal}' for graph, temporal in VARIANTS] + ['chronos2_frozen_head'],
            'graph_statistics': snapshot['stats'], 'split_counts': {key: int(value.sum()) for key, value in study['masks'].items()},
            'expanded_manifest': json.loads((args.expanded / 'manifest.json').read_text()),
            'upstream_sources': {file.stem: json.loads(file.read_text()) for file in EXTERNAL.glob('*_source.json')},
            'source_hashes': {str(p): sha256_file(p) for p in [args.expanded / 'events.jsonl', args.expanded / 'embeddings/manifest.json',
                             args.source / 'dataset/prices.csv', args.source / 'dataset/benchmark.csv']},
            'implementation_hashes': {str(p): sha256_file(p) for p in Path(__file__).parents[1].rglob('*.py')},
            'protocol': 'Three seeds per model; train-only scalers; purged chronological boundaries; validate all checkpoints before test evaluation.',
            'temporal_adapters': 'Official xLSTM all-mLSTM stack, official TimeMachine with official Mamba reference scan, TS-Library PatchTST/iTransformer/TimeMixer forecasting bodies with supervised readouts.',
            'SAGN': 'Feature-hop and attention adaptation; no self-label enhancement or label propagation.',
            'additional_predeclared_model': 'Official Chronos-2 native open/close median forecasting, no fine-tuning, evaluated in a separate module after checkpoint selection.',
            'GOFA': 'Assessed separately; not represented by a substitute architecture or invented score.'}
    if path.exists():
        old = json.loads(path.read_text())
        if old != plan:
            raise ValueError('Recorded benchmark plan changed; use a new output directory')
    else:
        save(path, plan)
    windows = torch.as_tensor(study['windows'], device='cuda')
    for graph, temporal in VARIANTS:
        name = f'{graph}_{temporal}'
        selected_data = no_events(data) if graph == 'gat_no_events' else data
        for seed in config['seeds']:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = ForecastAdapter('gat' if graph == 'gat_no_events' else graph, temporal).cuda()
            model, _ = fit(model, name, seed, selected_data, study, config, args.output)
            del model
    frozen = chronos_features(data, study, snapshot, args.output)
    for seed in config['seeds']:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = nn.Sequential(nn.LayerNorm(frozen.shape[1]), nn.Linear(frozen.shape[1], 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 3)).cuda()
        model, _ = fit(model, 'chronos2_frozen_head', seed, data, study, config, args.output, frozen)
        del model
    save(args.output / 'validation_selection_complete.json', {'complete': True, 'test_evaluated_during_selection': False})
    test = torch.as_tensor(np.flatnonzero(study['masks']['test']), device='cuda')
    targets = study['targets'][study['masks']['test']]
    dates = np.array([row['session_date'] for row in study['metadata']])[study['masks']['test']]
    predictions, results = {}, {}
    all_variants = VARIANTS + [('chronos2', 'frozen_head')]
    for graph, temporal in all_variants:
        name = f'{graph}_{temporal}'
        selected_data = no_events(data) if graph == 'gat_no_events' else data
        seed_predictions, seed_metrics = [], []
        for seed in config['seeds']:
            if graph == 'chronos2':
                model = nn.Sequential(nn.LayerNorm(frozen.shape[1]), nn.Linear(frozen.shape[1], 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 3)).cuda()
            else:
                model = ForecastAdapter('gat' if graph == 'gat_no_events' else graph, temporal).cuda()
            checkpoint = torch.load(args.output / 'checkpoints' / f'{name}_seed{seed}.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['state_dict']); model.eval()
            with torch.no_grad():
                raw = torch.cat([model(frozen[indices]) for indices in test.split(256)]).cpu().numpy() if graph == 'chronos2' else predict(model, selected_data, windows, test, batch_size=256)
            restored = restore_predictions(raw, study['scaler'])
            seed_predictions.append(restored)
            seed_metrics.append(evaluate(targets, restored, dates))
            predictions[f'{name}_seed{seed}'] = restored
            del model
        ensemble = np.mean(seed_predictions, axis=0)
        predictions[name] = ensemble
        correct = (ensemble[:, 1] >= .5) == (targets[:, 1] > .5)
        daily_correct = [correct[dates == day].mean() for day in sorted(set(dates))]
        results[name] = {'ensemble': evaluate(targets, ensemble, dates), 'seed_metrics': seed_metrics,
                         'seed_mean_accuracy': float(np.mean([m['direction_accuracy'] for m in seed_metrics])),
                         'seed_sd_accuracy': float(np.std([m['direction_accuracy'] for m in seed_metrics], ddof=1)),
                         'accuracy_interval': block_interval(daily_correct)}
    comparisons = {name: paired_comparison(targets, prediction, predictions['gat_gru'], dates)
                   for name, prediction in predictions.items() if '_seed' not in name and name != 'gat_gru'}
    comparisons['event_effect_gat_gru_vs_no_events'] = paired_comparison(targets, predictions['gat_gru'], predictions['gat_no_events_gru'], dates)
    np.savez_compressed(args.output / 'test_predictions.npz', targets=targets, dates=dates, **predictions)
    summary = {'models': results, 'paired_comparisons': comparisons,
               'split_counts': plan['split_counts'], 'graph_statistics': snapshot['stats'],
               'source_observations': plan['expanded_manifest']['combined_source_observations'],
               'GDELT_status': 'Discovery metadata acquired; not treated as extracted events.',
               'Wikidata_status': 'Current snapshot not used in historical graph.',
               'Chronos2_status': 'Frozen pretrained encoder plus supervised head; not zero-shot and not GAT; pretraining contamination unknown.',
               'GOFA_status': 'Not a forecasting result; implementation assessment recorded separately.',
               'interpretation': 'Exploratory reused 2019 test; no guaranteed improvement, paper-readiness or original-paper reproduction claim.'}
    save(args.output / 'summary.json', summary)
    render(args.output, summary)
    print(json.dumps({name: value['ensemble']['direction_accuracy'] for name, value in results.items()}), flush=True)


if __name__ == '__main__':
    main()
