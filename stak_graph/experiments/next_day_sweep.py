"""Matched next-trading-day OHLC regression; no SPY labels, inputs, or metrics."""
import argparse
import copy
import html
import json
from pathlib import Path
import shutil
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
from stak_graph.models.benchmark import graph_edges, EXTERNAL
from stak_graph.models.next_day import NextDayModel, FrozenHead, price_predictions
from .next_day_data import build_samples, metrics, TARGETS
from .train import make_tensor_data, predict
from .model_sweep import save, no_events, VARIANTS
from .metrics import block_interval
from .data import sha256_file


def prepare(args, config):
    root = args.source / 'dataset'
    calendar = TradingCalendar.from_csv(root / 'calendar.csv')
    bars = read_bars(root / 'prices.csv', calendar)
    registry = CompanyRegistry.from_csv(root / 'companies.csv')
    events = list(read_jsonl(args.expanded / 'events.jsonl'))
    cutoff = max(session.prediction_at for session in calendar.sessions if session.session_date <= config['sample_end'])
    snapshot = GraphBuilder(calendar, bars, registry, events, [], EventPolicy()).build(cutoff, len(calendar.sessions))
    study = build_samples(snapshot, bars, calendar, config)
    vectors, manifest = load_vectors(args.expanded / 'embeddings')
    if manifest['encoder'].get('simulated') or snapshot['contains_synthetic_data']:
        raise ValueError('Real data and real embeddings required')
    data = make_tensor_data(snapshot, vectors, manifest['dimension'], study, 'cuda', config)
    graph_edges(data, snapshot)
    return study, data, snapshot, bars, calendar


def loss(prediction, target, scale):
    return ((prediction - target) / scale).square().mean()


def fit(model, name, seed, study, data, config, output, frozen=None):
    checkpoint = output / f'checkpoints/{name}_seed{seed}.pt'
    history_path = output / f'histories/{name}_seed{seed}.json'
    if checkpoint.exists() and history_path.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['state_dict'])
        return
    windows = torch.as_tensor(study['windows'], device='cuda')
    target = torch.as_tensor(study['log_targets'], dtype=torch.float32, device='cuda')
    scale = torch.as_tensor(study['scaler']['target_scale'], dtype=torch.float32, device='cuda')
    train = torch.as_tensor(np.flatnonzero(study['masks']['train']), device='cuda')
    validation = torch.as_tensor(np.flatnonzero(study['masks']['validation']), device='cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state = float('inf'), 0, None
    history, start = [], time.monotonic()
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = train[torch.randperm(len(train), generator=generator).cuda()]
        losses = []
        for group in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(frozen[group]) if frozen is not None else model(data, windows[group])
            objective = loss(prediction, target[group], scale)
            if not torch.isfinite(objective):
                raise ValueError('Nonfinite OHLC regression objective')
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(objective.detach()))
        model.eval()
        with torch.no_grad():
            prediction = (torch.cat([model(frozen[group]) for group in validation.split(256)]) if frozen is not None
                          else torch.as_tensor(predict(model, data, windows, validation, batch_size=256), device='cuda'))
            validation_loss = float(loss(prediction, target[validation], scale))
        history.append({'epoch': epoch, 'training_loss': float(np.mean(losses)), 'validation_loss': validation_loss})
        if validation_loss < best - 1e-5:
            best, best_epoch = validation_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({'variant': name, 'seed': seed, **history[-1], 'selected_epoch': best_epoch}), flush=True)
        if epoch - best_epoch >= config['patience']:
            break
    if best_state is None:
        raise ValueError('No valid regression checkpoint')
    torch.save({'state_dict': best_state, 'variant': name, 'seed': seed, 'target_names': list(TARGETS),
                'horizon_sessions': 1, 'scaler': study['scaler'], 'selected_epoch': best_epoch}, checkpoint)
    save(history_path, {'epochs': history, 'selected_epoch': best_epoch, 'validation_loss': best,
                        'elapsed_seconds': time.monotonic() - start, 'checkpoint_sha256': sha256_file(checkpoint)})


def chronos_inputs(data, study, snapshot, output):
    values = data['events'].cpu().numpy()
    eligible = np.array([event['known_at'][:10] <= '2017-12-31' for event in snapshot['nodes']['event']])
    mean = values[eligible].mean(0)
    _, _, components = np.linalg.svd(values[eligible] - mean, full_matrices=False)
    components = components[:8]
    projected = torch.as_tensor((values - mean) @ components.T, device='cuda')
    event = projected.new_zeros((len(data['prices']), 8))
    event.index_add_(0, data['edge_day'], projected[data['edge_event']] * data['edge_weight'][:, None])
    count = projected.new_zeros((len(event), 1))
    count.index_add_(0, data['edge_day'], torch.ones((len(data['edge_day']), 1), device='cuda'))
    event /= count.clamp_min(1).sqrt()
    correlation = torch.zeros_like(data['prices'])
    correlation.index_add_(0, data['corr_target'], data['prices'][data['corr_source']] * data['corr_weight'][:, None])
    return torch.cat((data['prices'], event, correlation), -1).cpu()


def chronos_pipeline():
    sys.path.insert(0, str(EXTERNAL / 'chronos/src'))
    from chronos import Chronos2Pipeline
    checkpoint = json.loads((EXTERNAL / 'chronos_checkpoint.json').read_text())
    pipeline = Chronos2Pipeline.from_pretrained(checkpoint['model'], revision=checkpoint['revision'],
                                               device_map='cuda', cache_dir='/tmp/stak-graph-models',
                                               local_files_only=True, dtype=torch.float32)
    return pipeline, checkpoint


def frozen_features(context, study, output):
    path = output / 'chronos_features.npz'
    if path.exists():
        return torch.as_tensor(np.load(path)['features'], device='cuda')
    pipeline, checkpoint = chronos_pipeline()
    encoded = []
    for offset in range(0, len(study['windows']), 32):
        windows = study['windows'][offset:offset + 32]
        inputs = context[windows].transpose(1, 2)
        embeddings, _ = pipeline.embed(inputs, batch_size=256, context_length=30)
        pooled = torch.stack([torch.cat((item[:, -2].mean(0), item[:, :-1].mean((0, 1)))) for item in embeddings])
        encoded.append(torch.cat((pooled, inputs.mean(-1), inputs.std(-1)), -1).float().numpy())
        if offset % 1024 == 0:
            print(json.dumps({'stage': 'chronos_encoding', 'completed': offset, 'total': len(study['windows'])}), flush=True)
    values = np.concatenate(encoded)
    mean, std = values[study['masks']['train']].mean(0), values[study['masks']['train']].std(0)
    std[std < 1e-6] = 1
    features = ((values - mean) / std).clip(-10, 10).astype(np.float32)
    np.savez_compressed(path, features=features, mean=mean, scale=std)
    save(output / 'chronos_protocol.json', {'checkpoint': checkpoint, 'context_variables': context.shape[1],
         'market_inputs': False, 'mode': 'frozen encoder plus supervised four-price head',
         'pretraining_overlap': 'unknown', 'scalers_fit': 'train only'})
    del pipeline; torch.cuda.empty_cache()
    return torch.as_tensor(features, device='cuda')


def native_forecasts(context, study, snapshot, bars, output):
    pipeline, checkpoint = chronos_pipeline()
    lookup = {(bar.company_id, bar.session_date): bar for bar in bars}
    levels = np.array([[getattr(lookup[node['company_id'], node['session_date']], name) for name in TARGETS]
                       for node in snapshot['nodes']['stock_day']], dtype=np.float32)
    indices = np.flatnonzero(study['masks']['test'])
    forecasts = []
    for offset in range(0, len(indices), 32):
        inputs = []
        for index in indices[offset:offset + 32]:
            window = study['windows'][index]
            inputs.append({'target': levels[window].T,
                           'past_covariates': {str(j): context[window, j] for j in range(context.shape[1])}})
        _, median = pipeline.predict_quantiles(inputs, prediction_length=1, quantile_levels=[.5], batch_size=256,
                                                context_length=30, cross_learning=False)
        forecasts.append(torch.stack(median).cpu().numpy()[:, :, 0].astype(np.float64))
        if offset % 1024 == 0:
            print(json.dumps({'stage': 'chronos_native', 'completed': offset, 'total': len(indices)}), flush=True)
    raw = np.concatenate(forecasts)
    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError('Invalid native OHLC forecast')
    valid = raw.copy()
    valid[:, 1] = np.maximum.reduce((raw[:, 0], raw[:, 1], raw[:, 3]))
    valid[:, 2] = np.minimum.reduce((raw[:, 0], raw[:, 2], raw[:, 3]))
    np.savez_compressed(output / 'chronos_native_raw.npz', raw_prices=raw, prices=valid)
    save(output / 'chronos_native_protocol.json', {'checkpoint': checkpoint, 'horizon_sessions': 1,
         'targets': list(TARGETS), 'cross_learning': False, 'future_covariates': None, 'market_inputs': False,
         'candle_projection_fraction': float((np.abs(raw - valid).max(1) > 1e-10).mean()),
         'pretraining_overlap': 'unknown; no uncontaminated zero-shot claim'})
    del pipeline; torch.cuda.empty_cache()
    return valid


def report(output, summary):
    rows = []
    for name, result in summary['models'].items():
        metric = result['metrics']
        feature = metric['per_feature']
        rows.append('<tr><th>' + html.escape(name) + '</th>' + ''.join(f'<td>{feature[key]["MAE_percent_of_today_close"]:.3f}%</td>' for key in TARGETS)
                    + f'<td>{metric["mean_OHLC_MAE_percent_of_today_close"]:.3f}%</td><td>{metric["next_close_direction_accuracy"] * 100:.2f}%</td></tr>')
    document = '''<!doctype html><meta charset="utf-8"><title>STAK-Graph next-day OHLC</title>
    <style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#173044}
    table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}
    th{background:#eef4f8}.note{background:#fff3d6;padding:16px}pre{white-space:pre-wrap;font-size:12px}</style>
    <h1>STAK-Graph: next-trading-day stock OHLC forecasts</h1>
    <p>Predict tomorrow’s open, high, low and close using only information available by today’s cutoff.
    SPY is absent from labels, inputs and evaluation. Volume is historical input only, never a target.</p>
    <p class="note">The earlier five-day market-relative classification experiment answers a different question.
    Its accuracy is not evidence for this task. These new results use the previously inspected 2019 period and remain exploratory.</p>
    <p>Lower price error is better. The four error columns report mean absolute error as a percentage of today’s observed close.
    They are not classification accuracy. The last column is a secondary check: whether tomorrow’s close is above today’s close.</p>
    <table><tr><th>Model adaptation</th><th>Open MAE</th><th>High MAE</th><th>Low MAE</th><th>Close MAE</th><th>Mean OHLC MAE</th><th>Close direction</th></tr>'''
    document += ''.join(rows) + '</table><p>Supervised rows average price predictions across three fixed seeds. Native Chronos-2 uses its pretrained median forecasts and has an unmatched pretraining budget.</p>'
    document += '<p>Per-feature price-unit MAE/RMSE, MAPE, per-stock errors, paired calendar-block intervals, source hashes and protocols are provided below. Small price errors must be compared with persistence; they do not mean “99% prediction accuracy.”</p>'
    document += '<details><summary>Full metrics and protocol</summary><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre></details>'
    (output / 'results.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('checkpoints', 'histories'):
        (args.output / name).mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    config = {'sample_start': '2016-01-01', 'sample_end': '2019-12-31', 'train_end': '2017-12-31',
              'validation_end': '2018-12-31', 'lookback_sessions': 30, 'horizon_sessions': 1,
              'max_epochs': 40, 'patience': 8, 'seeds': [7, 19, 41], 'targets': list(TARGETS), 'SPY': False}
    study, data, snapshot, bars, calendar = prepare(args, config)
    source_paths = [args.source / 'dataset/prices.csv', args.source / 'dataset/calendar.csv',
                    args.expanded / 'events.jsonl', args.expanded / 'embeddings/manifest.json']
    plan = {'config': config, 'source_hashes': {str(path): sha256_file(path) for path in source_paths},
            'implementation_hashes': {str(path): sha256_file(path) for path in Path(__file__).parents[1].rglob('*.py')},
            'variants': [f'{graph}_{temporal}' for graph, temporal in VARIANTS] + ['chronos2_frozen_head', 'chronos2_native'],
            'splits': {name: int(mask.sum()) for name, mask in study['masks'].items()}, 'scaler': study['scaler'],
            'target': 'Next trading session OHLC prices, reconstructed from log(price_next / observed_close_today)',
            'loss': 'Mean squared log-ratio error divided by training-only target variances, equal weight per OHLC field',
            'selection': 'Validation loss only; all checkpoints selected before test scoring',
            'geometry': 'Positive prices; predicted high >= predicted open/close; predicted low <= predicted open/close',
            'persistence_baselines': ['last observed OHLC', 'today close repeated for all four outputs'],
            'price_basis': 'Existing FNSPID retrospectively adjusted OHLC; not contemporaneous unadjusted exchange quotes',
            'previous_market_relative_results_apply': False}
    if (args.output / 'plan.json').exists():
        if json.loads((args.output / 'plan.json').read_text()) != plan:
            raise ValueError('Frozen plan changed: use a new output folder')
    else:
        save(args.output / 'plan.json', plan)
    save(args.output / 'scaler.json', study['scaler'])
    for graph, temporal in VARIANTS:
        selected = no_events(data) if graph == 'gat_no_events' else data
        for seed in config['seeds']:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = NextDayModel('gat' if graph == 'gat_no_events' else graph, temporal, study['scaler']).cuda()
            fit(model, f'{graph}_{temporal}', seed, study, selected, config, args.output)
            del model
    context = chronos_inputs(data, study, snapshot, args.output)
    frozen = frozen_features(context, study, args.output)
    for seed in config['seeds']:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = FrozenHead(frozen.shape[1], study['scaler']).cuda()
        fit(model, 'chronos2_frozen_head', seed, study, data, config, args.output, frozen)
        del model
    save(args.output / 'validation_selection_complete.json', {'complete': True, 'test_used_for_selection': False})
    test = torch.as_tensor(np.flatnonzero(study['masks']['test']), device='cuda')
    windows = torch.as_tensor(study['windows'], device='cuda')
    anchor = study['anchors'][study['masks']['test']]
    truth = study['targets'][study['masks']['test']]
    dates = np.array([row['session_date'] for row in study['metadata']])[study['masks']['test']]
    companies = np.array([row['company_id'] for row in study['metadata']])[study['masks']['test']]
    predictions, result = {}, {}
    for graph, temporal in VARIANTS + [('chronos2', 'frozen_head')]:
        name = f'{graph}_{temporal}'
        selected = no_events(data) if graph == 'gat_no_events' else data
        individual = []
        for seed in config['seeds']:
            model = (FrozenHead(frozen.shape[1], study['scaler']) if graph == 'chronos2' else
                     NextDayModel('gat' if graph == 'gat_no_events' else graph, temporal, study['scaler'])).cuda()
            checkpoint = torch.load(args.output / f'checkpoints/{name}_seed{seed}.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['state_dict']); model.eval()
            with torch.no_grad():
                raw = (torch.cat([model(frozen[group]) for group in test.split(256)]).cpu().numpy() if graph == 'chronos2'
                       else predict(model, selected, windows, test, batch_size=256))
            price = price_predictions(raw, anchor)
            predictions[f'{name}_seed{seed}'] = price
            individual.append(price); del model
        ensemble = np.mean(individual, axis=0)
        predictions[name] = ensemble
        result[name] = {'metrics': metrics(truth, ensemble, anchor, companies),
                        'seed_metrics': [metrics(truth, value, anchor, companies) for value in individual]}
    predictions['chronos2_native'] = native_forecasts(context, study, snapshot, bars, args.output)
    predictions['last_OHLC'] = study['previous_ohlc'][study['masks']['test']]
    predictions['last_close'] = np.repeat(anchor[:, None], 4, axis=1)
    for name in ('chronos2_native', 'last_OHLC', 'last_close'):
        result[name] = {'metrics': metrics(truth, predictions[name], anchor, companies)}
    comparisons = {}
    for name, value in result.items():
        error = np.abs((predictions[name] - truth) / anchor[:, None]).mean(1)
        baseline_error = np.abs((predictions['last_close'] - truth) / anchor[:, None]).mean(1)
        daily = [(baseline_error[dates == day] - error[dates == day]).mean() for day in sorted(set(dates))]
        comparisons[name] = block_interval(daily)
    event_error = np.abs((predictions['gat_no_events_gru'] - truth) / anchor[:, None]).mean(1)
    event_with = np.abs((predictions['gat_gru'] - truth) / anchor[:, None]).mean(1)
    comparisons['event_effect_GAT_GRU_MAE_reduction'] = block_interval([(event_error[dates == day] - event_with[dates == day]).mean() for day in sorted(set(dates))])
    np.savez_compressed(args.output / 'test_predictions.npz', targets=truth, anchors=anchor, dates=dates, companies=companies, **predictions)
    summary = {'target': plan['target'], 'SPY_used': False, 'horizon_sessions': 1, 'target_names': list(TARGETS),
               'models': result, 'paired_MAE_reduction_vs_last_close': comparisons, 'split_counts': plan['splits'],
               'primary_metric': 'Mean OHLC absolute price error as a fraction of observed close_today; lower is better',
               'secondary_direction': 'Sign of predicted close_next minus observed close_today; equality is non-up',
               'limitations': ['2019 test reused; exploratory.', 'Retrospectively adjusted price basis.',
                              'Chronos pretraining overlap unknown.', 'Original headline availability dates unverified.',
                              'No numerical forecast error is reported as 100 minus error percent accuracy.']}
    save(args.output / 'summary.json', summary)
    report(args.output, summary)
    print(json.dumps({name: value['metrics']['mean_OHLC_MAE_percent_of_today_close'] for name, value in result.items()}), flush=True)


if __name__ == '__main__':
    main()
