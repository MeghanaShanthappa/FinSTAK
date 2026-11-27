"""Reproducible FinSTAK proposal pilot on the existing real-data panel.

Run in a new directory. This does not replace the previous frozen experiment.
"""
import argparse
import html
import json
from pathlib import Path
import time
import numpy as np
import torch

from stak_graph.io import read_jsonl
from stak_graph.extraction.registry import CompanyRegistry
from stak_graph.extraction.schema import EVENT_CATEGORIES
from stak_graph.embeddings.pipeline import load_vectors
from stak_graph.graph.builder import GraphBuilder, EventPolicy
from stak_graph.graph.market import TradingCalendar, read_bars
from stak_graph.models.finstak import FinSTAK
from stak_graph.models.next_day import price_predictions
from .next_day_data import build_samples, metrics
from .finstak_data import rolling_ohlcv, attach_history, tolerance_accuracy
from .train import make_tensor_data, predict
from .next_day_sweep import loss
from .model_sweep import save
from .metrics import block_interval
from .data import sha256_file

MODES = ('events', 'no_events', 'no_decay', 'lstm', 'tgcn')


def prepare(args, config):
    root = args.source / 'dataset'
    calendar = TradingCalendar.from_csv(root / 'calendar.csv')
    bars = read_bars(root / 'prices.csv', calendar)
    registry = CompanyRegistry.from_csv(root / 'companies.csv')
    events = list(read_jsonl(args.expanded / 'events.jsonl'))
    cutoff = max(s.prediction_at for s in calendar.sessions if s.session_date <= config['sample_end'])
    # Builder retains provenance/causality checks. Cancel its fixed decay below;
    # the learned decay is applied exactly once, inside null-sink attention.
    policy = EventPolicy(ttl_sessions=500)
    snapshot = GraphBuilder(calendar, bars, registry, events, [], policy).build(cutoff, len(calendar.sessions))
    horizons = {'earnings': 90, 'litigation': 500}
    categories = {n['id']: n['category'] for n in snapshot['nodes']['event']}
    snapshot['edges'] = [e for e in snapshot['edges'] if e['type'] != 'ACTIVE_ON'
                         or e['age_sessions'] < horizons.get(categories[e['source']], 15)]
    study = build_samples(snapshot, bars, calendar, config)
    normalized, counts = rolling_ohlcv(snapshot, bars)
    study['features'] = np.concatenate((normalized, study['features']), axis=1).astype(np.float32)
    study['scaler']['input_normalization'] = {
        'OHLC': 'Shared price mean/std over trailing 90 sessions, including observed today',
        'volume': 'Separate trailing 90-session log1p-volume mean/std',
        'warmup': 'Expanding prefix before 90 observations; no future padding',
        'technical_features': '14 original causal features, scaled using training rows only',
        'input_dimension': 19}
    vectors, manifest = load_vectors(args.expanded / 'embeddings')
    if manifest['encoder'].get('simulated') or snapshot['contains_synthetic_data']:
        raise ValueError('Real data and real embeddings required')
    data = make_tensor_data(snapshot, vectors, manifest['dimension'], study, args.device, config)
    data['edge_prior'] = data['edge_weight'] * torch.exp2(data['edge_age'].float() / policy.half_life_sessions)
    attach_history(data, snapshot, config['lookback_sessions'])
    return study, data, snapshot, counts


def fit(model, mode, seed, study, data, config, output):
    device = data['prices'].device
    windows = torch.as_tensor(study['windows'], device=device)
    targets = torch.as_tensor(study['log_targets'], dtype=torch.float32, device=device)
    scale = torch.as_tensor(study['scaler']['target_scale'], device=device, dtype=torch.float32)
    train = torch.as_tensor(np.flatnonzero(study['masks']['train']), device=device)
    validation = torch.as_tensor(np.flatnonzero(study['masks']['validation']), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, state = float('inf'), 0, None
    history, start = [], time.monotonic()
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        losses = []
        order = train[torch.randperm(len(train), generator=generator).to(device)]
        for group in order.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            objective = loss(model(data, windows[group]), targets[group], scale)
            if not torch.isfinite(objective):
                raise ValueError('Nonfinite regression loss')
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(objective.detach()))
        prediction = torch.as_tensor(predict(model, data, windows, validation, batch_size=512), device=device)
        value = float(loss(prediction, targets[validation], scale))
        if not np.isfinite(value):
            raise ValueError('Nonfinite validation loss')
        history.append({'epoch': epoch, 'training_loss': float(np.mean(losses)), 'validation_loss': value})
        if value < best - 1e-5:
            best, best_epoch = value, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps({'mode': mode, 'seed': seed, **history[-1], 'selected_epoch': best_epoch,
                          'elapsed_seconds': round(time.monotonic() - start, 1)}), flush=True)
        if epoch - best_epoch >= config['patience']:
            break
    checkpoint = output / 'checkpoints' / f'{mode}_seed{seed}.pt'
    torch.save({'state_dict': state, 'config': config, 'scaler': study['scaler'],
                'selected_epoch': best_epoch, 'mode': mode, 'seed': seed}, checkpoint)
    save(output / 'histories' / f'{mode}_seed{seed}.json', {'epochs': history, 'selected_epoch': best_epoch,
         'best_validation_loss': best, 'elapsed_seconds': time.monotonic() - start,
         'checkpoint_sha256': sha256_file(checkpoint)})


def report(output, summary):
    rows = []
    for name, result in summary['models'].items():
        metric = result['metrics']
        tolerance = result['tolerance_accuracy']['1%']
        rows.append('<tr><th>' + html.escape(name) + '</th>' +
                    ''.join(f'<td>{metric["per_feature"][key]["MAE_percent_of_today_close"]:.3f}%</td>'
                            for key in ('open', 'high', 'low', 'close')) +
                    f'<td>{metric["mean_OHLC_MAE_percent_of_today_close"]:.3f}%</td>' +
                    f'<td>{tolerance["pooled_fields_percent"]:.2f}%</td>' +
                    f'<td>{tolerance["whole_candle_percent"]:.2f}%</td></tr>')
    document = '''<!doctype html><meta charset="utf-8"><title>FinSTAK proposal pilot</title>
    <style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#173044}
    table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>
    <h1>FinSTAK: two-stage event graph pilot</h1>
    <p>Observed stock history → local GRU → typed event attention with a zero-value null sink and learned decay
    → one-hop diffusion of event shocks across same-session correlation edges → next-session OHLC.</p>
    <p>All models use the same 19 inputs, chronological splits, training budget, and valid candle decoder.
    LSTM and T-GCN are project comparators, not reproductions of published scores. No SPY and no volume target.</p>
    <p>MAE is normalized by today's observed close; lower is better. A 1% tolerance hit means absolute error
    ≤ 1% of today's close. Pooled counts each field separately; whole candle requires all four fields to pass.</p>
    <table><tr><th>Model</th><th>Open MAE</th><th>High MAE</th><th>Low MAE</th><th>Close MAE</th>
    <th>Mean MAE</th><th>1% pooled hits</th><th>1% whole candles</th></tr>'''
    document += ''.join(rows) + '</table><h2>Limits</h2><ul>'
    document += ''.join('<li>' + html.escape(x) + '</li>' for x in summary['limitations'])
    document += '</ul><details><summary>Full metrics, uncertainty and audit</summary><pre>'
    document += html.escape(json.dumps(summary, indent=2)) + '</pre></details>'
    (output / 'results.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 41])
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive epoch budget and unique seeds required')
    # Never silently reuse checkpoints from changed source/configuration.
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('checkpoints', 'histories'):
        (args.output / name).mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config = {'sample_start': '2016-01-01', 'sample_end': '2019-12-31', 'train_end': '2017-12-31',
              'validation_end': '2018-12-31', 'lookback_sessions': 30, 'horizon_sessions': 1,
              'max_epochs': args.epochs, 'patience': 5, 'seeds': args.seeds, 'batch_size': 1024,
              'targets': ['open', 'high', 'low', 'close'], 'modes': args.modes}
    study, data, snapshot, counts = prepare(args, config)
    sources = [args.source / ('dataset/' + n) for n in ('prices.csv', 'calendar.csv', 'companies.csv')]
    sources += [args.expanded / 'events.jsonl', args.expanded / 'embeddings/manifest.json',
                args.expanded / 'embeddings/vectors.f32', args.expanded / 'embeddings/index.jsonl']
    plan = {'config': config, 'source_hashes': {str(p): sha256_file(p) for p in sources},
            'implementation_hashes': {str(p): sha256_file(p) for p in Path(__file__).parents[1].rglob('*.py')},
            'split_counts': {k: int(v.sum()) for k, v in study['masks'].items()}, 'scaler': study['scaler'],
            'stock_days': len(data['prices']), 'events': len(data['events']), 'event_edges': len(data['edge_day']),
            'correlation_edges': len(data['corr_target']), 'normalization_warmup_rows': int((counts < 90).sum()),
            'horizons_sessions': {'earnings': 90, 'litigation': 500, 'other': 15},
            'decay': 'Positive learned rate per event category/directness/head, initialized to 5-session half-life',
            'cutoff': 'Existing end-of-source-day cutoff; intraday availability not established',
            'peer_edges': 'Prior-60-session correlation, top 3 positive neighbors >= .2; no invented supply chain',
            'loss': 'MSE in log(next OHLC / observed close), scaled by train-only target variance',
            'selection': 'Validation-only checkpoint selection, every model selected before test scoring'}
    save(args.output / 'plan.json', plan)
    save(args.output / 'scaler.json', study['scaler'])
    for mode in args.modes:
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = FinSTAK(study['scaler'], companies=len(data['company_ids']), mode=mode).to(args.device)
            fit(model, mode, seed, study, data, config, args.output)
            del model
    save(args.output / 'validation_selection_complete.json', {'complete': True, 'test_used_for_selection': False})
    test_mask = study['masks']['test']
    windows = torch.as_tensor(study['windows'], device=args.device)
    indices = torch.as_tensor(np.flatnonzero(test_mask), device=args.device)
    truth, anchor = study['targets'][test_mask], study['anchors'][test_mask]
    dates = np.array([r['session_date'] for r in study['metadata']])[test_mask]
    companies = np.array([r['company_id'] for r in study['metadata']])[test_mask]
    predictions, result, audit = {}, {}, {}
    for mode in args.modes:
        individual = []
        for seed in args.seeds:
            model = FinSTAK(study['scaler'], companies=len(data['company_ids']), mode=mode).to(args.device)
            model.load_state_dict(torch.load(args.output / f'checkpoints/{mode}_seed{seed}.pt',
                                           weights_only=True, map_location=args.device)['state_dict'])
            model.eval()
            raw = predict(model, data, windows, indices)
            prices = price_predictions(raw, anchor)
            individual.append(prices)
            predictions[f'{mode}_seed{seed}'] = prices
            if seed == args.seeds[0]:
                # Verify future stock inputs/event embeddings cannot alter earlier
                # forecasts, and reconstruction matches the selected checkpoint.
                cutoff = '2018-12-31'
                past = torch.as_tensor(np.flatnonzero(study['masks']['validation']), device=args.device)
                changed = dict(data)
                changed['prices'] = data['prices'].clone()
                future_days = torch.tensor([n['session_date'] > cutoff for n in snapshot['nodes']['stock_day']], device=args.device)
                changed['prices'][future_days] += 10
                changed['events'] = data['events'].clone()
                future_events = torch.tensor([n['known_at'][:10] > cutoff for n in snapshot['nodes']['event']], device=args.device)
                changed['events'][future_events] += 10
                before = predict(model, data, windows, past)
                after = predict(model, changed, windows, past)
                delta = float(np.abs(before - after).max())
                if delta > 1e-7:
                    raise ValueError('Future perturbation changed past forecast')
                audit[mode] = {'future_perturbation_max_abs_logratio_change': delta}
            del model
        predictions[mode] = np.mean(individual, axis=0)
        result[mode] = {'metrics': metrics(truth, predictions[mode], anchor, companies),
                        'seed_metrics': [metrics(truth, p, anchor, companies) for p in individual],
                        'tolerance_accuracy': tolerance_accuracy(truth, predictions[mode], anchor)}
    predictions['last_close'] = np.repeat(anchor[:, None], 4, axis=1)
    predictions['last_OHLC'] = study['previous_ohlc'][test_mask]
    for mode in ('last_close', 'last_OHLC'):
        result[mode] = {'metrics': metrics(truth, predictions[mode], anchor, companies),
                        'tolerance_accuracy': tolerance_accuracy(truth, predictions[mode], anchor)}
    comparisons = {}
    if 'events' in predictions:
        candidate = np.abs((predictions['events'] - truth) / anchor[:, None]).mean(1)
        for mode in ('no_events', 'no_decay', 'lstm', 'tgcn', 'last_close'):
            if mode in predictions:
                baseline = np.abs((predictions[mode] - truth) / anchor[:, None]).mean(1)
                comparisons[mode] = block_interval([(baseline[dates == day] - candidate[dates == day]).mean()
                                                    for day in sorted(set(dates))])
    np.savez_compressed(args.output / 'test_predictions.npz', targets=truth, anchors=anchor, dates=dates,
                        companies=companies, **predictions)
    summary = {'models': result, 'split_counts': plan['split_counts'], 'causality_audit': audit,
               'paired_MAE_reduction_events_vs_comparator': comparisons,
               'comparison_units': 'Fractions of observed close; multiply by 100 for percentage points',
               'limitations': ['Exploratory reused 2019 test, 16 stocks; not a five-year S&P 500 study.',
                              'Existing cached Qwen extraction and EmbeddingGemma vectors; FinBERT/FinDKG not added.',
                              'News arrival times unverified; EOD assumptions retained, not a 09:00 ET study.',
                              'Retrospectively adjusted OHLC. Filing/release observations may overlap one economic event.',
                              'No historical supply-chain, analyst-consensus or court-docket data invented.',
                              'Event expiry stops new messages; local histories contain realized effects of past events.',
                              'Tolerance hit rates depend on chosen bands; not 100 minus MAE.',
                              'Pilot budgets and model adaptations do not establish paper-worthy improvement.']}
    save(args.output / 'summary.json', summary)
    report(args.output, summary)
    print(json.dumps({k: v['metrics']['mean_OHLC_MAE_percent_of_today_close'] for k, v in result.items()}), flush=True)


if __name__ == '__main__':
    main()
