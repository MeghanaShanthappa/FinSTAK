"""GPU TimesFM 3.0 benchmark and frozen-prior FinSTAK adaptation.

Run into a fresh directory. Interrupted jobs resume only with identical code,
data, checkpoint revision and configuration. The reused test is exploratory.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from .finstak_run import prepare, fit
from .timesfm_features import context_rows, event_covariates, project_candle
from .model_sweep import save
from .data import sha256_file
from .next_day_data import metrics
from .finstak_data import tolerance_accuracy
from .train import predict
from .metrics import block_interval
from stak_graph.graph.market import TradingCalendar, read_bars
from stak_graph.models.timesfm_hybrid import TimesFMFinSTAK

NATIVE = ('price', 'market', 'direct', 'graph')
HYBRIDS = ('local', 'direct', 'graph', 'no_decay')


def native_forecast(model, study, days, prices, covariates, indices, length, batch, path):
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if not np.array_equal(cached['indices'], indices):
            raise ValueError('Cached forecast row mismatch')
        return cached['prediction']
    rows = context_rows(days, study['windows'][indices, -1], length, allow_short=True)
    values = []
    start = time.monotonic()
    for offset in range(0, len(indices), batch):
        group = indices[offset:offset + batch]
        history = rows[offset:offset + batch]
        # One scalar (observed own-stock close) is shared by all four targets;
        # this preserves cross-price geometry and introduces no future values.
        contexts = [np.log(prices[row] / study['anchors'][index]).T.astype('float32') for row, index in zip(history, group)]
        past = None if covariates is None else [covariates[row].T for row in history]
        with torch.inference_mode():
            output = list(model.predict_batch(contexts, horizon=1,
                          past_only_covariates=past, return_quantiles=False,
                          use_symmetric_averaging=False, make_positive=False))
        result = np.stack([np.asarray(v.forecast)[:, 0] for v in output])
        if result.shape != (len(group), 4) or not np.isfinite(result).all():
            raise ValueError('Invalid TimesFM output')
        values.append(result)
        if offset % (batch * 20) == 0:
            print(json.dumps({'phase': path.stem, 'completed': offset + len(group),
                              'total': len(indices), 'seconds': time.monotonic() - start}), flush=True)
    raw = np.concatenate(values)
    prediction = project_candle(raw)
    np.savez_compressed(path, indices=indices, raw=raw, prediction=prediction,
                        context_counts=np.array([len(row) for row in rows]),
                        elapsed_seconds=time.monotonic() - start,
                        repaired_candles=int(np.any(raw != prediction, axis=1).sum()))
    return prediction


def score(study, mask, prediction):
    companies = np.array([r['company_id'] for r in study['metadata']])[mask]
    truth, anchor = study['targets'][mask], study['anchors'][mask]
    return {'metrics': metrics(truth, prediction, anchor, companies),
            'tolerance_accuracy': tolerance_accuracy(truth, prediction, anchor)}


def render(output, summary):
    document = '''<!doctype html><meta charset="utf-8"><title>FinSTAK × TimesFM</title>
    <style>body{font:16px/1.6 system-ui;max-width:1200px;margin:36px auto;padding:24px;color:#173044}
    table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}
    pre{white-space:pre-wrap}small{color:#456}</style><h1>FinSTAK × TimesFM 3.0</h1>
    <p>Next-session OHLC, 16 stocks, chronological train/validation/test. All test results are exploratory:
    2019 has been inspected before and foundation-model pretraining overlap is unknown.</p>
    <p>MAE is absolute price error divided by the observed prediction-day close, expressed as a percent.
    Tolerance hits count individual OHLC fields within 1% of that close.</p>
    <table><tr><th>Model</th><th>Mean OHLC MAE ↓</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>1% hits ↑</th></tr>'''
    for name, value in summary['models'].items():
        metric = value['metrics']
        fields = [metric['mean_OHLC_MAE_percent_of_today_close']] + [metric['per_feature'][f]['MAE_percent_of_today_close'] for f in ('open', 'high', 'low', 'close')]
        document += '<tr><th>' + html.escape(name) + '</th>' + ''.join(f'<td>{v:.4f}%</td>' for v in fields)
        document += f'<td>{value["tolerance_accuracy"]["1%"]["pooled_fields_percent"]:.2f}%</td></tr>'
    document += '</table><h2>Limits and interpretation</h2><ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in summary['limitations'])
    document += '</ul><details><summary>Selection, uncertainty and audit</summary><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre></details>'
    (output / 'results.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--contexts', nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 19, 41])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=('native', 'all'), default='all')
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.device != 'cuda':
        raise RuntimeError('This GPU experiment requires CUDA; no silent CPU fallback')
    if min(args.contexts) < 32 or args.epochs < 1 or args.batch < 1:
        parser.error('Contexts >=32 and positive budgets required')
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('native', 'checkpoints', 'histories'):
        (args.output / name).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    config = {'sample_start': '2016-01-01', 'sample_end': '2019-12-31', 'train_end': '2017-12-31',
              'validation_end': '2018-12-31', 'lookback_sessions': 30, 'horizon_sessions': 1,
              'max_epochs': args.epochs, 'patience': 5, 'batch_size': 1024, 'seeds': args.seeds,
              'targets': ['open', 'high', 'low', 'close']}
    code_hashes = {str(p.relative_to(Path(__file__).parents[2])): sha256_file(p)
                   for p in Path(__file__).parents[1].rglob('*.py')}
    data_paths = [args.source / 'dataset' / n for n in ('prices.csv', 'companies.csv', 'calendar.csv')]
    data_paths += [args.expanded / 'events.jsonl', args.expanded / 'embeddings/vectors.f32',
                   args.expanded / 'embeddings/manifest.json', args.expanded / 'embeddings/index.jsonl']
    from huggingface_hub import HfApi
    revision = HfApi().model_info('google/timesfm-3.0-pytorch').sha
    plan = {'config': config, 'contexts': args.contexts, 'native_modes': NATIVE, 'hybrid_modes': HYBRIDS,
            'code_hashes': code_hashes, 'data_hashes': {str(p): sha256_file(p) for p in data_paths},
            'upstream_revision': subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
            'checkpoint_revision': revision, 'batch': args.batch,
            'native_selection': 'Per-mode context chosen by validation normalized MAE; no test selection',
            'hybrid_selection': 'Validation-only checkpoints; three-seed equal ensemble; no test selection',
            'primary_comparison': 'hybrid_graph vs hybrid_local; native_graph vs native_direct secondary',
            'pretraining_overlap': 'Unknown; 2019 test cannot establish zero-shot temporal isolation',
            'hardware': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    plan_path = args.output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError('Resume plan differs from frozen plan; use a fresh output directory')
    save(plan_path, plan)
    study, data, snapshot, _ = prepare(args, config)
    days = snapshot['nodes']['stock_day']
    calendar = TradingCalendar.from_csv(args.source / 'dataset/calendar.csv')
    bars = read_bars(args.source / 'dataset/prices.csv', calendar)
    lookup = {(b.company_id, b.session_date): b for b in bars}
    prices = np.array([[getattr(lookup[d['company_id'], d['session_date']], f) for f in ('open', 'high', 'low', 'close')] for d in days])
    direct, peer, feature_audit = event_covariates(data, days, config['train_end'])
    save(args.output / 'event_feature_transform.json', feature_audit)
    market = study['features'][:, 4:]  # normalized historical volume + 14 causal indicators
    covariates = {'price': None, 'market': market, 'direct': np.concatenate((market, direct), 1),
                  'graph': np.concatenate((market, direct, peer), 1)}
    validation = np.flatnonzero(study['masks']['validation'])
    test = np.flatnonzero(study['masks']['test'])
    from timesfm3 import TimesFM3Forecaster, ModelConfig
    foundation = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
                  revision=revision, cache_dir=str(args.cache), device='cuda', per_core_batch_size=args.batch))
    native_validation, selected = {}, {}
    for mode in NATIVE:
        native_validation[mode] = {}
        for context in args.contexts:
            forecast = native_forecast(foundation, study, days, prices, covariates[mode], validation,
                                      context, args.batch, args.output / 'native' / f'{mode}_c{context}_validation.npz')
            predicted = study['anchors'][validation, None] * np.exp(forecast)
            native_validation[mode][str(context)] = score(study, study['masks']['validation'], predicted)
        selected[mode] = min(args.contexts, key=lambda c: native_validation[mode][str(c)]['metrics']['mean_OHLC_MAE_percent_of_today_close'])
    save(args.output / 'native_validation_selection.json', {'selected_context': selected, 'validation': native_validation})
    # Generate training priors only for the validation-selected price model.
    all_indices = np.arange(len(study['windows']))
    base = native_forecast(foundation, study, days, prices, None, all_indices, selected['price'],
                           args.batch, args.output / 'native' / 'selected_price_all.npz')
    native_test = {'native_price': study['anchors'][test, None] * np.exp(base[test])}
    for mode in NATIVE[1:]:
        forecast = native_forecast(foundation, study, days, prices, covariates[mode], test, selected[mode],
                                  args.batch, args.output / 'native' / f'{mode}_selected_test.npz')
        native_test['native_' + mode] = study['anchors'][test, None] * np.exp(forecast)
    del foundation
    torch.cuda.empty_cache()
    predictions = native_test
    if args.stage == 'all':
        prior = torch.zeros((len(days), 4), device=args.device)
        sample_rows = torch.as_tensor(study['windows'][:, -1], device=args.device)
        prior[sample_rows] = torch.as_tensor(base, dtype=torch.float32, device=args.device)
        data['forecast_prior'] = prior
        data['prices'] = torch.cat((data['prices'], prior / torch.tensor(study['scaler']['target_scale'], device=args.device)), 1)
        windows = torch.as_tensor(study['windows'], device=args.device)
        test_indices = torch.as_tensor(test, device=args.device)
        # All hybrid checkpoints are selected before test inference.
        for mode in HYBRIDS:
            for seed in args.seeds:
                name = 'hybrid_' + mode
                checkpoint = args.output / 'checkpoints' / f'{name}_seed{seed}.pt'
                if not (args.output / 'histories' / f'{name}_seed{seed}.json').exists():
                    torch.manual_seed(seed)
                    model = TimesFMFinSTAK(study['scaler'], len(data['company_ids']), mode).to(args.device)
                    fit(model, name, seed, study, data, config, args.output)
                    del model
        save(args.output / 'hybrid_selection_complete.json', {'complete': True, 'test_used_for_selection': False})
        for mode in HYBRIDS:
            individual = []
            for seed in args.seeds:
                model = TimesFMFinSTAK(study['scaler'], len(data['company_ids']), mode).to(args.device)
                model.load_state_dict(torch.load(args.output / 'checkpoints' / f'hybrid_{mode}_seed{seed}.pt', weights_only=True, map_location=args.device)['state_dict'])
                model.eval()
                # Test is scored only after development selection completes.
                raw = predict(model, data, windows, test_indices)
                value = study['anchors'][test, None] * np.exp(raw.astype('float64'))
                individual.append(value)
                predictions[f'hybrid_{mode}_seed{seed}'] = value
                del model
            predictions['hybrid_' + mode] = np.mean(individual, 0)
    truth, anchor = study['targets'][test], study['anchors'][test]
    dates = np.array([r['session_date'] for r in study['metadata']])[test]
    companies = np.array([r['company_id'] for r in study['metadata']])[test]
    original = np.load(args.reference / 'test_predictions.npz', allow_pickle=False)
    for key, expected in (('targets', truth), ('anchors', anchor), ('dates', dates), ('companies', companies)):
        if not np.array_equal(original[key], expected):
            raise ValueError('Original FinSTAK comparison sample identity mismatch: ' + key)
    for mode in ('events', 'no_events', 'no_decay', 'lstm', 'tgcn', 'last_close', 'last_OHLC'):
        predictions['original_' + mode] = original[mode]
    result = {name: score(study, study['masks']['test'], value) for name, value in predictions.items() if '_seed' not in name}
    comparisons = {}
    pairs = [('native_graph', 'native_direct'), ('native_price', 'original_no_decay'),
             ('hybrid_graph', 'hybrid_local'), ('hybrid_graph', 'hybrid_direct'),
             ('hybrid_graph', 'original_no_decay'), ('hybrid_graph', 'hybrid_no_decay')]
    for candidate, baseline in pairs:
        if candidate not in predictions:
            continue
        gain = (np.abs(predictions[baseline] - truth) - np.abs(predictions[candidate] - truth)).mean(1) / anchor * 100
        comparisons[candidate + '_vs_' + baseline] = block_interval([gain[dates == day].mean() for day in sorted(set(dates))])
    summary = {'models': result, 'paired_MAE_reduction_percentage_points': comparisons,
               'selected_context': selected, 'split_counts': {k: int(v.sum()) for k, v in study['masks'].items()},
               'limitations': ['Exploratory reused 2019 test; 16 stocks; no prospective confirmation.',
               'Foundation pretraining overlap unknown; cannot claim a contamination-free zero-shot test.',
               'Historical publication times are end-of-source-day assumptions, not verified intraday arrival.',
               'Native TimesFM uses OHLC history plus optional past-only covariates; no future news is supplied.',
               'Native contexts are selected from ' + str(args.contexts) + ' sessions; original GRU uses 30 sessions.',
               'Hybrid uses frozen forecast priors, not TimesFM hidden-state embeddings or end-to-end foundation fine-tuning.',
               'Event PCA/scales fit training data only; confidence is not calibrated.',
               'OHLC candle projection is applied uniformly to native forecasts; repaired counts saved.',
               'Hybrid candle bounds use detached open/close for high/low gradients; correct geometry is not a predictive-value claim.',
               'All intervals are exploratory, not adjusted for multiple comparisons. Positive MAE reduction favors candidate.',
               'TimesFM 3.0 downloaded weights are non-commercial/non-production research assets.']}
    np.savez_compressed(args.output / 'test_predictions.npz', targets=truth, anchors=anchor, dates=dates, companies=companies, **predictions)
    save(args.output / 'summary.json', summary)
    render(args.output, summary)
    print(json.dumps({k: v['metrics']['mean_OHLC_MAE_percent_of_today_close'] for k, v in result.items()}), flush=True)


if __name__ == '__main__':
    main()
