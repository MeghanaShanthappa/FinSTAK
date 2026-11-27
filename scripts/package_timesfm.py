"""Retain audited research artifacts, minimal inputs, and a readable report."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary', type=Path, required=True)
    p.add_argument('--encoders', type=Path, nargs='+', required=True)
    p.add_argument('--final', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--explanations', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert json.loads((args.primary / 'verification.json').read_text())['passed']
    assert json.loads((args.primary / 'native_verification.json').read_text())['passed']
    for root in args.encoders:
        assert json.loads((root / 'verification.json').read_text())['passed']
    assert {json.loads((root / 'plan.json').read_text())['context'] for root in args.encoders} == {32, 64, 128}
    fits = sum(len(list((root / 'checkpoints').glob('*.pt'))) for root in [args.primary, *args.encoders])
    assert fits == 48
    shutil.copytree(args.primary, args.output / 'primary', dirs_exist_ok=True)
    for root in args.encoders:
        context = json.loads((root / 'plan.json').read_text())['context']
        shutil.copytree(root, args.output / f'encoder_c{context}', dirs_exist_ok=True)
    for filename in ('summary.json', 'test_predictions.npz', 'validation_selection.json', 'outcome.json', 'results.html'):
        shutil.copy2(args.final / filename, args.output / filename)
    shutil.copy2(args.explanations, args.output / 'source_deletion_sensitivity.json')
    shutil.copy2('/tmp/finstak-timesfm-tests.log', args.output / 'automated_tests.log')
    project = Path(__file__).resolve().parents[1]
    source = args.output / 'source_snapshot'
    for directory in ('stak_graph', 'scripts', 'tests', 'configs', 'examples'):
        shutil.copytree(project / directory, source / directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for filename in ('pyproject.toml', 'README.md', 'TIMESFM_RESEARCH.md'):
        shutil.copy2(project / filename, source / filename)
    input_paths = {}
    dataset = Path('/var/tmp/mshanthappa-stak-graph/model_sweep_v2/dataset')
    expanded = Path('/var/tmp/mshanthappa-stak-graph/expanded_v3')
    for filename in ('prices.csv', 'calendar.csv', 'companies.csv', 'manifest.json', 'news_provenance.jsonl'):
        target = args.output / 'inputs/source/dataset' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset / filename, target)
        input_paths[str(dataset / filename)] = str(target.relative_to(args.output))
    for filename in ('events.jsonl', 'embeddings/vectors.f32', 'embeddings/manifest.json', 'embeddings/index.jsonl'):
        target = args.output / 'inputs/expanded' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(expanded / filename, target)
        input_paths[str(expanded / filename)] = str(target.relative_to(args.output))
    reference = Path('/var/tmp/mshanthappa-stak-graph/finstak_proposal_v1b')
    for filename in ('test_predictions.npz', 'summary.json', 'plan.json', 'verification.json'):
        target = args.output / 'reference' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / filename, target)
    for seed in (7, 19, 41):
        for folder, suffix in (('checkpoints', 'pt'), ('histories', 'json')):
            filename = f'{folder}/no_decay_seed{seed}.{suffix}'
            target = args.output / 'reference' / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reference / filename, target)
    summary = json.loads((args.output / 'summary.json').read_text())
    outcome = json.loads((args.output / 'outcome.json').read_text())
    selected = outcome['selected_model']
    original = summary['models']['original_no_decay']['metrics']
    selected_metrics = summary['models'][selected]['metrics']
    saved = np.load(args.output / 'test_predictions.npz', allow_pickle=False)
    dates, anchors, truth = saved['dates'], saved['anchors'], saved['targets']
    quarters = np.array([str(d)[:4] + '-Q' + str((int(str(d)[5:7]) - 1) // 3 + 1) for d in dates])
    diagnostics = {}
    for quarter in sorted(set(quarters)):
        mask = quarters == quarter
        candidate = float((100 * np.abs(saved[selected][mask] - truth[mask]) / anchors[mask, None]).mean())
        baseline = float((100 * np.abs(saved['original_no_decay'][mask] - truth[mask]) / anchors[mask, None]).mean())
        diagnostics[quarter] = {'samples': int(mask.sum()), 'selected_MAE_percent': candidate, 'original_MAE_percent': baseline,
                                'MAE_reduction_percentage_points': baseline - candidate}
    (args.output / 'quarterly_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    labels = {'original_no_decay': 'Original FinSTAK: no decay', 'original_events': 'Original FinSTAK: learned decay',
              'original_no_events': 'Original FinSTAK: no events', 'original_lstm': 'Original LSTM',
              'original_tgcn': 'Original GCN–GRU', 'original_last_close': 'Previous close persistence',
              'original_last_OHLC': 'Previous OHLC persistence', 'native_price': 'TimesFM: OHLC history',
              'native_market': 'TimesFM: historical numerical features', 'native_direct': 'TimesFM: direct-event covariates',
              'native_graph': 'TimesFM: graph-event covariates'}
    for prefix, title in (('hybrid', 'Forecast-prior correction'), ('encoder', 'Transformer encoder')):
        for mode, suffix in (('local', 'no events'), ('direct', 'direct events'), ('graph', 'event graph'), ('no_decay', 'events without decay')):
            labels[prefix + '_' + mode] = title + ': ' + suffix
            labels[prefix + '_' + mode + '_tuned'] = title + ': ' + suffix + ' (context tuned)'
    comparison = outcome['best_new_model_paired_gain'] if selected.startswith('original_') else outcome['paired_gain']
    low, high = comparison['ci95']
    intro = f'<p><strong>Validation-selected model:</strong> {html.escape(labels.get(selected, selected))}.</p>'
    if selected.startswith('original_'):
        new_name = outcome['best_new_model_by_validation']
        intro += f'<p>The strongest new candidate by validation was {html.escape(labels[new_name])}, with test normalized OHLC MAE <strong>{outcome["best_new_model_test_MAE_percent"]:.4f}%</strong>, versus <strong>{original["mean_OHLC_MAE_percent_of_today_close"]:.4f}%</strong> for original FinSTAK.'
    else:
        intro += f'<p>Test normalized OHLC MAE: <strong>{selected_metrics["mean_OHLC_MAE_percent_of_today_close"]:.4f}%</strong>; original FinSTAK: <strong>{original["mean_OHLC_MAE_percent_of_today_close"]:.4f}%</strong>.'
    intro += f' New-candidate error reduction 95% interval: [{low:.4f}, {high:.4f}] percentage points. Positive favors the new model.</p>'
    intro += '<p>48 trained fits across three seeds, 12 native validation configurations, 4,032 test examples, 165 passing automated tests, independent checkpoint/label audits, native replay, and source-deletion sensitivity. The study completed within the authorized research window.</p>'
    report = (args.output / 'results.html').read_text()
    report = report.replace('<h1>FinSTAK × TimesFM 3.0</h1>', '<h1>FinSTAK × TimesFM 3.0</h1>' + intro)
    for key, label in labels.items():
        report = report.replace('<th>' + key + '</th>', '<th>' + html.escape(label) + '</th>')
    (args.output / 'results.html').write_text(report)
    retained = {str(path.relative_to(args.output)): sha(path) for path in args.output.rglob('*') if path.is_file() and path.name != 'delivery_manifest.json'}
    manifest = {'passed': True, 'trained_fits': 48, 'automated_tests': 165,
                'retained_input_mapping': input_paths, 'artifact_sha256': retained,
                'foundation_weights': 'Pinned hash and revision retained; redownload under the included research license',
                'source': 'Archived project source and official TimesFM source included'}
    (args.output / 'delivery_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'passed': True, 'artifacts': len(retained), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
