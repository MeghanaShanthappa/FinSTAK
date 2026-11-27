"""Build a readable, evidence-linked report from completed walk-forward fits."""
from collections import Counter
import html
import json
from pathlib import Path
import numpy as np

from stak_graph.experiments.scale_io import write_json
from stak_graph.experiments.data import sha256_file
from stak_graph.experiments.next_day_data import metrics
from stak_graph.experiments.finstak_data import tolerance_accuracy

ROOT = Path('/tmp/finstak-100x10-v1')


def load_predictions(stage, variant):
    result = []
    for year in ('2021', '2022', '2023'):
        path = ROOT / stage / year / f'{variant}_predictions.npz'
        if not path.exists():
            return None
        with np.load(path) as data:
            result.append({key: data[key] for key in ('dates', 'companies', 'truth', 'anchor', 'prediction')})
    return result


def paired_interval(reference, candidate, seed=1701):
    differences = []
    for a, b in zip(reference, candidate):
        if not all(np.array_equal(a[k], b[k]) for k in ('dates', 'companies', 'truth', 'anchor')):
            raise ValueError('Comparison must have identical labels, dates and stocks')
        loss_a = (np.abs(a['prediction'] - a['truth']) / a['anchor'][:, None]).mean(1) * 100
        loss_b = (np.abs(b['prediction'] - b['truth']) / b['anchor'][:, None]).mean(1) * 100
        dates, inv = np.unique(a['dates'], return_inverse=True)
        mean = np.bincount(inv, weights=loss_a - loss_b) / np.bincount(inv)
        differences.append(mean)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(2000):
        selected = []
        for delta in differences:
            starts = rng.integers(0, len(delta), size=int(np.ceil(len(delta) / 20)))
            indices = np.concatenate([(start + np.arange(20)) % len(delta) for start in starts])[:len(delta)]
            selected.append(delta[indices])
        boot.append(np.concatenate(selected).mean())
    return {'MAE_reduction_percentage_points': float(np.concatenate(differences).mean()),
            '95_percent_interval': np.quantile(boot, [.025, .975]).tolist(),
            'positive_favors_candidate': True, 'block_sessions': 20, 'replicates': 2000,
            'uncorrected_exploratory_interval': True}


def main():
    data = json.loads((ROOT / 'dataset/event_audit.json').read_text())
    merged = json.loads((ROOT / 'expanded/manifest.json').read_text())
    results = {}
    for stage in ('data_stage', 'architecture_stage'):
        for year in ('2021', '2022', '2023'):
            path = ROOT / stage / year / 'summary.json'
            if not path.exists():
                raise ValueError(f'Results are incomplete: {path}')
            for variant, measurement in json.loads(path.read_text()).items():
                if variant in ('previous_close', 'previous_OHLC') and stage == 'architecture_stage':
                    continue
                results.setdefault(variant, {})[year] = measurement
    comparisons = {}
    ref = load_predictions('data_stage', 'no_decay')
    native_path = ROOT / 'native_timesfm/predictions.npz'
    native_result = None
    if native_path.exists():
        panel = np.load(ROOT / 'packed/panel.npz')
        native = np.load(native_path)
        ids, company = native['date_ids'], panel['companies']
        native_price = native['prediction'].reshape(len(ids), len(company), 4)
        native_result, native_pairs = {}, []
        for year in ('2021', '2022', '2023'):
            keep = np.char.startswith(panel['dates'][ids], year) & np.char.startswith(panel['dates'][ids + 1], year)
            dates = ids[keep]
            p = native_price[keep].reshape(-1, 4)
            truth = panel['raw'][dates + 1, :, :4].reshape(-1, 4)
            anchor = panel['raw'][dates, :, 3].reshape(-1)
            companies = np.tile(company, len(dates))
            native_result[year] = {'price': metrics(truth, p, anchor, companies), 'tolerance': tolerance_accuracy(truth, p, anchor)}
            native_pairs.append({'dates': np.repeat(panel['dates'][dates], len(company)), 'companies': companies,
                                 'truth': truth, 'anchor': anchor, 'prediction': p})
        results['native_TimesFM_price_only'] = native_result
        comparisons['native_TimesFM_price_only_vs_no_decay'] = paired_interval(ref, native_pairs)
        write_json(ROOT / 'native_timesfm/aligned_summary.json', native_result)
    for stage in ('data_stage', 'architecture_stage'):
        for variant in json.loads((ROOT / stage / 'plan.json').read_text())['variants']:
            candidate = load_predictions(stage, variant)
            comparisons[variant + '_vs_no_decay'] = paired_interval(ref, candidate)
    for ref_name, name in [('events', 'hybrid_decay'), ('events', 'sector'), ('events', 'isolated'),
                           ('events', 'primary_only'), ('events', 'two_layer'), ('events', 'finbert'),
                           ('volume_shared', 'volume_pcgrad'), ('volume_shared', 'volume_uncertainty'),
                           ('volume_shared', 'dual_stream'), ('volume_shared', 'combined')]:
        ref_stage = 'data_stage' if ref_name == 'events' else 'architecture_stage'
        comparisons[f'{name}_vs_{ref_name}'] = paired_interval(load_predictions(ref_stage, ref_name),
                                                              load_predictions('architecture_stage', name))
    limitations = [
        'Automated source evidence checks are not human semantic review or independent verification of news truth.',
        'Near-duplicate rules cannot guarantee one observation per underlying economic event.',
        'News availability uses a conservative assumed two-day delay; a verified pre-market feed is not available.',
        'Fixed manually curated universe, retrospective corporate-action adjustments, static sector groups and unknown pretraining overlap limit historical realism.',
        'No historical supply-chain, analyst-consensus surprise or court-docket change feed was fabricated.',
        'Typed attention and sector aggregation are project implementations, not a full published HGT reproduction.',
        'Three-seed ensembles and 20-session date-block intervals are exploratory; multiple comparisons are not corrected.',
        'Forecast accuracy and volume retransformation metrics do not establish trading profitability.',
    ]
    summary = {'dataset': {'stocks': 100, 'years': '2014–2023, final source price 2023-12-28',
                'events': merged['events'], 'verification': merged['verification'], 'news_by_year': data['by_year']},
               'models': results, 'comparisons': comparisons, 'limitations': limitations}
    if native_result is not None:
        summary['native_TimesFM_price_only'] = native_result
    write_json(ROOT / 'summary.json', summary)
    rows = []
    for name, years in results.items():
        cells = [f'<th>{html.escape(name)}</th>']
        for year in ('2021', '2022', '2023'):
            value = years[year]
            cells.append(f'<td>{value["price"]["mean_OHLC_MAE_percent_of_today_close"]:.4f}%</td>')
        pooled = np.mean([years[y]['tolerance']['1%']['pooled_fields_percent'] for y in years])
        cells.append(f'<td>{pooled:.2f}%</td>')
        direction = 100 * np.mean([years[y]['price']['next_close_direction_accuracy'] for y in years])
        cells.append(f'<td>{direction:.2f}%</td>')
        comparison = comparisons.get(name + '_vs_no_decay')
        cells.append('<td>' + (f'{comparison["MAE_reduction_percentage_points"]:+.4f}' if comparison else '—') + '</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    volume_rows = []
    for baseline in ('previous_volume', 'trailing_log_volume_median'):
        values = [results['previous_close'][year]['volume_baselines'][baseline] for year in ('2021', '2022', '2023')]
        mape = np.mean([value['MAPE_positive_volume_percent'] for value in values])
        mse = np.mean([value['MSE_shares_squared'] for value in values])
        volume_rows.append(f'<tr><th>{baseline}</th><td>baseline</td><td>{mape:.2f}%</td><td>{mse:.4g}</td></tr>')
    for name, years in results.items():
        if 'volume' not in years['2021']:
            continue
        for correction in ('naive', 'gaussian', 'smearing'):
            mape = [years[y]['volume'][correction]['MAPE_positive_volume_percent'] for y in years]
            mse = [years[y]['volume'][correction]['MSE_shares_squared'] for y in years]
            volume_rows.append(f'<tr><th>{html.escape(name)}</th><td>{correction}</td><td>{np.mean(mape):.2f}%</td><td>{np.mean(mse):.4g}</td></tr>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>FinSTAK: 100-stock walk-forward study</title><style>
    body{font:16px/1.6 system-ui;color:#173044;max-width:1200px;margin:40px auto;padding:20px}
    table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}
    pre{white-space:pre-wrap;font-size:12px}aside{background:#eef4f7;padding:20px;border-radius:10px}
    </style><h1>FinSTAK: expanded data and architecture experiments</h1>'''
    document += f'<p>100 companies · 2014–2023 · {merged["events"]:,} source-evidence-checked event records · 3 chronological test years · 3 paired seeds.</p>'
    document += '<aside>Event verification is automated and source-based. The collection has not received human semantic certification. Historical arrival timestamps and perfect economic-event deduplication are not established.</aside>'
    document += '<h2>What was tested</h2><p>Stage 1 expands the data while retaining FinSTAK’s design, with price-only, no-decay, learned-decay, LSTM and T-GCN conditions. Stage 2 tests detached wick heads, volume tasks, PCGrad, uncertainty weighting, hybrid decay, sector routing, primary-subject filtering, two-layer GRUs, auxiliary direction classification, frozen FinBERT, dual streams and a combined recipe.</p>'
    document += '<p>Every fold refits training-only scalers. Train/validation ends are 2019/2020 for the 2021 test, 2020/2021 for 2022 and 2021/2022 for 2023. Boundary-crossing labels are purged. Checkpoints use validation OHLC error; test scores do not select checkpoints.</p>'
    document += '<h2>Price forecasting</h2><p>Normalized OHLC MAE is average absolute price error divided by observed decision-day close; lower is better. A tolerance hit means a field is within 1% of that close. It is not directional accuracy. Positive error reduction favors the candidate over no-decay FinSTAK.</p>'
    document += '<table><tr><th>Model</th><th>2021 MAE</th><th>2022 MAE</th><th>2023 MAE</th><th>Mean 1% field hits</th><th>Mean close direction</th><th>MAE reduction, pp</th></tr>' + ''.join(rows) + '</table>'
    majority = 100 * np.mean([results['previous_close'][year]['direction_baselines']['training_majority_accuracy'] for year in ('2021', '2022', '2023')])
    document += f'<p>The training-majority direction baseline averages {majority:.2f}% across the three test years. Directional accuracy must be assessed against this benchmark, rather than assuming that any score above 50% adds predictive value.</p>'
    document += '<h2>Volume forecasts</h2><p>Corrections use only the preceding validation year’s residuals, independently per stock. Means below average the three years. Neither Gaussian correction nor empirical smearing guarantees conditional mean accuracy.</p>'
    document += '<table><tr><th>Model</th><th>Correction</th><th>Mean MAPE</th><th>Mean MSE, shares²</th></tr>' + ''.join(volume_rows) + '</table>'
    document += '<h2>Limits</h2><ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in limitations) + '</ul>'
    document += '<details><summary>All metrics, paired intervals and audit context</summary><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre></details></html>'
    (ROOT / 'results.html').write_text(document)
    write_json(ROOT / 'report_manifest.json', {name: sha256_file(ROOT / name) for name in ('results.html', 'summary.json')})


if __name__ == '__main__':
    main()
