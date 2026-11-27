"""Lock a recommendation from validation, then report exploratory test results."""
import argparse
import json
from pathlib import Path
import numpy as np
from stak_graph.experiments.timesfm_run import render
from stak_graph.experiments.metrics import block_interval
from stak_graph.experiments.model_sweep import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--encoders', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    primary_summary = json.loads((args.primary / 'summary.json').read_text())
    native = json.loads((args.primary / 'native_validation_selection.json').read_text())
    hybrid_validation = json.loads((args.primary / 'hybrid_validation_metrics.json').read_text())
    encoder = {str(json.loads((p / 'plan.json').read_text())['context']):
               (p, json.loads((p / 'summary.json').read_text())) for p in args.encoders}
    mae = lambda entry: entry['metrics']['mean_OHLC_MAE_percent_of_today_close']
    validation_scores = dict(hybrid_validation)
    for mode, context in native['selected_context'].items():
        validation_scores['native_' + mode] = native['validation'][mode][str(context)]
    common = min(encoder, key=lambda c: mae(encoder[c][1]['validation_models']['encoder_local']))
    contexts = {}
    for mode in ('local', 'direct', 'graph', 'no_decay'):
        name = 'encoder_' + mode
        context = min(encoder, key=lambda c: mae(encoder[c][1]['validation_models'][name]))
        contexts[name + '_tuned'] = context
        validation_scores[name + '_tuned'] = encoder[context][1]['validation_models'][name]
    selected = min(validation_scores, key=lambda name: mae(validation_scores[name]))
    best_new = min((name for name in validation_scores if not name.startswith('original_')),
                   key=lambda name: mae(validation_scores[name]))
    selection = {'validation_selected_model': selected,
                 'best_new_model_by_validation': best_new,
                 'selection_criterion': 'Minimum 2018 normalized OHLC MAE; 2019 test never used by selection function',
                 'encoder_context_per_mode': contexts, 'common_encoder_context': int(common),
                 'native_context_per_mode': native['selected_context'],
                 'validation_MAE_percent': {name: mae(value) for name, value in validation_scores.items()}}
    # Persist the recommendation before reading any test predictions.
    save(args.output / 'validation_selection.json', selection)
    primary = np.load(args.primary / 'test_predictions.npz', allow_pickle=False)
    arrays = {k: primary[k] for k in primary.files}
    models = dict(primary_summary['models'])
    common_summary = encoder[common][1]
    common_arrays = np.load(encoder[common][0] / 'test_predictions.npz', allow_pickle=False)
    for mode in ('local', 'direct', 'graph', 'no_decay'):
        name = 'encoder_' + mode
        models[name] = common_summary['models'][name]
        arrays[name] = common_arrays[name]
        tuned = name + '_tuned'
        path, summary = encoder[contexts[tuned]]
        candidate_arrays = np.load(path / 'test_predictions.npz', allow_pickle=False)
        models[tuned] = summary['models'][name]
        arrays[tuned] = candidate_arrays[name]
    truth, anchors, dates = arrays['targets'], arrays['anchors'], arrays['dates']
    comparisons = {}
    pairs = [(selected, 'original_no_decay'), (best_new, 'original_no_decay'), ('hybrid_graph', 'hybrid_local'),
             ('hybrid_graph', 'hybrid_direct'), ('hybrid_graph', 'hybrid_no_decay'),
             ('encoder_graph', 'encoder_local'), ('encoder_graph', 'encoder_direct'),
             ('encoder_graph', 'encoder_no_decay'), ('encoder_graph_tuned', 'original_no_decay'),
             ('native_graph', 'native_direct')]
    for candidate, baseline in pairs:
        gain = (np.abs(arrays[baseline] - truth) - np.abs(arrays[candidate] - truth)).mean(1) / anchors * 100
        comparisons[candidate + '_vs_' + baseline] = block_interval([gain[dates == day].mean() for day in sorted(set(dates))])
    ordered = {selected: models[selected]}
    ordered['original_no_decay'] = models['original_no_decay']
    ordered.update(models)
    summary = {'models': ordered, 'validation_selection': selection,
               'paired_MAE_reduction_percentage_points': comparisons,
               'split_counts': primary_summary['split_counts'],
               'limitations': common_summary['limitations'] + [
                   'Encoder rows without tuned suffix share the context selected by the local encoder validation score.',
                   'Tuned encoder rows independently select each mode context by validation; compare common-context rows to isolate graph effects.',
                   'The final recommendation competes with the preserved original no-decay model on validation, including fallback to that original model.',
                   'Native direct/graph comparisons may use different validation-selected contexts; they compare pipelines rather than isolate graph topology.']}
    np.savez_compressed(args.output / 'test_predictions.npz', **arrays)
    save(args.output / 'summary.json', summary)
    render(args.output, summary)
    selected_mae, original_mae = mae(models[selected]), mae(models['original_no_decay'])
    outcome = {'selected_model': selected, 'test_normalized_MAE_percent': selected_mae,
               'best_new_model_by_validation': best_new,
               'best_new_model_test_MAE_percent': mae(models[best_new]),
               'best_new_model_paired_gain': comparisons[best_new + '_vs_original_no_decay'],
               'original_normalized_MAE_percent': original_mae,
               'relative_error_reduction_percent': 100 * (original_mae - selected_mae) / original_mae,
               'paired_gain': comparisons[selected + '_vs_original_no_decay']}
    save(args.output / 'outcome.json', outcome)
    print(json.dumps(outcome, indent=2))


if __name__ == '__main__':
    main()
