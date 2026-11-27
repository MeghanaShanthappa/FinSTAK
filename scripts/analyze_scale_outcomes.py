"""Summarize graph value and validation-only recipe selection after the run."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path('/tmp/finstak-100x10-v1')


def block_sensitivity(reference, candidate):
    daily = []
    for a, b in zip(reference, candidate):
        error_a = (np.abs(a['prediction'] - a['truth']) / a['anchor'][:, None]).mean(1) * 100
        error_b = (np.abs(b['prediction'] - b['truth']) / b['anchor'][:, None]).mean(1) * 100
        _, inverse = np.unique(a['dates'], return_inverse=True)
        daily.append(np.bincount(inverse, weights=error_a - error_b) / np.bincount(inverse))
    result = {}
    for block in (10, 20, 60, 120):
        rng = np.random.default_rng(1701)
        samples = []
        for _ in range(2000):
            pieces = []
            for delta in daily:
                starts = rng.integers(0, len(delta), size=int(np.ceil(len(delta) / block)))
                index = ((starts[:, None] + np.arange(block)) % len(delta)).reshape(-1)[:len(delta)]
                pieces.append(delta[index])
            samples.append(np.concatenate(pieces).mean())
        result[str(block)] = {'95_percent_interval': np.quantile(samples, [.025, .975]).tolist(),
                              'replicates': 2000, 'uncorrected_exploratory_interval': True}
    return result


def main():
    sys.path.insert(0, str(ROOT / 'source_snapshot'))
    from stak_graph.experiments.scale_io import write_json
    from stak_graph.experiments.data import sha256_file
    report = ROOT / 'source_snapshot/scripts/report_scale_study.py'
    spec = importlib.util.spec_from_file_location('frozen_scale_report', report)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    phases = {stage: json.loads((ROOT / stage / 'plan.json').read_text()) for stage in ('data_stage', 'architecture_stage')}
    comparisons = {}
    sensitivity = {}
    price_only = module.load_predictions('data_stage', 'no_events')
    for variant in ('events', 'no_decay'):
        comparisons[variant + '_vs_price_only'] = module.paired_interval(price_only, module.load_predictions('data_stage', variant))
        sensitivity[variant + '_vs_price_only'] = block_sensitivity(price_only, module.load_predictions('data_stage', variant))
    selected = {}
    selected_baseline, selected_new, selected_all = [], [], []
    for index, year in enumerate(('2021', '2022', '2023')):
        scores, stage_of = {}, {}
        for stage, plan in phases.items():
            for variant in plan['variants']:
                values = [json.loads((ROOT / stage / year / f'{variant}_seed{seed}_history.json').read_text())['best_validation_MAE']
                          for seed in plan['seeds']]
                scores[variant] = float(np.mean(values))
                stage_of[variant] = stage
        baseline = min(phases['data_stage']['variants'], key=lambda v: (scores[v], v))
        new = min(phases['architecture_stage']['variants'], key=lambda v: (scores[v], v))
        overall = min(scores, key=lambda v: (scores[v], v))
        selected[year] = {'baseline': baseline, 'new_components': new, 'overall': overall,
                          'mean_seed_validation_MAE': scores, 'selection_uses_test_scores': False}
        selected_baseline.append(module.load_predictions('data_stage', baseline)[index])
        selected_new.append(module.load_predictions('architecture_stage', new)[index])
        selected_all.append(module.load_predictions(stage_of[overall], overall)[index])
    comparisons['validation_selected_new_vs_validation_selected_baseline'] = module.paired_interval(selected_baseline, selected_new)
    comparisons['validation_selected_overall_vs_validation_selected_baseline'] = module.paired_interval(selected_baseline, selected_all)
    sensitivity['validation_selected_new_vs_validation_selected_baseline'] = block_sensitivity(selected_baseline, selected_new)
    sensitivity['validation_selected_overall_vs_validation_selected_baseline'] = block_sensitivity(selected_baseline, selected_all)
    def pooled(parts):
        losses = [np.abs(p['prediction'] - p['truth']) / p['anchor'][:, None] for p in parts]
        return float(np.concatenate(losses).mean() * 100)
    write_json(ROOT / 'outcome.json', {
        'selection_rule': 'Minimum mean seed validation MAE; checkpoints already validation-selected. Test scores never enter recipe selection.',
        'selection': selected, 'comparisons': comparisons,
        'block_length_sensitivity_sessions': sensitivity,
        'pooled_test_MAE_percent': {'selected_baseline': pooled(selected_baseline),
                                  'selected_new_components': pooled(selected_new), 'selected_overall': pooled(selected_all)},
        'latest_fold_validation_recommendation': selected['2023']['overall'],
        'analysis_code_sha256': sha256_file(Path(__file__)),
        'limitations': ['Source evidence checks are automated, not human semantic certification.',
                       'Intervals are exploratory, uncorrected for multiple comparisons, and conditioned on these three years.',
                       'Mean seed validation error is the recipe criterion; ensemble test forecasts average the three price forecasts.',
                       'PCGrad does not guarantee unbiased conditional mean estimates or eliminate every task conflict.']})


if __name__ == '__main__':
    main()
