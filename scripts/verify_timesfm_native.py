"""Replay fixed native validation batches and check cached feature transforms."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stak_graph.experiments.finstak_run import prepare
from stak_graph.experiments.timesfm_run import native_forecast
from stak_graph.experiments.timesfm_features import event_covariates
from stak_graph.experiments.model_sweep import save
from stak_graph.graph.market import TradingCalendar, read_bars


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--expanded', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    args = p.parse_args()
    args.device = 'cuda'
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    plan = json.loads((args.output / 'plan.json').read_text())
    study, data, snapshot, _ = prepare(args, plan['config'])
    direct, peer, audit = event_covariates(data, snapshot['nodes']['stock_day'], plan['config']['train_end'])
    assert audit == json.loads((args.output / 'event_feature_transform.json').read_text())
    provenance = json.loads((args.output / 'cache_provenance.json').read_text())
    previous = Path(next(iter(provenance['source']))).parents[1] / 'event_feature_transform.json'
    assert audit == json.loads(previous.read_text()), 'Cached native features used a different PCA transform'
    market = study['features'][:, 4:]
    covariates = {'price': None, 'market': market, 'direct': np.concatenate((market, direct), 1),
                  'graph': np.concatenate((market, direct, peer), 1)}
    calendar = TradingCalendar.from_csv(args.source / 'dataset/calendar.csv')
    bars = {(b.company_id, b.session_date): b for b in read_bars(args.source / 'dataset/prices.csv', calendar)}
    days = snapshot['nodes']['stock_day']
    prices = np.array([[getattr(bars[d['company_id'], d['session_date']], f) for f in ('open', 'high', 'low', 'close')] for d in days])
    from timesfm3 import TimesFM3Forecaster, ModelConfig
    model = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
                 revision=plan['checkpoint_revision'], cache_dir=str(args.cache), device='cuda', per_core_batch_size=16))
    selected = json.loads((args.output / 'native_validation_selection.json').read_text())['selected_context']
    folder = args.output / 'native_replay'
    folder.mkdir(exist_ok=False)
    checks = {}
    for mode, context in selected.items():
        cached = np.load(args.output / 'native' / f'{mode}_c{context}_validation.npz', allow_pickle=False)
        indices = cached['indices'][:16]
        actual = native_forecast(model, study, days, prices, covariates[mode], indices, context, 16, folder / (mode + '.npz'))
        error = float(np.abs(actual - cached['prediction'][:16]).max())
        assert error <= 5e-6, (mode, error)
        checks[mode] = {'replayed_examples': len(indices), 'max_abs_log_forecast_difference': error}
    save(args.output / 'native_verification.json', {'passed': True, 'feature_transform_exact_match': True,
          'cached_validation_transform_exact_match': True, 'native_batch_replay': checks})
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
