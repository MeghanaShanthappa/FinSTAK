"""Independent label joins, checkpoint reconstruction and future-input audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stak_graph.experiments.finstak_run import prepare
from stak_graph.experiments.timesfm_run import HYBRIDS, score
from stak_graph.experiments.timesfm_features import project_candle
from stak_graph.experiments.train import predict
from stak_graph.experiments.data import sha256_file
from stak_graph.experiments.model_sweep import save
from stak_graph.graph.market import TradingCalendar, read_bars
from stak_graph.models.timesfm_hybrid import TimesFMFinSTAK
from stak_graph.models.finstak import FinSTAK


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--expanded', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--reference', type=Path, default=Path('/var/tmp/mshanthappa-stak-graph/finstak_proposal_v1b'))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((args.output / 'plan.json').read_text())
    for filename, checksum in plan['data_hashes'].items():
        assert sha256_file(Path(filename)) == checksum, filename
    root = Path(__file__).resolve().parents[1]
    for filename, checksum in plan['code_hashes'].items():
        assert sha256_file(root / filename) == checksum, filename
    study, data, snapshot, _ = prepare(args, plan['config'])
    original_data = dict(data)
    saved = np.load(args.output / 'test_predictions.npz', allow_pickle=False)
    test_mask = study['masks']['test']
    test = np.flatnonzero(test_mask)
    np.testing.assert_array_equal(saved['targets'], study['targets'][test])
    np.testing.assert_array_equal(saved['anchors'], study['anchors'][test])
    calendar = TradingCalendar.from_csv(args.source / 'dataset/calendar.csv')
    bars = {(b.company_id, b.session_date): b for b in read_bars(args.source / 'dataset/prices.csv', calendar)}
    for i, (company, day) in enumerate(zip(saved['companies'], saved['dates'])):
        nxt = calendar.sessions[calendar.index[str(day)] + 1].session_date
        bar = bars[str(company), nxt]
        np.testing.assert_array_equal(saved['targets'][i], [getattr(bar, f) for f in ('open', 'high', 'low', 'close')])
        assert saved['anchors'][i] == bars[str(company), str(day)].close
    summary = json.loads((args.output / 'summary.json').read_text())
    for name, result in summary['models'].items():
        assert score(study, test_mask, saved[name]) == result, name
    for path in (args.output / 'native').glob('*.npz'):
        native = np.load(path, allow_pickle=False)
        np.testing.assert_array_equal(native['prediction'], project_candle(native['raw']))
    native_selection = json.loads((args.output / 'native_validation_selection.json').read_text())
    for mode, context in native_selection['selected_context'].items():
        scores = {}
        for candidate in plan['contexts']:
            cached = np.load(args.output / 'native' / f'{mode}_c{candidate}_validation.npz', allow_pickle=False)
            np.testing.assert_array_equal(cached['indices'], np.flatnonzero(study['masks']['validation']))
            value = study['anchors'][cached['indices'], None] * np.exp(cached['prediction'])
            actual = score(study, study['masks']['validation'], value)
            assert actual == native_selection['validation'][mode][str(candidate)]
            scores[candidate] = actual['metrics']['mean_OHLC_MAE_percent_of_today_close']
        assert context == min(plan['contexts'], key=lambda c: scores[c])
    assert json.loads((args.output / 'hybrid_selection_complete.json').read_text())['complete']
    base = np.load(args.output / 'native/selected_price_all.npz', allow_pickle=False)['prediction']
    prior = torch.zeros((len(data['prices']), 4), device=args.device)
    prior[torch.as_tensor(study['windows'][:, -1], device=args.device)] = torch.as_tensor(base, device=args.device, dtype=torch.float32)
    data['forecast_prior'] = prior
    data['prices'] = torch.cat((data['prices'], prior / torch.tensor(study['scaler']['target_scale'], device=args.device)), 1)
    windows = torch.as_tensor(study['windows'], device=args.device)
    indices = torch.as_tensor(test, device=args.device)
    checks, validation_predictions, validation_scores = {}, {}, {}
    validation_mask = study['masks']['validation']
    validation = torch.as_tensor(np.flatnonzero(validation_mask), device=args.device)
    for mode in HYBRIDS:
        individual, development = [], []
        for seed in plan['config']['seeds']:
            name = f'hybrid_{mode}_seed{seed}'
            path = args.output / 'checkpoints' / (name + '.pt')
            history = json.loads((args.output / 'histories' / (name + '.json')).read_text())
            assert sha256_file(path) == history['checkpoint_sha256']
            checkpoint = torch.load(path, weights_only=True, map_location=args.device)
            assert checkpoint['config'] == plan['config']
            assert checkpoint['scaler'] == study['scaler']
            chosen = next(r for r in history['epochs'] if r['epoch'] == checkpoint['selected_epoch'])
            assert chosen['validation_loss'] <= min(r['validation_loss'] for r in history['epochs']) + 1e-5
            model = TimesFMFinSTAK(study['scaler'], len(data['company_ids']), mode).to(args.device)
            model.load_state_dict(checkpoint['state_dict'])
            development.append(study['anchors'][validation_mask, None] * np.exp(predict(model, data, windows, validation).astype('float64')))
            predicted = saved['anchors'][:, None] * np.exp(predict(model, data, windows, indices).astype('float64'))
            np.testing.assert_allclose(predicted, saved[name], rtol=1e-7, atol=1e-7)
            individual.append(predicted)
            if seed == plan['config']['seeds'][0]:
                # All stock-day and event values beyond the validation boundary
                # are corrupted, while earlier forecasts must stay identical.
                changed = dict(data)
                future = torch.tensor([d['session_date'] > '2018-12-31' for d in snapshot['nodes']['stock_day']], device=args.device)
                event_future = torch.tensor([e['known_at'][:10] > '2018-12-31' for e in snapshot['nodes']['event']], device=args.device)
                for key in ('prices', 'forecast_prior'):
                    changed[key] = data[key].clone()
                    changed[key][future] += 100
                changed['events'] = data['events'].clone()
                changed['events'][event_future] += 100
                past = torch.as_tensor(np.flatnonzero(study['masks']['validation'])[::31], device=args.device)
                before = predict(model, data, windows, past)
                after = predict(model, changed, windows, past)
                difference = float(np.abs(before - after).max())
                assert difference <= 1e-7, (mode, difference)
                checks[mode] = {'future_perturbation_max_abs_change': difference}
            del model
        np.testing.assert_allclose(np.mean(individual, 0), saved['hybrid_' + mode], rtol=1e-7, atol=1e-7)
        validation_predictions['hybrid_' + mode] = np.mean(development, 0)
        validation_scores['hybrid_' + mode] = score(study, validation_mask, validation_predictions['hybrid_' + mode])
    original_development = []
    for seed in plan['config']['seeds']:
        model = FinSTAK(study['scaler'], companies=len(data['company_ids']), mode='no_decay').to(args.device)
        state = torch.load(args.reference / 'checkpoints' / f'no_decay_seed{seed}.pt', weights_only=True, map_location=args.device)
        model.load_state_dict(state['state_dict'])
        original_development.append(study['anchors'][validation_mask, None] * np.exp(predict(model, original_data, windows, validation).astype('float64')))
        del model
    validation_predictions['original_no_decay'] = np.mean(original_development, 0)
    validation_scores['original_no_decay'] = score(study, validation_mask, validation_predictions['original_no_decay'])
    save(args.output / 'hybrid_validation_metrics.json', validation_scores)
    np.savez_compressed(args.output / 'validation_predictions.npz', targets=study['targets'][validation_mask],
                        anchors=study['anchors'][validation_mask], **validation_predictions)
    audit = {'passed': True, 'independent_next_session_price_joins': len(test),
             'hybrid_checkpoints_reconstructed': len(HYBRIDS) * len(plan['config']['seeds']),
             'all_metrics_recomputed': True, 'native_candle_projection_recomputed': True,
             'source_and_implementation_hashes_match': True, 'future_input_audit': checks}
    save(args.output / 'verification.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
