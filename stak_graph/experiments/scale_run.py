"""Walk-forward, validation-selected experiments for the 100 x 10 study.

Stage 1 holds the original model design fixed while expanding the data.
Stage 2 tests proposal components with paired seeds and the same data folds.
No test score participates in checkpoints or experiment configuration.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .scale_io import write_json
from stak_graph.models.scale_finstak import ScaleFinSTAK, VARIANTS, task_losses, pcgrad_backward
from .scale_panel import PanelBatches
from .next_day_data import metrics
from .finstak_data import tolerance_accuracy
from .data import sha256_file

FOLDS = {
    '2021': ('2019-12-31', '2020-12-31', '2021-12-31'),
    '2022': ('2020-12-31', '2021-12-31', '2022-12-31'),
    '2023': ('2021-12-31', '2022-12-31', '2023-12-28'),
}
STAGE_ONE = ('no_events', 'no_decay', 'events', 'lstm', 'tgcn')
STAGE_TWO = tuple(v for v in VARIANTS if v not in STAGE_ONE)


def groups(indices, batch_dates):
    for begin in range(0, len(indices), batch_dates):
        yield indices[begin:begin + batch_dates]


@torch.no_grad()
def predict(model, panel, indices, batch_dates):
    model.eval()
    prices, volumes, directions = [], [], []
    for dates in groups(indices, batch_dates):
        data = panel.batch(dates)
        price, volume = model(data)
        prices.append(price.cpu().numpy())
        if volume is not None:
            volumes.append(volume.cpu().numpy())
        if model.direction_logits is not None:
            directions.append(model.direction_logits.cpu().numpy())
    return (np.concatenate(prices), np.concatenate(volumes) if volumes else None,
            np.concatenate(directions) if directions else None)


def normalized_error(log_prediction, panel, ids):
    actual = panel.raw[ids + 1, :, :4].reshape(-1, 4)
    anchor = panel.raw[ids, :, 3].reshape(-1)
    prediction = anchor[:, None] * np.exp(log_prediction.astype(np.float64))
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite price forecast')
    return float((np.abs(prediction - actual) / anchor[:, None]).mean() * 100)


def fit(panel, variant, seed, config, output):
    checkpoint = output / f'{variant}_seed{seed}.pt'
    history_path = output / f'{variant}_seed{seed}_history.json'
    if checkpoint.exists() and history_path.exists():
        record = json.loads(history_path.read_text())
        if record['checkpoint_sha256'] != sha256_file(checkpoint):
            raise ValueError('Existing checkpoint checksum mismatch')
        return record
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ScaleFinSTAK(panel.scaler, panel.c, panel.events.shape[1], variant, category_count=panel.category_count).to(panel.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    rng = np.random.default_rng(seed)
    gradient_rng = torch.Generator().manual_seed(seed)
    best, best_epoch, state = float('inf'), 0, None
    history, started = [], time.monotonic()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        losses = []
        order = rng.permutation(panel.splits['train'])
        for ids in groups(order, config['batch_dates']):
            data = panel.batch(ids)
            optimizer.zero_grad(set_to_none=True)
            task = task_losses(model(data), data, panel.scaler, model)
            if not all(torch.isfinite(value) for value in task):
                raise ValueError('Nonfinite training task loss')
            if variant in ('volume_pcgrad', 'combined'):
                pcgrad_backward(task, model.parameters(), gradient_rng)
                objective = torch.stack([v.detach() for v in task]).mean()
            elif variant == 'volume_uncertainty':
                variance = model.log_variances[:len(task)].clamp(-5, 5)
                objective = (torch.exp(-variance) * torch.stack(task) + variance).mean()
                objective.backward()
            else:
                objective = torch.stack(task).mean()
                objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(objective.detach()))
        validation, _, _ = predict(model, panel, panel.splits['validation'], config['batch_dates'])
        # Identical price objective for checkpoint selection in every condition.
        value = normalized_error(validation, panel, panel.splits['validation'])
        if value < best - 1e-6:
            best, best_epoch = value, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        record = {'variant': variant, 'seed': seed, 'epoch': epoch, 'training_objective': float(np.mean(losses)),
                  'validation_OHLC_MAE_percent': value, 'selected_epoch': best_epoch,
                  'elapsed_seconds': round(time.monotonic() - started, 2)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if epoch - best_epoch >= config['patience']:
            break
    torch.save({'state_dict': state, 'scaler': panel.scaler, 'companies': panel.companies.tolist(),
                'variant': variant, 'seed': seed, 'selected_epoch': best_epoch, 'config': config}, checkpoint)
    result = {'history': history, 'best_validation_MAE': best, 'selected_epoch': best_epoch,
              'checkpoint_sha256': sha256_file(checkpoint), 'elapsed_seconds': time.monotonic() - started}
    write_json(history_path, result)
    return result


def volume_metrics(truth, prediction):
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Invalid volume forecast')
    positive = truth > 0
    return {'MSE_shares_squared': float(((prediction - truth) ** 2).mean()),
            'MAE_shares': float(np.abs(prediction - truth).mean()),
            'MAPE_positive_volume_percent': float((np.abs(prediction[positive] - truth[positive]) / truth[positive]).mean() * 100),
            'MAPE_coverage': int(positive.sum()), 'samples': len(truth)}


def evaluate_one(panel, variant, seeds, config, output):
    test_ids, val_ids = panel.splits['test'], panel.splits['validation']
    truth = panel.raw[test_ids + 1, :, :4].reshape(-1, 4)
    anchor = panel.raw[test_ids, :, 3].reshape(-1)
    companies = np.tile(panel.companies, len(test_ids))
    predictions, individual, val_predictions, volume_test, volume_val, directions = [], {}, [], [], [], []
    for seed in seeds:
        saved = torch.load(output / f'{variant}_seed{seed}.pt', map_location=panel.device, weights_only=True)
        model = ScaleFinSTAK(panel.scaler, panel.c, panel.events.shape[1], variant, category_count=panel.category_count).to(panel.device)
        model.load_state_dict(saved['state_dict'])
        test, test_volume, direction = predict(model, panel, test_ids, config['batch_dates'])
        validation, val_volume, _ = predict(model, panel, val_ids, config['batch_dates'])
        price = anchor[:, None] * np.exp(test.astype(np.float64))
        predictions.append(price)
        individual[f'seed{seed}'] = price
        val_predictions.append(validation)
        if test_volume is not None:
            volume_test.append(test_volume)
            volume_val.append(val_volume)
        if direction is not None:
            directions.append(direction)
    price = np.mean(predictions, axis=0)
    result = {'price': metrics(truth, price, anchor, companies),
              'tolerance': tolerance_accuracy(truth, price, anchor), 'seeds': seeds}
    arrays = {'truth': truth, 'anchor': anchor, 'prediction': price, 'dates': np.repeat(panel.dates[test_ids], panel.c),
              'companies': companies, **individual}
    if directions:
        result['auxiliary_direction_accuracy'] = float(((np.mean(directions, axis=0) > 0) == (truth[:, 3] > anchor)).mean())
    if volume_test:
        test_mu = panel.volume_mean[test_ids].reshape(-1)
        test_scale = panel.volume_scale[test_ids].reshape(-1)
        val_mu = panel.volume_mean[val_ids].reshape(-1)
        val_scale = panel.volume_scale[val_ids].reshape(-1)
        test_log = test_mu + test_scale * np.mean(volume_test, axis=0)
        val_log = val_mu + val_scale * np.mean(volume_val, axis=0)
        val_truth = panel.raw[val_ids + 1, :, 4].reshape(-1)
        residual = (np.log1p(val_truth) - val_log).reshape(len(val_ids), panel.c)
        # Calibrate only on the completed validation year, independently per stock.
        gaussian = residual.mean(0) + .5 * residual.var(0)
        smearing = np.exp(residual).mean(0)
        volume_truth = panel.raw[test_ids + 1, :, 4].reshape(-1)
        result['volume'] = {}
        for name, correction in [('naive', np.zeros(panel.c)), ('gaussian', gaussian), ('smearing', np.log(smearing))]:
            predicted_volume = np.maximum(0, np.expm1(test_log + np.tile(correction, len(test_ids))))
            result['volume'][name] = volume_metrics(volume_truth, predicted_volume)
            arrays[f'volume_{name}'] = predicted_volume
        arrays['volume_truth'] = volume_truth
        write_json(output / f'{variant}_volume_calibration.json', {'fit_period': 'validation only',
                   'validation_observations_per_stock': len(val_ids), 'companies': panel.companies.tolist(),
                   'gaussian_log_correction': gaussian.tolist(), 'smearing_factor': smearing.tolist(),
                   'limitations': 'Per-stock pooled residual correction is not a guarantee of conditional mean calibration.'})
    np.savez(output / f'{variant}_predictions.npz', **arrays)
    write_json(output / f'{variant}_metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('data', 'architecture'), required=True)
    parser.add_argument('--folds', nargs='+', choices=tuple(FOLDS), default=list(FOLDS))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 41])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-dates', type=int, default=8)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_dates < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive budgets and unique seeds required')
    variants = args.variants or list(STAGE_ONE if args.stage == 'data' else STAGE_TWO)
    config = {'stage': args.stage, 'folds': args.folds, 'variants': variants, 'seeds': args.seeds,
              'epochs': args.epochs, 'patience': args.patience, 'batch_dates': args.batch_dates,
              'selection': 'Validation normalized OHLC MAE, same criterion for all variants',
              'event_arrays_sha256': sha256_file(args.packed / 'event_arrays.npz'),
              'panel_sha256': sha256_file(args.packed / 'panel.npz'),
              'finbert_arrays_sha256': sha256_file(args.packed / 'event_arrays_finbert.npz')
                  if (args.packed / 'event_arrays_finbert.npz').exists() else None,
              'code_hashes': {str(p.relative_to(Path(__file__).parents[2])): sha256_file(p)
                              for p in Path(__file__).parents[2].rglob('*.py')}}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != config:
        raise ValueError('Refusing to mix checkpoints from changed code/data/configuration')
    write_json(plan_path, config)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    for fold in args.folds:
        output = args.output / fold
        output.mkdir(exist_ok=True)
        panel = PanelBatches(args.packed, *FOLDS[fold], device=args.device)
        write_json(output / 'split_audit.json', {'periods': FOLDS[fold], 'scaler': panel.scaler,
                   'samples': {k: len(v) * panel.c for k, v in panel.splits.items()},
                   'split_dates': {k: [str(panel.dates[v[0]]), str(panel.dates[v[-1]])] for k, v in panel.splits.items()}})
        for variant in variants:
            panel.select_encoder(variant)
            for seed in args.seeds:
                write_json(args.output / 'status.json', {'state': 'training', 'fold': fold, 'variant': variant, 'seed': seed,
                           'updated_at': datetime.now(timezone.utc).isoformat()})
                fit(panel, variant, seed, config, output)
        # Lock every checkpoint in this fold before opening its test scores.
        write_json(output / 'validation_selection_complete.json', {'complete': True, 'test_used_for_checkpoint_selection': False})
        result = {}
        for variant in variants:
            panel.select_encoder(variant)
            result[variant] = evaluate_one(panel, variant, args.seeds, config, output)
        # Same-cohort trivial baselines, not old 16-stock headline scores.
        ids = panel.splits['test']
        truth = panel.raw[ids + 1, :, :4].reshape(-1, 4)
        anchor = panel.raw[ids, :, 3].reshape(-1)
        company = np.tile(panel.companies, len(ids))
        for name, price in [('previous_close', np.repeat(anchor[:, None], 4, axis=1)),
                            ('previous_OHLC', panel.raw[ids, :, :4].reshape(-1, 4))]:
            result[name] = {'price': metrics(truth, price, anchor, company), 'tolerance': tolerance_accuracy(truth, price, anchor)}
        up = truth[:, 3] > anchor
        training_majority_up = bool((panel.log_targets[panel.splits['train'], :, 3] > 0).mean() > .5)
        result['previous_close']['direction_baselines'] = {
            'always_up_accuracy': float(up.mean()), 'always_down_accuracy': float((~up).mean()),
            'training_majority_predicts_up': training_majority_up,
            'training_majority_accuracy': float((up == training_majority_up).mean())}
        volume_truth = panel.raw[ids + 1, :, 4].reshape(-1)
        result['previous_close']['volume_baselines'] = {
            'previous_volume': volume_metrics(volume_truth, panel.raw[ids, :, 4].reshape(-1)),
            'trailing_log_volume_median': volume_metrics(volume_truth, np.expm1(panel.volume_mean[ids].reshape(-1)))}
        write_json(output / 'summary.json', result)
        del panel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(args.output / 'status.json', {'state': 'complete', 'updated_at': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
