"""Independent label joins, checkpoint replay and future-input perturbations."""
import json
from pathlib import Path
import numpy as np
import torch

from stak_graph.io import read_jsonl
from stak_graph.experiments.scale_io import write_json
from stak_graph.experiments.scale_panel import PanelBatches
from stak_graph.experiments.scale_run import FOLDS
from stak_graph.experiments.data import sha256_file
from stak_graph.models.scale_finstak import ScaleFinSTAK

ROOT = Path('/tmp/finstak-100x10-v1')


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    events = list(read_jsonl(ROOT / 'expanded/events.jsonl'))
    assertions, checkpoints = [], 0
    for stage in ('data_stage', 'architecture_stage'):
        plan = json.loads((ROOT / stage / 'plan.json').read_text())
        if (sha256_file(ROOT / 'packed/panel.npz') != plan['panel_sha256']
                or sha256_file(ROOT / 'packed/event_arrays.npz') != plan['event_arrays_sha256']
                or sha256_file(ROOT / 'packed/event_arrays_finbert.npz') != plan['finbert_arrays_sha256']):
            raise ValueError('Training inputs changed after fitting')
        for name, expected in plan['code_hashes'].items():
            if sha256_file(ROOT / 'source_snapshot' / name) != expected:
                raise ValueError('Frozen training implementation changed: ' + name)
        for year, periods in FOLDS.items():
            panel = PanelBatches(ROOT / 'packed', *periods)
            selected = json.loads((ROOT / stage / year / 'validation_selection_complete.json').read_text())
            if not selected['complete'] or selected['test_used_for_checkpoint_selection']:
                raise ValueError('Missing pre-test checkpoint lock')
            for variant in plan['variants']:
                panel.select_encoder(variant)
                path = ROOT / stage / year / f'{variant}_predictions.npz'
                p = np.load(path)
                ids = panel.splits['test']
                expected_truth = panel.raw[ids + 1, :, :4].reshape(-1, 4)
                anchor = panel.raw[ids, :, 3].reshape(-1)
                if not np.array_equal(p['truth'], expected_truth) or not np.array_equal(p['anchor'], anchor):
                    raise ValueError('Saved labels disagree with independent next-session join')
                if not np.array_equal(p['dates'], np.repeat(panel.dates[ids], panel.c)):
                    raise ValueError('Saved forecast dates differ')
                for seed in plan['seeds']:
                    checkpoint = ROOT / stage / year / f'{variant}_seed{seed}.pt'
                    history = json.loads((ROOT / stage / year / f'{variant}_seed{seed}_history.json').read_text())
                    if sha256_file(checkpoint) != history['checkpoint_sha256']:
                        raise ValueError('Checkpoint checksum mismatch')
                    checkpoints += 1
                seed = plan['seeds'][0]
                saved = torch.load(ROOT / stage / year / f'{variant}_seed{seed}.pt', map_location='cuda', weights_only=True)
                model = ScaleFinSTAK(panel.scaler, panel.c, panel.events.shape[1], variant, category_count=panel.category_count).cuda().eval()
                model.load_state_dict(saved['state_dict'])
                # Reconstruct with the original evaluation batch shape. Changing
                # GRU/matrix batch dimensions changes floating-point kernels.
                replay_data = panel.batch(ids[:plan['batch_dates']])
                with torch.inference_mode():
                    reconstructed = model(replay_data)[0].cpu().numpy()
                old = np.log(p[f'seed{seed}'][:len(reconstructed)] / anchor[:len(reconstructed), None])
                replay_delta = float(np.abs(reconstructed - old).max())
                if replay_delta > 2e-6:
                    raise ValueError(f'Checkpoint replay differs: {stage} {year} {variant}: {replay_delta}')
                # Keep the stronger early two-date cutoff for causality checks.
                sample_ids = ids[:2]
                data = panel.batch(sample_ids)
                with torch.inference_mode():
                    before = model(data)[0].cpu().numpy()
                cutoff = str(panel.dates[sample_ids[-1]])
                future_events = torch.tensor([e['known_at'][:10] > cutoff for e in events], device='cuda')
                modified = dict(data)
                modified['events'] = data['events'].clone()
                modified['events'][future_events] += 10
                # Rebuild historical batches after altering all later prices.
                future_days = torch.tensor(panel.dates > cutoff, device='cuda')
                original_future = panel.features[future_days].clone()
                panel.features[future_days] += 10
                changed = panel.batch(sample_ids)
                changed['events'] = modified['events']
                changed['target'] = changed['target'] + 123
                changed['volume_target'] = changed['volume_target'] + 123
                with torch.inference_mode():
                    after = model(changed)[0].cpu().numpy()
                panel.features[future_days] = original_future
                future_delta = float(np.abs(before - after).max())
                if future_delta > 1e-7:
                    raise ValueError('Future inputs changed earlier forecasts')
                assertions.append({'stage': stage, 'year': year, 'variant': variant,
                                   'label_rows_joined': len(expected_truth), 'replay_max_logratio_difference': replay_delta,
                                   'future_perturbation_max_logratio_difference': future_delta})
                del model
            del panel
            torch.cuda.empty_cache()
    write_json(ROOT / 'independent_verification.json', {'passed': True, 'checkpoints_hashed': checkpoints,
               'checks': assertions, 'verification_code_sha256': sha256_file(Path(__file__)),
               'replay_batching': 'Original evaluation batch size; unchanged 2e-6 logratio tolerance. Causality retains the independent two-date cutoff.',
               'scope': 'Independent labels, saved-checkpoint reconstruction and model-input causality; not a pretraining-overlap or live-timestamp certification.'})
    print(json.dumps({'passed': True, 'checkpoints_hashed': checkpoints, 'model_fold_checks': len(assertions)}), flush=True)


if __name__ == '__main__':
    main()
