"""Reproduce the expanded study using its immutable released code and inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(study):
    plans = {}
    for stage in ('data_stage', 'architecture_stage'):
        plan = json.loads((study / stage / 'plan.json').read_text())
        for name, expected in plan['code_hashes'].items():
            if checksum(study / 'source_snapshot' / name) != expected:
                raise ValueError('Frozen source mismatch: ' + name)
        for name, key in [('panel.npz', 'panel_sha256'),
                          ('event_arrays.npz', 'event_arrays_sha256'),
                          ('event_arrays_finbert.npz', 'finbert_arrays_sha256')]:
            if checksum(study / 'packed' / name) != plan[key]:
                raise ValueError('Packed input mismatch: ' + name)
        plans[stage] = plan
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    study = args.study.resolve()
    plans = verify(study)
    if args.check_only:
        print('PASS: recorded frozen source and packed-input hashes match for both stages.')
        return
    if args.output is None:
        parser.error('--output is required unless --check-only is used')
    output = args.output.resolve()
    if output.exists():
        parser.error('Choose a new output directory; existing results will not be overwritten.')
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select an accessible GPU or explicitly use --device cpu.')
    output.mkdir(parents=True)
    environment = {**os.environ, 'PYTHONPATH': str(study / 'source_snapshot'),
                   'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    status = {'state': 'running', 'source_study': str(study),
              'device': args.device, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Reproduction of existing historical test folds; not a new holdout.'}
    status_path = output / 'reproduction_status.json'
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    try:
        for stage, plan in plans.items():
            command = [sys.executable, '-m', 'stak_graph.experiments.scale_run',
                       '--packed', str(study / 'packed'), '--output', str(output / stage),
                       '--stage', plan['stage'], '--device', args.device,
                       '--epochs', str(plan['epochs']), '--patience', str(plan['patience']),
                       '--batch-dates', str(plan['batch_dates']),
                       '--seeds', *[str(seed) for seed in plan['seeds']],
                       '--folds', *plan['folds'], '--variants', *plan['variants']]
            print('Starting ' + stage, flush=True)
            with (output / (stage + '.log')).open('w') as log:
                subprocess.run(command, cwd=study / 'source_snapshot', env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        status['state'] = 'training_complete'
    except Exception as error:
        status.update(state='failed', error=str(error))
        raise
    finally:
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
