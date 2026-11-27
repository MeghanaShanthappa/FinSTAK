"""Tracked, fail-closed orchestration of the two authorized research stages."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from stak_graph.experiments.scale_io import write_json
from stak_graph.experiments.data import sha256_file

ROOT = Path('/tmp/finstak-100x10-v1')
PROJECT = Path('/home/mshanthappa/STAK-Graph')
PYTHON = '/var/tmp/mshanthappa-robotics/venv/bin/python'


def status(state, **extra):
    record = {'state': state, 'updated_at': datetime.now(timezone.utc).isoformat(), **extra}
    write_json(ROOT / 'pipeline_status.json', record)
    print(json.dumps(record), flush=True)


def run(label, command, environment):
    status('running', stage=label, command=command)
    with (ROOT / f'{label}.log').open('ab') as log:
        result = subprocess.run(command, cwd=ROOT / 'source_snapshot', env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{label} failed with exit {result.returncode}; see {ROOT / (label + ".log")}')


def wait_news():
    status('waiting_for_event_audit')
    job = json.loads((ROOT / 'news_job.json').read_text())
    for _ in range(120):
        audit_path = ROOT / 'dataset/event_audit.json'
        if audit_path.exists():
            try:
                audit = json.loads(audit_path.read_text())
                expected = job.get('expected_extractor_version_prefix', '4-')
                if len(audit.get('scans', {})) == 2 and audit.get('extractor_version', '').startswith(expected):
                    return audit
            except json.JSONDecodeError:
                pass
        try:
            os.kill(job['pid'], 0)
            process_state = Path(f'/proc/{job["pid"]}/stat').read_text().split()[2]
            if process_state == 'Z':
                raise RuntimeError('Source audit exited without a complete export; inspect news_v4.log')
        except ProcessLookupError as error:
            raise RuntimeError('Source audit exited without a complete export; inspect news_v4.log') from error
        time.sleep(30)
    raise TimeoutError('Event audit did not finish within the tracked waiting window')


def freeze():
    destination = ROOT / 'source_snapshot'
    if destination.exists():
        manifest = json.loads((destination / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if sha256_file(destination / name) != digest:
                raise ValueError('Frozen implementation changed: ' + name)
        return destination
    destination.mkdir()
    for name in ('stak_graph', 'scripts', 'configs', 'tests', 'examples'):
        shutil.copytree(PROJECT / name, destination / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT / 'pyproject.toml', destination / 'pyproject.toml')
    write_json(destination / 'manifest.json', {'files': {str(p.relative_to(destination)): sha256_file(p)
               for p in destination.rglob('*') if p.is_file()}})
    return destination


def main():
    try:
        audit = wait_news()
        # The final count includes deduplicated official SEC disclosures, so the
        # combined-data gate is enforced by build_scale_artifacts, not news alone.
        snapshot = freeze()
        import torch, numpy as np
        write_json(ROOT / 'runtime_versions.json', {'python': sys.version, 'torch': torch.__version__,
                   'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(), 'GPU': torch.cuda.get_device_name(0),
                   'deterministic_training': True, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        env = {**os.environ, 'PYTHONPATH': f'/tmp/finstak-timesfm-upstream/src:/tmp/finstak-timesfm-deps:/tmp/stak-graph-python-deps:{snapshot}',
               'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1'}
        run('frozen_tests', [PYTHON, '-m', 'unittest', 'discover', '-s', str(snapshot / 'tests'), '-v'], env)
        run('build_artifacts', [PYTHON, str(snapshot / 'scripts/build_scale_artifacts.py')], env)
        common = ['--packed', str(ROOT / 'packed'), '--epochs', '25', '--patience', '5', '--batch-dates', '16']
        run('data_stage', [PYTHON, '-m', 'stak_graph.experiments.scale_run', '--stage', 'data',
                          '--output', str(ROOT / 'data_stage'), *common], env)
        run('architecture_stage', [PYTHON, '-m', 'stak_graph.experiments.scale_run', '--stage', 'architecture',
                                  '--output', str(ROOT / 'architecture_stage'), *common], env)
        run('independent_verification', [PYTHON, str(snapshot / 'scripts/verify_scale_study.py')], env)
        run('report', [PYTHON, str(snapshot / 'scripts/report_scale_study.py')], env)
        status('complete', report=str(ROOT / 'results.html'))
        return 0
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    sys.exit(main())
