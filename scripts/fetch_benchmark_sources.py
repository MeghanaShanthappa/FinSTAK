"""Restore pinned upstream repositories and the public Chronos-2 checkpoint.

Use an existing completed run as the manifest. This never changes an existing
checkout at another commit and does not install into the shared ML environment.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--external', type=Path, required=True)
    parser.add_argument('--model-cache', type=Path, default=Path('/tmp/stak-graph-models'))
    parser.add_argument('--download-chronos', action='store_true')
    args = parser.parse_args()
    plan = json.loads((args.run / 'plan.json').read_text())
    args.external.mkdir(parents=True, exist_ok=True)
    for name, source in plan['upstream_sources'].items():
        repository, commit = source['repository'], source['commit']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository) or not re.fullmatch(r'[a-f0-9]{40}', commit):
            raise ValueError('Invalid pinned repository metadata')
        target = args.external / name.removesuffix('_source')
        if not target.exists():
            target.mkdir()
            subprocess.run(['git', 'init', str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'https://github.com/' + repository + '.git', commit], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        observed = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if observed != commit:
            raise ValueError('Existing checkout differs; choose a fresh external directory: ' + str(target))
        (args.external / (name + '.json')).write_text(json.dumps(source, indent=2))
    checkpoint = json.loads((args.run / 'chronos_protocol.json').read_text())['checkpoint']
    (args.external / 'chronos_checkpoint.json').write_text(json.dumps(checkpoint, indent=2))
    if args.download_chronos:
        from huggingface_hub import snapshot_download
        snapshot_download(checkpoint['model'], revision=checkpoint['revision'], cache_dir=args.model_cache,
                          allow_patterns=['*.json', '*.safetensors'])
    print('Pinned source repositories restored. Set STAK_EXTERNAL to', args.external)


if __name__ == '__main__':
    main()
