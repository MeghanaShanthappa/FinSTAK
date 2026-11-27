"""Restore optional duplicate vector files from the retained packed matrices."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.study / 'embedding_restore_manifest.json').read_text())
    for row in manifest:
        destination = args.study / row['destination']
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Existing vector file differs; refusing to overwrite it')
            continue
        values = np.load(args.study / row['packed'])['embeddings'].astype('<f4', copy=False).tobytes()
        if hashlib.sha256(values).hexdigest() != row['sha256']:
            raise ValueError('Packed vectors do not reproduce the original binary')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as out:
            out.write(values)
        print(destination)


if __name__ == '__main__':
    main()
