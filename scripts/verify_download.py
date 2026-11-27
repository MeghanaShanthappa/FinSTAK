"""Verify the retained research release before extracting it."""
import argparse
import hashlib
from pathlib import Path

EXPECTED = '5779df4319d688e7084a8adfac94c2d26835c2d4d60ce552d882f4fc500dac74'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.archive.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != EXPECTED:
        raise SystemExit('Archive SHA256 mismatch; do not extract this file.')
    print('PASS: retained research archive SHA256 matches.')


if __name__ == '__main__':
    main()
