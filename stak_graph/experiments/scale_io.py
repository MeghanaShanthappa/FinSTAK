"""Atomic updates for explicitly resumable, derived study artifacts only."""
import json
import os
from pathlib import Path
import tempfile


def _write(path, records, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as out:
            temporary = Path(out.name)
            if lines:
                for record in records:
                    out.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            else:
                out.write(json.dumps(records, sort_keys=True, allow_nan=False) + '\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path, record):
    _write(path, record, False)


def write_jsonl(path, records):
    _write(path, records, True)
