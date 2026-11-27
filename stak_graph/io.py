"""Small shared serialization helpers; files are never interpreted as Python objects."""

from pathlib import Path

from .extraction.schema import canonical_json, strict_json


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = strict_json(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected a JSON object")
                yield record
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(canonical_json(value) + "\n")


def write_jsonl(path, records):
    with Path(path).open("x", encoding="utf-8") as output:
        for record in records:
            output.write(canonical_json(record) + "\n")
