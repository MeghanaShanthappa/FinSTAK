"""Static text vectors keyed by text, model revision and task configuration."""

from array import array
import math
from pathlib import Path
import sqlite3
import sys

from stak_graph.extraction.schema import canonical_json, digest, strict_json
from stak_graph.io import read_jsonl, write_json, write_jsonl

TEXT_VERSION = "1.0.0"


def event_text(event):
    if event.get("validation_status") != "evidence_checked":
        raise ValueError("Only evidence_checked events can be embedded by default")
    description = event.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Event description must be nonempty")
    quotes = list(event.get("evidence", []))
    quotes += [span["quote"] for span in event.get("evidence_spans", [])]
    if not quotes or not all(isinstance(quote, str) and quote for quote in quotes):
        raise ValueError("Event must retain its supporting quotations")
    unique_quotes = list(dict.fromkeys(quotes))
    return description + "\nEvidence:\n" + "\n".join(unique_quotes)


def checked_vector(vector, dimension):
    if len(vector) != dimension or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vector):
        raise ValueError("Invalid embedding dimension or non-finite values")
    norm = math.sqrt(sum(v * v for v in vector))
    if not 0.999 <= norm <= 1.001:
        raise ValueError("Encoder must return a unit-length embedding")
    return vector


class VectorCache:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (cache_key TEXT PRIMARY KEY, vector TEXT NOT NULL)")

    def get(self, key):
        row = self.db.execute("SELECT vector FROM vectors WHERE cache_key=?", (key,)).fetchone()
        return strict_json(row[0]) if row else None

    def put(self, key, vector):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?)", (key, canonical_json(vector)))

    def close(self):
        self.db.close()


def embed_events(events, backend, output, cache=None, batch_size=16):
    output = Path(output)
    if output.exists():
        raise ValueError("Embedding output already exists; choose a new directory")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    identity = backend.identity
    dimension = identity["dimension"]
    selected, skipped, seen, fingerprints = [], [], set(), []
    for event in events:
        if event.get("validation_status") != "evidence_checked":
            skipped.append({"event_id": event.get("event_id"), "reason": "not_evidence_checked"})
            continue
        identifier = event.get("event_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Event IDs must be nonempty and unique")
        seen.add(identifier)
        text = event_text(event)
        key = digest({"encoder": identity, "text_version": TEXT_VERSION, "text": text})
        selected.append((event, text, key))
        fingerprints.append(digest(event))
    vectors, hits, pending = {}, 0, set()
    missing = []
    for _, text, key in selected:
        if key in vectors:
            continue
        cached = cache.get(key) if cache else None
        if cached is not None:
            vectors[key] = checked_vector(cached, dimension)
            hits += 1
        elif key not in pending:
            missing.append((text, key))
            pending.add(key)
    for start in range(0, len(missing), batch_size):
        group = missing[start:start + batch_size]
        generated = backend.encode([text for text, _ in group])
        if len(generated) != len(group):
            raise ValueError("Encoder returned the wrong batch size")
        for (_, key), vector in zip(group, generated):
            vectors[key] = checked_vector(vector, dimension)
            if cache:
                cache.put(key, vector)
    output.mkdir(parents=True, exist_ok=False)
    index = []
    with (output / "vectors.f32").open("xb") as binary:
        for row, (event, text, key) in enumerate(selected):
            values = array("f", vectors[key])
            if values.itemsize != 4:
                raise ValueError("This platform does not have four-byte float arrays")
            if sys.byteorder != "little":
                values.byteswap()
            binary.write(values.tobytes())
            index.append({"event_id": event["event_id"], "row": row, "text_sha256": digest(text),
                          "cache_key": key, "source_hash": event["source_hash"], "known_at": event["known_at"]})
    write_jsonl(output / "index.jsonl", index)
    write_jsonl(output / "skipped.jsonl", skipped)
    manifest = {"encoder": identity, "text_version": TEXT_VERSION, "dimension": dimension,
                "dtype": "float32", "byte_order": "little", "count": len(index), "cache_hits": hits,
                "input_events_hash": digest(fingerprints), "contains_synthetic_data": any(e[0].get("is_synthetic", False) for e in selected),
                "is_forecasting_evaluation": False}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_vectors(directory):
    directory = Path(directory)
    manifest = strict_json((directory / "manifest.json").read_text())
    if manifest.get("dtype") != "float32" or manifest.get("byte_order") != "little":
        raise ValueError("Unsupported vector file encoding")
    raw = (directory / "vectors.f32").read_bytes()
    dimension, count = manifest["dimension"], manifest["count"]
    if type(dimension) is not int or type(count) is not int or dimension <= 0 or count < 0 or len(raw) != count * dimension * 4:
        raise ValueError("Embedding binary length does not match its manifest")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    index = list(read_jsonl(directory / "index.jsonl"))
    if len(index) != count or [row["row"] for row in index] != list(range(count)):
        raise ValueError("Invalid embedding row index")
    if len({r["event_id"] for r in index}) != count:
        raise ValueError("Duplicate embedding event ID")
    return {row["event_id"]: {**row, "vector": checked_vector(list(values[i * dimension:(i + 1) * dimension]), dimension)}
            for i, row in enumerate(index)}, manifest
