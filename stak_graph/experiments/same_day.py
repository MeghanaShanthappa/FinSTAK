"""Retiming of already-extracted news without repeating language-model inference."""

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from stak_graph.embeddings.pipeline import VectorCache, embed_events, load_vectors
from stak_graph.extraction.backends import BackendError, Generation
from stak_graph.extraction.pipeline import ExtractionPipeline, ResponseCache, participant_links
from stak_graph.extraction.registry import CompanyRegistry
from stak_graph.extraction.sources import Document
from stak_graph.extraction.schema import canonical_json, digest, schema_errors, strict_json
from stak_graph.io import read_jsonl, write_json, write_jsonl
from .data import extractor_documents, same_day_availability, sha256_file, write_csv
from .events import project


class ArchivedGeneratorIdentity:
    """All responses must come from the real, archived model's response cache."""
    def __init__(self, identity):
        if identity.get("simulated"):
            raise ValueError("Cannot retime a simulated extraction as real model output")
        self._identity = copy.deepcopy(identity)

    @property
    def identity(self): return copy.deepcopy(self._identity)

    def generate(self, messages, schema, request_id):
        raise BackendError("Missing archived response; no new model inference was requested")


class ArchivedVectorIdentity:
    def __init__(self, identity):
        if identity.get("simulated"):
            raise ValueError("Real semantic vectors are required")
        self._identity = copy.deepcopy(identity)

    @property
    def identity(self): return copy.deepcopy(self._identity)

    def encode(self, texts):
        raise ValueError("Event text changed; archived semantic vector cannot be reused")


def prepare_same_day(source, output, config):
    import csv
    source, output = Path(source), Path(output)
    if output.exists(): raise ValueError("Same-day prepared output already exists")
    original = json.loads((source / "config.json").read_text())
    for key in ("companies", "sample_start", "sample_end", "headline_sampling_salt", "horizon_sessions", "lookback_sessions"):
        if original[key] != config[key]: raise ValueError(f"Retiming must preserve the selected data: {key}")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("prices.csv", "benchmark.csv", "companies.csv"):
        shutil.copyfile(source / name, output / name)
    with (source / "calendar.csv").open(newline="") as file:
        sessions = list(csv.DictReader(file))
    for session in sessions:
        session["prediction_at"] = session["session_date"] + "T23:59:59Z"
    write_csv(output / "calendar.csv", list(sessions[0]), sessions)
    provenance = list(read_jsonl(source / "news_provenance.jsonl"))
    counts = Counter()
    for row in provenance:
        row["known_at"], precision = same_day_availability(row["published_at"])
        row["timestamp_precision"] = precision
        counts[precision] += 1
    inputs = extractor_documents([{key: value for key, value in row.items() if key != "timestamp_precision"} for row in provenance])
    write_jsonl(output / "documents.jsonl", inputs)
    write_jsonl(output / "news_provenance.jsonl", provenance)
    write_json(output / "config.json", config)
    manifest = copy.deepcopy(json.loads((source / "manifest.json").read_text()))
    manifest["config_sha256"] = digest(config)
    manifest["availability_policy"] = {"name": "same_day", "decision_cutoff": "23:59:59 UTC on trading-session date",
        "date_only_news": "available at end of recorded source date", "nonmidnight_news": "reported timestamp, unverified",
        "timestamp_counts": dict(counts), "source_prepared_manifest_sha256": sha256_file(source / "manifest.json")}
    manifest["limitations"] = [value for value in manifest["limitations"] if "two-calendar-day" not in value]
    manifest["limitations"].append("Date-only headlines are assumed available at 23:59:59 UTC on their recorded date; forecasts use that cutoff. Actual historical arrival times and source dates are unverified.")
    manifest["files"] = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "manifest.json", manifest)
    return manifest


def revalidate_extraction(source, prepared, output, cache_path):
    source, prepared, output = Path(source), Path(prepared), Path(output)
    if output.exists(): raise ValueError("Retimed extraction output already exists")
    old = json.loads((source / "manifest.json").read_text())
    backend = ArchivedGeneratorIdentity(old["extraction"]["generator"])
    cache = ResponseCache(cache_path)
    # Preserve the final schema-valid response, including any individual
    # records that will be quarantined again by evidence validation.
    try:
        for audit in read_jsonl(source / "audit.jsonl"):
            for chunk in audit["chunk_runs"]:
                valid = [attempt for attempt in chunk["attempts"] if "text" in attempt
                         and attempt.get("finish_reason") in ("stop", "eos", "completed")
                         and not schema_errors(strict_json(attempt["text"]))]
                if not valid: raise ValueError("Missing successful archived chunk response")
                attempt = valid[-1]
                cache.put(chunk["cache_key"], Generation(attempt["text"], attempt["finish_reason"], attempt.get("usage", {})))
        pipeline = ExtractionPipeline(backend, CompanyRegistry.from_csv(prepared / "companies.csv"),
                                      old["chunk_chars"], old["overlap_chars"], max_retries=0, cache=cache)
        if pipeline.provenance != old["extraction"]:
            raise ValueError("Extraction prompt/schema/generator changed; retiming cannot silently reuse incompatible outputs")
        output.mkdir(parents=True, exist_ok=False)
        counts = {"documents": 0, "document_errors": 0, "failed_chunks": 0, "events": 0,
                  "events_needing_review": 0, "relationships": 0, "participant_links": 0, "quarantined_records": 0, "cache_hits": 0}
        start = time.monotonic()
        with ExitStack() as stack:
            files = {name: stack.enter_context((output / f"{name}.jsonl").open("x"))
                     for name in ("documents", "events", "relationships", "event_company_links", "quarantine", "audit")}
            def write(name, value): files[name].write(canonical_json(value) + "\n")
            for row in read_jsonl(prepared / "documents.jsonl"):
                result = pipeline.extract(Document.from_dict(row))
                if result["status"] != "ok": raise ValueError("Retiming failed cached-response evidence validation")
                links = participant_links(result["events"])
                write("documents", result["document"])
                for name in ("events", "relationships", "quarantine"):
                    for record in result[name]: write(name, record)
                for link in links: write("event_company_links", link)
                write("audit", {"document_id": row["document_id"], "status": result["status"],
                                "extraction": result["extraction"], "chunk_runs": result["chunk_runs"],
                                "response_origin": "archived_real_model_generation_revalidated"})
                counts["documents"] += 1
                for key in ("events", "relationships"): counts[key] += len(result[key])
                counts["events_needing_review"] += sum(e["validation_status"] == "needs_review" for e in result["events"])
                counts["participant_links"] += len(links)
                counts["quarantined_records"] += len(result["quarantine"])
                counts["cache_hits"] += sum(chunk["cache_hit"] for chunk in result["chunk_runs"])
        counts["elapsed_seconds"] = round(time.monotonic() - start, 3)
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "extraction": pipeline.provenance,
                    "input_path": str((prepared / "documents.jsonl").resolve()),
                    "processed_input_sha256": sha256_file(prepared / "documents.jsonl"),
                    "registry_sha256": sha256_file(prepared / "companies.csv"),
                    "chunk_chars": old["chunk_chars"], "overlap_chars": old["overlap_chars"], "summary": counts,
                    "run_status": "completed", "is_accuracy_benchmark": False,
                    "retiming": {"policy": "same_day_end_of_recorded_date", "model_inference_repeated": False,
                                 "source_manifest_sha256": sha256_file(source / "manifest.json"),
                                 "source_audit_sha256": sha256_file(source / "audit.jsonl"),
                                 "validation": "Final archived model responses revalidated against unchanged source text and new availability metadata."}}
        write_json(output / "manifest.json", manifest)
        return manifest
    finally:
        cache.close()


def reuse_vectors(source, events_path, output, cache_path):
    vectors, manifest = load_vectors(source)
    cache = VectorCache(cache_path)
    try:
        for row in vectors.values(): cache.put(row["cache_key"], row["vector"])
        result = embed_events(list(read_jsonl(events_path)), ArchivedVectorIdentity(manifest["encoder"]), output, cache=cache)
        if result["cache_hits"] != result["count"]:
            raise ValueError("Expected every unchanged event text to reuse a real semantic embedding")
        return result
    finally:
        cache.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    data = prepare_same_day(args.source_run / "dataset", args.work / "prepared", config)
    extraction = revalidate_extraction(args.source_run / "source_extraction", args.work / "prepared", args.work / "extraction", args.work / "responses.sqlite3")
    projected = project(args.work / "extraction", args.work / "projected")
    embeddings = reuse_vectors(args.source_run / "event_embeddings", args.work / "projected/events.jsonl", args.work / "embeddings", args.work / "vectors.sqlite3")
    print(json.dumps({"availability": data["availability_policy"], "extraction": extraction["summary"],
                      "projected_events": projected["projected_events"], "embedding_count": embeddings["count"]}, indent=2))


if __name__ == "__main__": main()
