"""Batch JSONL interface, designed to be usable without any GPU dependencies."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

from .backends import BackendError, HTTPBackend, ReplayBackend, TransformersBackend
from .pipeline import ExtractionPipeline, ResponseCache, participant_links
from .registry import CompanyRegistry
from .schema import EXTRACTION_SCHEMA, canonical_json, strict_json
from .sources import Document


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="STAK-Graph financial event extraction")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("schema", help="Print the exact JSON schema used for generation")
    extract = commands.add_parser("extract", help="Extract events from document JSONL")
    extract.add_argument("--input", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path, help="A new output directory; existing directories are never overwritten")
    extract.add_argument("--registry", type=Path, help="CSV containing company_id,name,aliases,valid_from,valid_to")
    extract.add_argument("--backend", choices=["transformers", "http", "replay"], default="transformers")
    extract.add_argument("--model", default="Qwen/Qwen3-4B")
    extract.add_argument("--revision", help="Pinned model commit (recommended); actual local model commit is recorded")
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--model-cache", help="Hugging Face model cache; use temporary storage if home storage is limited")
    extract.add_argument("--local-files-only", action="store_true")
    extract.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    extract.add_argument("--replay-responses", type=Path)
    extract.add_argument("--cache", type=Path, help="Optional SQLite response cache")
    extract.add_argument("--chunk-chars", type=int, default=6000)
    extract.add_argument("--overlap-chars", type=int, default=500)
    extract.add_argument("--max-new-tokens", type=int, default=3000)
    extract.add_argument("--max-input-tokens", type=int, default=8192)
    extract.add_argument("--max-retries", type=int, default=1)
    extract.add_argument("--limit", type=int)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "schema":
        print(canonical_json(EXTRACTION_SCHEMA))
        return 0
    try:
        return extract_batch(arguments)
    except (ValueError, OSError, BackendError) as exc:
        print(f"Extraction error: {exc}", file=sys.stderr)
        return 2


def extract_batch(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}; choose a new directory")
    if not args.input.is_file():
        raise ValueError(f"Input file does not exist: {args.input}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.chunk_chars < 100 or args.overlap_chars < 0 or args.overlap_chars >= args.chunk_chars // 2 or args.max_retries < 0:
        raise ValueError("Invalid chunk/overlap/retry parameters")
    registry = CompanyRegistry.from_csv(args.registry) if args.registry else CompanyRegistry()
    if args.backend == "replay":
        if not args.replay_responses:
            raise ValueError("--replay-responses is required for the replay backend")
        backend = ReplayBackend(args.replay_responses)
    elif args.backend == "http":
        backend = HTTPBackend(args.base_url, args.model, args.revision, args.max_new_tokens)
    else:
        backend = TransformersBackend(
            args.model, args.revision, args.device, args.model_cache,
            args.max_new_tokens, args.max_input_tokens, args.local_files_only,
        )
    cache = ResponseCache(args.cache) if args.cache else None
    pipeline = ExtractionPipeline(backend, registry, args.chunk_chars, args.overlap_chars, args.max_retries, cache)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {"documents": 0, "document_errors": 0, "failed_chunks": 0,
               "events": 0, "events_needing_review": 0, "relationships": 0,
               "participant_links": 0, "quarantined_records": 0, "cache_hits": 0}
    seen_ids = set()
    input_hasher = hashlib.sha256()
    completed = False
    try:
        with ExitStack() as stack:
            outputs = {name: stack.enter_context((args.output / f"{name}.jsonl").open("x", encoding="utf-8"))
                       for name in ("documents", "events", "relationships", "event_company_links", "quarantine", "audit")}

            def write(name: str, record: dict) -> None:
                outputs[name].write(canonical_json(record) + "\n")

            with args.input.open("rb") as source:
                for line_number, line in enumerate(source, 1):
                    if args.limit and summary["documents"] + summary["document_errors"] >= args.limit:
                        break
                    input_hasher.update(line)
                    if not line.strip():
                        continue
                    try:
                        document = Document.from_dict(strict_json(line.decode("utf-8")))
                        if document.document_id in seen_ids:
                            raise ValueError(f"Duplicate document_id: {document.document_id}")
                        seen_ids.add(document.document_id)
                    except (ValueError, UnicodeError) as exc:
                        summary["document_errors"] += 1
                        summary["quarantined_records"] += 1
                        write("quarantine", {"kind": "input", "line": line_number, "error": str(exc)})
                        continue
                    result = pipeline.extract(document)
                    links = participant_links(result["events"])
                    write("documents", result["document"])
                    for event in result["events"]:
                        write("events", event)
                    for relationship in result["relationships"]:
                        write("relationships", relationship)
                    for link in links:
                        write("event_company_links", link)
                    for rejected in result["quarantine"]:
                        write("quarantine", rejected)
                    write("audit", {"document_id": document.document_id, "status": result["status"],
                                    "extraction": result["extraction"], "chunk_runs": result["chunk_runs"]})
                    summary["documents"] += 1
                    summary["events"] += len(result["events"])
                    summary["events_needing_review"] += sum(e["validation_status"] == "needs_review" for e in result["events"])
                    summary["relationships"] += len(result["relationships"])
                    summary["participant_links"] += len(links)
                    summary["quarantined_records"] += len(result["quarantine"])
                    summary["cache_hits"] += sum(r["cache_hit"] for r in result["chunk_runs"])
                    summary["failed_chunks"] += sum(r["error"] is not None for r in result["chunk_runs"])
                    print(f"{document.document_id}: {len(result['events'])} events, {len(result['relationships'])} relationships, {result['status']}", file=sys.stderr, flush=True)
            if summary["documents"] + summary["document_errors"] == 0:
                summary["document_errors"] += 1
                summary["quarantined_records"] += 1
                write("quarantine", {"kind": "input", "error": "Input contains no document records"})
            completed = True
    finally:
        if cache:
            cache.close()
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(), "extraction": pipeline.provenance,
            "input_path": str(args.input.resolve()), "processed_input_sha256": input_hasher.hexdigest(),
            "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest() if args.registry else None,
            "chunk_chars": args.chunk_chars, "overlap_chars": args.overlap_chars,
            "summary": summary, "is_accuracy_benchmark": False,
            "run_status": "aborted" if not completed else "completed_with_errors" if summary["document_errors"] or summary["failed_chunks"] else "completed",
        }
        (args.output / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(canonical_json(summary))
    return 2 if summary["document_errors"] or summary["failed_chunks"] else 0
