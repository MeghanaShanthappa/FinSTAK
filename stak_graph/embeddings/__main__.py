import argparse
import sys

from stak_graph.io import read_jsonl
from stak_graph.extraction.schema import canonical_json
from .backend import EmbeddingGemmaBackend, TestHashBackend
from .pipeline import VectorCache, embed_events


def main():
    parser = argparse.ArgumentParser(description="Encode validated STAK-Graph events")
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["gemma2", "test-hash"], default="gemma2")
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-cache", default="/tmp/stak-graph-models")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    cache = None
    try:
        backend = TestHashBackend() if args.backend == "test-hash" else EmbeddingGemmaBackend(
            args.model, args.revision, args.device, args.model_cache, args.local_files_only)
        cache = VectorCache(args.cache) if args.cache else None
        print(canonical_json(embed_events(read_jsonl(args.events), backend, args.output, cache, args.batch_size)))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Embedding error: {exc}", file=sys.stderr)
        return 2
    finally:
        if cache:
            cache.close()


raise SystemExit(main())
