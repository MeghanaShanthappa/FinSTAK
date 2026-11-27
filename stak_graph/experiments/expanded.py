"""Create the expanded source dataset and real semantic event embeddings."""
import argparse
import json
from pathlib import Path

from stak_graph.io import read_jsonl, write_json, write_jsonl
from stak_graph.embeddings.backend import EmbeddingGemmaBackend
from stak_graph.embeddings.pipeline import load_vectors, embed_events, VectorCache
from .data import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--knowledge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-releases', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = list(read_jsonl(args.source / 'event_projection/events.jsonl'))
    sec = list(read_jsonl(args.knowledge / 'sec_events.jsonl'))
    releases = list(read_jsonl(args.knowledge / 'sec_release_events.jsonl')) if args.include_releases else []
    events = sorted(original + sec + releases, key=lambda row: (row['known_at'], row['event_id']))
    if len({row['event_id'] for row in events}) != len(events):
        raise ValueError('Duplicate event IDs')
    write_jsonl(args.output / 'events.jsonl', events)
    vectors, manifest = load_vectors(args.source / 'event_embeddings')
    cache = VectorCache(args.output / 'vectors.sqlite3')
    try:
        for row in vectors.values():
            cache.put(row['cache_key'], row['vector'])
        encoder = EmbeddingGemmaBackend(revision=manifest['encoder']['revision'],
                                       cache_dir='/tmp/stak-graph-models', local_files_only=True)
        embedding = embed_events(events, encoder, args.output / 'embeddings', cache=cache, batch_size=16)
    finally:
        cache.close()
    write_json(args.output / 'manifest.json', {
        'original_headline_event_records': len(original), 'SEC_disclosure_records': len(sec),
        'SEC_release_evidence_records': len(releases),
        'combined_source_observations': len(events),
        'sources': {str(args.source / 'event_projection/events.jsonl'): sha256_file(args.source / 'event_projection/events.jsonl'),
                    str(args.knowledge / 'sec_events.jsonl'): sha256_file(args.knowledge / 'sec_events.jsonl'),
                    **({str(args.knowledge / 'sec_release_events.jsonl'): sha256_file(args.knowledge / 'sec_release_events.jsonl')} if releases else {})},
        'embedding': embedding,
        'GDELT_metadata_used_as_events': False, 'current_Wikidata_used_in_historical_graph': False,
        'limitations': ['Headline earnings releases and SEC disclosures are distinct source observations and can describe the same economic quarter.',
                       'SEC knowledge begins at end of filing date; it is not backdated to earnings announcement or financial period end.',
                       'Existing 2019 test has been inspected in previous studies; results remain exploratory.']})
    print(json.dumps({'events': len(events), 'SEC': len(sec), 'cached_embeddings': embedding['cache_hits']}))


if __name__ == '__main__':
    main()
