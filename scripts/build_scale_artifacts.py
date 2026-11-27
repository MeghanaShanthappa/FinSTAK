"""Merge disclosure evidence, preserve verification levels, embed and pack."""
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path
import re

from stak_graph.io import read_jsonl
from stak_graph.experiments.scale_io import write_json, write_jsonl
from stak_graph.embeddings.backend import EmbeddingGemmaBackend
from stak_graph.embeddings.finbert import FinBERTBackend
from stak_graph.embeddings.pipeline import VectorCache, embed_events
from stak_graph.experiments.scale_panel import build_panel, build_event_arrays
from stak_graph.experiments.data import sha256_file
from stak_graph.acquisition.sec_items import events_from_filing

ROOT = Path('/tmp/finstak-100x10-v1')


def merge():
    output = ROOT / 'expanded'
    output.mkdir(exist_ok=True)
    news = list(read_jsonl(ROOT / 'dataset/events.jsonl'))
    sec_path = ROOT / 'sec/sec_events.jsonl'
    sec = list(read_jsonl(sec_path)) if sec_path.exists() else []
    filings = ROOT / 'sec/sec_filings.jsonl'
    item_events = [event for filing in read_jsonl(filings) for event in events_from_filing(filing)] if filings.exists() else []
    write_jsonl(ROOT / 'sec/sec_item_events.jsonl', item_events)
    sec.extend(item_events)
    config = json.loads((ROOT / 'dataset/config.json').read_text())
    sec = [e for e in sec if config['sample_start'] <= e['known_at'][:10] <= config['sample_end']]
    for event in news:
        # A quoted headline-leading entity is an automated subject check, not
        # externally certified issuer identity. Keep the verification distinction.
        quote = event['companies'][0]['evidence']
        prefix = re.sub(r'^(?:UPDATE\s*\d*\s*[-:]|BRIEF\s*[-:])\s*', '', event['description'], flags=re.I)
        # Require an adjacent asserted issuer action, not merely a leading
        # company mention in a market-commentary sentence.
        action = r'\s*(?:\([^)]*\))?\s*[,\-:]?\s*(?:reports?|reported|posts?|posted|announces?|announced|declares?|declared|raises?|raised|cuts?|lowers?|acquires?|acquired|agrees?|completes?|issues?|prices?|sues|settles?|recalls?|halts?|suspends?)\b'
        event['primary_discloser_verified'] = bool(re.match(re.escape(quote) + action, prefix, re.I))
        event['primary_discloser_verification_level'] = 'quoted_leading_company_with_adjacent_asserted_action'
    for event in sec:
        event['primary_discloser_verified'] = True
        event['primary_discloser_verification_level'] = 'SEC_CIK_accession_reporting_company_join'
        event.setdefault('verification_level', 'structured_SEC_concept_accession_period_and_identifier_join')
    # Filing and earnings announcement can describe the same quarter; retain
    # both source observations but assign one conservative economic cluster.
    # They remain available at their own later known_at; never move a filing
    # back to an earlier announcement time.
    duplicates = []
    eligible_sec = []
    news_earnings = defaultdict(list)
    kept_sec_earnings = defaultdict(list)
    for row in news:
        if row['category'] == 'earnings':
            news_earnings[row['companies'][0]['company_id']].append(
                (datetime.fromisoformat(row['known_at'].replace('Z', '+00:00')), row['event_id']))
    for event in sorted(sec, key=lambda e: (e['known_at'], e['event_id'])):
        ticker = event['companies'][0]['company_id']
        known = datetime.fromisoformat(event['known_at'].replace('Z', '+00:00'))
        prior = [identifier for stamp, identifier in news_earnings[ticker] if 0 <= (known - stamp).days <= 30]
        prior_sec = [identifier for stamp, identifier in kept_sec_earnings[ticker] if 0 <= (known - stamp).days <= 45]
        if event['category'] == 'earnings' and (prior or prior_sec):
            duplicates.append({'sec_event_id': event['event_id'], 'possible_same_earnings_event': (prior or prior_sec)[-1],
                               'rule': 'same-company news earnings in prior 30 days or retained official disclosure in prior 45 days; later observation excluded'})
        else:
            eligible_sec.append(event)
            if event['category'] == 'earnings':
                kept_sec_earnings[ticker].append((known, event['event_id']))
    events = sorted(news + eligible_sec, key=lambda e: (e['known_at'], e['event_id']))
    if len(events) > 50000:
        # Fixed identifier ranking is label-blind and reproducible. Preserve
        # official disclosures, then cap news observations without replacement.
        news_budget = max(0, 50000 - len(eligible_sec))
        chosen = sorted(news, key=lambda e: e['event_id'])[:news_budget]
        events = sorted(chosen + eligible_sec, key=lambda e: (e['known_at'], e['event_id']))
    if len({e['event_id'] for e in events}) != len(events):
        raise ValueError('Duplicate event identifier')
    write_json(output / 'preparation_audit.json', {'events': len(events), 'minimum': 25000, 'maximum': 50000,
               'count_gate_passed': 25000 <= len(events) <= 50000, 'news_candidates': len(news),
               'SEC_item_candidates': len(item_events), 'SEC_retained': len(eligible_sec),
               'SEC_possible_duplicates_excluded': len(duplicates)})
    if not 25000 <= len(events) <= 50000:
        raise ValueError(f'Event count gate failed: {len(events)} after source checks and deduplication; do not call this the requested full study')
    write_jsonl(output / 'events.jsonl', events)
    write_jsonl(output / 'sec_duplicate_audit.jsonl', duplicates)
    write_json(output / 'manifest.json', {'events': len(events), 'news_observations': len(news),
               'SEC_observations_retained': len(eligible_sec), 'SEC_possible_duplicates_excluded': len(duplicates),
               'human_reviewed_count': 0, 'verification': 'Automated source evidence checks, not human semantic certification',
               'news_audit_sha256': sha256_file(ROOT / 'dataset/event_audit.json')})
    return output, events


def main():
    expanded, events = merge()
    import subprocess, sys
    subprocess.run([sys.executable, str(Path(__file__).with_name('verify_scale_sources.py'))], check=True)
    import torch
    torch.set_num_threads(4)
    for encoder in ('embeddinggemma', 'finbert'):
        directory = 'embeddings' if encoder == 'embeddinggemma' else 'finbert_embeddings'
        if (expanded / directory / 'manifest.json').exists():
            continue
        if encoder == 'embeddinggemma':
            backend = EmbeddingGemmaBackend(revision='f9972ac22e696a105c482202646979e33e4b5a1e',
                                           cache_dir='/tmp/stak-graph-models', local_files_only=True)
        else:
            revision = json.loads((ROOT / 'finbert_checkpoint.json').read_text())['revision']
            backend = FinBERTBackend(revision, local_files_only=True)
        cache = VectorCache(ROOT / 'vectors.sqlite3')
        try:
            manifest = embed_events(events, backend, expanded / directory, cache=cache, batch_size=64)
        finally:
            cache.close()
        print(json.dumps({'encoder': encoder, 'embedding_complete': manifest}), flush=True)
        del backend
        torch.cuda.empty_cache()
    packed = ROOT / 'packed'
    if not (packed / 'panel.npz').exists():
        build_panel(ROOT / 'dataset', packed)
    build_event_arrays(ROOT / 'dataset', expanded, packed)
    build_event_arrays(ROOT / 'dataset', expanded, packed, 'finbert_embeddings', 'event_arrays_finbert.npz')
    print(json.dumps({'packed_complete': True}), flush=True)


if __name__ == '__main__':
    main()
