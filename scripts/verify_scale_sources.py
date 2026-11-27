"""Independently rejoin every retained news assertion to its pinned raw row."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from stak_graph.io import read_jsonl
from stak_graph.experiments.scale_io import write_json
from stak_graph.experiments.scale_data import source_time, asserted_categories
from stak_graph.acquisition.public import known_at as sec_known_at

ROOT = Path('/tmp/finstak-100x10-v1')


def main():
    wanted = defaultdict(lambda: defaultdict(list))
    events = list(read_jsonl(ROOT / 'expanded/events.jsonl'))
    official = []
    for event in events:
        source = event.get('provenance')
        if source:
            wanted[source['archive']][source['source_row']].append(event)
        else:
            official.append(event)
    csv.field_size_limit(64 * 1024 * 1024)
    checked, archived = 0, {}
    for name, rows in wanted.items():
        matched = set()
        with (Path('/tmp/stak-graph-real-data/raw') / name).open(newline='', encoding='utf-8-sig') as source:
            for number, row in enumerate(csv.DictReader(source), 2):
                if number not in rows:
                    continue
                row_hash = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                for event in rows[number]:
                    provenance = event['provenance']
                    if provenance['source_row_sha256'] != row_hash:
                        raise ValueError('Source row hash mismatch')
                    published, known = source_time(row['Date'])
                    if event['known_at'] != known or event['published_at'] != published:
                        raise ValueError('Historical source timestamp/delay mismatch')
                    span = provenance['evidence_span']
                    source_field = row[span['field']]
                    if span['field'] == 'Article_title':
                        # Headline ingestion strips only outside whitespace;
                        # recorded headline spans use that canonical field.
                        source_field = source_field.strip()
                    text = source_field[span['start']:span['end']]
                    if text != event['description'] or hashlib.sha256(text.encode()).hexdigest() != event['source_hash']:
                        raise ValueError('Evidence quotation or source hash mismatch')
                    company = event['companies'][0]
                    if company['company_id'] != row['Stock_symbol'] or company['evidence'] not in text:
                        raise ValueError('Company/source attribution mismatch')
                    if event['category'] not in asserted_categories(text):
                        raise ValueError('Category is not supported by the recorded assertion rule')
                    if event['source_url'] != row['Url'].strip():
                        raise ValueError('Source URL mismatch')
                    checked += 1
                matched.add(number)
                if len(matched) == len(rows):
                    break
        if len(matched) != len(rows):
            raise ValueError('Retained source row is absent from pinned archive')
        archived[name] = {'raw_rows_rejoined': len(matched), 'assertions_rejoined': sum(map(len, rows.values()))}
        print(json.dumps({'archive_verified': name, **archived[name]}), flush=True)
    # Structured disclosures retain the actual accession/concept or item.
    official_rows = {r['accessionNumber']: r for r in read_jsonl(ROOT / 'sec/sec_filings.jsonl')}
    verified_official = 0
    fact_cache = {}
    for event in official:
        if event.get('source_type') == 'SEC_declared_form_item_metadata':
            quoted = json.loads(event['evidence'][0])
            actual = official_rows.get(quoted['accessionNumber'])
            if actual != quoted:
                raise ValueError('SEC item quotation does not match acquired filing metadata')
            if not all(item in str(actual.get('items', '')) for item in event['reported_items']):
                raise ValueError('SEC form item is absent from the acquired metadata')
        elif 'financial_observation' in event:
            observation = event['financial_observation']
            ticker = observation['ticker']
            if ticker not in fact_cache:
                fact_cache[ticker] = json.loads((ROOT / 'sec/raw' / f'{ticker}_companyfacts.json').read_text())
            namespace, concept = observation['concept'].split(':')
            facts = fact_cache[ticker]['facts'][namespace][concept]['units'][observation['unit']]
            matches = [r for r in facts if r.get('accn') == observation['accession'] and
                       r.get('start') == observation['period_start'] and r.get('end') == observation['period_end']
                       and r.get('val') == observation['value']]
            if not matches:
                raise ValueError('Quarterly financial observation does not match the acquired SEC facts')
            filing = official_rows.get(observation['accession'])
            if not filing or event['known_at'] != sec_known_at(filing):
                raise ValueError('Quarterly event availability does not match the original filing date')
        else:
            raise ValueError('Official event lacks a retained structured observation')
        verified_official += 1
    write_json(ROOT / 'source_verification.json', {'passed': True, 'news_assertions_rejoined': checked,
               'official_events_checked': verified_official, 'archives': archived, 'total_events': len(events),
               'human_reviewed': False, 'scope': 'Independent source attribution/quotation/timestamp/category-rule checks, not verification of news truth or observed delivery time.'})


if __name__ == '__main__':
    main()
