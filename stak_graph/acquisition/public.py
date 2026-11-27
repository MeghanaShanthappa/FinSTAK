"""Acquire SEC, OpenFIGI, GDELT and Wikidata evidence without backdating snapshots.

SEC_CONTACT_EMAIL is used only in HTTP headers, never serialized. Raw responses
are immutable cached files. GDELT records are discovery metadata, not extracted
financial events; Wikidata's current snapshot is excluded from historical fits.
"""
import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from stak_graph.extraction.schema import digest


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)


def write_jsonl(path, rows):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    temporary.replace(path)


class PublicClient:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.last_request = 0.0

    def get(self, name, url, payload=None, sec=False):
        path = self.root / name
        metadata = path.with_suffix(path.suffix + '.source.json')
        if path.exists() and metadata.exists():
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != json.loads(metadata.read_text())['sha256']:
                raise ValueError('Cached source checksum mismatch: ' + name)
            return raw
        headers = {'User-Agent': 'STAK-Graph academic research', 'Accept-Encoding': 'identity'}
        if sec:
            contact = os.environ.get('SEC_CONTACT_EMAIL', '')
            if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', contact):
                raise ValueError('SEC_CONTACT_EMAIL must contain the authorized contact')
            headers['User-Agent'] += ' ' + contact
        data = None if payload is None else json.dumps(payload).encode()
        if data is not None:
            headers['Content-Type'] = 'application/json'
        for attempt in range(4):
            time.sleep(max(0, .55 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=70) as response:
                    raw = response.read()
                path.write_bytes(raw)
                write_json(metadata, {'url': url, 'fetched_at': now(), 'sha256': hashlib.sha256(raw).hexdigest(),
                                      'bytes': len(raw), 'request_payload': payload})
                return raw
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError('HTTP retries exhausted')

    def json(self, *args, **kwargs):
        return json.loads(self.get(*args, **kwargs))


def filing_rows(columns):
    count = len(columns.get('accessionNumber', []))
    return [{key: values[index] for key, values in columns.items() if isinstance(values, list) and len(values) == count}
            for index in range(count)]


def known_at(filing):
    """Daily conservative public-availability policy, including late acceptance."""
    latest_date = filing['filingDate']
    if filing.get('acceptanceDateTime'):
        accepted = datetime.fromisoformat(filing['acceptanceDateTime'].replace('Z', '+00:00'))
        if accepted.tzinfo is None:
            raise ValueError('SEC acceptance time must have an explicit timezone')
        latest_date = max(latest_date, accepted.astimezone(timezone.utc).date().isoformat())
    return latest_date + 'T23:59:59Z'


def quarterly_records(ticker, cik, facts, filings, years):
    lookup = {row['accessionNumber']: row for row in filings}
    candidates = {}
    gaap = facts.get('facts', {}).get('us-gaap', {})
    # EPS is not additive: Q4 is accepted only when directly reported for a
    # 60–120 day duration. Annual EPS minus 9-month EPS is never used.
    for unit, rows in gaap.get('EarningsPerShareDiluted', {}).get('units', {}).items():
        for fact in rows:
            filing = lookup.get(fact.get('accn'))
            if not filing or filing.get('form') not in ('10-Q', '10-K') or fact.get('fy') not in years:
                continue
            if fact.get('end') != filing.get('reportDate') or not fact.get('start'):
                continue
            days = (date.fromisoformat(fact['end']) - date.fromisoformat(fact['start'])).days + 1
            if not 60 <= days <= 120:
                continue
            quarter = fact.get('fp') if filing['form'] == '10-Q' else 'Q4'
            if quarter not in ('Q1', 'Q2', 'Q3', 'Q4'):
                continue
            key = (fact['fy'], quarter)
            candidates.setdefault(key, []).append((filing, fact, unit))
    coverage, events = [], []
    for year in years:
        for quarter in ('Q1', 'Q2', 'Q3', 'Q4'):
            options = candidates.get((year, quarter), [])
            if not options:
                coverage.append({'ticker': ticker, 'fiscal_year': year, 'fiscal_quarter': quarter,
                                 'status': 'missing_direct_quarterly_eps'})
                continue
            filing, fact, unit = min(options, key=lambda value: (known_at(value[0]), value[0]['accessionNumber']))
            accession = filing['accessionNumber']
            url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace("-", "")}/{filing["primaryDocument"]}'
            source = {'ticker': ticker, 'cik': cik, 'fiscal_year': year, 'fiscal_quarter': quarter,
                      'period_start': fact['start'], 'period_end': fact['end'], 'concept': 'us-gaap:EarningsPerShareDiluted',
                      'value': fact['val'], 'unit': unit, 'accession': accession, 'form': filing['form'],
                      'filing_date': filing['filingDate'], 'acceptance_at': filing.get('acceptanceDateTime'),
                      'known_at': known_at(filing), 'source_url': url,
                      'availability_policy': 'end_of_later_SEC_filing_or_acceptance_UTC_date', 'actual_earnings_release_time_verified': False,
                      'status': 'directly_reported_quarterly_eps', 'is_derived': False}
            coverage.append(source)
            text = (f'{ticker} fiscal {year} {quarter} quarterly financial disclosure: '
                    f'diluted earnings per share {fact["val"]} {unit}; '
                    f'period {fact["start"]} to {fact["end"]}; SEC {filing["form"]} filed {filing["filingDate"]}.')
            document_id = 'sec_' + accession
            source_hash = digest(source)
            events.append({'event_id': 'secq_' + digest([ticker, year, quarter])[:24],
                           'category': 'earnings', 'description': text, 'title': text,
                           'document_id': document_id, 'source_hash': source_hash,
                           'source_url': url, 'source_type': 'sec_structured_quarterly_disclosure',
                           'known_at': known_at(filing), 'published_at': known_at(filing),
                           'companies': [{'name': ticker, 'company_id': ticker, 'resolution_status': 'resolved',
                                          'role': 'reporting_company', 'evidence': ticker}],
                           'amounts': [], 'financial_observation': source,
                           'expected_duration': None, 'confidence': None, 'calibration_status': 'uncalibrated',
                           'valid_until': None, 'expected_direction': None, 'is_synthetic': False,
                           'validation_status': 'evidence_checked',
                           'validation_level': 'structured_SEC_concept_accession_period_and_identifier_join',
                           'human_reviewed': False, 'semantic_support_verified': True,
                           'evidence': [text], 'relationship_basis': 'directly_reported'})
    return coverage, events


def sec_sources(client, config, output):
    mapping = client.json('sec_tickers.json', 'https://www.sec.gov/files/company_tickers_exchange.json', sec=True)
    identifiers, filings_all, coverage, events = [], [], [], []
    for company in config['companies']:
        ticker = company['company_id']
        matches = [dict(zip(mapping['fields'], row)) for row in mapping['data'] if row[2] == ticker]
        override = config.get('historical_sec_identifiers', {}).get(ticker)
        if override:
            if not override.get('source_url', '').startswith('https://www.sec.gov/Archives/edgar/data/'):
                raise ValueError('Historical SEC identifier requires filing evidence')
            matches = [{'cik': int(override['cik']), 'name': company['name'], 'ticker': ticker,
                        'exchange': 'historical_US_listing', 'historical_identity_evidence': override['source_url']}]
        if len(matches) != 1:
            raise ValueError(f'SEC ticker mapping ambiguous or absent: {ticker}')
        identity = matches[0]
        cik = str(identity['cik']).zfill(10)
        identity.update({'company_id': ticker, 'cik': cik, 'snapshot_known_at': now(),
                         'historical_identifier_validity_verified': False})
        identifiers.append(identity)
        submission = client.json(f'{ticker}_submissions.json', f'https://data.sec.gov/submissions/CIK{cik}.json', sec=True)
        rows = filing_rows(submission['filings']['recent'])
        for history in submission['filings'].get('files', []):
            if history['filingTo'] < config['sample_start'] or history['filingFrom'] > config['sample_end']:
                continue
            older = client.json(f'{ticker}_{history["name"]}', 'https://data.sec.gov/submissions/' + history['name'], sec=True)
            rows.extend(filing_rows(older))
        rows = list({row['accessionNumber']: row for row in rows}.values())
        for row in rows:
            if config['sample_start'] <= row['filingDate'] <= config['sample_end'] and row['form'] in ('8-K', '10-Q', '10-K', '6-K'):
                filings_all.append({**row, 'ticker': ticker, 'cik': cik, 'known_at': known_at(row),
                                    'source_kind': 'filing_metadata', 'full_document_extracted': False})
        facts = client.json(f'{ticker}_companyfacts.json', f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json', sec=True)
        years = list(range(int(config['sample_start'][:4]), int(config['sample_end'][:4]) + 1))
        quarters, quarter_events = quarterly_records(ticker, cik, facts, rows, years)
        coverage.extend(quarters)
        events.extend(quarter_events)
        print(json.dumps({'source': 'SEC', 'ticker': ticker, 'quarterly_disclosures': len(quarter_events),
                          'target_quarters': 4 * len(years)}), flush=True)
    write_jsonl(output / 'identifiers.jsonl', identifiers)
    write_jsonl(output / 'sec_filings.jsonl', filings_all)
    write_jsonl(output / 'quarterly_coverage.jsonl', coverage)
    write_jsonl(output / 'sec_events.jsonl', events)
    return identifiers, {'filing_metadata': len(filings_all), 'quarterly_events': len(events),
                         'target_quarters': len(coverage), 'missing_quarters': len(coverage) - len(events)}


def figi_sources(client, identifiers, output):
    records = []
    for offset in range(0, len(identifiers), 10):
        group = identifiers[offset:offset + 10]
        payload = [{'idType': 'TICKER', 'idValue': row['company_id'], 'exchCode': 'US', 'marketSecDes': 'Equity'} for row in group]
        response = client.json(f'openfigi_{offset}.json', 'https://api.openfigi.com/v3/mapping', payload)
        for company, result in zip(group, response):
            records.append({'ticker': company['company_id'], 'cik': company['cik'], 'matches': result.get('data', []),
                            'error': result.get('error'), 'known_at': now(), 'historical_validity_verified': False})
        time.sleep(2.5)
    write_jsonl(output / 'openfigi.jsonl', records)
    return {'mapped_tickers': sum(bool(row['matches']) for row in records), 'historical_security_master': False}


def wikidata_sources(client, identifiers, output):
    entities, relations, failures = [], [], []
    for company in identifiers:
        ticker = company['company_id']
        try:
            query = f'SELECT ?item WHERE {{ ?item wdt:P5531 "{company["cik"]}". }}'
            url = 'https://query.wikidata.org/sparql?' + urllib.parse.urlencode({'query': query, 'format': 'json'})
            result = client.json(f'{ticker}_wikidata_lookup.json', url)
            matches = result['results']['bindings']
            if len(matches) != 1:
                failures.append({'ticker': ticker, 'reason': 'CIK query not unique', 'matches': len(matches)})
                continue
            qid = matches[0]['item']['value'].rsplit('/', 1)[-1]
            raw = client.json(f'{qid}.json', f'https://www.wikidata.org/wiki/Special:EntityData/{qid}.json')
            entity = raw['entities'][qid]
            entities.append({'ticker': ticker, 'qid': qid, 'revision': entity['lastrevid'], 'known_at': now()})
            for prop, kind in [('P749', 'parent'), ('P355', 'subsidiary'), ('P452', 'industry')]:
                for claim in entity.get('claims', {}).get(prop, []):
                    value = claim.get('mainsnak', {}).get('datavalue', {}).get('value', {})
                    if not isinstance(value, dict) or 'id' not in value:
                        continue
                    relations.append({'subject_ticker': ticker, 'subject_qid': qid, 'relation': kind,
                                      'object_qid': value['id'], 'claim_id': claim.get('id'),
                                      'qualifiers': claim.get('qualifiers', {}), 'references': claim.get('references', []),
                                      'known_at': now(), 'historical_training_eligible': False,
                                      'reason': 'current snapshot; historical availability not established'})
        except Exception as error:
            failures.append({'ticker': ticker, 'reason': type(error).__name__})
    write_jsonl(output / 'wikidata_entities.jsonl', entities)
    write_jsonl(output / 'wikidata_relations.jsonl', relations)
    write_jsonl(output / 'wikidata_failures.jsonl', failures)
    return {'entities': len(entities), 'relations': len(relations), 'failures': failures,
            'used_in_historical_forecasting': False}


def parse_gkg(raw, aliases, archive_url):
    """GKG 2.1 fields: date, document URL, themes, organizations; no article text."""
    records = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            with archive.open(name) as binary:
                for fields in csv.reader(io.TextIOWrapper(binary, encoding='utf-8', errors='replace'), delimiter='\t'):
                    if len(fields) < 15:
                        continue
                    organizations = fields[13] + ';' + fields[14]
                    folded = organizations.casefold()
                    tickers = sorted(ticker for ticker, names in aliases.items() if any(re.search(r'(?<!\w)' + re.escape(alias.casefold()) + r'(?!\w)', folded) for alias in names))
                    if not tickers:
                        continue
                    stamp = datetime.strptime(fields[1][:14], '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
                    records.append({'document_id': 'gdelt_' + digest(fields[0])[:24], 'source_url': fields[4],
                                    'source_archive': archive_url, 'first_seen_at': stamp.isoformat().replace('+00:00', 'Z'),
                                    'organizations': fields[13], 'themes': fields[7], 'candidate_tickers': tickers,
                                    'entity_links_verified': False, 'article_text_acquired': False,
                                    'record_kind': 'news_discovery_metadata', 'accepted_financial_event': False})
    return records


def gdelt_sources(client, config, output, archives):
    aliases = {row['company_id']: [alias for alias in row['aliases'] if len(alias) > 3 and alias != row['company_id']]
               for row in config['companies']}
    records, failures = [], []
    # Deterministic coverage sample, not a claim of complete historical news.
    for index in range(archives):
        year, month = 2016 + index // 12, index % 12 + 1
        stamp = f'{year}{month:02}15150000'
        url = f'https://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip'
        try:
            raw = client.get(stamp + '.gkg.csv.zip', url)
            found = parse_gkg(raw, aliases, url)
            records.extend(found)
            print(json.dumps({'source': 'GDELT', 'archive': stamp, 'candidate_news': len(found)}), flush=True)
        except Exception as error:
            failures.append({'archive': stamp, 'reason': type(error).__name__})
    records = list({row['source_url']: row for row in records}.values())
    write_jsonl(output / 'gdelt_news.jsonl', records)
    return {'news_metadata_records': len(records), 'archives_requested': archives, 'failures': failures,
            'complete_historical_coverage': False, 'accepted_events': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gdelt-archives', type=int, default=48)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    client = PublicClient(args.output / 'raw')
    identifiers, sec = sec_sources(client, config, args.output)
    inventory = {'created_at': now(), 'SEC': sec, 'OpenFIGI': figi_sources(client, identifiers, args.output)}
    write_json(args.output / 'inventory.json', inventory)
    inventory['GDELT'] = gdelt_sources(client, config, args.output, args.gdelt_archives)
    write_json(args.output / 'inventory.json', inventory)
    inventory['Wikidata'] = wikidata_sources(client, identifiers, args.output)
    write_json(args.output / 'inventory.json', inventory)
    print(json.dumps(inventory), flush=True)


if __name__ == '__main__':
    main()
