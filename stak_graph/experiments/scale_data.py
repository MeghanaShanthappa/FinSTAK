"""Pinned 100-company study acquisition, conservative evidence checks and audit.

Automated verification means reproducible source/quote/entity/category checks.
It does NOT mean human review, independently verified news truth, or observed
historical delivery timestamps. Those limitations are explicit in every export.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import unicodedata
import zipfile

from .scale_io import write_json, write_jsonl
from .data import sha256_file, write_csv, ARTIFACTS

NEWS_ARTIFACTS = {
    'All_external.csv': ARTIFACTS['All_external.csv'][1:],
    'nasdaq_exteral_data.csv': (23232979597, '1a7a3eb8e6b97ec19f286f2cfca3371542bddb272ab1eb8f36e33ad98fa5c4da'),
}
FIELDS = ('open', 'high', 'low', 'close', 'volume')
# Deliberately narrow asserted-event surfaces. Do not infer amounts, surprise,
# direction, litigation outcome or causal effect from these headline checks.
RULES = {
    'earnings': r'\b(?:reports?|posts?|reported|posted|announces?|released?)\b.{0,85}\b(?:earnings|results|profit|loss|eps|revenue)\b|\bq[1-4]\b.{0,35}\b(?:earnings|results|eps)\b.{0,40}\b(?:beat|beats|miss|misses|rose|rise|fall|fell)\b',
    'guidance_change': r'\b(?:raises?|raised|cuts?|lowers?|lowered|revises?|revised|updates?|withdraws?|boosts?)\b.{0,55}\b(?:guidance|outlook|forecast)\b',
    'acquisition': r'\b(?:acquires?|acquired|agrees? to (?:buy|acquire)|to acquire|completes? acquisition)\b|\b(?:announces?|announced|completes?|completed|approves?|approved|signs?|signed|closes?|closed)\b.{0,60}\b(?:merger|buyout|takeover)\b',
    'financing': r'\b(?:announces?|prices?|priced|issues?|issued|completes?|launches?|secures?)\b.{0,70}\b(?:notes|bonds|offering|financing|credit facility|loan|debt)\b',
    'litigation': r'\b(?:sues|sued|settles?|settled|fined|indicted|charged|lawsuit filed|files? (?:a )?lawsuit|court orders?|jury awards?)\b',
    'operational_disruption': r'\b(?:recalls?|recalled|halts?|halted|shutdown|shuts? down|outage|plant closure|suspends?|suspended|workers strike)\b',
    'dividend': r'\b(?:declares|declared|announces|announced)\b.{0,55}\bdividends?\b',
}
SCALE_CATEGORIES = tuple(RULES)
EXCLUDE = re.compile(r'\?|\b(?:may|might|could|rumou?r|speculat\w*|preview|what to expect|earnings call transcript|should you|analyst\w* expect|price target|initiates? coverage)\b', re.I)


def normalized_title(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(re.findall(r'[a-z0-9]+', text))


def asserted_categories(title, allowed_categories=None):
    if not title.strip() or EXCLUDE.search(title):
        return []
    result = [category for category, pattern in RULES.items()
              if (allowed_categories is None or category in allowed_categories) and re.search(pattern, title, re.I)]
    if 'dividend' in result and re.search(r'\b(?:expected to|expects? to|likely to|would)\b', title, re.I):
        result.remove('dividend')
    return result


def company_quote(title, company):
    # Bare short ticker symbols are not entity evidence (A, C, F, T, etc.).
    aliases = sorted(set([company['name'], *company['aliases']]), key=len, reverse=True)
    for alias in aliases:
        match = re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', title, re.I)
        if match:
            return match.group()
    match = re.search(r'\(' + re.escape(company['company_id']) + r'\)', title)
    return match.group() if match else None


def source_time(stamp):
    published = datetime.strptime(stamp.strip(), '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    # Two days after reported time, not midnight before publication.
    known = published + timedelta(days=2)
    return published.isoformat().replace('+00:00', 'Z'), known.isoformat().replace('+00:00', 'Z')


def supported_assertions(title, body, company, allowed_categories=None):
    """Use a quoted headline or a named asserted sentence, not an article label.

    A headline may be commentary while the original body contains a concrete
    corporate disclosure. Preserve the exact sentence and its character span.
    At most one assertion per category per article is retained.
    """
    seen = set()
    texts = [('Article_title', 0, title)]
    # Sentence boundaries are deterministic; no generated paraphrase is used.
    boundary = re.compile(r'(?<!\bInc\.)(?<!\bCorp\.)(?<!\bCo\.)(?<!\bLtd\.)(?<=[.!?])\s+|\n+', re.I)
    begin = 0
    for end, following in [(m.start(), m.end()) for m in boundary.finditer(body)] + [(len(body), len(body))]:
        segment = body[begin:end]
        text = segment.strip()
        if 30 <= len(text) <= 700:
            start = begin + len(segment) - len(segment.lstrip())
            texts.append(('Article', start, text))
        begin = following
    for field, start, text in texts:
        categories = asserted_categories(text, allowed_categories)
        quote = company_quote(text, company) if categories else None
        if not quote:
            continue
        for category in categories:
            if category not in seen:
                seen.add(category)
                yield category, text, quote, {'field': field, 'start': start, 'end': start + len(text)}


def audit_universe(raw, config, output):
    """Freeze selection using pre-2014 coverage only; report later gaps separately."""
    from collections import defaultdict
    selected, audit = [], []
    per_sector = defaultdict(int)
    with zipfile.ZipFile(raw / 'full_history.zip') as archive:
        members = set(archive.namelist())
        for company in config['candidate_companies']:
            ticker, sector = company['company_id'], company['sector']
            member = f'full_history/{ticker}.csv'
            if member not in members:
                audit.append({'ticker': ticker, 'status': 'absent_from_pinned_archive'})
                continue
            with archive.open(member) as handle:
                rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8-sig')))
            prefix = [r for r in rows if config['calendar_start'] <= r['date'] < config['sample_start']]
            eligible = len(prefix) >= 50
            take = eligible and per_sector[sector] < config['sector_quotas'][sector]
            if take:
                selected.append(company)
                per_sector[sector] += 1
            observed = [r['date'] for r in rows if config['sample_start'] <= r['date'] <= config['price_end']]
            audit.append({'ticker': ticker, 'sector': sector, 'pre_study_observations': len(prefix),
                          'selected': take, 'study_observations': len(observed),
                          'study_first': min(observed, default=None), 'study_last': max(observed, default=None),
                          'future_coverage_used_for_selection': False})
    write_json(output / 'universe_audit.json', {'candidates': audit, 'selected_count': len(selected),
               'sector_counts': dict(per_sector), 'limitations': [config['universe_policy']]})
    if len(selected) != 100:
        raise ValueError(f'Prespecified eligibility/sector quotas yielded {len(selected)}, not 100; see audit')
    frozen = dict(config, companies=selected)
    write_json(output / 'config.json', frozen)
    return frozen


def prepare_prices(raw, config, output):
    import exchange_calendars as xcals
    import pandas as pd
    calendar = xcals.get_calendar('XNYS', start=config['calendar_start'], end=config['price_end'])
    sessions = [{'session_date': stamp.date().isoformat(), 'open_at': row['open'].isoformat(),
                 'close_at': row['close'].isoformat(), 'prediction_at': stamp.date().isoformat() + 'T23:59:59Z',
                 'is_synthetic': 'false'} for stamp, row in calendar.schedule.iterrows()]
    by_date = {r['session_date']: r for r in sessions}
    rows, rejected, coverage = [], [], {}
    with zipfile.ZipFile(raw / 'full_history.zip') as archive:
        for company in config['companies']:
            ticker = company['company_id']
            dates = set()
            with archive.open(f'full_history/{ticker}.csv') as handle:
                for source in csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8-sig')):
                    day = source['date']
                    if not config['calendar_start'] <= day <= config['price_end']:
                        continue
                    try:
                        if day not in by_date or day in dates:
                            raise ValueError('non_session_or_duplicate')
                        factor = float(source['adj close']) / float(source['close'])
                        values = [float(source[k]) * factor for k in FIELDS[:4]] + [float(source['volume'])]
                        o, h, l, c, v = values
                        if not all(math.isfinite(x) for x in values) or min(values[:4]) <= 0 or v < 0 or l > min(o, c) or h < max(o, c):
                            raise ValueError('invalid_OHLCV')
                    except (ValueError, ZeroDivisionError) as error:
                        rejected.append({'ticker': ticker, 'date': day, 'reason': str(error)})
                        continue
                    dates.add(day)
                    rows.append({'company_id': ticker, 'session_date': day, **dict(zip(FIELDS, values)),
                                 'available_at': (pd.Timestamp(by_date[day]['close_at']) + pd.Timedelta(minutes=15)).isoformat(),
                                 'is_synthetic': 'false'})
            coverage[ticker] = {'rows': len(dates), 'missing_sessions': sorted(set(by_date) - dates)}
    rows.sort(key=lambda r: (r['company_id'], r['session_date']))
    write_csv(output / 'prices.csv', list(rows[0]), rows)
    write_csv(output / 'calendar.csv', list(sessions[0]), sessions)
    registry = [{'company_id': c['company_id'], 'name': c['name'], 'aliases': '|'.join(c['aliases']),
                 'valid_from': config['calendar_start'], 'valid_to': config['price_end']} for c in config['companies']]
    write_csv(output / 'companies.csv', list(registry[0]), registry)
    write_json(output / 'price_audit.json', {'rows': len(rows), 'coverage': coverage, 'rejected_rows': rejected,
               'source_sha256': sha256_file(raw / 'full_history.zip'),
               'price_basis': 'Retrospective adjusted_close/close adjustment; not point-in-time corporate action data',
               'calendar': {'name': 'XNYS', 'version': xcals.__version__, 'sessions': len(sessions)}})


def scan_news(path, config, output, db, allowed_categories=None):
    completed = db.execute('SELECT count(*) FROM scanned WHERE name=?', (path.name,)).fetchone()[0]
    if completed:
        return
    expected_size, expected_sha = NEWS_ARTIFACTS[path.name]
    if path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
        raise ValueError(f'Pinned artifact verification failed: {path}')
    companies = {c['company_id']: c for c in config['companies']}
    counts, years = Counter(), Counter()
    csv.field_size_limit(64 * 1024 * 1024)
    with path.open(newline='', encoding='utf-8-sig') as source:
        for row_number, row in enumerate(csv.DictReader(source), 2):
            counts['rows_scanned'] += 1
            years[row.get('Date', '')[:4]] += 1
            if counts['rows_scanned'] % 500000 == 0:
                db.commit()
                print(json.dumps({'archive': path.name, **counts}), flush=True)
            ticker = row.get('Stock_symbol', '')
            if ticker not in companies:
                continue
            stamp = row.get('Date', '')
            if not config['sample_start'] <= stamp[:10] <= config['sample_end']:
                continue
            counts['company_period_rows'] += 1
            title, url = row.get('Article_title', '').strip(), row.get('Url', '').strip()
            body = row.get('Article', '') or ''
            assertions = list(supported_assertions(title, body, companies[ticker], allowed_categories))
            if not assertions or not url.startswith(('https://', 'http://')):
                counts['rejected_assertion_entity_or_url'] += 1
                continue
            try:
                published, known = source_time(stamp)
            except ValueError:
                counts['invalid_timestamp'] += 1
                continue
            base = {'ticker': ticker, 'article_title': title, 'source_url': url, 'published_at': published, 'known_at': known,
                       'archive': path.name, 'source_row': row_number,
                       'article_sha256': hashlib.sha256(body.encode()).hexdigest() if body else None,
                       'article_excerpt': body[:6000], 'has_article_body': bool(body.strip()),
                       'source_row_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}
            # Later versions of a headline never inherit the earlier timestamp.
            for category, assertion, quote, span in assertions:
                original_text = title if span['field'] == 'Article_title' else body
                if original_text[span['start']:span['end']] != assertion:
                    raise ValueError('Evidence span does not reproduce the source assertion')
                payload = dict(base, title=assertion, company_quote=quote, categories=[category], evidence_span=span,
                               evidence_context=original_text[max(0, span['start'] - 200):span['end'] + 200])
                key = hashlib.sha256((ticker + '\n' + category + '\n' + url + '\n' + normalized_title(assertion)).encode()).hexdigest()
                previous = db.execute('SELECT payload FROM candidates WHERE id=?', (key,)).fetchone()
                if previous:
                    old = json.loads(previous[0])
                    if old['known_at'] >= known:
                        counts['exact_duplicate'] += 1
                        continue
                db.execute('INSERT OR REPLACE INTO candidates VALUES (?,?)', (key, json.dumps(payload)))
                counts['accepted_observations'] += 1
            if counts['rows_scanned'] % 10000 == 0:
                db.commit()
    db.execute('INSERT INTO scanned VALUES (?,?)', (path.name, json.dumps({'counts': dict(counts), 'years': dict(years), 'sha256': expected_sha})))
    db.commit()
    print(json.dumps({'archive_complete': path.name, **counts}), flush=True)


def export_events(config, output, db):
    candidates = [json.loads(r[0]) for r in db.execute('SELECT payload FROM candidates')]
    candidates.sort(key=lambda r: (r['known_at'], r['ticker'], r['title']))
    events, duplicates, recent = [], [], defaultdict(list)
    for source in candidates:
        ticker, title = source['ticker'], source['title']
        words = set(normalized_title(title).split())
        day = datetime.fromisoformat(source['published_at'].replace('Z', '+00:00'))
        for category in source['categories']:
            bucket = (ticker, category)
            recent[bucket] = [(d, w, eid) for d, w, eid in recent[bucket] if (day - d).days <= 14]
            duplicate = next((eid for d, w, eid in recent[bucket]
                              if len(words & w) / max(1, len(words | w)) >= .65
                              or category == 'earnings' and abs((day - d).total_seconds()) <= 3 * 86400), None)
            if duplicate:
                duplicates.append({'source_url': source['source_url'], 'category': category, 'duplicate_of': duplicate})
                continue
            identifier = 'scale_' + hashlib.sha256(json.dumps([ticker, category, normalized_title(title), source['published_at'][:10]]).encode()).hexdigest()[:24]
            recent[bucket].append((day, words, identifier))
            events.append({'event_id': identifier, 'category': category, 'description': title, 'title': title,
                           'document_id': 'doc_' + hashlib.sha256((source['source_url'] + title + source['known_at']).encode()).hexdigest()[:24],
                           'source_hash': hashlib.sha256(title.encode()).hexdigest(), 'source_url': source['source_url'],
                           'source_type': 'fnspid_quoted_assertion',
                           'known_at': source['known_at'], 'published_at': source['published_at'],
                           'companies': [{'company_id': ticker, 'name': source['company_quote'], 'resolution_status': 'resolved',
                                          'role': 'named_participant', 'evidence': source['company_quote']}],
                           'evidence': [title], 'amounts': [], 'expected_duration': None, 'valid_until': None,
                           'confidence': None, 'calibration_status': 'uncalibrated', 'validation_status': 'evidence_checked',
                           'verification_level': 'automated_pinned_archive_quote_entity_assertion_and_duplicate_checks',
                           'semantic_support_verified': False, 'human_reviewed': False, 'is_synthetic': False,
                           'historical_delivery_timestamp_verified': False, 'provenance': source})
    # Deterministic category/company/year stratification, no target-price labels.
    if len(events) > config['maximum_events']:
        groups = defaultdict(list)
        for event in events:
            groups[event['companies'][0]['company_id'], event['category'], event['published_at'][:4]].append(event)
        for group in groups.values():
            group.sort(key=lambda e: e['event_id'])
        events = [event for rank in range(max(map(len, groups.values()))) for key in sorted(groups)
                  for event in groups[key][rank:rank + 1]][:config['maximum_events']]
    events.sort(key=lambda e: (e['known_at'], e['event_id']))
    write_jsonl(output / 'events.jsonl', events)
    write_jsonl(output / 'duplicate_audit.jsonl', duplicates)
    # Stable, price-blind review sample; preserve unreviewed status.
    review = sorted(events, key=lambda e: hashlib.sha256(('review-v1' + e['event_id']).encode()).hexdigest())[:600]
    write_jsonl(output / 'human_review_sample.jsonl', review)
    summary = {'events': len(events), 'target_minimum': config['minimum_events'], 'target_maximum': config['maximum_events'],
               'extractor_version': '5-source-checked-dividend-declarations-plus-original-six-categories',
               'category_names': list(SCALE_CATEGORIES),
               'implementation_sha256': sha256_file(Path(__file__)),
               'count_gate_passed': config['minimum_events'] <= len(events) <= config['maximum_events'],
               'human_verified_events': 0, 'by_category': dict(Counter(e['category'] for e in events)),
               'by_year': dict(sorted(Counter(e['published_at'][:4] for e in events).items())),
               'by_company': dict(Counter(e['companies'][0]['company_id'] for e in events)),
               'scans': {name: json.loads(payload) for name, payload in db.execute('SELECT name,payload FROM scanned')},
               'limitations': ['Automated evidence checks are not human semantic verification or verification that reported claims are true.',
                              'Two-calendar-day news delay is assumed; actual historical arrival is not established.',
                              'Near-duplicate clustering cannot guarantee one row per underlying economic event.',
                              'Manual fixed-company universe and retrospective corporate-action adjustments remain biased.',
                              'Historical tests are exploratory; pretrained-model temporal overlap is unknown.']}
    if db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scanned_original'").fetchone():
        summary['original_six_category_scans'] = {name: json.loads(payload) for name, payload in db.execute('SELECT name,payload FROM scanned_original')}
    write_json(output / 'event_audit.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('scans', 'by_company')}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prices', 'news', 'extend', 'export'), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == 'prices':
        config = json.loads(args.config.read_text())
        config = audit_universe(args.raw, config, args.output)
        prepare_prices(args.raw, config, args.output)
        print(json.dumps({'companies': len(config['companies']), 'prices_complete': True}), flush=True)
        return
    config = json.loads((args.output / 'config.json').read_text())
    with sqlite3.connect(args.output / 'news_v5.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS scanned(name TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        if args.phase in ('news', 'extend'):
            for name in NEWS_ARTIFACTS:
                path = args.raw / name
                if path.is_file():
                    scan_news(path, config, args.output, db, ('dividend',) if args.phase == 'extend' else None)
        export_events(config, args.output, db)


if __name__ == '__main__':
    main()
