"""Fill quarterly event gaps using actual SEC 8-K earnings-release evidence.

This does not infer missing EPS. A source can establish that earnings were
announced for a quarter without supplying a structured numerical EPS value.
"""
import argparse
from datetime import date, timedelta
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from stak_graph.io import read_jsonl
from stak_graph.extraction.schema import digest
from .public import PublicClient, filing_rows, known_at, write_json, write_jsonl


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        if tag in ('p', 'div', 'br', 'tr', 'td'):
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, value):
        if not self.skip:
            self.parts.append(value)

    def text(self):
        return re.sub(r'\s+', ' ', ''.join(self.parts)).strip()


def evidence(raw, quarter, year):
    source = raw.decode('utf-8', errors='replace')
    documents = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', source, flags=re.S | re.I)
    # SEC SGML contains immutable filed exhibit text, including press releases.
    texts = []
    for document in documents:
        kind = re.search(r'<TYPE>([^\r\n<]+)', document, flags=re.I)
        if kind and (kind.group(1).strip().upper().startswith('EX-99') or kind.group(1).strip().upper() == '8-K'):
            body = re.search(r'<TEXT>(.*?)(?:</TEXT>|$)', document, flags=re.S | re.I)
            parser = TextParser(); parser.feed(body.group(1) if body else document)
            texts.append(parser.text())
    ordinal = {'Q1': 'first', 'Q2': 'second', 'Q3': 'third', 'Q4': 'fourth'}[quarter]
    pattern = re.compile(r'\b(?:' + ordinal + r'[ -]+(?:fiscal[ -]+)?quarter|' + quarter + r')\b', flags=re.I)
    for text in texts:
        for match in pattern.finditer(text):
            quote = text[max(0, match.start() - 250):min(len(text), match.end() + 650)]
            # A quarter phrase and target fiscal-year reference are required
            # in the same excerpt, alongside financial-results language.
            if str(year) in quote and re.search(r'earnings|financial results|results|net income', quote, re.I):
                return quote, text
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knowledge', type=Path, required=True)
    args = parser.parse_args()
    root = args.knowledge
    client = PublicClient(root / 'raw')
    identifiers = {row['company_id']: row for row in read_jsonl(root / 'identifiers.jsonl')}
    missing = [row for row in read_jsonl(root / 'quarterly_coverage.jsonl') if row['status'].startswith('missing')]
    events, coverage = [], []
    for item in missing:
        ticker, year, quarter = item['ticker'], item['fiscal_year'], item['fiscal_quarter']
        cik = identifiers[ticker]['cik']
        submission = json.loads((root / f'raw/{ticker}_submissions.json').read_text())
        filings = filing_rows(submission['filings']['recent'])
        for path in (root / 'raw').glob(f'{ticker}_CIK*-submissions-*.json'):
            filings.extend(filing_rows(json.loads(path.read_text())))
        lookup = {row['accessionNumber']: row for row in filings}
        facts = json.loads((root / f'raw/{ticker}_companyfacts.json').read_text())
        # The last fiscal year's report can be filed in the following calendar
        # year and reside in a separate submissions-history block.
        for rows in facts['facts']['us-gaap']['EarningsPerShareDiluted']['units'].values():
            for fact in rows:
                if fact.get('fy') != year or fact.get('fp') != 'FY' or fact.get('form') != '10-K' or fact.get('accn') in lookup:
                    continue
                for history in submission['filings'].get('files', []):
                    if history['filingFrom'] <= fact['filed'] <= history['filingTo']:
                        older = client.json(f'{ticker}_{history["name"]}', 'https://data.sec.gov/submissions/' + history['name'], sec=True)
                        additions = filing_rows(older)
                        filings.extend(additions)
                        lookup.update({row['accessionNumber']: row for row in additions})
        endpoints = []
        for rows in facts['facts']['us-gaap']['EarningsPerShareDiluted']['units'].values():
            for fact in rows:
                filing = lookup.get(fact.get('accn'))
                if not filing or fact.get('fy') != year or fact.get('end') != filing.get('reportDate'):
                    continue
                if (quarter == 'Q4' and filing['form'] == '10-K' and fact.get('fp') == 'FY') or (quarter != 'Q4' and fact.get('fp') == quarter and filing['form'] == '10-Q'):
                    endpoints.append(fact['end'])
        if not endpoints and quarter != 'Q4':
            # For a confirmed December fiscal year-end, use the actual 10-Q
            # reporting period to identify a missing numerical XBRL context.
            annual_ends = [fact['end'] for rows in facts['facts']['us-gaap']['EarningsPerShareDiluted']['units'].values()
                           for fact in rows if fact.get('fy') == year and fact.get('fp') == 'FY'
                           and fact.get('form') == '10-K' and fact['end'] == f'{year}-12-31']
            expected = {'Q1': f'{year}-03-31', 'Q2': f'{year}-06-30', 'Q3': f'{year}-09-30'}[quarter]
            if annual_ends and any(row['form'] == '10-Q' and row.get('reportDate') == expected for row in filings):
                endpoints.append(expected)
        if not endpoints:
            coverage.append({**item, 'release_status': 'no_matching_period_endpoint'})
            continue
        end = date.fromisoformat(sorted(endpoints)[0])
        candidates = sorted((filing for filing in filings if filing['form'] == '8-K' and '2.02' in filing.get('items', '')
                             and end < date.fromisoformat(filing['filingDate']) <= end + timedelta(days=80)),
                            key=lambda filing: filing['filingDate'])
        accepted = None
        for filing in candidates:
            accession = filing['accessionNumber']
            url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace("-", "")}/{accession}.txt'
            try:
                raw = client.get(f'{ticker}_{accession}.txt', url, sec=True)
                quote, text = evidence(raw, quarter, year)
                if quote is None:
                    continue
                source = {**item, 'period_end': end.isoformat(), 'accession': accession,
                          'known_at': known_at(filing), 'source_url': url, 'quote': quote,
                          'raw_sha256': hashlib.sha256(raw).hexdigest(), 'quarterly_eps_value': None,
                          'release_status': 'quarter_earnings_release_evidence',
                          'validation_level': 'CIK_filing_period_window_and_explicit_quarter_year_results_excerpt',
                          'human_reviewed': False}
                event = {'event_id': 'secr_' + digest([ticker, year, quarter])[:24], 'category': 'earnings',
                         'description': quote, 'title': f'{ticker} fiscal {year} {quarter} earnings-release evidence',
                         'document_id': 'sec_' + accession, 'source_hash': hashlib.sha256(text.encode()).hexdigest(),
                         'source_url': url, 'source_type': 'sec_8k_earnings_release_excerpt',
                         'known_at': known_at(filing), 'published_at': known_at(filing),
                         'companies': [{'name': ticker, 'company_id': ticker, 'resolution_status': 'resolved',
                                        'role': 'reporting_company', 'evidence': f'SEC CIK {cik}'}],
                         'amounts': [], 'expected_duration': None, 'confidence': None, 'calibration_status': 'uncalibrated',
                         'valid_until': None, 'expected_direction': None, 'is_synthetic': False,
                         'validation_status': 'evidence_checked', 'semantic_support_verified': False,
                         'human_reviewed': False, 'evidence': [quote], 'financial_observation': source,
                         'relationship_basis': 'directly_reported'}
                events.append(event); accepted = source
                break
            except Exception as error:
                print(json.dumps({'ticker': ticker, 'quarter': quarter, 'source_error': type(error).__name__}), flush=True)
        coverage.append(accepted or {**item, 'release_status': 'no_verified_release_excerpt'})
        print(json.dumps({'ticker': ticker, 'fiscal_year': year, 'quarter': quarter, 'release_found': accepted is not None}), flush=True)
    write_jsonl(root / 'sec_release_events.jsonl', events)
    write_jsonl(root / 'quarterly_release_coverage.jsonl', coverage)
    write_json(root / 'release_inventory.json', {'missing_direct_EPS_quarters': len(missing),
               'quarters_with_additional_release_evidence': len(events), 'remaining_uncovered': len(missing) - len(events),
               'numerical_EPS_values_inferred': 0})


if __name__ == '__main__':
    main()
