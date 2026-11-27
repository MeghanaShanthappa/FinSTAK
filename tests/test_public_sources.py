import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from stak_graph.acquisition.public import known_at, quarterly_records, parse_gkg, PublicClient
from stak_graph.acquisition.earnings_releases import evidence


class PublicSourcesTest(unittest.TestCase):
    def test_release_requires_explicit_quarter_year_and_results(self):
        raw = b'<DOCUMENT>\n<TYPE>EX-99.1\n<TEXT><html>Issuer reports fourth quarter 2017 financial results.</html></TEXT></DOCUMENT>'
        quote, text = evidence(raw, 'Q4', 2017)
        self.assertIn('fourth quarter 2017', quote)
        self.assertEqual(evidence(raw, 'Q1', 2017), (None, None))
        self.assertEqual(evidence(raw, 'Q4', 2016), (None, None))
        self.assertIn(quote, text)

    def test_late_acceptance_never_precedes_public_filing_date(self):
        self.assertEqual(known_at({'filingDate': '2019-10-31', 'acceptanceDateTime': '2019-10-30T22:12:36Z'}),
                         '2019-10-31T23:59:59Z')

    def test_quarter_context_and_original_vintage(self):
        filings = [{'accessionNumber': 'first', 'form': '10-Q', 'filingDate': '2017-05-01',
                    'reportDate': '2017-03-31', 'primaryDocument': 'report.htm'},
                   {'accessionNumber': 'later', 'form': '10-Q', 'filingDate': '2018-05-01',
                    'reportDate': '2018-03-31', 'primaryDocument': 'report.htm'}]
        records = [{'accn': 'first', 'fy': 2017, 'fp': 'Q1', 'start': '2017-01-01', 'end': '2017-03-31', 'val': 1.2},
                   {'accn': 'later', 'fy': 2018, 'fp': 'Q1', 'start': '2017-01-01', 'end': '2017-03-31', 'val': 1.1}]
        facts = {'facts': {'us-gaap': {'EarningsPerShareDiluted': {'units': {'USD/shares': records}}}}}
        coverage, events = quarterly_records('TEST', '123', facts, filings, [2017, 2018])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['financial_observation']['value'], 1.2)
        self.assertEqual(events[0]['known_at'], '2017-05-01T23:59:59Z')
        self.assertEqual(sum(row['status'] == 'missing_direct_quarterly_eps' for row in coverage), 7)

    def test_annual_eps_not_misrepresented_as_fourth_quarter(self):
        filings = [{'accessionNumber': 'annual', 'form': '10-K', 'filingDate': '2018-02-01',
                    'reportDate': '2017-12-31', 'primaryDocument': 'report.htm'}]
        annual = {'accn': 'annual', 'fy': 2017, 'fp': 'FY', 'start': '2017-01-01', 'end': '2017-12-31', 'val': 5.2}
        facts = {'facts': {'us-gaap': {'EarningsPerShareDiluted': {'units': {'USD/shares': [annual]}}}}}
        _, events = quarterly_records('TEST', '123', facts, filings, [2017])
        self.assertEqual(events, [])
        quarter = dict(annual, start='2017-10-01', val=1.4)
        facts['facts']['us-gaap']['EarningsPerShareDiluted']['units']['USD/shares'].append(quarter)
        _, events = quarterly_records('TEST', '123', facts, filings, [2017])
        self.assertEqual(events[0]['financial_observation']['fiscal_quarter'], 'Q4')
        self.assertFalse(events[0]['financial_observation']['is_derived'])

    def test_gdelt_is_discovery_and_not_event_or_verified_entity(self):
        row = [''] * 27
        row[0], row[1], row[4], row[7], row[13] = 'id', '20190115150000', 'https://example.org/news', 'ECON;', 'apple inc;'
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('gkg.csv', '\t'.join(row) + '\n')
        records = parse_gkg(buffer.getvalue(), {'AAPL': ['Apple Inc'], 'KO': ['Coca-Cola']}, 'https://example.org/archive.zip')
        self.assertEqual(records[0]['candidate_tickers'], ['AAPL'])
        self.assertFalse(records[0]['accepted_financial_event'])
        self.assertFalse(records[0]['entity_links_verified'])
        self.assertEqual(records[0]['first_seen_at'], '2019-01-15T15:00:00Z')

    def test_modified_cached_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'source.json'
            path.write_text('{}')
            path.with_suffix('.json.source.json').write_text(json.dumps({'sha256': 'incorrect'}))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                PublicClient(root).get('source.json', 'https://example.org')
