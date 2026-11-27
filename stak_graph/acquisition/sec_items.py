"""Narrow SEC event records supported by declared 8-K item metadata.

These records assert that a filing reports an item. They do not pretend the
full filing was extracted, assign a numerical surprise, or backdate a filing
to an earlier announcement. Ambiguous generic items are deliberately omitted.
"""
import json
import re
from stak_graph.extraction.schema import digest

ITEMS = {'2.02': ('earnings', 'results of operations and financial condition'),
         '2.03': ('financing', 'a direct financial obligation or an off-balance-sheet obligation'),
         '3.02': ('financing', 'unregistered sales of equity securities')}


def events_from_filing(filing):
    if filing.get('form') != '8-K' or not filing.get('primaryDocument'):
        return []
    declared = set(re.findall(r'\b\d\.\d{2}\b', str(filing.get('items', ''))))
    grouped = {}
    for item in sorted(declared & ITEMS.keys()):
        category, label = ITEMS[item]
        grouped.setdefault(category, []).append((item, label))
    events = []
    for category, items in grouped.items():
        ticker, cik, accession = filing['ticker'], filing['cik'], filing['accessionNumber']
        text = (f'{ticker} SEC Form 8-K disclosure on {filing["filingDate"]}; reported ' +
                '; '.join(f'Item {item}: {label}' for item, label in items) + '.')
        url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace("-", "")}/{filing["primaryDocument"]}'
        events.append({'event_id': 'seci_' + digest([cik, accession, category])[:24], 'category': category,
                       'description': text, 'document_id': 'secitem_' + accession, 'source_hash': digest(filing),
                       'source_url': url, 'source_type': 'SEC_declared_form_item_metadata',
                       'known_at': filing['known_at'], 'published_at': filing['known_at'],
                       'companies': [{'company_id': ticker, 'name': ticker, 'resolution_status': 'resolved',
                                      'role': 'reporting_company', 'evidence': ticker}],
                       'evidence': [json.dumps(filing, sort_keys=True)], 'amounts': [], 'expected_duration': None,
                       'valid_until': None, 'confidence': None, 'calibration_status': 'uncalibrated',
                       'validation_status': 'evidence_checked', 'semantic_support_verified': True,
                       'verification_level': 'SEC_CIK_accession_filing_date_and_declared_item_metadata_join',
                       'human_reviewed': False, 'is_synthetic': False, 'full_document_extracted': False,
                       'reported_items': [item for item, _ in items], 'item_taxonomy_source': 'https://www.sec.gov/files/form8-k.pdf',
                       'claim_scope': 'Declared filing items, not unseen document content or actual announcement time'})
    return events
