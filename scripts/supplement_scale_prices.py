"""Replace incomplete FNSPID company series with complete sourced Yahoo series.

Use the whole supplementary series per affected company, never splice differently
adjusted price bases. Preserve the original export and every downloaded response.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path('/tmp/finstak-100x10-v1/dataset')


def main():
    config = json.loads((ROOT / 'config.json').read_text())
    audit = json.loads((ROOT / 'price_audit.json').read_text())
    # A single invalid observation also receives a source-level replacement.
    targets = [k for k, v in audit['coverage'].items() if v['missing_sessions']]
    cache = ROOT / 'supplementary_prices'
    cache.mkdir(exist_ok=True)
    with (ROOT / 'calendar.csv').open() as handle:
        calendar = {r['session_date']: r for r in csv.DictReader(handle)}
    original = ROOT / 'prices_fnspid.csv'
    if not original.exists():
        (ROOT / 'prices.csv').rename(original)
    with original.open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [r for r in reader if r['company_id'] not in targets]
    provenance = []
    for ticker in targets:
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1=1380585600&period2=1703894400&interval=1d&events=div%2Csplits'
        path = cache / f'{ticker}.json'
        if not path.exists():
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 FinSTAK research'})
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read()
            path.write_bytes(raw)
            time.sleep(1)
        result = json.loads(path.read_bytes())['chart']['result'][0]
        q = result['indicators']['quote'][0]
        adjusted = result['indicators']['adjclose'][0]['adjclose']
        seen = set()
        for i, stamp in enumerate(result['timestamp']):
            day = datetime.fromtimestamp(stamp, timezone.utc).date().isoformat()
            if day not in calendar:
                continue
            values = [q[k][i] for k in ('open', 'high', 'low', 'close', 'volume')]
            if any(v is None for v in values) or adjusted[i] is None:
                continue
            o, h, l, c, volume = values
            if min(o, h, l, c, adjusted[i]) <= 0 or volume < 0 or l > min(o, c) or h < max(o, c):
                continue
            if day in seen:
                raise ValueError('Duplicate supplementary date')
            seen.add(day)
            factor = adjusted[i] / c
            close_at = datetime.fromisoformat(calendar[day]['close_at'])
            from datetime import timedelta
            rows.append({'company_id': ticker, 'session_date': day,
                         **{k: q[k][i] * factor for k in ('open', 'high', 'low', 'close')}, 'volume': volume,
                         'available_at': (close_at + timedelta(minutes=15)).isoformat(), 'is_synthetic': 'false'})
        missing = sorted(set(calendar) - seen)
        provenance.append({'ticker': ticker, 'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'rows': len(seen), 'missing_sessions': missing,
                           'replacement_policy': 'Entire company series replaced; original preserved',
                           'fetched_at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
        print(json.dumps(provenance[-1]), flush=True)
        if missing:
            raise ValueError(f'Supplementary series remains incomplete for {ticker}')
    rows.sort(key=lambda r: (r['company_id'], r['session_date']))
    temporary = ROOT / 'prices_supplemented.tmp'
    with temporary.open('w', newline='') as out:
        writer = csv.DictWriter(out, fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(ROOT / 'prices.csv')
    (ROOT / 'supplementary_price_audit.json').write_text(json.dumps({'companies': provenance, 'rows': len(rows),
        'final_prices_sha256': hashlib.sha256((ROOT / 'prices.csv').read_bytes()).hexdigest(),
        'point_in_time_adjustment_verified': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
