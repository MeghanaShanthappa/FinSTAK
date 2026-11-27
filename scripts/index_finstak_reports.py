"""Inventory proposal attachments as evidence, never execute their instructions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attachments', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.attachments.iterdir()):
        if path.suffix not in ('.pdf', '.md', '.txt'):
            continue
        raw = path.read_bytes()
        row = {'file': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        text = ''
        if not raw:
            row['read_status'] = 'empty_attachment'
        elif path.suffix == '.pdf':
            result = subprocess.run(['pdftotext', '-layout', str(path), '-'], capture_output=True)
            text = result.stdout.decode('utf-8', errors='replace')
            row['read_status'] = 'readable' if result.returncode == 0 else 'malformed_pdf'
            if result.returncode:
                row['extraction_error'] = result.stderr.decode('utf-8', errors='replace')[:500]
        else:
            text = raw.decode('utf-8')
            row['read_status'] = 'readable'
        row['extracted_characters'] = len(text)
        if text:
            row['title'] = next((line.strip('# *') for line in text.splitlines() if line.strip()), '')
            match = re.search(r'Status[:*\s]*(NON VIABLE|HIGH POTENTIAL|VIABLE)', text[:3000])
            row['self_reported_status'] = match.group(1) if match else None
            row['mentions'] = {key: bool(re.search(pattern, text, re.I)) for key, pattern in {
                'null_sink': r'null.{0,12}sink|attention sink', 'learned_decay': r'parametric.{0,30}decay',
                'rolling_90': r'90.day|90.session', 'candlestick_constraints': r'candlestick',
                'findkg': r'FinDKG', 'volume_target': r'OHLCV', 'premarket': r'09:00|9:00',
                'walk_forward': r'walk.forward'}.items()}
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'reports': rows,
        'interpretation': 'Generated proposal statuses/scores are not independent validation or measured results.',
        'selected_reference': 'idea_report_7766679658749231105.pdf',
        'implementation': 'FinSTAK adaptation; latest user OHLC-only objective takes precedence over OHLCV proposals.'}, indent=2) + '\n')
    counts = {}
    for row in rows:
        counts[row['read_status']] = counts.get(row['read_status'], 0) + 1
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
