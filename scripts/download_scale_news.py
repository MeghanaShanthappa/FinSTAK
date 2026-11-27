"""Acquire the pinned second FNSPID archive, without overwriting prior data."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path('/tmp/stak-graph-real-data/raw')
NAME = 'nasdaq_exteral_data.csv'
SIZE = 23232979597
SHA = '1a7a3eb8e6b97ec19f286f2cfca3371542bddb272ab1eb8f36e33ad98fa5c4da'
REV = 'bf9189c41527198897d1af3e17b1a0095279fc45'


def main():
    target = ROOT / NAME
    partial = ROOT / (NAME + '.part')
    if not target.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        url = f'https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/{REV}/Stock_news/{NAME}?download=true&offset={offset}'
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
            if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                raise ValueError('Server did not honor resume offset')
            with partial.open('ab' if offset else 'xb') as out:
                while block := response.read(8 * 1024 * 1024):
                    out.write(block)
                    offset += len(block)
                    if offset // (1024**3) != (offset - len(block)) // (1024**3):
                        print(json.dumps({'downloaded_bytes': offset, 'expected_bytes': SIZE}), flush=True)
        if partial.stat().st_size != SIZE:
            raise ValueError('Incomplete download; rerun to resume')
        partial.rename(target)
    digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
    if target.stat().st_size != SIZE or digest != SHA:
        raise ValueError('Pinned source checksum mismatch')
    print(json.dumps({'complete': True, 'path': str(target), 'sha256': digest}), flush=True)


if __name__ == '__main__':
    main()
