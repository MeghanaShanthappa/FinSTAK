"""Acquire actual SEC facts/accessions for the expanded study, resumably."""
import json
import os
from pathlib import Path
from stak_graph.acquisition.public import PublicClient, sec_sources


def main():
    root = Path('/tmp/finstak-100x10-v1')
    config = json.loads((root / 'dataset/config.json').read_text())
    config['historical_sec_identifiers'] = {
        'BK': {'cik': '1390777', 'source_url': 'https://www.sec.gov/Archives/edgar/data/1390777/000162828015000217/ex991_earningsrelease4q14.htm'},
        'K': {'cik': '55067', 'source_url': 'https://www.sec.gov/Archives/edgar/data/55067/000119312514174802/d720489dex991.htm'},
        'EQR': {'cik': '906107', 'source_url': 'https://www.sec.gov/Archives/edgar/data/906107/000090610715000007/eqr-20141231x10k.htm'},
    }
    output = root / 'sec'
    output.mkdir(exist_ok=True)
    client = PublicClient(output / 'raw')
    (output / 'acquisition_config.json').write_text(json.dumps(config, indent=2) + '\n')
    # Contact is supplied by the authorized user; retained only in request headers.
    os.environ['SEC_CONTACT_EMAIL'] = 'mshanthappa@nvidia.com'
    _, summary = sec_sources(client, config, output)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
