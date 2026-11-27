"""Retain completed research artifacts within the user's available home quota."""
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import tarfile
import numpy as np

ROOT = Path('/tmp/finstak-100x10-v1')
PROJECT = Path('/home/mshanthappa/STAK-Graph')
OUTPUT = PROJECT / 'results/scale_100x10_v1_retained'


def checksum(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    if json.loads((ROOT / 'pipeline_status.json').read_text())['state'] != 'complete':
        raise ValueError('Research pipeline has not completed')
    for name in ('source_verification.json', 'independent_verification.json'):
        if not json.loads((ROOT / name).read_text())['passed']:
            raise ValueError('Required verification is incomplete')
    outcome = json.loads((ROOT / 'outcome.json').read_text())
    restore = []
    for folder, packed in [('embeddings', 'event_arrays.npz'), ('finbert_embeddings', 'event_arrays_finbert.npz')]:
        original = ROOT / 'expanded' / folder / 'vectors.f32'
        data = np.load(ROOT / 'packed' / packed)['embeddings'].astype('<f4', copy=False).tobytes()
        expected = checksum(original)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Packed/source vector mismatch')
        restore.append({'destination': str(original.relative_to(ROOT)), 'packed': 'packed/' + packed,
                        'sha256': expected, 'bytes': len(data)})
    (ROOT / 'embedding_restore_manifest.json').write_text(json.dumps(restore, indent=2) + '\n')
    delivery = ROOT / 'delivery_scripts'
    delivery.mkdir(exist_ok=True)
    for name in ('replay_scale_forecast.py', 'analyze_scale_outcomes.py', 'restore_scale_embeddings.py',
                 'verify_scale_native.py', 'verify_scale_study.py', 'report_scale_study.py', 'package_scale_study.py'):
        shutil.copy2(PROJECT / 'scripts' / name, delivery / name)
    (ROOT / 'README.txt').write_text('''FinSTAK: 100-stock, 2014–2023 research study

Start with results.html. Machine-readable measurements are in summary.json and
outcome.json. The study contains 27,030 automatically source-checked event records;
these are not human-certified events or guaranteed unique economic occurrences.

The archive retains frozen source, packed research inputs, fitted checkpoints,
held-out predictions and audit evidence. Pretrained foundation/text model weights
and the large original news archives are excluded; their revisions and source
checksums remain recorded. Historical checkpoint replay uses the retained packed
embeddings and does not require downloading those pretrained weights.

Install the research dependencies listed in source_snapshot/pyproject.toml.
From the extracted study directory, replay 100 historical price forecasts with:

python delivery_scripts/replay_scale_forecast.py --study . --stage data_stage --fold 2023 --variant no_decay --date 2023-12-27 --device cpu --output forecast.json

For historical OHLCV forecasts use --stage architecture_stage --variant combined.
The helper checks frozen code, packed data and checkpoint checksums. CPU and GPU
floating-point results can differ slightly. Replay excludes target labels from
model inputs; it is not a live data acquisition or deployment interface.

If duplicate raw embedding binaries were omitted to fit storage limits, recreate
them exactly with:

python delivery_scripts/restore_scale_embeddings.py --study .

The restoration helper verifies the reconstructed checksums. Packed arrays already
contain these same vectors, so restoration is optional for forecast replay.
Acquisition/training/report scripts preserve the original run paths. A fresh full
study requires configuring new paths, downloading the pinned sources and freezing
that new implementation before fitting. It must not overwrite this study.
''')
    # Supplement the frozen report without changing any training implementation.
    report = (ROOT / 'results.html').read_text()
    supplement = json.loads((ROOT / 'dataset/supplementary_price_audit.json').read_text())
    interval = outcome['comparisons']['validation_selected_new_vs_validation_selected_baseline']
    gain = interval['MAE_reduction_percentage_points']
    low, high = interval['95_percent_interval']
    findings = ('The new component family reduced error on this study.' if low > 0 else
                'The new component family increased error on this study.' if high < 0 else
                'The interval includes zero; this study does not establish a reliable error reduction from the new component family.')
    section = '<section id="delivery-summary"><h2>Validation-selected family comparison</h2>'
    section += '<p>Each year selects its recipe using mean seed validation error. Its test forecasts average the three selected checkpoints.</p>'
    section += f'<p>{findings} Error reduction: {gain:+.4f} percentage points; exploratory 95% date-block interval [{low:+.4f}, {high:+.4f}].</p>'
    summary = json.loads((ROOT / 'summary.json').read_text())
    baseline_error = outcome['pooled_test_MAE_percent']['selected_baseline']
    section += '<h2>Pooled test results</h2><p>All rows use the same 74,900 stock-date targets from 2021–2023. Error is normalized OHLC MAE as a percentage of the observed decision-day close; lower is better.</p>'
    section += '<table><tr><th>Model</th><th>Pooled price error</th></tr>'
    for label, key in [('Price-only recurrent control', 'no_events'), ('FinSTAK without decay', 'no_decay'),
                       ('Hybrid decay', 'hybrid_decay'), ('Combined new recipe', 'combined'),
                       ('Frozen native TimesFM', 'native_TimesFM_price_only')]:
        error = baseline_error - summary['comparisons'][key + '_vs_no_decay']['MAE_reduction_percentage_points']
        section += f'<tr><th>{html.escape(label)}</th><td>{error:.4f}%</td></tr>'
    section += '</table><p>Hybrid decay has the lowest observed pooled error, but its exploratory interval against no-decay FinSTAK includes zero. The latest-fold validation choice remains no-decay FinSTAK. Model selection results and fixed-recipe comparisons are reported separately.</p>'
    section += '<h2>Data provenance</h2><p>News and original prices come from the pinned '
    section += '<a href="https://huggingface.co/datasets/Zihan1004/FNSPID/tree/bf9189c41527198897d1af3e17b1a0095279fc45">FNSPID release</a>. '
    section += f'Complete Yahoo historical series replace {len(supplement["companies"])} incomplete or invalid company series; their responses, URLs and checksums are retained. '
    section += 'Official disclosures come from <a href="https://www.sec.gov/search-filings/edgar-application-programming-interfaces">SEC EDGAR</a>. '
    section += 'Automated source checks are not human semantic certification or independent verification of news truth.</p>'
    section += '<p>PCGrad does not guarantee conditional mean invariance. Sector groups are static research groups; the study does not reconstruct historical supply-chain, consensus or pre-market news feeds.</p></section>'
    section += '<section><h2>Architecture and implementation coverage</h2>'
    section += '<p>Each stock uses 19 causal price, volume and technical features. A 32-unit recurrent encoder summarizes its recent history. Frozen 768-dimensional news embeddings enter four-head typed event attention, with a null sink so absent or stale evidence can contribute zero. Event disturbances can then propagate through rolling correlation peers or the sector alternative. The output parameterization predicts open and close relative to the observed close and constructs high and low with positive upper and lower wicks.</p>'
    coverage = [
        ('Gradient-isolated candlesticks', 'Separate wick heads receive detached temporal features and detached body bounds; a separate dual-stream condition trains an independent dispersion encoder.'),
        ('Price and volume task conflicts', 'Shared OHLCV, learned uncertainty weighting, PCGrad and fully separate valuation/dispersion streams are compared.'),
        ('Event persistence', 'No decay, learned exponential decay and exponential/power-law mixtures are tested with category-specific age limits.'),
        ('Graph propagation', 'Correlation peers, static company–sector–company routing and strict primary-issuer event filtering are compared.'),
        ('Sequence and event encoders', 'One- and two-layer recurrent encoders, Gemma versus FinBERT event embeddings, and an auxiliary direction head are tested.'),
        ('Foundation-model benchmark', 'A frozen native TimesFM price-only forecast is evaluated on exactly matched stock-date targets.'),
        ('Still unavailable', 'Historically reconstructed supply-chain edges, analyst-consensus surprises, court-docket changes, observed pre-market arrival times and human event certification remain outside this study.'),
    ]
    section += '<table><tr><th>Summary proposal</th><th>Implemented experiment or remaining limit</th></tr>'
    section += ''.join('<tr><th>' + html.escape(a) + '</th><td>' + html.escape(b) + '</td></tr>' for a, b in coverage)
    section += '</table><p>The combined recipe uses a 20-session, two-layer recurrent encoder, FinBERT event vectors, hybrid decay, sector routing, primary-issuer filtering, gradient-isolated wicks, volume prediction, auxiliary direction classification and PCGrad. It is evaluated as a hypothesis; adding these components is not itself evidence of better forecasts.</p></section>'
    profile = json.loads((ROOT / 'dataset_profile.json').read_text())
    section += '<section><h2>Event coverage</h2><p>All 100 stocks have retained events. Over the decade, counts per stock range from '
    counts = profile['event_records_per_stock']
    section += f'{counts["minimum"]} to {counts["maximum"]}, with median {counts["median"]:g}. These are retained evidence records, not guaranteed distinct economic events. No records have human certification.</p>'
    section += '<table><tr><th>Known year</th><th>Event records</th></tr>'
    section += ''.join(f'<tr><td>{html.escape(year)}</td><td>{count:,}</td></tr>' for year, count in profile['by_known_year'].items())
    section += '</table><details><summary>Category and source coverage</summary><pre>' + html.escape(json.dumps(profile, indent=2)) + '</pre></details></section>'
    section += '<details><summary>Verification and reporting corrections</summary><p>The initial replay check used two dates while the saved forecasts used 16-date batches. Different floating-point kernels produced a small numerical difference. Replaying the original batch size reproduced all saved seed-7 model/year forecasts to rounding precision, without relaxing the original tolerance. Separate two-date future-input and target-label perturbations still produced zero change. All 153 checkpoint checksums passed. A report-only variable-name collision was also corrected. Training sources, fitted checkpoints and saved predictions were preserved; both original and revised verification/report code and correction provenance are retained.</p></details>'
    if 'id="delivery-summary"' not in report:
        report = report.replace('</html>', section + '</html>')
        (ROOT / 'results.html').write_text(report)
    (ROOT / 'report_manifest.json').write_text(json.dumps({name: checksum(ROOT / name) for name in ('results.html', 'summary.json', 'outcome.json')}, indent=2) + '\n')
    directories = ('source_snapshot', 'packed', 'expanded', 'dataset', 'sec', 'data_stage', 'architecture_stage',
                   'native_timesfm', 'delivery_scripts')
    members = []
    for directory in directories:
        for path in (ROOT / directory).rglob('*'):
            if (not path.is_file() or path.is_symlink() or '__pycache__' in path.parts or
                    path.suffix in ('.sqlite3', '.db', '.pyc', '.tmp') or '.sqlite3-' in path.name or
                    '_preflight_legacy' in path.name):
                continue
            if path == ROOT / 'native_timesfm/summary.json' and (ROOT / 'native_timesfm/aligned_summary.json').exists():
                continue
            members.append(path)
    roots = ('README.txt', 'results.html', 'summary.json', 'outcome.json', 'runtime_versions.json', 'dataset_profile.json',
             'source_verification.json', 'independent_verification.json', 'pipeline_status.json', 'report_manifest.json',
             'embedding_restore_manifest.json', 'example_historical_forecast.json', 'forecast_interface_verification.json',
             'example_ohlcv_forecast.json', 'native_forecast_verification.json', 'data_stage_outcome.json',
             'finbert_compatibility_verification.json', 'frozen_tests.log', 'verification_recovery.json',
             'replay_batch_diagnostic.json', 'independent_verification.log',
             'pipeline_status_before_verification_recovery.json', 'postprocessing_provenance.json',
             'report_recovered.log', 'ohlcv_forecast_verification.json')
    members.extend(ROOT / name for name in roots if (ROOT / name).is_file())
    members = sorted(set(members))
    hashes = {str(path.relative_to(ROOT)): checksum(path) for path in members}
    members.append(ROOT / 'delivery_manifest.json')
    reserve = 200 * 1024**2
    budget = shutil.disk_usage(PROJECT).free - reserve
    archive = ROOT / 'FinSTAK_100stocks_2014-2023_research.tar.gz'
    omit_vectors = False
    for attempt in range(2):
        omitted = {name: value for name, value in hashes.items()
                   if omit_vectors and Path(name).name == 'vectors.f32'}
        included = {name: value for name, value in hashes.items() if name not in omitted}
        (ROOT / 'delivery_manifest.json').write_text(json.dumps({
            'files': included, 'omitted_reconstructible_files': omitted,
            'raw_news_archives_included': False, 'pretrained_model_weights_included': False,
            'source_reference': 'Pinned URLs/revisions and checksums are retained.'}, indent=2) + '\n')
        with tarfile.open(archive, 'w:gz', compresslevel=6) as out:
            for path in members:
                if omit_vectors and path.name == 'vectors.f32':
                    continue
                out.add(path, arcname='finstak-100x10-v1/' + str(path.relative_to(ROOT)), recursive=False)
        if archive.stat().st_size <= budget:
            break
        omit_vectors = True
    else:
        raise ValueError('Archive exceeds available home quota even without duplicate vectors')
    verified = 0
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            relative = str(Path(member.name).relative_to('finstak-100x10-v1'))
            if relative not in hashes:
                continue
            handle = source.extractfile(member)
            if hashlib.file_digest(handle, 'sha256').hexdigest() != hashes[relative]:
                raise ValueError('Retained archive verification failed: ' + relative)
            verified += 1
    if verified != len(included):
        raise ValueError('Archive member count does not match its delivery manifest')
    OUTPUT.mkdir(exist_ok=True)
    destination = OUTPUT / archive.name
    partial = OUTPUT / (archive.name + '.part')
    try:
        shutil.copy2(archive, partial)
        if checksum(partial) != checksum(archive):
            raise ValueError('Home archive copy checksum mismatch')
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    for name in ('results.html', 'summary.json', 'outcome.json', 'dataset_profile.json', 'delivery_manifest.json'):
        shutil.copy2(ROOT / name, OUTPUT / name)
    manifest = {'archive': str(destination), 'bytes': destination.stat().st_size, 'sha256': checksum(destination),
                'archive_members_verified': verified, 'duplicate_vector_files_omitted': omit_vectors,
                'restore_optional_vectors': 'python delivery_scripts/restore_scale_embeddings.py --study STUDY_ROOT'}
    (OUTPUT / 'archive_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
