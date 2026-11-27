"""Compact causal panel and date-local graph batches for the larger study."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from stak_graph.graph.market import TradingCalendar, read_bars, candle_features
from .scale_data import SCALE_CATEGORIES
from stak_graph.io import read_jsonl
from .scale_io import write_json
from stak_graph.embeddings.pipeline import load_vectors, event_text
from stak_graph.extraction.schema import digest
from .data import sha256_file


def rolling_moments(values, window=90):
    count = np.minimum(np.arange(len(values)) + 1, window)
    sums = np.concatenate(([0.], np.cumsum(values, dtype=np.float64)))
    squares = np.concatenate(([0.], np.cumsum(values ** 2, dtype=np.float64)))
    end = np.arange(1, len(values) + 1)
    begin = np.maximum(0, end - window)
    mean = (sums[end] - sums[begin]) / count
    variance = (squares[end] - squares[begin]) / count - mean ** 2
    return mean, np.sqrt(np.maximum(variance, 0))


def build_panel(dataset, output):
    dataset, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((dataset / 'config.json').read_text())
    calendar = TradingCalendar.from_csv(dataset / 'calendar.csv')
    bars = read_bars(dataset / 'prices.csv', calendar)
    companies = sorted(c['company_id'] for c in config['companies'])
    company_index = {c: i for i, c in enumerate(companies)}
    n, c = len(calendar.sessions), len(companies)
    raw = np.full((n, c, 5), np.nan, dtype=np.float64)
    features = np.zeros((n, c, 19), dtype=np.float32)
    volume_mean = np.zeros((n, c), dtype=np.float64)
    volume_scale = np.zeros((n, c), dtype=np.float64)
    groups = {company: [] for company in companies}
    for bar in bars:
        groups[bar.company_id].append(bar)
        raw[calendar.index[bar.session_date], company_index[bar.company_id]] = [bar.open, bar.high, bar.low, bar.close, bar.volume]
    if not np.isfinite(raw).all():
        raise ValueError('Complete observed panel required; no silent repeated-price or future padding')
    for company, history in groups.items():
        j = company_index[company]
        features[:, j, 5:] = np.asarray(candle_features(history, calendar))
        # Pool all four OHLC channels in each trailing calendar-session window.
        price_mean, _ = rolling_moments(raw[:, j, :4].mean(-1))
        price_square, _ = rolling_moments((raw[:, j, :4] ** 2).mean(-1))
        price_scale = np.maximum(np.sqrt(np.maximum(price_square - price_mean ** 2, 0)), price_mean * 1e-6)
        features[:, j, :4] = (raw[:, j, :4] - price_mean[:, None]) / price_scale[:, None]
        vm, vs = rolling_moments(np.log1p(raw[:, j, 4]))
        vs = np.maximum(vs, .1)
        features[:, j, 4] = (np.log1p(raw[:, j, 4]) - vm) / vs
        volume_mean[:, j], volume_scale[:, j] = vm, vs
    features[:, :, :5] = features[:, :, :5].clip(-10, 10)
    log_return = np.zeros((n, c), dtype=np.float64)
    log_return[1:] = np.diff(np.log(raw[:, :, 3]), axis=0)
    peers = np.tile(np.arange(c)[None, :, None], (n, 1, 3)).astype(np.int64)
    peer_weight = np.zeros((n, c, 3), dtype=np.float32)
    for t in range(60, n):
        corr = np.corrcoef(log_return[t - 59:t + 1], rowvar=False)
        np.fill_diagonal(corr, -np.inf)
        order = np.argsort(-corr, axis=1, kind='stable')[:, :3]
        weights = np.take_along_axis(corr, order, axis=1)
        peers[t] = order
        peer_weight[t] = np.where(np.isfinite(weights) & (weights >= .2), weights, 0)
    sectors = sorted(set(c['sector'] for c in config['companies']))
    sector_map = {row['company_id']: sectors.index(row['sector']) for row in config['companies']}
    days = np.array([s.session_date for s in calendar.sessions])
    path = output / 'panel.npz'
    np.savez(path, raw=raw, features=features, volume_mean=volume_mean, volume_scale=volume_scale,
             peers=peers, peer_weight=peer_weight, dates=days, companies=np.array(companies),
             sectors=np.array([sector_map[c] for c in companies]))
    write_json(output / 'panel_manifest.json', {'config': config, 'panel_sha256': sha256_file(path),
               'source_hashes': {name: sha256_file(dataset / name) for name in ('prices.csv', 'calendar.csv', 'companies.csv', 'config.json')},
               'stocks': c, 'sessions': n, 'first': days[0], 'last': days[-1],
               'normalization': 'Trailing 90 completed sessions, pooled OHLC moments; separate log1p volume, floor .1',
               'sector_labels': 'Manually assigned static research groups; not reconstructed historical GICS',
               'correlations': 'Previous 60 observed returns, including decision date; top 3 positive >= .2'})
    return path


def build_event_arrays(dataset, expanded, output, embedding_subdirectory='embeddings', filename='event_arrays.npz'):
    dataset, expanded, output = map(Path, (dataset, expanded, output))
    panel = np.load(output / 'panel.npz')
    companies = panel['companies'].tolist()
    calendar = TradingCalendar.from_csv(dataset / 'calendar.csv')
    vectors, manifest = load_vectors(expanded / embedding_subdirectory)
    if manifest['encoder'].get('simulated'):
        raise ValueError('Real frozen text embeddings required')
    events = list(read_jsonl(expanded / 'events.jsonl'))
    embedding, category, primary, source, target, ages = [], [], [], [], [], []
    day_count, company_count = len(panel['dates']), len(companies)
    for i, event in enumerate(events):
        vector = vectors[event['event_id']]
        if (event['source_hash'] != vector['source_hash'] or digest(event_text(event)) != vector['text_sha256']
                or event['validation_status'] != 'evidence_checked' or event.get('is_synthetic')):
            raise ValueError('Unverified, stale or synthetic event')
        embedding.append(vector['vector'])
        category.append(SCALE_CATEGORIES.index(event['category']))
        primary.append(bool(event.get('primary_discloser_verified', False)))
        first = calendar.first_eligible(event['known_at'])
        if first is None:
            continue
        ttl = {'earnings': 90, 'litigation': 500}.get(event['category'], 15)
        for company in event['companies']:
            if company['resolution_status'] != 'resolved' or company['company_id'] not in companies:
                raise ValueError('Unresolved event company')
            j = companies.index(company['company_id'])
            for t in range(first, min(first + ttl, day_count)):
                if event.get('valid_until') and calendar.sessions[t].prediction_at > event['valid_until']:
                    continue
                source.append(i); target.append(t * company_count + j); ages.append(t - first)
    source, target, ages = map(np.asarray, (source, target, ages))
    order = np.argsort(target, kind='stable')
    counts = np.bincount(target.astype(np.int64), minlength=day_count * company_count)
    indptr = np.concatenate(([0], np.cumsum(counts)))
    np.savez(output / filename, embeddings=np.asarray(embedding, dtype=np.float32),
             categories=np.asarray(category, dtype=np.int64), source=source[order].astype(np.int64),
             age=ages[order].astype(np.float32), indptr=indptr, primary_discloser=np.asarray(primary),
             category_names=np.array(SCALE_CATEGORIES))
    write_json(output / (Path(filename).stem + '_manifest.json'), {'events': len(events), 'edges': len(source),
               'event_sha256': sha256_file(expanded / 'events.jsonl'),
               'arrays_sha256': sha256_file(output / filename), 'encoder': manifest['encoder'],
               'routing': 'Named source participants only, then one peer/sector hop; no inferred direct peer exposures'})


class PanelBatches:
    """Date-local computations keep minibatch memory independent of study length."""
    def __init__(self, root, train_end, validation_end, test_end, device='cuda', lookback=30):
        self.root, self.device = Path(root), device
        panel = np.load(self.root / 'panel.npz')
        self.raw, self.dates = panel['raw'], panel['dates']
        self.companies = panel['companies']
        self.n, self.c = self.raw.shape[:2]
        self.lookback = lookback
        fit = self.dates <= train_end
        features = panel['features'].copy()
        mean, scale = features[fit, :, 5:].mean((0, 1)), features[fit, :, 5:].std((0, 1))
        scale = np.maximum(scale, 1e-6)
        features[:, :, 5:] = ((features[:, :, 5:] - mean) / scale).clip(-10, 10)
        self.features = torch.as_tensor(features, device=device)
        self.peer = torch.as_tensor(panel['peers'], device=device)
        self.peer_weight = torch.as_tensor(panel['peer_weight'], device=device)
        self.sectors = torch.as_tensor(panel['sectors'], device=device)
        self.volume_mean, self.volume_scale = panel['volume_mean'], panel['volume_scale']
        self.log_targets = np.log(self.raw[1:, :, :4] / self.raw[:-1, :, 3:4])
        self.volume_targets = (np.log1p(self.raw[1:, :, 4]) - self.volume_mean[:-1]) / self.volume_scale[:-1]
        ids = np.arange(lookback - 1, self.n - 1)
        self.splits = {
            'train': ids[(self.dates[ids] >= '2014-01-01') & (self.dates[ids + 1] <= train_end)],
            'validation': ids[(self.dates[ids] > train_end) & (self.dates[ids + 1] <= validation_end)],
            'test': ids[(self.dates[ids] > validation_end) & (self.dates[ids + 1] <= test_end)],
        }
        if any(len(v) == 0 for v in self.splits.values()):
            raise ValueError('Empty chronological split')
        y = self.log_targets[self.splits['train']]
        upper = y[:, :, 1] - np.maximum(y[:, :, 0], y[:, :, 3])
        lower = np.minimum(y[:, :, 0], y[:, :, 3]) - y[:, :, 2]
        self.scaler = {'target_mean': y.mean((0, 1)).tolist(), 'target_scale': y.std((0, 1)).tolist(),
                       'geometry_scales': [float(upper.mean()), float(lower.mean())],
                       'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(),
                       'volume_target_scale': float(self.volume_targets[self.splits['train']].std()),
                       'fit_latest_date': str(self.dates[self.splits['train'][-1] + 1])}
        path = self.root / 'event_arrays.npz'
        if path.exists():
            e = np.load(path)
            self.events = torch.as_tensor(e['embeddings'], device=device)
            self.default_events = self.events
            alternate = self.root / 'event_arrays_finbert.npz'
            self.finbert_events = None
            if alternate.exists():
                alt = np.load(alternate)
                if not all(np.array_equal(e[key], alt[key]) for key in ('categories', 'source', 'age', 'indptr', 'primary_discloser')):
                    raise ValueError('FinBERT comparison must retain identical event ordering and graph edges')
                self.finbert_events = torch.as_tensor(alt['embeddings'], device=device)
            self.categories = torch.as_tensor(e['categories'], device=device)
            self.category_count = len(e['category_names']) if 'category_names' in e.files else 6
            self.primary_discloser = torch.as_tensor(e['primary_discloser'], device=device)
            self.indptr, self.source, self.age = e['indptr'], e['source'], e['age']
        else:
            raise ValueError('Complete event arrays before training')

    def select_encoder(self, variant):
        self.lookback = 20 if variant == 'combined' else 30
        if variant in ('finbert', 'combined'):
            if self.finbert_events is None or self.finbert_events.shape != self.default_events.shape:
                raise ValueError('Matched frozen FinBERT event vectors are missing')
            self.events = self.finbert_events
        else:
            self.events = self.default_events

    def batch(self, dates):
        dates = np.asarray(dates, dtype=np.int64)
        history = dates[:, None] + np.arange(1 - self.lookback, 1)[None, :]
        index = torch.as_tensor(history, device=self.device)
        # B x lookback x companies x features; recurrent model reorders axes.
        x = self.features[index]
        rows = (dates[:, None] * self.c + np.arange(self.c)[None, :]).reshape(-1)
        lengths = self.indptr[rows + 1] - self.indptr[rows]
        selected = np.concatenate([np.arange(self.indptr[r], self.indptr[r + 1]) for r in rows])
        target = np.repeat(np.arange(len(rows)), lengths)
        output = {'x': x, 'peers': self.peer[torch.as_tensor(dates, device=self.device)],
                  'peer_weight': self.peer_weight[torch.as_tensor(dates, device=self.device)],
                  'history_peers': self.peer[index], 'history_peer_weight': self.peer_weight[index],
                  'sectors': self.sectors, 'events': self.events, 'categories': self.categories,
                  'primary_discloser': self.primary_discloser,
                  'edge_source': torch.as_tensor(self.source[selected], device=self.device),
                  'edge_target': torch.as_tensor(target, device=self.device),
                  'edge_age': torch.as_tensor(self.age[selected], device=self.device),
                  'target': torch.as_tensor(self.log_targets[dates].reshape(-1, 4), dtype=torch.float32, device=self.device),
                  'volume_target': torch.as_tensor(self.volume_targets[dates].reshape(-1), dtype=torch.float32, device=self.device)}
        return output
