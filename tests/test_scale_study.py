import unittest
import torch

from stak_graph.experiments.scale_data import asserted_categories, company_quote, source_time, supported_assertions
from stak_graph.models.scale_finstak import HybridNullAttention, ScaleFinSTAK, pcgrad_backward


class ScaleStudyTests(unittest.TestCase):
    def fixture(self, variant):
        torch.manual_seed(17)
        c, b, t, h = 3, 2, 5, 8
        data = {'x': torch.randn(b, t, c, 19), 'events': torch.randn(2, 6),
                'categories': torch.tensor([0, 1]), 'edge_source': torch.tensor([0, 1]),
                'edge_target': torch.tensor([0, 3]), 'edge_age': torch.tensor([0., 4.]),
                'peers': torch.tensor([[[1, 2, 0], [0, 2, 1], [0, 1, 2]]] * b),
                'peer_weight': torch.ones(b, c, 3) * .5,
                'history_peers': torch.tensor([[[[1, 2, 0], [0, 2, 1], [0, 1, 2]]] * t] * b),
                'history_peer_weight': torch.ones(b, t, c, 3) * .5,
                'sectors': torch.tensor([0, 0, 1]), 'primary_discloser': torch.tensor([True, False])}
        scaler = {'target_mean': [0., .01, -.01, 0.], 'target_scale': [.02] * 4, 'geometry_scales': [.005, .006]}
        return ScaleFinSTAK(scaler, c, event_dimension=6, variant=variant, hidden=h), data

    def test_all_variants_have_valid_candles_and_gradients(self):
        from stak_graph.models.scale_finstak import VARIANTS, VOLUME_VARIANTS
        for variant in VARIANTS:
            model, data = self.fixture(variant)
            price, volume = model(data)
            self.assertTrue(torch.isfinite(price).all())
            self.assertTrue((price[:, 1] >= price[:, [0, 3]].max(1).values).all())
            self.assertTrue((price[:, 2] <= price[:, [0, 3]].min(1).values).all())
            self.assertEqual(volume is not None, variant in VOLUME_VARIANTS)
            (price.square().sum() + (volume.square().sum() if volume is not None else 0)).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_isolation_blocks_wick_gradient_from_body_and_shared_trunk(self):
        for variant in ('isolated', 'combined'):
            model, data = self.fixture(variant)
            price, _ = model(data)
            price[:, [1, 2]].sum().backward()
            self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.trunk.parameters()))
            self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.head.parameters()))
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.wick_head.parameters()))

    def test_dual_stream_has_disjoint_parameters(self):
        model, data = self.fixture('dual_stream')
        price, volume = model(data)
        volume.sum().backward()
        self.assertTrue(all(p.grad is None for p in model.trunk.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.dispersion.parameters()))

    def test_no_events_gives_exact_local_fallback_in_both_topologies(self):
        for variant in ('events', 'sector', 'combined'):
            model, data = self.fixture(variant)
            for key in ('edge_source', 'edge_target', 'edge_age'):
                data[key] = data[key][:0]
            full = model.trunk(data)
            model.trunk.variant = 'no_events'
            self.assertTrue(torch.equal(full, model.trunk(data)))

    def test_date_batches_are_independent(self):
        model, data = self.fixture('sector')
        before = model(data)[0]
        data['x'][1] += 100
        after = model(data)[0]
        self.assertTrue(torch.equal(before[:3], after[:3]))

    def test_hybrid_sink_suppresses_stale_singleton(self):
        torch.manual_seed(1)
        layer = HybridNullAttention(hidden=4, heads=1, relations=1)
        q, v, index = torch.zeros(1, 4), torch.ones(1, 4), torch.tensor([0])
        def output(age):
            return layer(q, v, index, index, index, torch.tensor([age]), torch.ones(1))
        self.assertTrue((output(100.).abs() < output(0.).abs()).all())
        self.assertTrue(torch.equal(output(-1.), torch.zeros(1, 4)))

    def test_pcgrad_projects_opposing_shared_gradients(self):
        shared = torch.nn.Parameter(torch.tensor([1., 1.]))
        private = torch.nn.Parameter(torch.tensor([2.]))
        a = shared[0] + private.sum()
        b = -shared[0] + shared[1]
        pcgrad_backward([a, b], [shared, private], torch.Generator().manual_seed(7))
        self.assertTrue(torch.allclose(shared.grad, torch.tensor([.25, .75])))
        self.assertTrue(torch.allclose(private.grad, torch.ones(1)))

    def test_source_checks_reject_speculation_and_ambiguous_ticker(self):
        self.assertEqual(asserted_categories('Apple could acquire Example'), [])
        self.assertIn('earnings', asserted_categories('Apple reports quarterly earnings'))
        self.assertIn('guidance_change', asserted_categories('Apple raises earnings guidance'))
        self.assertIsNone(company_quote('A company reports earnings', {'name': 'Agilent', 'aliases': [], 'company_id': 'A'}))
        self.assertEqual(source_time('2020-01-01 12:30:00 UTC'), ('2020-01-01T12:30:00Z', '2020-01-03T12:30:00Z'))

    def test_body_assertion_retains_exact_source_span(self):
        body = 'Market commentary. Apple reports quarterly earnings of $2 per share. Apple could acquire Example.'
        results = list(supported_assertions('Market update', body, {'name': 'Apple', 'aliases': [], 'company_id': 'AAPL'}))
        self.assertEqual(len(results), 1)
        category, text, company, span = results[0]
        self.assertEqual(category, 'earnings')
        self.assertEqual(body[span['start']:span['end']], text)
        self.assertEqual(company, 'Apple')

    def test_recurrent_headlines_from_different_years_remain_distinct_sources(self):
        import csv, hashlib, json, sqlite3, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from stak_graph.experiments.scale_data import scan_news, NEWS_ARTIFACTS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'test_news.csv'
            fields = ['Date', 'Article_title', 'Stock_symbol', 'Url', 'Article']
            with source.open('w', newline='') as out:
                writer = csv.DictWriter(out, fields)
                writer.writeheader()
                for day, url in [('2014-01-01', 'https://example.com/2014'), ('2023-01-01', 'https://example.com/2023'),
                                 ('2014-01-02', 'https://example.com/2014')]:
                    writer.writerow({'Date': day + ' 12:30:00 UTC', 'Article_title': 'Apple reports quarterly earnings',
                                     'Stock_symbol': 'AAPL', 'Url': url, 'Article': ''})
            config = {'sample_start': '2014-01-01', 'sample_end': '2023-12-31',
                      'companies': [{'name': 'Apple', 'aliases': [], 'company_id': 'AAPL'}]}
            with sqlite3.connect(':memory:') as db:
                db.execute('CREATE TABLE candidates(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
                db.execute('CREATE TABLE scanned(name TEXT PRIMARY KEY,payload TEXT NOT NULL)')
                with patch.dict(NEWS_ARTIFACTS, {source.name: (source.stat().st_size, hashlib.sha256(source.read_bytes()).hexdigest())}):
                    scan_news(source, config, root, db)
                values = [json.loads(r[0]) for r in db.execute('SELECT payload FROM candidates')]
                self.assertEqual(len(values), 2)
                self.assertEqual(sorted(v['known_at'] for v in values), ['2014-01-04T12:30:00Z', '2023-01-03T12:30:00Z'])

    def test_company_abbreviations_and_decimals_do_not_break_evidence(self):
        body = 'Apple Inc. reports quarterly earnings of $2.35 per share. Markets remained quiet.'
        values = list(supported_assertions('Market update', body, {'name': 'Apple', 'aliases': [], 'company_id': 'AAPL'}))
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][1], 'Apple Inc. reports quarterly earnings of $2.35 per share.')

    def test_sec_items_assert_only_supported_declared_categories(self):
        from stak_graph.acquisition.sec_items import events_from_filing
        filing = {'ticker': 'AAPL', 'cik': '320193', 'accessionNumber': '0000320193-20-000001',
                  'form': '8-K', 'primaryDocument': 'report.htm', 'filingDate': '2020-01-02',
                  'known_at': '2020-01-02T23:59:59Z', 'items': '2.02,2.03,3.02,7.01,8.01,9.01'}
        events = events_from_filing(filing)
        self.assertEqual({e['category'] for e in events}, {'earnings', 'financing'})
        finance = next(e for e in events if e['category'] == 'financing')
        self.assertEqual(finance['reported_items'], ['2.03', '3.02'])
        self.assertFalse(finance['full_document_extracted'])
        self.assertEqual(finance['known_at'], filing['known_at'])
        self.assertEqual(events_from_filing(dict(filing, items='7.01,8.01,9.01')), [])

    def test_resumable_status_updates_preserve_last_valid_record_on_failure(self):
        import json, tempfile
        from pathlib import Path
        from stak_graph.experiments.scale_io import write_json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'status.json'
            write_json(path, {'state': 'waiting'})
            write_json(path, {'state': 'running'})
            with self.assertRaises(ValueError):
                write_json(path, {'invalid': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'state': 'running'})
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def test_sec_availability_never_precedes_acceptance_on_next_utc_date(self):
        from stak_graph.acquisition.public import known_at
        self.assertEqual(known_at({'filingDate': '2020-07-30', 'acceptanceDateTime': '2020-07-31T00:44:46.000Z'}),
                         '2020-07-31T23:59:59Z')
        self.assertEqual(known_at({'filingDate': '2020-07-31', 'acceptanceDateTime': '2020-07-30T21:00:00Z'}),
                         '2020-07-31T23:59:59Z')

    def test_explicit_dividend_declarations_are_separate_from_generic_history(self):
        self.assertIn('dividend', asserted_categories('Microsoft declares quarterly cash dividend'))
        self.assertNotIn('dividend', asserted_categories('Microsoft pays a dividend'))
        self.assertNotIn('dividend', asserted_categories('Microsoft is expected to announce a dividend'))
        self.assertNotIn('dividend', asserted_categories('Microsoft increased its dividend for 20 years'))

    def test_extended_category_routes_without_changing_legacy_six_category_models(self):
        old, data = self.fixture('events')
        data['categories'][0] = 6
        for variant in ('events', 'hybrid_decay', 'combined', 'dual_stream'):
            model = ScaleFinSTAK({'target_mean': [0., .01, -.01, 0.], 'target_scale': [.02] * 4,
                                 'geometry_scales': [.005, .006]}, 3, event_dimension=6, hidden=8,
                                 category_count=7, variant=variant)
            self.assertTrue(torch.isfinite(model(data)[0]).all())
        self.assertEqual(old.trunk.category.num_embeddings, 6)

    def test_finbert_overflow_preserves_every_input_token(self):
        from stak_graph.embeddings.finbert import chunked_batch
        from pathlib import Path
        if not Path('/tmp/stak-graph-models/models--ProsusAI--finbert').exists():
            self.skipTest('Optional pinned FinBERT tokenizer is not cached')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert', revision='0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031',
                                                cache_dir='/tmp/stak-graph-models', local_files_only=True)
        texts = ['hello ' * 1100, 'short text']
        batch, owners, counts = chunked_batch(tokenizer, texts)
        self.assertLessEqual(batch['input_ids'].shape[1], 512)
        self.assertEqual(owners.tolist(), [0, 0, 0, 1])
        self.assertEqual([int(counts[owners == i].sum()) for i in range(2)], [1100, 2])

    def test_date_local_graph_matches_original_finstak_with_identical_weights(self):
        from stak_graph.models.finstak import FinSTAK
        from stak_graph.experiments.finstak_data import attach_history
        model, batch = self.fixture('events')
        scaler = {'target_mean': [0., .01, -.01, 0.], 'target_scale': [.02] * 4, 'geometry_scales': [.005, .006]}
        original = FinSTAK(scaler, event_dimension=6, companies=3, hidden=8)
        state = original.state_dict()
        for key in state:
            if key.startswith('decoder.'):
                state[key] = model.state_dict()[key]
            elif key.startswith('head.'):
                state[key] = model.state_dict()[key]
            else:
                state[key] = model.state_dict()['trunk.' + key]
        original.load_state_dict(state)
        b, t, c, _ = batch['x'].shape
        current = torch.tensor([i * t * c + (t - 1) * c + j for i in range(b) for j in range(c)])
        full = {'prices': batch['x'].reshape(-1, 19), 'company_index': torch.arange(c).repeat(b * t),
                'events': batch['events'], 'categories': batch['categories'],
                'edge_event': batch['edge_source'], 'edge_day': current[batch['edge_target']],
                'edge_relation': torch.zeros_like(batch['edge_source']), 'edge_age': batch['edge_age'],
                'edge_prior': torch.ones_like(batch['edge_age']),
                'corr_source': current[(torch.arange(b)[:, None, None] * c + batch['peers']).reshape(-1)],
                'corr_target': current.repeat_interleave(3), 'corr_weight': batch['peer_weight'].reshape(-1)}
        days = [{'company_id': str(j), 'session_index': i * t + k} for i in range(b) for k in range(t) for j in range(c)]
        attach_history(full, {'nodes': {'stock_day': days}}, t)
        expected = original(full, full['local_windows'][current])
        actual, _ = model(batch)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-7, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
