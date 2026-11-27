import copy
import unittest
import numpy as np
import torch

from stak_graph.models.finstak import NullAttention, FinSTAK
from stak_graph.experiments.finstak_data import rolling_ohlcv, attach_history, tolerance_accuracy
from stak_graph.graph.market import Bar


class FinSTAKTest(unittest.TestCase):
    def test_single_event_decays_against_null_and_learns_rate(self):
        layer = NullAttention(hidden=4, heads=1, relations=1)
        with torch.no_grad():
            layer.query.weight.zero_()
            layer.relation_bias.weight.zero_()
            layer.null.weight.zero_(); layer.null.bias.zero_()
            layer.value.fill_(1.)
        q, v = torch.zeros(1, 4), torch.ones(1, 4)
        index, weight = torch.tensor([0]), torch.ones(1)
        fresh = layer(q, v, index, index, index, torch.tensor([0.]), weight)
        old = layer(q, v, index, index, index, torch.tensor([100.]), weight)
        self.assertTrue((old.abs() < fresh.abs()).all())
        self.assertLess(float(old.abs().max().detach()), 1e-4)
        old.sum().backward()
        self.assertTrue(torch.isfinite(layer.rate.grad).all())
        self.assertTrue((layer.rate.grad.abs() > 0).all())

    def test_masked_future_and_zero_weight_give_exact_zero(self):
        layer = NullAttention(hidden=4, heads=1, relations=1)
        q, v = torch.randn(3, 4), torch.randn(2, 4)
        out = layer(q, v, torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 0]),
                    torch.tensor([-1., 0.]), torch.tensor([1., 0.]))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        out.sum().backward()
        self.assertTrue(torch.isfinite(layer.key.grad).all())

    def fixture(self):
        torch.manual_seed(7)
        days = [{'company_id': c, 'session_index': t, 'session_date': str(t)} for c in ('A', 'B') for t in range(4)]
        snapshot = {'nodes': {'stock_day': days}}
        data = {'prices': torch.randn(8, 19), 'company_index': torch.tensor([0] * 4 + [1] * 4),
                'events': torch.randn(1, 6), 'categories': torch.tensor([0]),
                'edge_event': torch.tensor([0]), 'edge_day': torch.tensor([2]),
                'edge_relation': torch.tensor([0]), 'edge_age': torch.tensor([0]), 'edge_prior': torch.tensor([1.]),
                'corr_source': torch.tensor([2]), 'corr_target': torch.tensor([6]), 'corr_weight': torch.tensor([.8])}
        attach_history(data, snapshot, 3)
        scaler = {'target_mean': [0, .01, -.01, 0], 'target_scale': [.02] * 4, 'geometry_scales': [.005, .006]}
        model = FinSTAK(scaler, event_dimension=6, companies=2, hidden=8)
        return model, data

    def test_exact_local_fallback_and_event_two_hop_path(self):
        model, data = self.fixture()
        model.mode = 'no_events'
        local = model.encode_days(data)
        model.mode = 'events'
        full = model.encode_days(data)
        # The event only touches A at t=2; one peer hop reaches B at t=2.
        self.assertGreater(float((full[2] - local[2]).abs().max().detach()), 0)
        self.assertGreater(float((full[6] - local[6]).abs().max().detach()), 0)
        self.assertTrue(torch.equal(full[[0, 1, 3, 4, 5, 7]], local[[0, 1, 3, 4, 5, 7]]))
        empty = dict(data)
        for key in ('edge_event', 'edge_day', 'edge_relation', 'edge_age', 'edge_prior'):
            empty[key] = data[key][:0]
        self.assertTrue(torch.equal(model.encode_days(empty), local))

    def test_causal_histories_and_all_modes_train_valid_candles(self):
        model, data = self.fixture()
        modified = copy.deepcopy(data)
        modified['prices'][[3, 7]] += 100
        original = model.encode_days(data)
        updated = model.encode_days(modified)
        self.assertTrue(torch.equal(original[[0, 1, 2, 4, 5, 6]], updated[[0, 1, 2, 4, 5, 6]]))
        windows = torch.tensor([[0, 1, 2], [4, 5, 6]])
        for mode in ('events', 'no_events', 'no_decay', 'lstm', 'tgcn'):
            test = FinSTAK({'target_mean': [0, .01, -.01, 0], 'target_scale': [.02] * 4,
                           'geometry_scales': [.005, .006]}, event_dimension=6, companies=2, hidden=8, mode=mode)
            output = test(data, windows)
            self.assertEqual(tuple(output.shape), (2, 4))
            self.assertTrue((output[:, 1] >= output[:, [0, 3]].max(1).values).all())
            self.assertTrue((output[:, 2] <= output[:, [0, 3]].min(1).values).all())
            output.square().mean().backward()
            for parameter in test.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_rolling_shared_price_scale_flat_volume_and_future_invariance(self):
        dates = [f'2020-01-0{i + 1}' for i in range(5)]
        bars = [Bar('A', dates[i], 100 + i, 103 + i, 99 + i, 102 + i, 1000, dates[i] + 'T21:15:00Z', True) for i in range(5)]
        snapshot = {'nodes': {'stock_day': [{'company_id': 'A', 'session_index': i, 'session_date': dates[i]} for i in range(5)]}}
        first, counts = rolling_ohlcv(snapshot, bars, window=3)
        changed = bars[:4] + [Bar('A', dates[4], 900, 930, 890, 920, 9000, dates[4] + 'T21:15:00Z', True)]
        second, _ = rolling_ohlcv(snapshot, changed, window=3)
        np.testing.assert_array_equal(first[:4], second[:4])
        np.testing.assert_array_equal(counts, [1, 2, 3, 3, 3])
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((first[:, 1] >= first[:, [0, 3]].max(1)).all())
        self.assertTrue((first[:, 2] <= first[:, [0, 3]].min(1)).all())
        self.assertTrue((first[:, 4] == 0).all())

    def test_tolerance_accuracy_is_not_one_minus_mae(self):
        actual = np.full((2, 4), 100.)
        predicted = actual + np.array([[0, .5, 1, 2], [2, 2, 2, 2]])
        result = tolerance_accuracy(actual, predicted, np.full(2, 100.), [.01])['1%']
        self.assertEqual(result['per_feature_percent']['open'], 50.)
        self.assertEqual(result['pooled_fields_percent'], 37.5)
        self.assertEqual(result['whole_candle_percent'], 0.)
        with self.assertRaises(ValueError):
            tolerance_accuracy(actual, predicted, np.array([0, 100.]))


if __name__ == '__main__':
    unittest.main()
