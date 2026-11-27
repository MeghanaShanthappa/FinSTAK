import unittest
import numpy as np
import torch
from stak_graph.experiments.timesfm_features import context_rows, event_covariates, project_candle
from stak_graph.models.timesfm_hybrid import TimesFMFinSTAK
from stak_graph.models.timesfm_encoder import TimesFMEncoderFinSTAK


class TimesFMTests(unittest.TestCase):
    def test_context_never_crosses_stock_or_forecast_cutoff(self):
        days = [{'company_id': c, 'session_index': t, 'session_date': str(t)}
                for c in ('A', 'B') for t in range(6)]
        rows = context_rows(days, [4, 10], 3)
        np.testing.assert_array_equal(rows, [[2, 3, 4], [8, 9, 10]])
        with self.assertRaises(ValueError):
            context_rows(days, [1], 3)
        short = context_rows(days, [1, 7], 3, allow_short=True)
        np.testing.assert_array_equal(short, [[0, 1], [6, 7]])
        gap = [d for d in days if not (d['company_id'] == 'A' and d['session_index'] == 3)]
        with self.assertRaises(ValueError):
            context_rows(gap, [3], 3, allow_short=True)

    def test_projection_preserves_open_close(self):
        raw = np.array([[.1, -.2, .5, -.1], [.2, .3, -.2, .1]])
        projected = project_candle(raw)
        np.testing.assert_array_equal(projected[:, [0, 3]], raw[:, [0, 3]])
        self.assertTrue((projected[:, 1] >= projected[:, [0, 3]].max(1)).all())
        self.assertTrue((projected[:, 2] <= projected[:, [0, 3]].min(1)).all())

    def test_future_events_cannot_change_past_features_or_pca(self):
        torch.manual_seed(2)
        days = [{'session_date': '2017-01-01'} for _ in range(5)] + [{'session_date': '2019-01-01'}]
        data = {'edge_event': torch.arange(6), 'edge_day': torch.arange(6), 'edge_age': torch.zeros(6),
                'edge_prior': torch.ones(6), 'events': torch.randn(6, 8), 'categories': torch.arange(6),
                'corr_source': torch.tensor([0]), 'corr_target': torch.tensor([1]), 'corr_weight': torch.ones(1)}
        direct, peer, audit = event_covariates(data, days, '2017-12-31', components=2)
        altered = dict(data)
        altered['events'] = data['events'].clone()
        altered['events'][5] += 1000
        newer, newer_peer, new_audit = event_covariates(altered, days, '2017-12-31', components=2)
        np.testing.assert_array_equal(direct[:5], newer[:5])
        np.testing.assert_array_equal(peer[:5], newer_peer[:5])
        self.assertEqual(audit, new_audit)

    def test_hybrid_validity_and_isolated_geometry_gradients(self):
        scaler = {'target_mean': [0., .01, -.01, 0.], 'target_scale': [.01] * 4,
                  'geometry_scales': [.01, .01]}
        model = TimesFMFinSTAK(scaler, companies=1)
        model.forecast_prior = torch.tensor([[.001, .02, -.02, -.001]])
        # Directly exercise the decoder using an identity head.
        model.head = torch.nn.Identity()
        delta = torch.zeros((1, 4), requires_grad=True)
        result = model.predict_windows(delta, torch.tensor([[0]]))
        np.testing.assert_allclose(result.detach(), model.forecast_prior, atol=1e-7)
        result[:, 1].sum().backward()
        self.assertEqual(delta.grad[0, 0].item(), 0)
        self.assertEqual(delta.grad[0, 3].item(), 0)
        self.assertGreater(delta.grad[0, 1].item(), 0)
        extreme = torch.tensor([[20., -20., -20., -20.]])
        candle = model.predict_windows(extreme, torch.tensor([[0]]))
        self.assertTrue(torch.isfinite(candle).all())
        self.assertTrue((candle[:, 1] >= candle[:, [0, 3]].max(1).values).all())
        self.assertTrue((candle[:, 2] <= candle[:, [0, 3]].min(1).values).all())
        model.forecast_prior = torch.zeros(1, 4)
        cold = torch.zeros(1, 4, requires_grad=True)
        model.predict_windows(cold, torch.tensor([[0]]))[:, 1].sum().backward()
        self.assertGreater(cold.grad[0, 1].item(), 0, 'Zero native wick must be learnable')

    def test_inference_helper_initializes_prior_without_forward(self):
        scaler = {'target_mean': [0., .01, -.01, 0.], 'target_scale': [.01] * 4,
                  'geometry_scales': [.01, .01]}
        model = TimesFMFinSTAK(scaler, companies=1, mode='local')
        data = {'forecast_prior': torch.tensor([[0., .02, -.02, .001]]),
                'prices': torch.zeros(1, 23), 'company_index': torch.tensor([0]),
                'local_windows': torch.tensor([[0]]), 'local_mask': torch.ones(1, 1)}
        from stak_graph.experiments.train import predict
        result = predict(model, data, torch.tensor([[0]]), torch.tensor([0]))
        np.testing.assert_allclose(result, data['forecast_prior'], atol=1e-7)

    def test_frozen_encoder_no_event_fallback_and_training_gradient(self):
        scaler = {'target_mean': [0., .01, -.01, 0.], 'target_scale': [.01] * 4,
                  'geometry_scales': [.01, .01]}
        empty = torch.empty(0, dtype=torch.long)
        data = {'prices': torch.randn(2, 19), 'temporal_embeddings': torch.randn(2, 32),
                'company_index': torch.zeros(2, dtype=torch.long), 'events': torch.empty(0, 768),
                'categories': empty, 'edge_event': empty, 'edge_day': empty,
                'edge_relation': empty, 'edge_age': torch.empty(0), 'edge_prior': torch.empty(0),
                'corr_source': torch.tensor([0]), 'corr_target': torch.tensor([1]), 'corr_weight': torch.ones(1)}
        local = TimesFMEncoderFinSTAK(scaler, companies=1, mode='local')
        windows = torch.tensor([[0], [1]])
        expected = local(data, windows)
        for mode in ('direct', 'graph', 'no_decay'):
            model = TimesFMEncoderFinSTAK(scaler, companies=1, mode=mode)
            model.load_state_dict(local.state_dict())
            torch.testing.assert_close(model(data, windows), expected)
        expected.square().sum().backward()
        self.assertTrue(torch.isfinite(local.foundation_projection[-1].weight.grad).all())


if __name__ == '__main__':
    unittest.main()
