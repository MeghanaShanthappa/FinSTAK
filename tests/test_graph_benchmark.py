import unittest
import torch
from stak_graph.models.benchmark import GraphEncoder, ForecastAdapter, EXTERNAL


class CausalGraphBenchmarkTest(unittest.TestCase):
    @unittest.skipUnless((EXTERNAL / 'tslib').exists(), 'Optional upstream repositories are not installed')
    def test_upstream_models_accept_trainable_graph_outputs(self):
        for name in ('xlstm', 'patchtst', 'itransformer', 'timemixer', 'timemachine'):
            with self.subTest(model=name):
                model = ForecastAdapter('gat', name)
                daily = torch.randn(60, 32, requires_grad=True)
                output = model.predict_windows(daily, torch.arange(30).repeat(2, 1))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(daily.grad).all())
                self.assertGreater(float(daily.grad.abs().sum()), 0)

    def data(self):
        # Stock days 0→1→2; event 3 arrives on day 1, event 4 on day 2.
        return {'prices': torch.randn(3, 18), 'company_index': torch.zeros(3, dtype=torch.long),
                'events': torch.randn(2, 768), 'categories': torch.tensor([0, 1]),
                'graph_source': torch.tensor([0, 1, 2, 3, 4, 0, 1, 3, 4]),
                'graph_target': torch.tensor([0, 1, 2, 3, 4, 1, 2, 1, 2]),
                'graph_relation': torch.tensor([3, 3, 3, 3, 3, 2, 2, 0, 0]),
                'graph_weight': torch.ones(9)}

    def test_future_stock_and_event_cannot_change_past_states(self):
        for kind in ('typed', 'gat', 'sagn'):
            with self.subTest(kind=kind):
                torch.manual_seed(3)
                data = self.data()
                model = GraphEncoder(18, 768, 16, 1, kind).eval()
                changed = {key: value.clone() for key, value in data.items()}
                changed['prices'][2] = 1000
                changed['events'][1] = -1000
                with torch.no_grad():
                    reference = model(data)
                    perturbed = model(changed)
                torch.testing.assert_close(reference[:2], perturbed[:2], atol=0, rtol=0)
                self.assertFalse(torch.allclose(reference[2], perturbed[2]))

    def test_graph_receives_gradients(self):
        for kind in ('gat', 'sagn'):
            model = GraphEncoder(18, 768, 16, 1, kind)
            model(self.data()).square().sum().backward()
            self.assertTrue(torch.isfinite(model.event.weight.grad).all())
            self.assertGreater(float(model.event.weight.grad.abs().sum()), 0)

    def test_removing_event_edges_removes_event_influence(self):
        from stak_graph.experiments.model_sweep import no_events
        for kind in ('gat', 'sagn'):
            data = self.data()
            selected = data['graph_relation'] != 0
            for key in ('graph_source', 'graph_target', 'graph_relation', 'graph_weight'):
                data[key] = data[key][selected]
            changed = {key: value.clone() for key, value in data.items()}
            changed['events'] += 1000
            model = GraphEncoder(18, 768, 16, 1, kind).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(data), model(changed), atol=0, rtol=0)
