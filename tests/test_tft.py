"""Check TFT structure, actual gradients, causal attention, and same-day timing."""

import copy
from datetime import datetime, timedelta, timezone
import unittest

from stak_graph.experiments.data import same_day_availability
from stak_graph.graph.market import Session, TradingCalendar

try:
    import torch
    from stak_graph.experiments.train import EXCLUDED_MODELS, VARIANTS, make_model, multitask_loss
    from stak_graph.models.tft import InterpretableAttention, TemporalFusionEncoder, VariableSelection
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class SameDayTimingTests(unittest.TestCase):
    def test_date_only_uses_end_of_same_recorded_day(self):
        known, precision = same_day_availability("2019-01-02T00:00:00Z")
        self.assertEqual(known, "2019-01-02T23:59:59Z")
        self.assertEqual(precision, "date_only_assumed_end_of_day")

    def test_reported_clock_is_preserved(self):
        known, precision = same_day_availability("2019-01-02T21:30:00Z")
        self.assertEqual(known, "2019-01-02T21:30:00Z")
        self.assertEqual(precision, "reported_time_unverified")

    def calendar(self):
        return TradingCalendar([Session(day, day+"T14:30:00Z", day+"T21:00:00Z", day+"T23:59:59Z")
                                for day in ("2019-01-04", "2019-01-07", "2019-01-08")])

    def test_after_close_headline_can_reach_same_day_eod_forecast(self):
        self.assertEqual(self.calendar().first_eligible("2019-01-04T22:30:00Z"), 0)

    def test_next_date_headline_does_not_reach_previous_forecast(self):
        self.assertEqual(self.calendar().first_eligible("2019-01-05T00:01:00Z"), 1)

    def test_weekend_headline_uses_next_supplied_trading_decision(self):
        known, _ = same_day_availability("2019-01-05T00:00:00Z")
        self.assertEqual(self.calendar().first_eligible(known), 1)


@unittest.skipUnless(ML_AVAILABLE, "TFT tests require torch")
class TemporalFusionTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(10)

    def data(self):
        return {"prices": torch.randn(8, 3), "events": torch.randn(3, 4), "categories": torch.tensor([0, 1, 2]),
                "company_index": torch.tensor([0]*4+[1]*4), "edge_event": torch.tensor([0, 1, 2]),
                "edge_day": torch.tensor([1, 5, 7]), "edge_relation": torch.tensor([0, 0, 1]),
                "edge_age": torch.tensor([0, 2, 0]), "edge_weight": torch.tensor([1., .5, .25]),
                "corr_source": torch.tensor([1, 5]), "corr_target": torch.tensor([5, 1]), "corr_weight": torch.tensor([.8, .8])}

    def test_attention_masks_every_future_position(self):
        layer = InterpretableAttention(8, heads=2, dropout=0).eval()
        values = torch.randn(2, 7, 8)
        output, weights = layer(values)
        self.assertEqual(output.shape, (2, 7, 8))
        self.assertEqual(torch.count_nonzero(weights.triu(1)).item(), 0)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 7))

    def test_later_values_cannot_change_earlier_attention_outputs(self):
        layer = InterpretableAttention(8, heads=2, dropout=0).eval()
        values = torch.randn(2, 7, 8)
        before = layer(values)[0][:, :3]
        values[:, 3:] = 1e4
        torch.testing.assert_close(before, layer(values)[0][:, :3], rtol=0, atol=0)

    def test_interpretable_attention_shares_one_value_projection(self):
        layer = InterpretableAttention(12, heads=3, dropout=0)
        self.assertEqual(layer.value.weight.shape, (4, 12))
        self.assertEqual(layer.query.weight.shape, (12, 12))

    def test_variable_weights_form_a_distribution(self):
        selector = VariableSelection([1, 1, 8], hidden=8, dropout=0).eval()
        output, weights = selector([torch.randn(2, 5, 1), torch.randn(2, 5, 1), torch.randn(2, 5, 8)], torch.randn(2, 8))
        self.assertEqual(output.shape, (2, 5, 8))
        self.assertTrue(bool((weights >= 0).all()))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 5))

    def test_full_tft_returns_five_forecast_states_and_diagnostics(self):
        tft = TemporalFusionEncoder(3, hidden=8, horizon=5, heads=2, dropout=0).eval()
        states, info = tft(torch.randn(2, 6, 3), torch.randn(2, 6, 8), torch.randn(2, 6, 8), torch.randn(2, 8), diagnostics=True)
        self.assertEqual(states.shape, (2, 5, 8))
        self.assertEqual(info["historical_variable_weights"].shape, (2, 6, 5))
        self.assertEqual(info["attention_weights"].shape, (2, 11, 11))
        self.assertEqual(tft.future_positions.shape, (1, 5, 1))

    def test_future_decoder_positions_do_not_change_first_horizon(self):
        tft = TemporalFusionEncoder(3, hidden=8, horizon=5, heads=2, dropout=0).eval()
        args = [torch.randn(2, 6, 3), torch.randn(2, 6, 8), torch.randn(2, 6, 8), torch.randn(2, 8)]
        before = tft(*args)[:, 0]
        tft.future_positions[:, 1:] = 100
        torch.testing.assert_close(before, tft(*args)[:, 0], rtol=0, atol=0)

    def test_factory_uses_lstm_and_attention_for_tft(self):
        model = make_model("event_graph_correlation_tft", 3, 4, 8, 2)
        self.assertIsInstance(model.tft.encoder, torch.nn.LSTM)
        self.assertIsInstance(model.tft.decoder, torch.nn.LSTM)
        self.assertFalse(hasattr(model, "temporal"))

    def test_real_loss_reaches_graph_tft_and_static_parameters(self):
        model = make_model("event_graph_correlation_tft", 3, 4, 8, 2)
        target = torch.tensor([[.3, 1., -.2], [-.4, 0., .5]])
        loss = multitask_loss(model(self.data(), torch.tensor([[0, 1], [4, 5]])), target)
        loss.backward()
        for parameter in (model.tft.encoder.weight_ih_l0, model.tft.decoder.weight_ih_l0,
                          model.tft.attention.query.weight, model.tft.past_selection.selection.input.weight,
                          model.event_encoder[0].weight, model.correlation_message.weight, model.company_embedding.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_future_graph_information_cannot_change_earlier_tft_forecast(self):
        model = make_model("event_graph_correlation_tft", 3, 4, 8, 2).eval()
        data = self.data(); windows = torch.tensor([[0, 1], [4, 5]])
        before = model(data, windows)
        data["events"][2] = 1000
        data["prices"][[2, 3, 6, 7]] = 1000
        torch.testing.assert_close(before, model(data, windows), rtol=0, atol=0)

    def test_excluded_baselines_are_not_training_conditions(self):
        self.assertFalse(set(EXCLUDED_MODELS) & set(VARIANTS))
        self.assertEqual(len(VARIANTS), 10)
        self.assertEqual(sum(name.endswith("_tft") for name in VARIANTS), 5)


if __name__ == "__main__": unittest.main()
