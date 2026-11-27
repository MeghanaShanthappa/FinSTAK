import copy
import csv
import hashlib
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import tempfile
import unittest

from stak_graph.experiments.data import delayed_availability, extractor_documents, select_headlines
from stak_graph.extraction.sources import Document
from stak_graph.experiments.events import project_event

try:
    import numpy as np
    import torch
    from stak_graph.experiments.dataset import build_samples, causal_replacement, correlation_edges, split_of, target_at
    from stak_graph.experiments.metrics import auc, average_rank, block_interval, evaluate, paired_comparison
    from stak_graph.experiments.train import WindowForecastModel, multitask_loss
    from stak_graph.graph.market import Bar, Session, TradingCalendar
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class RealSourceAdapterTests(unittest.TestCase):
    def test_conservative_date_availability(self):
        published, known = delayed_availability("2019-01-04 23:59:59 UTC", 2)
        self.assertEqual(published, "2019-01-04T23:59:59Z")
        self.assertEqual(known, "2019-01-06T00:00:00Z")

    def test_reject_ambiguous_date_and_zero_delay(self):
        with self.assertRaises(ValueError):
            delayed_availability("2019-01-04", 2)
        with self.assertRaises(ValueError):
            delayed_availability("2019-01-04 12:00:00 UTC", 0)

    def test_provenance_sidecar_is_not_sent_as_document_fields(self):
        row = {"document_id": "d1", "text": "Acme reported earnings.", "source_url": "https://example.com/d1",
               "published_at": "2019-01-04T00:00:00Z", "known_at": "2019-01-06T00:00:00Z",
               "source_row": 12, "source_stock_symbols": ["ACME"]}
        result = extractor_documents([row])
        self.assertNotIn("source_row", result[0])
        self.assertEqual(Document.from_dict(result[0]).text, row["text"])
        self.assertIn("source_row", row)

    def test_headline_sampling_is_order_independent_and_label_free(self):
        fields = ["Date", "Article_title", "Stock_symbol", "Url"]
        rows = [dict(zip(fields, ["2019-01-04 00:00:00 UTC", f"Acme announcement {i}", "ACME", f"https://example.com/{i}"])) for i in range(10)]
        config = {"companies": [{"company_id": "ACME", "name": "Acme", "aliases": []}],
                  "sample_start": "2019-01-01", "sample_end": "2019-12-31", "availability_delay_calendar_days": 2,
                  "headline_sampling_salt": "fixed", "headlines_per_company_month": 1}
        with tempfile.TemporaryDirectory() as temp:
            outputs = []
            for i, values in enumerate((rows, list(reversed(rows)))):
                path = Path(temp) / f"{i}.csv"
                with path.open("w", newline="") as target:
                    writer = csv.DictWriter(target, fieldnames=fields); writer.writeheader(); writer.writerows(values)
                selected, stats = select_headlines(path, config)
                outputs.append(selected[0]["document_id"])
                self.assertEqual(stats["selected_documents"], 1)
                self.assertFalse(stats["selection_uses_price_labels"])
            self.assertEqual(*outputs)


class PartialEventProjectionTests(unittest.TestCase):
    def event(self):
        text = "Acme Reports Q2 EPS $1.20, Revenue $50M"
        return {"event_id": "original", "document_id": "doc1", "source_hash": hashlib.sha256(text.encode()).hexdigest(),
                "validation_status": "needs_review", "validation_errors": [], "review_flags": ["currency_not_explicit:0", "unit_not_explicit:0"],
                "claim_status": "reported", "category": "earnings", "description": "Acme earnings in USD",
                "companies": [{"company_id": "ACME", "name": "Acme", "role": "issuer", "resolution_status": "resolved", "evidence": text}],
                "amounts": [{"value": 50, "unit": "million", "currency": "USD"}], "evidence": [text]}, {"text": text}

    def test_ambiguous_fields_are_omitted_without_relabeling_source(self):
        event, doc = self.event()
        original = copy.deepcopy(event)
        result, reason = project_event(event, doc)
        self.assertEqual(reason, "projected")
        self.assertEqual(result["description"], doc["text"])
        self.assertEqual(result["amounts"], [])
        self.assertEqual(result["source_validation_status"], "needs_review")
        self.assertFalse(result["semantic_support_verified"])
        self.assertFalse(result["human_reviewed"])
        self.assertNotEqual(result["event_id"], event["event_id"])
        self.assertEqual(event, original)

    def test_unresolved_counterparty_does_not_get_an_invented_stock_identity(self):
        event, doc = self.event()
        event["companies"].append({"name": "Unknown", "resolution_status": "unresolved", "company_id": None})
        event["review_flags"].append("company_unresolved:Unknown")
        result, _ = project_event(event, doc)
        self.assertEqual(len(result["companies"]), 1)
        self.assertEqual(result["projection_omissions"]["unresolved_participants"], ["Unknown"])

    def test_analyst_price_target_is_not_management_guidance(self):
        event, doc = self.event()
        text = "Broker Raises Acme Price Target After Strong Outlook"
        doc["text"] = text
        event.update({"category": "guidance_change", "source_hash": hashlib.sha256(text.encode()).hexdigest()})
        result, reason = project_event(event, doc)
        self.assertIsNone(result)
        self.assertEqual(reason, "category_not_supported_by_headline_surface")

    def test_source_validation_errors_are_never_bypassed(self):
        event, doc = self.event(); event["validation_errors"] = ["invented_quote"]
        self.assertIsNone(project_event(event, doc)[0])

    def test_unhandled_review_flags_are_never_bypassed(self):
        event, doc = self.event(); event["review_flags"].append("claim_uncertain")
        self.assertIsNone(project_event(event, doc)[0])

    def test_rejects_stale_source(self):
        event, doc = self.event(); doc["text"] += " updated"
        with self.assertRaises(ValueError): project_event(event, doc)


@unittest.skipUnless(ML_AVAILABLE, "Experiment tests require numpy and torch")
class TargetAndSplitTests(unittest.TestCase):
    def test_target_uses_next_open_and_matched_benchmark(self):
        class Price:
            def __init__(self, opening, close): self.open, self.close = opening, close
        prices = {("A", 1): Price(100, 101), ("A", 2): Price(102, 110), ("A", 0): Price(1, 1)}
        benchmark = {1: Price(50, 51), 2: Price(52, 52.5)}
        excess, label, log_vol, last = target_at("A", 0, prices, benchmark, horizon=2)
        self.assertAlmostEqual(excess, math.log(1.1) - math.log(1.05))
        self.assertEqual((label, last), (1, 2))
        self.assertAlmostEqual(math.exp(log_vol), math.sqrt((math.log(1.01)**2 + math.log(110/101)**2)/2))

    def test_target_requires_every_intermediate_bar(self):
        with self.assertRaises(ValueError):
            target_at("A", 0, {}, {}, 5)

    def test_split_purges_overlapping_training_labels(self):
        config = {"sample_start": "2016-01-01", "sample_end": "2019-12-31", "train_end": "2017-12-31", "validation_end": "2018-12-31"}
        self.assertEqual(split_of("2017-12-22", "2018-01-02", config), "purged")
        self.assertEqual(split_of("2018-12-28", "2019-01-07", config), "purged")
        self.assertEqual(split_of("2019-12-31", "2020-01-08", config), "test")
        self.assertEqual(split_of("2017-12-15", "2017-12-22", config), "train")

    def test_scaler_ignores_future_prices_and_labels(self):
        start = datetime(2017, 12, 20, 14, tzinfo=timezone.utc)
        calendar = TradingCalendar([Session((start + timedelta(days=i)).date().isoformat(),
                                           (start + timedelta(days=i)).isoformat(),
                                           (start + timedelta(days=i, hours=7)).isoformat(),
                                           (start + timedelta(days=i, hours=7)).isoformat()) for i in range(30)])
        bars = [Bar("A", s.session_date, 100+i, 102+i, 99+i, 101+i, 1000+i, s.close_at, False) for i, s in enumerate(calendar.sessions)]
        benchmark = [Bar("SPY", s.session_date, 200+i, 202+i, 199+i, 201+i, 1000+i, s.close_at, False) for i, s in enumerate(calendar.sessions)]
        days = [{"id": f"d{i}", "company_id": "A", "session_index": i, "session_date": s.session_date,
                 "timestamp": s.close_at, "features": [float(i)]*14} for i, s in enumerate(calendar.sessions[:25])]
        snapshot = {"nodes": {"stock_day": days}}
        config = {"sample_start": "2017-12-20", "sample_end": "2018-01-13", "train_end": "2017-12-28",
                  "validation_end": "2018-01-04", "lookback_sessions": 3, "horizon_sessions": 2}
        original = build_samples(snapshot, bars, benchmark, calendar, config)
        future = copy.deepcopy(snapshot)
        for day in future["nodes"]["stock_day"]:
            if day["session_date"] > config["train_end"]: day["features"] = [1e6]*14
        changed = build_samples(future, bars, benchmark, calendar, config)
        self.assertEqual(original["scaler"], changed["scaler"])
        training = original["masks"]["train"]
        np.testing.assert_array_equal(original["features"][original["windows"][training]], changed["features"][changed["windows"][training]])


@unittest.skipUnless(ML_AVAILABLE, "Experiment tests require numpy and torch")
class MetricsTests(unittest.TestCase):
    def test_auc_handles_ties(self):
        self.assertEqual(auc([0, 1, 0, 1], [.5]*4), .5)
        self.assertEqual(auc([0, 0, 1, 1], [0, .2, .7, 1]), 1)
        self.assertIsNone(auc([1, 1], [.1, .2]))

    def test_majority_accuracy_is_distinct_from_balanced_accuracy(self):
        target = np.array([[.1, 1, -3]]*3 + [[-.1, 0, -3]])
        result = evaluate(target, np.array([[0, .75, -3]]*4), ["2019-01-02"]*4)
        self.assertEqual(result["direction_accuracy"], .75)
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertIsNone(result["mean_daily_rank_ic"])

    def test_block_interval_on_constant_effect(self):
        result = block_interval(np.full(90, .02), block=20, replicates=200)
        np.testing.assert_allclose(result["ci95"], [.02, .02])

    def test_paired_comparison_keeps_cross_stock_dependence(self):
        dates = [f"d{day:02}" for day in range(30) for _ in range(4)]
        target = np.tile([[.1, 1, -3], [-.1, 0, -3]], (60, 1))
        candidate = np.column_stack((target[:, 0], target[:, 1]*.8+.1, target[:, 2]))
        baseline = np.column_stack((-target[:, 0], (1-target[:, 1])*.8+.1, target[:, 2]+1))
        result = paired_comparison(target, candidate, baseline, dates, block=10, replicates=100)
        self.assertEqual(result["direction_accuracy_gain"]["sessions"], 30)
        np.testing.assert_allclose(result["direction_accuracy_gain"]["ci95"], [1, 1])
        np.testing.assert_allclose(result["daily_rank_ic_gain"]["ci95"], [2, 2])


@unittest.skipUnless(ML_AVAILABLE, "Experiment tests require numpy and torch")
class ForecastCausalityTests(unittest.TestCase):
    def data(self):
        torch.manual_seed(10)
        return {"prices": torch.randn(8, 3), "events": torch.randn(3, 4), "categories": torch.tensor([0, 1, 2]),
                "edge_event": torch.tensor([0, 1, 2]), "edge_day": torch.tensor([1, 5, 7]),
                "edge_relation": torch.tensor([0, 0, 1]), "edge_age": torch.tensor([0, 2, 0]),
                "edge_weight": torch.tensor([1., .5, .25]), "corr_source": torch.tensor([1, 5]),
                "corr_target": torch.tensor([5, 1]), "corr_weight": torch.tensor([.8, .8])}

    def test_future_event_values_do_not_change_earlier_forecast(self):
        data = self.data()
        model = WindowForecastModel(3, 4, 8, "event_graph_correlation_gru").eval()
        # Days 0..3 and 4..7 are two stock timelines; days 1 and 5 share a date.
        windows = torch.tensor([[0, 1], [4, 5]])
        with torch.no_grad():
            before = model(data, windows)
            data["events"][2] = 1e5
            data["prices"][[2, 3, 6, 7]] = 1e5
            after = model(data, windows)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_same_day_control_does_not_repeat_old_event(self):
        data = self.data()
        data["edge_age"][:] = 3
        torch.manual_seed(42)
        price = WindowForecastModel(3, 4, 8, "price_gru").eval()
        event = WindowForecastModel(3, 4, 8, "same_day_event_gru").eval()
        event.load_state_dict(price.state_dict())
        window = torch.tensor([[4, 5]])
        with torch.no_grad():
            torch.testing.assert_close(price(data, window), event(data, window), rtol=0, atol=0)

    def test_real_training_loss_reaches_event_and_correlation_parameters(self):
        data = self.data()
        model = WindowForecastModel(3, 4, 8, "event_graph_correlation_gru")
        loss = multitask_loss(model(data, torch.tensor([[0, 1], [4, 5]])), torch.tensor([[.5, 1., -.5], [-.1, 0., .5]]))
        loss.backward()
        self.assertGreater(model.event_encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.correlation_message.weight.grad.abs().sum().item(), 0)

    def test_negative_control_never_imports_future_content(self):
        events = [{"id": str(i), "known_at": f"2019-01-{i+1:02}T00:00:00Z"} for i in range(10)]
        replacements = causal_replacement(events)
        for i, replacement in enumerate(replacements):
            self.assertLessEqual(events[replacement]["known_at"], events[i]["known_at"])
        self.assertTrue(any(i != value for i, value in enumerate(replacements)))

    def test_rolling_correlation_does_not_use_future_returns(self):
        rng = np.random.default_rng(7)
        base = rng.normal(size=90)
        rows = []
        for company in range(3):
            for j in range(90):
                features = [0.0]*14; features[0] = float(base[j]+rng.normal(scale=.1))
                rows.append({"id": f"{company}-{j}", "company_id": str(company), "session_index": j, "features": features})
        snapshot = {"nodes": {"stock_day": rows}}
        source, target, weights = correlation_edges(snapshot, window=10, neighbors=1)
        changed = copy.deepcopy(snapshot)
        for row in changed["nodes"]["stock_day"]:
            if row["session_index"] > 40: row["features"][0] *= -100
        source2, target2, weights2 = correlation_edges(changed, window=10, neighbors=1)
        def earlier(s, t, w):
            return [(int(a), int(b), float(c)) for a, b, c in zip(s, t, w) if rows[int(b)]["session_index"] <= 40]
        self.assertEqual(earlier(source, target, weights), earlier(source2, target2, weights2))
        for a, b in zip(source, target):
            self.assertNotEqual(rows[a]["company_id"], rows[b]["company_id"])
            self.assertEqual(rows[a]["session_index"], rows[b]["session_index"])


if __name__ == "__main__":
    unittest.main()
