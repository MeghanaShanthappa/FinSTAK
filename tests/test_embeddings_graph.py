from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

from stak_graph.embeddings.backend import TestHashBackend
from stak_graph.embeddings.pipeline import VectorCache, checked_vector, embed_events, event_text, load_vectors
from stak_graph.extraction.backends import ReplayBackend
from stak_graph.extraction.pipeline import ExtractionPipeline
from stak_graph.extraction.registry import CompanyRegistry
from stak_graph.extraction.schema import digest
from stak_graph.extraction.sources import Document
from stak_graph.graph.builder import EventPolicy, GraphBuilder, audit_snapshot, company_node
from stak_graph.graph.market import Bar, Session, TradingCalendar, candle_features, read_bars
from stak_graph.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = CompanyRegistry.from_csv(ROOT / "examples/companies.csv")
CALENDAR = TradingCalendar.from_csv(ROOT / "examples/calendar.csv")
BARS = read_bars(ROOT / "examples/prices.csv", CALENDAR)
PIPELINE = ExtractionPipeline(ReplayBackend(ROOT / "examples/replay_responses.json"), REGISTRY)
RESULTS = [PIPELINE.extract(Document.from_dict(row)) for row in read_jsonl(ROOT / "examples/documents.jsonl")]
EVENTS = [row for result in RESULTS for row in result["events"]]
RELATIONSHIPS = [row for result in RESULTS for row in result["relationships"]]
CUTOFF = "2026-10-08T20:00:00Z"


def build(cutoff=CUTOFF, events=None, relationships=None, policy=None, bars=None):
    return GraphBuilder(CALENDAR, BARS if bars is None else bars, REGISTRY,
                        EVENTS if events is None else events,
                        RELATIONSHIPS if relationships is None else relationships, policy).build(cutoff)


def active(graph):
    return [edge for edge in graph["edges"] if edge["type"] == "ACTIVE_ON"]


class CountingEncoder(TestHashBackend):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return super().encode(texts)


class EmbeddingTests(unittest.TestCase):
    def test_text_preserves_duration_evidence_and_financial_values(self):
        shutdown = event_text(EVENTS[0])
        guidance = event_text(EVENTS[1])
        self.assertIn("ten calendar days", shutdown)
        self.assertIn("5.0 billion", guidance)
        self.assertIn("4.5 billion", guidance)

    def test_bad_vector_is_rejected(self):
        for vector in ([0.0] * 8, [float("nan")] * 8, [1.0]):
            with self.assertRaises(ValueError):
                checked_vector(vector, 8)

    def test_binary_vector_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "vectors"
            manifest = embed_events(EVENTS, TestHashBackend(), output)
            vectors, loaded = load_vectors(output)
            self.assertEqual(manifest, loaded)
            self.assertEqual(len(vectors), 2)
            self.assertEqual((output / "vectors.f32").stat().st_size, 2 * 8 * 4)
            self.assertTrue(loaded["encoder"]["simulated"])
            self.assertAlmostEqual(sum(v * v for v in next(iter(vectors.values()))["vector"]), 1, places=6)

    def test_cache_reuses_unchanged_event_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = VectorCache(Path(tmp) / "cache.db")
            encoder = CountingEncoder()
            embed_events(EVENTS, encoder, Path(tmp) / "first", cache)
            manifest = embed_events(EVENTS, encoder, Path(tmp) / "second", cache)
            self.assertEqual(encoder.calls, 1)
            self.assertEqual(manifest["cache_hits"], 2)
            cache.close()

    def test_changing_encoder_dimension_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = VectorCache(Path(tmp) / "cache.db")
            embed_events(EVENTS, TestHashBackend(8), Path(tmp) / "first", cache)
            manifest = embed_events(EVENTS, TestHashBackend(16), Path(tmp) / "second", cache)
            self.assertEqual(manifest["cache_hits"], 0)
            cache.close()

    def test_changed_source_text_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = VectorCache(Path(tmp) / "cache.db")
            embed_events(EVENTS, TestHashBackend(), Path(tmp) / "first", cache)
            changed = deepcopy(EVENTS)
            changed[0]["description"] += " Additional supported context."
            manifest = embed_events(changed, TestHashBackend(), Path(tmp) / "second", cache)
            self.assertEqual(manifest["cache_hits"], 1)
            cache.close()

    def test_unreviewed_event_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = deepcopy(EVENTS)
            events[0]["validation_status"] = "needs_review"
            manifest = embed_events(events, TestHashBackend(), Path(tmp) / "out")
            self.assertEqual(manifest["count"], 1)

    def test_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                embed_events(EVENTS + EVENTS, TestHashBackend(), Path(tmp) / "out")

    def test_corrupt_vector_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            embed_events(EVENTS, TestHashBackend(), output)
            with (output / "vectors.f32").open("ab") as handle:
                handle.write(b"x")
            with self.assertRaises(ValueError):
                load_vectors(output)

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                embed_events(EVENTS, TestHashBackend(), Path(tmp))


class MarketTests(unittest.TestCase):
    def test_after_close_event_maps_to_next_session(self):
        index = CALENDAR.first_eligible("2026-10-01T20:32:00Z")
        self.assertEqual(CALENDAR.sessions[index].session_date, "2026-10-02")

    def test_friday_after_close_maps_to_monday(self):
        index = CALENDAR.first_eligible("2026-10-02T20:01:00Z")
        self.assertEqual(CALENDAR.sessions[index].session_date, "2026-10-05")

    def test_exact_cutoff_is_eligible(self):
        index = CALENDAR.first_eligible("2026-10-02T20:00:00Z")
        self.assertEqual(CALENDAR.sessions[index].session_date, "2026-10-02")

    def test_calendar_handles_timezone_offsets(self):
        self.assertEqual(CALENDAR.first_eligible("2026-10-01T13:32:00-07:00"), CALENDAR.first_eligible("2026-10-01T20:32:00Z"))

    def test_earlier_event_age_is_not_reset(self):
        with self.assertRaises(ValueError):
            CALENDAR.first_eligible("2020-01-01T00:00:00Z")

    def test_bad_ohlc_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(BARS[0], high=1)

    def test_features_are_invariant_to_appended_future_prices(self):
        history = [bar for bar in BARS if bar.company_id == "DEMO_ACME"]
        prefix = candle_features(history[:3], CALENDAR)
        self.assertEqual(prefix, candle_features(history, CALENDAR)[:3])

    def test_first_return_has_a_missingness_mask(self):
        features = candle_features([BARS[0]], CALENDAR)[0]
        self.assertEqual(features[0], 0)
        self.assertEqual(features[9], 1)

    def test_missing_session_is_not_treated_as_one_day_return(self):
        history = [bar for bar in BARS if bar.company_id == "DEMO_ACME"]
        features = candle_features([history[0], history[2]], CALENDAR)
        self.assertEqual(features[-1][9], 1)
        self.assertEqual(features[-1][-1], 2)

    def test_calendar_rejects_inverted_sessions(self):
        with self.assertRaises(ValueError):
            Session("2026-10-01", "2026-10-01T20:00:00Z", "2026-10-01T13:30:00Z", "2026-10-01T20:00:00Z")


class GraphTests(unittest.TestCase):
    def test_graph_has_expected_daily_and_event_nodes(self):
        snapshot = build()
        self.assertEqual(snapshot["stats"]["stock_days"], 21)
        self.assertEqual(snapshot["stats"]["events"], 2)
        self.assertEqual(snapshot["stats"]["direct_active_edges"], 10)
        self.assertEqual(snapshot["stats"]["propagated_active_edges"], 5)
        self.assertTrue(snapshot["contains_synthetic_data"])

    def test_no_event_before_known_at(self):
        snapshot = build("2026-10-01T20:00:00Z")
        self.assertEqual(snapshot["nodes"]["event"], [])
        self.assertEqual(active(snapshot), [])

    def test_no_injection_into_previous_day(self):
        snapshot = build()
        dates = {node["id"]: node["session_date"] for node in snapshot["nodes"]["stock_day"]}
        self.assertTrue(all(dates[edge["target"]] >= "2026-10-02" for edge in active(snapshot)))

    def test_half_life_uses_trading_sessions(self):
        snapshot = build(policy=EventPolicy(half_life_sessions=1))
        direct = [edge for edge in active(snapshot) if edge["relationship_basis"] == "directly_stated"]
        self.assertTrue(all(math.isclose(edge["weight"], 2 ** (-edge["age_sessions"])) for edge in direct))

    def test_expiry_stops_new_event_injection(self):
        snapshot = build(policy=EventPolicy(ttl_sessions=2))
        self.assertTrue(all(edge["age_sessions"] < 2 for edge in active(snapshot)))
        self.assertEqual(snapshot["stats"]["direct_active_edges"], 4)

    def test_events_expired_before_the_window_are_not_loaded(self):
        snapshot = GraphBuilder(CALENDAR, BARS, REGISTRY, EVENTS, RELATIONSHIPS,
                                EventPolicy(ttl_sessions=1)).build(CUTOFF, lookback=2)
        self.assertEqual(snapshot["nodes"]["event"], [])
        self.assertTrue(all(row["reason"] == "expired_before_lookback" for row in snapshot["excluded"]))

    def test_direct_only_removes_inferred_connections(self):
        snapshot = build(policy=EventPolicy(propagate_relations=False))
        self.assertEqual(snapshot["stats"]["propagated_active_edges"], 0)
        self.assertEqual(snapshot["stats"]["direct_active_edges"], 10)

    def test_fixed_weights_are_not_calibrated_confidence(self):
        self.assertTrue(all(e["weight_basis"] == "fixed_weight_baseline" and e["confidence"] is None for e in active(build())))

    def test_uncalibrated_probability_is_rejected(self):
        events = deepcopy(EVENTS)
        events[0]["confidence"] = 0.8
        with self.assertRaises(ValueError):
            build(events=events)

    def test_future_prices_do_not_change_current_snapshot_inputs(self):
        changed = [replace(bar, open=1000, high=1002, low=999, close=1001) if bar.session_date > "2026-10-02" else bar for bar in BARS]
        original = build("2026-10-02T20:00:00Z")
        updated = build("2026-10-02T20:00:00Z", bars=changed)
        self.assertEqual(original["nodes"], updated["nodes"])
        self.assertEqual(original["edges"], updated["edges"])

    def test_future_event_does_not_change_current_snapshot_inputs(self):
        future = deepcopy(EVENTS[0])
        future["event_id"] = "future-event"
        future["known_at"] = "2030-01-01T00:00:00Z"
        original = build()
        updated = build(events=EVENTS + [future])
        self.assertEqual(original["nodes"], updated["nodes"])
        self.assertEqual(original["edges"], updated["edges"])

    def test_later_relationship_is_not_projected_onto_earlier_days(self):
        relation = deepcopy(RELATIONSHIPS[0])
        relation["known_at"] = "2026-10-06T12:00:00Z"
        relation["document_id"] = "late-business-relation-source"
        snapshot = build(relationships=[relation])
        dates = {node["id"]: node["session_date"] for node in snapshot["nodes"]["stock_day"]}
        inferred = [e for e in active(snapshot) if e["relationship_basis"] != "directly_stated"]
        self.assertTrue(inferred)
        self.assertTrue(all(dates[e["target"]] >= "2026-10-06" for e in inferred))
        self.assertTrue(all(e["age_sessions"] >= 2 for e in inferred))

    def test_explicit_expiration_is_respected(self):
        events = deepcopy(EVENTS)
        events[0]["valid_until"] = "2026-10-05T20:00:00Z"
        snapshot = build(events=events)
        dates = {node["id"]: node["session_date"] for node in snapshot["nodes"]["stock_day"]}
        self.assertTrue(all(dates[e["target"]] <= "2026-10-05" for e in active(snapshot) if e["source"] == events[0]["event_id"]))

    def test_unrestricted_company_reverse_edge_is_rejected(self):
        snapshot = build()
        snapshot["edges"].append({"type": "REVERSE_OBSERVATION", "source": company_node("DEMO_ACME"), "target": snapshot["nodes"]["stock_day"][0]["id"]})
        with self.assertRaises(ValueError):
            audit_snapshot(snapshot)

    def test_corrupted_event_timestamp_is_rejected(self):
        snapshot = build()
        snapshot["nodes"]["event"][0]["known_at"] = "2030-01-01T00:00:00Z"
        with self.assertRaises(ValueError):
            audit_snapshot(snapshot)

    def test_nonfinite_market_features_are_rejected(self):
        snapshot = build()
        snapshot["nodes"]["stock_day"][0]["features"][0] = float("nan")
        with self.assertRaises(ValueError):
            audit_snapshot(snapshot)

    def test_price_cannot_flow_into_a_shared_event_node(self):
        snapshot = build()
        snapshot["edges"].append({"type": "SUPPORTED_BY", "source": snapshot["nodes"]["stock_day"][-1]["id"],
                                  "target": snapshot["nodes"]["event"][0]["id"]})
        with self.assertRaises(ValueError):
            audit_snapshot(snapshot)

    def test_late_relationship_timestamp_cannot_be_hidden_in_a_path(self):
        snapshot = build()
        for edge in snapshot["edges"]:
            if "relationship_id" in edge:
                edge["known_at"] = "2026-10-08T19:00:00Z"
        with self.assertRaises(ValueError):
            audit_snapshot(snapshot)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch is optional")
class ModelTests(unittest.TestCase):
    def data(self, snapshot):
        from stak_graph.models.event_gru import tensorize
        vectors = {event["event_id"]: {"source_hash": event["source_hash"], "text_sha256": digest(event_text(event)),
                                      "vector": TestHashBackend().encode([event_text(event)])[0]} for event in EVENTS}
        return tensorize(snapshot, vectors, 8)

    def model(self):
        from stak_graph.models.event_gru import EventGraphGRU
        torch.manual_seed(7)
        return EventGraphGRU(14, 8, hidden=16)

    def test_forward_and_backward_have_finite_gradients(self):
        model = self.model()
        outputs = model(self.data(build()))
        loss = sum(value.square().mean() for value in outputs.values())
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
        self.assertGreater(float(model.event_encoder[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.relation_messages[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.relation_messages[1].weight.grad.abs().sum()), 0)

    def test_future_daily_nodes_cannot_change_earlier_outputs(self):
        model = self.model().eval()
        prefix = self.data(build("2026-10-02T20:00:00Z"))
        extended = self.data(build())
        left, right = model(prefix), model(extended)
        lookup = {identifier: i for i, identifier in enumerate(extended["day_ids"])}
        rows = [lookup[identifier] for identifier in prefix["day_ids"]]
        for key in left:
            self.assertTrue(torch.allclose(left[key], right[key][rows], atol=1e-6))

    def test_snapshot_without_events_is_supported(self):
        outputs = self.model()(self.data(build("2026-10-01T20:00:00Z")))
        self.assertEqual(outputs["return_score"].shape, (6,))

    def test_text_vector_must_match_event_version(self):
        from stak_graph.models.event_gru import tensorize
        snapshot = build()
        vectors = {event["event_id"]: {"source_hash": event["source_hash"], "text_sha256": "stale", "vector": [0.0] * 8} for event in EVENTS}
        with self.assertRaises(ValueError):
            tensorize(snapshot, vectors, 8)

    def test_zero_event_weights_match_no_event_baseline(self):
        model = self.model().eval()
        weighted = self.data(build())
        weighted["edge_weight"].zero_()
        baseline = self.data(build(events=[], relationships=[]))
        left, right = model(weighted), model(baseline)
        for key in left:
            self.assertTrue(torch.allclose(left[key], right[key], atol=1e-6))

    def test_absolute_event_weights_affect_model_outputs(self):
        model = self.model().eval()
        weighted = self.data(build())
        original = model(weighted)["return_score"]
        weighted["edge_weight"] *= 0.25
        reduced = model(weighted)["return_score"]
        self.assertGreater(float((original - reduced).abs().max().detach()), 1e-7)


if __name__ == "__main__":
    unittest.main()
