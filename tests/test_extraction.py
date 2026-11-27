"""CPU-only regression tests: evidence fidelity and temporal correctness."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stak_graph.extraction.backends import BackendError, Generation, HTTPBackend, ReplayBackend
from stak_graph.extraction.cli import main
from stak_graph.extraction.pipeline import ExtractionPipeline, ResponseCache, participant_links
from stak_graph.extraction.registry import Company, CompanyRegistry
from stak_graph.extraction.schema import EXTRACTION_SCHEMA, canonical_json, schema_errors, strict_json
from stak_graph.extraction.sources import Chunk, Document, chunk_document
from stak_graph.extraction.validation import check_event, check_relationship, numbers_in

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = strict_json((ROOT / "examples/replay_responses.json").read_text())
DOCUMENTS = [Document.from_dict(strict_json(line)) for line in (ROOT / "examples/documents.jsonl").read_text().splitlines()]


def registry() -> CompanyRegistry:
    return CompanyRegistry.from_csv(ROOT / "examples/companies.csv")


def chunk(document: Document) -> Chunk:
    return Chunk(0, 0, len(document.text), document.text)


class FakeBackend:
    def __init__(self, responses, revision="test-v1"):
        self.responses = list(responses)
        self.calls = 0
        self.revision = revision

    @property
    def identity(self):
        return {"backend": "test", "revision": self.revision, "simulated": True}

    def generate(self, messages, schema, request_id):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, Generation) else Generation(canonical_json(response))


class SchemaTests(unittest.TestCase):
    def test_fixture_responses_match_schema(self):
        for response in FIXTURES.values():
            self.assertEqual(schema_errors(response), [])

    def test_no_event_is_valid(self):
        self.assertEqual(schema_errors({"events": [], "relationships": []}), [])

    def test_unrequested_confidence_is_rejected(self):
        payload = deepcopy(FIXTURES["demo-shutdown"])
        payload["events"][0]["confidence"] = 0.99
        self.assertTrue(schema_errors(payload))

    def test_invalid_category_is_rejected(self):
        payload = deepcopy(FIXTURES["demo-shutdown"])
        payload["events"][0]["category"] = "stock_will_rise"
        self.assertTrue(schema_errors(payload))

    def test_impossible_occurrence_date_is_rejected(self):
        payload = deepcopy(FIXTURES["demo-shutdown"])
        payload["events"][0]["occurrence_date"] = "2026-02-30"
        self.assertTrue(schema_errors(payload))

    def test_boolean_is_not_an_amount(self):
        payload = deepcopy(FIXTURES["demo-guidance"])
        payload["events"][0]["amounts"][0]["value"] = True
        self.assertTrue(schema_errors(payload))

    def test_nonfinite_json_is_rejected(self):
        for value in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                strict_json(value)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"events":[],"events":[1]}')


class SourceTests(unittest.TestCase):
    def test_naive_time_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(DOCUMENTS[0], published_at="2026-10-01T20:30:00")

    def test_known_time_cannot_precede_publication(self):
        with self.assertRaises(ValueError):
            replace(DOCUMENTS[0], known_at="2026-10-01T19:30:00Z")

    def test_offsets_are_normalized_to_utc(self):
        item = replace(DOCUMENTS[0], published_at="2026-10-01T13:30:00-07:00")
        self.assertEqual(item.published_at, "2026-10-01T20:30:00Z")

    def test_fractional_times_are_compared_chronologically(self):
        item = replace(DOCUMENTS[0], known_at="2026-10-01T20:30:00.001Z")
        self.assertEqual(item.known_at, "2026-10-01T20:30:00.001000Z")

    def test_empty_source_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(DOCUMENTS[0], text=" ")

    def test_chunks_preserve_all_characters_and_offsets(self):
        item = replace(DOCUMENTS[0], text=("A paragraph of market information. " * 30) + "\n\nFinal paragraph.")
        coverage = set()
        for part in chunk_document(item, 200, 30):
            self.assertEqual(part.text, item.text[part.start:part.end])
            self.assertLessEqual(len(part.text), 200)
            coverage.update(range(part.start, part.end))
        self.assertEqual(coverage, set(range(len(item.text))))

    def test_zero_overlap_is_respected(self):
        item = replace(DOCUMENTS[0], text="One repeated sentence. " * 40)
        parts = chunk_document(item, 200, 0)
        self.assertTrue(all(left.end == right.start for left, right in zip(parts, parts[1:])))

    def test_bad_chunk_configuration_fails(self):
        with self.assertRaises(ValueError):
            chunk_document(DOCUMENTS[0], 100, 60)


class RegistryTests(unittest.TestCase):
    def test_alias_case_and_punctuation(self):
        self.assertEqual(registry().resolve("ACME COMPONENTS, INC.", "2026-10-01T00:00:00Z")["company_id"], "DEMO_ACME")

    def test_ambiguous_alias_is_not_guessed(self):
        companies = CompanyRegistry([Company("A", "Alpha", ("Shared",)), Company("B", "Beta", ("Shared",))])
        resolved = companies.resolve("Shared", "2026-10-01")
        self.assertIsNone(resolved["company_id"])
        self.assertEqual(resolved["resolution_status"], "ambiguous")

    def test_historical_alias_validity(self):
        companies = CompanyRegistry([
            Company("OLD", "Old issuer", ("XYZ",), valid_to="2020-12-31"),
            Company("NEW", "New issuer", ("XYZ",), valid_from="2021-01-01"),
        ])
        self.assertEqual(companies.resolve("XYZ", "2019-01-01")["company_id"], "OLD")
        self.assertEqual(companies.resolve("XYZ", "2026-01-01")["company_id"], "NEW")

    def test_unknown_company_remains_unknown(self):
        self.assertEqual(registry().resolve("Unknown", "2026-01-01")["resolution_status"], "unresolved")


class EvidenceTests(unittest.TestCase):
    def event(self):
        return deepcopy(FIXTURES["demo-shutdown"]["events"][0])

    def test_offsets_point_to_exact_source_text(self):
        result = check_event(self.event(), DOCUMENTS[0], chunk(DOCUMENTS[0]), registry())
        self.assertEqual(result.errors, [])
        self.assertEqual(result.record["validation_status"], "evidence_checked")
        for span in result.record["evidence_spans"]:
            self.assertEqual(DOCUMENTS[0].text[span["start"]:span["end"]], span["quote"])

    def test_fabricated_quote_is_rejected(self):
        event = self.event()
        event["evidence"] = ["The stock will rise tomorrow."]
        self.assertTrue(check_event(event, DOCUMENTS[0], chunk(DOCUMENTS[0]), registry()).errors)

    def test_unmentioned_participant_is_rejected(self):
        event = self.event()
        event["companies"][0]["name"] = "Alder Technologies"
        self.assertTrue(check_event(event, DOCUMENTS[0], chunk(DOCUMENTS[0]), registry()).errors)

    def test_fabricated_duration_is_rejected(self):
        event = self.event()
        event["expected_duration"]["value"] = 99
        self.assertIn("number_not_in_evidence:expected_duration", check_event(event, DOCUMENTS[0], chunk(DOCUMENTS[0]), registry()).errors)

    def test_unsupported_calendar_unit_is_rejected(self):
        event = self.event()
        event["expected_duration"]["unit"] = "trading_days"
        self.assertIn("duration_unit_not_in_evidence", check_event(event, DOCUMENTS[0], chunk(DOCUMENTS[0]), registry()).errors)

    def test_spelled_out_numbers_are_supported(self):
        self.assertIn(Decimal(10), numbers_in("ten calendar days"))
        self.assertIn(Decimal(25), numbers_in("twenty-five days"))
        self.assertIn(Decimal("4.5"), numbers_in("USD 4.5 billion"))

    def test_currency_is_not_inferred_from_dollar_symbol(self):
        event = deepcopy(FIXTURES["demo-guidance"]["events"][0])
        item = replace(DOCUMENTS[1], text=DOCUMENTS[1].text.replace("USD ", "$"))
        event["evidence"] = [value.replace("USD ", "$") for value in event["evidence"]]
        event["companies"][0]["evidence"] = event["companies"][0]["evidence"].replace("USD ", "$")
        for amount in event["amounts"]:
            amount["evidence"] = amount["evidence"].replace("USD ", "$")
        result = check_event(event, item, chunk(item), registry())
        self.assertTrue(any(flag.startswith("currency_not_explicit") for flag in result.review_flags))

    def test_unresolved_company_requires_review_and_no_link(self):
        result = check_event(self.event(), DOCUMENTS[0], chunk(DOCUMENTS[0]), CompanyRegistry())
        self.assertEqual(result.record["validation_status"], "needs_review")
        self.assertEqual(participant_links([result.record]), [])

    def test_hypothetical_claim_never_becomes_direct_link(self):
        event = self.event()
        event["claim_status"] = "hypothetical"
        result = check_event(event, DOCUMENTS[0], chunk(DOCUMENTS[0]), registry())
        self.assertEqual(participant_links([result.record]), [])

    def test_duration_does_not_become_decay_or_confidence(self):
        result = check_event(self.event(), DOCUMENTS[0], chunk(DOCUMENTS[0]), registry())
        self.assertIsNone(result.record["half_life_days"])
        self.assertIsNone(result.record["confidence"])
        self.assertIsNone(result.record["expected_direction"])
        self.assertFalse(result.record["semantic_support_verified"])

    def test_publication_date_is_not_event_date_evidence(self):
        event = self.event()
        event["occurrence_date"] = "2026-10-01"
        event["occurrence_date_evidence"] = event["evidence"][0]
        result = check_event(event, DOCUMENTS[0], chunk(DOCUMENTS[0]), registry())
        self.assertIn("occurrence_date_not_in_evidence", result.errors)

    def test_explicit_iso_date_has_a_source_span(self):
        event = deepcopy(FIXTURES["demo-guidance"]["events"][0])
        result = check_event(event, DOCUMENTS[1], chunk(DOCUMENTS[1]), registry())
        self.assertEqual(result.errors, [])
        self.assertTrue(any(span["field"] == "occurrence_date" for span in result.record["evidence_spans"]))

    def test_reversed_customer_relation_is_rejected(self):
        relation = deepcopy(FIXTURES["demo-shutdown"]["relationships"][0])
        relation["relation"] = "supplier_of"
        result = check_relationship(relation, DOCUMENTS[0], chunk(DOCUMENTS[0]), registry())
        self.assertIn("relationship_direction_contradicts_evidence", result.errors)

    def test_customer_relation_direction_is_checked(self):
        relation = deepcopy(FIXTURES["demo-shutdown"]["relationships"][0])
        result = check_relationship(relation, DOCUMENTS[0], chunk(DOCUMENTS[0]), registry())
        self.assertEqual(result.errors, [])
        self.assertEqual(result.review_flags, [])


class PipelineTests(unittest.TestCase):
    def test_customer_is_not_automatically_affected(self):
        result = ExtractionPipeline(FakeBackend([FIXTURES["demo-shutdown"]]), registry()).extract(DOCUMENTS[0])
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual([item["company_id"] for item in participant_links(result["events"])], ["DEMO_ACME"])

    def test_negative_document_returns_no_events(self):
        result = ExtractionPipeline(FakeBackend([FIXTURES["demo-no-event"]]), registry()).extract(DOCUMENTS[2])
        self.assertEqual(result["events"], [])
        self.assertEqual(result["status"], "ok")

    def test_invalid_json_is_retried_and_audited(self):
        backend = FakeBackend([Generation("not JSON"), FIXTURES["demo-shutdown"]])
        result = ExtractionPipeline(backend, registry()).extract(DOCUMENTS[0])
        self.assertEqual(backend.calls, 2)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["chunk_runs"][0]["attempts"][0]["schema_errors"])

    def test_truncated_generation_cannot_silently_become_no_events(self):
        backend = FakeBackend([Generation('{"events":[],"relationships":[]}', "length")])
        result = ExtractionPipeline(backend, registry(), max_retries=0).extract(DOCUMENTS[0])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["quarantine"][0]["kind"], "chunk")

    def test_backend_failure_is_audited(self):
        result = ExtractionPipeline(FakeBackend([BackendError("offline")]), registry(), max_retries=0).extract(DOCUMENTS[0])
        self.assertEqual(result["status"], "failed")
        self.assertIn("offline", result["chunk_runs"][0]["error"])

    def test_bad_evidence_is_quarantined(self):
        payload = deepcopy(FIXTURES["demo-shutdown"])
        payload["events"][0]["evidence"] = ["An invented sentence."]
        result = ExtractionPipeline(FakeBackend([payload]), registry()).extract(DOCUMENTS[0])
        self.assertEqual(result["events"], [])
        self.assertEqual(result["quarantine"][0]["kind"], "event")

    def test_exact_duplicate_events_are_merged(self):
        payload = deepcopy(FIXTURES["demo-shutdown"])
        payload["events"] *= 2
        result = ExtractionPipeline(FakeBackend([payload]), registry()).extract(DOCUMENTS[0])
        self.assertEqual(len(result["events"]), 1)

    def test_model_revision_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResponseCache(Path(tmp) / "cache.sqlite3")
            first = FakeBackend([FIXTURES["demo-shutdown"]], "revision-1")
            ExtractionPipeline(first, registry(), cache=cache).extract(DOCUMENTS[0])
            second = FakeBackend([FIXTURES["demo-shutdown"]], "revision-2")
            result = ExtractionPipeline(second, registry(), cache=cache).extract(DOCUMENTS[0])
            self.assertEqual(second.calls, 1)
            self.assertFalse(result["chunk_runs"][0]["cache_hit"])
            cache.close()

    def test_cached_response_revalidates_company_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResponseCache(Path(tmp) / "cache.sqlite3")
            first = FakeBackend([FIXTURES["demo-shutdown"]])
            ExtractionPipeline(first, registry(), cache=cache).extract(DOCUMENTS[0])
            backend = FakeBackend([BackendError("must not be called")])
            result = ExtractionPipeline(backend, CompanyRegistry(), cache=cache).extract(DOCUMENTS[0])
            self.assertEqual(backend.calls, 0)
            self.assertTrue(result["chunk_runs"][0]["cache_hit"])
            self.assertEqual(result["events"][0]["validation_status"], "needs_review")
            cache.close()

    def test_replay_is_explicitly_labeled_simulated(self):
        backend = ReplayBackend(ROOT / "examples/replay_responses.json")
        result = ExtractionPipeline(backend, registry()).extract(DOCUMENTS[0])
        self.assertTrue(result["events"][0]["extraction"]["generator"]["simulated"])

    def test_evidence_failure_gets_one_bounded_correction(self):
        bad = deepcopy(FIXTURES["demo-shutdown"])
        bad["relationships"][0]["relation"] = "supplier_of"
        backend = FakeBackend([bad, FIXTURES["demo-shutdown"]])
        result = ExtractionPipeline(backend, registry()).extract(DOCUMENTS[0])
        self.assertEqual(backend.calls, 2)
        self.assertEqual(result["quarantine"], [])
        self.assertTrue(result["chunk_runs"][0]["attempts"][0]["evidence_errors"])

    def test_corrected_cached_response_replaces_old_bad_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResponseCache(Path(tmp) / "cache.sqlite3")
            backend = FakeBackend([FIXTURES["demo-shutdown"]])
            pipeline = ExtractionPipeline(backend, registry(), cache=cache)
            first = pipeline.extract(DOCUMENTS[0])
            cache_key = first["chunk_runs"][0]["cache_key"]
            bad = deepcopy(FIXTURES["demo-shutdown"])
            bad["relationships"][0]["relation"] = "supplier_of"
            cache.put(cache_key, Generation(canonical_json(bad)))
            corrected = pipeline.extract(DOCUMENTS[0])
            self.assertEqual(corrected["quarantine"], [])
            self.assertEqual(backend.calls, 2)
            final = pipeline.extract(DOCUMENTS[0])
            self.assertEqual(final["quarantine"], [])
            self.assertEqual(backend.calls, 2)
            cache.close()


class HTTPTests(unittest.TestCase):
    def test_request_uses_json_schema_and_non_thinking_mode(self):
        response = {"choices": [{"message": {"content": '{"events":[],"relationships":[]}'}, "finish_reason": "stop"}]}
        with patch("stak_graph.extraction.backends.urlopen") as mock:
            mock.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            backend = HTTPBackend("http://127.0.0.1:8000/v1", "test-model")
            generated = backend.generate([], EXTRACTION_SCHEMA, "id:0")
            body = strict_json(mock.call_args.args[0].data.decode())
            self.assertEqual(body["response_format"]["json_schema"]["schema"], EXTRACTION_SCHEMA)
            self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(generated.finish_reason, "stop")

    def test_credentials_must_not_be_embedded_in_endpoint(self):
        with self.assertRaises(ValueError):
            HTTPBackend("https://user:password@example.com/v1", "test")


class CLITests(unittest.TestCase):
    def run_cli(self, output, input_path=None):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main([
                "extract", "--input", str(input_path or ROOT / "examples/documents.jsonl"),
                "--output", str(output), "--registry", str(ROOT / "examples/companies.csv"),
                "--backend", "replay", "--replay-responses", str(ROOT / "examples/replay_responses.json"),
            ])

    def test_full_jsonl_export_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            self.assertEqual(self.run_cli(output), 0)
            events = [strict_json(line) for line in (output / "events.jsonl").read_text().splitlines()]
            manifest = strict_json((output / "manifest.json").read_text())
            self.assertEqual(len(events), 2)
            self.assertEqual(manifest["summary"]["documents"], 3)
            self.assertFalse(manifest["is_accuracy_benchmark"])
            self.assertTrue(manifest["extraction"]["generator"]["simulated"])
            self.assertEqual((output / "quarantine.jsonl").read_text(), "")

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            self.assertEqual(self.run_cli(output), 0)
            before = (output / "manifest.json").read_bytes()
            self.assertEqual(self.run_cli(output), 2)
            self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_bad_metadata_is_quarantined_and_batch_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.jsonl"
            bad = json.loads((ROOT / "examples/documents.jsonl").read_text().splitlines()[0])
            bad["known_at"] = "2025-01-01T00:00:00Z"
            good = (ROOT / "examples/documents.jsonl").read_text().splitlines()[2]
            source.write_text(json.dumps(bad) + "\n" + good + "\n")
            output = Path(tmp) / "out"
            self.assertEqual(self.run_cli(output, source), 2)
            summary = strict_json((output / "manifest.json").read_text())["summary"]
            self.assertEqual(summary["documents"], 1)
            self.assertEqual(summary["document_errors"], 1)

    def test_empty_input_is_a_visible_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "empty.jsonl"
            source.write_text("\n")
            output = Path(tmp) / "out"
            self.assertEqual(self.run_cli(output, source), 2)
            manifest = strict_json((output / "manifest.json").read_text())
            self.assertEqual(manifest["run_status"], "completed_with_errors")


if __name__ == "__main__":
    unittest.main()
