"""Deterministic evidence checks. These are not semantic entailment or calibration."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from .registry import CompanyRegistry, normalized_name
from .schema import digest
from .sources import Chunk, Document


@dataclass
class CheckedRecord:
    record: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)


SMALL_NUMBERS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
    range(21),
))
SMALL_NUMBERS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90})


def numbers_in(text: str) -> set[Decimal]:
    values = set()
    for number in re.findall(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?", text.replace("−", "-")):
        try:
            values.add(Decimal(number.replace(",", "")))
        except InvalidOperation:
            pass
    total, active = 0, False
    for word in re.findall(r"[a-z]+", text.casefold()) + ["END"]:
        if word in SMALL_NUMBERS:
            total += SMALL_NUMBERS[word]
            active = True
        elif word == "hundred" and active:
            total *= 100
        elif word == "and" and active:
            continue
        else:
            if active:
                values.add(Decimal(total))
            total, active = 0, False
    return values


def _quoted_names(quote: str, *names: str) -> bool:
    haystack = f" {normalized_name(quote)} "
    return all(f" {normalized_name(name)} " in haystack for name in names)


def _spans(quote: str, document: Document, chunk: Chunk, field_path: str,
           errors: list[str]) -> dict[str, Any] | None:
    offset = chunk.text.find(quote)
    if offset < 0:
        errors.append(f"evidence_not_in_source:{field_path}")
        return None
    start = chunk.start + offset
    return {"field": field_path, "quote": quote, "start": start, "end": start + len(quote),
            "document_id": document.document_id}


def _check_number(value: float, quote: str, key: str, errors: list[str]) -> None:
    if Decimal(str(value)) not in numbers_in(quote):
        errors.append(f"number_not_in_evidence:{key}")


def _date_in_evidence(stamp: str, evidence: str) -> bool:
    parsed = date.fromisoformat(stamp)
    if re.search(rf"(?<!\d){re.escape(stamp)}(?!\d)", evidence):
        return True
    month = rf"(?:{parsed.strftime('%B')}|{parsed.strftime('%b')}\.?)"
    patterns = [rf"\b{month}\s+0?{parsed.day},?\s+{parsed.year}\b",
                rf"\b0?{parsed.day}\s+{month}\s+{parsed.year}\b"]
    return any(re.search(pattern, evidence, re.I) for pattern in patterns)


def check_event(event: dict, document: Document, chunk: Chunk, registry: CompanyRegistry) -> CheckedRecord:
    errors: list[str] = []
    flags: list[str] = []
    spans = []
    for index, quote in enumerate(event["evidence"]):
        span = _spans(quote, document, chunk, f"evidence/{index}", errors)
        if span:
            spans.append(span)
    companies = []
    for index, company in enumerate(event["companies"]):
        span = _spans(company["evidence"], document, chunk, f"companies/{index}", errors)
        if span:
            spans.append(span)
        if not _quoted_names(company["evidence"], company["name"]):
            errors.append(f"company_not_in_evidence:{index}")
        resolved = registry.resolve(company["name"], document.known_at)
        if resolved["resolution_status"] != "resolved":
            flags.append(f"company_{resolved['resolution_status']}:{company['name']}")
        companies.append({**company, **resolved})
    for index, amount in enumerate(event["amounts"]):
        path = f"amounts/{index}"
        span = _spans(amount["evidence"], document, chunk, path, errors)
        if span:
            spans.append(span)
        _check_number(amount["value"], amount["evidence"], path, errors)
        if amount["currency"] and not re.search(rf"\b{re.escape(amount['currency'])}\b", amount["evidence"], re.I):
            flags.append(f"currency_not_explicit:{index}")
        unit = amount["unit"].casefold().replace("_", " ")
        quote = amount["evidence"].casefold()
        if unit in ("percent", "percentage"):
            supported = "%" in quote or "percent" in quote
        else:
            supported = unit.rstrip("s") in quote
        if not supported:
            flags.append(f"unit_not_explicit:{index}")
    duration = event["expected_duration"]
    if duration:
        span = _spans(duration["evidence"], document, chunk, "expected_duration", errors)
        if span:
            spans.append(span)
        _check_number(duration["value"], duration["evidence"], "expected_duration", errors)
        words = normalized_name(duration["evidence"])
        unit = duration["unit"]
        if unit in ("calendar_days", "trading_days"):
            valid_unit = unit.split("_")[0] in words and bool(re.search(r"\bdays?\b", words))
        elif unit == "days_unspecified":
            valid_unit = bool(re.search(r"\bdays?\b", words))
        else:
            valid_unit = bool(re.search(rf"\b{unit.rstrip('s')}s?\b", words))
        if not valid_unit:
            errors.append("duration_unit_not_in_evidence")
    if event["occurrence_date"]:
        date_quote = event["occurrence_date_evidence"]
        if not date_quote:
            errors.append("occurrence_date_missing_evidence")
        else:
            span = _spans(date_quote, document, chunk, "occurrence_date", errors)
            if span:
                spans.append(span)
            if not _date_in_evidence(event["occurrence_date"], date_quote):
                errors.append("occurrence_date_not_in_evidence")
    elif event["occurrence_date_evidence"] is not None:
        errors.append("date_evidence_without_date")
    if event["claim_status"] in ("hypothetical", "uncertain"):
        flags.append(f"claim_{event['claim_status']}")
    fingerprint = {
        "document_id": document.document_id, "source_hash": document.source_hash,
        "category": event["category"], "companies": sorted((c["name"], c["role"]) for c in companies),
        "evidence": sorted(event["evidence"]), "occurrence_date": event["occurrence_date"],
        "occurrence_date_evidence": event["occurrence_date_evidence"],
        "amounts": event["amounts"], "expected_duration": duration, "claim_status": event["claim_status"],
    }
    record = {
        **event, "companies": companies, "event_id": "evt_" + digest(fingerprint)[:24],
        **document.metadata(), "evidence_spans": spans,
        "validation_status": "rejected" if errors else "needs_review" if flags else "evidence_checked",
        "validation_errors": sorted(set(errors)), "review_flags": sorted(set(flags)),
        "semantic_support_verified": False, "human_reviewed": False,
        "confidence": None, "calibration_status": "uncalibrated",
        "expected_direction": None, "half_life_days": None, "valid_until": None,
        "relationship_basis": "directly_stated", "chunk_indices": [chunk.index],
    }
    return CheckedRecord(record, errors, flags)


def check_relationship(relation: dict, document: Document, chunk: Chunk, registry: CompanyRegistry) -> CheckedRecord:
    errors: list[str] = []
    flags: list[str] = []
    span = _spans(relation["evidence"], document, chunk, "relationship", errors)
    if not _quoted_names(relation["evidence"], relation["subject"], relation["object"]):
        errors.append("relationship_entities_not_in_evidence")
    if relation["relation"] in ("customer_of", "supplier_of"):
        quoted = normalized_name(relation["evidence"])
        subject_name = re.escape(normalized_name(relation["subject"]))
        object_name = re.escape(normalized_name(relation["object"]))

        def purchase(left, right):
            return re.search(rf"\b{left}\b\s+(?:(?:is|was) (?:a )?customer of|(?:purchases|buys|sources|procures)\b.{{0,140}}?\bfrom)\s+{right}\b", quoted)

        def supply(left, right):
            return re.search(rf"\b{left}\b\s+(?:(?:is|was) (?:a )?supplier (?:of|to)|(?:supplies|sells|provides)\b.{{0,140}}?\bto)\s+{right}\b", quoted)

        observed = set()
        if purchase(subject_name, object_name) or supply(object_name, subject_name):
            observed.add("customer_of")
        if purchase(object_name, subject_name) or supply(subject_name, object_name):
            observed.add("supplier_of")
        if observed and relation["relation"] not in observed:
            errors.append("relationship_direction_contradicts_evidence")
            errors.append("With subject=" + repr(relation["subject"]) + " and object=" + repr(relation["object"])
                          + ", the quoted purchase/supply phrase supports relation=" + repr(sorted(observed)[0])
                          + ". Keep the subject and object fixed when correcting this relation.")
        elif not observed:
            flags.append("relationship_direction_requires_review")
    else:
        flags.append("relationship_semantics_requires_review")
    subject = registry.resolve(relation["subject"], document.known_at)
    target = registry.resolve(relation["object"], document.known_at)
    if subject["company_id"] is None or target["company_id"] is None:
        flags.append("relationship_entity_unresolved_or_ambiguous")
    record = {
        **relation, **document.metadata(),
        "relationship_id": "rel_" + digest({"document_id": document.document_id, "source_hash": document.source_hash, **relation})[:24],
        "subject_resolution": subject, "object_resolution": target,
        "evidence_spans": [span] if span else [],
        "validation_status": "rejected" if errors else "needs_review" if flags else "evidence_checked",
        "validation_errors": errors, "review_flags": flags, "relationship_basis": "directly_stated",
        "semantic_support_verified": False, "confidence": None,
        "calibration_status": "uncalibrated", "chunk_indices": [chunk.index],
    }
    return CheckedRecord(record, errors, flags)
