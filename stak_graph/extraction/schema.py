"""One versioned schema shared by prompting, constrained generation and validation."""

import hashlib
import json
import math
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_VERSION = "1.1.0"
PROMPT_VERSION = "1.1.1"
EVENT_CATEGORIES = (
    "earnings", "guidance_change", "acquisition", "financing", "litigation",
    "operational_disruption",
)


def object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object", "properties": properties,
        "required": list(properties), "additionalProperties": False,
    }


def text(maximum: int = 1000) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


EVIDENCE = text(2000)
PARTICIPANT = object_schema({"name": text(200), "role": text(100), "evidence": EVIDENCE})
AMOUNT = object_schema({
    "value": {"type": "number"},
    "unit": text(80),
    "currency": {"type": ["string", "null"], "pattern": "^[A-Z]{3}$"},
    "metric": text(120),
    "period": {"type": ["string", "null"], "maxLength": 120},
    "evidence": EVIDENCE,
})
DURATION = object_schema({
    "value": {"type": "number", "minimum": 0},
    "unit": {"enum": ["calendar_days", "trading_days", "weeks", "months", "years", "days_unspecified"]},
    "is_estimate": {"type": "boolean"},
    "evidence": EVIDENCE,
})
EVENT = object_schema({
    "category": {"enum": list(EVENT_CATEGORIES)},
    "description": text(),
    "companies": {"type": "array", "items": PARTICIPANT, "minItems": 1, "maxItems": 12},
    "occurrence_date": {"type": ["string", "null"], "format": "date"},
    "occurrence_date_evidence": {"type": ["string", "null"], "minLength": 1, "maxLength": 2000},
    "claim_status": {"enum": ["reported", "expected", "hypothetical", "uncertain"]},
    "amounts": {"type": "array", "items": AMOUNT, "maxItems": 20},
    "expected_duration": {"anyOf": [DURATION, {"type": "null"}]},
    "evidence": {"type": "array", "items": EVIDENCE, "minItems": 1, "maxItems": 12},
})
RELATIONSHIP = object_schema({
    "subject": text(200),
    "relation": {"enum": ["supplier_of", "customer_of", "parent_of", "subsidiary_of", "competes_with"]},
    "object": text(200),
    "evidence": EVIDENCE,
})
EXTRACTION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    **object_schema({
        "events": {"type": "array", "items": EVENT, "maxItems": 12},
        "relationships": {"type": "array", "items": RELATIONSHIP, "maxItems": 20},
    }),
}
VALIDATOR = Draft202012Validator(EXTRACTION_SCHEMA, format_checker=FormatChecker())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def strict_json(value: str) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"Non-finite JSON number: {token}")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = item
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("Non-finite JSON number")
        return value

    return json.loads(value, parse_constant=reject_constant, parse_float=finite_float, object_pairs_hook=unique_keys)


def schema_errors(payload: Any) -> list[str]:
    return [
        f"{'/'.join(map(str, error.absolute_path)) or '$'}: {error.message}"
        for error in sorted(VALIDATOR.iter_errors(payload), key=lambda item: str(item.absolute_path))
    ]
