"""Conservative, explicit partial projections for a headline-only pilot.

Raw extraction records are never relabeled. A new derived event retains the
verbatim headline and independently resolved, quoted participants. Unverified
numeric/role/counterparty fields are omitted from this separate projection.
This is field-level filtering, not human review or semantic certification.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

from stak_graph.extraction.schema import digest
from stak_graph.io import read_jsonl, write_json, write_jsonl
from .data import sha256_file

PROJECTION_VERSION = "1.0.0"
CATEGORY_PATTERNS = {
    "acquisition": r"\b(acqui(?:re[sd]?|ring|sition)|buyout|takeover|merger|merge[sd]?)\b",
    "guidance_change": r"\b(guidance|outlook|forecast)\b",
    "earnings": r"\b(earnings|eps|net (?:income|loss)|revenue|sales|profit)\b",
    "financing": r"\b(debt|bonds?|financing|offering|convertible|credit facility|loans?|funding|notes)\b",
    "litigation": r"\b(sue[sd]?|suing|lawsuits?|litigation|settle[sd]?|settlement|court|fines?|probe|investigat\w*|charges?)\b",
    "operational_disruption": r"\b(shutdown|shut down|halt\w*|outage|disrupt\w*|suspend\w*|strike|recall\w*|plant closure|factory clos\w*)\b",
}


def category_surface_supported(category, text):
    if not re.search(CATEGORY_PATTERNS[category], text, re.I):
        return False
    if category in ("earnings", "guidance_change"):
        if re.search(r"\b(price target|pt|maintains?|initiates? coverage|upgrades?|downgrades?)\b", text, re.I):
            return False
    if category == "earnings":
        return bool(re.search(r"\b(reports?|reported|beats?|beat|miss(?:es|ed)?|posts?|posted|results|q[1-4])\b", text, re.I))
    return True


def project_event(event, document):
    if event.get("validation_errors") or event.get("validation_status") not in ("evidence_checked", "needs_review"):
        return None, "source_validation_error"
    if event["claim_status"] in ("uncertain", "hypothetical"):
        return None, "uncertain_or_hypothetical"
    flags = event.get("review_flags", [])
    allowed = ("company_unresolved:", "currency_not_explicit:", "unit_not_explicit:")
    if any(not flag.startswith(allowed) for flag in flags):
        return None, "other_unresolved_review_flag"
    text = document["text"]
    if event["source_hash"] != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError("Projection document/source hash mismatch")
    if not category_surface_supported(event["category"], text):
        return None, "category_not_supported_by_headline_surface"
    participants = []
    for company in event["companies"]:
        if company.get("resolution_status") == "resolved" and company.get("company_id"):
            if company["evidence"] not in text:
                raise ValueError("Resolved participant quotation is absent from the source")
            participants.append({**company, "role": "named_participant"})
    if not participants:
        return None, "no_independently_resolved_participant"
    identifiers = sorted({c["company_id"] for c in participants})
    record = copy.deepcopy(event)
    record.update({
        "event_id": "evp_" + digest([PROJECTION_VERSION, event["document_id"], event["category"], identifiers])[:24],
        "source_event_id": event["event_id"], "source_validation_status": event["validation_status"],
        "source_review_flags": flags, "projection_version": PROJECTION_VERSION,
        "projection_validation_level": "verbatim_headline_category_surface_and_quoted_resolved_participants",
        "description": text, "companies": participants, "amounts": [], "expected_duration": None,
        "occurrence_date": None, "occurrence_date_evidence": None,
        "evidence": [text], "evidence_spans": [{"document_id": event["document_id"], "field": "projected_headline",
                                                "quote": text, "start": 0, "end": len(text)}],
        "validation_status": "evidence_checked", "review_flags": [], "validation_errors": [],
        "semantic_support_verified": False, "human_reviewed": False,
        "projection_omissions": {
            "structured_amounts": len(event["amounts"]),
            "unresolved_participants": [c["name"] for c in event["companies"] if c.get("resolution_status") != "resolved"],
            "participant_role_assertions": True,
        },
    })
    return record, "projected"


def project(extraction, output):
    extraction, output = Path(extraction), Path(output)
    if output.exists():
        raise ValueError("Projection output already exists")
    source_manifest = json.loads((extraction / "manifest.json").read_text())
    if source_manifest["run_status"] == "aborted" or source_manifest["summary"]["document_errors"]:
        raise ValueError("Projection requires completed extraction without source input errors")
    documents = {row["document_id"]: row for row in read_jsonl(extraction / "documents.jsonl")}
    events = list(read_jsonl(extraction / "events.jsonl"))
    projected, audit, seen = [], [], set()
    for event in events:
        record, reason = project_event(event, documents[event["document_id"]])
        if record and record["event_id"] not in seen:
            projected.append(record)
            seen.add(record["event_id"])
        audit.append({"source_event_id": event["event_id"], "source_validation_status": event["validation_status"],
                      "projected_event_id": record["event_id"] if record else None, "reason": reason,
                      "source_review_flags": event.get("review_flags", [])})
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "events.jsonl", projected)
    write_jsonl(output / "projection_audit.jsonl", audit)
    manifest = {"projection_version": PROJECTION_VERSION, "source_events_sha256": sha256_file(extraction / "events.jsonl"),
                "source_documents_sha256": sha256_file(extraction / "documents.jsonl"),
                "raw_events": len(events), "projected_events": len(projected), "outcomes": dict(Counter(row["reason"] for row in audit)),
                "source_statuses": dict(Counter(row["validation_status"] for row in events)),
                "semantics": "New partial events only; original records retain their review flags. No human or semantic certification.",
                "omitted_fields": ["structured amounts", "duration", "occurrence date", "unresolved participants", "specific participant roles"],
                "category_surface_checks": CATEGORY_PATTERNS, "is_accuracy_benchmark": False}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(project(args.extraction, args.output), indent=2))


if __name__ == "__main__":
    main()
