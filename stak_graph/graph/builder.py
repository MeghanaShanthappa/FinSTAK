"""Project events onto eligible daily nodes without unrestricted reverse paths."""

from dataclasses import asdict, dataclass
import math

from stak_graph.extraction.schema import digest
from stak_graph.extraction.sources import utc_timestamp
from stak_graph.embeddings.pipeline import event_text
from .market import FEATURE_NAMES, candle_features, moment


@dataclass(frozen=True)
class EventPolicy:
    half_life_sessions: float = 5.0
    ttl_sessions: int = 20
    baseline_weight: float = 1.0
    propagation_weight: float = 0.5
    propagate_relations: bool = True

    def __post_init__(self):
        if not math.isfinite(self.half_life_sessions) or self.half_life_sessions <= 0:
            raise ValueError("Half-life must be positive and finite")
        if isinstance(self.ttl_sessions, bool) or not isinstance(self.ttl_sessions, int) or self.ttl_sessions <= 0:
            raise ValueError("TTL must be a positive integer number of trading sessions")
        if not all(math.isfinite(v) and v >= 0 for v in (self.baseline_weight, self.propagation_weight)):
            raise ValueError("Baseline weights must be finite and nonnegative")


def company_node(company_id):
    return "co_" + digest(company_id)[:20]


def stock_day_node(company_id, session_date):
    return "sd_" + digest([company_id, session_date])[:20]


def document_node(document_id):
    return "doc_" + digest(document_id)[:20]


class GraphBuilder:
    def __init__(self, calendar, bars, registry, events, relationships=None, policy=None):
        self.calendar, self.bars, self.registry = calendar, bars, registry
        self.events, self.relationships = list(events), list(relationships or [])
        self.policy = policy or EventPolicy()
        if len({event["event_id"] for event in self.events}) != len(self.events):
            raise ValueError("Duplicate input event IDs")

    def build(self, cutoff, lookback=30):
        cutoff = utc_timestamp(cutoff)
        if type(lookback) is not int or lookback <= 0:
            raise ValueError("lookback must be positive")
        completed = [i for i, stamp in enumerate(self.calendar.times) if stamp <= moment(cutoff)]
        if not completed:
            raise ValueError("No trading session is complete at this cutoff")
        selected = set(completed[-lookback:])
        nodes = {"company": [], "stock_day": [], "event": [], "document": []}
        edges, excluded, documents, company_ids = [], [], {}, set()

        def add_document(row):
            identifier = row["document_id"]
            if identifier in documents:
                previous = documents[identifier]
                if previous["source_hash"] != row["source_hash"] or moment(previous["known_at"]) != moment(row["known_at"]):
                    raise ValueError("Conflicting document versions require distinct document IDs")
            documents[identifier] = row

        price_groups = {}
        for bar in self.bars:
            session = self.calendar.sessions[self.calendar.index[bar.session_date]]
            if moment(bar.available_at) <= moment(cutoff) and moment(session.prediction_at) <= moment(cutoff):
                price_groups.setdefault(bar.company_id, []).append(bar)
        valid_ids = {company.company_id for company in self.registry.companies}
        day_by_company = {}
        for company, history in sorted(price_groups.items()):
            if company not in valid_ids:
                raise ValueError(f"OHLCV company is absent from the registry: {company}")
            history.sort(key=lambda row: row.session_date)
            for bar in history:
                if not any(c.company_id == company and (not c.valid_from or c.valid_from <= bar.session_date)
                           and (not c.valid_to or bar.session_date <= c.valid_to) for c in self.registry.companies):
                    raise ValueError("Price observation falls outside the registry's security validity interval")
            features = candle_features(history, self.calendar)
            prior = None
            for bar, feature in zip(history, features):
                index = self.calendar.index[bar.session_date]
                if index not in selected:
                    continue
                session = self.calendar.sessions[index]
                node = {"id": stock_day_node(company, bar.session_date), "company_id": company,
                        "session_date": bar.session_date, "session_index": index,
                        "timestamp": session.prediction_at, "available_at": bar.available_at,
                        "features": feature, "is_synthetic": bar.is_synthetic}
                nodes["stock_day"].append(node)
                day_by_company.setdefault(company, []).append(node)
                company_ids.add(company)
                edges.append({"type": "OBSERVATION_OF", "source": node["id"], "target": company_node(company), "known_at": session.prediction_at})
                if prior:
                    gap = index - prior["session_index"]
                    edges.append({"type": "NEXT_DAY" if gap == 1 else "NEXT_OBSERVATION", "source": prior["id"],
                                  "target": node["id"], "gap_sessions": gap, "weight": 1.0})
                prior = node
        relations_by_company = {}
        for relation in self.relationships:
            if relation.get("validation_status") != "evidence_checked" or moment(relation["known_at"]) > moment(cutoff):
                continue
            source = relation["subject_resolution"]["company_id"]
            target = relation["object_resolution"]["company_id"]
            if not source or not target:
                continue
            company_ids.update((source, target))
            relations_by_company.setdefault(source, []).append(relation)
            relations_by_company.setdefault(target, []).append(relation)
            edges.append({"type": relation["relation"].upper(), "source": company_node(source), "target": company_node(target),
                          "known_at": relation["known_at"], "relationship_id": relation["relationship_id"],
                          "relationship_basis": "directly_stated", "semantic_support_verified": False})
            add_document(relation)
        for event in sorted(self.events, key=lambda row: row["event_id"]):
            if event.get("validation_status") != "evidence_checked" or moment(event["known_at"]) > moment(cutoff):
                excluded.append({"event_id": event["event_id"], "reason": "not_validated_or_not_known_at_cutoff"})
                continue
            try:
                first = self.calendar.first_eligible(event["known_at"])
            except ValueError:
                excluded.append({"event_id": event["event_id"], "reason": "predates_calendar_coverage"})
                continue
            if first is not None and (first + self.policy.ttl_sessions <= min(selected)
                                      or event.get("valid_until") and moment(event["valid_until"]) < self.calendar.times[min(selected)]):
                excluded.append({"event_id": event["event_id"], "reason": "expired_before_lookback"})
                continue
            confidence = event.get("confidence")
            calibrated = event.get("calibration_status") == "calibrated"
            if confidence is not None and (not calibrated or isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("A supplied confidence must be finite, in [0,1], and explicitly calibrated")
            confidence_factor = confidence if confidence is not None else 1.0
            nodes["event"].append({"id": event["event_id"], "category": event["category"], "description": event["description"],
                                   "known_at": event["known_at"], "source_hash": event["source_hash"], "document_id": event["document_id"],
                                   "amounts": event["amounts"], "expected_duration": event["expected_duration"],
                                   "embedding_ref": event["event_id"], "is_synthetic": event.get("is_synthetic", False)})
            nodes["event"][-1]["embedding_text_sha256"] = digest(event_text(event))
            add_document(event)
            edges.append({"type": "SUPPORTED_BY", "source": event["event_id"], "target": document_node(event["document_id"]),
                          "known_at": event["known_at"]})
            exposures = {}
            for company in event["companies"]:
                identifier = company.get("company_id")
                if not identifier or identifier not in valid_ids or company.get("resolution_status") != "resolved":
                    raise ValueError("Graph events must have resolved participants")
                company_ids.add(identifier)
                exposures[identifier] = {"basis": "directly_stated", "factor": 1.0, "known_at": event["known_at"], "path": [], "role": company["role"]}
                edges.append({"type": "PARTICIPATES_IN", "source": event["event_id"], "target": company_node(identifier),
                              "known_at": event["known_at"], "role": company["role"], "expected_direction": None})
            if self.policy.propagate_relations:
                for participant in list(exposures):
                    for relation in relations_by_company.get(participant, []):
                        if relation["relation"] not in ("supplier_of", "customer_of"):
                            continue
                        left = relation["subject_resolution"]["company_id"]
                        right = relation["object_resolution"]["company_id"]
                        target = right if participant == left else left if participant == right else None
                        if target is None or target in exposures:
                            continue
                        known_at = max((event["known_at"], relation["known_at"]), key=moment)
                        exposures[target] = {"basis": "inferred_via_business_relation", "factor": self.policy.propagation_weight,
                                             "known_at": known_at, "path": [relation["relationship_id"]], "role": "related_company"}
            if first is None:
                continue
            for company, exposure in exposures.items():
                for day in day_by_company.get(company, []):
                    age = day["session_index"] - first
                    if age < 0 or age >= self.policy.ttl_sessions or moment(exposure["known_at"]) > moment(day["timestamp"]):
                        continue
                    if event.get("valid_until") and moment(day["timestamp"]) > moment(event["valid_until"]):
                        continue
                    weight = self.policy.baseline_weight * exposure["factor"] * confidence_factor * 2 ** (-age / self.policy.half_life_sessions)
                    edges.append({"type": "ACTIVE_ON", "source": event["event_id"], "target": day["id"],
                                  "known_at": exposure["known_at"], "event_known_at": event["known_at"],
                                  "age_sessions": age, "weight": weight,
                                  "weight_basis": "calibrated_confidence" if confidence is not None else "fixed_weight_baseline",
                                  "confidence": confidence, "relationship_basis": exposure["basis"],
                                  "path": exposure["path"], "role": exposure["role"], "expected_direction": None})
        nodes["company"] = [{"id": company_node(identifier), "company_id": identifier} for identifier in sorted(company_ids)]
        nodes["document"] = [{"id": document_node(identifier), "document_id": identifier, "source_url": row["source_url"],
                              "source_hash": row["source_hash"], "known_at": row["known_at"]} for identifier, row in sorted(documents.items())]
        snapshot = {"schema_version": "1.0.0", "cutoff": cutoff, "feature_names": list(FEATURE_NAMES),
                    "nodes": nodes, "edges": edges, "policy": asdict(self.policy), "excluded": excluded,
                    "contains_synthetic_data": any(s.is_synthetic for s in self.calendar.sessions)
                    or any(bar.is_synthetic for bar in self.bars)
                    or any(e.get("is_synthetic", False) for e in self.events),
                    "stats": {"stock_days": len(nodes["stock_day"]), "events": len(nodes["event"]), "edges": len(edges),
                              "direct_active_edges": sum(e["type"] == "ACTIVE_ON" and e["relationship_basis"] == "directly_stated" for e in edges),
                              "propagated_active_edges": sum(e["type"] == "ACTIVE_ON" and e["relationship_basis"] != "directly_stated" for e in edges)},
                    "is_forecasting_evaluation": False}
        audit_snapshot(snapshot)
        return snapshot


def audit_snapshot(snapshot):
    all_nodes = [node for group in snapshot["nodes"].values() for node in group]
    lookup = {node["id"]: node for node in all_nodes}
    kinds = {node["id"]: kind for kind, group in snapshot["nodes"].items() for node in group}
    if len(lookup) != len(all_nodes):
        raise ValueError("Graph contains duplicate node IDs")
    for node in snapshot["nodes"]["stock_day"]:
        if moment(node["timestamp"]) > moment(snapshot["cutoff"]) or moment(node["available_at"]) > moment(node["timestamp"]):
            raise ValueError("StockDay uses unavailable data")
        if len(node["features"]) != len(snapshot["feature_names"]) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in node["features"]):
            raise ValueError("Invalid market feature dimension or non-finite values")
    for kind in ("event", "document"):
        if any(moment(node["known_at"]) > moment(snapshot["cutoff"]) for node in snapshot["nodes"][kind]):
            raise ValueError("Graph contains future event or document information")
    relation_times = {edge["relationship_id"]: edge["known_at"] for edge in snapshot["edges"] if "relationship_id" in edge}
    allowed = {
        "NEXT_DAY": ("stock_day", "stock_day"), "NEXT_OBSERVATION": ("stock_day", "stock_day"),
        "OBSERVATION_OF": ("stock_day", "company"), "ACTIVE_ON": ("event", "stock_day"),
        "SUPPORTED_BY": ("event", "document"), "PARTICIPATES_IN": ("event", "company"),
        **{name: ("company", "company") for name in ("SUPPLIER_OF", "CUSTOMER_OF", "PARENT_OF", "SUBSIDIARY_OF", "COMPETES_WITH")},
    }
    for edge in snapshot["edges"]:
        if edge["source"] not in lookup or edge["target"] not in lookup:
            raise ValueError("Graph edge has a missing endpoint")
        if edge["type"] not in allowed or (kinds[edge["source"]], kinds[edge["target"]]) != allowed[edge["type"]]:
            raise ValueError("Edge type or endpoint types violate the directed graph schema")
        if edge["type"] in ("NEXT_DAY", "NEXT_OBSERVATION"):
            if moment(lookup[edge["source"]]["timestamp"]) >= moment(lookup[edge["target"]]["timestamp"]):
                raise ValueError("Temporal edge points backward")
            if lookup[edge["source"]]["company_id"] != lookup[edge["target"]]["company_id"]:
                raise ValueError("Temporal edges cannot cross company timelines")
        if edge["type"] == "ACTIVE_ON":
            if moment(edge["known_at"]) > moment(lookup[edge["target"]]["timestamp"]):
                raise ValueError("Event or business relationship reached a pre-publication StockDay")
            if moment(lookup[edge["source"]]["known_at"]) > moment(lookup[edge["target"]]["timestamp"]):
                raise ValueError("Event reached a pre-publication StockDay")
            for relationship in edge["path"]:
                if relationship not in relation_times or moment(relation_times[relationship]) > moment(lookup[edge["target"]]["timestamp"]):
                    raise ValueError("Inferred influence used a future or missing business relationship")
            if not math.isfinite(edge["weight"]) or edge["weight"] < 0:
                raise ValueError("Invalid influence weight")
        # Static company/document messages cannot return to market observations.
        if edge["type"] != "ACTIVE_ON" and edge["type"] not in ("NEXT_DAY", "NEXT_OBSERVATION") and edge["target"].startswith("sd_"):
            raise ValueError("Unrestricted company/document reverse paths into StockDay nodes are forbidden")
    return True
