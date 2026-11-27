"""A small trainable graph baseline, not a fitted forecasting model."""

import math
import torch
from torch import nn

from stak_graph.extraction.schema import EVENT_CATEGORIES
from stak_graph.graph.builder import audit_snapshot


def tensorize(snapshot, vectors, dimension, device="cpu"):
    audit_snapshot(snapshot)
    days, events = snapshot["nodes"]["stock_day"], snapshot["nodes"]["event"]
    if not days:
        raise ValueError("Model requires at least one StockDay")
    day_index = {node["id"]: i for i, node in enumerate(days)}
    event_index = {node["id"]: i for i, node in enumerate(events)}
    event_vectors, categories = [], []
    for node in events:
        row = vectors.get(node["embedding_ref"])
        if row is None or row["source_hash"] != node["source_hash"] or row["text_sha256"] != node["embedding_text_sha256"]:
            raise ValueError("Missing or stale event embedding")
        if len(row["vector"]) != dimension or not all(math.isfinite(v) for v in row["vector"]):
            raise ValueError("Invalid event vector dimension or values")
        event_vectors.append(row["vector"])
        categories.append(EVENT_CATEGORIES.index(node["category"]))
    active = [edge for edge in snapshot["edges"] if edge["type"] == "ACTIVE_ON"]
    sequences = {}
    for i, node in enumerate(days):
        sequences.setdefault(node["company_id"], []).append(i)
    for indices in sequences.values():
        indices.sort(key=lambda i: days[i]["session_index"])
    return {
        "prices": torch.tensor([node["features"] for node in days], dtype=torch.float32, device=device),
        "events": torch.tensor(event_vectors, dtype=torch.float32, device=device).reshape(len(events), dimension),
        "categories": torch.tensor(categories, dtype=torch.long, device=device),
        "edge_event": torch.tensor([event_index[e["source"]] for e in active], dtype=torch.long, device=device),
        "edge_day": torch.tensor([day_index[e["target"]] for e in active], dtype=torch.long, device=device),
        "edge_relation": torch.tensor([0 if e["relationship_basis"] == "directly_stated" else 1 for e in active], dtype=torch.long, device=device),
        "edge_weight": torch.tensor([e["weight"] for e in active], dtype=torch.float32, device=device),
        "sequences": sequences, "day_ids": [node["id"] for node in days],
    }


class EventGraphGRU(nn.Module):
    """Typed event-to-day messages followed by a forward-only GRU per stock.

    Static Company/Document edges are for provenance and relation projection;
    they are not automatically reversed or used to mix daily states backward.
    """

    def __init__(self, price_dimension, event_dimension, hidden=64):
        super().__init__()
        self.price_encoder = nn.Sequential(nn.Linear(price_dimension, hidden), nn.GELU())
        self.event_encoder = nn.Sequential(nn.Linear(event_dimension, hidden), nn.GELU())
        self.category_embedding = nn.Embedding(len(EVENT_CATEGORIES), hidden)
        self.relation_messages = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(2)])
        self.fusion = nn.Sequential(nn.LayerNorm(hidden), nn.GELU())
        self.temporal = nn.GRU(hidden, hidden, batch_first=True, bidirectional=False)
        self.return_head = nn.Linear(hidden, 1)
        self.direction_head = nn.Linear(hidden, 1)
        self.volatility_head = nn.Linear(hidden, 1)

    def encode_days(self, data):
        """Local causal messages; reusable with bounded training windows."""
        price_states = self.price_encoder(data["prices"])
        event_states = self.event_encoder(data["events"]) + self.category_embedding(data["categories"])
        aggregate = torch.zeros_like(price_states)
        counts = torch.zeros((len(price_states), 1), device=price_states.device)
        for relation, layer in enumerate(self.relation_messages):
            selected = data["edge_relation"] == relation
            indices = data["edge_day"][selected]
            if indices.numel():
                messages = layer(event_states[data["edge_event"][selected]]) * data["edge_weight"][selected, None]
                aggregate.index_add_(0, indices, messages)
                counts.index_add_(0, indices, torch.ones((len(indices), 1), device=price_states.device))
        # Count normalization preserves absolute decay/confidence weights.
        return self.fusion(price_states + aggregate / counts.clamp_min(1).sqrt())

    def readout(self, states):
        return {
            "return_score": self.return_head(states).squeeze(-1),
            "direction_logit": self.direction_head(states).squeeze(-1),
            "volatility": torch.nn.functional.softplus(self.volatility_head(states).squeeze(-1)),
        }

    def forward(self, data):
        combined = self.encode_days(data)
        states = torch.zeros_like(combined)
        for rows in data["sequences"].values():
            indices = torch.tensor(rows, dtype=torch.long, device=combined.device)
            encoded, _ = self.temporal(combined[indices].unsqueeze(0))
            states = states.index_copy(0, indices, encoded[0])
        return self.readout(states)
