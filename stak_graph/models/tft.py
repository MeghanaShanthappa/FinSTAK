"""Compact Temporal Fusion Transformer adapted to the STAK forecast task.

Architecture follows Lim et al., arXiv:1912.09363v3, sections 4.1–4.5:
variable selection, four static contexts, LSTM encoder/decoder, static
enrichment, shared-value interpretable attention and gated residual paths.
https://arxiv.org/html/1912.09363v3
https://github.com/google-research/google-research/tree/master/tft

The original quantile heads/loss are replaced with the same return, binary
direction and log-volatility heads/loss used by the GRU comparison. The sole
known-future input is the relative forecast-horizon index; future observed
prices, news, and targets are never decoder inputs. Company identity is static.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from stak_graph.experiments.train import WindowForecastModel


class GateAddNorm(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden, hidden * 2)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, value, residual):
        return self.norm(residual + F.glu(self.gate(self.dropout(value)), dim=-1))


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dimension, hidden, output_dimension=None, context_dimension=None, dropout=0.1):
        super().__init__()
        output_dimension = output_dimension or hidden
        self.input = nn.Linear(input_dimension, hidden)
        self.context = nn.Linear(context_dimension, hidden, bias=False) if context_dimension else None
        self.hidden = nn.Linear(hidden, output_dimension)
        self.skip = nn.Identity() if input_dimension == output_dimension else nn.Linear(input_dimension, output_dimension)
        self.output = GateAddNorm(output_dimension, dropout)

    def forward(self, value, context=None):
        hidden = self.input(value)
        if self.context is not None:
            if context is None:
                raise ValueError("GRN requires its static context")
            while context.ndim < hidden.ndim:
                context = context.unsqueeze(1)
            hidden = hidden + self.context(context)
        return self.output(self.hidden(F.elu(hidden)), self.skip(value))


class VariableSelection(nn.Module):
    def __init__(self, dimensions, hidden, context=True, dropout=0.1):
        super().__init__()
        self.projections = nn.ModuleList(nn.Linear(dimension, hidden) for dimension in dimensions)
        self.transforms = nn.ModuleList(GatedResidualNetwork(hidden, hidden, dropout=dropout) for _ in dimensions)
        self.selection = GatedResidualNetwork(len(dimensions) * hidden, hidden, len(dimensions),
                                              hidden if context else None, dropout)

    def forward(self, values, context=None):
        if len(values) != len(self.projections):
            raise ValueError("Wrong number of temporal variables")
        projected = [layer(value) for layer, value in zip(self.projections, values)]
        logits = self.selection(torch.cat(projected, dim=-1), context)
        weights = torch.softmax(logits, dim=-1)
        transformed = torch.stack([layer(value) for layer, value in zip(self.transforms, projected)], dim=-2)
        selected = (transformed * weights.unsqueeze(-1)).sum(dim=-2)
        return selected, weights


class InterpretableAttention(nn.Module):
    """Head-specific queries/keys, a shared value projection, averaged heads."""

    def __init__(self, hidden, heads=4, dropout=0.1):
        super().__init__()
        if heads <= 0 or hidden % heads:
            raise ValueError("TFT state size must be divisible by its attention heads")
        self.heads, self.head_dimension = heads, hidden // heads
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, self.head_dimension, bias=False)
        self.output = nn.Linear(self.head_dimension, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        batch, steps, _ = values.shape
        query = self.query(values).reshape(batch, steps, self.heads, self.head_dimension).transpose(1, 2)
        key = self.key(values).reshape(batch, steps, self.heads, self.head_dimension).transpose(1, 2)
        logits = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        forbidden = torch.ones((steps, steps), dtype=torch.bool, device=values.device).triu(1)
        weights = torch.softmax(logits.masked_fill(forbidden, float("-inf")), dim=-1)
        shared_weights = self.dropout(weights).mean(dim=1)
        result = self.output(shared_weights @ self.value(values))
        return result, weights.mean(dim=1)


class TemporalFusionEncoder(nn.Module):
    def __init__(self, price_dimension, hidden, horizon=5, heads=4, dropout=0.1):
        super().__init__()
        if horizon < 1:
            raise ValueError("Forecast horizon must be positive")
        self.price_dimension, self.horizon = price_dimension, horizon
        # Raw continuous inputs remain separate variables. Graph outputs are
        # two vector-valued variable groups, fixed across every condition.
        self.past_selection = VariableSelection([1] * price_dimension + [hidden, hidden], hidden, dropout=dropout)
        self.future_selection = VariableSelection([1], hidden, dropout=dropout)
        self.static_selection = GatedResidualNetwork(hidden, hidden, dropout=dropout)
        self.static_context = GatedResidualNetwork(hidden, hidden, dropout=dropout)
        self.static_enrichment = GatedResidualNetwork(hidden, hidden, dropout=dropout)
        self.static_hidden = GatedResidualNetwork(hidden, hidden, dropout=dropout)
        self.static_cell = GatedResidualNetwork(hidden, hidden, dropout=dropout)
        self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.local_skip = GateAddNorm(hidden, dropout)
        self.enrichment = GatedResidualNetwork(hidden, hidden, context_dimension=hidden, dropout=dropout)
        self.attention = InterpretableAttention(hidden, heads, dropout)
        self.attention_skip = GateAddNorm(hidden, dropout)
        self.feedforward = GatedResidualNetwork(hidden, hidden, dropout=dropout)
        self.final_skip = GateAddNorm(hidden, dropout)
        self.register_buffer("future_positions", torch.arange(1, horizon + 1, dtype=torch.float32).view(1, horizon, 1) / horizon)

    def forward(self, prices, event_messages, correlation_messages, static_embedding, diagnostics=False):
        batch, history, dimension = prices.shape
        if dimension != self.price_dimension:
            raise ValueError("Unexpected historical feature dimension")
        static = self.static_selection(static_embedding)
        context = self.static_context(static)
        enrichment = self.static_enrichment(static)
        initial = (self.static_hidden(static).unsqueeze(0), self.static_cell(static).unsqueeze(0))
        variables = [prices[..., i:i+1] for i in range(dimension)] + [event_messages, correlation_messages]
        past, selection_weights = self.past_selection(variables, context)
        future, future_weights = self.future_selection([self.future_positions.expand(batch, -1, -1)], context)
        encoded, state = self.encoder(past, initial)
        decoded, _ = self.decoder(future, state)
        selected = torch.cat((past, future), dim=1)
        local = self.local_skip(torch.cat((encoded, decoded), dim=1), selected)
        enriched = self.enrichment(local, enrichment)
        attended, attention_weights = self.attention(enriched)
        attended = self.attention_skip(attended, enriched)
        forecast_states = self.final_skip(self.feedforward(attended), local)[:, history:]
        if diagnostics:
            return forecast_states, {"historical_variable_weights": selection_weights,
                                     "known_future_variable_weights": future_weights,
                                     "attention_weights": attention_weights}
        return forecast_states


class WindowTFTModel(WindowForecastModel):
    def __init__(self, price_dimension, event_dimension, hidden, variant, company_count=16, horizon=5):
        if not variant.endswith("_tft"):
            raise ValueError("TFT requires a TFT variant name")
        super().__init__(price_dimension, event_dimension, hidden, variant[:-4] + "_gru", company_count)
        del self.temporal
        self.tft = TemporalFusionEncoder(price_dimension, hidden, horizon=horizon)

    def encode_days(self, data):
        _, event_messages, correlation_messages = self.graph_channels(data)
        return {"prices": data["prices"], "event_messages": event_messages, "correlation_messages": correlation_messages,
                "company_index": data["company_index"]}

    def predict_windows(self, daily_states, windows):
        company = daily_states["company_index"][windows[:, -1]]
        state = self.tft(daily_states["prices"][windows], daily_states["event_messages"][windows],
                         daily_states["correlation_messages"][windows], self.company_embedding(company))[:, -1]
        return torch.stack((self.return_head(state)[:, 0], self.direction_head(state)[:, 0], self.volatility_head(state)[:, 0]), dim=1)
