"""Local-history first, typed event attention, then event-only peer diffusion.

An adaptation of the attached FinSTAK proposals, not an upstream HGT replica.
No labels enter message passing. Null messages have exactly zero value.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .next_day import CandleDecoder


class NullAttention(nn.Module):
    """Typed multihead attention with a competing zero-valued null neighbor.

    Decay appears inside softmax alongside the null, so even one event loses
    mass with age. Relation-specific K/V matrices implement heterogeneous
    messages. With no eligible messages the output is exactly zero.
    """
    def __init__(self, hidden=32, heads=4, relations=12, learned_decay=True):
        super().__init__()
        if hidden % heads:
            raise ValueError('Hidden size must be divisible by heads')
        self.heads, self.width = heads, hidden // heads
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Parameter(torch.empty(relations, heads, self.width, hidden))
        self.value = nn.Parameter(torch.empty(relations, heads, self.width, hidden))
        nn.init.xavier_uniform_(self.key.flatten(0, 2))
        nn.init.xavier_uniform_(self.value.flatten(0, 2))
        self.relation_bias = nn.Embedding(relations, heads)
        self.null = nn.Linear(hidden, heads)
        # Initially a five-session half-life, separately learned per relation/head.
        self.rate = nn.Parameter(torch.full((relations, heads), math.log(math.expm1(math.log(2) / 5))))
        self.learned_decay = learned_decay

    def forward(self, queries, values, source, target, relation, age, weight):
        if len(source) == 0:
            return torch.zeros_like(queries)
        q = self.query(queries).reshape(-1, self.heads, self.width)
        selected = values[source]
        k = torch.einsum('ehwi,ei->ehw', self.key[relation], selected)
        v = torch.einsum('ehwi,ei->ehw', self.value[relation], selected)
        logits = (q[target] * k).sum(-1) / math.sqrt(self.width) + self.relation_bias(relation)
        if self.learned_decay:
            logits = logits - F.softplus(self.rate[relation]) * age[:, None]
        valid = (age >= 0) & (weight > 0)
        logits = logits + weight.clamp_min(torch.finfo(weight.dtype).tiny).log()[:, None]
        logits = logits.masked_fill(~valid[:, None], -torch.inf)
        sink = self.null(queries)
        index = target[:, None].expand(-1, self.heads)
        maximum = sink.clone().scatter_reduce(0, index, logits, reduce='amax', include_self=True).detach()
        mass = (logits - maximum[target]).exp()
        denominator = (sink - maximum).exp().index_add(0, target, mass)
        alpha = mass / denominator[target]
        return torch.zeros_like(q).index_add(0, target, alpha[..., None] * v).flatten(1)


class FinSTAK(nn.Module):
    """Bounded local history -> event shock -> one peer hop -> valid OHLC.

    Peer edges transport only event shocks, not local baseline states. Hence
    no events anywhere gives an exact local-only fallback even with peers.
    Event-only and peer-only stages cannot accidentally compound the same edge.
    """
    def __init__(self, scaler, price_dimension=19, event_dimension=768, companies=16,
                 hidden=32, mode='events'):
        super().__init__()
        if mode not in ('events', 'no_events', 'no_decay', 'lstm', 'tgcn'):
            raise ValueError('Unknown FinSTAK mode')
        self.mode = mode
        self.input = nn.Linear(price_dimension, hidden)
        self.company = nn.Embedding(companies, hidden)
        self.temporal = (nn.LSTM if mode == 'lstm' else nn.GRU)(hidden, hidden, batch_first=True)
        self.event = nn.Linear(event_dimension, hidden, bias=False)
        self.category = nn.Embedding(6, hidden)
        self.cross = NullAttention(hidden, relations=12, learned_decay=mode != 'no_decay')
        self.peer = NullAttention(hidden, relations=1, learned_decay=False)
        self.peer_projection = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.decoder = CandleDecoder(scaler)

    def encode_days(self, data):
        # All windows end at their own StockDay, including peer-query nodes.
        inputs = self.input(data['prices']) + self.company(data['company_index'])
        if self.mode == 'tgcn':
            # GCN-before-GRU comparator on the same dynamic correlation edges.
            total, degree = torch.zeros_like(inputs), inputs.new_zeros((len(inputs), 1))
            target, source, weight = data['corr_target'], data['corr_source'], data['corr_weight']
            total.index_add_(0, target, self.peer_projection(inputs[source]) * weight[:, None])
            degree.index_add_(0, target, weight[:, None])
            inputs = inputs + total / degree.clamp_min(1)
        sequence = inputs[data['local_windows']]
        sequence = sequence * data['local_mask'][..., None]
        encoded, _ = self.temporal(sequence)
        local = encoded[:, -1]
        if self.mode in ('no_events', 'lstm', 'tgcn'):
            return local
        events = self.event(data['events']) + self.category(data['categories'])
        relation = 2 * data['categories'][data['edge_event']] + data['edge_relation']
        shock = self.cross(local, events, data['edge_event'], data['edge_day'], relation,
                           data['edge_age'], data['edge_prior'])
        peer = self.peer(local, shock, data['corr_source'], data['corr_target'],
                         torch.zeros_like(data['corr_source']), torch.zeros_like(data['corr_weight']), data['corr_weight'])
        return local + shock + peer

    def predict_windows(self, daily, windows):
        return self.decoder(self.head(daily[windows[:, -1]]))

    def forward(self, data, windows):
        return self.predict_windows(self.encode_days(data), windows)
