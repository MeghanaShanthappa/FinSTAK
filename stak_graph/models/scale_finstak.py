"""Controlled FinSTAK extensions: isolation, OHLCV, hybrid decay and sectors.

This is a custom typed-attention implementation, not a reproduction of HGT.
No supply-chain edge or analyst-consensus value is fabricated.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from .finstak import NullAttention
from .next_day import CandleDecoder

VARIANTS = ('no_events', 'no_decay', 'events', 'lstm', 'tgcn', 'isolated',
            'volume_shared', 'volume_uncertainty', 'volume_pcgrad', 'hybrid_decay',
            'sector', 'primary_only', 'two_layer', 'direction_aux', 'finbert', 'dual_stream', 'combined')
VOLUME_VARIANTS = ('volume_shared', 'volume_uncertainty', 'volume_pcgrad', 'dual_stream', 'combined')


class HybridNullAttention(NullAttention):
    """Bounded exponential/power-law mixture, with null mass in the softmax."""
    def __init__(self, hidden=32, heads=4, relations=12):
        super().__init__(hidden, heads, relations)
        self.mixture = nn.Parameter(torch.zeros(relations, heads))
        self.power = nn.Parameter(torch.zeros(relations, heads))
        self.time_scale = nn.Parameter(torch.full((relations, heads), math.log(math.expm1(5.))))

    def forward(self, queries, values, source, target, relation, age, weight):
        if len(source) == 0:
            return torch.zeros_like(queries)
        q = self.query(queries).reshape(-1, self.heads, self.width)
        k = torch.einsum('ehwi,ei->ehw', self.key[relation], values[source])
        v = torch.einsum('ehwi,ei->ehw', self.value[relation], values[source])
        a = age.clamp_min(0)[:, None]
        mixture = self.mixture[relation]
        exponential = F.logsigmoid(mixture) - F.softplus(self.rate[relation]) * a
        power = F.logsigmoid(-mixture) - F.softplus(self.power[relation]) * torch.log1p(a / F.softplus(self.time_scale[relation]))
        log_decay = torch.logaddexp(exponential, power)
        logits = (q[target] * k).sum(-1) / math.sqrt(self.width) + self.relation_bias(relation)
        logits = logits + log_decay + weight.clamp_min(torch.finfo(weight.dtype).tiny).log()[:, None]
        logits = logits.masked_fill(((age < 0) | (weight <= 0))[:, None], -torch.inf)
        sink = self.null(queries)
        index = target[:, None].expand(-1, self.heads)
        maximum = sink.clone().scatter_reduce(0, index, logits, reduce='amax', include_self=True).detach()
        mass = (logits - maximum[target]).exp()
        denominator = (sink - maximum).exp().index_add(0, target, mass)
        alpha = mass / denominator[target]
        return torch.zeros_like(q).index_add(0, target, alpha[..., None] * v).flatten(1)


class ScaleTrunk(nn.Module):
    def __init__(self, companies, event_dimension, variant, hidden=32, category_count=6):
        super().__init__()
        self.variant, self.hidden = variant, hidden
        self.input = nn.Linear(19, hidden)
        self.company = nn.Embedding(companies, hidden)
        layers = 2 if variant in ('two_layer', 'combined') else 1
        self.temporal = (nn.LSTM if variant == 'lstm' else nn.GRU)(hidden, hidden, num_layers=layers, batch_first=True)
        self.event = nn.Linear(event_dimension, hidden, bias=False)
        self.category = nn.Embedding(category_count, hidden)
        self.cross = (HybridNullAttention(hidden, relations=2 * category_count) if variant in ('hybrid_decay', 'combined') else
                      NullAttention(hidden, relations=2 * category_count, learned_decay=variant != 'no_decay'))
        self.peer = NullAttention(hidden, relations=1, learned_decay=False)
        self.peer_projection = nn.Linear(hidden, hidden, bias=False)
        self.sector_projection = nn.Linear(hidden, hidden, bias=False)
        self.sector_gate = nn.Linear(hidden, 1)

    def forward(self, data):
        b, t, c, _ = data['x'].shape
        x = self.input(data['x']) + self.company.weight[None, None]
        if self.variant == 'tgcn':
            batch = torch.arange(b, device=x.device)[:, None, None, None]
            history = torch.arange(t, device=x.device)[None, :, None, None]
            peer = x[batch, history, data['history_peers']]
            weights = data['history_peer_weight'][..., None]
            x = x + (self.peer_projection(peer) * weights).sum(-2) / weights.sum(-2).clamp_min(1)
        sequence = x.permute(0, 2, 1, 3).reshape(b * c, t, self.hidden)
        state, _ = self.temporal(sequence)
        local = state[:, -1]
        if self.variant in ('no_events', 'lstm', 'tgcn'):
            return local
        events = self.event(data['events']) + self.category(data['categories'])
        source, target, age = data['edge_source'], data['edge_target'], data['edge_age']
        if self.variant in ('primary_only', 'combined'):
            eligible = data['primary_discloser'][source]
            source, target, age = source[eligible], target[eligible], age[eligible]
        relation = 2 * data['categories'][source]
        shock = self.cross(local, events, source, target, relation, age, torch.ones_like(age))
        if self.variant in ('sector', 'combined'):
            # Explicit company -> sector -> company disturbance routing.
            # Remove the recipient's own contribution before lateral diffusion.
            sector = data['sectors']
            membership = F.one_hot(sector, int(sector.max()) + 1).to(local.dtype)
            grouped = torch.einsum('cs,bch->bsh', membership, shock.reshape(b, c, -1))
            count = membership.sum(0)
            others = (grouped[:, sector] - shock.reshape(b, c, -1)) / (count[sector] - 1).clamp_min(1)[None, :, None]
            peer = self.sector_projection(others.reshape(b * c, -1)) * torch.sigmoid(self.sector_gate(local))
        else:
            peer_source = (torch.arange(b, device=x.device)[:, None, None] * c + data['peers']).reshape(-1)
            peer_target = torch.arange(b * c, device=x.device).repeat_interleave(3)
            weights = data['peer_weight'].reshape(-1)
            peer = self.peer(local, shock, peer_source, peer_target, torch.zeros_like(peer_source),
                             torch.zeros_like(weights), weights)
        return local + shock + peer


class ScaleFinSTAK(nn.Module):
    def __init__(self, scaler, companies, event_dimension=768, variant='events', hidden=32, category_count=6):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown expanded-study variant')
        self.variant = variant
        self.trunk = ScaleTrunk(companies, event_dimension, variant, hidden, category_count)
        self.decoder = CandleDecoder(scaler)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.wick_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.volume_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.direction_head = nn.Linear(hidden, 1)
        self.log_variances = nn.Parameter(torch.zeros(4))
        self.isolate = variant in ('isolated', 'combined', 'dual_stream')
        self.dispersion = copy.deepcopy(self.trunk) if variant == 'dual_stream' else None

    def forward(self, data):
        state = self.trunk(data)
        values = self.head(state)
        opening = self.decoder.target_mean[0] + self.decoder.target_scale[0] * values[:, 0]
        closing = self.decoder.target_mean[3] + self.decoder.target_scale[3] * values[:, 3]
        top, bottom = torch.maximum(opening, closing), torch.minimum(opening, closing)
        dispersion = self.dispersion(data) if self.dispersion is not None else state
        if self.isolate:
            wick_state = dispersion if self.dispersion is not None else dispersion.detach()
            wick_values = self.wick_head(wick_state)
            top, bottom = top.detach(), bottom.detach()
            upper_raw, lower_raw = wick_values[:, 0], wick_values[:, 1]
        else:
            upper_raw, lower_raw = values[:, 1], values[:, 2]
        upper = self.decoder.geometry_scales[0] * F.softplus(upper_raw + .5413248546)
        lower = self.decoder.geometry_scales[1] * F.softplus(lower_raw + .5413248546)
        ohlc = torch.stack((opening, top + upper, bottom - lower, closing), -1)
        volume = self.volume_head(dispersion)[:, 0] if self.variant in VOLUME_VARIANTS else None
        self.direction_logits = self.direction_head(state)[:, 0] if self.variant in ('direction_aux', 'combined') else None
        return ohlc, volume


def task_losses(output, data, scaler, model=None):
    price, volume = output
    scale = price.new_tensor(scaler['target_scale'])
    error = ((price - data['target']) / scale) ** 2
    losses = [error[:, [0, 3]].mean(), error[:, [1, 2]].mean()]
    if volume is not None:
        losses.append((((volume - data['volume_target']) / scaler['volume_target_scale']) ** 2).mean())
    if model is not None and model.direction_logits is not None:
        losses.append(F.binary_cross_entropy_with_logits(model.direction_logits, (data['target'][:, 3] > 0).to(price.dtype)))
    return losses


def pcgrad_backward(losses, parameters, generator):
    """Project shared gradients only; task-specific gradients retain their scale.

    Non-shared entries are summed; entries used by every task are averaged,
    following PCGrad's shared/non-shared aggregation convention.
    """
    params = [p for p in parameters if p.requires_grad]
    gradients, used = [], []
    for index, loss in enumerate(losses):
        grad = torch.autograd.grad(loss, params, retain_graph=index < len(losses) - 1, allow_unused=True)
        used.append(torch.cat([torch.full((p.numel(),), g is not None, dtype=torch.bool, device=p.device) for p, g in zip(params, grad)]))
        gradients.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grad)]))
    projected = [g.clone() for g in gradients]
    for i, g in enumerate(projected):
        for j in torch.randperm(len(gradients), generator=generator).tolist():
            if i == j:
                continue
            # Restrict projection to parameters shared by this pair of tasks.
            common = used[i] & used[j]
            other = gradients[j] * common
            inner = (g * other).sum()
            if inner < 0:
                g.sub_(inner * other / other.square().sum().clamp_min(1e-20))
    merged = torch.stack(projected).sum(0)
    common_all = torch.stack(used).all(0)
    merged[common_all] /= len(losses)
    offset = 0
    for p in params:
        p.grad = merged[offset:offset + p.numel()].reshape_as(p).clone()
        offset += p.numel()
