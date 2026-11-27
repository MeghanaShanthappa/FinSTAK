"""Causal graph encoders and adapters to pinned upstream temporal models.

These are stock forecasting adaptations, not reproductions of the source
papers' datasets or scores. SAGN uses feature hops only, with no label diffusion.
"""
import ast
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import torch
from torch import nn
from torch.nn import functional as F


EXTERNAL = Path(os.environ.get('STAK_EXTERNAL', '/var/tmp/mshanthappa-stak-graph/external'))


def upstream(name):
    path = EXTERNAL / 'tslib'
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    key = 'stak_upstream_' + name
    if key not in sys.modules:
        source = path / 'models' / (name + '.py')
        module = ModuleType(key)
        module.__file__ = str(source)
        tree = OutOfPlaceDivision().visit(ast.parse(source.read_text()))
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(source), 'exec'), module.__dict__)
        sys.modules[key] = module
    return sys.modules[key].Model


class OutOfPlaceDivision(ast.NodeTransformer):
    """Preserve upstream normalization equations while permitting input gradients.

    Upstream predictors receive data without gradients. Here those inputs are
    outputs of a trained graph encoder, so x /= std must be x = x / std.
    """
    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div) and isinstance(node.target, ast.Name):
            return ast.copy_location(ast.Assign(targets=[node.target], value=ast.BinOp(
                left=ast.Name(id=node.target.id, ctx=ast.Load()), op=ast.Div(), right=node.value)), node)
        return node


def reference_mamba():
    """Use official Mamba weights/forward with its official Torch scan reference.

    No native CUDA compiler is installed. Only the scan binding is changed;
    model equations, convolution and selective recurrence remain upstream code.
    The fused kernels and incremental inference cache are not used.
    """
    if 'stak_reference_mamba' in sys.modules:
        return sys.modules['stak_reference_mamba'].Mamba
    import einops
    root = EXTERNAL / 'mamba' / 'mamba_ssm'
    tree = ast.parse((root / 'ops/selective_scan_interface.py').read_text())
    reference = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'selective_scan_ref')
    namespace = {'torch': torch, 'F': F, 'rearrange': einops.rearrange, 'repeat': einops.repeat}
    exec(compile(ast.Module(body=[reference], type_ignores=[]), str(root / 'ops/selective_scan_interface.py'), 'exec'), namespace)
    tree = ast.parse((root / 'modules/mamba_simple.py').read_text())
    # Extract the class and ordinary imports; optional native-kernel imports
    # are bypassed explicitly in favor of the author's reference function.
    body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    module = ModuleType('stak_reference_mamba')
    module.__dict__.update({'torch': torch, 'nn': nn, 'F': F, 'Tensor': torch.Tensor,
                           'math': __import__('math'), 'Optional': __import__('typing').Optional,
                           'rearrange': einops.rearrange, 'repeat': einops.repeat,
                           'selective_scan_fn': namespace['selective_scan_ref'],
                           'causal_conv1d_fn': None, 'causal_conv1d_update': None,
                           'selective_state_update': None, 'RMSNorm': None})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(root / 'modules/mamba_simple.py'), 'exec'), module.__dict__)
    original = module.Mamba
    class ReferenceMamba(original):
        def __init__(self, *args, **kwargs):
            kwargs['use_fast_path'] = False
            super().__init__(*args, **kwargs)
    module.Mamba = ReferenceMamba
    sys.modules[module.__name__] = module
    return module.Mamba


def time_machine(config):
    root = EXTERNAL / 'TimeMachine' / 'TimeMachine_supervised'
    sys.path.insert(0, str(root))
    tree = ast.parse((root / 'models/TimeMachine.py').read_text())
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == 'mamba_ssm')]
    tree = OutOfPlaceDivision().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = {'Mamba': reference_mamba(), '__name__': 'stak_upstream_timemachine'}
    exec(compile(tree, str(root / 'models/TimeMachine.py'), 'exec'), namespace)
    # Avoid the upstream's generic models/ directory shadowing TS-Library.
    sys.path.remove(str(root))
    return namespace['Model'](config)


def scatter_mean(values, source, target, weights):
    result = torch.zeros_like(values)
    count = values.new_zeros((len(values), 1))
    result.index_add_(0, target, values[source] * weights[:, None])
    count.index_add_(0, target, torch.ones_like(weights[:, None]))
    return result / count.clamp_min(1).sqrt()


class GraphAttention(nn.Module):
    """Multihead additive GAT with typed edges and absolute decay after softmax."""
    def __init__(self, hidden, heads=4):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.projection = nn.Linear(hidden, hidden, bias=False)
        self.source_attention = nn.Parameter(torch.empty(heads, self.width))
        self.target_attention = nn.Parameter(torch.empty(heads, self.width))
        self.relation_bias = nn.Embedding(4, heads)
        self.norm = nn.LayerNorm(hidden)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)

    def forward(self, values, source, target, weights, relations):
        transformed = self.projection(values).reshape(-1, self.heads, self.width)
        logits = F.leaky_relu((transformed[source] * self.source_attention).sum(-1)
                              + (transformed[target] * self.target_attention).sum(-1)
                              + self.relation_bias(relations), negative_slope=.2)
        index = target[:, None].expand(-1, self.heads)
        maxima = logits.new_full((len(values), self.heads), -torch.inf)
        maxima.scatter_reduce_(0, index, logits, reduce='amax', include_self=True)
        exponential = (logits - maxima[target].detach()).exp()
        denominator = torch.zeros_like(maxima).index_add_(0, target, exponential)
        alpha = exponential / denominator[target].clamp_min(1e-12)
        messages = transformed[source] * (alpha * weights[:, None])[..., None]
        result = torch.zeros_like(transformed).index_add_(0, target, messages).flatten(1)
        return self.norm(values + F.elu(result))


class GraphEncoder(nn.Module):
    def __init__(self, price_dimension, event_dimension, hidden, companies, kind):
        super().__init__()
        self.kind = kind
        self.price = nn.Linear(price_dimension, hidden)
        self.event = nn.Linear(event_dimension, hidden)
        self.company = nn.Embedding(companies, hidden)
        self.category = nn.Embedding(6, hidden)
        self.norm = nn.LayerNorm(hidden)
        if kind == 'gat':
            self.attention = nn.ModuleList([GraphAttention(hidden) for _ in range(2)])
        elif kind == 'sagn':
            # SAGN's multi-hop feature propagation and learned hop attention.
            # No pseudo-labels or target labels enter this inductive adaptation.
            self.hops = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(.1)) for _ in range(3)])
            self.hop_score = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(3)])
        elif kind == 'typed':
            self.messages = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(3)])
        else:
            raise ValueError('Unknown graph encoder')

    def forward(self, data):
        days = self.price(data['prices']) + self.company(data['company_index'])
        events = self.event(data['events']) + self.category(data['categories'])
        values = torch.cat((days, events))
        source, target, weights, relations = (data[name] for name in ('graph_source', 'graph_target', 'graph_weight', 'graph_relation'))
        if self.kind == 'gat':
            for layer in self.attention:
                values = layer(values, source, target, weights, relations)
            return values[:len(days)]
        if self.kind == 'sagn':
            raw = [values]
            for _ in range(2):
                raw.append(scatter_mean(raw[-1], source, target, weights))
            hops = [layer(hop) for layer, hop in zip(self.hops, raw)]
            scores = torch.cat([layer(torch.cat((hop, hops[0]), -1)) for layer, hop in zip(self.hop_score, hops)], -1).softmax(-1)
            mixed = sum(hop * scores[:, index:index + 1] for index, hop in enumerate(hops))
            return self.norm(days + mixed[:len(days)])
        result = torch.zeros_like(values)
        for index, layer in enumerate(self.messages):
            selected = relations == index
            result = result + scatter_mean(layer(values), source[selected], target[selected], weights[selected])
        return self.norm(days + F.gelu(result[:len(days)]))


class ForecastAdapter(nn.Module):
    def __init__(self, graph, temporal, price_dimension=18, event_dimension=768, hidden=32, companies=16):
        super().__init__()
        self.graph = GraphEncoder(price_dimension, event_dimension, hidden, companies, graph)
        self.temporal_name = temporal
        self.dropout = nn.Dropout(.1)
        if temporal == 'gru':
            self.temporal = nn.GRU(hidden, hidden, batch_first=True)
            output = hidden
        elif temporal == 'xlstm':
            sys.path.insert(0, str(EXTERNAL / 'xlstm'))
            # The all-mLSTM stack uses the official pure-Torch backend. Its
            # package eagerly imports an unused sLSTM compiler helper; defer
            # that helper rather than requiring a CUDA SDK for an unused cell.
            name = 'xlstm.blocks.slstm.src.cuda_init'
            if name not in sys.modules:
                deferred = ModuleType(name)
                def unavailable(*args, **kwargs):
                    raise RuntimeError('sLSTM CUDA compilation is unavailable; this adapter uses mLSTM only')
                deferred.load = unavailable
                sys.modules[name] = deferred
            from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig
            self.temporal = xLSTMBlockStack(xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(num_heads=4)),
                context_length=30, num_blocks=2, embedding_dim=hidden, dropout=.1))
            output = hidden
        elif temporal in ('patchtst', 'itransformer', 'timemixer', 'timemachine'):
            config = SimpleNamespace(task_name='long_term_forecast', seq_len=30, pred_len=5,
                                     enc_in=hidden, c_out=hidden, d_model=hidden, d_ff=hidden * 2,
                                     n_heads=4, e_layers=2, dropout=.1, factor=3,
                                     activation='gelu', embed='timeF', freq='d',
                                     channel_independence=0, down_sampling_layers=1,
                                     down_sampling_window=2, down_sampling_method='avg',
                                     moving_avg=5, decomp_method='moving_avg', use_norm=1,
                                     top_k=3, label_len=0, use_future_temporal_feature=False,
                                     n1=32, n2=16, d_state=8, dconv=4, e_fact=2, revin=0, ch_ind=0, residual=1)
            if temporal == 'timemachine':
                self.temporal = time_machine(config)
            else:
                name = {'patchtst': 'PatchTST', 'itransformer': 'iTransformer', 'timemixer': 'TimeMixer'}[temporal]
                cls = upstream(name)
                self.temporal = cls(config, patch_len=8, stride=4) if temporal == 'patchtst' else cls(config)
            output = 5 * hidden
        elif temporal == 'tft':
            from .tft import TemporalFusionEncoder
            self.temporal = TemporalFusionEncoder(18, hidden, heads=4, horizon=5)
            output = hidden
        else:
            raise ValueError('Unsupported temporal architecture')
        self.head = nn.Linear(output, 3)

    def encode_days(self, data):
        states = self.graph(data)
        if self.temporal_name == 'tft':
            return {'states': states, 'prices': data['prices'],
                    'static': self.graph.company(data['company_index'])}
        return states

    def predict_windows(self, daily, windows):
        if self.temporal_name == 'tft':
            messages = daily['states'][windows]
            state = self.temporal(daily['prices'][windows], messages, torch.zeros_like(messages),
                                  daily['static'][windows[:, -1]])[:, -1]
            return self.head(state)
        sequence = self.dropout(daily[windows])
        if self.temporal_name == 'gru':
            state, _ = self.temporal(sequence)
            state = state[:, -1]
        elif self.temporal_name == 'xlstm':
            state = self.temporal(sequence)[:, -1]
        elif self.temporal_name == 'timemachine':
            state = self.temporal(sequence.clone()).flatten(1)
        elif self.temporal_name in ('patchtst', 'itransformer', 'timemixer'):
            state = self.temporal(sequence.clone(), None, None, None).flatten(1)
        else:
            raise ValueError('TFT uses its existing matched model adapter')
        return self.head(state)

    def forward(self, data, windows):
        return self.predict_windows(self.encode_days(data), windows)


def graph_edges(data, snapshot):
    """Only Event→StockDay, same-date correlation and forward temporal edges."""
    device = data['prices'].device
    count = len(data['prices'])
    # Self loops are explicit, as required for GAT neighborhood attention.
    loops = torch.arange(count + len(data['events']), device=device)
    lookup = {node['id']: index for index, node in enumerate(snapshot['nodes']['stock_day'])}
    temporal = [edge for edge in snapshot['edges'] if edge['type'] in ('NEXT_DAY', 'NEXT_OBSERVATION')]
    left = torch.tensor([lookup[edge['source']] for edge in temporal], device=device, dtype=torch.long)
    right = torch.tensor([lookup[edge['target']] for edge in temporal], device=device, dtype=torch.long)
    data['graph_source'] = torch.cat((data['edge_event'] + count, data['corr_source'], left, loops))
    data['graph_target'] = torch.cat((data['edge_day'], data['corr_target'], right, loops))
    data['graph_weight'] = torch.cat((data['edge_weight'], data['corr_weight'], torch.ones(len(left) + len(loops), device=device)))
    data['graph_relation'] = torch.cat((torch.zeros(len(data['edge_day']), device=device, dtype=torch.long),
                                        torch.ones(len(data['corr_target']), device=device, dtype=torch.long),
                                        torch.full((len(left),), 2, device=device, dtype=torch.long),
                                        torch.full((len(loops),), 3, device=device, dtype=torch.long)))
    return data
