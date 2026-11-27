"""FinSTAK with frozen TimesFM history embeddings replacing its GRU."""
import torch
from torch import nn
from .finstak import FinSTAK


class TimesFMEncoderFinSTAK(FinSTAK):
    def __init__(self, scaler, companies, mode='graph'):
        if mode not in ('local', 'direct', 'graph', 'no_decay'):
            raise ValueError('Unknown encoder ablation')
        super().__init__(scaler, companies=companies,
                         mode='no_events' if mode == 'local' else ('no_decay' if mode == 'no_decay' else 'events'))
        self.encoder_mode = mode
        self.temporal = nn.Identity()
        self.foundation_projection = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32))

    def encode_days(self, data):
        local = (self.foundation_projection(data['temporal_embeddings'])
                 + self.input(data['prices']) + self.company(data['company_index']))
        if self.encoder_mode == 'local':
            return local
        events = self.event(data['events']) + self.category(data['categories'])
        relation = 2 * data['categories'][data['edge_event']] + data['edge_relation']
        shock = self.cross(local, events, data['edge_event'], data['edge_day'], relation,
                           data['edge_age'], data['edge_prior'])
        if self.encoder_mode == 'direct':
            return local + shock
        peer = self.peer(local, shock, data['corr_source'], data['corr_target'],
                         torch.zeros_like(data['corr_source']), torch.zeros_like(data['corr_weight']), data['corr_weight'])
        return local + shock + peer
