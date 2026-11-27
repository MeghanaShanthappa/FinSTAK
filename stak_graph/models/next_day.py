"""Four-price forecasting adapters with positive, consistent candle geometry."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .benchmark import ForecastAdapter, upstream, time_machine
from .tft import TemporalFusionEncoder


class NextDayModel(ForecastAdapter):
    def __init__(self, graph, temporal, scaler):
        super().__init__(graph, temporal, price_dimension=14)
        hidden = 32
        if temporal in ('patchtst', 'itransformer', 'timemixer', 'timemachine'):
            cfg = SimpleNamespace(task_name='long_term_forecast', seq_len=30, pred_len=1, enc_in=hidden, c_out=hidden,
                                  d_model=hidden, d_ff=64, n_heads=4, e_layers=2, dropout=.1, factor=3,
                                  activation='gelu', embed='timeF', freq='d', channel_independence=0,
                                  down_sampling_layers=1, down_sampling_window=2, down_sampling_method='avg',
                                  moving_avg=5, decomp_method='moving_avg', use_norm=1, top_k=3, label_len=0,
                                  use_future_temporal_feature=False, n1=32, n2=16, d_state=8, dconv=4, e_fact=2,
                                  revin=0, ch_ind=0, residual=1)
            if temporal == 'timemachine':
                self.temporal = time_machine(cfg)
            else:
                cls = upstream({'patchtst': 'PatchTST', 'itransformer': 'iTransformer', 'timemixer': 'TimeMixer'}[temporal])
                self.temporal = cls(cfg, patch_len=8, stride=4) if temporal == 'patchtst' else cls(cfg)
        elif temporal == 'tft':
            self.temporal = TemporalFusionEncoder(14, hidden, horizon=1, heads=4)
        self.head = nn.Linear(hidden, 4)
        self.decoder = CandleDecoder(scaler)

    def predict_windows(self, daily, windows):
        return self.decoder(super().predict_windows(daily, windows))


class CandleDecoder(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        self.register_buffer('target_mean', torch.tensor(scaler['target_mean'], dtype=torch.float32))
        self.register_buffer('target_scale', torch.tensor(scaler['target_scale'], dtype=torch.float32))
        self.register_buffer('geometry_scales', torch.tensor(scaler['geometry_scales'], dtype=torch.float32))

    def forward(self, values):
        # Outputs are log(price_{t+1}/close_t), in OHLC order. High and low
        # depend only on predicted open/close, never on actual future values.
        opening = self.target_mean[0] + self.target_scale[0] * values[..., 0]
        closing = self.target_mean[3] + self.target_scale[3] * values[..., 3]
        upper = self.geometry_scales[0] * F.softplus(values[..., 1] + .5413248546)
        lower = self.geometry_scales[1] * F.softplus(values[..., 2] + .5413248546)
        return torch.stack((opening, torch.maximum(opening, closing) + upper,
                            torch.minimum(opening, closing) - lower, closing), -1)


class FrozenHead(nn.Module):
    def __init__(self, dimension, scaler):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 4))
        self.decoder = CandleDecoder(scaler)

    def forward(self, values):
        return self.decoder(self.layers(values))


def price_predictions(log_ratios, anchors):
    values = np.asarray(anchors)[:, None] * np.exp(np.asarray(log_ratios, dtype=np.float64))
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Nonfinite or nonpositive price forecast; no silent clipping')
    return values
