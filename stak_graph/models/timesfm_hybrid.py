"""Frozen TimesFM forecast prior with a trainable FinSTAK correction."""
import torch
from .finstak import FinSTAK


class TimesFMFinSTAK(FinSTAK):
    def __init__(self, scaler, companies, mode='graph', isolate_geometry=True):
        if mode not in ('local', 'direct', 'graph', 'no_decay'):
            raise ValueError('Unknown hybrid mode')
        super().__init__(scaler, price_dimension=23, companies=companies,
                         mode='no_events' if mode == 'local' else ('no_decay' if mode == 'no_decay' else 'events'))
        self.hybrid_mode = mode
        self.isolate_geometry = isolate_geometry
        # Start exactly at the foundation forecast; learn a bounded correction.
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def encode_days(self, data):
        # The shared inference helper calls encode_days/predict_windows directly.
        self.forecast_prior = data['forecast_prior']
        if self.hybrid_mode == 'direct':
            data = dict(data)
            data['corr_source'] = data['corr_source'][:0]
            data['corr_target'] = data['corr_target'][:0]
            data['corr_weight'] = data['corr_weight'][:0]
        return super().encode_days(data)

    def predict_windows(self, daily, windows):
        row = windows[:, -1]
        delta = self.head(daily[row])
        base = self.forecast_prior[row]
        scale = self.decoder.target_scale
        opening = base[:, 0] + scale[0] * delta[:, 0]
        closing = base[:, 3] + scale[3] * delta[:, 3]
        # Additive nonnegative wick corrections can grow a zero-width wick
        # introduced by native candle projection. Multiplicative corrections
        # would leave those forecasts effectively unable to learn a range.
        upper = base[:, 1] - torch.maximum(base[:, 0], base[:, 3])
        lower = torch.minimum(base[:, 0], base[:, 3]) - base[:, 2]
        upper = (upper + self.decoder.geometry_scales[0] * delta[:, 1]).clamp_min(0)
        lower = (lower + self.decoder.geometry_scales[1] * delta[:, 2]).clamp_min(0)
        top, bottom = torch.maximum(opening, closing), torch.minimum(opening, closing)
        if self.isolate_geometry:
            top, bottom = top.detach(), bottom.detach()
        return torch.stack((opening, top + upper, bottom - lower, closing), -1)

    def forward(self, data, windows):
        self.forecast_prior = data['forecast_prior']
        return super().forward(data, windows)
