import copy
import unittest
import numpy as np
import torch

from stak_graph.graph.market import Bar, Session, TradingCalendar, candle_features
from stak_graph.experiments.next_day_data import build_samples, next_target, metrics
from stak_graph.models.next_day import NextDayModel, CandleDecoder, price_predictions
from stak_graph.models.benchmark import EXTERNAL


class NextDayTest(unittest.TestCase):
    def fixture(self):
        dates = ['2017-12-20', '2017-12-21', '2017-12-22', '2017-12-26', '2017-12-27', '2017-12-28', '2017-12-29',
                 '2018-01-02', '2018-01-03', '2018-12-28', '2018-12-31', '2019-01-02', '2019-01-03', '2019-01-04']
        calendar = TradingCalendar([Session(day, day + 'T14:30:00Z', day + 'T21:00:00Z', day + 'T23:59:59Z', True) for day in dates])
        bars = []
        for i, day in enumerate(dates):
            opening = 90 + .5 * i + i * i
            closing = opening + .6 + .2 * i
            bars.append(Bar('A', day, opening, closing + .6, opening - .9, closing, 1000 + i,
                            day + 'T21:15:00Z', True))
        config = {'sample_start': dates[0], 'sample_end': dates[-2], 'train_end': '2017-12-31',
                  'validation_end': '2018-12-31', 'lookback_sessions': 2}
        return calendar, bars, config

    def snapshot(self, calendar, bars):
        features = candle_features(bars, calendar)
        return {'nodes': {'stock_day': [{'company_id': 'A', 'session_date': bar.session_date,
                                        'session_index': i, 'timestamp': bar.session_date + 'T23:59:59Z', 'features': features[i]}
                                       for i, bar in enumerate(bars)]}}

    def test_targets_follow_trading_sessions_across_holidays(self):
        calendar, bars, config = self.fixture()
        lookup = {('A', i): bar for i, bar in enumerate(bars)}
        actual, anchor = next_target('A', 6, lookup)
        np.testing.assert_array_equal(actual, [bars[7].open, bars[7].high, bars[7].low, bars[7].close])
        self.assertEqual(calendar.sessions[7].session_date, '2018-01-02')
        self.assertEqual(anchor, bars[6].close)
        study = build_samples(self.snapshot(calendar, bars), bars, calendar, config)
        index = next(i for i, row in enumerate(study['metadata']) if row['session_date'] == '2017-12-29')
        self.assertTrue(study['masks']['purged'][index])
        self.assertEqual(study['targets'].shape[1], 4)
        self.assertEqual(study['features'].shape[1], 14)
        self.assertFalse(study['scaler']['market_relative'])

    def test_future_prices_do_not_change_training_scalers_or_labels(self):
        calendar, bars, config = self.fixture()
        original = build_samples(self.snapshot(calendar, bars), bars, calendar, config)
        changed = list(bars)
        for index, bar in enumerate(changed):
            if bar.session_date.startswith('2019'):
                changed[index] = Bar('A', bar.session_date, bar.open * 2, bar.high * 2, bar.low * 2, bar.close * 2,
                                     bar.volume, bar.available_at, True)
        modified = build_samples(self.snapshot(calendar, changed), changed, calendar, config)
        self.assertEqual(original['scaler'], modified['scaler'])
        np.testing.assert_array_equal(original['targets'][original['masks']['train']], modified['targets'][modified['masks']['train']])

    def test_candle_decoder_has_valid_geometry_and_input_gradients(self):
        scaler = {'target_mean': [0, .01, -.01, 0], 'target_scale': [.02] * 4, 'geometry_scales': [.005, .006]}
        values = torch.randn(25, 4, requires_grad=True)
        result = CandleDecoder(scaler)(values)
        prices = price_predictions(result.detach().numpy(), np.full(25, 100.))
        self.assertTrue((prices[:, 1] >= np.maximum(prices[:, 0], prices[:, 3])).all())
        self.assertTrue((prices[:, 2] <= np.minimum(prices[:, 0], prices[:, 3])).all())
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    @unittest.skipUnless((EXTERNAL / 'tslib').exists(), 'Optional upstream models unavailable')
    def test_all_backbones_produce_four_next_day_prices(self):
        scaler = {'target_mean': [0, .01, -.01, 0], 'target_scale': [.02] * 4, 'geometry_scales': [.005, .006]}
        windows = torch.arange(30).repeat(2, 1)
        for name in ('gru', 'xlstm', 'patchtst', 'itransformer', 'timemixer', 'timemachine', 'tft'):
            with self.subTest(model=name):
                model = NextDayModel('gat', name, scaler)
                daily = torch.randn(40, 32, requires_grad=True)
                if name == 'tft':
                    daily = {'states': daily, 'prices': torch.randn(40, 14), 'static': torch.randn(40, 32)}
                    self.assertEqual(model.temporal.horizon, 1)
                elif name in ('patchtst', 'itransformer', 'timemixer'):
                    self.assertEqual(model.temporal.pred_len, 1)
                result = model.predict_windows(daily, windows)
                self.assertEqual(tuple(result.shape), (2, 4))
                result.square().mean().backward()

    def test_metrics_are_price_errors_and_absolute_close_direction(self):
        truth = np.array([[100., 104., 99., 102.], [102., 103., 97., 98.]])
        prediction = np.array([[100., 103., 99., 101.], [101., 103., 97., 99.]])
        result = metrics(truth, prediction, np.array([100., 100.]), np.array(['A', 'B']))
        self.assertEqual(result['per_feature']['close']['MAE_price_units'], 1.)
        self.assertEqual(result['next_close_direction_accuracy'], 1.)
        self.assertNotIn('direction_accuracy', result)
