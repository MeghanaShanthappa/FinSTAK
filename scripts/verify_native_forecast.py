"""Reconstruct native forecasts and test cross-task isolation using cached weights."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, default=Path('/var/tmp/mshanthappa-stak-graph/model_sweep_v2'))
args = parser.parse_args()
root = args.run
arrays = np.load(root / 'chronos_native_predictions.npz')
main = np.load(root / 'test_predictions.npz')
summary = json.loads((root / 'summary.json').read_text())
np.testing.assert_array_equal(arrays['targets'], main['targets'])
np.testing.assert_array_equal(arrays['dates'], main['dates'])
prices = arrays['price_forecasts']
assert prices.shape == (len(arrays['targets']), 4, 5)
assert (prices > 0).all() and np.isfinite(prices).all()
stock = np.log(prices[:, 1, 4]) - np.log(prices[:, 0, 0])
spy = np.log(prices[:, 3, 4]) - np.log(prices[:, 2, 0])
score = stock - spy
np.testing.assert_allclose(score, arrays['predictions'][:, 0], atol=1e-12, rtol=0)
labels = arrays['targets'][:, 1] > .5
accuracy = float(((score >= 0) == labels).mean())
assert accuracy == summary['models']['chronos2_native_forecast']['ensemble']['direction_accuracy']
increments = np.empty((len(prices), 5))
increments[:, 0] = np.log(prices[:, 1, 0]) - np.log(prices[:, 0, 0])
increments[:, 1:] = np.diff(np.log(prices[:, 1]), axis=1)
expected_vol = np.log(np.maximum(np.sqrt(np.mean(increments * increments, axis=1)), 1e-8))
np.testing.assert_allclose(expected_vol, arrays['predictions'][:, 2], atol=1e-10, rtol=0)

sys.path.insert(0, '/var/tmp/mshanthappa-stak-graph/external/chronos/src')
from chronos import Chronos2Pipeline
checkpoint = json.loads((root / 'chronos_native_protocol.json').read_text())['checkpoint']
pipeline = Chronos2Pipeline.from_pretrained(checkpoint['model'], revision=checkpoint['revision'],
    device_map='cuda', local_files_only=True, cache_dir='/tmp/stak-graph-models', dtype=torch.float32)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.manual_seed(4921)
first = {'target': torch.rand(4, 30) + 100,
         'past_covariates': {str(index): torch.randn(30) for index in range(44)}}
unrelated = {'target': torch.rand(4, 30) * 1000,
             'past_covariates': {str(index): torch.randn(30) * 1000 for index in range(44)}}
alone = pipeline.predict_quantiles([first], prediction_length=5, quantile_levels=[.5], context_length=30,
                                    cross_learning=False, batch_size=256)[1][0]
batched = pipeline.predict_quantiles([first, unrelated], prediction_length=5, quantile_levels=[.5], context_length=30,
                                      cross_learning=False, batch_size=256)[1][0]
torch.testing.assert_close(alone, batched, atol=1e-4, rtol=1e-6)
embedding_first = torch.randn(1, 44, 30)
embedding_unrelated = torch.randn(1, 44, 30) * 1000
alone_embedding = pipeline.embed(embedding_first, batch_size=256, context_length=30)[0][0]
batched_embedding = pipeline.embed(torch.cat((embedding_first, embedding_unrelated)), batch_size=256, context_length=30)[0][0]
torch.testing.assert_close(alone_embedding, batched_embedding, atol=1e-4, rtol=1e-5)
record = {'passed': True, 'samples': len(labels), 'accuracy_independently_reconstructed': accuracy,
          'forecast_price_shape': list(prices.shape), 'direction_and_volatility_reconstructed_from_native_forecast_prices': True,
          'test_label_and_date_order_matches_supervised_study': True,
          'unrelated_task_cannot_change_native_forecast_or_frozen_embedding': True,
          'max_native_forecast_batch_difference': float((alone - batched).abs().max()),
          'max_frozen_embedding_batch_difference': float((alone_embedding - batched_embedding).abs().max()),
          'later_pretraining_overlap_with_old_prices_remains_unknown': True}
(root / 'chronos_native_verification.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
