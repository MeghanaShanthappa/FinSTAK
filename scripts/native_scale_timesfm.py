"""Prespecified frozen TimesFM price-only comparator on the expanded panel."""
import json
from pathlib import Path
import time
import numpy as np
import torch

from timesfm3 import TimesFM3Forecaster, ModelConfig
from stak_graph.experiments.timesfm_features import project_candle
from stak_graph.experiments.next_day_data import metrics
from stak_graph.experiments.finstak_data import tolerance_accuracy
from stak_graph.experiments.data import sha256_file
from stak_graph.experiments.scale_io import write_json

ROOT = Path('/tmp/finstak-100x10-v1')


def main():
    torch.set_num_threads(4)
    panel = np.load(ROOT / 'packed/panel.npz')
    raw, dates, companies = panel['raw'], panel['dates'], panel['companies']
    context, batch_size = 128, 16
    output = ROOT / 'native_timesfm'
    chunks = output / 'chunks'
    chunks.mkdir(parents=True, exist_ok=True)
    plan = {'context': context, 'context_selection': 'Prespecified from prior study, no new test tuning',
            'model': 'google/timesfm-3.0-pytorch', 'revision': '43046b85ec22d584a13f8098c2ed39c889e129c2',
            'panel_sha256': sha256_file(ROOT / 'packed/panel.npz'),
            'mode': 'Frozen joint OHLC price-only, no future covariates, no finetuning',
            'pretraining_overlap': 'Unknown; historical dates do not establish a post-pretraining holdout'}
    if (output / 'plan.json').exists() and json.loads((output / 'plan.json').read_text()) != plan:
        raise ValueError('Changed native forecast plan')
    write_json(output / 'plan.json', plan)
    model = TimesFM3Forecaster(ModelConfig(checkpoint_path=plan['model'], revision=plan['revision'],
              cache_dir='/tmp/finstak-timesfm-cache', local_files_only=True, device='cuda', per_core_batch_size=batch_size))
    ids = np.flatnonzero((dates >= '2020-01-01') & (np.arange(len(dates)) < len(dates) - 1))
    start = time.monotonic()
    all_predictions, all_dates = [], []
    for offset in range(0, len(ids), 8):
        group = ids[offset:offset + 8]
        path = chunks / f'{dates[group[0]]}.npz'
        if path.exists():
            cached = np.load(path)
            if not np.array_equal(cached['date_ids'], group):
                raise ValueError('Native cache dates differ')
            predicted = cached['prediction']
        else:
            contexts = [np.log(raw[t - context + 1:t + 1, company, :4] / raw[t, company, 3]).T.astype(np.float32)
                        for t in group for company in range(len(companies))]
            forecasts = []
            for begin in range(0, len(contexts), batch_size):
                with torch.inference_mode():
                    result = list(model.predict_batch(contexts[begin:begin + batch_size], horizon=1,
                                  return_quantiles=False, use_symmetric_averaging=False, make_positive=False))
                forecasts.extend(np.asarray(v.forecast)[:, 0] for v in result)
            original = np.stack(forecasts)
            log_forecast = project_candle(original)
            predicted = raw[group, :, 3].reshape(-1, 1) * np.exp(log_forecast)
            np.savez(path, date_ids=group, raw_log_forecasts=original, prediction=predicted)
        all_predictions.append(predicted)
        all_dates.extend(group)
        if offset % 40 == 0:
            print(json.dumps({'date_chunks_complete': offset + len(group), 'total_dates': len(ids),
                              'seconds': time.monotonic() - start}), flush=True)
    all_dates = np.asarray(all_dates)
    prediction = np.concatenate(all_predictions)
    np.savez(output / 'predictions.npz', date_ids=all_dates, prediction=prediction)
    result = {}
    for year in ('2020', '2021', '2022', '2023'):
        select = np.flatnonzero(np.char.startswith(dates[all_dates], year)
                                & np.char.startswith(dates[all_dates + 1], year))
        selected_dates = all_dates[select]
        p = prediction.reshape(len(all_dates), len(companies), 4)[select].reshape(-1, 4)
        truth = raw[selected_dates + 1, :, :4].reshape(-1, 4)
        anchor = raw[selected_dates, :, 3].reshape(-1)
        result[year] = {'price': metrics(truth, p, anchor, np.tile(companies, len(select))),
                        'tolerance': tolerance_accuracy(truth, p, anchor)}
    write_json(output / 'summary.json', result)
    print(json.dumps({'native_complete': True, 'seconds': time.monotonic() - start}), flush=True)


if __name__ == '__main__':
    main()
