"""Replay prespecified native foundation forecasts from their historical inputs."""
import json
from pathlib import Path
import numpy as np
import torch
from timesfm3 import TimesFM3Forecaster, ModelConfig
from stak_graph.experiments.timesfm_features import project_candle
from stak_graph.experiments.data import sha256_file

ROOT = Path('/tmp/finstak-100x10-v1')


def main():
    torch.set_num_threads(4)
    plan = json.loads((ROOT / 'native_timesfm/plan.json').read_text())
    if sha256_file(ROOT / 'packed/panel.npz') != plan['panel_sha256']:
        raise ValueError('Native forecast input panel changed')
    panel = np.load(ROOT / 'packed/panel.npz')
    cached = np.load(ROOT / 'native_timesfm/predictions.npz')
    raw, dates, companies = panel['raw'], panel['dates'], panel['companies']
    saved = cached['prediction'].reshape(len(cached['date_ids']), len(companies), 4)
    examples = [('2021-01-04', 'AAPL'), ('2022-06-13', 'MSFT'), ('2023-12-27', 'EQR')]
    contexts, anchors, expected = [], [], []
    for day, company in examples:
        t = int(np.flatnonzero(dates == day)[0])
        c = int(np.flatnonzero(companies == company)[0])
        position = int(np.flatnonzero(cached['date_ids'] == t)[0])
        anchor = raw[t, c, 3]
        contexts.append(np.log(raw[t - plan['context'] + 1:t + 1, c, :4] / anchor).T.astype(np.float32))
        anchors.append(anchor)
        expected.append(saved[position, c])
    model = TimesFM3Forecaster(ModelConfig(checkpoint_path=plan['model'], revision=plan['revision'],
              cache_dir='/tmp/finstak-timesfm-cache', local_files_only=True, device='cuda', per_core_batch_size=16))
    with torch.inference_mode():
        result = list(model.predict_batch(contexts, horizon=1, return_quantiles=False,
                      use_symmetric_averaging=False, make_positive=False))
    log_prediction = project_candle(np.stack([np.asarray(row.forecast)[:, 0] for row in result]))
    price = np.asarray(anchors)[:, None] * np.exp(log_prediction)
    relative = np.abs(price - np.asarray(expected)) / np.asarray(anchors)[:, None]
    maximum = float(relative.max())
    if maximum > 2e-6:
        raise ValueError('Native forecast replay differs from retained output')
    output = {'passed': True, 'examples': examples, 'max_price_difference_fraction_of_anchor': maximum,
              'checkpoint_revision': plan['revision'], 'context': plan['context'], 'future_inputs_used': False}
    (ROOT / 'native_forecast_verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
