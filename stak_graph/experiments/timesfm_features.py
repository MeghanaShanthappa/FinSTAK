"""Causal TimesFM contexts and train-only event covariates for the FinSTAK pilot."""
import numpy as np


def project_candle(log_prices):
    values = np.asarray(log_prices, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError('Expected finite four-price log forecasts')
    values[:, 1] = np.maximum(values[:, 1], np.maximum(values[:, 0], values[:, 3]))
    values[:, 2] = np.minimum(values[:, 2], np.minimum(values[:, 0], values[:, 3]))
    return values


def event_covariates(data, days, train_end, components=4):
    """Training-only PCA; absolute decaying evidence mass plus semantic moments.

    No self-rated confidence is invented. Existing validated edge priors are
    used as evidence weights, not calibrated probabilities. PCA fits only
    events linked to stock days on/before the training boundary.
    """
    get = lambda key: data[key].detach().cpu().numpy()
    source, target, age = get('edge_event'), get('edge_day'), get('edge_age')
    prior, vectors, categories = get('edge_prior'), get('events'), get('categories')
    eligible_days = np.array([d['session_date'] <= train_end for d in days])
    eligible = np.unique(source[eligible_days[target]])
    if len(eligible) < components + 1:
        raise ValueError('Insufficient training events for PCA')
    mean = vectors[eligible].mean(0)
    _, _, basis = np.linalg.svd(vectors[eligible] - mean, full_matrices=False)
    basis = basis[:components]
    semantic = (vectors - mean) @ basis.T
    # Fixed five-session half-life is a prespecified feature policy, not fitted
    # on the test period. Learned decay is separately tested in the hybrid.
    weight = prior * np.exp2(-age / 5.)
    direct = np.zeros((len(days), 6 + components), dtype=np.float32)
    np.add.at(direct, (target, categories[source]), weight)
    for k in range(components):
        np.add.at(direct[:, 6 + k], target, weight * semantic[source, k])
    direct[:, :6] = np.log1p(direct[:, :6])
    cs, ct, cw = get('corr_source'), get('corr_target'), get('corr_weight')
    if any(days[int(s)]['session_date'] != days[int(t)]['session_date'] for s, t in zip(cs, ct)):
        raise ValueError('Peer graph must connect same-date stock days')
    peer = np.zeros_like(direct)
    degree = np.zeros(len(days), dtype=np.float32)
    np.add.at(peer, ct, direct[cs] * cw[:, None])
    np.add.at(degree, ct, cw)
    peer /= np.maximum(degree[:, None], 1)
    scale = direct[eligible_days].std(0)
    scale[scale < 1e-6] = 1
    direct = (direct / scale).clip(-10, 10)
    peer = (peer / scale).clip(-10, 10)
    return direct, peer, {'pca_mean': mean.tolist(), 'pca_basis': basis.tolist(),
                         'scale': scale.tolist(), 'fit_events': eligible.tolist(),
                         'fit_latest_day': max(d['session_date'] for d in days if d['session_date'] <= train_end),
                         'half_life_sessions': 5, 'components': components}


def context_rows(days, sample_rows, length, allow_short=False):
    if length < 1:
        raise ValueError('Positive context length required')
    lookup = {(d['company_id'], d['session_index']): i for i, d in enumerate(days)}
    rows = []
    for last in sample_rows:
        day = days[int(last)]
        keys = [(day['company_id'], j) for j in range(day['session_index'] - length + 1, day['session_index'] + 1)]
        if allow_short:
            while keys and keys[0] not in lookup:
                keys.pop(0)
        if not keys or any(k not in lookup for k in keys):
            raise ValueError('Missing historical context; no future or repeated padding')
        rows.append([lookup[k] for k in keys])
    return [np.asarray(row, dtype=np.int64) for row in rows] if allow_short else np.asarray(rows, dtype=np.int64)
