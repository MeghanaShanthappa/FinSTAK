"""Source-linked event-deletion sensitivity for the two trainable graph routes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stak_graph.experiments.finstak_run import prepare
from stak_graph.experiments.train import predict
from stak_graph.experiments.model_sweep import save
from stak_graph.io import read_jsonl
from stak_graph.graph.market import moment
from stak_graph.models.timesfm_hybrid import TimesFMFinSTAK
from stak_graph.models.timesfm_encoder import TimesFMEncoderFinSTAK


def without_events(data, event=None):
    changed = dict(data)
    keep = (data['edge_event'] != event) if event is not None else torch.zeros_like(data['edge_event'], dtype=torch.bool)
    for key in ('edge_event', 'edge_day', 'edge_age', 'edge_relation', 'edge_prior', 'edge_weight'):
        changed[key] = data[key][keep]
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.device = 'cuda'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((args.primary / 'plan.json').read_text())
    study, original_data, snapshot, _ = prepare(args, plan['config'])
    test_rows = np.flatnonzero(study['masks']['test'])
    test = torch.as_tensor(test_rows, device='cuda')
    windows = torch.as_tensor(study['windows'], device='cuda')
    anchors = study['anchors'][test_rows]
    sources = {e['event_id']: e for e in read_jsonl(args.expanded / 'events.jsonl')}
    result = {'interpretation': 'Model sensitivity to deleting event messages; realized historical prices and frozen foundation representations stay fixed. These are not causal effects.', 'routes': {}}
    for route, output, cls in (('hybrid', args.primary, TimesFMFinSTAK), ('encoder', args.encoder, TimesFMEncoderFinSTAK)):
        data = dict(original_data)
        day_rows = torch.as_tensor(study['windows'][:, -1], device='cuda')
        if route == 'hybrid':
            base = np.load(output / 'native/selected_price_all.npz')['prediction']
            prior = torch.zeros((len(data['prices']), 4), device='cuda')
            prior[day_rows] = torch.as_tensor(base, dtype=torch.float32, device='cuda')
            data['forecast_prior'] = prior
            data['prices'] = torch.cat((data['prices'], prior / torch.tensor(study['scaler']['target_scale'], device='cuda')), 1)
        else:
            reduced = np.load(output / 'frozen_encoder.npz')['reduced']
            data['temporal_embeddings'] = torch.zeros((len(data['prices']), 32), device='cuda')
            data['temporal_embeddings'][day_rows] = torch.as_tensor(reduced, device='cuda')
        models, full, deleted = [], [], []
        for seed in plan['config']['seeds']:
            model = cls(study['scaler'], len(data['company_ids']), mode='graph').to('cuda')
            model.load_state_dict(torch.load(output / 'checkpoints' / f'{route}_graph_seed{seed}.pt', weights_only=True, map_location='cuda')['state_dict'])
            models.append(model)
            full.append(anchors[:, None] * np.exp(predict(model, data, windows, test).astype('float64')))
            deleted.append(anchors[:, None] * np.exp(predict(model, without_events(data), windows, test).astype('float64')))
        full, deleted = np.mean(full, 0), np.mean(deleted, 0)
        saved = np.load(output / 'test_predictions.npz', allow_pickle=False)
        np.testing.assert_allclose(full, saved[route + '_graph'], rtol=1e-7, atol=1e-7)
        effect = 100 * (full - deleted) / anchors[:, None]
        ranked = np.argsort(np.abs(effect).mean(1))[::-1][:3]
        examples = []
        for i in ranked:
            sample_index = int(test_rows[i]); row = int(study['windows'][sample_index, -1])
            peer_nodes = data['corr_source'][data['corr_target'] == row].tolist()
            relevant = torch.tensor([row] + peer_nodes, device='cuda')
            edges = torch.isin(data['edge_day'], relevant)
            candidates = sorted(set(data['edge_event'][edges].tolist()), key=lambda e: snapshot['nodes']['event'][e]['known_at'], reverse=True)[:4]
            evidence = []
            one = torch.tensor([sample_index], device='cuda')
            for event in candidates:
                source = sources[snapshot['nodes']['event'][event]['id']]
                assert moment(source['known_at']) <= moment(study['metadata'][sample_index]['prediction_at'])
                removed = []
                for model in models:
                    removed.append(anchors[i] * np.exp(predict(model, without_events(data, event), windows, one).astype('float64')[0]))
                change = 100 * (full[i] - np.mean(removed, 0)) / anchors[i]
                evidence.append({'event_id': source['event_id'], 'description': source['description'],
                                 'source_url': source.get('source_url'), 'known_at': source['known_at'],
                                 'category': source['category'], 'source_hash': source['source_hash'],
                                 'forecast_shift_percent_of_close_OHLC': change.tolist()})
            examples.append({'company_id': study['metadata'][sample_index]['company_id'],
                             'observed_close': float(anchors[i]),
                             'prediction_date': study['metadata'][sample_index]['session_date'],
                             'target_date': study['metadata'][sample_index]['target_date'],
                             'full_forecast_OHLC': full[i].tolist(), 'without_all_events_OHLC': deleted[i].tolist(),
                             'all_event_shift_percent_of_close_OHLC': effect[i].tolist(),
                             'individual_source_deletions': evidence})
        result['routes'][route] = {'mean_abs_event_forecast_shift_percent_of_close': float(np.abs(effect).mean()),
                                   'examples': examples, 'candidate_policy': 'Three largest all-event forecast shifts; up to four most recent eligible direct/one-hop sources per example'}
        del models
        torch.cuda.empty_cache()
    save(args.output, result)
    print(json.dumps({key: value['mean_abs_event_forecast_shift_percent_of_close'] for key, value in result['routes'].items()}))


if __name__ == '__main__':
    main()
