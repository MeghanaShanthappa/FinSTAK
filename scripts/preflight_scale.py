"""Real-data GPU smoke check; never a completed experiment or accuracy claim."""
import json
from pathlib import Path
import time
import numpy as np
import torch

from stak_graph.io import read_jsonl
from stak_graph.experiments.scale_io import write_jsonl, write_json
from stak_graph.embeddings.backend import EmbeddingGemmaBackend
from stak_graph.embeddings.pipeline import VectorCache, embed_events
from stak_graph.experiments.scale_panel import build_event_arrays, PanelBatches
from stak_graph.models.scale_finstak import ScaleFinSTAK, VARIANTS, task_losses, pcgrad_backward


def main():
    root = Path('/tmp/finstak-100x10-v1')
    expanded = root / 'preflight_expanded'
    expanded.mkdir(exist_ok=True)
    events = sorted(read_jsonl(root / 'dataset/events.jsonl'), key=lambda e: e['event_id'])[:512]
    for event in events:
        quote = event['companies'][0]['evidence']
        event['primary_discloser_verified'] = event['description'].lower().startswith(quote.lower())
        event['primary_discloser_verification_level'] = 'headline_leading_quoted_entity_only'
    write_jsonl(expanded / 'events.jsonl', events)
    torch.set_num_threads(4)
    if not (expanded / 'embeddings').exists():
        backend = EmbeddingGemmaBackend(revision='f9972ac22e696a105c482202646979e33e4b5a1e',
                                       cache_dir='/tmp/stak-graph-models', local_files_only=True)
        cache = VectorCache(root / 'vectors.sqlite3')
        try:
            embed_events(events, backend, expanded / 'embeddings', cache=cache, batch_size=64)
        finally:
            cache.close()
        del backend
        torch.cuda.empty_cache()
    packed = root / 'preflight_packed'
    packed.mkdir(exist_ok=True)
    if not (packed / 'panel.npz').exists():
        (packed / 'panel.npz').symlink_to(root / 'packed/panel.npz')
    build_event_arrays(root / 'dataset', expanded, packed)
    panel = PanelBatches(packed, '2019-12-31', '2020-12-31', '2021-12-31')
    result = {}
    for variant in VARIANTS:
        started = time.monotonic()
        model = ScaleFinSTAK(panel.scaler, panel.c, panel.events.shape[1], variant, category_count=panel.category_count).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for ids in [panel.splits['train'][100:108], panel.splits['train'][-8:]]:
            data = panel.batch(ids)
            output = model(data)
            losses = task_losses(output, data, panel.scaler, model)
            optimizer.zero_grad(set_to_none=True)
            if variant in ('volume_pcgrad', 'combined'):
                pcgrad_backward(losses, model.parameters(), torch.Generator().manual_seed(7))
            else:
                torch.stack(losses).mean().backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError(f'Nonfinite gradients: {variant}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        result[variant] = {'two_updates_finite': True, 'seconds': time.monotonic() - started}
        print(json.dumps({variant: result[variant]}), flush=True)
        del model, optimizer
    write_json(root / 'gpu_preflight.json', {'mode': '512-event real-data smoke check, not full-study results',
               'stocks': panel.c, 'variants': result, 'peak_GPU_bytes': torch.cuda.max_memory_allocated(),
               'primary_mask': 'Automated headline-leading quoted entity check, not human certification'})


if __name__ == '__main__':
    main()
