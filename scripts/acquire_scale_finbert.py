"""Resolve and retain a pinned FinBERT checkpoint for the encoder ablation."""
import json
from pathlib import Path
from huggingface_hub import HfApi
from stak_graph.embeddings.finbert import FinBERTBackend

root = Path('/tmp/finstak-100x10-v1')
path = root / 'finbert_checkpoint.json'
if path.exists():
    revision = json.loads(path.read_text())['revision']
else:
    revision = HfApi().model_info('ProsusAI/finbert').sha
    path.write_text(json.dumps({'model': 'ProsusAI/finbert', 'revision': revision,
                    'source': 'https://huggingface.co/ProsusAI/finbert'}, indent=2) + '\n')
backend = FinBERTBackend(revision, device='cpu')
print(json.dumps(backend.identity), flush=True)
