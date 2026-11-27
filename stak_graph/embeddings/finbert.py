"""Frozen FinBERT token-mean embeddings for a controlled encoder comparison."""
import importlib.metadata


def chunked_batch(tokenizer, texts):
    """Keep every overflow segment using the installed tokenizer's public API."""
    batch = tokenizer(texts, padding=True, truncation=True, max_length=512,
                      return_overflowing_tokens=True, return_special_tokens_mask=True, return_tensors='pt')
    if 'overflow_to_sample_mapping' not in batch:
        raise ValueError('Tokenizer did not provide complete overflow ownership')
    owners = batch.pop('overflow_to_sample_mapping')
    special = batch.pop('special_tokens_mask')
    content_counts = ((batch['attention_mask'] == 1) & (special == 0)).sum(1)
    return batch, owners, content_counts


class FinBERTBackend:
    def __init__(self, revision, cache_dir='/tmp/stak-graph-models', device='cuda', local_files_only=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device, self.revision = torch, device, revision
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert', revision=revision,
                            cache_dir=cache_dir, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained('ProsusAI/finbert', revision=revision,
                            cache_dir=cache_dir, local_files_only=local_files_only).to(device).eval()
        self.dimension = self.model.config.hidden_size
        self.model.requires_grad_(False)

    @property
    def identity(self):
        return {'model': 'ProsusAI/finbert', 'revision': self.revision, 'dimension': self.dimension,
                'backend': 'transformers', 'pooling': 'masked_token_mean_with_token_length_weighted_chunks', 'normalized': True,
                'max_tokens': 512, 'simulated': False, 'torch': self.torch.__version__,
                'transformers': importlib.metadata.version('transformers')}

    def encode(self, texts):
        batch, owners, _ = chunked_batch(self.tokenizer, texts)
        batch = batch.to(self.device)
        with self.torch.inference_mode():
            values = self.model(**batch).last_hidden_state.float()
            mask = batch['attention_mask'][..., None]
            total = values.new_zeros((len(texts), self.dimension))
            counts = values.new_zeros((len(texts), 1))
            owner = owners.to(self.device)
            total.index_add_(0, owner, (values * mask).sum(1))
            counts.index_add_(0, owner, mask.sum(1).to(values.dtype))
            pooled = total / counts.clamp_min(1)
            pooled = self.torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.cpu().tolist()
