"""EmbeddingGemma 2 text-only inference using its native Transformers model."""

import hashlib
import importlib.metadata


class EmbeddingGemmaBackend:
    def __init__(self, model="google/embeddinggemma-2", revision=None, device="cuda",
                 cache_dir=None, local_files_only=False, max_tokens=8192):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; explicitly select CPU or use a process with GPU access")
        options = dict(revision=revision, cache_dir=cache_dir, local_files_only=local_files_only, trust_remote_code=False)
        config = AutoConfig.from_pretrained(model, **options)
        if config.model_type != "embedding_gemma2":
            raise ValueError("This backend implements EmbeddingGemma 2 pooling; other encoders require their own backend")
        config.vision_config = None
        config.audio_config = None
        self.tokenizer = AutoTokenizer.from_pretrained(model, **options)
        self.model = AutoModel.from_pretrained(model, config=config, **options, dtype=torch.float32).to(device).eval()
        self.torch, self.device, self.max_tokens = torch, device, max_tokens
        self.dimension = config.text_config.embedding_dim
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.model_name = model
        self.prompt = "task: classification | query: "

    @property
    def identity(self):
        return {
            "model": self.model_name, "revision": self.revision, "backend": "transformers",
            "modality": "text_only", "pooling": "masked_mean_of_projected_tokens",
            "task_prefix": self.prompt, "dimension": self.dimension, "normalized": True,
            "weights_dtype": "float32", "max_tokens": self.max_tokens,
            "torch": self.torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "simulated": False,
        }

    def encode(self, texts):
        batch = self.tokenizer([self.prompt + text for text in texts], padding=True,
                               truncation=False, return_tensors="pt")
        if batch["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("Event text exceeds the encoder token limit; text was not silently truncated")
        batch = batch.to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vectors = self.torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return vectors.cpu().tolist()


class TestHashBackend:
    """Deterministic placeholder for CPU integration tests, never semantic embeddings."""

    def __init__(self, dimension=8):
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.dimension = dimension

    @property
    def identity(self):
        return {"model": "TEST_ONLY_sha256", "revision": "1", "dimension": self.dimension,
                "normalized": True, "simulated": True}

    def encode(self, texts):
        import math
        vectors = []
        for text in texts:
            vector = [int.from_bytes(hashlib.sha256(f"{i}:{text}".encode()).digest()[:4], "little") / 2**31 - 1
                      for i in range(self.dimension)]
            length = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / length for value in vector])
        return vectors
