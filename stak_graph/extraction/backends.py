"""Interchangeable model backends. Heavy ML imports are confined to the local backend."""

from dataclasses import dataclass, field
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .schema import strict_json


class BackendError(RuntimeError):
    pass


@dataclass
class Generation:
    text: str
    finish_reason: str = "stop"
    usage: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    @property
    def identity(self) -> dict[str, Any]: ...

    def generate(self, messages: list[dict[str, str]], schema: dict, request_id: str) -> Generation: ...


class SchemaLogitsProcessor:
    """Use XGrammar's matcher with eager Torch masking, without a Triton build.

    The CLI generates one source chunk at a time. Ordinary Torch bitwise kernels
    keep masking on the selected device and need no Python development headers.
    """

    def __init__(self, torch_module, grammar_module, compiled_grammar):
        self.torch = torch_module
        self.matcher = grammar_module.GrammarMatcher(compiled_grammar)
        self.vocab_size = compiled_grammar.tokenizer_info.vocab_size
        self.bitmask = grammar_module.allocate_token_bitmask(1, self.vocab_size)
        self.prefilled = False
        self.positions = None

    def __call__(self, input_ids, scores):
        if scores.shape != (1, self.vocab_size):
            raise BackendError("Schema decoder requires a single sequence with the model's full vocabulary")
        if self.prefilled:
            if not self.matcher.accept_token(int(input_ids[0, -1])):
                raise BackendError("Generated token violated the compiled JSON grammar")
        self.prefilled = True
        self.matcher.fill_next_token_bitmask(self.bitmask, 0)
        if self.positions is None:
            self.positions = self.torch.arange(self.vocab_size, device=scores.device)
        words = self.bitmask.to(scores.device)[:, self.positions // 32]
        allowed = ((words >> (self.positions % 32)) & 1).bool()
        return scores.masked_fill(~allowed, float("-inf"))


class TransformersBackend:
    """Single-device local generation with an actual JSON-schema token constraint."""

    def __init__(self, model: str = "Qwen/Qwen3-4B", revision: str | None = None,
                 device: str = "cuda", model_cache: str | None = None,
                 max_new_tokens: int = 3000, max_input_tokens: int = 8192,
                 local_files_only: bool = False):
        if max_new_tokens <= 0 or max_input_tokens <= 0:
            raise ValueError("Token limits must be positive")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import xgrammar
        except ImportError as exc:
            raise BackendError(f"Local runtime import failed ({exc}). Install compatible packages with: pip install -e '.[local]'") from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise BackendError("CUDA is unavailable to this process. Use an environment with GPU access, or explicitly select --device cpu.")
        self.torch = torch
        self.model_name = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self._xgrammar = xgrammar
        self._compiled_grammars = {}
        options = {"revision": revision, "cache_dir": model_cache, "local_files_only": local_files_only,
                   "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(model, **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, **options, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.model.config.vocab_size)
        self._grammar_compiler = xgrammar.GrammarCompiler(tokenizer_info)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "transformers", "model": self.model_name, "revision": self.revision,
            "schema_constrained": True, "simulated": False, "do_sample": False,
            "enable_thinking": False, "max_new_tokens": self.max_new_tokens,
            "max_input_tokens": self.max_input_tokens, "device": self.device,
            "torch": self.torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "xgrammar": importlib.metadata.version("xgrammar"),
            "schema_masking": "eager_torch",
        }

    def generate(self, messages: list[dict[str, str]], schema: dict, request_id: str) -> Generation:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_count = inputs["input_ids"].shape[1]
        context_limit = getattr(self.model.config, "max_position_embeddings", 32768)
        if input_count > self.max_input_tokens or input_count + self.max_new_tokens > context_limit:
            raise BackendError(f"Prompt has {input_count} tokens; reduce --chunk-chars or the output token limit. Input was not truncated.")
        schema_text = json.dumps(schema, sort_keys=True)
        if schema_text not in self._compiled_grammars:
            self._compiled_grammars[schema_text] = self._grammar_compiler.compile_json_schema(schema_text)
        processor = SchemaLogitsProcessor(self.torch, self._xgrammar, self._compiled_grammars[schema_text])
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                logits_processor=[processor],
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = output[0, input_count:]
        eos_ids = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids or []
        ended = bool(len(generated)) and int(generated[-1]) in eos_ids
        return Generation(
            self.tokenizer.decode(generated, skip_special_tokens=True),
            "length" if len(generated) >= self.max_new_tokens and not ended else "stop",
            {"input_tokens": input_count, "output_tokens": len(generated),
             "generation_seconds": round(time.monotonic() - started, 3)},
        )


class HTTPBackend:
    """Schema-constrained chat endpoint, including local vLLM servers."""

    def __init__(self, base_url: str, model: str, revision: str | None = None,
                 max_new_tokens: int = 3000, timeout: float = 180):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be HTTP(S), without credentials, query parameters or fragments")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model_name, self.revision = model, revision
        self.max_new_tokens, self.timeout = max_new_tokens, timeout
        if max_new_tokens <= 0 or timeout <= 0:
            raise ValueError("Token limit and timeout must be positive")

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "http", "endpoint": self.url, "model": self.model_name,
            "revision": self.revision, "revision_verification": "operator_supplied",
            "schema_constrained": True, "simulated": False,
            "temperature": 0, "max_new_tokens": self.max_new_tokens,
        }

    def generate(self, messages: list[dict[str, str]], schema: dict, request_id: str) -> Generation:
        payload = {
            "model": self.model_name, "messages": messages, "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "stak_financial_events", "strict": True, "schema": schema,
            }},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("STAK_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(self.url, json.dumps(payload).encode(), headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = strict_json(response.read().decode())
            choice = data["choices"][0]
            content = choice["message"].get("content")
            if not isinstance(content, str):
                raise BackendError("Model returned no textual JSON content")
            return Generation(content, choice.get("finish_reason", "stop"), data.get("usage", {}))
        except HTTPError as exc:
            raise BackendError(f"Extraction endpoint returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
            raise BackendError(f"Extraction endpoint failed: {type(exc).__name__}") from exc


class ReplayBackend:
    """Explicit test fixture replay. Never label this output as model inference."""

    def __init__(self, path: str | Path):
        self.payload = strict_json(Path(path).read_text(encoding="utf-8"))
        if not isinstance(self.payload, dict):
            raise ValueError("Replay responses must be a document-ID-to-response object")
        from .schema import digest
        self.fixture_hash = digest(self.payload)

    @property
    def identity(self) -> dict[str, Any]:
        return {"backend": "replay", "model": "fixture-replay", "revision": self.fixture_hash,
                "schema_constrained": False, "simulated": True}

    def generate(self, messages: list[dict[str, str]], schema: dict, request_id: str) -> Generation:
        document_id, _, chunk_index = request_id.rpartition(":")
        try:
            document = self.payload[document_id]
            result = document[int(chunk_index)] if isinstance(document, list) else document
        except (KeyError, IndexError, ValueError) as exc:
            raise BackendError(f"No replay response for {request_id}") from exc
        return Generation(json.dumps(result, ensure_ascii=False))
