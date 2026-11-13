"""Lazy local dense BGE-M3 adapter; no weights are loaded on import."""

import logging
import math
import threading
from dataclasses import dataclass
from typing import Protocol

from .config import DIMENSION, SemanticConfig

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Embedding:
    vector: list[float]
    original_tokens: int
    used_tokens: int

    @property
    def truncated(self) -> bool:
        return self.used_tokens < self.original_tokens


class Encoder(Protocol):
    @property
    def signature(self) -> str: ...

    def encode(self, texts: list[str]) -> list[Embedding]: ...


def normalized_vector(vector: list[float]) -> list[float]:
    if len(vector) != DIMENSION or not all(math.isfinite(v) for v in vector):
        raise ValueError("Embedding must contain 1024 finite values")
    norm = math.sqrt(sum(v * v for v in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Zero embeddings cannot be used for cosine search")
    return [float(v / norm) for v in vector]


def fit_text(text: str, tokenizer, limit: int) -> tuple[str, int, int]:
    """Explicit prefix budget, with re-tokenization to prevent implicit truncation."""
    original = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    if original <= limit:
        return text, original, original
    budget = limit - tokenizer.num_special_tokens_to_add(pair=False)
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    candidate = tokenizer.decode(ids[:budget], skip_special_tokens=True)
    used = len(tokenizer.encode(candidate, add_special_tokens=True, truncation=False))
    # Decoding and re-encoding can shift token boundaries; validate the final text.
    while used > limit:
        candidate = candidate[:max(0, len(candidate) - max(1, len(candidate) // 20))]
        used = len(tokenizer.encode(candidate, add_special_tokens=True, truncation=False))
    return candidate, original, used


class BGEEncoder:
    def __init__(self, config: SemanticConfig) -> None:
        self.config = config
        self._model = None
        self._lock = threading.RLock()

    @property
    def signature(self) -> str:
        return self.config.signature

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelLoadError("Install requirements-semantic.txt before loading BGE-M3") from exc
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info("Loading %s@%s on %s (offline=%s)", self.config.model_name,
                    self.config.model_revision, device, self.config.local_files_only)
        try:
            model = SentenceTransformer(self.config.model_name, revision=self.config.model_revision,
                device=device, cache_folder=self.config.cache_folder,
                local_files_only=self.config.local_files_only, trust_remote_code=False)
            if model.get_sentence_embedding_dimension() != DIMENSION:
                raise ValueError("Model must produce dense embeddings of dimension 1024")
            if self.config.max_tokens > model.max_seq_length:
                raise ValueError(f"max_tokens exceeds model limit {model.max_seq_length}")
            model.max_seq_length = self.config.max_tokens
            model.eval()
        except Exception as exc:
            raise ModelLoadError("Cannot load BGE-M3; check dependencies, model cache, network and device. "
                                 "For MPS errors try SEMANTIC_DEVICE=cpu") from exc
        self._model = model
        return model

    def encode(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        # All calls, including initial load, are serialized for this model instance.
        with self._lock:
            model = self._load()
            prepared = [fit_text(text, model.tokenizer, self.config.max_tokens) for text in texts]
            vectors = model.encode([p[0] for p in prepared], batch_size=self.config.batch_size,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, prompt="")
            return [Embedding(normalized_vector(vector.tolist()), original, used)
                    for vector, (_, original, used) in zip(vectors, prepared, strict=True)]

    def warmup(self) -> None:
        with self._lock:
            self._load()
