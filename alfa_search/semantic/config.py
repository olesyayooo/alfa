"""Semantic configuration; immutable model revision prevents silent drift."""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DIMENSION = 1024
TEXT_VERSION = "material-fields-v1"


@dataclass(frozen=True)
class SemanticConfig:
    model_name: str = "BAAI/bge-m3"
    model_revision: str = field(default_factory=lambda: os.getenv("SEMANTIC_MODEL_REVISION", MODEL_REVISION))
    batch_size: int = 8
    max_tokens: int = 2048
    device: str = field(default_factory=lambda: os.getenv("SEMANTIC_DEVICE", "auto"))
    cache_folder: str | None = field(default_factory=lambda: os.getenv("SEMANTIC_MODEL_CACHE"))
    local_files_only: bool = field(default_factory=lambda: os.getenv("SEMANTIC_OFFLINE", "0") == "1")

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-f0-9]{40}", self.model_revision):
            raise ValueError("model_revision must be an immutable Hugging Face commit SHA (40 hex characters)")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or not 8 <= self.max_tokens <= 8192:
            raise ValueError("max_tokens must be in [8, 8192]")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps or cuda")

    @property
    def signature(self) -> str:
        value = {"model": self.model_name, "revision": self.model_revision,
                 "max_tokens": self.max_tokens, "text_version": TEXT_VERSION,
                 "dimension": DIMENSION, "normalized": True, "encoding": "dense-float32-v1"}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
