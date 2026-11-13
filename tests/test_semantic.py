import asyncio
import sys
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from alfa_search import SemanticConfig, SemanticSearch
from alfa_search.models import Material
from alfa_search.semantic.encoder import BGEEncoder, Embedding, fit_text, normalized_vector
from alfa_search.semantic.repository import MissingPgvectorError, SemanticRepository
from alfa_search.semantic.text import content_hash, material_text


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True, truncation=False):
        ids = [ord(char) for char in text]
        return [0, *ids, 0] if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class CountingEncoder:
    def __init__(self, config):
        self.signature = config.signature
        self.calls = []
        self.fail_at = None

    def encode(self, texts):
        self.calls.append(list(texts))
        if self.fail_at == len(self.calls):
            raise RuntimeError("Injected model failure")
        embeddings = []
        for text in texts:
            lower = text.lower()
            if "смешанный" in lower:
                vector = [0.6, 0.8]
            elif "маркетинг" in lower:
                vector = [1.0, 0.0]
            elif "налог" in lower:
                vector = [0.0, 1.0]
            else:
                vector = [0.0, 0.0, 1.0]
            vector += [0.0] * (1024 - len(vector))
            tokens = len(text.split()) + 2
            embeddings.append(Embedding(vector, tokens, tokens))
        return embeddings


def test_material_text_and_deduplication():
    material = Material(UUID(int=1), "course", "<b>Как привлечь клиентов?</b>",
                        "Для бизнеса и команды", ("Продажи", " Продажи ", "<b>Продажи</b>"),
                        ("CRM", "CRM"), ("Торговля",))
    text = material_text(material)
    assert text == ("Название: Как привлечь клиентов?\n\nТемы: Продажи\n\nТеги: CRM\n\n"
                    "Отрасли: Торговля\n\nОписание: Для бизнеса и команды")
    assert "Для" in text and "клиентов" in text  # no lowercase/lemmatization/stop-word deletion
    assert material_text(Material(UUID(int=1), "article")) == ""
    assert content_hash(text) == content_hash(text)
    assert content_hash(text + "!") != content_hash(text)


def test_token_limit_preserves_prefix_and_reports_trimming():
    tokenizer = CharacterTokenizer()
    text = "Название: ИИ\n\nТеги: CRM\n\nОписание: " + "длинное описание " * 50
    result, original, used = fit_text(text, tokenizer, 64)
    assert result.startswith("Название: ИИ\n\nТеги: CRM")
    assert used <= 64 < original
    assert fit_text("ИИ", tokenizer, 64) == ("ИИ", 4, 4)


def test_vector_validation_and_configuration():
    assert normalized_vector([2.0] + [0.0] * 1023) == [1.0] + [0.0] * 1023
    for vector in ([1.0], [0.0] * 1024, [float("nan")] * 1024):
        with pytest.raises(ValueError):
            normalized_vector(vector)
    for kwargs in ({"model_revision": "main"}, {"max_tokens": 9000}, {"batch_size": 0}, {"device": "bad"}):
        with pytest.raises(ValueError):
            SemanticConfig(**kwargs)
    config = SemanticConfig()
    assert config.signature != replace(config, model_revision="a" * 40).signature
    assert config.signature != replace(config, max_tokens=1024).signature
    assert config.signature == replace(config, batch_size=1).signature


def test_real_adapter_loads_model_once_and_requests_dense_normalized_vectors(monkeypatch):
    loaded = []
    arguments = []
    class FakeVector(list):
        def tolist(self):
            return list(self)
    class FakeSentenceTransformer:
        max_seq_length = 8192
        tokenizer = CharacterTokenizer()
        def __init__(self, name, **kwargs):
            loaded.append((name, kwargs))
        def get_sentence_embedding_dimension(self):
            return 1024
        def eval(self):
            return self
        def encode(self, texts, **kwargs):
            arguments.append(kwargs)
            return [FakeVector([2.0] + [0.0] * 1023) for _ in texts]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    encoder = BGEEncoder(SemanticConfig(max_tokens=64, local_files_only=True))
    assert loaded == []
    first = encoder.encode(["a" * 100])[0]
    second = encoder.encode(["короткий"])[0]
    assert len(loaded) == 1
    assert loaded[0][1]["device"] == "mps"
    assert loaded[0][1]["local_files_only"] is True
    assert loaded[0][1]["trust_remote_code"] is False
    assert first.truncated and first.used_tokens <= 64
    assert not second.truncated
    assert len(first.vector) == 1024 and first.vector[0] == 1
    assert all(kwargs["normalize_embeddings"] for kwargs in arguments)


async def test_empty_queries_and_top_k_do_not_load_model_or_connect():
    service = SemanticSearch("postgresql://unused", encoder=CountingEncoder(SemanticConfig()))
    service._repository.search = AsyncMock(side_effect=AssertionError("Unexpected DB access"))
    for query in ("", "  ", "<p> </p>"):
        assert await service.search(query) == []
    assert await service.search("налог", 0) == []
    assert await service.search("налог", -1) == []
    for top_k in (True, 2.5, "5"):
        with pytest.raises(TypeError):
            await service.search("налог", top_k)
    with pytest.raises(TypeError):
        await service.search(None)


async def test_model_compute_does_not_block_event_loop():
    config = SemanticConfig()
    encoder = CountingEncoder(config)
    original = encoder.encode
    def slow_encode(texts):
        time.sleep(0.1)
        return original(texts)
    encoder.encode = slow_encode
    service = SemanticSearch("postgresql://unused", config, encoder=encoder)
    task = asyncio.create_task(service._encode(["маркетинг"]))
    await asyncio.sleep(0.01)
    assert not task.done()
    assert len((await task)[0].vector) == 1024


async def test_missing_pgvector_errors_are_actionable():
    repository = SemanticRepository("postgresql://unused")
    connection = SimpleNamespace(fetchval=AsyncMock(side_effect=[None, None]))
    with pytest.raises(MissingPgvectorError, match="server files are missing"):
        await repository._check_vector(connection, allow_install=True)
    connection.fetchval = AsyncMock(side_effect=[None, "0.8.0"])
    with pytest.raises(MissingPgvectorError, match="not enabled"):
        await repository._check_vector(connection, allow_install=False)
