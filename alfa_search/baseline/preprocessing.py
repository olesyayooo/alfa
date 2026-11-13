"""Shared, frequency-preserving preprocessing for documents and queries."""

import re
from functools import lru_cache
from importlib.metadata import version

import pymorphy3

from ..text import clean_text

STOP_WORDS = frozenset("и в во на с со к ко по из за у о об от до для а но или как что это при через без не the a an and or of to in for with".split())
TERMS = frozenset({"ии", "ип", "ооо", "ндс", "усн", "осно", "аусн", "псн", "есхн", "рф"})


class Preprocessor:
    def __init__(self) -> None:
        self._morph = pymorphy3.MorphAnalyzer()
        self.signature = "v2:" + version("pymorphy3") + ":" + version("pymorphy3-dicts-ru")

    @lru_cache(maxsize=50000)
    def _lemma(self, token: str) -> str:
        if token in TERMS or not re.fullmatch("[а-я]+", token):
            return token
        return self._morph.parse(token)[0].normal_form.replace("ё", "е")

    def tokens(self, value: str | None) -> list[str]:
        text = clean_text(value).lower().replace("ё", "е")
        words = re.findall(r"[а-яa-z0-9]+", text)
        return [lemma for word in words if word not in STOP_WORDS
                for lemma in [self._lemma(word)] if lemma not in STOP_WORDS]
