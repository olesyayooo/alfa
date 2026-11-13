"""Canonical text without BM25 lemmatization or stop-word removal."""

import hashlib

from ..models import Material
from ..text import clean_text


def _labels(values: tuple[str, ...]) -> str:
    labels = {text for value in values for text in [clean_text(value)] if text}
    return ", ".join(sorted(labels))


def material_text(material: Material) -> str:
    # Priority order: title and metadata survive trimming of long descriptions.
    fields = (("Название", clean_text(material.title)), ("Темы", _labels(material.topics)),
              ("Теги", _labels(material.tags)), ("Отрасли", _labels(material.industries)),
              ("Описание", clean_text(material.description)))
    return "\n\n".join(f"{label}: {value}" for label, value in fields if value)


def content_hash(text: str) -> str:
    # Hash the entire source text, including the portion potentially truncated.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
