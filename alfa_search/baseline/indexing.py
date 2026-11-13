"""CPU-only preparation of five separate field postings."""

from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from ..config import FIELDS
from ..models import Material
from .preprocessing import Preprocessor


@dataclass
class PreparedIndex:
    materials: list[tuple[str, UUID, list[int]]]
    frequencies: Counter[str]
    postings: list[tuple[str, str, UUID, int, int]]
    averages: list[float]


def prepare_index(materials: list[Material], preprocessor: Preprocessor) -> PreparedIndex:
    rows: list[tuple[str, UUID, list[int]]] = []
    frequencies: Counter[str] = Counter()
    postings: list[tuple[str, str, UUID, int, int]] = []
    seen: set[tuple[str, UUID]] = set()
    totals = [0] * 5
    for material in materials:
        key = (material.type, material.id)
        if key in seen:
            raise ValueError(f"Duplicate source material: {key}")
        seen.add(key)
        lengths = []
        document_terms: set[str] = set()
        for field_id, field in enumerate(FIELDS, 1):
            value = getattr(material, field)
            # Preserve token boundaries across separate labels.
            tokens = (preprocessor.tokens(value) if isinstance(value, str) or value is None
                      else [token for label in value for token in preprocessor.tokens(label)])
            counts = Counter(tokens)
            lengths.append(len(tokens))
            totals[field_id - 1] += len(tokens)
            document_terms.update(counts)
            postings.extend((term, material.type, material.id, field_id, tf)
                            for term, tf in sorted(counts.items()))
        frequencies.update(document_terms)
        rows.append((material.type, material.id, lengths))
    averages = [total / len(rows) if rows else 0.0 for total in totals]
    return PreparedIndex(rows, frequencies, postings, averages)
