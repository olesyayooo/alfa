from dataclasses import dataclass
from typing import TypedDict
from uuid import UUID


@dataclass(frozen=True)
class Material:
    id: UUID
    type: str
    title: str | None = None
    description: str | None = None
    topics: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    industries: tuple[str, ...] = ()


class SearchResult(TypedDict):
    material_id: UUID
    material_type: str
    score: float
