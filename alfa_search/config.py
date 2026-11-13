"""Ranking settings can change without rebuilding the index."""

import math
from dataclasses import dataclass

FIELDS = ("title", "description", "topics", "tags", "industries")


@dataclass(frozen=True)
class SearchConfig:
    weights: tuple[float, ...] = (3.0, 1.0, 2.0, 2.0, 1.5)
    b: tuple[float, ...] = (0.75, 0.75, 0.75, 0.75, 0.75)
    k1: float = 1.2
    exclude_closed: bool = True
    public_articles_only: bool = True

    def __post_init__(self) -> None:
        if len(self.weights) != 5 or any(not math.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError("Provide five finite positive field weights")
        if len(self.b) != 5 or any(not math.isfinite(b) or not 0 <= b <= 1 for b in self.b):
            raise ValueError("Provide five field b values in [0, 1]")
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and positive")
