from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any


class KwargsConfig(Mapping[str, Any]):
    """Base class for dataclass configs that can be unpacked as kwargs."""

    def to_kwargs(self) -> dict[str, Any]:
        """Return this dataclass config as a plain kwargs dictionary."""
        if not is_dataclass(self):
            raise TypeError("KwargsConfig subclasses must be dataclasses.")
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_kwargs()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_kwargs())

    def __len__(self) -> int:
        return len(self.to_kwargs())
