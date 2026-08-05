from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class Checkpoint:
    """Own one checkpoint run's ID, paths, persistence, and registry access."""

    id: str
    architecture_dir: Path
    run_dir: Path
    path: Path
    last_path: Path
    registry_path: Path

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        subdir: str,
        run_id: str | None = None,
        checkpoint_base: str | Path = "/home/junshi/vitalive/model/checkpoints",
    ) -> "Checkpoint":
        """Create a timestamped run directory and its standard paths."""
        checkpoint_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        architecture_dir = Path(project_root) / checkpoint_base / subdir
        run_dir = architecture_dir / checkpoint_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            id=checkpoint_id,
            architecture_dir=architecture_dir,
            run_dir=run_dir,
            path=run_dir / "best.pt",
            last_path=run_dir / "last.pt",
            registry_path=architecture_dir / "registry.jsonl",
        )

    def save(
        self,
        payload: Mapping[str, Any],
        path: str | Path | None = None,
    ) -> Path:
        """Persist a prepared payload; pipeline modules define its schema."""
        destination = Path(path) if path is not None else self.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(payload), destination)
        return destination

    def update_registry(self, record: Mapping[str, Any]) -> Path:
        """Append a run summary after adding standard checkpoint identity fields."""
        registry_record = dict(record)
        registry_record.setdefault("timestamp", self.id)
        registry_record.setdefault("run_id", self.id)
        registry_record.setdefault("run_dir", self.id)
        self.architecture_dir.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(registry_record, default=str) + "\n")
        return self.registry_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load one checkpoint payload."""
        return torch.load(Path(path), map_location=map_location, weights_only=False)

    @staticmethod
    def load_registry(architecture_dir: str | Path) -> list[dict[str, Any]]:
        """Load registry records newest first."""
        registry_path = Path(architecture_dir) / "registry.jsonl"
        if not registry_path.exists():
            return []
        records = [
            json.loads(line)
            for line in registry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return list(reversed(records))

    @classmethod
    def best(
        cls,
        architecture_dir: str | Path,
        metric: str = "test_recon_loss",
    ) -> Path:
        """Return the registered best.pt path with the lowest selected metric."""
        records = cls.load_registry(architecture_dir)
        candidates = [record for record in records if record.get(metric) is not None]
        if not candidates:
            raise FileNotFoundError(
                f"No registry records with metric {metric!r} in {architecture_dir}."
            )
        record = min(candidates, key=lambda item: item[metric])
        return Path(architecture_dir) / record["run_dir"] / "best.pt"


__all__ = ["Checkpoint"]
