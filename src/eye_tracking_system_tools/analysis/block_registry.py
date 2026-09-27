"""YAML block registry: animal → list of analyzed block paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BlockSpec:
    animal: str
    block_path: Path
    block_num: str
    left_eye_csv: Path | None = None
    right_eye_csv: Path | None = None

    @property
    def analysis_path(self) -> Path:
        return self.block_path / "analysis"

    @property
    def block_key(self) -> str:
        return f"{self.animal}_block_{self.block_num}"


def _infer_block_num(block_path: Path) -> str:
    name = block_path.name
    m = re.match(r"block[_-]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1).zfill(3)
    digits = re.findall(r"\d+", name)
    if digits:
        return digits[-1].zfill(3)
    return name


def _resolve_eye_override(value: Any, analysis: Path, registry: Path) -> Path | None:
    """Resolve an ``eye_csv`` entry against ``analysis/``; must name an existing file."""
    if value is None:
        return None
    csv_path = Path(str(value)).expanduser()
    if not csv_path.is_absolute():
        csv_path = analysis / csv_path
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"{registry}: eye CSV override does not exist: {csv_path}")
    return csv_path


def load_registry(path: Path | str) -> list[BlockSpec]:
    """
    Load an ``animals:`` registry into a :class:`BlockSpec` list.

    A block entry is either a path or a mapping ``{path, left_eye_csv,
    right_eye_csv}``. The two CSV keys pin a specific trace file instead of
    letting :func:`resolve_eye_csv` pick the newest ``*raw_verified*``; relative
    values are resolved against the block's ``analysis/`` folder.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    animals = data.get("animals")
    if not isinstance(animals, dict):
        raise ValueError(f"{path}: expected top-level 'animals' mapping")

    specs: list[BlockSpec] = []
    for animal, blocks in animals.items():
        if isinstance(blocks, (str, Path, dict)):
            blocks = [blocks]
        if not isinstance(blocks, list):
            raise ValueError(f"{path}: animals[{animal!r}] must be a path or list of paths")
        for bp in blocks:
            if isinstance(bp, dict):
                entry = bp
                raw_path = entry.get("path")
                if raw_path is None:
                    raise ValueError(f"{path}: animals[{animal!r}] block mapping needs a 'path'")
            else:
                entry = {}
                raw_path = bp
            block_path = Path(raw_path).expanduser().resolve()
            if not block_path.exists():
                raise FileNotFoundError(f"Block path does not exist: {block_path}")
            analysis = block_path / "analysis"
            if not analysis.is_dir():
                raise FileNotFoundError(f"Missing analysis/ under {block_path}")
            specs.append(
                BlockSpec(
                    animal=str(animal),
                    block_path=block_path,
                    block_num=_infer_block_num(block_path),
                    left_eye_csv=_resolve_eye_override(entry.get("left_eye_csv"), analysis, path),
                    right_eye_csv=_resolve_eye_override(entry.get("right_eye_csv"), analysis, path),
                )
            )
    return specs
