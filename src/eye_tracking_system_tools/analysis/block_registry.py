"""YAML block registry: animal → list of analyzed block paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

PAPER_REGISTRY_HEADER = """\
# Paper / flexible-figures block registry (animals → analyzed block paths).
# Produced by development/flexible_paper_figures_tool.ipynb (section 0.5)
# or edited by hand. Each path must contain an analysis/ folder with eye CSVs.
#
"""


@dataclass(frozen=True)
class BlockSpec:
    animal: str
    block_path: Path
    block_num: str

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


def _infer_animal(block_path: Path | str) -> str:
    """Best-effort animal id from a block path (``.../PV_143/date/block_001``)."""
    parts = Path(block_path).resolve().parts
    for part in parts:
        if re.match(r"^(PV|M|Turtle)_\d+", part, flags=re.IGNORECASE):
            return part
    return Path(block_path).parent.parent.name


def read_paper_registry(path: Path | str) -> list[BlockSpec]:
    """Tolerant read of an ``animals:`` registry (missing file → ``[]``; no FS checks)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    animals = data.get("animals")
    if not isinstance(animals, dict):
        return []
    specs: list[BlockSpec] = []
    for animal, blocks in animals.items():
        if isinstance(blocks, (str, Path)):
            blocks = [blocks]
        if not isinstance(blocks, list):
            continue
        for bp in blocks:
            block_path = Path(bp).expanduser()
            specs.append(
                BlockSpec(
                    animal=str(animal),
                    block_path=block_path,
                    block_num=_infer_block_num(block_path),
                )
            )
    return specs


def write_paper_registry(
    path: Path | str,
    specs: Iterable[Any],
    *,
    header: bool = True,
) -> Path:
    """
    Write an ``animals: {name: [block paths]}`` registry.

    ``specs`` may be :class:`BlockSpec`, jitter specs, or any objects with
    ``.animal`` / ``.block_path`` attributes (or ``(animal, path)`` tuples).
    De-duplicates on ``block_path``; animal order follows first appearance.
    """
    path = Path(path)
    animals: dict[str, list[str]] = {}
    seen: set[str] = set()
    for spec in specs:
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            animal, block_path = str(spec[0]), Path(spec[1])
        else:
            animal = str(getattr(spec, "animal", None) or _infer_animal(spec.block_path))
            block_path = Path(spec.block_path)
        key = str(block_path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        animals.setdefault(animal, []).append(key)

    body = yaml.safe_dump({"animals": animals}, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(PAPER_REGISTRY_HEADER)
        f.write(body)
    return path


def load_registry(path: Path | str) -> list[BlockSpec]:
    """Load ``configs/sample_blocks.yaml``-style registry into BlockSpec list."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    animals = data.get("animals")
    if not isinstance(animals, dict):
        raise ValueError(f"{path}: expected top-level 'animals' mapping")

    specs: list[BlockSpec] = []
    for animal, blocks in animals.items():
        if isinstance(blocks, (str, Path)):
            blocks = [blocks]
        if not isinstance(blocks, list):
            raise ValueError(f"{path}: animals[{animal!r}] must be a path or list of paths")
        for bp in blocks:
            block_path = Path(bp).expanduser().resolve()
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
                )
            )
    return specs


def iter_by_animal(specs: list[BlockSpec]) -> Iterator[tuple[str, list[BlockSpec]]]:
    seen: dict[str, list[BlockSpec]] = {}
    for s in specs:
        seen.setdefault(s.animal, []).append(s)
    for animal, group in seen.items():
        yield animal, group


def registry_summary(specs: list[BlockSpec]) -> dict[str, Any]:
    return {
        "n_blocks": len(specs),
        "animals": sorted({s.animal for s in specs}),
        "blocks": [
            {
                "animal": s.animal,
                "block_num": s.block_num,
                "block_path": str(s.block_path),
            }
            for s in specs
        ],
    }
