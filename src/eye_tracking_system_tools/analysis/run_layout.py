"""Organized run output dirs: figures/ + metadata/, never under reproduction/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPRODUCTION_MARKER = Path("figures") / "reproduction"


@dataclass(frozen=True)
class RunDirs:
    run_dir: Path
    figures_dir: Path
    metadata_dir: Path

    def ensure(self) -> "RunDirs":
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        return self


def assert_not_reproduction(path: Path) -> None:
    """Raise if ``path`` resolves under the archived reproduction tree."""
    resolved = Path(path).resolve()
    parts = resolved.parts
    for i in range(len(parts) - 1):
        if parts[i] == "figures" and parts[i + 1] == "reproduction":
            raise ValueError(
                f"Refusing to write under figures/reproduction: {resolved}\n"
                "Copy baselines out to outputs/<run>/figures/ instead."
            )
    # Also catch .../eye_tracking_system_tools/figures/reproduction/...
    text = str(resolved)
    if "/figures/reproduction/" in text or text.endswith("/figures/reproduction"):
        raise ValueError(
            f"Refusing to write under figures/reproduction: {resolved}"
        )


def resolve_run_name(
    tag: str | None = None,
    *,
    prefix: str = "phase2",
    default_name: str = "phase2_latest",
) -> str:
    """
    Default (empty/None tag) → overwrite working folder ``default_name``.
    Non-empty tag → ``{prefix}_{tag}`` unique snapshot folder.
    """
    if tag is None or str(tag).strip() == "":
        return default_name
    safe = str(tag).strip().replace(" ", "_")
    if safe.startswith(f"{prefix}_"):
        return safe
    return f"{prefix}_{safe}"


def resolve_figure_dirs(out_dir: Path | str) -> tuple[Path, Path]:
    """
    Return ``(figures_dir, metadata_dir)`` for a run root or either child.

    Creates ``figures/`` and ``metadata/`` under a run root when needed.
    """
    out_dir = Path(out_dir)
    if out_dir.name == "figures":
        figures_dir, metadata_dir = out_dir, out_dir.parent / "metadata"
    elif out_dir.name == "metadata":
        figures_dir, metadata_dir = out_dir.parent / "figures", out_dir
    else:
        figures_dir, metadata_dir = out_dir / "figures", out_dir / "metadata"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    assert_not_reproduction(figures_dir)
    assert_not_reproduction(metadata_dir)
    return figures_dir, metadata_dir


def resolve_run_dir(
    out_root: Path | str,
    tag: str | None = None,
    *,
    prefix: str = "phase2",
    default_name: str = "phase2_latest",
) -> RunDirs:
    """
    Resolve ``outputs/<run_name>/{figures,metadata}/``.

    * No tag → ``phase2_latest`` (overwrite).
    * ``--tag paper_events_v2`` → ``phase2_paper_events_v2``.
    """
    out_root = Path(out_root)
    run_name = resolve_run_name(tag, prefix=prefix, default_name=default_name)
    run_dir = out_root / run_name
    assert_not_reproduction(run_dir)
    dirs = RunDirs(
        run_dir=run_dir,
        figures_dir=run_dir / "figures",
        metadata_dir=run_dir / "metadata",
    )
    return dirs.ensure()
