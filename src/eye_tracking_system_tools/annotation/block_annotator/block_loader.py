"""Load BlockSync-backed sessions for the annotator."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from eye_tracking_system_tools.annotation.block_annotator.models import (
    AnnotatorConfig,
    BlockSession,
    compute_ms_axis,
)
from eye_tracking_system_tools.preprocessing.block_sync_core import load_final_sync_df


_DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")
_BLOCK_RE = re.compile(r"^block_(\d+)$", re.IGNORECASE)


def find_blocks_with_sync(root: Path) -> list[Path]:
    """Return block folders containing analysis/final_sync_df.csv (or blocksync_df)."""
    root = Path(root)
    if (root / "analysis" / "final_sync_df.csv").exists() or (
        root / "analysis" / "blocksync_df.csv"
    ).exists():
        return [root]
    found: list[Path] = []
    for p in root.rglob("analysis"):
        if p.is_dir():
            if (p / "final_sync_df.csv").exists() or (p / "blocksync_df.csv").exists():
                found.append(p.parent)
    return sorted(set(found))


def infer_metadata(block_path: Path) -> tuple[str, str | None, str, Path]:
    """
    Infer animal_call, experiment_date, block_num, path_to_animal_folder from path.

    Expected layouts:
      .../animal/yyyy_mm_dd/block_NNN
      .../animal/block_NNN
    """
    block_path = Path(block_path).resolve()
    m_block = _BLOCK_RE.match(block_path.name)
    if not m_block:
        raise ValueError(f"Folder name must be block_NNN, got: {block_path.name}")
    block_num = m_block.group(1).zfill(3) if len(m_block.group(1)) <= 3 else m_block.group(1)

    parent = block_path.parent
    if _DATE_RE.match(parent.name):
        experiment_date = parent.name
        animal_call = parent.parent.name
        path_to_animal = parent.parent.parent
    else:
        experiment_date = None
        animal_call = parent.name
        path_to_animal = parent.parent

    return animal_call, experiment_date, block_num, path_to_animal


def discover_block_videos(
    block_path: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Public wrapper: arena, left-eye, right-eye mp4 paths under a block."""
    return _discover_videos(Path(block_path))


def _discover_videos(block_path: Path) -> tuple[list[Path], list[Path], list[Path]]:
    arena_nested = block_path / "arena_videos" / "videos"
    arena_flat = block_path / "arena_videos"
    arena_dir = arena_nested if arena_nested.is_dir() else arena_flat
    arena = sorted(arena_dir.glob("*.mp4")) if arena_dir.is_dir() else []

    le_dir = block_path / "eye_videos" / "LE"
    re_dir = block_path / "eye_videos" / "RE"
    le = sorted(le_dir.rglob("*.mp4")) if le_dir.exists() else []
    re = sorted(re_dir.rglob("*.mp4")) if re_dir.exists() else []
    le = _prefer_eye_mp4([p for p in le if "DLC" not in str(p)], "_LE")
    re = _prefer_eye_mp4([p for p in re if "DLC" not in str(p)], "_RE")
    return arena, le, re


def _prefer_eye_mp4(paths: list[Path], stamp: str) -> list[Path]:
    """Prefer stamped eye videos (e.g. *_LE.mp4) over other MP4s in the folder."""
    if not paths:
        return paths
    stamped = [p for p in paths if stamp.lower() in p.stem.lower()]
    return stamped if stamped else paths


def _load_ellipse_csv(analysis_path: Path, side: str) -> pd.DataFrame | None:
    candidates = [
        analysis_path / f"{side}_eye_data.csv",
        analysis_path / f"{'le' if side == 'left' else 're'}_df.csv",
        analysis_path / f"{'left' if side == 'left' else 'right'}_eye_data.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


def load_block_session(
    block_path: Path,
    output_folder: Path,
    config: AnnotatorConfig,
    *,
    animal_call: str | None = None,
    experiment_date: str | None = None,
    block_num: str | None = None,
) -> BlockSession:
    block_path = Path(block_path).resolve()
    sync_path = block_path / "analysis" / "final_sync_df.csv"
    if not sync_path.exists() and not (block_path / "analysis" / "blocksync_df.csv").exists():
        raise FileNotFoundError(
            f"No final_sync_df.csv in {block_path / 'analysis'}"
        )

    inf_animal, inf_date, inf_block, path_to_animal = infer_metadata(block_path)
    animal_call = animal_call or inf_animal
    experiment_date = experiment_date if experiment_date is not None else inf_date
    block_num = block_num or inf_block

    from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync

    block = BlockSync(
        animal_call,
        experiment_date,
        block_num,
        str(path_to_animal),
    )
    block.block_path = block_path
    block.analysis_path = block_path / "analysis"

    load_final_sync_df(block, verbose=False)
    sample_rate = float(getattr(block, "sample_rate", 30000) or 30000)
    ms_axis = compute_ms_axis(block.final_sync_df, sample_rate)

    arena, le, re = _discover_videos(block_path)
    if not arena:
        try:
            block.handle_arena_files()
            arena = [Path(p) for p in (block.arena_videos or [])]
        except Exception:
            arena = []

    le_df = _load_ellipse_csv(block.analysis_path, "left")
    re_df = _load_ellipse_csv(block.analysis_path, "right")
    if le_df is None and getattr(block, "le_df", None) is not None:
        le_df = block.le_df
    if re_df is None and getattr(block, "re_df", None) is not None:
        re_df = block.re_df

    oe_rec = getattr(block, "oe_rec", None)

    return BlockSession(
        animal_call=animal_call,
        experiment_date=experiment_date,
        block_num=block_num,
        block_path=block_path,
        output_folder=Path(output_folder),
        config=config,
        final_sync_df=block.final_sync_df,
        ms_axis=ms_axis,
        sample_rate_hz=sample_rate,
        arena_videos=arena,
        le_videos=le,
        re_videos=re,
        le_ellipse_df=le_df,
        re_ellipse_df=re_df,
        oe_rec=oe_rec,
    )
