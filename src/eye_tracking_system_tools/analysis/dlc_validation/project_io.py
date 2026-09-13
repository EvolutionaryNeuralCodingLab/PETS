"""DeepLabCut project discovery, config parsing, and path resolution."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_PUPIL_RE = re.compile(r"pupil", re.IGNORECASE)
_EDGE_RE = re.compile(r"edge", re.IGNORECASE)
_EYE_SUFFIX_RE = re.compile(r"_(LE|RE)(?:_|\.|$)", re.IGNORECASE)


def _normalize_frame_key(key: str | int) -> str:
    """Normalize DLC frame index strings for cross-platform joins."""
    return str(key).replace("\\", "/")


def is_pupil_bodypart(name: str) -> bool:
    """Return True if ``name`` is a pupil perimeter landmark (not an eye edge)."""
    return bool(_PUPIL_RE.search(name)) and not bool(_EDGE_RE.search(name))


def sort_pupil_bodyparts(names: list[str]) -> list[str]:
    """Sort pupil bodyparts by numeric suffix when present (Pupil_1 … Pupil_12)."""

    def _key(n: str) -> tuple[int, str]:
        m = re.search(r"(\d+)\s*$", n)
        return (int(m.group(1)), n) if m else (999, n)

    return sorted(names, key=_key)


@dataclass
class DlcProject:
    """Resolved DeepLabCut project paths and metadata."""

    root: Path
    config: dict[str, Any]
    scorer: str
    bodyparts: list[str]
    pupil_bodyparts: list[str]
    pcutoff: float
    iteration: int
    shuffle: int
    training_fraction: float
    is_pytorch: bool = False
    pose_cfg: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def species_note(self) -> str:
        return str(self.config.get("_qc_species", ""))

    def training_dataset_dir(self, iteration: int | None = None) -> Path:
        it = self.iteration if iteration is None else iteration
        return self.root / "training-datasets" / f"iteration-{it}"

    def evaluation_dir(self, iteration: int | None = None) -> Path:
        it = self.iteration if iteration is None else iteration
        return self.root / "evaluation-results" / f"iteration-{it}"

    def model_dir(self, iteration: int | None = None, shuffle: int | None = None) -> Path:
        it = self.iteration if iteration is None else iteration
        sh = self.shuffle if shuffle is None else shuffle
        task = self.config.get("Task", "task")
        date = self.config.get("date", "date")
        return (
            self.root
            / "dlc-models"
            / f"iteration-{it}"
            / f"{task}{date}-trainset{int(self.training_fraction * 100)}shuffle{sh}"
        )


def load_dlc_project(
    root: Path | str,
    *,
    iteration: int | None = None,
    shuffle: int = 1,
    species: str = "",
) -> DlcProject:
    """Load ``config.yaml`` and resolve bodyparts from pose_cfg when available."""
    root = Path(root).resolve()
    cfg_path = root / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"DLC config not found: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    it = int(iteration if iteration is not None else config.get("iteration", 0))
    train_fracs = config.get("TrainingFraction", [0.95])
    training_fraction = float(train_fracs[0]) if train_fracs else 0.95

    proj = DlcProject(
        root=root,
        config=config,
        scorer=str(config.get("scorer", "scorer")),
        bodyparts=[],
        pupil_bodyparts=[],
        pcutoff=float(config.get("pcutoff", 0.6)),
        iteration=it,
        shuffle=int(shuffle),
        training_fraction=training_fraction,
        is_pytorch=_detect_pytorch_backend(config, root, it, shuffle, training_fraction),
    )
    if species:
        proj.config["_qc_species"] = species

    pose_cfg = _load_pose_cfg(proj)
    proj.pose_cfg = pose_cfg
    bodyparts = _resolve_bodyparts(config, pose_cfg)
    proj.bodyparts = bodyparts
    proj.pupil_bodyparts = sort_pupil_bodyparts([b for b in bodyparts if is_pupil_bodypart(b)])

    if str(config.get("project_path", "")).replace("\\", "/") != str(root).replace("\\", "/"):
        proj.notes.append(
            f"config project_path ({config.get('project_path')}) differs from QC root ({root})"
        )
    return proj


def _detect_pytorch_backend(
    config: dict[str, Any],
    root: Path,
    iteration: int,
    shuffle: int,
    training_fraction: float,
) -> bool:
    engine = str(config.get("engine", config.get("model", ""))).lower()
    if "pytorch" in engine or "dlc3" in engine:
        return True
    task = config.get("Task", "task")
    date = config.get("date", "date")
    trainset = int(training_fraction * 100)
    pytorch_cfg = (
        root
        / "dlc-models"
        / f"iteration-{iteration}"
        / f"{task}{date}-trainset{trainset}shuffle{shuffle}"
        / "train"
        / "pytorch_config.yaml"
    )
    return pytorch_cfg.is_file()


def _load_pose_cfg(project: DlcProject) -> dict[str, Any]:
    pose_path = project.model_dir() / "train" / "pose_cfg.yaml"
    if not pose_path.is_file():
        return {}
    with open(pose_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_bodyparts(config: dict[str, Any], pose_cfg: dict[str, Any]) -> list[str]:
    if pose_cfg.get("all_joints_names"):
        return list(pose_cfg["all_joints_names"])
    if pose_cfg.get("bodyparts"):
        return list(pose_cfg["bodyparts"])
    raw = config.get("bodyparts", [])
    # config may contain duplicates; preserve order, drop dupes
    seen: set[str] = set()
    out: list[str] = []
    for b in raw:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def find_collected_data_h5(project: DlcProject, iteration: int | None = None) -> Path:
    """Return merged training-set CollectedData H5 for the given iteration."""
    it = project.iteration if iteration is None else iteration
    base = project.training_dataset_dir(it)
    matches = sorted(base.rglob(f"CollectedData_{project.scorer}.h5"))
    if not matches:
        raise FileNotFoundError(f"No CollectedData_{project.scorer}.h5 under {base}")
    # Prefer UnaugmentedDataSet merged file
    unaug = [p for p in matches if "UnaugmentedDataSet" in str(p)]
    return unaug[0] if unaug else matches[-1]


def find_evaluation_results_csv(project: DlcProject, iteration: int | None = None) -> Path | None:
    it = project.iteration if iteration is None else iteration
    ev_dir = project.evaluation_dir(it)
    if not ev_dir.is_dir():
        return None
    matches = sorted(ev_dir.rglob("*-results.csv"))
    return matches[-1] if matches else None


def find_evaluation_h5(project: DlcProject, iteration: int | None = None) -> Path | None:
    it = project.iteration if iteration is None else iteration
    ev_dir = project.evaluation_dir(it)
    if not ev_dir.is_dir():
        return None
    matches = sorted(ev_dir.rglob("*-snapshot-*.h5"))
    return matches[-1] if matches else None


def find_analyzed_h5_files(project: DlcProject, videos_dir: Path | None = None) -> list[Path]:
    """Discover analyzed-video H5 outputs (typically under ``videos/``)."""
    root = videos_dir or (project.root / "videos")
    if not root.is_dir():
        return []
    return sorted(root.glob("*DLC*.h5"))


def find_source_video_for_h5(h5_path: Path, videos_dir: Path | None = None) -> Path | None:
    """Match an analyzed H5 to its source ``.mp4`` when co-located."""
    parent = videos_dir or h5_path.parent
    stem = h5_path.name.split("DLC")[0]
    for ext in (".mp4", ".avi", ".mov"):
        candidate = parent / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_train_test_split(project: DlcProject, iteration: int | None = None) -> dict[str, Any]:
    """
    Load train/test frame indices from DLC ``Documentation_data`` pickle.

    Returns dict with keys ``train_indices``, ``test_indices``, ``training_fraction``,
    and ``frame_keys`` (normalized CollectedData index order).
    """
    it = project.iteration if iteration is None else iteration
    doc_dir = project.training_dataset_dir(it)
    matches = sorted(doc_dir.rglob("Documentation_data*.pickle"))
    if not matches:
        raise FileNotFoundError(f"No Documentation_data pickle under {doc_dir}")

    with open(matches[0], "rb") as f:
        doc = pickle.load(f, encoding="latin1")

    train_idx = test_idx = None
    training_fraction = project.training_fraction
    if isinstance(doc, (list, tuple)) and len(doc) >= 3:
        train_idx = list(doc[1]) if doc[1] is not None else []
        test_idx = list(doc[2]) if doc[2] is not None else []
        if len(doc) >= 4:
            try:
                training_fraction = float(doc[3])
            except (TypeError, ValueError):
                pass

    collected = find_collected_data_h5(project, it)
    gt = pd.read_hdf(collected)
    frame_keys = [_normalize_frame_key(k) for k in gt.index]

    split_labels = ["unknown"] * len(frame_keys)
    for i in train_idx or []:
        if 0 <= int(i) < len(split_labels):
            split_labels[int(i)] = "train"
    for i in test_idx or []:
        if 0 <= int(i) < len(split_labels):
            split_labels[int(i)] = "test"

    return {
        "train_indices": [int(i) for i in (train_idx or [])],
        "test_indices": [int(i) for i in (test_idx or [])],
        "training_fraction": training_fraction,
        "frame_keys": frame_keys,
        "split_labels": split_labels,
        "documentation_path": matches[0],
        "collected_data_path": collected,
    }


def parse_video_metadata(
    h5_path: Path,
    *,
    species: str = "",
    animal: str = "",
) -> dict[str, str]:
    """Infer video / eye metadata from an analyzed H5 filename."""
    stem = h5_path.name.split("DLC")[0].rstrip("_")
    eye = "unknown"
    m = _EYE_SUFFIX_RE.search(stem)
    if m:
        eye = m.group(1).upper()
    return {
        "video": stem,
        "eye": eye,
        "animal": animal or "unknown",
        "species": species or "unknown",
        "h5_path": str(h5_path),
    }


def load_native_eval_summary(csv_path: Path) -> pd.DataFrame:
    """Load DLC ``evaluate_network`` summary CSV without recomputing aggregates."""
    return pd.read_csv(csv_path)
