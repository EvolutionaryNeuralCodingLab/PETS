"""Pickle + sidecar metadata helpers."""

from __future__ import annotations

import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def write_pickle_with_meta(
    payload: Any,
    pickle_path: Path,
    *,
    meta: dict[str, Any],
    entrypoint: str,
) -> Path:
    """Write ``payload`` to ``pickle_path`` and a sibling ``*.meta.yaml``."""
    pickle_path = Path(pickle_path)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pickle": str(pickle_path.resolve()),
        "entrypoint": entrypoint,
        "git_hash": git_hash(),
        **meta,
    }
    meta_path = pickle_path.with_suffix(pickle_path.suffix + ".meta.yaml")
    if pickle_path.suffix == ".pickle":
        meta_path = Path(str(pickle_path) + ".meta.yaml")
    elif pickle_path.suffix == ".pkl":
        meta_path = Path(str(pickle_path) + ".meta.yaml")

    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta_out, f, sort_keys=False)
    return meta_path


def load_params_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data
