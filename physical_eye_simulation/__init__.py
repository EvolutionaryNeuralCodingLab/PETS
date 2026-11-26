"""Physical-eye Kerr vs Swirski (conicoid) simulation package.

Default path is analytic (exact pinhole pupil ellipses). An optional Blender
mode builds a lightweight sphere+pupil scene or uses annotated MRI landmarks.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
GEOMETRY_DIR = PACKAGE_ROOT / "geometry"
DEFAULT_CONFIG = PACKAGE_ROOT / "config_default.yaml"

__all__ = ["PACKAGE_ROOT", "GEOMETRY_DIR", "DEFAULT_CONFIG"]
