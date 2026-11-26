"""Configurable pinhole camera (defaults: 3.7 mm lens on OV5647 at 640×480)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Paper / PETS fallback (matches conicoid.camera)
LENS_FOCAL_MM = 3.7
OV5647_PIXEL_PITCH_MM = 0.0014
OV5647_NATIVE_WIDTH = 2592
WIDTH = 640
HEIGHT = 480
FOCAL_PX = LENS_FOCAL_MM / OV5647_PIXEL_PITCH_MM * (WIDTH / OV5647_NATIVE_WIDTH)
SENSOR_WIDTH_MM = OV5647_NATIVE_WIDTH * OV5647_PIXEL_PITCH_MM  # 3.6288 mm
CX = WIDTH / 2.0
CY = HEIGHT / 2.0

CAMERA_DISTANCES_MM = (10.0, 13.0, 16.0)
PIVOTS = ("eye_centre", "sclera_sphere_centre", "lens_centre")


@dataclass(frozen=True)
class PaperCamera:
    width: int = WIDTH
    height: int = HEIGHT
    focal_px: float = FOCAL_PX
    cx: float = CX
    cy: float = CY
    lens_mm: float = LENS_FOCAL_MM
    sensor_width_mm: float = SENSOR_WIDTH_MM

    @classmethod
    def from_optics(
        cls,
        *,
        lens_mm: float,
        sensor_width_mm: float,
        width: int,
        height: int,
        cx: float | None = None,
        cy: float | None = None,
    ) -> "PaperCamera":
        """Build a camera from lens and sensor size (horizontal fit)."""
        focal_px = float(lens_mm) / float(sensor_width_mm) * float(width)
        return cls(
            width=int(width),
            height=int(height),
            focal_px=focal_px,
            cx=float(width) / 2.0 if cx is None else float(cx),
            cy=float(height) / 2.0 if cy is None else float(cy),
            lens_mm=float(lens_mm),
            sensor_width_mm=float(sensor_width_mm),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def project_points(xyz_mm, camera: PaperCamera = PaperCamera()):
    """Perspective project (N,3) points in the camera frame to pixel centres.

    Camera frame: origin at the pinhole, +Z toward the scene, +X right, +Y down
    (OpenCV). Points must have Z > 0.
    """
    import numpy as np

    xyz = np.asarray(xyz_mm, dtype=float)
    z = xyz[..., 2]
    u = camera.cx + camera.focal_px * xyz[..., 0] / z
    v = camera.cy + camera.focal_px * xyz[..., 1] / z
    return np.stack([u, v], axis=-1)
