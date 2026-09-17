"""Geometric primitives for Safaee-Rad / Swirski pupil unprojection.

Ports of ``Ellipse.h``, ``Circle.h``, ``Sphere.h``, ``Conic.h``, and
``Conicoid.h`` from https://github.com/LeszekSwirski/singleeyefitter.

A 2D conic is ``A x^2 + B xy + C y^2 + D x + E y + F = 0``.
A 3D conicoid is
``A x^2 + B y^2 + C z^2 + 2 F yz + 2 G zx + 2 H xy + 2 U x + 2 V y + 2 W z + D = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ellipse2D:
    """Axis-aligned-in-its-own-frame ellipse on the image plane.

    ``angle`` is the counterclockwise rotation (radians) from the x-axis to
    the **major** axis. Radii are semi-axes. Centre is in the same 2D units
    as the camera-centred image plane (principal point at the origin).
    """

    centre: tuple[float, float]
    major_radius: float
    minor_radius: float
    angle: float

    @property
    def cx(self) -> float:
        return float(self.centre[0])

    @property
    def cy(self) -> float:
        return float(self.centre[1])


@dataclass(frozen=True)
class Circle3D:
    """Circle in camera space: centre, unit normal, radius."""

    centre: np.ndarray
    normal: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", np.asarray(self.centre, dtype=float).reshape(3))
        object.__setattr__(self, "normal", np.asarray(self.normal, dtype=float).reshape(3))


@dataclass(frozen=True)
class Sphere:
    centre: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", np.asarray(self.centre, dtype=float).reshape(3))


@dataclass(frozen=True)
class Conic:
    """Implicit 2D conic ``A x^2 + B xy + C y^2 + D x + E y + F = 0``."""

    A: float
    B: float
    C: float
    D: float
    E: float
    F: float

    def evaluate(self, x: float, y: float) -> float:
        return (
            self.A * x * x
            + self.B * x * y
            + self.C * y * y
            + self.D * x
            + self.E * y
            + self.F
        )


@dataclass(frozen=True)
class Conicoid:
    """Quadric ``A x^2 + B y^2 + C z^2 + 2F yz + 2G zx + 2H xy + 2U x + 2V y + 2W z + D = 0``."""

    A: float
    B: float
    C: float
    F: float
    G: float
    H: float
    U: float
    V: float
    W: float
    D: float


def conic_from_ellipse(ellipse: Ellipse2D) -> Conic:
    """Swirski ``Conic(const Ellipse2D &)``."""
    ax = float(np.cos(ellipse.angle))
    ay = float(np.sin(ellipse.angle))
    a2 = ellipse.major_radius ** 2
    b2 = ellipse.minor_radius ** 2
    x0, y0 = ellipse.cx, ellipse.cy

    A = ax * ax / a2 + ay * ay / b2
    B = 2.0 * ax * ay / a2 - 2.0 * ax * ay / b2
    C = ay * ay / a2 + ax * ax / b2
    D = (
        (-2.0 * ax * ay * y0 - 2.0 * ax * ax * x0) / a2
        + (2.0 * ax * ay * y0 - 2.0 * ay * ay * x0) / b2
    )
    E = (
        (-2.0 * ax * ay * x0 - 2.0 * ay * ay * y0) / a2
        + (2.0 * ax * ay * x0 - 2.0 * ax * ax * y0) / b2
    )
    F = (
        (2.0 * ax * ay * x0 * y0 + ax * ax * x0 * x0 + ay * ay * y0 * y0) / a2
        + (-2.0 * ax * ay * x0 * y0 + ay * ay * x0 * x0 + ax * ax * y0 * y0) / b2
        - 1.0
    )
    return Conic(A=A, B=B, C=C, D=D, E=E, F=F)


def ellipse_from_conic(conic: Conic) -> Ellipse2D:
    """Swirski ``Ellipse2D(const Conic &)``."""
    angle = 0.5 * float(np.arctan2(conic.B, conic.A - conic.C))
    cost = float(np.cos(angle))
    sint = float(np.sin(angle))
    sin_sq = sint * sint
    cos_sq = cost * cost

    ao = conic.F
    au = conic.D * cost + conic.E * sint
    av = -conic.D * sint + conic.E * cost
    auu = conic.A * cos_sq + conic.C * sin_sq + conic.B * sint * cost
    avv = conic.A * sin_sq + conic.C * cos_sq - conic.B * sint * cost

    tu = -au / (2.0 * auu)
    tv = -av / (2.0 * avv)
    w_centre = ao - auu * tu * tu - avv * tv * tv

    cx = tu * cost - tv * sint
    cy = tu * sint + tv * cost
    major = float(np.sqrt(abs(-w_centre / auu)))
    minor = float(np.sqrt(abs(-w_centre / avv)))
    if major < minor:
        major, minor = minor, major
        angle = angle + 0.5 * np.pi
    if angle > np.pi:
        angle = angle - np.pi
    return Ellipse2D(centre=(cx, cy), major_radius=major, minor_radius=minor, angle=float(angle))


def conicoid_from_conic(conic: Conic, vertex: np.ndarray) -> Conicoid:
    """Cone with the given conic as base on ``z = 0`` and vertex ``vertex``.

    Swirski ``Conicoid(const Conic &, vertex)``.
    """
    alpha, beta, gamma = (float(v) for v in np.asarray(vertex, dtype=float).reshape(3))
    g2 = gamma * gamma
    return Conicoid(
        A=g2 * conic.A,
        B=g2 * conic.C,
        C=(
            conic.A * alpha * alpha
            + conic.B * alpha * beta
            + conic.C * beta * beta
            + conic.D * alpha
            + conic.E * beta
            + conic.F
        ),
        F=-gamma * (conic.C * beta + conic.B / 2.0 * alpha + conic.E / 2.0),
        G=-gamma * (conic.B / 2.0 * beta + conic.A * alpha + conic.D / 2.0),
        H=g2 * conic.B / 2.0,
        U=g2 * conic.D / 2.0,
        V=g2 * conic.E / 2.0,
        W=-gamma * (conic.E / 2.0 * beta + conic.D / 2.0 * alpha + conic.F),
        D=g2 * conic.F,
    )


def ellipse_from_pets(
    center_x: float,
    center_y: float,
    major_ax: float,
    minor_ax: float,
    phi: float,
    *,
    cx: float,
    cy: float,
    width: float | None = None,
    height: float | None = None,
) -> Ellipse2D:
    """Convert a PETS pixel-space ellipse to a camera-centred ``Ellipse2D``.

    ``LsqEllipse.as_parameters`` stores semi-axis ``width`` along ``phi`` and
    ``height`` along ``phi + π/2``; those may be swapped relative to major/
    minor. When ``width``/``height`` are provided they take precedence so the
    major-axis angle is recovered correctly. Otherwise ``phi`` is treated as
    the angle of ``major_ax``.

    Image origin is top-left; the principal point ``(cx, cy)`` is subtracted
    so Swirski's cone vertex at ``(0, 0, -f)`` is valid.
    """
    if (
        width is not None
        and height is not None
        and np.isfinite(width)
        and np.isfinite(height)
        and width > 0
        and height > 0
    ):
        major = float(width)
        minor = float(height)
        angle = float(phi)
    else:
        major = float(major_ax)
        minor = float(minor_ax)
        angle = float(phi)
    if major < minor:
        major, minor = minor, major
        angle = angle + 0.5 * np.pi
    return Ellipse2D(
        centre=(float(center_x) - float(cx), float(center_y) - float(cy)),
        major_radius=major,
        minor_radius=minor,
        angle=angle,
    )
