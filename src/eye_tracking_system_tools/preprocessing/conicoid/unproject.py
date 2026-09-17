"""Safaee-Rad 1992 circular unprojection of an image ellipse.

Python port of ``singleeyefitter/projection.h`` (Leszek Świrski). The cone
through the camera and the pupil ellipse is reconstructed as a conicoid; its
two circular sections are the candidate 3D pupils.

Camera convention (same as Swirski): origin at the camera centre, looking
along +z, image plane at z = f. Ellipse coordinates on that plane have the
principal point at the origin. Gaze normals are flipped to point **toward**
the camera (``n · centre < 0``).
"""

from __future__ import annotations

import numpy as np

from .geometry import (
    Circle3D,
    Conic,
    Ellipse2D,
    conic_from_ellipse,
    conicoid_from_conic,
    ellipse_from_conic,
)


class UnprojectionError(ValueError):
    """Raised when an ellipse has no valid circular unprojection."""


def project_point(point: np.ndarray, focal_length: float) -> np.ndarray:
    """Pinhole projection of a 3-vector onto the image plane at z = f."""
    p = np.asarray(point, dtype=float).reshape(3)
    if p[2] == 0:
        raise UnprojectionError("Cannot project a point with z = 0")
    return focal_length * p[:2] / p[2]


def project_circle(circle: Circle3D, focal_length: float) -> Ellipse2D:
    """Project a 3D circle to an image ellipse (Swirski ``project(Circle3D)``)."""
    c = np.asarray(circle.centre, dtype=float).reshape(3)
    n = np.asarray(circle.normal, dtype=float).reshape(3)
    r = float(circle.radius)
    f = float(focal_length)

    cn = float(np.dot(c, n))
    c2r2 = float(np.dot(c, c) - r * r)
    abc = cn * cn - 2.0 * cn * c * n + c2r2 * n * n
    F = 2.0 * (c2r2 * n[1] * n[2] - cn * (n[1] * c[2] + n[2] * c[1]))
    G = 2.0 * (c2r2 * n[2] * n[0] - cn * (n[2] * c[0] + n[0] * c[2]))
    H = 2.0 * (c2r2 * n[0] * n[1] - cn * (n[0] * c[1] + n[1] * c[0]))

    conic = Conic(
        A=float(abc[0]),
        B=float(H),
        C=float(abc[1]),
        D=float(G * f),
        E=float(F * f),
        F=float(abc[2] * f * f),
    )
    return ellipse_from_conic(conic)


def _canonical_frame(cone) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of the cone quadratic form, descending eigenvalues.

    Returns ``(lambda, T1)`` where ``T1`` rows are right-handed eigenvectors
    matching Safaee-Rad eq. (8) / Swirski ``T1``.
    """
    q = np.array(
        [
            [cone.A, cone.H, cone.G],
            [cone.H, cone.B, cone.F],
            [cone.G, cone.F, cone.C],
        ],
        dtype=float,
    )
    evals, evecs = np.linalg.eigh(q)
    order = np.argsort(evals)[::-1]
    lam = evals[order]
    # Columns of T1 are eigenvectors (Safaee-Rad eq. 8 / Swirski T1).
    t1 = evecs[:, order].copy()
    # Right-handed: (row0 × row1) · row2 = det(T1) > 0.
    if np.linalg.det(t1) < 0:
        t1 = -t1
    return lam, t1


def unproject(
    ellipse: Ellipse2D,
    circle_radius: float,
    focal_length: float,
) -> tuple[Circle3D, Circle3D]:
    """Unproject an image ellipse to the two 3D circles of the given radius.

    Parameters
    ----------
    ellipse
        Camera-centred image ellipse (principal point at the origin).
    circle_radius
        Assumed 3D pupil radius. Distance–size is otherwise ambiguous; a
        dummy of 1 is fine for gaze-line construction (Swirski).
    focal_length
        Pinhole focal length in the same pixels as ``ellipse``.
    """
    if ellipse.major_radius <= 0 or ellipse.minor_radius <= 0:
        raise UnprojectionError("Ellipse radii must be positive")
    if circle_radius <= 0 or focal_length <= 0:
        raise UnprojectionError("circle_radius and focal_length must be positive")

    conic = conic_from_ellipse(ellipse)
    pupil_cone = conicoid_from_conic(conic, np.array([0.0, 0.0, -float(focal_length)]))
    lam, t1 = _canonical_frame(pupil_cone)

    if not np.all(np.isfinite(lam)):
        raise UnprojectionError("Cone quadratic form is degenerate")
    # Safaee-Rad: λ0 ≥ λ1 > 0 > λ2 for an elliptical cone section.
    if not (lam[0] >= lam[1] > 0.0 and lam[2] < 0.0):
        raise UnprojectionError(
            f"Cone eigenvalues are not an elliptical section: {lam}"
        )

    n = float(np.sqrt((lam[1] - lam[2]) / (lam[0] - lam[2])))
    l_abs = float(np.sqrt((lam[0] - lam[1]) / (lam[0] - lam[2])))

    uvw = np.array([pupil_cone.U, pupil_cone.V, pupil_cone.W], dtype=float)
    t2 = -(t1.T @ uvw) / lam

    solutions: list[Circle3D] = []
    for l_val in (l_abs, -l_abs):
        gaze = t1 @ np.array([l_val, 0.0, n])

        if l_val == 0.0:
            t3 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        else:
            sgn = 1.0 if l_val > 0.0 else -1.0
            t3 = np.array(
                [
                    [0.0, -n * sgn, l_val],
                    [sgn, 0.0, 0.0],
                    [0.0, abs(l_val), n],
                ]
            )

        col0 = t3[:, 0]
        col1 = t3[:, 1]
        col2 = t3[:, 2]
        A = float(np.dot(lam, col0 * col0))
        B = float(np.dot(lam, col0 * col2))
        C = float(np.dot(lam, col1 * col2))
        D = float(np.dot(lam, col2 * col2))
        disc = B * B + C * C - A * D
        if disc <= 0 or abs(A) < 1e-18:
            raise UnprojectionError("Circular section discriminant is non-positive")

        centre_xp = np.zeros(3)
        centre_xp[2] = A * float(circle_radius) / float(np.sqrt(disc))
        centre_xp[0] = -B / A * centre_xp[2]
        centre_xp[1] = -C / A * centre_xp[2]

        def _to_camera(xp: np.ndarray) -> np.ndarray:
            return t1 @ (t3 @ xp + t2) + np.array([0.0, 0.0, float(focal_length)])

        centre = _to_camera(centre_xp)
        if centre[2] < 0:
            centre = _to_camera(-centre_xp)

        if np.dot(gaze, centre) > 0:
            gaze = -gaze
        gn = float(np.linalg.norm(gaze))
        if gn == 0:
            raise UnprojectionError("Zero gaze vector")
        gaze = gaze / gn
        solutions.append(Circle3D(centre=centre, normal=gaze, radius=float(circle_radius)))

    return solutions[0], solutions[1]


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest angle between two 3-vectors, in degrees."""
    ua = np.asarray(a, dtype=float).reshape(3)
    ub = np.asarray(b, dtype=float).reshape(3)
    na = np.linalg.norm(ua)
    nb = np.linalg.norm(ub)
    if na == 0 or nb == 0:
        return float("nan")
    c = float(np.clip(np.dot(ua, ub) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))
