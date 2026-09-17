"""Swirski non-iterative 3D eye-model initialisation from pupil ellipses.

Ports of ``EyeModelFitter::unproject_observations`` and
``EyeModelFitter::initialise_model`` from
https://github.com/LeszekSwirski/singleeyefitter, **without** the final
rescale to a 12 mm human eyeball. Gaze direction is invariant to that
scale; lizard geometry uses an ~8 mm eye diameter at ~13 mm camera
distance (see the synthetic validation table).

Image-based per-frame refine lives in ``refine.py``. Dierkes 2019 3D
line-intersection of eyeball centre is available via
``sphere_method="dierkes_3d"``. Refraction maps are optional (see
``refraction.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Circle3D, Ellipse2D, Sphere
from .unproject import UnprojectionError, project_point, unproject


class EyeModelError(ValueError):
    """Raised when the sphere cannot be estimated from the observations."""


@dataclass
class PupilObservation:
    ellipse: Ellipse2D
    pair: tuple[Circle3D, Circle3D] | None = None
    circle: Circle3D | None = None
    inlier: bool = True


@dataclass
class EyeModelFit:
    sphere: Sphere
    pupils: list[PupilObservation] = field(default_factory=list)
    n_inliers: int = 0
    n_observations: int = 0


def nearest_intersect(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Least-squares intersection of lines ``x = p + s d`` (unit ``d``)."""
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    dim = origins.shape[1]
    a = np.zeros((dim, dim))
    b = np.zeros(dim)
    for p, d in zip(origins, directions):
        nrm = float(np.linalg.norm(d))
        if nrm == 0.0 or not np.isfinite(nrm):
            continue
        v = d / nrm
        iv = np.eye(dim) - np.outer(v, v)
        a += iv
        b += iv @ p
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        raise EyeModelError("Gaze lines are degenerate; cannot intersect") from exc


def _line_sphere_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    sphere: Sphere,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(direction, dtype=float)
    nrm = float(np.linalg.norm(v))
    if nrm == 0:
        raise EyeModelError("Zero direction for line–sphere intersection")
    v = v / nrm
    p = np.asarray(origin, dtype=float)
    c = sphere.centre - p
    r = float(sphere.radius)
    disc = float(np.dot(v, c) ** 2 - np.dot(c, c) + r * r)
    if disc < 0:
        raise EyeModelError("Line and sphere do not intersect")
    root = float(np.sqrt(disc))
    s1 = float(np.dot(v, c) - root)
    s2 = float(np.dot(v, c) + root)
    return p + s1 * v, p + s2 * v


def _projected_gaze_line(
    circle: Circle3D,
    focal_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    c_proj = project_point(circle.centre, focal_length)
    v_proj = project_point(circle.centre + circle.normal, focal_length) - c_proj
    nrm = float(np.linalg.norm(v_proj))
    if nrm == 0:
        raise EyeModelError("Projected gaze has zero length")
    return c_proj, v_proj / nrm


def unproject_observations(
    ellipses: list[Ellipse2D],
    *,
    focal_length: float,
    pupil_radius: float = 1.0,
    eye_z: float = 13.0,
    use_ransac: bool = True,
    ransac_iters: int | None = None,
    ransac_epsilon: float = 10.0,
    rng: np.random.Generator | None = None,
) -> EyeModelFit:
    """Unproject ellipses, intersect projected gaze lines, disambiguate.

    ``eye_z`` places the 3D sphere centre on the ray through the projected
    intersection; default 13 mm matches the lizard synthetic camera distance.
    Dummy ``pupil_radius`` only sets the unprojection scale (Swirski uses 1).
    """
    if len(ellipses) < 2:
        raise EyeModelError("Need at least two pupil ellipses")

    pupils: list[PupilObservation] = []
    origins = []
    directions = []
    for ell in ellipses:
        obs = PupilObservation(ellipse=ell)
        try:
            pair = unproject(ell, pupil_radius, focal_length)
            c_proj, v_proj = _projected_gaze_line(pair[0], focal_length)
        except (UnprojectionError, EyeModelError):
            obs.pair = None
            obs.inlier = False
            pupils.append(obs)
            continue
        obs.pair = pair
        pupils.append(obs)
        origins.append(c_proj)
        directions.append(v_proj)

    if len(origins) < 2:
        raise EyeModelError("Fewer than two ellipses unprojected successfully")

    origins_a = np.asarray(origins)
    dirs_a = np.asarray(directions)
    valid_idx = [i for i, p in enumerate(pupils) if p.pair is not None]

    if use_ransac and len(valid_idx) >= 2:
        eye_centre_proj, inlier_local = _ransac_intersection(
            origins_a,
            dirs_a,
            epsilon=ransac_epsilon,
            n_iters=ransac_iters,
            rng=rng,
        )
        inlier_set = {valid_idx[j] for j in inlier_local}
        for i, obs in enumerate(pupils):
            obs.inlier = i in inlier_set
    else:
        eye_centre_proj = nearest_intersect(origins_a, dirs_a)
        for obs in pupils:
            if obs.pair is not None:
                obs.inlier = True

    n_inliers = sum(1 for p in pupils if p.inlier and p.pair is not None)
    if n_inliers < 2:
        raise EyeModelError("RANSAC found fewer than two inlier gaze lines")

    sphere = Sphere(
        centre=np.array(
            [
                eye_centre_proj[0] * eye_z / focal_length,
                eye_centre_proj[1] * eye_z / focal_length,
                float(eye_z),
            ]
        ),
        radius=1.0,
    )

    for obs in pupils:
        if obs.pair is None:
            continue
        first, second = obs.pair
        c_proj, v_proj = _projected_gaze_line(first, focal_length)
        # Choose the circle whose projected normal points away from the
        # projected sphere centre (Swirski / Dierkes eq. 1).
        if np.dot(c_proj - eye_centre_proj, v_proj) >= 0:
            obs.circle = first
        else:
            obs.circle = second

    return EyeModelFit(
        sphere=sphere,
        pupils=pupils,
        n_inliers=n_inliers,
        n_observations=len(ellipses),
    )


def _ransac_intersection(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    epsilon: float,
    n_iters: int | None,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, list[int]]:
    rng = rng or np.random.default_rng()
    n = origins.shape[0]
    w = 0.3
    p = 0.9999
    k_default = int(np.ceil(np.log(1.0 - p) / np.log(1.0 - w ** 2)))
    k = int(n_iters) if n_iters is not None else max(k_default, 32)
    eps2 = float(epsilon) ** 2

    best_err = np.inf
    best_centre = None
    best_inliers: list[int] = []
    indices = np.arange(n)

    for _ in range(k):
        sample = rng.choice(indices, size=2, replace=False)
        try:
            sample_c = nearest_intersect(origins[sample], directions[sample])
        except EyeModelError:
            continue
        inliers = [
            i
            for i in indices
            if _point_line_dist2(sample_c, origins[i], directions[i]) < eps2
        ]
        if len(inliers) <= w * n:
            continue
        try:
            centre = nearest_intersect(origins[inliers], directions[inliers])
        except EyeModelError:
            continue
        err = sum(
            min(_point_line_dist2(centre, origins[i], directions[i]), eps2)
            for i in indices
        )
        if err < best_err:
            best_err = err
            best_centre = centre
            best_inliers = inliers

    if best_centre is None:
        best_centre = nearest_intersect(origins, directions)
        best_inliers = list(indices)
    return best_centre, list(best_inliers)


def _point_line_dist2(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    v = direction / np.linalg.norm(direction)
    d = point - origin
    return float(np.dot(d, d) - np.dot(d, v) ** 2)


def _estimate_sphere_radius(fit: EyeModelFit) -> float:
    camera = np.zeros(3)
    radii: list[float] = []
    for obs in fit.pupils:
        if obs.circle is None or not obs.inlier:
            continue
        try:
            p_hat = nearest_intersect(
                np.vstack([fit.sphere.centre, camera]),
                np.vstack([obs.circle.normal, obs.circle.centre]),
            )
        except EyeModelError:
            continue
        radii.append(float(np.linalg.norm(p_hat - fit.sphere.centre)))
    if not radii:
        raise EyeModelError("Could not estimate sphere radius from inliers")
    return float(np.mean(radii))


def snap_pupils_to_sphere(fit: EyeModelFit) -> EyeModelFit:
    """Intersect each camera ray with the sphere; gaze is the radial."""
    camera = np.zeros(3)
    for obs in fit.pupils:
        if obs.circle is None:
            continue
        try:
            near, _far = _line_sphere_intersect(
                camera, obs.circle.centre, fit.sphere
            )
        except EyeModelError:
            obs.circle = None
            continue
        radial = near - fit.sphere.centre
        nrm = float(np.linalg.norm(radial))
        if nrm == 0:
            obs.circle = None
            continue
        normal = radial / nrm
        if np.dot(normal, near) > 0:
            normal = -normal
        z = float(obs.circle.centre[2])
        r_at_1 = obs.circle.radius / z if z else obs.circle.radius
        obs.circle = Circle3D(
            centre=near, normal=normal, radius=float(r_at_1 * near[2])
        )
    return fit


def dierkes_eyeball_centre(circles: list[Circle3D], sphere_radius: float) -> np.ndarray:
    """Dierkes 2019 §3.3: LS intersection of lines ``p_i - R n_i`` (vary pupil r).

    Each unprojected circle centre lies on a camera ray ``d_i = p_i/||p_i||``.
    With fixed sphere radius ``R``, the eyeball centre is constrained to the
    line through ``-R n_i`` in direction ``d_i``.
    """
    if len(circles) < 2:
        raise EyeModelError("Need at least two circles for Dierkes intersection")
    origins = []
    directions = []
    r = float(sphere_radius)
    for circ in circles:
        p = np.asarray(circ.centre, dtype=float).reshape(3)
        n = np.asarray(circ.normal, dtype=float).reshape(3)
        pn = float(np.linalg.norm(p))
        nn = float(np.linalg.norm(n))
        if pn == 0 or nn == 0:
            continue
        origins.append(-r * (n / nn))
        directions.append(p / pn)
    if len(origins) < 2:
        raise EyeModelError("Dierkes lines are degenerate")
    return nearest_intersect(np.asarray(origins), np.asarray(directions))


def sph_to_cart(theta: float, psi: float) -> np.ndarray:
    """Swirski spherical parametrisation: y-up, ``psi`` in the x–z plane."""
    st = float(np.sin(theta))
    return np.array(
        [st * np.cos(psi), np.cos(theta), st * np.sin(psi)], dtype=float
    )


def cart_to_sph(vector: np.ndarray) -> tuple[float, float]:
    v = np.asarray(vector, dtype=float).reshape(3)
    r = float(np.linalg.norm(v))
    if r == 0:
        return 0.0, 0.0
    theta = float(np.arccos(np.clip(v[1] / r, -1.0, 1.0)))
    psi = float(np.arctan2(v[2], v[0]))
    return theta, psi


def circle_from_params(
    sphere: Sphere, theta: float, psi: float, radius: float
) -> Circle3D:
    radial = sph_to_cart(theta, psi)
    nrm = float(np.linalg.norm(radial))
    if nrm == 0:
        radial = np.array([0.0, 0.0, -1.0])
    else:
        radial = radial / nrm
    centre = sphere.centre + float(sphere.radius) * radial
    if np.dot(radial, centre) > 0:
        radial = -radial
        centre = sphere.centre + float(sphere.radius) * radial
    return Circle3D(centre=centre, normal=radial, radius=float(radius))


def params_from_circle(sphere: Sphere, circle: Circle3D) -> tuple[float, float, float]:
    theta, psi = cart_to_sph(circle.centre - sphere.centre)
    return theta, psi, float(circle.radius)


def initialise_model(fit: EyeModelFit) -> EyeModelFit:
    """Estimate sphere radius and snap each pupil to the sphere surface.

    Does **not** rescale to a 12 mm human radius. Gaze after this step is
    the radial from the sphere centre through the ray–sphere intersection
    (original unprojected normals are discarded, as in Swirski).
    """
    radius = _estimate_sphere_radius(fit)
    fit.sphere = Sphere(centre=fit.sphere.centre, radius=radius)
    return snap_pupils_to_sphere(fit)


def fit_sphere_from_ellipses(
    ellipses: list[Ellipse2D],
    *,
    focal_length: float,
    pupil_radius: float = 1.0,
    eye_z: float = 13.0,
    use_ransac: bool = True,
    rng: np.random.Generator | None = None,
    sphere_method: str = "dierkes_3d",
) -> EyeModelFit:
    """Unproject + initialise. ``sphere_method`` is ``swirski_2d`` or ``dierkes_3d``."""
    method = str(sphere_method).strip().lower()
    if method not in ("swirski_2d", "dierkes_3d"):
        raise ValueError(
            f"sphere_method must be 'swirski_2d' or 'dierkes_3d', got {sphere_method!r}"
        )
    fit = unproject_observations(
        ellipses,
        focal_length=focal_length,
        pupil_radius=pupil_radius,
        eye_z=eye_z,
        use_ransac=use_ransac,
        rng=rng,
    )
    fit = initialise_model(fit)
    if method == "dierkes_3d":
        circles = [
            obs.circle
            for obs in fit.pupils
            if obs.circle is not None and obs.inlier
        ]
        centre = dierkes_eyeball_centre(circles, fit.sphere.radius)
        fit.sphere = Sphere(centre=centre, radius=fit.sphere.radius)
        fit = snap_pupils_to_sphere(fit)
    return fit
