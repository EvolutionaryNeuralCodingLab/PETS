"""Least-squares ellipse fitting with a canonical geometry contract.

Geometry contract (ellipse_geometry_version = 2)
------------------------------------------------
- ``width`` is the major semiaxis; ``height`` is the minor semiaxis.
- ``width >= height > 0`` when geometry is valid.
- ``phi`` is the major-axis orientation in **radians**, modulo ``π``,
  stored in ``[0, π)``. Image coordinates: x right, y down.
- Near-circular ellipses (``height / width >= NEAR_CIRCLE_AXIS_RATIO``)
  keep valid centers and axes but set ``phi`` to NaN (orientation
  unreliable). Exact circles have no orientation.
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as la

__version__ = "2.1.0"

# Ellipse schema written at storage boundaries after this fix.
ELLIPSE_GEOMETRY_VERSION = 2

# Orientation is poorly constrained when the axis ratio is this close to 1.
NEAR_CIRCLE_AXIS_RATIO = 0.98


def canonicalize_ellipse_axes_phi(width, height, phi):
    """Order semi-axes so ``width >= height`` and wrap ``phi`` to ``[0, π)``.

    When ``height > width``, swap axes and add ``π/2`` to ``phi`` so the
    returned ``phi`` still describes the major axis. This preserves geometry.
    """
    w = np.asarray(width, dtype=float)
    h = np.asarray(height, dtype=float)
    ang = np.asarray(phi, dtype=float)
    scalar = w.ndim == 0 and h.ndim == 0 and ang.ndim == 0
    w = np.atleast_1d(np.array(w, dtype=float, copy=True))
    h = np.atleast_1d(np.array(h, dtype=float, copy=True))
    ang = np.atleast_1d(np.array(ang, dtype=float, copy=True))
    swap = np.isfinite(w) & np.isfinite(h) & (h > w)
    w2 = np.where(swap, h, w)
    h2 = np.where(swap, w, h)
    ang2 = np.where(swap, ang + 0.5 * np.pi, ang)
    ang2 = np.mod(ang2, np.pi)
    bad = ~(np.isfinite(w2) & np.isfinite(h2) & np.isfinite(ang2))
    # Preserve NaN phi (unreliable orientation) when axes are otherwise fine.
    ang_nan = ~np.isfinite(ang)
    ang2 = np.where(ang_nan & np.isfinite(w2) & np.isfinite(h2), np.nan, ang2)
    ang2 = np.where(bad & ~ang_nan, np.nan, ang2)
    if scalar:
        return float(w2[0]), float(h2[0]), float(ang2[0])
    return w2, h2, ang2


def orientation_is_reliable(width, height, *, ratio_threshold: float = NEAR_CIRCLE_AXIS_RATIO) -> bool:
    """Return True when major-axis orientation is considered well constrained."""
    w = float(width)
    h = float(height)
    if not (np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0):
        return False
    if w < h:
        w, h = h, w
    return (h / w) < float(ratio_threshold)


def unwrap_ellipse_phi(phi: np.ndarray) -> np.ndarray:
    """Unwrap π-periodic ellipse orientation within contiguous finite segments.

    Uses ``0.5 * unwrap(2 * phi)``. Does not interpolate across NaN gaps.
    """
    phi = np.asarray(phi, dtype=float)
    out = np.full_like(phi, np.nan, dtype=float)
    finite = np.isfinite(phi)
    if not np.any(finite):
        return out
    # Process contiguous True runs separately so gaps are not bridged.
    padded = np.concatenate([[False], finite, [False]])
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    for s, e in zip(starts, ends):
        seg = phi[s:e]
        out[s:e] = 0.5 * np.unwrap(2.0 * seg)
    return out


def conic_coefficients_to_parameters(
    a: float,
    b: float,
    c: float,
    d: float,
    f: float,
    g: float,
    *,
    near_circle_ratio: float = NEAR_CIRCLE_AXIS_RATIO,
):
    """Convert general conic coefficients to canonical ellipse parameters.

    Coefficients follow ``ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0``.

    Returns
    -------
    center : list[float]
    width : float
        Major semiaxis.
    height : float
        Minor semiaxis.
    phi : float
        Major-axis angle in radians in ``[0, π)``, or NaN if unreliable.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)
    f = float(f)
    g = float(g)

    det = b * b - a * c
    if not np.isfinite(det) or abs(det) < 1e-15:
        raise ValueError("degenerate conic: b^2 - a*c ~ 0")

    x0 = (c * d - b * f) / det
    y0 = (a * f - b * d) / det
    if not (np.isfinite(x0) and np.isfinite(y0)):
        raise ValueError("non-finite ellipse center")

    # Value of the conic polynomial at the center (translation constant).
    mu = a * x0**2 + 2.0 * b * x0 * y0 + c * y0**2 + 2.0 * d * x0 + 2.0 * f * y0 + g
    if not np.isfinite(mu) or abs(mu) < 1e-15:
        raise ValueError("degenerate translated conic")

    # Centered form: X^T A X = -mu  =>  X^T (A / -mu) X = 1
    A = np.array([[a, b], [b, c]], dtype=float)
    M = A / (-mu)

    evals, evecs = np.linalg.eigh(M)
    # Sign-normalize so both eigenvalues are positive for a real ellipse.
    if evals[0] < 0 and evals[1] < 0:
        M = -M
        evals, evecs = np.linalg.eigh(M)

    if evals[0] <= 0 or evals[1] <= 0:
        raise ValueError("quadratic form is not positive-definite (not an ellipse)")

    # eigh returns ascending eigenvalues: smaller λ => larger semiaxis.
    semi_major = 1.0 / np.sqrt(evals[0])
    semi_minor = 1.0 / np.sqrt(evals[1])
    if not (np.isfinite(semi_major) and np.isfinite(semi_minor)):
        raise ValueError("non-finite semiaxes")
    if semi_major <= 0 or semi_minor <= 0:
        raise ValueError("non-positive semiaxes")

    v_major = evecs[:, 0]
    phi = float(np.arctan2(v_major[1], v_major[0]))
    width, height, phi = canonicalize_ellipse_axes_phi(semi_major, semi_minor, phi)

    if not orientation_is_reliable(width, height, ratio_threshold=near_circle_ratio):
        phi = float("nan")

    return [float(x0), float(y0)], float(width), float(height), float(phi)


class LsqEllipse:
    """Least Squares fitting of Elliptical data

    Attributes
    ----------
    coef_ : array
        Estimated coefficients for the Least squares fit to the elliptical data
        containing the values [a,b,c,d,f,g].T corresponding to
        ax**2 + 2bxy + cy**2 + 2dx + 2fy + g

    References
    ----------
    (*) Halir R., Flusser J. 'Numerically Stable Direct Least Squares
    Fitting of Ellipses'
    (**) Weisstein, Eric W. "Ellipse." From MathWorld--A Wolfram Web Resource.
    http://mathworld.wolfram.com/Ellipse.html
    """

    ALLOWED_FEATURES = 2

    def __init__(self):
        self.coef_ = None

    def _check_data(self, X):

        n_samples, n_features = X.shape
        if not n_features == self.ALLOWED_FEATURES:
            raise ValueError(
                "Incorrect number of features. "
                f"Got {n_features} features, expected 2. "
            )

        if n_samples < 5:
            raise ValueError(
                "Received too few samples"
                f"Got {n_samples} features, 5 or more required. "
            )

        return X

    def fit(self, X):
        """Fit the data

        Parameters
        ----------
        X : array, shape (n_points, 2)
            Data values for the x-y data pairs to fit

        Returns
        -------
        self : returns an instance of self.
        """
        X = self._check_data(X)

        # extract x-y pairs
        x, y = X.T

        # Quadratic part of design matrix [eqn. 15] from (*)
        D1 = np.vstack([x**2, x * y, y**2]).T
        # Linear part of design matrix [eqn. 16] from (*)
        D2 = np.vstack([x, y, np.ones_like(x)]).T

        # Forming scatter matrix [eqn. 17] from (*)
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2

        # Constraint matrix [eqn. 18]
        C1 = np.array([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])

        # Reduced scatter matrix [eqn. 29]
        M = la.inv(C1) @ (S1 - S2 @ la.inv(S3) @ S2.T)

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this
        # equation [eqn. 28]
        #
        # M is real but non-symmetric, so np.linalg.eig may return a complex
        # eigendecomposition when the point set does not admit a real ellipse.
        # Those complex eigenvectors are not valid Halir–Flusser solutions and
        # must not propagate into float ellipse parameters / CSV exports.
        _eigval, eigvec = np.linalg.eig(M)
        if np.iscomplexobj(eigvec):
            real_cols = np.all(np.abs(eigvec.imag) <= 1e-12, axis=0)
            eigvec = np.real(eigvec[:, real_cols])
        else:
            eigvec = np.asarray(eigvec, dtype=float)

        if eigvec.ndim != 2 or eigvec.shape[1] == 0:
            raise ValueError("No real eigenvector for ellipse fit")

        # Eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4 * np.multiply(eigvec[0, :], eigvec[2, :]) - np.power(
            eigvec[1, :], 2
        )
        idxs = np.nonzero(cond > 0)[0]
        if idxs.size == 0:
            raise ValueError(
                "No eigenvector satisfies the ellipse constraint 4ac - b^2 > 0"
            )
        a1 = eigvec[:, idxs]

        # |d f g> = -S3^(-1) * S2^(T)*|a b c> [eqn. 24]
        a2 = la.inv(-S3) @ S2.T @ a1

        # Eigenvectors |a b c d f g>
        # list of the coefficients describing an ellipse [a,b,c,d,f,g]
        # corresponding to ax**2 + 2bxy + cy**2 + 2dx + 2fy + g
        self.coef_ = np.vstack([a1, a2])

        return self

    @property
    def coefficients(self):
        """
        List of the coefficients describing the fitted ellipse

        Returns
        -------
        [a,b,c,d,f,g] corresponding to ax**2 + 2bxy + cy**2 + 2dx + 2fy + g
        """
        return np.asarray(self.coef_).ravel()

    def as_parameters(self):
        """Returns the definition of the fitted ellipse as localized parameters

        Returns
        -------
        center : list
            [x0, y0]
        width : float
            Semimajor axis (``width >= height``)
        height : float
            Semiminor axis
        phi : float
            Counterclockwise angle from the x-axis to the major axis, in
            radians in ``[0, π)``. NaN when orientation is unreliable
            (near-circular).
        """
        # Coefficients of ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0
        a = self.coefficients[0]
        b = self.coefficients[1] / 2.0
        c = self.coefficients[2]
        d = self.coefficients[3] / 2.0
        f = self.coefficients[4] / 2.0
        g = self.coefficients[5]
        return conic_coefficients_to_parameters(a, b, c, d, f, g)

    def return_fit(self, n_points=None, t=None):
        """Return the X, Y values of the predicted ellipse

        Points are returned along the parametric curve of the ellipse as evenly
        spaced points starting at t=0 to t=2pi

        Parameters
        ---------
        n_points : int
            Number of points to return
        t : array
            Parametric points used to generate x-y pairs, If provided,
            `n_points` will be ignored

        Returns
        -------
        X : array, shape (n_points, 2)
            data values for the x-y data pairs
        """
        if self.coef_ is None:
            raise ValueError("Must call .fit() before using .return_fit()")

        if n_points is None and t is None:
            raise AttributeError(
                "A value for `n_points` or `t` must be ", "provided"
            )

        if t is None:
            t = np.linspace(0, 2 * np.pi, n_points)

        center, width, height, phi = self.as_parameters()
        if not np.isfinite(phi):
            phi = 0.0  # circle / near-circle: orientation unused for contour

        x = (
            center[0]
            + width * np.cos(t) * np.cos(phi)
            - height * np.sin(t) * np.sin(phi)
        )
        y = (
            center[1]
            + width * np.cos(t) * np.sin(phi)
            + height * np.sin(t) * np.cos(phi)
        )

        return np.c_[x, y]
