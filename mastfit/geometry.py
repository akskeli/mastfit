"""Geometric primitives: robust circle fitting, PCA axis seeding, IRLS axis refinement.

All estimators here are *deterministic*: no random sampling is used anywhere, so
repeated runs on identical input return bit-identical results. This is a
requirement for a dimensional-control instrument and is the key contrast with
RANSAC-based pipelines.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigh, inv, lstsq, svd

__all__ = [
    "fit_ellipse",
    "orthonormal_basis",
    "taubin_circle",
    "robust_circle",
    "harmonic_ovality",
    "pca_axis",
    "irls_axis",
    "irls_weights",
    "irls_axis_tapered",
    "robust_cone_fit",
]


def orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a 2x3 basis spanning the plane perpendicular to `direction`.

    Parameters
    ----------
    direction : (3,) array
        Axis direction (need not be normalised).

    Returns
    -------
    B : (2, 3) array
        Rows are orthonormal vectors spanning the plane perpendicular to d.
    d : (3,) array
        The normalised direction.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    seed = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - d * (seed @ d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return np.vstack([u, v]), d


def taubin_circle(xy: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Algebraic circle fit (Taubin 1991) with first-order centre covariance.

    Taubin's normalisation minimises the bias of the algebraic distance and is
    the standard non-iterative initialiser for geometric circle fitting.

    Parameters
    ----------
    xy : (n, 2) array
        Points in the slice plane.

    Returns
    -------
    centre : (2,) array
    radius : float
    sigma : float
        RMS of the geometric (radial) residuals.
    cov : (2, 2) array
        First-order covariance of the fitted centre,
        ``cov = sigma^2 (J^T J)^-1`` restricted to the centre block.
    """
    xy = np.asarray(xy, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    z = u * u + v * v
    zm = z.mean()

    A = np.array([[(u * u).mean(), (u * v).mean()],
                  [(u * v).mean(), (v * v).mean()]])
    b = 0.5 * np.array([(u * z).mean(), (v * z).mean()])
    sol, *_ = lstsq(A, b, rcond=None)
    uc, vc = sol
    cx, cy = uc + xm, vc + ym
    radius = float(np.sqrt(uc * uc + vc * vc + zm))

    dx, dy = x - cx, y - cy
    dist = np.hypot(dx, dy)
    resid = dist - radius
    n = len(x)
    # Jacobian of the residual w.r.t. (cx, cy, R)
    J = np.column_stack([-dx / dist, -dy / dist, -np.ones(n)])
    sigma2 = float(resid @ resid) / max(n - 3, 1)
    try:
        cov = sigma2 * inv(J.T @ J)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate slice
        cov = np.full((3, 3), np.inf)
    return np.array([cx, cy]), radius, float(np.sqrt(sigma2)), cov[:2, :2]


def robust_circle(
    xy: np.ndarray, n_iter: int = 4, k_mad: float = 2.5, min_points: int = 20
):
    """Deterministic trimmed circle fit.

    Fit by Taubin, then iteratively discard points whose *radial* residual lies
    more than ``k_mad`` robust scales (MAD) from the median residual, refitting
    each time. This removes scattered off-ring points.

    Note
    ----
    By construction this does *not* remove a dense one-sided cluster (e.g. a
    welded track occupying one arc): such a cluster shifts the whole circle
    rather than producing radial outliers. That case is handled at the slice
    level by :mod:`mastfit.quality`.

    Returns
    -------
    centre, radius, sigma, cov, inliers
    """
    xy = np.asarray(xy, dtype=float)
    centre, radius, sigma, cov = taubin_circle(xy)
    inliers = np.ones(len(xy), dtype=bool)
    for _ in range(n_iter):
        dx, dy = xy[:, 0] - centre[0], xy[:, 1] - centre[1]
        resid = np.hypot(dx, dy) - radius
        med = np.median(resid)
        mad = 1.4826 * np.median(np.abs(resid - med)) + 1e-9
        new = np.abs(resid - med) < k_mad * mad
        if new.sum() < min_points:
            break
        inliers = new
        centre, radius, sigma, cov = taubin_circle(xy[inliers])
    return centre, radius, sigma, cov, inliers


def fit_ellipse(xy: np.ndarray):
    """Direct least-squares ellipse fit (Halir and Flusser's stable form of Fitzgibbon's method).

    Returns ``(centre, semi_axes, angle)`` or ``None`` if the conic is not an ellipse.
    Coordinates are centred and scaled internally for conditioning.
    """
    xy = np.asarray(xy, dtype=float)
    m = xy.mean(axis=0)
    s = np.sqrt(((xy - m) ** 2).sum(axis=1).mean()) + 1e-12
    x, y = ((xy - m) / s).T
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ T
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])
    w, V = np.linalg.eig(M)
    V = np.real(V)
    cond = 4 * V[0] * V[2] - V[1] ** 2
    idx = np.where(cond > 0)[0]
    if len(idx) == 0:
        return None
    a1 = V[:, idx[0]]
    A, B, C = a1
    Dc, E, F = T @ a1
    den = B * B - 4 * A * C
    if den >= 0:
        return None
    x0 = (2 * C * Dc - B * E) / den
    y0 = (2 * A * E - B * Dc) / den
    # semi-axes
    num = 2 * (A * E * E + C * Dc * Dc - B * Dc * E + den * F)
    root = np.sqrt((A - C) ** 2 + B * B)
    with np.errstate(invalid="ignore"):
        ax1 = -np.sqrt(num * (A + C + root)) / den
        ax2 = -np.sqrt(num * (A + C - root)) / den
    if not (np.isfinite(ax1) and np.isfinite(ax2)):
        return None
    angle = 0.5 * np.arctan2(-B, C - A)
    centre = np.array([x0, y0]) * s + m
    return centre, np.array([max(ax1, ax2), min(ax1, ax2)]) * s, float(angle)


def harmonic_ovality(xy: np.ndarray, centre: np.ndarray) -> float:
    """Ovality (%) from the second angular harmonic of the radial profile.

    Fits ``r(theta) ~ R0 + a1 cos + b1 sin + a2 cos2 + b2 sin2`` and returns
    ``e = 2 c2 / (R0 + c2)``, equivalent to ``1 - r_min / r_max`` for a small
    ellipse. This Fourier-roundness measure is markedly more stable on noisy
    partial arcs than a direct algebraic ellipse fit.

    Interpretation warning
    ----------------------
    A nonzero value does not by itself imply structural ovalisation: the same
    estimator applied to a perfectly round ring with the slice's own sampling
    and noise returns a nonzero *null floor* (see
    :func:`mastfit.validate.ovality_null_floor`). Only the excess over that
    floor is meaningful, and even then the cause may be distributed hardware
    rather than tube flattening.
    """
    xy = np.asarray(xy, dtype=float)
    dx, dy = xy[:, 0] - centre[0], xy[:, 1] - centre[1]
    theta = np.arctan2(dy, dx)
    r = np.hypot(dx, dy)
    X = np.column_stack([
        np.ones_like(theta), np.cos(theta), np.sin(theta),
        np.cos(2 * theta), np.sin(2 * theta),
    ])
    coef, *_ = lstsq(X, r, rcond=None)
    r0 = coef[0]
    c2 = np.hypot(coef[3], coef[4])
    if r0 + c2 <= 0:  # pragma: no cover
        return float("nan")
    return float(2 * c2 / (r0 + c2) * 100.0)


def pca_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seed the mast axis with the dominant principal direction.

    For a slender object (aspect ratio ~25:1 here) the axial eigenvalue
    dominates and the top eigenvector recovers the axis to sub-degree accuracy.
    The eigenvalue ratio lambda1/lambda2 is returned as a built-in confidence.

    Returns
    -------
    centroid : (3,) array
    direction : (3,) array
        Unit axis direction, sign-fixed to point upwards (+z).
    eigenvalues : (3,) array, descending.
    """
    P = np.asarray(points, dtype=float)
    c = P.mean(axis=0)
    w, V = eigh((P - c).T @ (P - c))
    order = np.argsort(w)[::-1]
    w = w[order]
    d = V[:, order][:, 0]
    if d[2] < 0:
        d = -d
    return c, d, w


def irls_axis(
    points: np.ndarray,
    centroid: np.ndarray,
    direction: np.ndarray,
    n_iter: int = 5,
    tukey_c: float = 4.685,
):
    """Deterministic robust axis refinement by IRLS with Tukey's biweight.

    Weights are applied to the *radial* residual (distance to the axis minus the
    median radius), so residual rigging and brackets are down-weighted smoothly
    and without any random sampling. Because the input clouds are pre-segmented,
    this converges in two to three reweightings and moves the PCA seed only
    marginally: it is a safety net, not the primary estimator.
    """
    P = np.asarray(points, dtype=float)
    c = np.asarray(centroid, dtype=float).copy()
    d = np.asarray(direction, dtype=float).copy()
    for _ in range(n_iter):
        dp = P - c
        t = dp @ d
        radial = np.linalg.norm(dp - np.outer(t, d), axis=1)
        R = np.median(radial)
        rho = radial - R
        scale = 1.4826 * np.median(np.abs(rho - np.median(rho))) + 1e-9
        cutoff = tukey_c * scale
        w = np.where(np.abs(rho) < cutoff, (1 - (rho / cutoff) ** 2) ** 2, 0.0)
        if w.sum() <= 0:  # pragma: no cover
            break
        c = (w[:, None] * P).sum(axis=0) / w.sum()
        Q = (P - c) * np.sqrt(w)[:, None]
        _, _, Vt = svd(Q, full_matrices=False)
        d = Vt[0]
        if d[2] < 0:
            d = -d
    return c, d


def irls_weights(points, centroid, direction, tukey_c: float = 4.685, radius_model: str = "median"):
    """Biweight weights of the IRLS axis step, for diagnostics.

    ``radius_model="median"`` is the operating rule (one median radius per mast);
    ``"linear"`` measures the radial residual from a robust linear radius trend
    r(t) = a + b t instead, which removes the taper from the residual.
    Returns ``(weights, t)`` with ``t`` the position along the axis.
    """
    P = np.asarray(points, dtype=float)
    c = np.asarray(centroid, dtype=float)
    d = np.asarray(direction, dtype=float)
    dp = P - c
    t = dp @ d
    radial = np.linalg.norm(dp - np.outer(t, d), axis=1)
    if radius_model == "linear":
        R = _robust_radius_trend(t, radial)
    else:
        R = np.median(radial)
    rho = radial - R
    scale = 1.4826 * np.median(np.abs(rho - np.median(rho))) + 1e-9
    cutoff = tukey_c * scale
    w = np.where(np.abs(rho) < cutoff, (1 - (rho / cutoff) ** 2) ** 2, 0.0)
    return w, t


def _robust_radius_trend(t, radial, n_bins: int = 20):
    """Straight line through the per-bin median radii (robust to rigging and flare)."""
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    tc, rc = [], []
    for k in range(n_bins):
        m = (t >= edges[k]) & (t <= edges[k + 1])
        if m.sum() >= 20:
            tc.append(np.median(t[m]))
            rc.append(np.median(radial[m]))
    tc, rc = np.asarray(tc), np.asarray(rc)
    # Theil-Sen slope: median of pairwise slopes, then median intercept
    i, j = np.triu_indices(len(tc), 1)
    slope = np.median((rc[j] - rc[i]) / (tc[j] - tc[i]))
    icpt = np.median(rc - slope * tc)
    return icpt + slope * t


def irls_axis_tapered(points, centroid, direction, n_iter: int = 5, tukey_c: float = 4.685):
    """The IRLS axis step with taper-aware weights (radius_model="linear").

    Identical to :func:`irls_axis` except that the radial residual is measured
    from a robust linear radius trend instead of one median radius. Used for the
    sensitivity study in ``scripts/15_significance_and_ablations.py``.
    """
    P = np.asarray(points, dtype=float)
    c = np.asarray(centroid, dtype=float).copy()
    d = np.asarray(direction, dtype=float).copy()
    for _ in range(n_iter):
        w, _ = irls_weights(P, c, d, tukey_c=tukey_c, radius_model="linear")
        if w.sum() <= 0:  # pragma: no cover
            break
        c = (w[:, None] * P).sum(axis=0) / w.sum()
        Q = (P - c) * np.sqrt(w)[:, None]
        _, _, Vt = svd(Q, full_matrices=False)
        d = Vt[0]
        if d[2] < 0:
            d = -d
    return c, d


def robust_cone_fit(points, centroid, direction, n_iter: int = 5, tukey_c: float = 4.685):
    """Robust fit of a straight truncated cone (tapered cylinder) to the surface points.

    Solves  min over (c, d, a, b)  sum_i rho_Tukey( dist(p_i, line(c, d)) - (a + b t_i) )
    by iteratively reweighted nonlinear least squares (Gauss-Newton steps via
    scipy.optimize.least_squares with the biweight weights fixed per iteration),
    where t_i is the position of p_i along the axis. This is the M-estimator of the
    cylinder-surface objective, used as the reference against which the operating
    weighted-PCA step is compared (``scripts/15_significance_and_ablations.py``).

    Returns ``(centroid, direction, a, b)``.
    """
    from scipy.optimize import least_squares

    P = np.asarray(points, dtype=float)
    c0 = np.asarray(centroid, dtype=float)
    d0 = np.asarray(direction, dtype=float)
    d0 = d0 / np.linalg.norm(d0)
    U, d0 = orthonormal_basis(d0)
    u1, u2 = U[0], U[1]

    def unpack(x):
        c = c0 + x[0] * u1 + x[1] * u2
        d = d0 + x[2] * u1 + x[3] * u2
        return c, d / np.linalg.norm(d)

    def resid(x):
        c, d = unpack(x)
        dp = P - c
        t = dp @ d
        dist = np.linalg.norm(dp - np.outer(t, d), axis=1)
        return dist - (x[4] + x[5] * t)

    t0 = (P - c0) @ d0
    dist0 = np.linalg.norm((P - c0) - np.outer(t0, d0), axis=1)
    b0, a0 = np.polyfit(t0, dist0, 1)
    x = np.array([0.0, 0.0, 0.0, 0.0, a0, b0])
    w = np.ones(len(P))
    for _ in range(n_iter):
        sw = np.sqrt(w)
        x = least_squares(lambda y: sw * resid(y), x, method="lm").x
        e = resid(x)
        scale = 1.4826 * np.median(np.abs(e - np.median(e))) + 1e-9
        cutoff = tukey_c * scale
        w = np.where(np.abs(e) < cutoff, (1 - (e / cutoff) ** 2) ** 2, 0.0)
    c, d = unpack(x)
    if d[2] < 0:
        d = -d
    return c, d, float(x[4]), float(x[5])
