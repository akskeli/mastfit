"""Validation without ground truth: synthetic phantoms and error models.

The central result in this module is that the *error model matters more than the
error magnitude* (``scripts/05_synthetic_validation.py``, 8 seeds):

    white (independent) surface noise    -> bending-profile RMSE ~= 6 mm
    spatially-correlated error, same RMS -> bending-profile RMSE ~= 8.5 mm (5-13 mm)

Structure-from-motion error is spatially correlated (it comes from camera
geometry and lens-model mis-calibration, not from per-point independent jitter),
so the *correlated* figure is the operative accuracy of the method; with a
phantom matched to the measured cloud statistics (correlated field of 45 mm RMS) it is
~21 mm, and per mast 20-27 mm (mean) and 29-40 mm (90th percentile), against ~7 mm for
the same phantoms with independent noise. White noise averages out along a slice;
correlated error masquerades as shape and does not.

The terrestrial-laser cross-validation corroborates this: the RMS difference
between the two sensors is 14 mm (transverse) and 33 mm (longitudinal).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq, svd

__all__ = [
    "make_phantom",
    "gp_correlated_field",
    "matched_phantom",
    "phantom_geometry",
    "slice_statistics",
    "detrend",
    "recovered_profile",
    "compare_to_truth",
    "ovality_null_floor",
    "ovality_study",
]

#: Per-slice statistics of the real UAV clouds (pooled over the five masts).
#: A phantom used to transfer an accuracy figure to the real data must be shown
#: to resemble these -- see ``scripts/08_phantom_match.py``.
REAL_CLOUD_STATS = {
    "radial_residual_sigma_median_mm": 45.6,
    "radial_residual_sigma_p90_mm": 112.2,
    "trimmed_fraction_median_pct": 5.6,
    "angular_coverage_p10_pct": 91.7,
    "point_density_median_per_m": 386.6,
}

#: Measured geometry of the five real masts (bow -> stern), used so that a phantom
#: can reproduce a *specific* mast rather than a generic 50 m tube. Matching the noise
#: statistics but not the geometry would still leave the phantom unrepresentative:
#: the masts differ by 12 m in height and by a factor of six in taper.
REAL_MAST_GEOMETRY = [
    # (name,          height_m, radius_base_m, taper, points_per_m)
    ("Foremast",         51.3,  0.426, 0.0035, 287),
    ("Mainmast",         52.7,  0.478, 0.0049, 343),
    ("Middle Mast",      53.7,  0.489, 0.0049, 319),
    ("Mizzenmast",       53.2,  0.538, 0.0008, 486),
    ("Jiggermast",       42.1,  0.541, 0.0030, 417),
]

#: Incoherent (within-slice) surface noise needed to reproduce the measured
#: per-slice radial residual of the real clouds. Correlated error alone does not
#: produce it, because a smooth field displaces a slice rather than scattering it.
INCOHERENT_NOISE_M = 0.040


def make_phantom(
    height_m: float = 50.0,
    radius_base_m: float = 0.50,
    taper: float = 0.006,
    rake_deg: float = 2.0,
    mode: str = "C",
    amplitude_m: float = 0.15,
    points_per_m: int = 380,
    noise_m: float = 0.02,
    correlated: bool = False,
    clutter_frac: float = 0.0,
    base_flare: bool = False,
    common_mode=None,
    seed: int = 0,
    corr_len_m: float | None = None,
    corr_anisotropy: float = 1.0,
    corr_angular_frac: float = 0.4,
    wavelength_m: float | None = None,
):
    """Generate a tapered mast with a *known* curved axis.

    Parameters
    ----------
    mode : {"C", "lean", "S", "wave"}
        Shape of the injected bend; ``"wave"`` is a sinusoid of wavelength
        ``wavelength_m`` (used to measure the response of the smoother).
    correlated : bool
        If True, replace independent per-point noise with a smooth,
        spatially-correlated field of the same RMS (the SfM-realistic model).
    clutter_frac : float
        Fraction of added off-surface points, clustered on one side at six
        bands (the rigging-attachment analogue).
    base_flare : bool
        Add a deck-collar flare at the base.
    common_mode : callable or None
        ``f(z_norm) -> transverse offset (m)`` added to *every* mast; used to
        inject a known doming-like common curvature.
    corr_len_m : float or None
        Correlated-error model. ``None`` (default, used for every number in the
        paper) is the low-order Fourier field: horizontal displacements built from
        five modes ``sin(kz*pi*z/H + kt*theta + phase)`` with
        ``(kz, kt) in {(1,0), (2,0), (3,0), (1,1), (2,1)}``, i.e. vertical
        wavelengths 2H, H and 2H/3, angular orders 0 (slice translation) and 1,
        Gaussian random amplitudes and uniform random phases, rescaled so that the
        RMS of each horizontal component equals ``noise_m``. A number instead uses
        a stationary Gaussian-process field along the height with squared-
        exponential correlation of length ``corr_len_m`` (metres); see
        :func:`gp_correlated_field`.
    corr_anisotropy : float
        GP field only: RMS ratio of the displacement in the plane of the injected
        bend (x) to the perpendicular one (y), at fixed mean-square amplitude.
    corr_angular_frac : float
        GP field only: fraction of the variance in the angle-dependent (order-1)
        part; the rest translates whole slices.

    Returns
    -------
    points : (n, 3) array
    z_truth : (m,) array
    bend_truth_mm : (m,) array
        The injected transverse bend (excluding any common mode).
    """
    rg = np.random.default_rng(seed)
    n = int(height_m * points_per_m)
    z = rg.uniform(0, height_m, n)
    zf = z / height_m
    r = np.maximum(radius_base_m - taper * z, 0.12)

    def shape(zz):
        if mode == "C":
            return amplitude_m * np.sin(np.pi * zz / height_m)
        if mode == "lean":
            return amplitude_m * zz / height_m
        if mode == "S":
            return amplitude_m * np.sin(2 * np.pi * zz / height_m)
        if mode == "wave":
            return amplitude_m * np.sin(2 * np.pi * zz / wavelength_m)
        raise ValueError(f"unknown mode {mode!r}")  # pragma: no cover

    bend = shape(z)

    axis_x = bend + (common_mode(zf) if common_mode is not None else 0.0)
    axis_y = np.tan(np.deg2rad(rake_deg)) * z

    theta = rg.uniform(0, 2 * np.pi, n)
    P = np.column_stack([
        axis_x + r * np.cos(theta),
        axis_y + r * np.sin(theta),
        z,
    ])

    if correlated and corr_len_m is not None:
        A = gp_correlated_field(z, theta, height_m, noise_m, corr_len_m,
                                anisotropy=corr_anisotropy,
                                angular_frac=corr_angular_frac, rng=rg)
        P[:, 0] += A[:, 0]
        P[:, 1] += A[:, 1]
    elif correlated:
        # Smooth low-order field: a few Fourier modes in (height, angle),
        # scaled to the requested RMS. This mimics SfM systematic error.
        A = np.zeros((n, 2))
        for kz, kt in [(1, 0), (2, 0), (1, 1), (2, 1), (3, 0)]:
            ph = rg.uniform(0, 2 * np.pi, 2)
            amp = rg.normal(0, 1, 2)
            A[:, 0] += amp[0] * np.sin(kz * np.pi * zf + kt * theta + ph[0])
            A[:, 1] += amp[1] * np.sin(kz * np.pi * zf + kt * theta + ph[1])
        A *= noise_m / (A.std() + 1e-9)
        P[:, 0] += A[:, 0]
        P[:, 1] += A[:, 1]
    else:
        P += rg.normal(0, noise_m, P.shape)

    if base_flare:
        nb = int(0.05 * n)
        zb = rg.uniform(0, 2.5, nb)
        rb = radius_base_m + rg.uniform(0.10, 0.60, nb)
        tb = rg.uniform(0, 2 * np.pi, nb)
        P = np.vstack([P, np.column_stack([
            rb * np.cos(tb),
            np.tan(np.deg2rad(rake_deg)) * zb + rb * np.sin(tb),
            zb,
        ])])

    if clutter_frac > 0:
        nc = int(clutter_frac * n)
        zc = rg.choice(np.linspace(4, height_m - 4, 6), nc) + rg.normal(0, 0.6, nc)
        rc = (radius_base_m - taper * zc) + rg.uniform(0.02, 0.28, nc)
        tc = rg.normal(0, 0.35, nc)  # clustered on one side
        zcf = zc / height_m
        bc = shape(zc)
        if common_mode is not None:
            bc = bc + common_mode(zcf)
        P = np.vstack([P, np.column_stack([
            bc + rc * np.cos(tc),
            np.tan(np.deg2rad(rake_deg)) * zc + rc * np.sin(tc),
            zc,
        ])])

    zg = np.linspace(0, height_m, 60 if mode != "wave" else 400)
    bt = shape(zg)
    return P, zg, bt * 1000.0


def gp_correlated_field(z, theta, height_m, rms_m, corr_len_m, anisotropy=1.0,
                        angular_frac=0.4, rng=None, grid_step_m=None):
    """Horizontal displacement field with an explicit correlation length.

    Each of four independent zero-mean Gaussian processes along the height has a
    squared-exponential covariance ``exp(-dz^2 / (2 l^2))`` with ``l = corr_len_m``.
    Two of them translate whole slices (x and y); two modulate an angle-dependent
    (order-1) term ``g(z) * sin(theta + phase)``. ``angular_frac`` sets the
    variance share of the angular part, ``anisotropy`` the x:y RMS ratio, and the
    mean-square displacement per horizontal component equals ``rms_m**2``.

    Returns an ``(n, 2)`` array of x and y displacements for points at heights
    ``z`` (metres) and angles ``theta`` (radians).
    """
    rng = np.random.default_rng() if rng is None else rng
    step = grid_step_m or min(0.25, corr_len_m / 4.0)
    zg = np.arange(0.0, height_m + step, step)
    dz = zg[:, None] - zg[None, :]
    K = np.exp(-0.5 * (dz / corr_len_m) ** 2)
    w, V = np.linalg.eigh(K)
    L = V * np.sqrt(np.clip(w, 0.0, None))
    G = L @ rng.normal(0.0, 1.0, (len(zg), 4))          # four unit-variance processes
    G /= G.std(axis=0, keepdims=True) + 1e-12
    gi = np.column_stack([np.interp(z, zg, G[:, k]) for k in range(4)])
    ph = rng.uniform(0.0, 2.0 * np.pi, 2)
    a_t, a_a = np.sqrt(1.0 - angular_frac), np.sqrt(angular_frac)
    norm = np.sqrt((1.0 + anisotropy ** 2) / 2.0)
    sx, sy = rms_m * anisotropy / norm, rms_m / norm
    ax = sx * (a_t * gi[:, 0] + a_a * np.sqrt(2.0) * gi[:, 2] * np.sin(theta + ph[0]))
    ay = sy * (a_t * gi[:, 1] + a_a * np.sqrt(2.0) * gi[:, 3] * np.sin(theta + ph[1]))
    return np.column_stack([ax, ay])


def detrend(z: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Remove the best-fit straight line -- the same operation the estimator does.

    This is essential for a fair comparison: the pipeline reports deviation from
    the mast's own straight raked axis, so the ground truth must be reduced the
    same way. Comparing against the raw axis offset instead produces a large
    spurious 'bias' that is purely an artefact of mismatched definitions.
    """
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = lstsq(A, b, rcond=None)
    return b - A @ coef


def recovered_profile(points: np.ndarray, **kwargs):
    """Run the pipeline and project the 2-D bend onto its dominant direction.

    Returns the profile on *physical* heights (the vertical coordinate of the input
    cloud), not on :attr:`MastResult.height`, which is measured from the lowest
    retained slice. The distinction matters for phantoms: when the flare test removes
    the bottom slices, ``height == 0`` lies several metres above the phantom base, and
    comparing the two height axes directly would offset the recovered profile from the
    truth.
    """
    from .pipeline import fit_mast

    res = fit_mast(points, **kwargs)
    B = np.column_stack([res.bend_transverse_mm, res.bend_longitudinal_mm])
    Bd = B - B.mean(axis=0)
    _, _, Vt = svd(Bd, full_matrices=False)
    z = res.world_z()
    order = np.argsort(z)
    return z[order], (B @ Vt[0])[order]


def compare_to_truth(points, z_truth, bend_truth, n_grid: int = 50, **kwargs):
    """Compare a recovered profile with ground truth, rotation- and sign-invariant.

    Both profiles are taken at the same physical heights, over the height range that
    the recovered profile covers (slices removed at the ends are not extrapolated),
    and the straight line is removed from each over that range, as in the pipeline.
    """
    h, b = recovered_profile(points, **kwargs)
    lo, hi = max(float(z_truth.min()), float(h.min())), min(float(z_truth.max()), float(h.max()))
    g = np.linspace(lo, hi, n_grid)
    T = detrend(g, np.interp(g, z_truth, bend_truth))
    R = detrend(g, np.interp(g, h, b))
    if np.corrcoef(R, T)[0, 1] < 0:
        R = -R
    rmse = float(np.sqrt(np.mean((R - T) ** 2)))
    peak_err = float(abs(np.max(np.abs(R)) - np.max(np.abs(T))))
    return g, T, R, rmse, peak_err


def ovality_null_floor(xy: np.ndarray, centre: np.ndarray, radius: float, seed: int = 0) -> float:
    """Ovality that a *perfectly round* ring would report, given this slice's sampling and noise.

    Any measured ovality below this floor is not evidence of ovalisation.
    """
    from .geometry import harmonic_ovality

    rg = np.random.default_rng(seed)
    d = np.asarray(xy) - np.asarray(centre)
    r = np.hypot(d[:, 0], d[:, 1])
    theta = np.arctan2(d[:, 1], d[:, 0])
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    r_round = radius + rg.normal(0, mad, len(r))
    synthetic = np.column_stack([r_round * np.cos(theta), r_round * np.sin(theta)])
    return harmonic_ovality(synthetic, np.zeros(2))


def slice_statistics(points, n_slices: int = 100, min_points: int = 40) -> dict:
    """Per-slice statistics that govern fitting accuracy.

    These are the quantities a synthetic phantom must reproduce before any
    accuracy figure derived from it may be transferred to real data:

    * ``sigma``    -- robust per-slice radial residual (mm)
    * ``trimmed``  -- fraction of points rejected as outliers (%)
    * ``coverage`` -- angular coverage of the ring (%)
    * ``density``  -- points per metre of height
    * ``z``        -- normalised height of each slice

    See :data:`REAL_CLOUD_STATS` for the measured values on the Golden Horizon
    UAV clouds, and ``scripts/08_phantom_match.py`` for the comparison.
    """
    from .geometry import irls_axis, orthonormal_basis, pca_axis, robust_circle

    P = np.asarray(points, dtype=float)
    c0, d0, _ = pca_axis(P)
    c, d = irls_axis(P, c0, d0)
    B, d = orthonormal_basis(d)
    t = (P - c) @ d
    xy = (P - c) @ B.T
    edges = np.linspace(t.min(), t.max(), n_slices + 1)
    span = t.max() - t.min()

    sigma, trimmed, coverage, density, z = [], [], [], [], []
    for k in range(n_slices):
        upper = t < edges[k + 1] if k < n_slices - 1 else t <= edges[k + 1]
        m = (t >= edges[k]) & upper
        if m.sum() < min_points:
            continue
        s = xy[m]
        cen, _, sg, _, inl = robust_circle(s)
        ang = np.degrees(np.arctan2(s[:, 1] - cen[1], s[:, 0] - cen[0]))
        sigma.append(sg * 1000.0)
        trimmed.append((1.0 - inl.mean()) * 100.0)
        coverage.append(len(set(((ang + 180) // 30).astype(int))) / 12.0 * 100.0)
        density.append(m.sum() / (span / n_slices))
        z.append((0.5 * (edges[k] + edges[k + 1]) - t.min()) / span)

    return {
        "sigma": np.asarray(sigma),
        "trimmed": np.asarray(trimmed),
        "coverage": np.asarray(coverage),
        "density": np.asarray(density),
        "z": np.asarray(z),
    }


def matched_phantom(seed: int = 0, correlated: bool = True, scatter_frac: float = 0.06, **kwargs):
    """A phantom calibrated to the *measured* per-slice statistics of the UAV clouds.

    The generic phantom of :func:`make_phantom` is much milder than the real data
    (radial residual sigma 9 mm against 46 mm; 0.2% trimmed against 5.6%; no heavy
    tail). This variant raises the surface noise and contamination to match, and adds a
    scattered heavy-tailed off-surface component that the band-only clutter model lacks
    (``scripts/08_phantom_match.py``, Table 2 of the paper).

    Each slice centre averages over hundreds of points, so per-point residual spread
    largely averages away; the limiting term is the spatially correlated component.

    Remaining differences: the real clouds have a heavier tail (sigma p90 112 mm
    against ~75 mm here) and mild angular-coverage gaps behind rigging (10th percentile
    92% against 100%), which this phantom does not reproduce.
    """
    # The real cloud carries BOTH error types, and they show up differently:
    #   * incoherent (within-slice) scatter -> the per-slice radial residual sigma
    #   * spatially-correlated error        -> masquerades as SHAPE, and leaves the
    #                                          slice residual almost untouched
    # Matching only the correlated field would therefore *undershoot* the measured
    # slice sigma (~23 mm against 46 mm), so both are injected: a correlated field
    # at the requested level, plus incoherent noise calibrated to the real sigma.
    params = dict(mode="C", amplitude_m=0.15, noise_m=0.045, clutter_frac=0.06,
                  base_flare=True, correlated=correlated, seed=seed)
    params.update(kwargs)
    P, zt, bt = make_phantom(**params)

    if correlated:
        # add the incoherent component that the smooth field does not produce
        rg_w = np.random.default_rng(seed + 4242)
        P = P + rg_w.normal(0, INCOHERENT_NOISE_M, P.shape)

    if scatter_frac > 0:
        rg = np.random.default_rng(seed + 991)
        n = int(scatter_frac * len(P))
        z = rg.uniform(P[:, 2].min(), P[:, 2].max(), n)
        th = rg.uniform(0, 2 * np.pi, n)
        # heavy-tailed radial offset: the real clouds' outliers are not Gaussian
        r = 0.5 + rg.exponential(0.35, n)
        rake = np.tan(np.deg2rad(params.get("rake_deg", 2.0)))
        P = np.vstack([P, np.column_stack([
            r * np.cos(th), r * np.sin(th) + rake * z, z,
        ])])
    return P, zt, bt


def phantom_geometry(mast_index: int) -> dict:
    """Geometry keywords reproducing a *specific* real mast (0 = Foremast .. 4 = Jiggermast).

    Pass straight into :func:`matched_phantom` or :func:`make_phantom`::

        P, z, b = matched_phantom(seed=i, **phantom_geometry(i))

    Without this, every phantom is the same generic 50 m tube: the seed changes only
    the noise realisation, not the shape. The real masts differ by 12 m in height and
    by a factor of six in taper, so each mast gets its own phantom.
    """
    name, height, radius, taper, density = REAL_MAST_GEOMETRY[mast_index]
    return {
        "height_m": height,
        "radius_base_m": radius,
        "taper": taper,
        "points_per_m": density,
    }


def ovality_study(points, n_slices: int = 100, k_mad: float = 2.5, min_points: int = 40,
                  min_coverage: float = 0.55, kappa: float = 3.0) -> dict:
    """Per-slice ovality, its round-ring null floor, and a one-sector masking test.

    Uses the same slicing, coverage gate, trimmed circle fit and slice-level quality
    control as :func:`mastfit.pipeline.fit_mast`, and reports, for every retained
    slice:

    ``ovality``  second-harmonic ovality (%) of the trimmed ring,
    ``null``     the ovality a perfectly round ring returns with that slice's own
                 angular sampling and radial noise (:func:`ovality_null_floor`),
    ``masked``   the ovality after removing the 30-degree sector with the largest
                 mean absolute radial residual and refitting -- the test of whether
                 a single one-sided attachment (e.g. a welded mast track) explains
                 the excess.
    """
    from .geometry import harmonic_ovality, irls_axis, orthonormal_basis, pca_axis, robust_circle
    from .quality import slice_quality_mask

    P = np.asarray(points, dtype=float)
    c0, d0, _ = pca_axis(P)
    c, d = irls_axis(P, c0, d0)
    B, d = orthonormal_basis(d)
    t = (P - c) @ d
    xy = (P - c) @ B.T
    edges = np.linspace(t.min(), t.max(), n_slices + 1)
    h, rad, res, ova, nul, msk = [], [], [], [], [], []
    for k in range(n_slices):
        upper = t < edges[k + 1] if k < n_slices - 1 else t <= edges[k + 1]
        m = (t >= edges[k]) & upper
        if m.sum() < min_points:
            continue
        s = xy[m]
        c1 = np.median(s, axis=0)
        ang = np.degrees(np.arctan2(s[:, 1] - c1[1], s[:, 0] - c1[0]))
        if len(set(((ang + 180) // 30).astype(int))) / 12.0 < min_coverage:
            continue
        cen, r, sg, _cov, inl = robust_circle(s, k_mad=k_mad)
        si = s[inl]
        if len(si) < min_points:
            continue
        h.append(0.5 * (edges[k] + edges[k + 1]))
        rad.append(r)
        res.append(sg * 1000.0)
        ova.append(harmonic_ovality(si, cen))
        nul.append(ovality_null_floor(si, cen, r, seed=k))
        dd = si - cen
        sector = ((np.degrees(np.arctan2(dd[:, 1], dd[:, 0])) + 180) // 30).astype(int)
        rr = np.abs(np.hypot(dd[:, 0], dd[:, 1]) - r)
        worst = max(set(sector), key=lambda q: rr[sector == q].mean())
        s2 = si[sector != worst]
        cen2, _r2, _s2, _c2, inl2 = robust_circle(s2, k_mad=k_mad)
        msk.append(harmonic_ovality(s2[inl2], cen2))
    h, rad, res, ova, nul, msk = map(np.asarray, (h, rad, res, ova, nul, msk))
    keep = slice_quality_mask(h, rad, res, ova, kappa=kappa)
    return {"ovality": ova[keep], "null": nul[keep], "masked": msk[keep],
            "height": h[keep] - h[keep].min()}
