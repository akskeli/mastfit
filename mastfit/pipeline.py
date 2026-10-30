"""The deterministic mast-deformation pipeline.

Stages
------
I.   PCA axis seed -> IRLS robust refinement -> taper-aware sectioning ->
     per-slice robust circle fit (with centre covariance).
II.  Contamination rejection: base-flare removal, rigging-band suppression.
III. As-built centreline (fixed-bandwidth kernel smoother) -> straight raked
     axis (least-squares line) -> rake/bend decomposition.

Why a *fixed* smoothing bandwidth?
----------------------------------
A cross-validated (GCV) penalty does not guarantee that the recovered bend is
independent of the slice thickness ``delta``: with fine slicing the centreline
chases per-slice noise and the apparent bend inflates severalfold. A fixed
bandwidth ``L`` (metres) makes the estimate delta-independent by construction
and gives an explicit *resolved bending wavelength* (>~ L) below which localized
deformation is not reported. See ``scripts/06_sensitivity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import lstsq

from .geometry import (
    fit_ellipse,
    harmonic_ovality,
    irls_axis,
    orthonormal_basis,
    pca_axis,
    robust_circle,
)
from .quality import slice_quality_breakdown

__all__ = ["MastResult", "fit_mast"]


@dataclass
class MastResult:
    """Result of fitting one mast."""

    name: str = ""
    n_points: int = 0
    eig_ratio: float = 0.0
    rake_deg: float = 0.0
    rake_azimuth_deg: float = 0.0
    median_diameter_mm: float = 0.0

    # centreline sampled on `height` (metres above the lowest retained slice)
    height: np.ndarray = field(default_factory=lambda: np.array([]))
    bend_transverse_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    bend_longitudinal_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    bend_total_mm: np.ndarray = field(default_factory=lambda: np.array([]))

    # per-slice diagnostics (retained slices only)
    slice_height: np.ndarray = field(default_factory=lambda: np.array([]))
    slice_sigma_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    slice_radius_m: np.ndarray = field(default_factory=lambda: np.array([]))
    slice_ovality_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    n_slices_kept: int = 0
    n_slices_total: int = 0

    # geometry needed to place the profile in the input frame (cross-sensor comparison)
    axis_centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    height_origin_m: float = 0.0   # along-axis coordinate of `height == 0`, from axis_centroid
    # smoothed centreline in the coordinates of the input cloud (n_eval x 3), before the
    # straight line is removed; used to compare two sensors without re-referencing
    centre_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # every accepted slice, before slice-level quality control (Fig. 5)
    all_slice_height: np.ndarray = field(default_factory=lambda: np.array([]))
    all_slice_radius_m: np.ndarray = field(default_factory=lambda: np.array([]))
    all_slice_ovality_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    all_slice_residual_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    flare_removed: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    band_flagged: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    @property
    def max_bend_mm(self) -> float:
        return float(np.nanmax(self.bend_total_mm)) if self.bend_total_mm.size else float("nan")

    @property
    def median_sigma_mm(self) -> float:
        return float(np.nanmedian(self.slice_sigma_mm)) if self.slice_sigma_mm.size else float("nan")

    def world_z(self, height=None) -> np.ndarray:
        """Vertical coordinate, in the frame of the input cloud, of profile heights.

        ``height`` defaults to :attr:`height`. Used to pair two sensors at the same
        physical height after co-registration (:mod:`mastfit.crossval`).
        """
        h = self.height if height is None else np.asarray(height, dtype=float)
        return self.axis_centroid[2] + (h + self.height_origin_m) * self.axis_direction[2]


def _extract_slices(points, centroid, direction, n_slices, k_mad, min_points, min_coverage,
                    section_model="circle"):
    """Section the cloud perpendicular to the axis and robustly fit each slice."""
    B, d = orthonormal_basis(direction)
    t = (points - centroid) @ d          # along-axis coordinate
    xy = (points - centroid) @ B.T       # in-plane coordinates
    edges = np.linspace(t.min(), t.max(), n_slices + 1)

    h, u, v, radius, sigma, resid, ovality = [], [], [], [], [], [], []
    for k in range(n_slices):
        upper = t < edges[k + 1] if k < n_slices - 1 else t <= edges[k + 1]
        m = (t >= edges[k]) & upper
        if m.sum() < min_points:
            continue
        s = xy[m]

        # Angular-coverage gate: a slice seen over a short or lopsided arc gives
        # a poorly constrained centre, so require points across enough sectors.
        c0 = np.median(s, axis=0)
        ang = np.degrees(np.arctan2(s[:, 1] - c0[1], s[:, 0] - c0[0]))
        sectors = set(((ang + 180) // 30).astype(int))
        if len(sectors) / 12.0 < min_coverage:
            continue

        cen, r, sg, cov, inl = robust_circle(s, k_mad=k_mad)
        s_in = s[inl]
        if len(s_in) < min_points:
            continue

        if section_model == "ellipse":
            # centre of a direct ellipse fit to the circle-fit inliers; the circle centre is
            # kept if the ellipse is degenerate, implausibly elongated or far off
            el = fit_ellipse(s_in)
            if el is not None:
                c_e, ax_e, _ = el
                if ax_e[0] / ax_e[1] < 1.6 and np.hypot(*(c_e - cen)) < 0.5 * r:
                    cen = c_e
        h.append(0.5 * (edges[k] + edges[k + 1]))
        u.append(cen[0])
        v.append(cen[1])
        radius.append(r)
        sigma.append(np.sqrt(max(np.trace(cov), 1e-12)) * 1000.0)  # mm
        resid.append(sg * 1000.0)                                   # mm
        ovality.append(harmonic_ovality(s_in, cen))

    return (
        np.asarray(h), np.asarray(u), np.asarray(v), np.asarray(radius),
        np.asarray(sigma), np.asarray(resid), np.asarray(ovality), B, d,
    )


def _kernel_smooth(z_knots, values, weights, z_eval, bandwidth):
    """Weighted Gaussian kernel smoother with fixed bandwidth (metres)."""
    D = (z_eval[:, None] - z_knots[None, :]) / bandwidth
    K = np.exp(-0.5 * D * D) * weights[None, :]
    return (K @ values) / K.sum(axis=1)


def fit_mast(
    points: np.ndarray,
    name: str = "",
    n_slices: int = 100,
    bandwidth_m: float = 3.0,
    k_mad: float = 2.5,
    kappa: float = 3.0,
    min_points_per_slice: int = 40,
    min_coverage: float = 0.55,
    apply_quality_control: bool = True,
    n_eval: int = 200,
    tukey_c: float = 4.685,
    irls_iter: int = 5,
    flare_removal: bool = True,
    band_suppression: bool = True,
    variance_weighting: bool = True,
    axis=None,
    section_model: str = "circle",
) -> MastResult:
    """Fit one mast and return its rake and bending profile.

    Parameters
    ----------
    points : (n, 3) array
        Segmented point cloud for a single mast (metres).
    n_slices : int
        Number of slices along the axis. With the fixed-bandwidth smoother the
        result is insensitive to this (see ``scripts/06_sensitivity.py``).
    bandwidth_m : float
        Kernel bandwidth L. Sets the resolved bending wavelength.
    apply_quality_control : bool
        If False, skip base-flare/band rejection (used to *demonstrate* the
        effect of contamination; not a supported operating mode).
    tukey_c : float
        Tuning constant of Tukey's biweight in the IRLS axis refinement, in units
        of the robust scale (1.4826 MAD). 4.685 gives 95% asymptotic efficiency
        under Gaussian residuals.
    irls_iter : int
        Number of IRLS reweightings; 0 keeps the PCA seed (used only for the
        sensitivity study in ``scripts/14_robustness_checks.py``).
    flare_removal, band_suppression, variance_weighting : bool
        Switch off one component at a time (ablation study,
        ``scripts/15_significance_and_ablations.py``); the defaults are the operating pipeline.
    axis : (centroid, direction) or None
        Use this axis instead of the PCA + IRLS estimate (e.g. a robust cone fit).
    section_model : {"circle", "ellipse"}
        Slice centre from the trimmed circle fit (default) or from a direct ellipse fit to
        its inliers (sensitivity study, ``scripts/19_section_model.py``).
    """
    P = np.asarray(points, dtype=float)
    c0, d0, eig = pca_axis(P)
    if axis is None:
        centroid, direction = irls_axis(P, c0, d0, n_iter=irls_iter, tukey_c=tukey_c)
    else:
        centroid, direction = (np.asarray(a, dtype=float) for a in axis)

    h, u, v, radius, sigma, resid, ovality, B, d = _extract_slices(
        P, centroid, direction, n_slices, k_mad, min_points_per_slice, min_coverage,
        section_model=section_model,
    )
    n_total = len(h)
    if n_total < 8:
        raise ValueError(f"only {n_total} usable slices for mast {name!r}: too sparse to fit")

    qc = slice_quality_breakdown(h, radius, resid, ovality, kappa=kappa)
    if not apply_quality_control:
        keep = np.ones(n_total, dtype=bool)
    elif flare_removal and band_suppression:
        keep = qc["keep"]
    else:
        keep = np.ones(n_total, dtype=bool)
        if flare_removal:
            keep &= ~qc["flare"]
        if band_suppression:
            keep &= ~qc["band"]
        if keep.sum() < 8:          # same guard as slice_quality_breakdown
            keep = np.ones(n_total, dtype=bool)

    hk, uk, vk = h[keep], u[keep], v[keep]
    sig_k, rad_k, ova_k = sigma[keep], radius[keep], ovality[keep]
    order = np.argsort(hk)
    hk, uk, vk, sig_k, rad_k, ova_k = (
        hk[order], uk[order], vk[order], sig_k[order], rad_k[order], ova_k[order]
    )

    # Inverse-variance weights: well-determined slices dominate the centreline.
    w = 1.0 / np.maximum(sig_k, 1e-3) ** 2 if variance_weighting else np.ones_like(sig_k)
    z = np.linspace(hk.min(), hk.max(), n_eval)
    cu = _kernel_smooth(hk, uk, w, z, bandwidth_m)
    cv = _kernel_smooth(hk, vk, w, z, bandwidth_m)

    # Straight raked reference axis: ordinary least-squares line through the
    # smoothed *clean* centreline (one line per in-plane coordinate). Deformation is measured against the mast's own axis, so no CAD
    # model or external alignment is needed (self-referential).
    A = np.column_stack([np.ones_like(z), z])
    au, *_ = lstsq(A, cu, rcond=None)
    av, *_ = lstsq(A, cv, rcond=None)
    bx = (cu - A @ au) * 1000.0
    by = (cv - A @ av) * 1000.0

    # Rake: tilt of the fitted straight axis from vertical, in world coordinates.
    slope_world = au[1] * B[0] + av[1] * B[1] + d
    slope_world = slope_world / np.linalg.norm(slope_world)
    rake = np.degrees(np.arccos(abs(slope_world[2])))
    azimuth = np.degrees(np.arctan2(slope_world[1], slope_world[0]))

    return MastResult(
        name=name,
        n_points=len(P),
        eig_ratio=float(eig[0] / eig[1]),
        rake_deg=float(rake),
        rake_azimuth_deg=float(azimuth),
        median_diameter_mm=float(np.nanmedian(rad_k) * 2000.0),
        height=z - z.min(),
        bend_transverse_mm=bx,
        bend_longitudinal_mm=by,
        bend_total_mm=np.hypot(bx, by),
        slice_height=hk - hk.min(),
        slice_sigma_mm=sig_k,
        slice_radius_m=rad_k,
        slice_ovality_pct=ova_k,
        n_slices_kept=int(keep.sum()),
        n_slices_total=n_total,
        axis_centroid=np.asarray(centroid, dtype=float),
        axis_direction=np.asarray(d, dtype=float),
        height_origin_m=float(z.min()),
        centre_world=(np.asarray(centroid, float)[None, :] + z[:, None] * np.asarray(d, float)[None, :]
                      + cu[:, None] * B[0][None, :] + cv[:, None] * B[1][None, :]),
        all_slice_height=h - h.min(),
        all_slice_radius_m=radius,
        all_slice_ovality_pct=ovality,
        all_slice_residual_mm=resid,
        flare_removed=qc["flare"],
        band_flagged=qc["band"],
    )
