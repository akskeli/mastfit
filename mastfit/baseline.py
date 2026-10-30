"""The unsmoothed baseline variant of the pipeline (Section 4.2, Figures 6 and 10a).

This is the pipeline a practitioner would write without the kernel smoother: slice
centres are strung directly onto a weighted straight line, and the bend is the
departure of the centres from that line. It is *not* an operating mode. It exists to
expose the base-flare mechanism, which the kernel-smoothed operating pipeline
largely masks (changes of at most 3 mm on the real masts).

Paper values (40 slices per mast):

    real data   Mizzenmast 87 -> 410 mm, Middle Mast 104 -> 324 mm when the flare
                test is disabled; Foremast 218 -> 257, Mainmast 132 -> 135,
                Jiggermast 115 -> 131 mm
    phantom     true 94 mm; recovered 127 mm without and 86 mm with the flare test
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq

from .geometry import harmonic_ovality, irls_axis, orthonormal_basis, pca_axis, robust_circle
from .quality import slice_quality_mask

__all__ = ["naive_pipeline"]


def naive_pipeline(P, n_slices: int = 40, quality_control: bool = True,
                   return_profile: bool = False):
    """Slice centres fitted directly with a weighted best-fit line (no smoothing).

    Returns the maximum total lateral bend (mm). With ``return_profile=True`` it
    returns ``(max_bend_mm, height_m, bend_total_mm)`` for the retained slices,
    with height measured from the lowest accepted slice *before* quality control, so
    that profiles with and without the flare test share one height axis.
    """
    c0, d0, _ = pca_axis(P)
    c, d = irls_axis(P, c0, d0)
    B, d = orthonormal_basis(d)
    t = (P - c) @ d
    xy = (P - c) @ B.T
    edges = np.linspace(t.min(), t.max(), n_slices + 1)
    H, CEN, RAD, SIG, RES, OVA = [], [], [], [], [], []
    for k in range(n_slices):
        upper = t < edges[k + 1] if k < n_slices - 1 else t <= edges[k + 1]
        m = (t >= edges[k]) & upper
        if m.sum() < 40:
            continue
        s = xy[m]
        c1 = np.median(s, axis=0)
        ang = np.degrees(np.arctan2(s[:, 1] - c1[1], s[:, 0] - c1[0]))
        if len(set(((ang + 180) // 30).astype(int))) / 12 < 0.6:
            continue
        cen, r, sg, cov, inl = robust_circle(s)
        if inl.sum() < 40:
            continue
        h = 0.5 * (edges[k] + edges[k + 1])
        H.append(h)
        CEN.append(c + cen[0] * B[0] + cen[1] * B[1] + h * d)
        RAD.append(r)
        SIG.append(np.sqrt(max(np.trace(cov), 1e-12)) * 1000)
        RES.append(sg * 1000)
        OVA.append(harmonic_ovality(s[inl], cen))
    H, RAD, SIG, RES, OVA = map(np.asarray, (H, RAD, SIG, RES, OVA))
    CEN = np.asarray(CEN)
    keep = (slice_quality_mask(H, RAD, RES, OVA, min_kept=6) if quality_control
            else np.ones(len(H), bool))
    Hk, CENk, SIGk = H[keep], CEN[keep], SIG[keep]
    w = 1 / np.maximum(SIGk, 1e-3)
    A = np.column_stack([np.ones_like(Hk), Hk])
    coef, *_ = lstsq(A * w[:, None], CENk * w[:, None], rcond=None)
    bend = CENk - A @ coef
    total = np.hypot(bend[:, 0], bend[:, 1]) * 1000
    peak = float(np.nanmax(total))
    if return_profile:
        return peak, Hk - H.min(), total
    return peak
