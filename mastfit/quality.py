"""Slice-level quality control: base-flare removal and rigging-band suppression.

Two structural contaminations bias the centreline even after the per-slice
trimmed circle fit, because they act at the level of *whole slices* rather than
individual points.

*Contamination* here means points that do not lie on the mast surface (rigging,
stays, welded attachments, the deck-collar flare) -- as distinct from correlated
reconstruction error on the surface points themselves.

1. **Base flare (deck collar).** The lowest metres flare where the tube steps
   through the deck, producing anomalously large, non-circular cross-sections.
   Left in, these slices anchor and tilt the straight reference line and inject
   false bending along the *entire* mast. This is not a subtle effect: on a
   naive best-fit-line pipeline it inflates one mast's apparent bend from
   87 mm to 410 mm.

2. **Rigging bands.** Yard, gaff and stay attachments occur at quasi-periodic
   heights (roughly six yards on each of the four square-rigged masts of this
   five-masted barque) and add clustered off-tube points that survive the radial
   trim. They show up as *local spikes*
   in the circle-fit residual and slice ovality relative to their neighbours.

Both tests are fixed-constant threshold rules on a statistic, so the procedure
remains deterministic and free of manually chosen heights.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq

__all__ = [
    "rolling_median",
    "base_flare_mask",
    "rigging_band_mask",
    "slice_quality_mask",
    "slice_quality_breakdown",
]


def rolling_median(a: np.ndarray, window: int = 7) -> np.ndarray:
    """Centred rolling median with edge clamping."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.nanmedian(a[lo:hi])
    return out


def base_flare_mask(
    heights: np.ndarray,
    radii: np.ndarray,
    kappa: float = 3.0,
    core_percentiles: tuple[float, float] = (20.0, 80.0),
) -> np.ndarray:
    """Flag contiguous bottom/top slices that violate the taper trend.

    A robust straight taper ``r(z) = alpha + beta z`` is fitted on the central
    portion of the mast (where taper is regular), and any *contiguous run* of
    slices at the bottom or top whose radius departs from the trend by more than
    ``kappa`` robust scales is rejected (Section 3.6 of the paper).

    Only contiguous runs from the ends are removed, so a mid-mast slice is never
    dropped by this test.

    Returns
    -------
    keep : bool array
        True for slices to retain.
    """
    heights = np.asarray(heights, dtype=float)
    radii = np.asarray(radii, dtype=float)
    keep = np.ones(len(heights), dtype=bool)
    if len(heights) < 8:
        return keep

    lo, hi = np.percentile(heights, core_percentiles)
    core = (heights >= lo) & (heights <= hi)
    if core.sum() < 3:  # pragma: no cover
        return keep

    A = np.column_stack([np.ones(core.sum()), heights[core]])
    taper, *_ = lstsq(A, radii[core], rcond=None)
    resid = radii - (taper[0] + taper[1] * heights)
    mad = 1.4826 * np.median(np.abs(resid[core] - np.median(resid[core]))) + 1e-9
    ok = np.abs(resid) < kappa * mad

    i = 0
    while i < len(heights) and not ok[i]:
        keep[i] = False
        i += 1
    i = len(heights) - 1
    while i >= 0 and not ok[i]:
        keep[i] = False
        i -= 1
    return keep


def rigging_band_mask(
    fit_residual: np.ndarray,
    ovality: np.ndarray,
    kappa: float = 3.0,
    window: int = 7,
) -> np.ndarray:
    """Flag slices that spike above a rolling-median baseline.

    A rigging band manifests as a *local* excursion (relative to vertical
    neighbours) in the circle-fit residual and/or the slice ovality.

    Returns
    -------
    band : bool array
        True where a band is detected (these slices should be dropped).
    """
    fit_residual = np.asarray(fit_residual, dtype=float)
    ovality = np.asarray(ovality, dtype=float)

    res_dev = fit_residual - rolling_median(fit_residual, window)
    ova_dev = np.nan_to_num(ovality - rolling_median(ovality, window))

    res_scale = 1.4826 * np.median(np.abs(res_dev)) + 1e-9
    ova_scale = 1.4826 * np.median(np.abs(ova_dev)) + 1e-9

    return (res_dev > kappa * res_scale) | (ova_dev > kappa * ova_scale)


def slice_quality_breakdown(
    heights, radii, fit_residual, ovality, kappa=3.0, min_kept=8
) -> dict:
    """Both slice tests, reported separately (used for Fig. 5 and the numbers ledger).

    Returns a dict of boolean arrays over the input slices:

    ``flare``  slices removed by the taper-consistency (base/top flare) test,
    ``band``   slices flagged by the rigging-band spike test,
    ``keep``   slices retained (``~flare & ~band``), and the flag
    ``guard``  True if fewer than ``min_kept`` slices survived, in which case every
               slice is retained so that a sparse mast is not trimmed into an
               unfittable state.
    """
    flare = ~base_flare_mask(heights, radii, kappa=kappa)
    band = rigging_band_mask(fit_residual, ovality, kappa=kappa)
    keep = ~flare & ~band
    guard = bool(keep.sum() < min_kept)
    if guard:
        keep = np.ones(len(heights), dtype=bool)
    return {"flare": flare, "band": band, "keep": keep, "guard": guard}


def slice_quality_mask(
    heights, radii, fit_residual, ovality, kappa=3.0, min_kept=8
) -> np.ndarray:
    """Combined mask: keep slices that pass both the flare and band tests."""
    return slice_quality_breakdown(
        heights, radii, fit_residual, ovality, kappa=kappa, min_kept=min_kept
    )["keep"]
