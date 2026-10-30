"""Five-mast shared-curvature test (candidate photogrammetric doming).

This is the primary exportable idea in this repository, and it is not
mast-specific: it works for *any* scene in which several like structures are
reconstructed together (turbine farms, chimney clusters, pipe racks, tank yards).

Where the literature mitigates doming at capture time (oblique imagery, flight
design) or corrects a finished terrain product with a modelled function, this test
exploits a property of the *scene itself* -- redundancy across instances -- and so
needs no extra capture, no reference surface, and no calibration model.

Rationale
---------
Structure-from-motion reconstructions can contain a broad-scale systematic
curvature ("doming"/"bowling") that bends a perfectly straight object. Such an
artefact is a property of the *reconstruction volume*, not of any one mast, so it
imprints a common low-order curvature on every mast that was reconstructed
together. Genuine structural bends, by contrast, are independent per mast.

Decomposing the five bending profiles into a shared component plus mast-specific
residuals therefore *bounds* the artefact, and this is available for free
because the masts were imaged jointly.

What it can and cannot say
--------------------------
* A shared component is **not proof** of doming. A common structural or
  environmental cause (identical rig, common berth and thermal state) would look
  the same. On the real data (:func:`positional_trend`, exact permutation test):
  the transverse loading shows no trend with mast position (r = -0.29, p = 0.65);
  the longitudinal loading falls monotonically from bow to stern (r = -0.96,
  p = 0.017), but the trend is carried by the Foremast, whose large longitudinal
  bend dominates the mean profile, and it disappears when the shared component is
  recomputed without the Foremast (r = +0.56, p = 0.50). The test is therefore
  inconclusive, and the shared component is called a *candidate* common-mode
  artefact.
* Subtracting the shared component also removes any genuinely common structural
  lean, so the corrected bends are a **conservative lower bound**; the raw bends
  are the upper bound.
* On synthetic data the test recovers an injected 49% shared fraction as 51%, so it
  recovers the common mode only approximately (``scripts/05_synthetic_validation.py``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["shared_fraction", "doming_decomposition", "shared_loadings", "positional_trend"]


def _resample(profiles, heights, grid):
    return np.array([
        np.interp(grid, h / h[-1], p) for h, p in zip(heights, profiles)
    ])


def shared_fraction(M: np.ndarray) -> tuple[float, np.ndarray]:
    """Fraction of bending-shape variance explained by the component common to all masts.

    Parameters
    ----------
    M : (n_masts, n_heights) array
        Bending profiles resampled on a common normalised-height grid.

    Returns
    -------
    fraction : float in [0, 1]
    shared : (n_heights,) array
        The common (mean) profile.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    shared = M.mean(axis=0)
    shared_var = n * np.sum((shared - shared.mean()) ** 2)
    resid_var = np.sum((M - shared) ** 2)
    denom = shared_var + resid_var
    if denom <= 0:  # pragma: no cover
        return 0.0, shared
    return float(shared_var / denom), shared


def doming_decomposition(results, n_grid: int = 36):
    """Split the five bending profiles into shared + mast-specific components.

    Returns a dict with the shared fractions (transverse and longitudinal), the
    shared profiles, and the doming-corrected per-mast profiles.
    """
    grid = np.linspace(0.05, 0.95, n_grid)
    heights = [r.height for r in results]
    BX = _resample([r.bend_transverse_mm for r in results], heights, grid)
    BY = _resample([r.bend_longitudinal_mm for r in results], heights, grid)

    fx, sx = shared_fraction(BX)
    fy, sy = shared_fraction(BY)

    return {
        "grid": grid,
        "transverse": BX,
        "longitudinal": BY,
        "shared_fraction_transverse": fx,
        "shared_fraction_longitudinal": fy,
        "shared_transverse": sx,
        "shared_longitudinal": sy,
        "corrected_transverse": BX - sx,
        "corrected_longitudinal": BY - sy,
    }


def shared_loadings(M: np.ndarray) -> np.ndarray:
    """Least-squares loading of each (mean-removed) profile on the shared profile.

    ``M`` is ``(n_masts, n_heights)``. The loadings average to 1 by construction.
    """
    M = np.asarray(M, dtype=float)
    s = M.mean(axis=0)
    sc = s - s.mean()
    return np.array([np.dot(m - m.mean(), sc) / np.dot(sc, sc) for m in M])


def positional_trend(M: np.ndarray, positions=None) -> dict:
    """Does the loading on the shared component vary with mast position?

    A reconstruction-wide distortion would tend to imprint a trend along the deck.
    With five masts an asymptotic test is meaningless, so the p-value is exact: the
    Pearson correlation between position and loading is compared with its value
    under all ``n!`` orderings of the masts (two-sided).

    Returns ``{"loadings", "r", "p", "n_perm"}``.
    """
    from itertools import permutations

    M = np.asarray(M, dtype=float)
    x = np.arange(len(M), dtype=float) if positions is None else np.asarray(positions, float)
    load = shared_loadings(M)
    r0 = float(np.corrcoef(x, load)[0, 1])
    rs = np.array([np.corrcoef(x, np.asarray(p))[0, 1] for p in permutations(load)])
    p = float(np.mean(np.abs(rs) >= abs(r0) - 1e-12))
    return {"loadings": load, "r": r0, "p": p, "n_perm": len(rs)}
