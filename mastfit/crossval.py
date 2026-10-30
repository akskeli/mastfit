"""Cross-validation of the UAV result against an independent terrestrial laser scan.

A second sensor with different error physics (TLS has no SfM doming and no
correlated photogrammetric error) measures the same masts. Both clouds are
co-registered into the common ship frame (:func:`mastfit.io.coregister`), the
*identical* pipeline is run on each, and the bending profiles are compared at the
same physical heights by Bland-Altman analysis.

Pairing rule
------------
Profiles are paired at the same ship-frame height over the height span that both
sensors cover, and each profile is re-referenced to the least-squares straight line
over that common span. This makes the two estimands identical: bending is defined
relative to a straight line, so two profiles referenced to lines fitted over
different spans differ by a linear trend that is not a measurement disagreement.
No sign is chosen from the data; both clouds are mapped into the ship frame in the
same way.

Because each re-referenced profile has zero mean, the Bland-Altman bias is zero by
construction; the informative quantities are the spread of the differences (limits
of agreement), the correlation, and the per-mast RMS difference.

Result on the Golden Horizon data (four masts with adequate laser coverage;
scripts/04_crossvalidation.py):

    transverse   : 95% LoA +-28 mm (mast-block bootstrap CI +-18 to +-36), r = +0.65
    longitudinal : 95% LoA +-65 mm (mast-block bootstrap CI +-49 to +-77), r = +0.44

The Foremast is excluded: its laser cloud has only 1,930 points, too sparse to fit.
The Foremast carries the largest UAV bend (160 mm), so the largest signal in the
study is the one that cannot be independently confirmed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bland_altman",
    "common_span_profiles",
    "cross_validate",
    "lag1_autocorrelation",
    "block_bootstrap_loa",
]


def bland_altman(a: np.ndarray, b: np.ndarray) -> dict:
    """Bias, limits of agreement, and correlation between two measurements."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    bias = float(diff.mean())
    sd = float(diff.std())
    return {
        "bias_mm": bias,
        "sd_mm": sd,
        "loa_mm": 1.96 * sd,
        "corr": float(np.corrcoef(a, b)[0, 1]),
        "rms_diff_mm": float(np.sqrt((diff ** 2).mean())),
        "mean": (a + b) / 2,
        "diff": diff,
    }


def _detrend(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def common_span_profiles(res_a, res_b, attr: str, n: int = 60):
    """Pair two fitted masts at the same height and re-reference both to one line.

    ``res_a`` and ``res_b`` are :class:`~mastfit.pipeline.MastResult` objects fitted
    to clouds that share a vertical datum (e.g. both in the common ship frame).

    Returns ``(z, a, b)``: ``n`` heights spanning the overlap of the two retained
    height ranges, and the two ``attr`` profiles (e.g. ``"bend_transverse_mm"``) at
    those heights, each with its least-squares line over the overlap removed.
    """
    za, zb = res_a.world_z(), res_b.world_z()
    lo, hi = max(za.min(), zb.min()), min(za.max(), zb.max())
    if hi - lo <= 0:
        raise ValueError("the two profiles do not overlap in height")
    z = np.linspace(lo, hi, n)
    a = _detrend(z, np.interp(z, za, getattr(res_a, attr)))
    b = _detrend(z, np.interp(z, zb, getattr(res_b, attr)))
    return z, a, b


def cross_validate(drone_results, laser_results, indices, n_per_mast: int = 60,
                   n_boot: int = 4000, seed: int = 1) -> dict:
    """Compare the two sensors per component, pooled over the usable masts.

    Parameters
    ----------
    drone_results, laser_results : list[MastResult]
        Indexed by physical mast index (0 = Foremast .. 4 = Jiggermast), fitted to
        clouds in the common ship frame.
    indices : iterable[int]
        Masts with adequate coverage in *both* sensors.

    Returns
    -------
    dict with, per component (``"transverse"``, ``"longitudinal"``), the pooled
    Bland-Altman statistics plus ``per_mast`` (heights, both profiles and their
    difference), the mast-block bootstrap CI on the limits of agreement, the lag-1
    autocorrelation and the effective sample size; and ``span`` (overlap per mast)
    and ``peak_total`` (maximum total bend of each sensor over the common span).
    """
    out = {"span": {}, "peak_total": {}}
    for comp, attr in (("transverse", "bend_transverse_mm"),
                       ("longitudinal", "bend_longitudinal_mm")):
        per_mast, A, B = {}, [], []
        for i in indices:
            z, a, b = common_span_profiles(drone_results[i], laser_results[i], attr, n_per_mast)
            per_mast[i] = {"z": z, "uav": a, "tls": b, "diff": a - b}
            A.append(a)
            B.append(b)
            out["span"][i] = (float(z[0]), float(z[-1]))
        stats = bland_altman(np.concatenate(A), np.concatenate(B))
        loa, lo, hi, r1, neff = block_bootstrap_loa(
            [per_mast[i]["diff"] for i in indices], n_boot=n_boot, seed=seed)
        stats.update({"per_mast": per_mast, "loa_ci_mm": (lo, hi), "lag1": r1,
                      "n_effective": neff, "n": int(sum(len(v["diff"]) for v in per_mast.values()))})
        out[comp] = stats
    for i in indices:
        tu = np.hypot(out["transverse"]["per_mast"][i]["uav"], out["longitudinal"]["per_mast"][i]["uav"])
        tt = np.hypot(out["transverse"]["per_mast"][i]["tls"], out["longitudinal"]["per_mast"][i]["tls"])
        out["peak_total"][i] = (float(tu.max()), float(tt.max()))
    return out


def lag1_autocorrelation(series_list):
    """Mean lag-1 autocorrelation of per-mast difference series.

    Computed per mast and averaged, never across mast boundaries. The kernel-smoothed
    profiles make consecutive per-height differences highly dependent (r1 ~ 0.98 on the
    real data), so the nominal sample count greatly overstates the information content:
    the effective sample size is n * (1 - r1) / (1 + r1), roughly 2-3 per component here.
    The independent unit is the mast.
    """
    rs = []
    for d in series_list:
        d = np.asarray(d, float)
        if len(d) >= 3:
            rs.append(np.corrcoef(d[:-1], d[1:])[0, 1])
    return float(np.mean(rs)) if rs else float("nan")


def block_bootstrap_loa(series_list, n_boot=4000, seed=1):
    """95% CI on the Bland-Altman limit of agreement, resampling MASTS, not heights.

    Because heights within a mast are autocorrelated, resampling individual heights
    would fabricate precision. Whole masts are the exchangeable unit, so the bootstrap
    draws masts with replacement and recomputes 1.96*SD of the pooled differences.

    Returns (loa, ci_low, ci_high, r1, n_effective) in the units of the input.
    """
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.asarray(d, float) for d in series_list])
    n = len(x)
    r1 = lag1_autocorrelation(series_list)
    neff = n * (1 - r1) / (1 + r1) if np.isfinite(r1) else float("nan")
    loa = 1.96 * x.std()
    boots = np.empty(n_boot)
    k = len(series_list)
    for b in range(n_boot):
        pick = [series_list[i] for i in rng.integers(0, k, k)]
        boots[b] = 1.96 * np.concatenate(pick).std()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(loa), float(lo), float(hi), float(r1), float(neff)
