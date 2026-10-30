"""Pipeline tests, including the determinism guarantee."""
import numpy as np
import pytest

from mastfit.pipeline import fit_mast
from mastfit.validate import compare_to_truth, make_phantom


@pytest.fixture(scope="module")
def phantom():
    return make_phantom(mode="C", amplitude_m=0.15, noise_m=0.01,
                        clutter_frac=0.02, base_flare=True, seed=7)


def test_pipeline_is_bit_identical_across_runs(phantom):
    """The central claim of the method: no randomness anywhere."""
    P, _, _ = phantom
    a = fit_mast(P, n_slices=60)
    b = fit_mast(P, n_slices=60)
    assert np.array_equal(a.bend_total_mm, b.bend_total_mm)
    assert a.rake_deg == b.rake_deg


def test_recovers_known_bend(phantom):
    P, zt, bt = phantom
    _, _, _, rmse, peak_err = compare_to_truth(P, zt, bt)
    assert rmse < 25, f"RMSE {rmse:.1f} mm too large on a clean phantom"
    assert peak_err < 30


def test_recovers_known_rake():
    P, _, _ = make_phantom(mode="C", amplitude_m=0.05, rake_deg=2.0, noise_m=0.01, seed=8)
    r = fit_mast(P, n_slices=60)
    assert r.rake_deg == pytest.approx(2.0, abs=0.5)


def test_straight_mast_has_small_bend():
    P, _, _ = make_phantom(mode="C", amplitude_m=0.0, noise_m=0.01, seed=9)
    r = fit_mast(P, n_slices=60)
    assert r.max_bend_mm < 30, "a straight phantom must not manufacture bending"


def test_quality_control_suppresses_flare_inflation():
    P, _, _ = make_phantom(mode="C", amplitude_m=0.05, noise_m=0.01,
                           base_flare=True, clutter_frac=0.03, seed=11)
    on = fit_mast(P, n_slices=60, apply_quality_control=True).max_bend_mm
    off = fit_mast(P, n_slices=60, apply_quality_control=False).max_bend_mm
    assert on <= off + 1e-9


def test_bend_is_insensitive_to_slice_count():
    """The fixed bandwidth must decouple the estimate from the slice count."""
    P, _, _ = make_phantom(mode="C", amplitude_m=0.15, noise_m=0.02, seed=12)
    vals = [fit_mast(P, n_slices=n, bandwidth_m=3.0).max_bend_mm for n in (40, 100, 200)]
    spread = max(vals) - min(vals)
    assert spread < 0.35 * np.mean(vals), f"delta-instability: {vals}"


def test_too_sparse_cloud_raises():
    rg = np.random.default_rng(0)
    P = rg.normal(size=(50, 3))
    with pytest.raises(ValueError, match="too sparse"):
        fit_mast(P, name="tiny")


def test_correlated_error_is_worse_than_white():
    """At equal magnitude, correlated error costs more than white noise (modestly); with
    the matched amplitudes it dominates."""
    from mastfit.validate import matched_phantom
    white, corr = [], []
    for sd in range(3):
        Pw, zt, bt = make_phantom(mode="C", amplitude_m=0.15, noise_m=0.02,
                                  correlated=False, seed=sd)
        Pc, _, _ = make_phantom(mode="C", amplitude_m=0.15, noise_m=0.02,
                                correlated=True, seed=sd)
        white.append(compare_to_truth(Pw, zt, bt)[3])
        corr.append(compare_to_truth(Pc, zt, bt)[3])
    assert np.mean(corr) > np.mean(white)
    mw = [compare_to_truth(*matched_phantom(seed=sd, correlated=False))[3] for sd in range(3)]
    mc = [compare_to_truth(*matched_phantom(seed=sd, correlated=True))[3] for sd in range(3)]
    assert np.mean(mc) > 2.0 * np.mean(mw)


def test_matched_phantom_reproduces_real_slice_statistics():
    """The accuracy transfer must be checkable, not asserted."""
    from mastfit.validate import REAL_CLOUD_STATS, matched_phantom, slice_statistics

    st = slice_statistics(matched_phantom(seed=1)[0])
    sigma = np.median(st["sigma"])
    trimmed = np.median(st["trimmed"])
    target_sigma = REAL_CLOUD_STATS["radial_residual_sigma_median_mm"]
    target_trim = REAL_CLOUD_STATS["trimmed_fraction_median_pct"]
    assert abs(sigma - target_sigma) < 0.25 * target_sigma, (
        f"matched phantom sigma {sigma:.1f} mm does not match the real {target_sigma} mm"
    )
    assert abs(trimmed - target_trim) < 3.0


def test_accuracy_is_robust_to_phantom_realism():
    """Matching the phantom to the data raises the floor (its correlated field is larger)
    but keeps it bounded."""
    from mastfit.validate import compare_to_truth, make_phantom, matched_phantom

    mild = [compare_to_truth(*make_phantom(mode="C", amplitude_m=0.15, noise_m=0.02,
                                           clutter_frac=0.03, base_flare=True,
                                           correlated=True, seed=sd))[3] for sd in range(3)]
    real = [compare_to_truth(*matched_phantom(seed=sd))[3] for sd in range(3)]
    assert np.mean(real) > np.mean(mild)
    assert np.mean(real) < 40, "transferred accuracy must remain within the quoted floor"


def test_phantom_geometry_tracks_each_real_mast():
    """A phantom must reproduce the mast it stands in for, not a generic tube.

    Matching noise statistics while leaving every phantom 50 m tall would make the
    synthetic row five copies of the same object, and would quietly undermine any
    per-mast accuracy claim derived from it.
    """
    from mastfit.validate import REAL_MAST_GEOMETRY, matched_phantom, phantom_geometry

    heights = []
    for i, (_, h_true, _, _, _) in enumerate(REAL_MAST_GEOMETRY):
        P, _, _ = matched_phantom(seed=i, **phantom_geometry(i))
        h = P[:, 2].max() - P[:, 2].min()
        assert abs(h - h_true) < 1.0, f"phantom {i} height {h:.1f} != real {h_true:.1f}"
        heights.append(h)
    # the Jiggermast is markedly the shortest; the phantoms must not be uniform
    assert max(heights) - min(heights) > 8.0, "phantoms are all the same size"


def test_lag1_autocorrelation_of_smooth_vs_white_series():
    """The autocorrelation diagnostic must separate smooth from independent series."""
    import numpy as np

    from mastfit.crossval import lag1_autocorrelation

    rng = np.random.default_rng(0)
    smooth = [np.cumsum(rng.normal(size=200)) for _ in range(4)]   # highly correlated
    white = [rng.normal(size=200) for _ in range(4)]               # independent
    assert lag1_autocorrelation(smooth) > 0.9
    assert abs(lag1_autocorrelation(white)) < 0.2


def test_block_bootstrap_loa_is_deterministic_and_brackets_point_estimate():
    """The mast-block bootstrap must be seeded (repo promise: bit-identical runs)
    and its CI must contain the point LoA."""
    import numpy as np

    from mastfit.crossval import block_bootstrap_loa

    rng = np.random.default_rng(3)
    series = [rng.normal(scale=10, size=60) + rng.normal(scale=5) for _ in range(4)]
    a = block_bootstrap_loa(series, n_boot=500, seed=7)
    b = block_bootstrap_loa(series, n_boot=500, seed=7)
    assert a == b, "bootstrap must be deterministic under a fixed seed"
    loa, lo, hi, r1, neff = a
    assert 0 < lo <= hi, "CI must be ordered and positive"
    assert loa > 0 and np.isfinite(neff) and neff > 0
    # a strongly autocorrelated series must yield a small effective sample size
    smooth = [np.cumsum(np.random.default_rng(9).normal(size=60)) for _ in range(4)]
    *_, neff_smooth = block_bootstrap_loa(smooth, n_boot=100, seed=1)
    assert neff_smooth < 4 * 60 / 5, "ESS must collapse for smooth series"
