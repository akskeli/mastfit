"""Tests for the validation tools: cross-validation pairing, positional test, unsmoothed
baseline, slice-quality breakdown and ovality study."""
import numpy as np

from mastfit.baseline import naive_pipeline
from mastfit.crossval import common_span_profiles, cross_validate
from mastfit.doming import positional_trend, shared_loadings
from mastfit.pipeline import fit_mast
from mastfit.quality import slice_quality_breakdown, slice_quality_mask
from mastfit.validate import detrend, make_phantom, ovality_study


def test_positional_trend_exact_p_for_a_perfect_ramp():
    g = np.linspace(0, 1, 30)
    base = np.sin(np.pi * g)
    M = np.array([(1 + 0.5 * k) * base for k in range(5)])   # loading grows with position
    out = positional_trend(M)
    assert out["n_perm"] == 120
    assert out["r"] > 0.99
    assert abs(out["p"] - 2 / 120) < 1e-12                    # the ramp and its reverse


def test_shared_loadings_average_to_one():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(5, 40)) + np.sin(np.linspace(0, 3, 40))
    assert abs(shared_loadings(M).mean() - 1.0) < 1e-12


def test_common_span_profiles_share_heights_and_have_zero_mean():
    P1, _, _ = make_phantom(mode="C", amplitude_m=0.12, seed=1)
    P2, _, _ = make_phantom(mode="C", amplitude_m=0.12, seed=2)
    P2 = P2[P2[:, 2] > 5.0]                                   # second sensor misses the base
    ra, rb = fit_mast(P1), fit_mast(P2)
    z, a, b = common_span_profiles(ra, rb, "bend_transverse_mm")
    assert z[0] >= rb.world_z().min() - 1e-9 and z[-1] <= ra.world_z().max() + 1e-9
    assert abs(a.mean()) < 1e-9 and abs(b.mean()) < 1e-9


def test_cross_validate_identical_clouds_agree_exactly():
    P, _, _ = make_phantom(mode="S", amplitude_m=0.1, seed=4)
    r = fit_mast(P)
    cv = cross_validate([r] * 2, [r] * 2, [0, 1], n_boot=50)
    assert cv["transverse"]["loa_mm"] < 1e-9
    assert cv["longitudinal"]["rms_diff_mm"] < 1e-9


def test_unsmoothed_baseline_shows_the_flare_mechanism():
    P, zt, bt = make_phantom(mode="C", amplitude_m=0.15, noise_m=0.02, clutter_frac=0.03,
                             base_flare=True, seed=3)
    off = naive_pipeline(P, quality_control=False)
    on, h, b = naive_pipeline(P, quality_control=True, return_profile=True)
    truth = np.max(np.abs(detrend(zt, bt)))
    assert off > on
    assert abs(on - truth) < abs(off - truth)
    assert len(h) == len(b) and h.min() >= 0


def test_quality_breakdown_matches_combined_mask():
    rng = np.random.default_rng(0)
    h = np.linspace(0, 50, 100)
    r = 0.5 - 0.004 * h + rng.normal(0, 0.003, 100)
    r[:3] += 0.3                                              # flare
    res = 20 + rng.normal(0, 2, 100)
    res[50] = 90                                              # a band
    ova = 10 + rng.normal(0, 1, 100)
    qc = slice_quality_breakdown(h, r, res, ova)
    assert qc["flare"][:3].all() and qc["band"][50]
    assert np.array_equal(qc["keep"], slice_quality_mask(h, r, res, ova))


def test_ovality_study_null_is_small_for_a_round_phantom():
    P, _, _ = make_phantom(mode="C", amplitude_m=0.05, noise_m=0.01, seed=5)
    o = ovality_study(P)
    assert np.median(o["null"]) < 5.0
    assert len(o["ovality"]) == len(o["null"]) == len(o["masked"]) > 50
