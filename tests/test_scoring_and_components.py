"""Tests for the physical-height scoring of phantoms, the component switches of
fit_mast, the robust cone fit, the response of the smoother and the section model."""
import numpy as np

from mastfit.geometry import irls_axis, pca_axis, robust_cone_fit
from mastfit.pipeline import fit_mast
from mastfit.validate import compare_to_truth, make_phantom, recovered_profile


def test_recovered_profile_uses_physical_heights():
    # with a base flare, the lowest retained slice lies above the phantom base; the
    # profile must be returned on the phantom's own height axis, not from zero
    P, zt, bt = make_phantom(mode="C", amplitude_m=0.15, noise_m=0.005, base_flare=True, seed=1)
    h, b = recovered_profile(P)
    assert h[0] > 0.5 and np.all(np.diff(h) > 0)
    res = fit_mast(P)
    assert abs((h[-1] - h[0]) - (res.height[-1] - res.height[0]) * abs(res.axis_direction[2])) < 0.05


def test_scoring_is_not_offset_by_removed_base_slices():
    P, zt, bt = make_phantom(mode="C", amplitude_m=0.15, noise_m=0.005, base_flare=True, seed=1)
    g, T, R, rmse, _ = compare_to_truth(P, zt, bt)
    assert g[0] > 0.5            # compared only where the profile exists
    assert rmse < 10.0           # an offset of the removed base (~3 m) would give far more


def test_component_switches_run_and_matter():
    P, _, _ = make_phantom(mode="C", amplitude_m=0.10, noise_m=0.02, base_flare=True,
                           clutter_frac=0.05, seed=3)
    full = fit_mast(P).max_bend_mm
    for kw in ({"flare_removal": False}, {"band_suppression": False},
               {"variance_weighting": False}, {"irls_iter": 0}):
        assert np.isfinite(fit_mast(P, **kw).max_bend_mm)
    assert fit_mast(P, variance_weighting=False).max_bend_mm != full


def test_robust_cone_axis_agrees_with_the_irls_axis():
    P, _, _ = make_phantom(mode="C", amplitude_m=0.05, noise_m=0.01, clutter_frac=0.03, seed=5)
    c0, d0 = pca_axis(P)[:2]
    c1, d1 = irls_axis(P, c0, d0)
    c2, d2, a, b = robust_cone_fit(P, c1, d1)
    ang = np.degrees(np.arccos(min(1.0, abs(float(d1 @ d2)))))
    assert ang < 0.5 and np.isfinite(a) and np.isfinite(b)


def test_smoother_transfer_follows_the_gaussian_response():
    # a 16 m wave is passed at about half its amplitude for L = 3 m
    lam, L = 16.0, 3.0
    P, zt, bt = make_phantom(height_m=50.0, mode="wave", wavelength_m=lam, amplitude_m=0.05,
                             noise_m=0.005, rake_deg=2.0, seed=0)
    h, b = recovered_profile(P, bandwidth_m=L)
    m = (h > h.min() + 2 * L) & (h < h.max() - 2 * L)
    A = np.column_stack([np.sin(2 * np.pi * h[m] / lam), np.cos(2 * np.pi * h[m] / lam),
                         np.ones(m.sum()), h[m]])
    coef = np.linalg.lstsq(A, b[m], rcond=None)[0]
    ratio = np.hypot(coef[0], coef[1]) / 50.0
    assert abs(ratio - np.exp(-2 * np.pi ** 2 * L ** 2 / lam ** 2)) < 0.1


def test_ellipse_fit_recovers_centre_on_full_and_partial_rings():
    from mastfit.geometry import fit_ellipse
    rg = np.random.default_rng(0)
    t = rg.uniform(0, 2 * np.pi, 600)
    a, b, ang, c = 0.5, 0.4, 0.6, np.array([0.1, -0.2])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    xy = (R @ np.vstack([a * np.cos(t), b * np.sin(t)])).T + c + rg.normal(0, 0.002, (600, 2))
    for m in (np.ones_like(t, bool), t < np.radians(300)):
        cen, axes, _ = fit_ellipse(xy[m])
        assert np.allclose(cen, c, atol=0.005) and np.allclose(axes, [a, b], atol=0.01)


def test_section_model_and_world_centreline():
    P, zt, bt = make_phantom(mode="C", amplitude_m=0.10, noise_m=0.01, seed=2)
    rc, re_ = fit_mast(P), fit_mast(P, section_model="ellipse")
    assert abs(rc.max_bend_mm - re_.max_bend_mm) < 5.0
    cw = rc.centre_world
    assert cw.shape == (len(rc.height), 3)
    # the world centreline follows the phantom axis (x = bend, y = rake * z) to within a few mm
    xt = np.interp(cw[:, 2], zt, bt / 1000.0)
    yt = np.tan(np.radians(2.0)) * cw[:, 2]
    assert np.sqrt(np.mean((cw[:, 1] - yt) ** 2)) < 0.005
    assert np.sqrt(np.mean((cw[:, 0] - xt) ** 2)) < 0.015
