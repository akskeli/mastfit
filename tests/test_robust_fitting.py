"""Tests for the biweight constant exposed in fit_mast, the Gaussian-process
correlated-error field, and the TLS band filter on oval sections."""
import numpy as np

from mastfit.geometry import irls_axis, pca_axis
from mastfit.io import robust_clean
from mastfit.pipeline import fit_mast
from mastfit.validate import gp_correlated_field, make_phantom


def test_biweight_weights_have_the_outer_square():
    # w = (1 - u^2)^2 for |u| < 1: at u = 0.5 the weight is 0.5625, not 0.75
    u = 0.5
    assert abs((1 - u ** 2) ** 2 - 0.5625) < 1e-12


def test_irls_zero_iterations_returns_the_pca_seed():
    P, _, _ = make_phantom(mode="C", amplitude_m=0.1, seed=2)
    c0, d0, _ = pca_axis(P)
    c, d = irls_axis(P, c0, d0, n_iter=0)
    assert np.allclose(c, c0) and np.allclose(d, d0)


def test_profile_accuracy_is_insensitive_to_the_tukey_constant():
    from mastfit.validate import compare_to_truth
    P, zt, bt = make_phantom(mode="C", amplitude_m=0.15, clutter_frac=0.05, seed=4)
    e = [compare_to_truth(P, zt, bt, tukey_c=c)[3] for c in (3.0, 4.685, 9.0)]
    assert max(e) - min(e) < 3.0


def test_gp_field_has_the_requested_rms_and_correlation_length():
    rng = np.random.default_rng(0)
    z = np.linspace(0, 50, 4000)
    th = rng.uniform(0, 2 * np.pi, z.size)
    A = gp_correlated_field(z, th, 50.0, 0.045, 5.0, angular_frac=0.0, rng=rng)
    assert abs(A[:, 0].std() - 0.045) < 0.01
    # translation-only field: correlation at a 5 m lag ~ exp(-1/2)
    k = np.searchsorted(z, 5.0)
    x = A[:, 0] - A[:, 0].mean()
    r = np.dot(x[:-k], x[k:]) / np.dot(x, x)
    assert 0.2 < r < 0.95


def test_gp_field_anisotropy_ratio():
    rng = np.random.default_rng(1)
    z = np.linspace(0, 50, 4000)
    th = rng.uniform(0, 2 * np.pi, z.size)
    A = gp_correlated_field(z, th, 50.0, 0.03, 10.0, anisotropy=2.0, angular_frac=0.0, rng=rng)
    ratio = A[:, 0].std() / A[:, 1].std()
    assert 1.3 < ratio < 3.0


def test_default_phantom_unchanged_by_new_options():
    P1, _, _ = make_phantom(correlated=True, noise_m=0.02, seed=1)
    P2, _, _ = make_phantom(correlated=True, noise_m=0.02, seed=1, corr_len_m=None)
    assert np.array_equal(P1, P2)


def test_band_filter_keeps_oval_sections():
    rg = np.random.default_rng(5)
    n = 60000
    z = rg.uniform(0, 30, n)
    th = rg.uniform(0, 2 * np.pi, n)
    a = 0.3 / (2 - 0.3)                       # 30% ovality
    r = 0.45 * (1 + a * np.cos(2 * th)) + rg.normal(0, 0.003, n)
    P = np.column_stack([r * np.cos(th), r * np.sin(th), z])
    kept = len(robust_clean(P)) / n
    assert kept > 0.99
