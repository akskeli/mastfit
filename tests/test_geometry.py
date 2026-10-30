"""Unit tests for the geometric primitives."""
import numpy as np
import pytest

from mastfit.geometry import (harmonic_ovality, irls_axis, orthonormal_basis,
                              pca_axis, robust_circle, taubin_circle)


def _ring(cx, cy, r, n=200, noise=0.0, seed=0):
    rg = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r + (rg.normal(0, noise, n) if noise else 0.0)
    return np.column_stack([cx + rr * np.cos(th), cy + rr * np.sin(th)])


def test_taubin_recovers_exact_circle():
    xy = _ring(3.0, -2.0, 0.75)
    c, r, sigma, _ = taubin_circle(xy)
    assert np.allclose(c, [3.0, -2.0], atol=1e-9)
    assert r == pytest.approx(0.75, abs=1e-9)
    assert sigma == pytest.approx(0.0, abs=1e-9)


def test_taubin_is_noise_tolerant():
    xy = _ring(0.0, 0.0, 0.5, n=400, noise=0.002, seed=1)
    c, r, _, _ = taubin_circle(xy)
    assert np.linalg.norm(c) < 0.002
    assert r == pytest.approx(0.5, abs=0.002)


def test_robust_circle_rejects_radial_outliers():
    xy = _ring(1.0, 1.0, 0.5, n=200, noise=0.001, seed=2)
    outliers = np.array([[3.0, 3.0], [2.5, 1.0], [1.0, 4.0]])
    contaminated = np.vstack([xy, outliers])
    c, r, _, _, inl = robust_circle(contaminated)
    assert np.allclose(c, [1.0, 1.0], atol=0.01)
    assert r == pytest.approx(0.5, abs=0.01)
    assert not inl[-3:].any(), "gross outliers should be trimmed"


def test_orthonormal_basis_is_orthonormal():
    B, d = orthonormal_basis(np.array([0.3, -0.5, 1.0]))
    assert np.allclose(B @ B.T, np.eye(2), atol=1e-12)
    assert np.allclose(B @ d, 0.0, atol=1e-12)
    assert np.linalg.norm(d) == pytest.approx(1.0)


def test_pca_axis_recovers_known_direction():
    rg = np.random.default_rng(3)
    z = rg.uniform(0, 50, 4000)
    th = rg.uniform(0, 2 * np.pi, 4000)
    P = np.column_stack([0.5 * np.cos(th), 0.5 * np.sin(th), z])
    _, d, eig = pca_axis(P)
    assert abs(d[2]) > 0.999, "axis of a slender tube must be near-vertical"
    assert eig[0] / eig[1] > 100, "slender geometry must be well conditioned"


def test_irls_axis_is_robust_to_a_side_cluster():
    rg = np.random.default_rng(4)
    z = rg.uniform(0, 50, 6000)
    th = rg.uniform(0, 2 * np.pi, 6000)
    P = np.column_stack([0.5 * np.cos(th), 0.5 * np.sin(th), z])
    junk = np.column_stack([rg.uniform(1.5, 3.0, 400), rg.uniform(-.3, .3, 400),
                            rg.uniform(0, 50, 400)])
    c0, d0, _ = pca_axis(np.vstack([P, junk]))
    _, d = irls_axis(np.vstack([P, junk]), c0, d0)
    assert abs(d[2]) > 0.99


def test_ovality_zero_for_round_ring():
    assert harmonic_ovality(_ring(0, 0, 1.0), np.zeros(2)) == pytest.approx(0.0, abs=1e-6)


def test_ovality_detects_a_known_ellipse():
    th = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    a, b = 1.0, 0.9
    xy = np.column_stack([a * np.cos(th), b * np.sin(th)])
    e = harmonic_ovality(xy, np.zeros(2))
    assert e == pytest.approx((1 - b / a) * 100, rel=0.15)
