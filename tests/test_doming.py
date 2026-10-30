"""Tests for the five-mast shared-curvature decomposition."""
import numpy as np

from mastfit.doming import shared_fraction


def test_identical_profiles_are_fully_shared():
    p = np.sin(np.linspace(0, np.pi, 40))
    M = np.tile(p, (5, 1))
    frac, shared = shared_fraction(M)
    assert frac > 0.99
    assert np.allclose(shared, p)


def test_independent_profiles_are_mostly_unshared():
    rg = np.random.default_rng(0)
    M = rg.normal(size=(5, 40))
    frac, _ = shared_fraction(M)
    assert frac < 0.5


def test_injected_common_mode_is_detected():
    rg = np.random.default_rng(1)
    z = np.linspace(0, 1, 40)
    common = 3.0 * np.sin(np.pi * z)
    M = np.array([common + rg.normal(0, 0.3, 40) for _ in range(5)])
    frac, shared = shared_fraction(M)
    assert frac > 0.8
    assert np.corrcoef(shared, common)[0, 1] > 0.95
