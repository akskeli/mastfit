#!/usr/bin/env python3
"""Significance test of the total bend, axis refinement, smoother response and ablations.

  A  IRLS axis step: what it optimizes, how the single median radius weights the
     points along a tapered mast, and comparison with a robust cone fit (the
     M-estimator of the cylinder-surface objective) and with taper-aware weights.
  B  Null distribution of the reported total bend: the full pipeline on straight
     synthetic masts under the assumed error model; false-positive rate of the
     floor-based rule; p-values and classification of the measured bends.
  C  Response of the Gaussian smoother: amplitude transfer of sinusoidal bends of
     known wavelength (theory and full pipeline); wavelength content of the
     measured profiles.
  D  Ablations under the final configuration (100 slices, L = 3 m).

Writes results/significance_<sections>.json. Usage:
    python scripts/15_significance_and_ablations.py --sections ACD
    python scripts/15_significance_and_ablations.py --sections B --null 200
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from _common import base_parser, ensure_dirs, load_drone  # noqa: E402
from mastfit.geometry import (irls_axis, irls_axis_tapered, irls_weights,  # noqa: E402
                              pca_axis, robust_cone_fit)
from mastfit.io import MAST_NAMES  # noqa: E402
from mastfit.pipeline import fit_mast  # noqa: E402
from mastfit.validate import (compare_to_truth, detrend, make_phantom,  # noqa: E402
                              matched_phantom, phantom_geometry, recovered_profile)

NAMES = [MAST_NAMES[i][1] for i in range(5)]
BENDS = [159.6, 89.4, 45.4, 36.0, 70.2]          # Table 1 total bends (mm)
FLOOR_P90 = [40.1, 29.0, 31.7, 33.2, 34.7]       # Table 3, 90th percentile of the floor (mm)
FLOOR_MEAN = [27.1, 20.1, 21.1, 21.0, 20.0]      # Table 3, mean floor (mm)
OUT = {}


def put(sec, key, val):
    OUT.setdefault(sec, {})[key] = val
    print(f"  [{sec}] {key}: {val}", flush=True)


def ang(a, b):
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


# ---------------------------------------------------------------------------- A
def section_a(drone, seeds):
    print("\nA. IRLS AXIS STEP")
    rows = {}
    for i, P in enumerate(drone):
        c0, d0, _ = pca_axis(P)
        c1, d1 = irls_axis(P, c0, d0)
        c2, d2, a, b = robust_cone_fit(P, c1, d1)
        c3, d3 = irls_axis_tapered(P, c0, d0)
        w, t = irls_weights(P, c1, d1)
        wl, _ = irls_weights(P, c1, d1, radius_model="linear")
        q = np.quantile(t, [0.1, 0.9])
        lo, hi = t < q[0], t > q[1]
        mid = ~lo & ~hi
        rows[NAMES[i]] = {
            "cone_vs_irls_deg": round(ang(d1, d2), 3),
            "tapered_vs_irls_deg": round(ang(d1, d3), 3),
            "cone_taper_mm_per_m": round(-b * 1000, 2),
            "bend_irls_cone_tapered_mm": [round(fit_mast(P).max_bend_mm, 1),
                                          round(fit_mast(P, axis=(c2, d2)).max_bend_mm, 1),
                                          round(fit_mast(P, axis=(c3, d3)).max_bend_mm, 1)],
            "mean_weight_bottom_mid_top_single_radius": [round(float(w[lo].mean()), 2),
                                                         round(float(w[mid].mean()), 2),
                                                         round(float(w[hi].mean()), 2)],
            "mean_weight_bottom_mid_top_tapered": [round(float(wl[lo].mean()), 2),
                                                   round(float(wl[mid].mean()), 2),
                                                   round(float(wl[hi].mean()), 2)],
            "zero_weight_fraction_pct": round(float((w == 0).mean() * 100), 2),
        }
        put("A", NAMES[i], rows[NAMES[i]])
    # accuracy on matched phantoms (correlated error), same seeds for all axes
    acc = {"irls": [], "cone": [], "tapered": []}
    for i in range(5):
        g = phantom_geometry(i)
        for sd in range(seeds):
            P, zt, bt = matched_phantom(seed=100 * i + sd, correlated=True, **g)
            c0, d0, _ = pca_axis(P)
            c1, d1 = irls_axis(P, c0, d0)
            c2, d2, _, _ = robust_cone_fit(P, c1, d1)
            c3, d3 = irls_axis_tapered(P, c0, d0)
            acc["irls"].append(compare_to_truth(P, zt, bt)[3])
            acc["cone"].append(compare_to_truth(P, zt, bt, axis=(c2, d2))[3])
            acc["tapered"].append(compare_to_truth(P, zt, bt, axis=(c3, d3))[3])
    put("A", f"phantom bending-profile RMSE, mean over {5 * seeds} [mm]",
        {k: round(float(np.mean(v)), 1) for k, v in acc.items()})
    put("A", "phantom RMSE difference cone - irls, max |.| [mm]",
        round(float(np.max(np.abs(np.subtract(acc["cone"], acc["irls"])))), 1))


# ---------------------------------------------------------------------------- B
def null_bends(i, n, field_kw, seed0):
    g = phantom_geometry(i)
    out = []
    for sd in range(n):
        P, _, _ = matched_phantom(seed=seed0 + 1000 * i + sd, correlated=True, amplitude_m=0.0,
                                  **g, **field_kw)
        out.append(fit_mast(P).max_bend_mm)
    return np.asarray(out)


def section_b(n_null, n_sens):
    print("\nB. NULL DISTRIBUTION OF THE TOTAL BEND (straight masts)")
    res = {}
    allnull = {}
    for i in range(5):
        b0 = null_bends(i, n_null, {}, seed0=50000)
        allnull[NAMES[i]] = b0
        p = (1 + np.sum(b0 >= BENDS[i])) / (n_null + 1)
        res[NAMES[i]] = {
            "null_mean": round(float(b0.mean()), 1), "null_median": round(float(np.median(b0)), 1),
            "null_p95": round(float(np.percentile(b0, 95)), 1),
            "null_p99": round(float(np.percentile(b0, 99)), 1),
            "fpr_rule_2xP90_pct": round(float(np.mean(b0 > 2 * FLOOR_P90[i]) * 100), 1),
            "fpr_rule_2xMean_pct": round(float(np.mean(b0 > 2 * FLOOR_MEAN[i]) * 100), 1),
            "rate_above_P90_pct": round(float(np.mean(b0 > FLOOR_P90[i]) * 100), 1),
            "measured_mm": BENDS[i], "p_value": round(float(p), 4),
            "significant_at_5pct": bool(p < 0.05),
            "significant_at_1pct_bonferroni": bool(p < 0.01),
        }
        put("B", NAMES[i], res[NAMES[i]])
    pooled2 = np.mean([np.mean(allnull[NAMES[i]] > 2 * FLOOR_P90[i]) for i in range(5)]) * 100
    pooled1 = np.mean([np.mean(allnull[NAMES[i]] > FLOOR_P90[i]) for i in range(5)]) * 100
    put("B", "pooled false-positive rate of the rule bend > 2 x p90 floor [%]", round(float(pooled2), 1))
    put("B", "pooled rate of bend > p90 floor on straight masts [%]",
        round(float(pooled1), 1))
    OUT["B_null_samples"] = {k: [round(float(x), 2) for x in v] for k, v in allnull.items()}
    sens = {}
    for label, kw in (("GP l = 2.5 m, 45 mm", {"corr_len_m": 2.5}),
                      ("default field, 30 mm", {"noise_m": 0.030})):
        row = {}
        for i in range(5):
            b0 = null_bends(i, n_sens, kw, seed0=70000)
            p = (1 + np.sum(b0 >= BENDS[i])) / (n_sens + 1)
            row[NAMES[i]] = {"null_p95": round(float(np.percentile(b0, 95)), 1), "p_value": round(float(p), 4)}
        sens[label] = row
        put("B", f"sensitivity: {label}", row)
    OUT["B_sensitivity"] = sens


# ---------------------------------------------------------------------------- C
def amplitude_of(z, y, lam, lo, hi):
    m = (z >= lo) & (z <= hi)
    zz = z[m]
    A = np.column_stack([np.sin(2 * np.pi * zz / lam), np.cos(2 * np.pi * zz / lam),
                         np.ones_like(zz), zz])
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    return float(np.hypot(coef[0], coef[1]))


def section_c(drone):
    print("\nC. RESPONSE OF THE GAUSSIAN SMOOTHER")
    lams = [4, 6, 8, 10, 12, 16, 20, 30, 50]
    out = {}
    for L in (2.0, 3.0, 5.0):
        row = {}
        for lam in lams:
            ratios = []
            for sd in range(3):
                P, zt, bt = make_phantom(height_m=50.0, mode="wave", wavelength_m=lam, amplitude_m=0.05,
                                         noise_m=0.005, rake_deg=2.0, seed=sd)
                h, b = recovered_profile(P, bandwidth_m=L)
                lo, hi = h.min() + 2 * L, h.max() - 2 * L
                a_rec = amplitude_of(h, b, lam, lo, hi)
                a_tru = amplitude_of(zt, bt, lam, lo, hi)
                ratios.append(a_rec / a_tru)
            theory = float(np.exp(-2 * np.pi ** 2 * L ** 2 / lam ** 2))
            row[lam] = {"pipeline": round(float(np.mean(ratios)), 3), "theory": round(theory, 3)}
        out[L] = row
        put("C", f"amplitude transfer, L = {L:g} m", row)
    for L in (2.0, 3.0, 5.0):
        put("C", f"wavelength at 50% transfer (theory), L = {L:g} m [m]",
            round(float(np.pi * L * np.sqrt(2 / np.log(2))), 1))
    # wavelength content of the measured profiles: power at wavelengths >= 16 m,
    # from the lightly smoothed (L = 1 m) total-bend profile
    frac = []
    for P in drone:
        r = fit_mast(P, bandwidth_m=1.0)
        z = r.height
        b = np.hypot(r.bend_transverse_mm, r.bend_longitudinal_mm)
        bx = detrend(z, r.bend_transverse_mm)
        by = detrend(z, r.bend_longitudinal_mm)
        n = len(z)
        dz = float(np.mean(np.diff(z)))
        f = np.fft.rfftfreq(n, dz)
        pw = np.abs(np.fft.rfft(bx)) ** 2 + np.abs(np.fft.rfft(by)) ** 2
        lam = np.where(f > 0, 1 / np.maximum(f, 1e-12), np.inf)
        frac.append(float(pw[lam >= 16].sum() / pw[1:].sum() * 100))
    put("C", "measured profiles (L = 1 m): share of bending power at wavelengths >= 16 m [%]",
        [round(x, 1) for x in frac])
    OUT["C_transfer"] = out


# ---------------------------------------------------------------------------- D
VARIANTS = (("full pipeline", {}),
            ("no base-flare removal", {"flare_removal": False}),
            ("no rigging-band suppression", {"band_suppression": False}),
            ("no covariance weighting", {"variance_weighting": False}),
            ("no axis refinement (PCA seed)", {"irls_iter": 0}),
            ("no flare removal, no covariance weighting", {"flare_removal": False, "variance_weighting": False}),
            ("no slice quality control (flare and band)", {"flare_removal": False, "band_suppression": False}),
            ("no band suppression, no covariance weighting", {"band_suppression": False, "variance_weighting": False}),
            ("no quality control, no covariance weighting",
             {"flare_removal": False, "band_suppression": False, "variance_weighting": False}))


def section_d(drone, seeds):
    print("\nD. ABLATIONS (100 slices, L = 3 m)")
    real = {}
    for label, kw in VARIANTS:
        real[label] = [round(fit_mast(P, **kw).max_bend_mm, 1) for P in drone]
        put("D", f"real data, total bend [mm]: {label}", real[label])
    syn = {}
    for err, corr in (("correlated", True), ("white", False)):
        for label, kw in VARIANTS:
            e = []
            for i in range(5):
                g = phantom_geometry(i)
                for sd in range(seeds):
                    P, zt, bt = matched_phantom(seed=100 * i + sd, correlated=corr, **g)
                    e.append(compare_to_truth(P, zt, bt, **kw)[3])
            syn[(err, label)] = (round(float(np.mean(e)), 1), round(float(np.percentile(e, 90)), 1))
            put("D", f"phantom RMSE mean / p90 [mm], {err} error: {label}", syn[(err, label)])
    OUT["D_phantom"] = {f"{a} | {b}": v for (a, b), v in syn.items()}


def main():
    p = base_parser(__doc__)
    p.add_argument("--sections", default="ABCD")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--null", type=int, default=200)
    p.add_argument("--sens", type=int, default=100)
    args = p.parse_args()
    ensure_dirs("results")
    drone = load_drone(args)
    if "A" in args.sections:
        section_a(drone, max(4, args.seeds // 2))
    if "B" in args.sections:
        section_b(args.null, args.sens)
    if "C" in args.sections:
        section_c(drone)
    if "D" in args.sections:
        section_d(drone, args.seeds)
    out = f"results/significance_{args.sections}.json"
    with open(out, "w") as f:
        json.dump(OUT, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
