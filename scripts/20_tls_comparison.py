#!/usr/bin/env python3
"""UAV-TLS comparison: per mast, without straight-line re-referencing, and the TLS-side
uncertainty (Section 4.15, Table 8).

A  The two centrelines are compared at the same ship-frame heights in four ways:
     (1) absolute positions after co-registration only,
     (2) after removing the common TLS frame tilt (one rotation fitted to all masts),
     (3) after (2) and a constant offset per mast,
     (4) after removing a straight line per mast (the bending estimand; Section 4.8).
   RMS of the differences per mast and component.
B  Per-mast statistics of (4): RMS, 95% limits, correlation, maximum difference.
C  TLS-side uncertainty: per-slice centre uncertainty and angular coverage of the TLS
   slices, and the bending-profile RMSE of TLS-like phantoms (3 mm noise, the measured
   point density and one-sided angular coverage of each mast, rigging returns, the same
   band filter and pipeline). The expected UAV-TLS difference if the errors are
   independent is sqrt(eps_UAV^2 + eps_TLS^2).

Writes results/tls_comparison.json. Usage: python scripts/20_tls_comparison.py --seeds 5
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from _common import base_parser, load_drone, load_laser  # noqa: E402
from mastfit.crossval import common_span_profiles  # noqa: E402
from mastfit.geometry import robust_circle  # noqa: E402
from mastfit.io import MAST_NAMES, coregister, robust_clean  # noqa: E402
from mastfit.pipeline import fit_mast  # noqa: E402
from mastfit.validate import REAL_MAST_GEOMETRY, compare_to_truth  # noqa: E402

NAMES = [MAST_NAMES[i][1] for i in range(5)]
MIN_LASER_POINTS = 5000
FLOOR_MEAN = [27.1, 20.1, 21.1, 21.0, 20.0]      # Table 3 (mm)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def tls_coverage(P, n_slices=100):
    """Median and 10th percentile of the occupied share of 10-degree sectors per slice."""
    c0 = P.mean(axis=0)
    t = P[:, 2]
    edges = np.linspace(t.min(), t.max(), n_slices + 1)
    cov = []
    for k in range(n_slices):
        m = (t >= edges[k]) & (t < edges[k + 1])
        if m.sum() < 40:
            continue
        s = P[m, :2]
        cen = robust_circle(s)[0]
        ang = np.degrees(np.arctan2(s[:, 1] - cen[1], s[:, 0] - cen[0]))
        cov.append(len(set(((ang + 180) // 10).astype(int))) / 36.0)
    cov = np.asarray(cov)
    return float(np.median(cov)), float(np.percentile(cov, 10))


def tls_phantom(i, seed, coverage, n_per_m, amp=0.10, noise=0.003, rigging=0.05, ovality=0.0):
    name, H, r0, taper, _ = REAL_MAST_GEOMETRY[i]
    rg = np.random.default_rng(seed)
    n = int(H * n_per_m)
    z = rg.uniform(0, H, n)
    half = np.pi * coverage                      # visible arc, one-sided (scanner side)
    th = rg.uniform(-half, half, n)
    r = r0 - taper * z
    if ovality > 0:                              # orientation turning once over the height
        q = ovality / (2 - ovality)
        r = r * (1 + q * np.cos(2 * (th - 2 * np.pi * z / H)))
    bend = amp * np.sin(np.pi * z / H)
    rake = np.tan(np.radians(2.0))
    P = np.column_stack([bend + r * np.cos(th), rake * z + r * np.sin(th), z]) + rg.normal(0, noise, (n, 3))
    nr = int(rigging * n)                         # rigging returns near the surface
    zr = rg.uniform(0, H, nr)
    tr = rg.uniform(-half, half, nr)
    rr = (r0 - taper * zr) + rg.uniform(0.05, 0.6, nr)
    P = np.vstack([P, np.column_stack([amp * np.sin(np.pi * zr / H) + rr * np.cos(tr),
                                       rake * zr + rr * np.sin(tr), zr])])
    zt = np.linspace(0, H, 400)
    return P, zt, amp * np.sin(np.pi * zt / H) * 1000.0


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    drone, laser = load_drone(a), load_laser(a)
    usable = [i for i in range(5) if len(laser[i]) >= MIN_LASER_POINTS]
    dt, lt = coregister(drone, laser)
    rd = {i: fit_mast(dt[i]) for i in usable}
    rl = {i: fit_mast(robust_clean(lt[i])) for i in usable}
    out = {"usable": [NAMES[i] for i in usable]}

    # common TLS frame tilt (small-angle), fitted to the axis directions as in script 14
    A = np.array([rd[i].axis_direction * np.sign(rd[i].axis_direction[2]) for i in usable])
    B = np.array([rl[i].axis_direction * np.sign(rl[i].axis_direction[2]) for i in usable])
    tilt = (B[:, :2] / B[:, 2:3] - A[:, :2] / A[:, 2:3]).mean(axis=0)
    out["common_tilt_deg"] = [round(float(np.degrees(np.arctan(x))), 2) for x in tilt]
    # rotate the TLS clouds so that their vertical matches the UAV vertical (one rotation for
    # all masts), then rebuild the ship frame from the rotated clouds: this also corrects the
    # per-mast height datum, which the tilt offsets by up to ~2 m at the ends of the ship
    v = np.array([-tilt[0], -tilt[1], 1.0]); v /= np.linalg.norm(v)
    k = np.cross([0, 0, 1.0], v); sn = np.linalg.norm(k); k /= sn; cs = v[2]
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Rot = np.eye(3) + sn * Kx + (1 - cs) * Kx @ Kx
    lt_rot = [m @ Rot.T for m in lt]
    # re-centre as in mastfit.io.ship_frame (centroid of the mast medians, z = 0 at the 1st
    # percentile), keeping the axes: the clouds are already in ship-frame coordinates
    cen = np.mean([[np.median(m[:, 0]), np.median(m[:, 1])] for m in lt_rot], axis=0)
    zb = float(np.percentile(np.concatenate([m[:, 2] for m in lt_rot]), 1))
    lt2 = [m - np.array([cen[0], cen[1], zb]) for m in lt_rot]
    rl2 = {i: fit_mast(robust_clean(lt2[i])) for i in usable}
    # height shift of each mast as a whole (1st percentile of its cleaned TLS cloud)
    out["height_shift_by_rotation_m"] = {
        NAMES[i]: round(float(np.percentile(robust_clean(lt2[i])[:, 2], 1)
                              - np.percentile(robust_clean(lt[i])[:, 2], 1)), 2) for i in usable}
    print("height shift of each mast by the rotation [m]:", out["height_shift_by_rotation_m"], flush=True)

    def paired(ru, rt):
        cu, cl = ru.centre_world, rt.centre_world
        lo, hi = max(cu[:, 2].min(), cl[:, 2].min()), min(cu[:, 2].max(), cl[:, 2].max())
        z = np.linspace(lo, hi, 60)
        U = np.column_stack([np.interp(z, cu[:, 2], cu[:, k_]) for k_ in (0, 1)])
        T = np.column_stack([np.interp(z, cl[:, 2], cl[:, k_]) for k_ in (0, 1)])
        return z, (U - T) * 1000.0

    pairs1 = {i: paired(rd[i], rl[i]) for i in usable}
    pairs2 = {i: paired(rd[i], rl2[i]) for i in usable}
    # one horizontal translation for all masts after the rotation (global rigid alignment)
    shift = np.mean(np.vstack([pairs2[i][1] for i in usable]), axis=0)

    print("A. UAV-TLS centreline differences [mm RMS], transverse / longitudinal")
    var = {}
    for i in usable:
        z, d1 = pairs1[i]
        z2, d2 = pairs2[i]
        d2g = d2 - shift
        d3 = d2 - d2.mean(axis=0)
        Az = np.column_stack([np.ones_like(z2), z2])
        d4 = d2 - Az @ np.linalg.lstsq(Az, d2, rcond=None)[0]
        row = {}
        for label, dd in (("common ship frame as co-registered", d1),
                          ("global rigid alignment (one rotation, one translation)", d2g),
                          ("global alignment and per-mast offset", d3),
                          ("per-mast line removed", d4)):
            row[label] = [round(rms(dd[:, 0]), 1), round(rms(dd[:, 1]), 1)]
        # the bending estimand as in Section 4.8 (each profile re-referenced over the common span)
        za, at, bt_ = common_span_profiles(rd[i], rl[i], "bend_transverse_mm")
        _, al, bl = common_span_profiles(rd[i], rl[i], "bend_longitudinal_mm")
        row["bending profiles (Section 4.8)"] = [round(rms(at - bt_), 1), round(rms(al - bl), 1)]
        za2, at2, bt2 = common_span_profiles(rd[i], rl2[i], "bend_transverse_mm")
        _, al2, bl2 = common_span_profiles(rd[i], rl2[i], "bend_longitudinal_mm")
        row["bending profiles, TLS rotated (height datum corrected)"] = [round(rms(at2 - bt2), 1),
                                                                         round(rms(al2 - bl2), 1)]
        dtv, dlg = at - bt_, al - bl
        row["per_mast_stats"] = {
            "loa_mm": [round(1.96 * float(np.std(dtv)), 1), round(1.96 * float(np.std(dlg)), 1)],
            "corr": [round(float(np.corrcoef(at, bt_)[0, 1]), 2), round(float(np.corrcoef(al, bl)[0, 1]), 2)],
            "max_abs_mm": [round(float(np.max(np.abs(dtv))), 1), round(float(np.max(np.abs(dlg))), 1)],
            "common_span_m": [round(float(za[0]), 1), round(float(za[-1]), 1)],
        }
        var[NAMES[i]] = row
        print(f"  {NAMES[i]}: {row}", flush=True)
    out["A_variants"] = var

    print("\nC. TLS-side uncertainty")
    from mastfit.validate import ovality_study
    tls = {}
    for i in usable:
        P = robust_clean(lt[i])
        cov_med, cov_p10 = tls_coverage(P)
        r = rl[i]
        n_per_m = min(len(P) / float(np.ptp(P[:, 2])), 3000.0)   # cap for run time
        e = [compare_to_truth(*tls_phantom(i, sd, cov_med, n_per_m))[3] for sd in range(a.seeds)]
        eps_t = float(np.mean(e))
        # with the ovality that TLS measures where it covers the full circumference (2.5-5.5%)
        e_ov = {ov: float(np.mean([compare_to_truth(*tls_phantom(i, sd, cov_med, n_per_m, ovality=ov))[3]
                                   for sd in range(a.seeds)])) for ov in (0.025, 0.055)}
        st = ovality_study(P)
        st_u = ovality_study(dt[i])
        row = {"points": len(P), "points_per_m": round(len(P) / float(np.ptp(P[:, 2]))),
               "sigma_stat_median_mm": round(float(np.median(r.slice_sigma_mm)), 1),
               "coverage_median_pct": round(100 * cov_med), "coverage_p10_pct": round(100 * cov_p10),
               "ovality_tls_median_pct": round(float(np.nanmedian(st["ovality"])), 1),
               "ovality_tls_null_pct": round(float(np.nanmedian(st["null"])), 1),
               "ovality_uav_median_pct": round(float(np.nanmedian(st_u["ovality"])), 1),
               "tls_phantom_rmse_mm": round(eps_t, 1),
               "tls_phantom_rmse_ovality_2.5_5.5_mm": [round(e_ov[0.025], 1), round(e_ov[0.055], 1)],
               "expected_difference_mm": round(float(np.hypot(FLOOR_MEAN[i], eps_t)), 1)}
        tls[NAMES[i]] = row
        print(f"  {NAMES[i]}: {row}", flush=True)
    out["C_tls"] = tls
    os.makedirs("results", exist_ok=True)
    with open("results/tls_comparison.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
