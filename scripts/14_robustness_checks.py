#!/usr/bin/env python3
"""Robustness checks of the axis refinement, the error model and the data handling.

Writes results/robustness_checks.json and prints a summary. Sections:

  A  Tukey biweight: sensitivity of the bends to the tuning constant c (and to
     switching the IRLS refinement off).
  B  Correlated error model: what the default field is (correlation length,
     anisotropy, amplitude), sensitivity of the per-mast error floor to each, and
     the correlation length of the measured UAV-TLS difference profiles.
  C  Shared-curvature test without the Foremast; Foremast share of the shared mode.
  D  TLS frame tilt: one rotation fitted to the mast axes of both sensors.
  E  TLS band filter on oval and dented rings: are true surface points removed?
  F  Cloud-derived proxies for the missing UAV acquisition metadata.

Usage:  python scripts/14_robustness_checks.py [--seeds 6] [--skip-b]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from _common import base_parser, ensure_dirs, load_drone, load_laser  # noqa: E402
from mastfit.crossval import cross_validate  # noqa: E402
from mastfit.doming import doming_decomposition, shared_fraction  # noqa: E402
from mastfit.io import MAST_NAMES, coregister, robust_clean  # noqa: E402
from mastfit.pipeline import fit_mast  # noqa: E402
from mastfit.validate import (REAL_MAST_GEOMETRY, compare_to_truth,  # noqa: E402
                              matched_phantom, phantom_geometry)

NAMES = [MAST_NAMES[i][1] for i in range(5)]
MIN_LASER_POINTS = 5000
OUT = {}


def put(sec, key, val):
    OUT.setdefault(sec, {})[key] = val
    print(f"  [{sec}] {key}: {val}")


def efold(z, series_list):
    """Lag (in the units of z) at which the mean autocorrelation first drops below 1/e."""
    dz = float(np.mean(np.diff(z)))
    nlag = len(z) // 2
    acf = np.zeros(nlag)
    for s in series_list:
        s = np.asarray(s, float) - np.mean(s)
        v = np.dot(s, s)
        acf += np.array([np.dot(s[:len(s) - k], s[k:]) / v for k in range(nlag)])
    acf /= len(series_list)
    below = np.where(acf < np.exp(-1))[0]
    if len(below) == 0:
        return float("nan")
    k = below[0]
    # linear interpolation between k-1 and k
    a0, a1 = acf[k - 1], acf[k]
    frac = (a0 - np.exp(-1)) / (a0 - a1) if a0 != a1 else 0.0
    return float((k - 1 + frac) * dz)


# ---------------------------------------------------------------------------- A
def section_a(drone, args):
    print("\nA. TUKEY BIWEIGHT CONSTANT")
    base = [fit_mast(P, n_slices=args.slices, bandwidth_m=args.bandwidth) for P in drone]
    put("A", "default c = 4.685: total bend [mm]", [round(r.max_bend_mm, 1) for r in base])
    rows = {}
    for label, kw in (("c=3.0", dict(tukey_c=3.0)), ("c=6.0", dict(tukey_c=6.0)),
                      ("c=9.0", dict(tukey_c=9.0)), ("IRLS off (PCA seed)", dict(irls_iter=0))):
        rs = [fit_mast(P, n_slices=args.slices, bandwidth_m=args.bandwidth, **kw) for P in drone]
        dang = [float(np.degrees(np.arccos(np.clip(abs(a.axis_direction @ b.axis_direction), -1, 1))))
                for a, b in zip(rs, base)]
        dmax = [abs(a.max_bend_mm - b.max_bend_mm) for a, b in zip(rs, base)]
        rows[label] = {"bend_mm": [round(r.max_bend_mm, 1) for r in rs],
                       "axis_change_deg": [round(x, 3) for x in dang],
                       "max_abs_bend_change_mm": round(float(max(dmax)), 1)}
        put("A", label, rows[label])
    worst = max(v["max_abs_bend_change_mm"] for v in rows.values())
    put("A", "largest bend change over all variants [mm]", worst)
    # accuracy against truth on matched (white-noise) phantoms of the Mainmast geometry
    acc = {}
    for c in (3.0, 4.685, 6.0, 9.0):
        e = [compare_to_truth(*matched_phantom(seed=100 + sd, correlated=False, **phantom_geometry(1)),
                              tukey_c=c)[3] for sd in range(6)]
        acc[c] = round(float(np.mean(e)), 1)
    put("A", "phantom bending-profile RMSE vs c (6 seeds, Mainmast geometry) [mm]", acc)
    accc = {}
    for c in (3.0, 4.685, 6.0, 9.0):
        e = [compare_to_truth(*matched_phantom(seed=100 + sd, correlated=True, **phantom_geometry(1)),
                              tukey_c=c)[3] for sd in range(6)]
        accc[c] = round(float(np.mean(e)), 1)
    put("A", "same, correlated error (default field) [mm]", accc)
    return base


# ---------------------------------------------------------------------------- B
def legacy_field_efold(height_m):
    """e-folding length of the slice-translation part of the default Fourier field.

    With uniform random phases the field is stationary in z and its autocorrelation
    is (1/3) * sum_{k=1..3} cos(k pi tau / H).
    """
    tau = np.linspace(0, height_m, 20001)
    acf = np.mean([np.cos(k * np.pi * tau / height_m) for k in (1, 2, 3)], axis=0)
    return float(tau[np.argmax(acf < np.exp(-1))])


def floors(field_kw, seeds, masts=range(5)):
    """Bending-profile RMSE over matched per-mast phantoms; returns (all errors, per-mast mean)."""
    errs, per, eprof = [], [], []
    for i in masts:
        g = phantom_geometry(i)
        e_i = []
        for sd in range(seeds):
            P, zt, bt = matched_phantom(seed=100 * i + sd, correlated=True, **g, **field_kw)
            gz, T, R, rmse, _ = compare_to_truth(P, zt, bt)
            e_i.append(rmse)
            eprof.append((gz, R - T))
        errs += e_i
        per.append(float(np.mean(e_i)))
        PER_P90.append(float(np.percentile(e_i, 90)))
    return np.asarray(errs), per, eprof


PER_P90 = []
def summarize(errs):
    """Pooled statistics plus the per-mast 90th percentiles of the floors() call that
    produced ``errs`` (read from PER_P90)."""
    p90 = PER_P90[-5:]
    return {"mean": round(float(np.mean(errs)), 1), "p90": round(float(np.percentile(errs, 90)), 1),
            "min": round(float(np.min(errs)), 1), "max": round(float(np.max(errs)), 1),
            "per_mast_p90": [round(x, 1) for x in p90]}


def section_b(drone, laser, args):
    print("\nB. CORRELATED ERROR MODEL")
    H = [g[1] for g in REAL_MAST_GEOMETRY]
    put("B", "default field: vertical wavelengths 2H, H, 2H/3 for H = 42.1-53.7 m [m]",
        [round(2 * min(H), 1), round(2 * max(H), 1), round(min(H), 1), round(max(H), 1),
         round(2 * min(H) / 3, 1), round(2 * max(H) / 3, 1)])
    ef = [legacy_field_efold(h) for h in H]
    put("B", "default field: e-folding correlation length of the slice translation [m]",
        [round(x, 1) for x in ef])
    put("B", "default field: angular-order-1 share of the variance (2 of 5 modes) [%]", 40)
    put("B", "default field amplitude, per horizontal component [mm]",
        {"Section 4.5 error-model experiment": 20, "matched phantoms (Sections 4.6-4.7)": 45,
         "plus incoherent noise in matched phantoms": 40})

    seeds = args.seeds
    res = {}
    t0 = time.time()
    e, per, prof = floors({}, seeds)
    res["default"] = {**summarize(e), "per_mast_mean": [round(x, 1) for x in per]}
    zz = np.linspace(0, 1, 50)
    res["default"]["error_profile_efold_m"] = round(float(np.median(
        [efold(gz, [d]) for gz, d in prof])), 1)
    put("B", "default field (45 mm), floor", res["default"])

    sweep_l = {}
    for L in (1.0, 2.5, 5.0, 10.0, 20.0, 40.0):
        e, per, prof = floors({"corr_len_m": L}, seeds)
        s = {**summarize(e), "per_mast_mean": [round(x, 1) for x in per],
             "error_profile_efold_m": round(float(np.median([efold(gz, [d]) for gz, d in prof])), 1)}
        sweep_l[L] = s
        put("B", f"GP field l = {L:g} m (45 mm, isotropic)", s)
    res["corr_len"] = sweep_l

    sweep_a = {}
    for A in (0.015, 0.030, 0.045, 0.060):
        e, per, _ = floors({"noise_m": A}, seeds)
        sweep_a[int(round(A * 1000))] = {**summarize(e), "per_mast_mean": [round(x, 1) for x in per]}
        put("B", f"default field, amplitude {A * 1000:.0f} mm", sweep_a[int(round(A * 1000))])
    res["amplitude"] = sweep_a

    sweep_n = {}
    for an in (0.5, 1.0, 2.0):
        e, per, _ = floors({"corr_len_m": 10.0, "corr_anisotropy": an}, seeds)
        sweep_n[an] = {**summarize(e), "per_mast_mean": [round(x, 1) for x in per]}
        put("B", f"GP l = 10 m, anisotropy (bend plane : perpendicular) = {an:g}", sweep_n[an])
    res["anisotropy"] = sweep_n

    sweep_f = {}
    for fr in (0.0, 0.4, 0.8):
        e, per, _ = floors({"corr_len_m": 10.0, "corr_angular_frac": fr}, seeds)
        sweep_f[fr] = {**summarize(e)}
        put("B", f"GP l = 10 m, angular share {fr:.0%}", sweep_f[fr])
    res["angular_share"] = sweep_f
    put("B", "sweep run time [s]", round(time.time() - t0))

    # empirical correlation length of the UAV-TLS difference profiles
    usable = [i for i in range(5) if len(laser[i]) >= MIN_LASER_POINTS]
    dt, lt = coregister(drone, laser)
    rd = [fit_mast(dt[i], n_slices=args.slices, bandwidth_m=args.bandwidth) for i in range(5)]
    rl = [fit_mast(robust_clean(lt[i]), n_slices=args.slices, bandwidth_m=args.bandwidth)
          if i in usable else None for i in range(5)]
    cv = cross_validate(rd, rl, usable, n_boot=10)
    emp = {}
    for comp in ("transverse", "longitudinal"):
        pm = cv[comp]["per_mast"]
        vals = [efold(pm[i]["z"], [pm[i]["diff"]]) for i in usable]
        emp[comp] = {"per_mast_m": [round(v, 1) for v in vals], "median_m": round(float(np.nanmedian(vals)), 1),
                     "rms_mm": round(float(cv[comp]["rms_diff_mm"]), 1)}
        put("B", f"UAV-TLS difference e-folding length, {comp}", emp[comp])
    res["empirical"] = emp
    OUT["B_detail"] = res
    return rd, rl, usable, cv


# ---------------------------------------------------------------------------- C
def section_c(base):
    print("\nC. SHARED-CURVATURE TEST WITHOUT THE FOREMAST")
    dm = doming_decomposition(base)
    for comp in ("transverse", "longitudinal"):
        M = dm[comp]
        f5, s5 = shared_fraction(M)
        f4, s4 = shared_fraction(M[1:])
        put("C", f"{comp}: shared fraction 5 masts / without Foremast [%]",
            [round(f5 * 100, 1), round(f4 * 100, 1)])
        put("C", f"{comp}: peak |shared profile| 5 masts / without Foremast [mm]",
            [round(float(np.abs(s5 - s5.mean()).max()), 1), round(float(np.abs(s4 - s4.mean()).max()), 1)])
        contrib = (M[0] - M[0].mean()) / len(M)
        put("C", f"{comp}: Foremast contribution to the 5-mast shared profile, peak [mm]",
            round(float(np.abs(contrib).max()), 1))
    put("C", "smallest attainable exact two-sided p with five masts", round(2 / 120, 3))


# ---------------------------------------------------------------------------- D
def section_d(rd, rl, usable):
    print("\nD. TLS FRAME TILT")
    # The axes are within ~2 deg of vertical, so a rotation of the TLS frame acts on
    # them as a common horizontal offset of the unit direction vectors (small-angle
    # tilt). A heading term is not identifiable from near-vertical vectors and the
    # headings are already aligned by the mast-base line, so only the tilt is fitted.
    A = np.array([rd[i].axis_direction for i in usable])
    B = np.array([rl[i].axis_direction for i in usable])
    A = A * np.sign(A[:, 2:3])
    B = B * np.sign(B[:, 2:3])
    ah, bh = A[:, :2] / A[:, 2:3], B[:, :2] / B[:, 2:3]          # tan of the axis tilt
    t = (bh - ah).mean(axis=0)
    deg = lambda v: float(np.degrees(np.arctan(np.linalg.norm(v))))
    put("D", "common TLS-minus-UAV tilt: magnitude [deg], transverse and longitudinal components [deg]",
        [round(deg(t), 2), round(float(np.degrees(np.arctan(t[0]))), 2), round(float(np.degrees(np.arctan(t[1]))), 2)])
    put("D", "direction of the TLS tilt", "toward the bow" if t[1] < 0 else "toward the stern")
    before = [deg(b - a) for a, b in zip(ah, bh)]
    after = [deg(b - t - a) for a, b in zip(ah, bh)]
    put("D", "per-mast axis difference UAV vs TLS, before / after removing the common tilt [deg]",
        {NAMES[i]: [round(x, 2), round(y, 2)] for i, x, y in zip(usable, before, after)})
    put("D", "max residual after removing the common tilt [deg]", round(max(after), 2))
    rake_u = [deg(a) for a in ah]
    rake_t = [deg(b - t) for b in bh]
    put("D", "rake UAV / tilt-corrected TLS [deg]",
        {NAMES[i]: [round(x, 2), round(y, 2)] for i, x, y in zip(usable, rake_u, rake_t)})
    put("D", "rake differences UAV - corrected TLS [deg]", [round(x - y, 2) for x, y in zip(rake_u, rake_t)])
    put("D", "raking aft: UAV, corrected TLS", [[bool(a[1] > 0) for a in ah], [bool((b - t)[1] > 0) for b in bh]])
    put("D", "TLS raking aft before correction", [bool(b[1] > 0) for b in bh])


# ---------------------------------------------------------------------------- E
def ring_mast(height=40.0, r0=0.45, taper=0.004, ovality=0.0, dent_depth=0.0, noise=0.003,
              n_per_m=3000, rigging_frac=0.05, seed=0):
    """Synthetic TLS-like mast: oval cross-sections (orientation varying with height),
    an optional 30-degree dent over 2 m, TLS-level noise and off-surface rigging returns.
    Returns (points, is_surface, true_centres_z, true_centres_xy)."""
    rg = np.random.default_rng(seed)
    n = int(height * n_per_m)
    z = rg.uniform(0, height, n)
    th = rg.uniform(0, 2 * np.pi, n)
    r = r0 - taper * z
    # ovality e = 1 - rmin/rmax  ->  r(theta) = rbar * (1 + a cos 2(theta - phi)), a = e / (2 - e)
    a = ovality / (2 - ovality)
    phi = 0.3 * z
    rr = r * (1 + a * np.cos(2 * (th - phi)))
    dent = (np.abs(((th - 1.0 + np.pi) % (2 * np.pi)) - np.pi) < np.radians(15)) & (np.abs(z - 20) < 1.0)
    rr = rr - dent_depth * dent
    rr = rr + rg.normal(0, noise, n)
    P = np.column_stack([rr * np.cos(th), rr * np.sin(th), z])
    surf = np.ones(n, bool)
    nr = int(rigging_frac * n)
    zr = rg.choice(np.linspace(3, height - 3, 6), nr) + rg.normal(0, 0.4, nr)
    thr = rg.normal(2.5, 0.3, nr)
    rrr = (r0 - taper * zr) + rg.uniform(0.05, 0.6, nr)
    P = np.vstack([P, np.column_stack([rrr * np.cos(thr), rrr * np.sin(thr), zr])])
    surf = np.concatenate([surf, np.zeros(nr, bool)])
    return P, surf, dent


def section_e():
    print("\nE. TLS BAND FILTER ON OVAL AND DENTED RINGS")
    out = {}
    for ov in (0.0, 0.10, 0.20, 0.30):
        P, surf, _ = ring_mast(ovality=ov, seed=1)
        C = robust_clean(P)
        keep = np.isin(P.view([("", P.dtype)] * 3).ravel(), C.view([("", C.dtype)] * 3).ravel())
        lost = 100 * (1 - keep[surf].mean())
        rig = 100 * (1 - keep[~surf].mean())
        r_bef = fit_mast(P[surf], n_slices=60)
        r_aft = fit_mast(C, n_slices=60)
        dmax = abs(r_aft.max_bend_mm - r_bef.max_bend_mm)
        out[f"ovality {ov:.0%}"] = {"surface_points_removed_pct": round(lost, 2),
                                    "rigging_points_removed_pct": round(rig, 1),
                                    "bend_change_vs_surface_only_mm": round(dmax, 1)}
        put("E", f"ovality {ov:.0%}", out[f"ovality {ov:.0%}"])
    for dd in (0.010, 0.030, 0.060):
        P, surf, dent = ring_mast(ovality=0.10, dent_depth=dd, seed=2)
        C = robust_clean(P)
        keep = np.isin(P.view([("", P.dtype)] * 3).ravel(), C.view([("", C.dtype)] * 3).ravel())
        dent_full = np.concatenate([dent, np.zeros((~surf).sum(), bool)])
        lost_dent = 100 * (1 - keep[dent_full].mean())
        lost_other = 100 * (1 - keep[surf & ~dent_full].mean())
        r_bef = fit_mast(P[surf], n_slices=60)
        r_aft = fit_mast(C, n_slices=60)
        out[f"dent {dd * 1000:.0f} mm"] = {"dent_points_removed_pct": round(lost_dent, 1),
                                           "other_surface_removed_pct": round(lost_other, 2),
                                           "bend_change_mm": round(abs(r_aft.max_bend_mm - r_bef.max_bend_mm), 1)}
        put("E", f"dent {dd * 1000:.0f} mm deep, 30 deg x 2 m, on 10% ovality", out[f"dent {dd * 1000:.0f} mm"])
    OUT["E_detail"] = out


# ---------------------------------------------------------------------------- F
def section_f(drone):
    print("\nF. CLOUD-DERIVED PROXIES FOR THE MISSING METADATA")
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover
        put("F", "nearest-neighbour spacing", "scipy unavailable")
        return
    sp = []
    for P in drone:
        d, _ = cKDTree(P).query(P, k=2)
        sp.append(float(np.median(d[:, 1]) * 1000))
    put("F", "UAV median nearest-neighbour point spacing per mast [mm]", [round(x) for x in sp])
    dens = [len(P) / (np.ptp(P[:, 2])) for P in drone]
    put("F", "UAV points per metre of mast height", [round(x) for x in dens])


def main():
    p = base_parser(__doc__)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--skip-b", action="store_true")
    args = p.parse_args()
    ensure_dirs("results")
    drone, laser = load_drone(args), load_laser(args)
    base = section_a(drone, args)
    if args.skip_b:
        dt, lt = coregister(drone, laser)
        usable = [i for i in range(5) if len(laser[i]) >= MIN_LASER_POINTS]
        rd = [fit_mast(dt[i], n_slices=args.slices, bandwidth_m=args.bandwidth) for i in range(5)]
        rl = [fit_mast(robust_clean(lt[i]), n_slices=args.slices, bandwidth_m=args.bandwidth)
              if i in usable else None for i in range(5)]
    else:
        rd, rl, usable, _ = section_b(drone, laser, args)
    section_c(base)
    section_d(rd, rl, usable)
    section_e()
    section_f(drone)
    with open("results/robustness_checks.json", "w") as f:
        json.dump(OUT, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\nwrote results/robustness_checks.json")


if __name__ == "__main__":
    main()
