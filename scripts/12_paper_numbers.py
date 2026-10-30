#!/usr/bin/env python3
"""The numbers ledger: every number quoted in the paper, recomputed from the data.

Each line prints the paper section, a key, and the value as it should appear in the
text. The full-precision values are written to ``results/paper_numbers.json``.

Runtime is about two minutes (the synthetic studies dominate). Requires both
``.mat`` files in ``./data/``.

Sections follow the paper: 3.3-3.5 (geometry used in the method description),
4.1-4.10 (results), and a frame check that the transverse/longitudinal labels of the
bending components are correct.
"""
import importlib
import json
import os
from statistics import NormalDist

import numpy as np

from _common import RESDIR, base_parser, ensure_dirs, label_of, load_drone, load_laser
from mastfit.baseline import naive_pipeline
from mastfit.crossval import cross_validate
from mastfit.doming import doming_decomposition, positional_trend, shared_fraction
from mastfit.geometry import irls_axis, orthonormal_basis, pca_axis
from mastfit.io import coregister, robust_clean, ship_frame
from mastfit.pipeline import fit_mast
from mastfit.validate import REAL_MAST_GEOMETRY, ovality_study

NAMES = ["Foremast", "Mainmast", "Middle Mast", "Mizzenmast", "Jiggermast"]
MIN_LASER_POINTS = 2000
LEDGER = {}


def put(section, key, value, text=None):
    """Record a value and print it with the text form used in the paper."""
    LEDGER.setdefault(section, {})[key] = value
    shown = text if text is not None else value
    print(f"  [{section}] {key}: {shown}")


def rng_str(a, fmt="{:.0f}", sep="–"):
    a = np.asarray(a, dtype=float)
    return fmt.format(a.min()) + sep + fmt.format(a.max())


def band_runs(height, flagged):
    """Contiguous runs of flagged slices, excluding runs that touch either end."""
    idx = np.flatnonzero(flagged)
    runs, cur = [], []
    for k in idx:
        if cur and k != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(k)
    if cur:
        runs.append(cur)
    n = len(height)
    inner = [r for r in runs if r[0] > 0 and r[-1] < n - 1]
    return [float(np.mean(height[r])) for r in inner]


def main():
    args = base_parser(__doc__).parse_args()
    ensure_dirs(RESDIR)
    drone, laser = load_drone(args), load_laser(args)
    res = [fit_mast(P, name=NAMES[i], n_slices=args.slices, bandwidth_m=args.bandwidth)
           for i, P in enumerate(drone)]

    # ------------------------------------------------------------------ geometry (I, 3.3)
    print("\nGEOMETRY OF THE MASTS (Introduction, Sections 3.2-3.3)")
    heights = [P[:, 2].max() - P[:, 2].min() for P in drone]
    diam = [r.median_diameter_mm for r in res]
    put("3.3", "mast height from the UAV cloud [m]", [round(h, 1) for h in heights],
        rng_str(heights, "{:.0f}") + " m")
    put("3.3", "median diameter [m]", [round(d / 1000, 2) for d in diam], rng_str(np.array(diam) / 1000, "{:.1f}") + " m")
    aspect = [h / (d / 1000) for h, d in zip(heights, diam)]
    put("3.3", "height-to-diameter ratio (min)", round(min(aspect), 1), f">{int(min(aspect) // 10 * 10)}:1")
    put("3.3", "PCA eigenvalue ratio lambda1/lambda2", [round(r.eig_ratio) for r in res],
        rng_str([r.eig_ratio for r in res], "{:.0f}"))

    # ------------------------------------------------------------------ IRLS (3.4)
    print("\nAXIS REFINEMENT (Section 3.4)")
    shifts, conv = [], []
    for P in drone:
        c0, d0, _ = pca_axis(P)
        prev, steps = d0, []
        for k in range(1, 6):
            _, dk = irls_axis(P, c0, d0, n_iter=k)
            steps.append(np.degrees(np.arccos(min(1.0, abs(dk @ prev)))))
            prev = dk
        shifts.append(np.degrees(np.arccos(min(1.0, abs(prev @ d0)))))
        conv.append(next((k + 1 for k, s in enumerate(steps) if s < 0.01), 5))
    put("3.4", "IRLS shift of the PCA seed [deg]", [round(s, 3) for s in shifts], rng_str(shifts, "{:.2f}") + "°")
    put("3.4", "iterations until the axis changes by < 0.01 deg", conv, rng_str(conv, "{:.0f}"))

    # ------------------------------------------------------------------ slicing (3.5)
    print("\nSECTIONING (Section 3.5)")
    deltas, cov_all = [], []
    for P in drone:
        c0, d0, _ = pca_axis(P)
        c, d = irls_axis(P, c0, d0)
        B, d = orthonormal_basis(d)
        t = (P - c) @ d
        xy = (P - c) @ B.T
        edges = np.linspace(t.min(), t.max(), args.slices + 1)
        deltas.append((t.max() - t.min()) / args.slices)
        for k in range(args.slices):
            m = (t >= edges[k]) & ((t < edges[k + 1]) if k < args.slices - 1 else (t <= edges[k + 1]))
            if m.sum() < 40:
                continue
            s = xy[m]
            c1 = np.median(s, axis=0)
            ang = np.degrees(np.arctan2(s[:, 1] - c1[1], s[:, 0] - c1[0]))
            cov_all.append(len(set(((ang + 180) // 30).astype(int))) / 12 * 100)
    cov_all = np.asarray(cov_all)
    taper_change = [g[3] * dl * 1000 for g, dl in zip(REAL_MAST_GEOMETRY, deltas)]
    put("3.5", "slice thickness Delta [m]", [round(x, 3) for x in deltas], rng_str(deltas, "{:.1f}") + " m")
    put("3.5", "taper-induced radius change per slice [mm]", [round(x, 2) for x in taper_change],
        f"< {np.ceil(max(taper_change)):.0f} mm")
    put("3.5", "angular coverage of candidate slices: median / min [%]",
        [round(float(np.median(cov_all)), 1), round(float(cov_all.min()), 1)],
        f"median {np.median(cov_all):.0f}%, minimum {cov_all.min():.0f}%")
    put("3.5", "slices rejected by the coverage gate", int((cov_all / 100 < 0.55).sum()))

    # ------------------------------------------------------------------ 4.1
    print("\nDATASET AND PIPELINE EXECUTION (Section 4.1)")
    npts = [len(P) for P in drone]
    put("4.1", "points per mast", npts, f"{min(npts):,}–{max(npts):,}; {sum(npts):,} total")
    base_rm, top_rm, bands_flagged, n_bands, intervals = [], [], [], [], []
    for r in res:
        fl = r.flare_removed
        k = 0
        while k < len(fl) and fl[k]:
            k += 1
        j = 0
        while j < len(fl) and fl[len(fl) - 1 - j]:
            j += 1
        base_rm.append(k)
        top_rm.append(j)
        flagged = r.band_flagged & ~fl
        bands_flagged.append(int(flagged.sum()))
        centres = band_runs(r.all_slice_height, flagged)
        n_bands.append(len(centres))
        intervals += list(np.diff(centres))
    put("4.1", "base slices removed by the flare test", base_rm, rng_str(base_rm))
    put("4.1", "top slices removed by the flare test", top_rm, rng_str(top_rm))
    put("4.1", "rigging-band slices flagged", bands_flagged, rng_str(bands_flagged))
    put("4.1", "bands per mast (runs not touching the ends)", n_bands, rng_str(n_bands))
    put("4.1", "vertical interval between bands, median [m]", round(float(np.median(intervals)), 2),
        f"≈{np.median(intervals):.0f} m (IQR {np.percentile(intervals, 25):.1f}–{np.percentile(intervals, 75):.1f} m)")

    # ------------------------------------------------------------------ 4.2
    print("\nBASE-FLARE REMOVAL AND SMOOTHING SCALE (Section 4.2)")
    abl = {NAMES[i]: (naive_pipeline(P, quality_control=False), naive_pipeline(P, quality_control=True))
           for i, P in enumerate(drone)}
    put("4.2", "unsmoothed baseline, 40 slices: flare test off -> on [mm]",
        {k: [round(v[0]), round(v[1])] for k, v in abl.items()},
        "; ".join(f"{k} {v[1]:.0f}→{v[0]:.0f}" for k, v in abl.items()))
    off = [fit_mast(P, apply_quality_control=False).max_bend_mm for P in drone]
    dq = [abs(o - r.max_bend_mm) for o, r in zip(off, res)]
    put("4.2", "kernel-smoothed: |change| when QC is disabled [mm]", [round(x, 1) for x in dq],
        f"at most {max(dq):.0f} mm")
    counts = (40, 100, 200)          # Delta ~ 1.25, 0.5, 0.25 m on a ~50 m mast
    fore_unsmoothed = [naive_pipeline(drone[0], n_slices=n) for n in counts]
    put("4.2", "Foremast unsmoothed bend at 40, 100, 200 slices [mm]",
        [round(x) for x in fore_unsmoothed], "→".join(f"{x:.0f}" for x in fore_unsmoothed))
    sweep = {NAMES[i]: [fit_mast(P, n_slices=n).max_bend_mm for n in counts] for i, P in enumerate(drone)}
    rel = {k: max(abs(v[0] - v[1]), abs(v[2] - v[1])) / v[1] * 100 for k, v in sweep.items()}
    put("4.2", "smoothed bend at 40, 100, 200 slices [mm]",
        {k: [round(x) for x in v] for k, v in sweep.items()},
        "; ".join(f"{k} " + "/".join(f"{x:.0f}" for x in v) for k, v in sweep.items()))
    put("4.2", "largest change relative to 100 slices [%]", {k: round(v, 1) for k, v in rel.items()},
        f"at most {max(rel.values()):.0f}% ({max(rel, key=rel.get)})")
    bw = {NAMES[i]: [fit_mast(P, bandwidth_m=L).max_bend_mm for L in (2, 3, 4, 5)] for i, P in enumerate(drone)}
    put("4.2", "bend at L = 2, 3, 4, 5 m [mm]", {k: [round(x) for x in v] for k, v in bw.items()},
        "; ".join(f"{k} " + ", ".join(f"{x:.0f}" for x in v) for k, v in bw.items()))

    # ------------------------------------------------------------------ Table 1, 4.3
    print("\nTABLE 1 AND SECTION 4.3")
    for i, r in enumerate(res):
        put("Table 1", NAMES[i], {
            "points": r.n_points, "rake_deg": round(r.rake_deg, 2),
            "median_diameter_mm": round(r.median_diameter_mm),
            "transverse_mm": round(float(np.nanmax(np.abs(r.bend_transverse_mm))), 1),
            "longitudinal_mm": round(float(np.nanmax(np.abs(r.bend_longitudinal_mm))), 1),
            "total_mm": round(r.max_bend_mm, 1), "sigma_stat_mm": round(r.median_sigma_mm, 1),
            "slices_kept": r.n_slices_kept, "slices_accepted": r.n_slices_total})
    rakes = [r.rake_deg for r in res]
    put("4.3", "rake range [deg]", [round(x, 2) for x in rakes], rng_str(rakes, "{:.1f}") + "°")
    # rake direction in the ship frame (+y = toward the stern)
    fd = ship_frame(drone)
    aft = []
    for P in drone:
        c0, d0, _ = pca_axis(P)
        _, dd = irls_axis(P, c0, d0)
        aft.append(float(dd[:2] @ fd.longitudinal))
    put("4.3", "rake points aft (toward the stern) on every mast", bool(all(a > 0 for a in aft)))
    sig = [r.median_sigma_mm for r in res]
    put("4.3", "sigma_stat range [mm]", [round(s, 1) for s in sig], rng_str(sig, "{:.0f}") + " mm")
    m = sum(r.n_slices_kept for r in res)
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * m))
    put("4.3", "Bonferroni: comparisons m (retained slices, five masts)", m)
    put("4.3", "Bonferroni threshold z (two-sided, alpha = 0.05)", round(z, 2), f"≈{z:.1f}σ_stat")
    put("4.3", "Bonferroni threshold z*sigma_stat [mm]", [round(z * s, 1) for s in sig], rng_str([z * s for s in sig]) + " mm")

    # ------------------------------------------------------------------ 4.4
    print("\nSHARED-CURVATURE TEST (Section 4.4)")
    dm = doming_decomposition(res)
    put("4.4", "shared fraction transverse / longitudinal [%]",
        [round(dm["shared_fraction_transverse"] * 100, 1), round(dm["shared_fraction_longitudinal"] * 100, 1)],
        f"{dm['shared_fraction_transverse'] * 100:.0f}% / {dm['shared_fraction_longitudinal'] * 100:.0f}%")
    for comp in ("transverse", "longitudinal"):
        pt = positional_trend(dm[comp])
        put("4.4", f"{comp}: loading vs position, r (exact p)", [round(pt["r"], 3), round(pt["p"], 3)],
            f"r = {pt['r']:+.2f}, p = {pt['p']:.3f} ({pt['n_perm']} orderings)")
        pt4 = positional_trend(dm[comp][1:], positions=np.arange(1, 5))
        put("4.4", f"{comp}: without the Foremast, r (exact p)", [round(pt4["r"], 3), round(pt4["p"], 3)],
            f"r = {pt4['r']:+.2f}, p = {pt4['p']:.2f}")
        put("4.4", f"{comp}: shared fraction without the Foremast [%]",
            round(shared_fraction(dm[comp][1:])[0] * 100, 1))

    # ------------------------------------------------------------------ 4.5
    print("\nSYNTHETIC VALIDATION (Section 4.5)  -- scripts/05_synthetic_validation.py")
    syn = importlib.import_module("05_synthetic_validation")
    em = syn.experiment_error_model()
    fl = syn.experiment_flare()
    dom = syn.experiment_doming()
    put("4.5", "white-noise RMSE [mm]", round(float(em["white"].mean()), 1), f"≈{em['white'].mean():.0f} mm")
    put("4.5", "correlated-error RMSE, mean (range) [mm]",
        [round(float(em["correlated"].mean()), 1), round(float(em["correlated"].min()), 1), round(float(em["correlated"].max()), 1)],
        f"≈{em['correlated'].mean():.0f} mm ({em['correlated'].min():.0f}–{em['correlated'].max():.0f} mm)")
    put("4.5", "ratio correlated / white", round(float(em["correlated"].mean() / em["white"].mean()), 2))
    c = fl["C"]
    put("4.5", "phantom flare: truth, baseline off, baseline on [mm]",
        [round(c["truth"]), round(c["baseline_off"]), round(c["baseline_on"])],
        f"{c['truth']:.0f} → {c['baseline_off']:.0f} (off), {c['baseline_on']:.0f} (on)")
    put("4.5", "doming test: injected -> recovered shared fraction [%]",
        [round(dom["fraction_injected"] * 100, 1), round(dom["fraction_recovered"] * 100, 1)],
        f"{dom['fraction_injected'] * 100:.0f}% → {dom['fraction_recovered'] * 100:.0f}%")
    put("4.5", "doming test: per-mast bend error [mm]", round(dom["per_mast_error_mm"], 1))

    # ------------------------------------------------------------------ 4.6
    print("\nPHANTOM REPRESENTATIVENESS (Section 4.6, Table 2)  -- scripts/08_phantom_match.py")
    pm = importlib.import_module("08_phantom_match").compute(args)
    for key in ("sigma_med", "sigma_p90", "trimmed", "coverage_p10", "density"):
        put("Table 2", key, [round(pm["real"][key], 1), round(pm["orig"][key], 1), round(pm["match"][key], 1)])
    put("4.6", "transferred RMSE original -> matched [mm]",
        [round(float(pm["errs_orig"].mean()), 1), round(float(pm["errs_match"].mean()), 1)],
        f"{pm['errs_orig'].mean():.0f} → {pm['errs_match'].mean():.0f} mm "
        f"(matched range {pm['errs_match'].min():.0f}–{pm['errs_match'].max():.0f} mm)")

    # ------------------------------------------------------------------ 4.7
    print("\nPER-MAST ERROR FLOOR (Section 4.7, Table 3)  -- scripts/10_per_mast_accuracy.py")
    pmf = importlib.import_module("10_per_mast_accuracy")
    f10, f30 = pmf.compute_floors(10), pmf.compute_floors(30)
    bends = {NAMES[i]: r.max_bend_mm for i, r in enumerate(res)}
    means10, p90_10, d_mean, d_p90, ci_m, ci_p = [], [], [], [], [], []
    for i, n in enumerate(NAMES):
        e10, e30 = f10[n], f30[n]
        mu10, q10, mu30, q30 = e10.mean(), np.percentile(e10, 90), e30.mean(), np.percentile(e30, 90)
        means10.append(mu10)
        p90_10.append(q10)
        d_mean.append(abs(mu30 - mu10))
        d_p90.append(abs(q30 - q10))
        (mlo, mhi), (plo, phi) = pmf.bootstrap_ci(e30, seed=i)
        ci_m.append((mhi - mlo) / 2)
        ci_p.append((phi - plo) / 2)
        put("Table 3", n, {"height_m": REAL_MAST_GEOMETRY[i][1], "taper": REAL_MAST_GEOMETRY[i][3],
                             "bend_mm": round(bends[n], 1), "floor_mean": round(mu10, 1),
                             "floor_p90": round(q10, 1),
                             "floor_mean_30": round(mu30, 1), "floor_p90_30": round(q30, 1)})
    put("4.7", "floor means (10 realizations) [mm]", rng_str(means10))
    put("4.7", "floor 90th percentiles (10 realizations) [mm]", rng_str(p90_10))
    put("4.7", "30 realizations: change of mean / p90 [mm]",
        [round(max(d_mean), 1), round(max(d_p90), 1)], f"at most {max(max(d_mean), max(d_p90)):.0f} mm")
    put("4.7", "30 realizations: bootstrap 95% CI half-width, mean / p90 [mm]",
        [rng_str(ci_m), rng_str(ci_p)], f"±{rng_str(ci_m)} mm / ±{rng_str(ci_p)} mm")

    # ------------------------------------------------------------------ 4.8
    print("\nCROSS-VALIDATION AGAINST TLS (Section 4.8)  -- scripts/04_crossvalidation.py")
    tls_n = [len(P) for P in laser]
    put("4.8", "TLS points per mast", tls_n)
    usable = [i for i in range(5) if len(laser[i]) >= MIN_LASER_POINTS]
    clean_frac = {NAMES[i]: 100 * (1 - len(robust_clean(laser[i])) / len(laser[i])) for i in usable}
    put("4.8", "TLS points removed by the band-wise radial filter [%]",
        {k: round(v, 1) for k, v in clean_frac.items()}, rng_str(list(clean_frac.values())) + "%")
    dt, lt = coregister(drone, laser)
    rd = [fit_mast(dt[i], n_slices=args.slices, bandwidth_m=args.bandwidth) for i in range(5)]
    rl = [fit_mast(robust_clean(lt[i]), n_slices=args.slices, bandwidth_m=args.bandwidth)
          if i in usable else None for i in range(5)]
    cv = cross_validate(rd, rl, usable)
    for comp in ("transverse", "longitudinal"):
        s = cv[comp]
        rms = {NAMES[i]: float(np.sqrt(np.mean(v["diff"] ** 2))) for i, v in s["per_mast"].items()}
        put("4.8", f"{comp}: LoA [mm] (mast-block 95% CI)",
            [round(s["loa_mm"], 1), round(s["loa_ci_mm"][0], 1), round(s["loa_ci_mm"][1], 1)],
            f"±{s['loa_mm']:.0f} mm (CI ±{s['loa_ci_mm'][0]:.0f} to ±{s['loa_ci_mm'][1]:.0f} mm)")
        put("4.8", f"{comp}: correlation, SD and RMS of the differences",
            [round(s["corr"], 3), round(s["sd_mm"], 1), round(s["rms_diff_mm"], 1)],
            f"r = {s['corr']:+.2f}; RMS difference {s['rms_diff_mm']:.0f} mm")
        put("4.8", f"{comp}: per-mast RMS difference [mm]", {k: round(v, 1) for k, v in rms.items()})
        put("4.8", f"{comp}: lag-1 autocorrelation, n, effective n",
            [round(s["lag1"], 3), s["n"], round(s["n_effective"], 2)])
    import copy
    Ts, Ls = [], []
    rng = np.random.default_rng(0)
    for _ in range(40):
        sh = rng.uniform(-1, 1, len(usable))
        rr = [None] * 5
        for k, i in enumerate(usable):
            rr[i] = copy.copy(rl[i])
            rr[i].height_origin_m = rl[i].height_origin_m + sh[k] / rl[i].axis_direction[2]
        c2 = cross_validate(rd, rr, usable, n_boot=10)
        Ts.append(c2["transverse"]["loa_mm"] - cv["transverse"]["loa_mm"])
        Ls.append(c2["longitudinal"]["loa_mm"] - cv["longitudinal"]["loa_mm"])
    put("4.8", "LoA change for random vertical shifts of the TLS masts within +-1 m [mm]",
        [round(float(np.abs(Ts).max()), 1), round(float(np.abs(Ls).max()), 1)],
        f"at most {max(np.abs(Ts).max(), np.abs(Ls).max()):.0f} mm")
    put("4.8", "common height span per mast [m]", {NAMES[i]: [round(a, 1), round(b, 1)] for i, (a, b) in cv["span"].items()})
    pk = {NAMES[i]: [round(a), round(b)] for i, (a, b) in cv["peak_total"].items()}
    put("4.8", "peak total bend over the common span, UAV / TLS [mm]", pk,
        "differences " + rng_str([abs(a - b) for a, b in pk.values()]) + " mm")
    full = {NAMES[i]: [round(res[i].max_bend_mm), round(rl[i].max_bend_mm)] for i in usable}
    put("4.8", "peak total bend over each sensor's own span, UAV / TLS [mm]", full,
        "differences " + rng_str([abs(a - b) for a, b in full.values()]) + " mm")
    # positions and axis directions in the common frame
    pd_ = np.array([[np.median(t[:, 0]), np.median(t[:, 1])] for t in dt])
    pl_ = np.array([[np.median(t[:, 0]), np.median(t[:, 1])] for t in lt])
    dx = np.abs(pd_[usable, 0] - pl_[usable, 0])
    dy = pd_[usable, 1] - pl_[usable, 1]
    put("4.8", "mast positions UAV - TLS: max |transverse|, max |longitudinal - mean| [m]",
        [round(float(dx.max()), 2), round(float(np.abs(dy - dy.mean()).max()), 2)])
    put("4.8", "mast positions UAV - TLS: max |longitudinal| [m]", round(float(np.abs(dy).max()), 2))
    ang, tilt = [], []
    for i in usable:
        a, b = rd[i].axis_direction, rl[i].axis_direction
        ha, hb = a[:2], b[:2]
        ang.append(np.degrees(np.arccos(np.clip(ha @ hb / np.linalg.norm(ha) / np.linalg.norm(hb), -1, 1))))
        tilt.append(b - a)
    tilt = np.mean(tilt, axis=0)
    put("4.8", "angle between UAV and TLS lean directions [deg]", [round(x) for x in ang],
        f"≈{np.mean(ang):.0f}°")
    put("4.8", "mean TLS-minus-UAV axis difference: magnitude [deg], longitudinal sign",
        [round(float(np.degrees(np.linalg.norm(tilt[:2]))), 2), "toward the bow" if tilt[1] < 0 else "toward the stern"])
    put("4.8", "TLS rake of each usable mast [deg]", [round(rl[i].rake_deg, 2) for i in usable])

    # ------------------------------------------------------------------ 4.9
    print("\nCROSS-SECTION OVALITY (Section 4.9)")
    ov = [ovality_study(P) for P in drone]
    med = [float(np.median(o["ovality"])) for o in ov]
    nul = [float(np.median(o["null"])) for o in ov]
    msk = [float(np.median(o["masked"])) for o in ov]
    put("4.9", "median ovality per mast [%]", [round(x, 1) for x in med], rng_str(med) + "%")
    put("4.9", "median round-ring null [%]", [round(x, 1) for x in nul], rng_str(nul) + "%")
    put("4.9", "median ovality after masking the worst 30-degree sector [%]", [round(x, 1) for x in msk],
        f"e.g. Foremast {med[0]:.1f} → {msk[0]:.1f}%")
    put("4.9", "masked ovality / null (min over masts)", round(min(a / b for a, b in zip(msk, nul)), 1))

    # ------------------------------------------------------------------ 4.10
    print("\nINTER-MAST GEOMETRY (Section 4.10)")
    for tag, T in (("UAV", dt), ("TLS", lt)):
        pos = np.array([[np.median(t[:, 0]), np.median(t[:, 1])] for t in T])
        gaps = np.diff(np.sort(pos[:, 1]))
        put("4.10", f"{tag} spacing from median positions [m]", [round(g, 2) for g in gaps],
            ", ".join(f"{g:.1f}" for g in gaps) + f" m (mean {gaps.mean():.1f} m)")
    # axis positions at a common height near the deck (removes the effect of rake x coverage)
    for tag, T in (("UAV", dt), ("TLS", lt)):
        ys = []
        for t in T:
            if len(t) < MIN_LASER_POINTS:
                ys.append(np.nan)
                continue
            c0, d0, _ = pca_axis(t)
            c, d = irls_axis(t, c0, d0)
            ys.append(c[1] + (5.0 - c[2]) / d[2] * d[1])
        ys = np.asarray(ys)
        put("4.10", f"{tag} axis position at z = 5 m, spacing [m]",
            [None if not np.isfinite(g) else round(float(g), 2) for g in np.diff(ys)])
        LEDGER["4.10"][f"{tag}_axis_y_at_5m"] = [None if not np.isfinite(y) else float(y) for y in ys]
    yu = np.array(LEDGER["4.10"]["UAV_axis_y_at_5m"], dtype=float)
    yt = np.array(LEDGER["4.10"]["TLS_axis_y_at_5m"], dtype=float)
    span_u, span_t = yu[4] - yu[1], yt[4] - yt[1]
    put("4.10", "Mainmast-to-Jiggermast distance UAV / TLS [m], ratio",
        [round(float(span_u), 3), round(float(span_t), 3), round(float(span_u / span_t), 4)],
        f"UAV/TLS = {span_u / span_t:.4f} ({(span_u / span_t - 1) * 100:+.2f}%)")
    dia_t = {NAMES[i]: rl[i].median_diameter_mm for i in usable}
    put("4.10", "median diameter UAV / TLS [mm]", {k: [round(res[NAMES.index(k)].median_diameter_mm), round(v)] for k, v in dia_t.items()})
    # TLS angular coverage per mast (fraction of 12 sectors, median over slices)
    covs = {}
    for i in usable:
        P = robust_clean(lt[i])
        c0, d0, _ = pca_axis(P)
        c, dd = irls_axis(P, c0, d0)
        B, dd = orthonormal_basis(dd)
        t = (P - c) @ dd
        xy = (P - c) @ B.T
        edges = np.linspace(t.min(), t.max(), args.slices + 1)
        cv_ = []
        for k in range(args.slices):
            msk = (t >= edges[k]) & (t < edges[k + 1])
            if msk.sum() < 40:
                continue
            q = xy[msk]
            c1 = np.median(q, axis=0)
            a_ = np.degrees(np.arctan2(q[:, 1] - c1[1], q[:, 0] - c1[0]))
            cv_.append(len(set(((a_ + 180) // 30).astype(int))) / 12 * 100)
        covs[NAMES[i]] = float(np.median(cv_))
    put("4.10", "TLS angular coverage, median over slices [%]", {k: round(v) for k, v in covs.items()})
    full_cov = [k for k, v in covs.items() if v >= 99]
    excess = {k: (res[NAMES.index(k)].median_diameter_mm - dia_t[k]) for k in full_cov}
    put("4.10", "UAV minus TLS diameter where TLS covers the full ring [mm, %]",
        {k: [round(v), round(100 * v / dia_t[k], 1)] for k, v in excess.items()},
        ", ".join(f"{k} +{100 * v / dia_t[k]:.0f}% (radius +{v / 2:.0f} mm)" for k, v in excess.items()))

    # ------------------------------------------------------------------ frame check
    print("\nFRAME CHECK: component labels")
    ang_b0 = []
    for P in drone:
        c0, d0, _ = pca_axis(P)
        _, dd = irls_axis(P, c0, d0)
        B, _ = orthonormal_basis(dd)
        b0 = B[0][:2] / np.linalg.norm(B[0][:2])
        ang_b0.append(np.degrees(np.arccos(abs(b0 @ fd.transverse))))
    put("frame", "angle between the 'transverse' bend axis and the ship's transverse axis [deg]",
        [round(a, 2) for a in ang_b0], f"≤ {max(ang_b0):.1f}°")

    out = os.path.join(RESDIR, "paper_numbers.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(LEDGER, f, indent=1, ensure_ascii=False, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
