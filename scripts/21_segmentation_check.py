#!/usr/bin/env python3
"""Validation of the automatic UAV mast segmentation (Section 3.1).

No manual reference segmentation of the UAV vessel cloud exists, so the segmented masts are
checked against what can be measured:

A  Completeness along the mast: largest gap between consecutive points along the robust axis,
   share of 0.5 m bands holding points; and the vertical extent of each UAV segment against the
   independently segmented TLS cloud of the same mast (four masts with usable TLS coverage), in
   the common ship frame after the TLS clouds are rotated by the common tilt (as in script 20).
   The upper and lower ends are the highest and lowest 0.25 m bands that hold at least five
   points, a definition that is largely insensitive to the point density (which differs by a
   factor of about 50 between the TLS masts). Each cloud has its own height datum (Section 3.1),
   so the datum-free segment lengths are the cleaner comparison.
B  Correctness: radial residual of every point from the fitted surface (trimmed circle centres
   and radii of the slices the pipeline retains, interpolated along the axis). Points far
   outside the surface include material that the segmentation retained but that is not mast
   (deck around the foot, rigging and fittings close to the mast, yard attachments); points far
   inside cannot be mast surface and show the scatter of the reconstruction. Shares beyond
   +-0.15, 0.3 and 0.5 m, where the far points lie, and the largest outward residual.
C  Effect on the result: bends after removing all points more than 0.5 m and 0.3 m outside the
   fitted surface.

Writes results/segmentation_check.json and, with --figure, figures/segmentation_check.png
(the segmented masts coloured by radial residual).
Usage: python scripts/21_segmentation_check.py [--figure]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from _common import base_parser, load_drone, load_laser  # noqa: E402
from mastfit.geometry import orthonormal_basis, robust_circle  # noqa: E402
from mastfit.io import MAST_NAMES, coregister, robust_clean  # noqa: E402
from mastfit.pipeline import fit_mast  # noqa: E402

NAMES = [MAST_NAMES[i][1] for i in range(5)]
MIN_LASER_POINTS = 5000


def surface_residual(P, c, d, exclude=(), n_slices=100, min_points=40):
    """Axial coordinate, in-plane coordinates and radial residual of every point (m), about the
    axis (c, d) of the pipeline and with its slicing. The reference surface is interpolated from
    the trimmed circle fits of the slices that the pipeline retains (slices in `exclude`, those
    removed by the flare and band tests, do not enter it), so contamination in removed slices is
    measured against the clean surface rather than absorbed by it."""
    B, d = orthonormal_basis(d)
    t = (P - c) @ d
    xy = (P - c) @ B.T
    edges = np.linspace(t.min(), t.max(), n_slices + 1)
    tk, ck, rk = [], [], []
    for k in range(n_slices):
        m = (t >= edges[k]) & ((t < edges[k + 1]) if k < n_slices - 1 else (t <= edges[k + 1]))
        if m.sum() < min_points or k in exclude:
            continue
        cen, r, *_ = robust_circle(xy[m])
        tk.append(0.5 * (edges[k] + edges[k + 1]))
        ck.append(cen)
        rk.append(r)
    tk, ck, rk = np.asarray(tk), np.asarray(ck), np.asarray(rk)
    cu, cv, rr = (np.interp(t, tk, ck[:, 0]), np.interp(t, tk, ck[:, 1]), np.interp(t, tk, rk))
    rho = np.hypot(xy[:, 0] - cu, xy[:, 1] - cv) - rr
    return t, xy, rho


def removed_slices(P, r, n_slices=100):
    """Slice index of every point and the indices of the slices that the base-flare and the
    rigging-band tests remove (the slicing of fit_mast, about its axis)."""
    t = (P - r.axis_centroid) @ r.axis_direction
    edges = np.linspace(t.min(), t.max(), n_slices + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_slices - 1)
    hs = r.all_slice_height                       # accepted slices, measured from the lowest one
    for j in range(n_slices):                     # locate the lowest accepted slice
        idx = np.round((hs + cen[j] - cen[0]) / (cen[1] - cen[0])).astype(int)
        if idx.max() < n_slices and np.allclose(cen[idx], hs + cen[j], atol=1e-6):
            break
    flare = idx[r.flare_removed]
    band = idx[r.band_flagged & ~r.flare_removed]
    return t, k, edges, flare, band


def ends(z, n_min=5, w=0.25):
    b = np.arange(np.floor(z.min()), np.ceil(z.max()) + w, w)
    cnt, _ = np.histogram(z, b)
    ok = np.where(cnt >= n_min)[0]
    return float(b[ok[0]]), float(b[ok[-1] + 1])


def rotated_tls(dt, lt, usable):
    """TLS clouds rotated by the common tilt and re-centred, as in scripts/20."""
    rd = {i: fit_mast(dt[i]) for i in usable}
    rl = {i: fit_mast(robust_clean(lt[i])) for i in usable}
    A = np.array([rd[i].axis_direction * np.sign(rd[i].axis_direction[2]) for i in usable])
    B = np.array([rl[i].axis_direction * np.sign(rl[i].axis_direction[2]) for i in usable])
    tilt = (B[:, :2] / B[:, 2:3] - A[:, :2] / A[:, 2:3]).mean(axis=0)
    v = np.array([-tilt[0], -tilt[1], 1.0])
    v /= np.linalg.norm(v)
    k = np.cross([0, 0, 1.0], v)
    sn = np.linalg.norm(k)
    k /= sn
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Rot = np.eye(3) + sn * Kx + (1 - v[2]) * Kx @ Kx
    lt_rot = [m @ Rot.T for m in lt]
    cen = np.mean([[np.median(m[:, 0]), np.median(m[:, 1])] for m in lt_rot], axis=0)
    zb = float(np.percentile(np.concatenate([m[:, 2] for m in lt_rot]), 1))
    return [m - np.array([cen[0], cen[1], zb]) for m in lt_rot]


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    drone, laser = load_drone(a), load_laser(a)
    dt, lt = coregister(drone, laser)
    usable = [i for i in range(5) if len(laser[i]) >= MIN_LASER_POINTS]
    lt2 = rotated_tls(dt, lt, usable)
    out, keep_for_fig = {}, {}
    for i in range(5):
        P = dt[i]
        res = fit_mast(P)
        b0 = res.max_bend_mm
        _, k, edges, flare, band = removed_slices(P, res)
        t, xy, rho = surface_residual(P, res.axis_centroid, res.axis_direction,
                                      exclude=set(flare.tolist()) | set(band.tolist()))
        far = rho > 0.3
        ts = np.sort(t)
        bands = np.histogram(t, np.arange(t.min(), t.max() + 0.5, 0.5))[0]
        row = {
            "points": int(len(P)),
            "axial_length_m": round(float(np.ptp(t)), 2),
            "largest_axial_gap_m": round(float(np.max(np.diff(ts))), 2),
            "bands_0p5m_with_points_pct": round(100 * float(np.mean(bands > 0)), 1),
            "outside_0p15_pct": round(100 * float(np.mean(rho > 0.15)), 2),
            "inside_0p15_pct": round(100 * float(np.mean(rho < -0.15)), 2),
            "outside_0p3_pct": round(100 * float(np.mean(rho > 0.3)), 2),
            "inside_0p3_pct": round(100 * float(np.mean(rho < -0.3)), 2),
            "outside_0p5_pct": round(100 * float(np.mean(rho > 0.5)), 2),
            "max_outward_m": round(float(rho.max()), 2),
            "outside_0p3_in_lowest_2m_pct": round(100 * float(np.mean((t - t.min() < 2.0)[far])), 1),
            "outside_0p3_in_band_slices_pct": round(100 * float(np.mean(np.isin(k, band)[far])), 1),
            "outside_0p3_in_flare_slices_pct": round(100 * float(np.mean(np.isin(k, flare)[far])), 1),
            "outside_0p3_at_foot_or_removed_slices_pct": round(100 * float(np.mean(
                ((t - t.min() < 2.0) | np.isin(k, flare) | np.isin(k, band))[far])), 1),
            "bend_mm": round(b0, 1),
            "bend_without_outside_0p5_mm": round(fit_mast(P[rho <= 0.5]).max_bend_mm, 1),
            "bend_without_outside_0p3_mm": round(fit_mast(P[rho <= 0.3]).max_bend_mm, 1),
        }
        if i in usable:
            u, q = ends(P[:, 2]), ends(lt2[i][:, 2])
            row.update({"uav_ends_m": [round(u[0], 2), round(u[1], 2)],
                        "tls_ends_m": [round(q[0], 2), round(q[1], 2)],
                        "length_difference_m": round((u[1] - u[0]) - (q[1] - q[0]), 2),
                        "lower_end_difference_m": round(u[0] - q[0], 2),
                        "upper_end_difference_m": round(u[1] - q[1], 2)})
        out[NAMES[i]] = row
        keep_for_fig[i] = (t, xy, rho, edges, flare, band)
        print(f"{NAMES[i]}: {row}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/segmentation_check.json", "w") as f:
        json.dump(out, f, indent=1)
    if a.figure:
        figure(keep_for_fig, a.outdir)


def figure(data, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(11, 6.2), sharey=True)
    classes = [(-np.inf, 0.15, "#b8b8b8", "within 0.15 m of the fitted surface"),
               (0.15, 0.3, "#e0a030", "0.15–0.3 m outside"),
               (0.3, np.inf, "#c0392b", "more than 0.3 m outside")]
    for ax, i in zip(axes, range(5)):
        t, xy, rho, edges, flare, band = data[i]
        z = t - t.min()
        for j in np.concatenate([flare, band]).astype(int):
            ax.axhspan(edges[j] - t.min(), edges[j + 1] - t.min(), color="#dfe8f3", lw=0, zorder=0,
                       label="slice removed by the flare or band test" if j == (list(flare) + list(band))[0] else None)
        for lo, hi, col, lab in classes:
            m = (rho > lo) & (rho <= hi)
            ax.scatter(xy[m, 0], z[m], s=1.2 if lo < 0 else 3, c=col, lw=0, label=lab,
                       rasterized=True, zorder=2 if lo < 0 else 3)
        ax.set_title(NAMES[i], fontsize=10)
        ax.set_xlim(-1.6, 1.6)
        ax.set_xlabel("across axis (m)", fontsize=8)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("height along axis (m)", fontsize=9)
    hl = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(lab, h)
    order = [c[3] for c in classes] + ["slice removed by the flare or band test"]
    fig.legend([hl[o] for o in order if o in hl], [o for o in order if o in hl], loc="lower center",
               ncol=4, frameon=False, fontsize=9, markerscale=5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "segmentation_check.png"), dpi=200)
    print("figure ->", os.path.join(outdir, "segmentation_check.png"))


if __name__ == "__main__":
    main()
