#!/usr/bin/env python3
"""Is the circle-fit centre an unbiased estimate of the centreline? (Section 4.14)

A  Real masts: centrelines and bends with slice centres from the trimmed circle fit and from
   a direct ellipse fit to the same inliers.
B  Phantoms with a known straight-plus-bend axis whose cross-sections are modified:
   uniform radial offset (reconstructed surface outside the tube), ovality with fixed or
   rotating orientation, missing angular sectors, and a first-harmonic (one-sided) radial
   bias that is constant or varies with height (a one-sided radial bias b cos(theta) moves
   the fitted centre by b). Bending-profile RMSE with circle and ellipse centres.

Writes results/section_model.json. Usage: python scripts/19_section_model.py --seeds 5
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from _common import base_parser, load_drone  # noqa: E402
from mastfit.io import MAST_NAMES  # noqa: E402
from mastfit.pipeline import fit_mast  # noqa: E402
from mastfit.validate import compare_to_truth  # noqa: E402

NAMES = [MAST_NAMES[i][1] for i in range(5)]
H, R0, TAPER, RAKE, AMP = 50.0, 0.50, 0.006, 2.0, 0.10


def phantom(seed, offset=0.0, ovality=0.0, rotate=False, gaps=0.0, h1=0.0, h1_vary=False,
            noise=0.02, ppm=380):
    rg = np.random.default_rng(seed)
    n = int(H * ppm)
    z = rg.uniform(0, H, n)
    th = rg.uniform(0, 2 * np.pi, n)
    r = R0 - TAPER * z + offset
    if ovality > 0:
        q = ovality / (2 - ovality)               # 1 - rmin/rmax = ovality
        phi = 2 * np.pi * z / H if rotate else 0.3
        r = r * (1 + q * np.cos(2 * (th - phi)))
    if h1 > 0:
        amp = h1 * np.sin(2 * np.pi * z / H) if h1_vary else h1   # S-shaped along the height
        r = r + amp * np.cos(th)                   # one-sided bias toward +x
    bend = AMP * np.sin(np.pi * z / H)
    P = np.column_stack([bend + r * np.cos(th), np.tan(np.radians(RAKE)) * z + r * np.sin(th), z])
    P += rg.normal(0, noise, P.shape)
    if gaps > 0:                                   # occluded sectors in a share of 1 m bands
        band = np.floor(z).astype(int)
        hit = rg.random(int(H) + 1) < gaps
        centre = rg.uniform(0, 2 * np.pi, int(H) + 1)
        d = np.angle(np.exp(1j * (th - centre[band])))
        P = P[~(hit[band] & (np.abs(d) < np.radians(30)))]
    zt = np.linspace(0, H, 400)
    return P, zt, AMP * np.sin(np.pi * zt / H) * 1000.0


CASES = [
    ("round sections (reference)", {}),
    ("uniform radial offset +50 mm", {"offset": 0.05}),
    ("ovality 10%, fixed orientation", {"ovality": 0.10}),
    ("ovality 25%, orientation rotating with height", {"ovality": 0.25, "rotate": True}),
    ("ovality 25% + 60 deg gaps in 30% of 1 m bands", {"ovality": 0.25, "gaps": 0.3}),
    ("one-sided radial bias 20 mm, constant", {"h1": 0.02}),
    ("one-sided radial bias 20 mm, varying with height (S-shaped)", {"h1": 0.02, "h1_vary": True}),
    ("combined: offset 50 mm, ovality 25% rotating, gaps 30%",
     {"offset": 0.05, "ovality": 0.25, "rotate": True, "gaps": 0.3}),
]


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    out = {"A_real": {}, "B_phantom": {}}
    drone = load_drone(a)
    print("A. REAL MASTS: circle vs ellipse centres")
    for i, P in enumerate(drone):
        rc, re_ = fit_mast(P), fit_mast(P, section_model="ellipse")
        dx = np.asarray(re_.bend_transverse_mm) - np.asarray(rc.bend_transverse_mm)
        dy = np.asarray(re_.bend_longitudinal_mm) - np.asarray(rc.bend_longitudinal_mm)
        row = {"bend_circle_mm": round(rc.max_bend_mm, 1), "bend_ellipse_mm": round(re_.max_bend_mm, 1),
               "profile_rms_difference_mm": round(float(np.sqrt(np.mean(dx ** 2 + dy ** 2))), 1),
               "profile_max_difference_mm": round(float(np.max(np.hypot(dx, dy))), 1)}
        out["A_real"][NAMES[i]] = row
        print(f"  {NAMES[i]}: {row}", flush=True)
    print("\nB. PHANTOMS: bending-profile RMSE [mm], circle / ellipse centres")
    for label, kw in CASES:
        ec, ee = [], []
        for sd in range(a.seeds):
            P, zt, bt = phantom(sd, **kw)
            ec.append(compare_to_truth(P, zt, bt)[3])
            ee.append(compare_to_truth(P, zt, bt, section_model="ellipse")[3])
        row = {"circle_mm": round(float(np.mean(ec)), 1), "ellipse_mm": round(float(np.mean(ee)), 1)}
        out["B_phantom"][label] = row
        print(f"  {label}: {row}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/section_model.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
