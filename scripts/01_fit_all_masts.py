#!/usr/bin/env python3
"""Fit all five masts and print Table 1 of the paper.

Reproduces:
    Foremast     rake 1.38  diam  678  bend  61 / 148 -> 160 mm
    Mainmast     rake 1.86  diam  704  bend  82 /  48 ->  89 mm
    Middle Mast  rake 1.73  diam  749  bend  30 /  34 ->  45 mm
    Mizzenmast   rake 1.45  diam 1030  bend  35 /  28 ->  36 mm
    Jiggermast   rake 1.20  diam  959  bend  37 /  60 ->  70 mm

The diameters describe the reconstructed UAV surface: where the TLS survey covers
most of the circumference they exceed the TLS diameters by 8-13% (Section 4.10).

Significance is not judged here: see scripts/15_significance_and_ablations.py
(section B), which tests each bend against the null distribution of the total bend
of straight synthetic masts (Table 5).
"""
import csv, os
import numpy as np
from _common import base_parser, ensure_dirs, label_of, load_drone, RESDIR
from mastfit.pipeline import fit_mast


def main():
    args = base_parser(__doc__).parse_args()
    ensure_dirs(RESDIR)
    masts = load_drone(args)

    rows = []
    print(f"{'Mast':<14}{'Points':>8}{'Rake':>7}{'Diam':>7}"
          f"{'Trans':>7}{'Long':>7}{'Total':>7}{'sigma':>7}")
    print("-" * 67)
    for i, P in enumerate(masts):
        r = fit_mast(P, name=label_of(i), n_slices=args.slices, bandwidth_m=args.bandwidth)
        tr = float(np.nanmax(np.abs(r.bend_transverse_mm)))
        lo = float(np.nanmax(np.abs(r.bend_longitudinal_mm)))
        tot = r.max_bend_mm
        print(f"{r.name:<14}{r.n_points:>8,}{r.rake_deg:>7.2f}{r.median_diameter_mm:>7.0f}"
              f"{tr:>7.0f}{lo:>7.0f}{tot:>7.0f}{r.median_sigma_mm:>7.1f}")
        rows.append(dict(mast=r.name, points=r.n_points, rake_deg=round(r.rake_deg, 2),
                         diameter_mm=round(r.median_diameter_mm), transverse_mm=round(tr),
                         longitudinal_mm=round(lo), total_mm=round(tot),
                         sigma_stat_mm=round(r.median_sigma_mm, 1),
                         slices_kept=r.n_slices_kept, slices_total=r.n_slices_total))

    out = os.path.join(RESDIR, "table1.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nsigma is the random slice-centre uncertainty; the limiting term is the correlated")
    print("reconstruction error (per-mast floors: scripts/10_per_mast_accuracy.py). Significance:")
    print("scripts/15_significance_and_ablations.py --sections B.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
