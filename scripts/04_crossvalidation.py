#!/usr/bin/env python3
"""Independent cross-validation against the terrestrial laser scan (Section 4.8, Figures 14 and 15).

Both clouds are co-registered into the common ship frame and processed by the
identical pipeline. The profiles are then paired at the same height over the
height range covered by both sensors and re-referenced to one straight line over
that range (``mastfit.crossval``), so that both estimate the same quantity.

Reproduces (four masts; the Foremast has only 1,930 TLS points):
    transverse   : RMS difference 14 mm, r = +0.65, 95% LoA +-28 mm (CI +-18 to +-36)
    longitudinal : RMS difference 33 mm, r = +0.44, 95% LoA +-65 mm (CI +-49 to +-77)

The Bland-Altman bias is zero by construction (both re-referenced profiles have
zero mean), so it is printed for completeness only.
"""
import numpy as np

from _common import base_parser, label_of, load_drone, load_laser
from mastfit.crossval import cross_validate
from mastfit.io import coregister, robust_clean
from mastfit.pipeline import fit_mast

MIN_LASER_POINTS = 2000


def main():
    args = base_parser(__doc__).parse_args()
    drone, laser = load_drone(args), load_laser(args)
    dt, lt = coregister(drone, laser)

    print("laser coverage per mast:")
    usable = []
    for i, P in enumerate(laser):
        ok = len(P) >= MIN_LASER_POINTS
        print(f"  {label_of(i):<14}{len(P):>9,} pts  {'usable' if ok else 'TOO SPARSE - excluded'}")
        if ok:
            usable.append(i)

    dres = [fit_mast(dt[i], name=label_of(i), n_slices=args.slices, bandwidth_m=args.bandwidth)
            for i in range(5)]
    lres = [fit_mast(robust_clean(lt[i]), name=label_of(i), n_slices=args.slices,
                     bandwidth_m=args.bandwidth) if i in usable else None for i in range(5)]
    cv = cross_validate(dres, lres, usable)

    print("\ncommon height span and peak total bend over it (mm):")
    print(f"{'Mast':<14}{'span [m]':>14}{'UAV':>8}{'TLS':>8}")
    for i in usable:
        a, b = cv["span"][i]
        u, t = cv["peak_total"][i]
        print(f"{label_of(i):<14}{f'{a:.1f}-{b:.1f}':>14}{u:>8.0f}{t:>8.0f}")

    print("\nBland-Altman, pooled per-height (usable masts), UAV - TLS:")
    for comp in ("transverse", "longitudinal"):
        s = cv[comp]
        print(f"  {comp:<13} RMS diff {s['rms_diff_mm']:4.0f} mm | corr {s['corr']:+.2f} | "
              f"LoA +-{s['loa_mm']:3.0f} mm (mast-block 95% CI +-{s['loa_ci_mm'][0]:.0f} to "
              f"+-{s['loa_ci_mm'][1]:.0f}) | bias {s['bias_mm']:+.1f} mm (zero by construction)")
        print(f"  {'':<13} lag-1 r = {s['lag1']:.2f}, n = {s['n']} -> effective n ~ {s['n_effective']:.1f}")
    print("\nThe RMS differences are of the order of the phantom-derived error floor, which the")
    print("comparison therefore corroborates. The independent unit is the mast (four).")


if __name__ == "__main__":
    main()
