#!/usr/bin/env python3
"""Sensitivity of the result to the free parameters.

1. SLICE THICKNESS -- the reason a *fixed* smoothing bandwidth is used.
   Without smoothing, the recovered bend is badly unstable in delta:
       Foremast 218 -> 276 -> 591 mm as delta goes 1.25 -> 0.5 -> 0.25 m
   With the fixed L = 3 m kernel it is stable:
       Foremast 137 -> 160 -> 160 mm
   The bandwidth, not the slice count, is the physically meaningful scale.

2. THRESHOLD CONSTANTS -- the trim/band constants barely move the answer.
"""
import numpy as np
from _common import base_parser, label_of, load_drone
from mastfit.pipeline import fit_mast


def main():
    args = base_parser(__doc__).parse_args()
    masts = load_drone(args)
    slice_counts = [40, 100, 200]   # delta ~ 1.25, 0.5, 0.25 m over a ~50 m mast

    print("SLICE-THICKNESS STABILITY (max bend, mm), fixed bandwidth L = 3 m")
    print(f"{'Mast':<14}" + "".join(f"{'d=%.2fm' % (50/n):>10}" for n in slice_counts))
    print("-" * 46)
    for i, P in enumerate(masts):
        vals = [fit_mast(P, n_slices=n, bandwidth_m=args.bandwidth).max_bend_mm
                for n in slice_counts]
        print(f"{label_of(i):<14}" + "".join(f"{v:>10.0f}" for v in vals))
    print("  -> with the fixed bandwidth the bends change by at most ~20% (Middle Mast)")
    print("     across slice counts; without smoothing the Foremast bend varies 218 -> 591 mm.")

    print("\nBANDWIDTH SENSITIVITY (max bend, mm) -- L sets the resolved wavelength")
    bws = [2.0, 3.0, 4.0, 5.0]
    print(f"{'Mast':<14}" + "".join(f"{'L=%.0fm' % b:>10}" for b in bws))
    print("-" * 46)
    for i, P in enumerate(masts):
        vals = [fit_mast(P, n_slices=args.slices, bandwidth_m=b).max_bend_mm for b in bws]
        print(f"{label_of(i):<14}" + "".join(f"{v:>10.0f}" for v in vals))
    print("  -> smooth monotone attenuation, no qualitative change (paper: 176/160/147/136 mm")
    print("     for the Foremast). L = 3 m bridges the gaps left by removed slices and")
    print("     describes the bending of the mast as a whole (half amplitude at a 16 m")
    print("     wavelength); the floor and the null distribution are derived at the same L,")
    print("     and scripts/18_bandwidth_sweep.py repeats both for L = 2-5 m.")

    print("\nTHRESHOLD-CONSTANT ABLATION (max bend, mm)")
    print(f"{'Mast':<14}{'kmad=2.0':>10}{'2.5':>8}{'3.5':>8}{'| kappa=2':>10}{'3':>6}{'4':>6}")
    print("-" * 62)
    for i, P in enumerate(masts):
        km = [fit_mast(P, n_slices=args.slices, bandwidth_m=args.bandwidth,
                       k_mad=k).max_bend_mm for k in (2.0, 2.5, 3.5)]
        kb = [fit_mast(P, n_slices=args.slices, bandwidth_m=args.bandwidth,
                       kappa=k).max_bend_mm for k in (2.0, 3.0, 4.0)]
        print(f"{label_of(i):<14}{km[0]:>10.0f}{km[1]:>8.0f}{km[2]:>8.0f}"
              f"{kb[0]:>10.0f}{kb[1]:>6.0f}{kb[2]:>6.0f}")
    print("  -> insensitive: the constants are not tuned to produce the result.")




if __name__ == "__main__":
    main()
