#!/usr/bin/env python3
"""Five-mast shared-curvature (candidate doming) test.

Reproduces: shared transverse ~28%, shared longitudinal ~14%.

The shared component is a *candidate* reconstruction artefact, not a proven one:
a common structural or environmental cause would look identical. The corrected
bends are therefore a conservative lower bound.
"""
import numpy as np
from _common import base_parser, ensure_dirs, label_of, load_drone, RESDIR
from mastfit.doming import doming_decomposition
from mastfit.pipeline import fit_mast


def main():
    args = base_parser(__doc__).parse_args()
    ensure_dirs(RESDIR)
    masts = load_drone(args)
    results = [fit_mast(P, name=label_of(i), n_slices=args.slices,
                        bandwidth_m=args.bandwidth) for i, P in enumerate(masts)]
    d = doming_decomposition(results)

    print(f"shared transverse   : {d['shared_fraction_transverse']*100:.0f}%")
    print(f"shared longitudinal : {d['shared_fraction_longitudinal']*100:.0f}%")
    print("\nPer-mast peak bend, raw -> doming-corrected (mm):")
    for i, r in enumerate(results):
        tr_raw = np.nanmax(np.abs(d["transverse"][i]))
        tr_cor = np.nanmax(np.abs(d["corrected_transverse"][i]))
        lo_raw = np.nanmax(np.abs(d["longitudinal"][i]))
        lo_cor = np.nanmax(np.abs(d["corrected_longitudinal"][i]))
        print(f"  {r.name:<14} transverse {tr_raw:5.0f} -> {tr_cor:5.0f} | "
              f"longitudinal {lo_raw:5.0f} -> {lo_cor:5.0f}")

    print("\nNOTE: bow-stern (longitudinal) bending is mostly mast-specific and is the")
    print("more trustworthy component; a larger share of athwartship bending is common-mode.")
    np.savez(f"{RESDIR}/doming.npz", **{k: v for k, v in d.items() if isinstance(v, np.ndarray)})


if __name__ == "__main__":
    main()
