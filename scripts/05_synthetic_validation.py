#!/usr/bin/env python3
"""Synthetic validation: accuracy, the error model, and the doming test (Section 4.5, Figure 10).

Error model at equal magnitude (20 mm RMS, 8 seeds):

    white (independent) noise             -> RMSE ~= 6 mm (6-7)
    spatially-correlated error, same RMS  -> RMSE ~= 8.5 mm (5-13)

At equal magnitude the penalty of correlated error is modest; it grows with the
amplitude of the correlated component. The OPERATIVE accuracy of the method is the
per-mast floor of the matched phantoms (correlated field of 45 mm RMS): 20-27 mm (mean)
and 29-40 mm (90th percentile), from scripts/10_per_mast_accuracy.py (Sections 4.6 and
4.7; its sensitivity to the error model: scripts/14_robustness_checks.py), against
~7 mm for the same phantoms with independent noise.

Also reproduces:
  * flare mechanism on a phantom with known truth (true 94 mm): the unsmoothed
    baseline returns 127 mm without and 86 mm with the flare test; the
    kernel-smoothed pipeline is barely affected (74 / 70 mm);
  * five-mast doming recovery: an injected 49% shared fraction is recovered as
    51%, with the independent per-mast bends recovered to 10 mm (mean abs error).

Every experiment function returns its numbers, so ``scripts/12_paper_numbers.py``
and ``scripts/13_paper_figures.py`` use exactly the values printed here.
"""
import numpy as np
from _common import base_parser
from mastfit.doming import shared_fraction
from mastfit.baseline import naive_pipeline
from mastfit.pipeline import fit_mast
from mastfit.validate import compare_to_truth, detrend, make_phantom, recovered_profile

N_SEEDS = 8


def experiment_error_model():
    print("=" * 68)
    print("1. ERROR MODEL: white vs spatially-correlated noise (same RMS)")
    print("=" * 68)
    out = {}
    for label, corr in [("white / independent", False), ("correlated (SfM-like)", True)]:
        errs = []
        for sd in range(N_SEEDS):
            P, zt, bt = make_phantom(mode="C", amplitude_m=0.15, noise_m=0.02,
                                     clutter_frac=0.03, base_flare=True,
                                     correlated=corr, seed=sd)
            errs.append(compare_to_truth(P, zt, bt)[3])
        print(f"  {label:<24} RMSE {np.mean(errs):5.1f} mm "
              f"(range {min(errs):.0f}-{max(errs):.0f} over {N_SEEDS} seeds)")
        out["correlated" if corr else "white"] = np.asarray(errs)
    print()
    print("  At equal magnitude, correlated error costs some accuracy, and its spread is")
    print("  wide because the outcome depends on the realised error field. The operative")
    print("  floor, with a correlated field matched to the data, is quoted per mast")
    print("  (script 10) and is about three times the white-noise figure.")
    return out


def experiment_noise_clutter_sweep():
    print("\n" + "=" * 68)
    print("2. ROBUSTNESS: RMSE vs surface noise and one-sided contamination")
    print("=" * 68)
    noises = [0.005, 0.01, 0.02, 0.03, 0.05]
    print("  clutter |" + "".join(f"{n*1000:>7.0f}mm" for n in noises))
    for cf in [0.0, 0.02, 0.05, 0.10]:
        row = []
        for ns in noises:
            errs = [compare_to_truth(*make_phantom(mode="C", amplitude_m=0.15, noise_m=ns,
                                                   clutter_frac=cf, base_flare=True, seed=sd))[3]
                    for sd in range(N_SEEDS)]
            row.append(np.mean(errs))
        print(f"  {cf*100:5.0f}%  |" + "".join(f"{v:>9.1f}" for v in row))
    print("  (RMSE in mm against a true peak bend of ~110 mm)")


def experiment_flare():
    print("\n" + "=" * 68)
    print("3. FLARE MECHANISM on a phantom with known truth")
    print("=" * 68)
    out = {}
    for mode in ("C", "lean"):
        P, zt, bt = make_phantom(mode=mode, amplitude_m=0.15, noise_m=0.02,
                                 clutter_frac=0.03, base_flare=True, seed=3)
        truth = float(np.max(np.abs(detrend(zt, bt))))
        naive_off = naive_pipeline(P, quality_control=False)
        naive_on = naive_pipeline(P, quality_control=True)
        off = fit_mast(P, apply_quality_control=False).max_bend_mm
        on = fit_mast(P, apply_quality_control=True).max_bend_mm
        out[mode] = {"truth": truth, "baseline_off": naive_off, "baseline_on": naive_on,
                     "smoothed_off": off, "smoothed_on": on}
        print(f"  mode {mode:<5} truth {truth:5.0f} | unsmoothed baseline: flare test off "
              f"{naive_off:5.0f}, on {naive_on:5.0f} | kernel-smoothed: off {off:5.0f}, on {on:5.0f} mm")
    print("  -> without the flare test the unsmoothed baseline inflates the bend; the")
    print("     kernel-smoothed pipeline masks the effect (Section 4.2).")
    return out


def experiment_doming():
    print("\n" + "=" * 68)
    print("4. FIVE-MAST DOMING TEST on synthetic data with a KNOWN common mode")
    print("=" * 68)
    amp_common = 0.06
    common = lambda zf: amp_common * np.sin(np.pi * zf)
    modes = ["C", "lean", "S", "C", "lean"]
    amps = [0.12, 0.10, 0.14, 0.08, 0.11]
    g = np.linspace(0.05, 0.95, 36)
    REC, INJ = [], []
    for i in range(5):
        P, _, _ = make_phantom(mode=modes[i], amplitude_m=amps[i], noise_m=0.02,
                               clutter_frac=0.03, base_flare=True,
                               common_mode=common, seed=10 + i)
        h, b = recovered_profile(P)            # physical heights of the retained span
        # as in the real-data test, the profile is normalized over its own retained span;
        # the injected truth is evaluated at the same physical heights (phantom height 50 m)
        REC.append(np.interp(g, (h - h[0]) / (h[-1] - h[0]), b))
        zf = (h[0] + g * (h[-1] - h[0])) / 50.0
        indep = (amps[i] * np.sin(np.pi * zf) if modes[i] == "C" else
                 amps[i] * zf if modes[i] == "lean" else
                 amps[i] * np.sin(2 * np.pi * zf))
        INJ.append((indep + common(zf)) * 1000)
    REC = np.array([detrend(g, x) for x in REC])
    INJ = np.array([detrend(g, x) for x in INJ])
    for i in range(5):
        if np.corrcoef(REC[i], INJ[i])[0, 1] < 0:
            REC[i] = -REC[i]
    f_inj, s_inj = shared_fraction(INJ)
    f_rec, s_rec = shared_fraction(REC)
    resid_err = [abs(np.max(np.abs(REC[i] - s_rec)) - np.max(np.abs(INJ[i] - s_inj)))
                 for i in range(5)]
    print(f"  injected shared fraction  : {f_inj*100:.0f}%")
    print(f"  recovered shared fraction : {f_rec*100:.0f}%")
    print(f"  per-mast independent bend recovered to {np.mean(resid_err):.1f} mm (mean abs err)")
    print("  -> the test recovers the common mode approximately.")
    return {"grid": g, "injected": INJ, "recovered": REC, "shared_injected": s_inj,
            "shared_recovered": s_rec, "fraction_injected": f_inj,
            "fraction_recovered": f_rec, "per_mast_error_mm": float(np.mean(resid_err))}


def main():
    base_parser(__doc__).parse_args()
    experiment_error_model()
    experiment_noise_clutter_sweep()
    experiment_flare()
    experiment_doming()


if __name__ == "__main__":
    main()
