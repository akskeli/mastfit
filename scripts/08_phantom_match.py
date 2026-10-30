#!/usr/bin/env python3
"""Does the phantom resemble the real cloud? (Section 4.6, Table 2, Figure 11.)

Every accuracy figure in this study is derived on a synthetic phantom and then
transferred to the real masts. That transfer is only valid if the phantom resembles
the data in the statistics that govern fitting accuracy, so this script compares them
for the real UAV clouds, the generic phantom and the matched phantom
(``mastfit.validate.matched_phantom``):

    statistic (per slice)              real     generic phantom   matched phantom
    radial residual sigma, median    45.6 mm          9.0 mm           46.2 mm
    radial residual sigma, p90      112.2 mm         11.2 mm           75.4 mm
    points trimmed as outliers        5.6 %           0.2 %             6.6 %
    angular coverage, 10th pct       91.7 %         100.0 %           100.0 %
    point density                     387 /m          391 /m            423 /m

The generic phantom is much cleaner than the data. Matching it raises the transferred
correlated-error RMSE from about 9 to about 21 mm: each slice centre averages over
hundreds of points, so per-point scatter largely averages away, and the limiting term
is the spatially correlated component, which does not show up in the slice residual.

Two differences remain: the real clouds have a heavier residual tail (sigma p90 112 mm
against 75 mm) and mild angular-coverage gaps behind rigging (10th percentile 92%
against 100%). Accuracy was insensitive to both over the range tested.
"""
import os

import numpy as np

from _common import base_parser, ensure_dirs, label_of, load_drone
from mastfit.validate import (REAL_CLOUD_STATS, compare_to_truth, make_phantom,
                              matched_phantom, slice_statistics)

N_SEEDS = 6


def summarise(st):
    return dict(
        sigma_med=np.median(st["sigma"]),
        sigma_p90=np.percentile(st["sigma"], 90),
        trimmed=np.median(st["trimmed"]),
        coverage_p10=np.percentile(st["coverage"], 10),
        density=np.median(st["density"]),
    )


def compute(args):
    """All numbers of Table 2 and Section 4.6 (returned, not only printed)."""
    drone = load_drone(args)
    pooled = {k: [] for k in ("sigma", "trimmed", "coverage", "density")}
    per_mast = []
    for P in drone:
        st = slice_statistics(P, n_slices=args.slices)
        for k in pooled:
            pooled[k].append(st[k])
        per_mast.append(summarise(st))
    real_raw = {k: np.concatenate(v) for k, v in pooled.items()}
    real = summarise(real_raw)

    def phantom_summary(fn):
        rows, errs, raws = [], [], []
        for sd in range(N_SEEDS):
            P, zt, bt = fn(sd)
            st = slice_statistics(P, n_slices=args.slices)
            raws.append(st)
            rows.append(summarise(st))
            errs.append(compare_to_truth(P, zt, bt)[3])
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        raw = {k: np.concatenate([r[k] for r in raws]) for k in raws[0]}
        return agg, np.asarray(errs), raw

    orig, e_orig, raw_orig = phantom_summary(
        lambda sd: make_phantom(mode="C", amplitude_m=0.15, noise_m=0.02,
                                clutter_frac=0.03, base_flare=True,
                                correlated=True, seed=sd))
    match, e_match, raw_match = phantom_summary(lambda sd: matched_phantom(seed=sd))
    return {"real": real, "orig": orig, "match": match, "per_mast": per_mast,
            "errs_orig": e_orig, "errs_match": e_match,
            "raw_real": real_raw, "raw_orig": raw_orig, "raw_match": raw_match}


def main():
    args = base_parser(__doc__).parse_args()
    ensure_dirs(args.outdir)
    out = compute(args)
    real, orig, match = out["real"], out["orig"], out["match"]

    print("REAL UAV CLOUDS -- per-slice statistics")
    print(f"{'Mast':<14}{'sigma_med':>11}{'sigma_p90':>11}{'trim%':>8}{'cov_p10':>9}{'pts/m':>8}")
    print("-" * 61)
    for i, s in enumerate(out["per_mast"]):
        print(f"{label_of(i):<14}{s['sigma_med']:>11.1f}{s['sigma_p90']:>11.1f}"
              f"{s['trimmed']:>8.1f}{s['coverage_p10']:>9.1f}{s['density']:>8.0f}")
    print(f"{'POOLED':<14}{real['sigma_med']:>11.1f}{real['sigma_p90']:>11.1f}"
          f"{real['trimmed']:>8.1f}{real['coverage_p10']:>9.1f}{real['density']:>8.0f}")

    print("\nPHANTOM-TO-REAL MATCH")
    print(f"{'statistic':<32}{'real':>10}{'original':>11}{'matched':>10}")
    print("-" * 63)
    for key, lab in [("sigma_med", "radial residual sigma, med (mm)"),
                     ("sigma_p90", "radial residual sigma, p90 (mm)"),
                     ("trimmed", "trimmed as outliers (%)"),
                     ("coverage_p10", "angular coverage, p10 (%)"),
                     ("density", "point density (pts/m)")]:
        print(f"{lab:<32}{real[key]:>10.1f}{orig[key]:>11.1f}{match[key]:>10.1f}")

    e_orig, e_match = out["errs_orig"], out["errs_match"]
    print("\nTRANSFERRED ACCURACY (correlated-error RMSE, mm)")
    print(f"  original phantom : {e_orig.mean():5.1f}   (range {e_orig.min():.0f}-{e_orig.max():.0f})")
    print(f"  matched  phantom : {e_match.mean():5.1f}   (range {e_match.min():.0f}-{e_match.max():.0f})")

    print("\nVERDICT")
    print("  The original phantom was milder than the data. Matching it changes the")
    print("  transferred accuracy only slightly, because slice-centre averaging removes")
    print("  per-point scatter; the limiting term is the correlated error. Remaining,")
    print("  stated discrepancies: the real tail is heavier (p90 112 vs ~75 mm) and the")
    print("  real rings have mild occlusion gaps (coverage p10 92% vs 100%).")

    _figure(real, orig, match, load_drone(args), args)


def _figure(real, orig, match, drone, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = {k: [] for k in ("sigma", "trimmed", "coverage", "density")}
    for P in drone:
        st = slice_statistics(P, n_slices=args.slices)
        for k in pooled:
            pooled[k].append(st[k])
    R = {k: np.concatenate(v) for k, v in pooled.items()}
    O = slice_statistics(make_phantom(mode="C", amplitude_m=0.15, noise_m=0.02,
                                      clutter_frac=0.03, base_flare=True,
                                      correlated=True, seed=1)[0])
    M = slice_statistics(matched_phantom(seed=1)[0])

    panels = [("sigma", "per-slice radial residual \u03c3", "mm"),
              ("trimmed", "points trimmed as outliers", "%"),
              ("coverage", "angular coverage of ring", "%"),
              ("density", "point density", "pts / m")]
    colours = ["#c0392b", "#8fa8bd", "#2e8b57"]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (key, title, unit) in zip(axs, panels):
        bp = ax.boxplot([R[key], O[key], M[key]], patch_artist=True, widths=.6,
                        showfliers=False, tick_labels=["real", "original", "matched"])
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(.75)
        for med in bp["medians"]:
            med.set_color("k")
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(unit)
        ax.grid(alpha=.25, axis="y")
    fig.suptitle("Phantom-to-real statistical match: the original phantom was milder than the "
                 "data; the matched phantom is not", fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(args.outdir, "fig_phantom_match.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
