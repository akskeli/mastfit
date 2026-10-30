#!/usr/bin/env python3
"""Per-mast accuracy floor from geometry- and noise-matched phantoms (Section 4.7, Table 3).

The five masts are not the same estimation problem: they differ by 12 m in height and
by a factor of six in taper (the Mizzenmast is nearly a straight tube, the Mainmast and
Middle Mast taper most), and both affect the slice count, the angular coverage and the
radius-trend fit that drives flare rejection. Each mast therefore gets its own phantom,
with that mast's height, base radius, taper and point density plus the measured
per-slice noise statistics, under spatially correlated error. The bending-profile RMSE
over many seeds gives that mast's floor as a distribution:

    Foremast 27.1 mm (mean) / 40.1 mm (90th percentile), Mainmast 20.1 / 29.0,
    Middle Mast 21.1 / 31.7, Mizzenmast 21.0 / 33.2, Jiggermast 20.0 / 34.7.

The floor is the accuracy of the recovered profile. It is not a significance threshold:
significance is tested against the null distribution of the total bend of straight
masts (scripts/15_significance_and_ablations.py, section B).
"""
import numpy as np

from _common import base_parser, ensure_dirs
from mastfit.validate import (REAL_MAST_GEOMETRY, compare_to_truth, matched_phantom,
                              phantom_geometry)

#: Measured bends (mm) from scripts/01_fit_all_masts.py (Table 1)
MEASURED_BEND_MM = {
    "Foremast": 160,
    "Mainmast": 89,
    "Middle Mast": 45,
    "Mizzenmast": 36,
    "Jiggermast": 70,
}


def compute_floors(seeds: int = 10) -> dict:
    """Correlated-error RMSE of each mast's geometry- and noise-matched phantom."""
    floors = {}
    for i, (name, _h, _r0, _taper, _d) in enumerate(REAL_MAST_GEOMETRY):
        errs = []
        for sd in range(seeds):
            P, zt, bt = matched_phantom(seed=100 * i + sd, correlated=True,
                                        **phantom_geometry(i))
            errs.append(compare_to_truth(P, zt, bt)[3])
        floors[name] = np.asarray(errs)
    return floors


def bootstrap_ci(errs, seed=0, n_boot=2000):
    """95% bootstrap CIs on the mean and the 90th percentile of a floor distribution."""
    rng = np.random.default_rng(seed)
    bm, bp = [], []
    for _ in range(n_boot):
        b = rng.choice(errs, size=len(errs), replace=True)
        bm.append(b.mean())
        bp.append(np.percentile(b, 90))
    return tuple(np.percentile(bm, [2.5, 97.5])), tuple(np.percentile(bp, [2.5, 97.5]))


def main():
    p = base_parser(__doc__)
    p.add_argument("--seeds", type=int, default=10,
               help="phantom realizations per mast (Table 3 uses 10; "
                    "see scripts/12_paper_numbers.py for the 30-realization check)")
    args = p.parse_args()
    ensure_dirs(args.outdir)

    print("PER-MAST ACCURACY FLOOR  (geometry + noise matched, correlated error)")
    print(f"{'Mast':<14}{'height':>8}{'taper':>8}{'mean':>8}{'median':>8}{'p90':>7}{'range':>11}")
    print("-" * 64)

    floors = compute_floors(args.seeds)
    for i, (name, h, _r0, taper, _d) in enumerate(REAL_MAST_GEOMETRY):
        errs = floors[name]
        (mlo, mhi), (plo, phi) = bootstrap_ci(errs, seed=i)
        print(f"{name:<14}{h:>8.1f}{taper:>8.4f}{errs.mean():>8.1f}"
              f"{np.median(errs):>8.1f}{np.percentile(errs, 90):>7.1f}"
              f"{f'{errs.min():.0f}-{errs.max():.0f}':>11}"
              f"   CI(mean) [{mlo:.0f},{mhi:.0f}]  CI(p90) [{plo:.0f},{phi:.0f}]")

    pooled = np.concatenate(list(floors.values()))
    print("-" * 64)
    print(f"{'POOLED':<14}{'':>8}{'':>8}{pooled.mean():>8.1f}"
          f"{np.median(pooled):>8.1f}{np.percentile(pooled, 90):>7.1f}"
          f"{f'{pooled.min():.0f}-{pooled.max():.0f}':>11}")
    print("\n  -> the floors are similar across masts (mean 20-27 mm, 90th pct 29-40 mm);")
    print("     each is a DISTRIBUTION, not a point. The floor is the accuracy of the")
    print("     recovered profile; significance is tested against the null distribution of")
    print("     the total bend (scripts/15_significance_and_ablations.py --sections B).")

    _figure(floors, args)


def _figure(floors, args):
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(floors)
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot([floors[n] for n in names], patch_artist=True, widths=.55,
                    showfliers=False, tick_labels=[n.replace(" ", "\n") for n in names])
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(plt.cm.viridis(i / 4 * 0.85))
        patch.set_alpha(.75)
    for med in bp["medians"]:
        med.set_color("k")

    for i, n in enumerate(names):
        ax.scatter([i + 1], [MEASURED_BEND_MM[n]], marker="D", s=55, zorder=5,
                   color="#c0392b", edgecolor="k", linewidth=.5,
                   label="measured bend" if i == 0 else None)
        ax.annotate(f"{MEASURED_BEND_MM[n]}", (i + 1, MEASURED_BEND_MM[n]),
                    textcoords="offset points", xytext=(10, -3), fontsize=8.5)

    ax.set_ylabel("mm")
    ax.set_yscale("log")
    ax.set_title("Per-mast error floor (box) and the measured bend (diamond)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=.25, axis="y", which="both")
    fig.tight_layout()
    out = os.path.join(args.outdir, "fig_per_mast_floor.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
