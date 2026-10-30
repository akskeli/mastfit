#!/usr/bin/env python3
"""Why base-flare removal matters (Section 4.2; reproduces the 87 -> 410 mm inflation).

The kernel-smoothed operating pipeline largely masks the flare (changes of at most
3 mm), so the mechanism is demonstrated on the unsmoothed baseline variant
(:mod:`mastfit.baseline`), the pipeline a practitioner would write without this
quality-control stage. Without flare rejection, the straightest mast would be
reported as the most deformed.

Reproduces (unsmoothed baseline, 40 slices; bend without -> with the flare test):
    Foremast     257 -> 218 mm   (clean base)
    Mainmast     135 -> 132 mm   (clean base)
    Middle Mast  324 -> 104 mm
    Mizzenmast   410 ->  87 mm
    Jiggermast   131 -> 115 mm   (clean base)
"""
from _common import base_parser, label_of, load_drone
from mastfit.baseline import naive_pipeline  # noqa: F401  (re-exported for older imports)


def main():
    args = base_parser(__doc__).parse_args()
    masts = load_drone(args)
    print(f"{'Mast':<14}{'untrimmed':>11}{'trimmed':>10}   effect")
    print("-" * 52)
    for i, P in enumerate(masts):
        off = naive_pipeline(P, quality_control=False)
        on = naive_pipeline(P, quality_control=True)
        eff = "COLLAPSES" if off > 2 * on else "clean base"
        print(f"{label_of(i):<14}{off:>11.0f}{on:>10.0f}   {eff}")
    print("\nWithout flare rejection the Mizzenmast (the straightest mast) would be")
    print("reported as the most deformed. This is the decisive quality-control stage.")


if __name__ == "__main__":
    main()
