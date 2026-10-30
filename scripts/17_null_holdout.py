#!/usr/bin/env python3
"""Out-of-sample check of the null test of the total bend.

The 95th-percentile thresholds of scripts/15_significance_and_ablations.py (section B, 200 straight
phantoms per mast, seeds 50000+) are applied to an independent set of straight phantoms
(seeds 95000+). The fraction of these that exceed the threshold is the empirical
false-positive rate of the test at a nominal 5%. The same set gives the false-positive rates
of floor-based rules and three distinct error measures of a straight mast: the
pointwise standard deviation of the bend at a given height, the RMS of the bend profile over
height and the maximum over height (the total bend).

Requires results/significance_B.json. Writes results/null_holdout.json.
Usage: python scripts/17_null_holdout.py --n 100
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mastfit.io import MAST_NAMES  # noqa: E402
from mastfit.pipeline import fit_mast  # noqa: E402
from mastfit.validate import matched_phantom, phantom_geometry  # noqa: E402

NAMES = [MAST_NAMES[i][1] for i in range(5)]
FLOOR_P90 = [40.1, 29.0, 31.7, 33.2, 34.7]       # Table 3, 90th percentile of the RMSE (mm)
FLOOR_MEAN = [27.1, 20.1, 21.1, 21.0, 20.0]      # Table 3, mean RMSE (mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    with open("results/significance_B.json") as f:
        ref = json.load(f)["B"]
    out = {}
    grid = np.linspace(0.05, 0.95, 37)               # normalized height, away from the ends
    pooled = {"new": [], "2xp90": [], "p90": [], "2xmean": []}
    samples = {}
    for i in range(5):
        g = phantom_geometry(i)
        thr = ref[NAMES[i]]["null_p95"]
        B, rms, cols = [], [], []
        for sd in range(a.n):
            P, _, _ = matched_phantom(seed=95000 + 1000 * i + sd, correlated=True, amplitude_m=0.0, **g)
            r = fit_mast(P)
            bx, by = np.asarray(r.bend_transverse_mm), np.asarray(r.bend_longitudinal_mm)
            B.append(r.max_bend_mm)
            rms.append(float(np.sqrt(np.mean(bx ** 2 + by ** 2))))
            zn = (r.height - r.height.min()) / np.ptp(r.height)
            cols.append(np.concatenate([np.interp(grid, zn, bx), np.interp(grid, zn, by)]))
        B = np.asarray(B)
        cols = np.asarray(cols)
        sd_point = float(np.median(cols.std(axis=0, ddof=1)))
        row = {
            "threshold_p95_from_B": thr,
            "fpr_new_test_pct": round(float(np.mean(B > thr) * 100), 1),
            "fpr_2xp90_pct": round(float(np.mean(B > 2 * FLOOR_P90[i]) * 100), 1),
            "rate_above_p90_pct": round(float(np.mean(B > FLOOR_P90[i]) * 100), 1),
            "fpr_2xmean_pct": round(float(np.mean(B > 2 * FLOOR_MEAN[i]) * 100), 1),
            "pointwise_sd_per_component_mm": round(sd_point, 1),
            "profile_rms_mm_mean": round(float(np.mean(rms)), 1),
            "total_bend_mm_mean": round(float(B.mean()), 1),
            "total_bend_mm_p95": round(float(np.percentile(B, 95)), 1),
        }
        out[NAMES[i]] = row
        samples[NAMES[i]] = [round(float(x), 2) for x in B]
        pooled["new"].append(np.mean(B > thr))
        pooled["2xp90"].append(np.mean(B > 2 * FLOOR_P90[i]))
        pooled["p90"].append(np.mean(B > FLOOR_P90[i]))
        pooled["2xmean"].append(np.mean(B > 2 * FLOOR_MEAN[i]))
        print(NAMES[i], row, flush=True)
    out["pooled_pct"] = {k: round(float(np.mean(v) * 100), 1) for k, v in pooled.items()}
    out["n_per_mast"] = a.n
    out["holdout_samples"] = samples
    print("pooled", out["pooled_pct"], flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/null_holdout.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
