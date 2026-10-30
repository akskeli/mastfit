#!/usr/bin/env python3
"""Kernel bandwidth L: bends, error floors, null distributions and verdicts for each L
(Table 7).

For L = 2, 3, 4 and 5 m, everything that depends on L is recomputed with that L:
  * the total bend of each real mast;
  * the per-mast error floor (bending-profile RMSE on 10 matched phantoms per mast);
  * the null distribution of the total bend (200 straight matched phantoms per mast, the
    seeds of scripts/15 section B, so L = 3 m reproduces Table 5) and the p-value of the
    measured bend.
Also reports the largest gap left between retained slices (removed band or flare slices),
which the kernel must bridge.

Writes results/bandwidth_sweep.json. Usage:
    python scripts/18_bandwidth_sweep.py --L 2 3 4 5 --null 200
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from _common import base_parser, load_drone  # noqa: E402
from mastfit.io import MAST_NAMES  # noqa: E402
from mastfit.pipeline import fit_mast  # noqa: E402
from mastfit.validate import compare_to_truth, matched_phantom, phantom_geometry  # noqa: E402

NAMES = [MAST_NAMES[i][1] for i in range(5)]


def holm(p):
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--L", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0])
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    drone = load_drone(a)
    out = {}
    # gaps between retained slices (independent of L)
    gaps = []
    for P in drone:
        r = fit_mast(P)
        gaps.append(round(float(np.max(np.diff(np.sort(r.slice_height)))), 2))
    out["max_gap_between_retained_slices_m"] = gaps
    print("max gap between retained slices [m]:", gaps, flush=True)
    for L in a.L:
        row = {"bend_mm": [round(fit_mast(P, bandwidth_m=L).max_bend_mm, 1) for P in drone]}
        fm, f90, p95, pv, nmean = [], [], [], [], []
        for i in range(5):
            g = phantom_geometry(i)
            e = []
            for sd in range(a.seeds):
                P, zt, bt = matched_phantom(seed=100 * i + sd, correlated=True, **g)
                e.append(compare_to_truth(P, zt, bt, bandwidth_m=L)[3])
            fm.append(round(float(np.mean(e)), 1))
            f90.append(round(float(np.percentile(e, 90)), 1))
            b0 = np.array([fit_mast(matched_phantom(seed=50000 + 1000 * i + sd, correlated=True,
                                                    amplitude_m=0.0, **g)[0],
                                    bandwidth_m=L).max_bend_mm for sd in range(a.null)])
            nmean.append(round(float(b0.mean()), 1))
            p95.append(round(float(np.percentile(b0, 95)), 1))
            pv.append(float((1 + np.sum(b0 >= row["bend_mm"][i])) / (a.null + 1)))
        row.update({"floor_mean_mm": fm, "floor_p90_mm": f90, "null_mean_mm": nmean,
                    "null_p95_mm": p95, "p_value": [round(x, 4) for x in pv],
                    "holm_p": [round(float(x), 4) for x in holm(np.array(pv))],
                    "significant_5pct": [bool(x < 0.05) for x in pv]})
        out[f"L={L:g}"] = row
        print(f"L = {L:g} m:", row, flush=True)
    os.makedirs("results", exist_ok=True)
    with open(f"results/bandwidth_sweep{a.tag}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
