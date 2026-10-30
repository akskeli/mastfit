#!/usr/bin/env python3
"""Null-distribution test of the total bend under each of the 14 error models of Table 4.

For every model and mast, the full pipeline is run on straight synthetic masts (matched
phantoms with zero bend); the measured total bend is significant at the 5% level if it
exceeds the 95th percentile of that null distribution. Writes results/null_by_model.json
(or results/null_by_model_<part>.json when the models are split over several processes).

Usage: python scripts/16_null_by_error_model.py --part 0 --parts 2 --n 100
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mastfit.pipeline import fit_mast  # noqa: E402
from mastfit.validate import matched_phantom, phantom_geometry  # noqa: E402

BENDS = [159.6, 89.4, 45.4, 36.0, 70.2]
MODELS = [
    ("Default field, 45 mm (Table 3)", {}),
    ("GP, l = 1 m", {"corr_len_m": 1.0}),
    ("GP, l = 2.5 m", {"corr_len_m": 2.5}),
    ("GP, l = 5 m", {"corr_len_m": 5.0}),
    ("GP, l = 10 m", {"corr_len_m": 10.0}),
    ("GP, l = 20 m", {"corr_len_m": 20.0}),
    ("GP, l = 40 m", {"corr_len_m": 40.0}),
    ("Default field, 15 mm", {"noise_m": 0.015}),
    ("Default field, 30 mm", {"noise_m": 0.030}),
    ("Default field, 60 mm", {"noise_m": 0.060}),
    ("GP, l = 10 m, anisotropy 1:2", {"corr_len_m": 10.0, "corr_anisotropy": 0.5}),
    ("GP, l = 10 m, anisotropy 2:1", {"corr_len_m": 10.0, "corr_anisotropy": 2.0}),
    ("GP, l = 10 m, translation only", {"corr_len_m": 10.0, "corr_angular_frac": 0.0}),
    ("GP, l = 10 m, 80% angular", {"corr_len_m": 10.0, "corr_angular_frac": 0.8}),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    out = {}
    for k, (label, kw) in enumerate(MODELS):
        if k % a.parts != a.part:
            continue
        row = {}
        for i in range(5):
            g = phantom_geometry(i)
            b0 = np.array([fit_mast(matched_phantom(seed=90000 + 1000 * i + sd, correlated=True,
                                                    amplitude_m=0.0, **g, **kw)[0]).max_bend_mm
                           for sd in range(a.n)])
            p = (1 + np.sum(b0 >= BENDS[i])) / (a.n + 1)
            row[i] = {"p95": round(float(np.percentile(b0, 95)), 1), "p": round(float(p), 4)}
        out[label] = row
        print(label, {i: (v["p95"], v["p"]) for i, v in row.items()}, flush=True)
    os.makedirs("results", exist_ok=True)
    name = "results/null_by_model.json" if a.parts == 1 else f"results/null_by_model_{a.part}.json"
    with open(name, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
