"""Shared helpers for the reproduction scripts."""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mastfit.io import MAST_NAMES, load_masts  # noqa: E402

DEFAULT_DRONE = "data/korig_dron_JARBOLI_SEGMENTED.mat"
DEFAULT_LASER = "data/laser_JARBOLI_SEGMENTED.mat"
FIGDIR = "figures"
RESDIR = "results"


def name_of(i: int) -> str:
    return MAST_NAMES[i][1]


def label_of(i: int) -> str:
    num, nm = MAST_NAMES[i]
    return f"{num} {nm}"


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--drone", default=DEFAULT_DRONE, help="segmented drone .mat")
    p.add_argument("--laser", default=DEFAULT_LASER, help="segmented laser .mat")
    p.add_argument("--slices", type=int, default=100, help="slices per mast")
    p.add_argument("--bandwidth", type=float, default=3.0, help="kernel bandwidth L [m]")
    p.add_argument("--outdir", default=FIGDIR, help="figure output directory")
    return p


def require(path: str) -> str:
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: missing data file {path!r}.\n"
            "The point clouds are not redistributed with this repository.\n"
            "Place the .mat files in ./data/ (see README, 'Data')."
        )
    return path


def load_drone(args):
    return load_masts(require(args.drone), sensor="drone")


def load_laser(args):
    return load_masts(require(args.laser), sensor="laser")


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def viridis(i: int):
    import matplotlib.pyplot as plt
    return plt.cm.viridis(i / 4 * 0.85)
