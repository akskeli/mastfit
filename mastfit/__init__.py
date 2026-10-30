"""mastfit -- deterministic generalized-cylinder fitting for mast deformation.

Reference implementation for:

    "Self-Referential Generalized-Cylinder Fitting for Mast Deformation
     Measurement from UAV Point Clouds"

Quick start
-----------
>>> from mastfit.io import load_masts, MAST_NAMES
>>> from mastfit.pipeline import fit_mast
>>> masts = load_masts("data/korig_dron_JARBOLI_SEGMENTED.mat", sensor="drone")
>>> res = fit_mast(masts[0], name="Foremast")
>>> round(res.max_bend_mm)
160
"""

from .doming import doming_decomposition, shared_fraction
from .io import MAST_NAMES, coregister, load_masts, robust_clean, ship_frame
from .pipeline import MastResult, fit_mast

__version__ = "1.6.0"

__all__ = [
    "fit_mast",
    "MastResult",
    "doming_decomposition",
    "shared_fraction",
    "load_masts",
    "coregister",
    "ship_frame",
    "robust_clean",
    "MAST_NAMES",
]
