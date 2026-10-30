"""Data loading, mast naming, and co-registration into a common ship frame.

Mast naming
-----------
Masts are named bow -> stern:

    1 Foremast, 2 Mainmast, 3 Middle Mast, 4 Mizzenmast, 5 Jiggermast

The bow/stern orientation is *derived from the data*, not assumed: the
Jiggermast (stern-most) is markedly the shortest mast (~40 m against 49-52 m for
the others in both sensors), which fixes the stern end unambiguously.

Cell mapping
------------
The two ``.mat`` files store the masts in different cell orders, and the two
clouds were delivered in independent coordinate frames: both are right-handed with
+z up and +y along the ship toward the stern, but their origins differ (a ~29.5 m
longitudinal offset). Both differences are resolved here.

Acquisition (from the survey records available to the authors)
--------------------------------------------------------------
* UAV photogrammetry: acquired and processed by an external contractor (Agisoft
  Metashape, ground-control points for scale). Camera, flight and ground-control
  metadata were not supplied with the point cloud.
* TLS: FARO Focus3D X130, 18 scans of which 16 were registered with reference
  spheres in FARO SCENE; inclinometer switched off (Brodosplit reverse-engineering
  report, December 2021). The TLS vertical axis is therefore defined by the
  registration, not by gravity, which explains its global tilt relative to the UAV
  cloud (Section 4.8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.io as sio

__all__ = [
    "MAST_NAMES",
    "DRONE_CELL",
    "LASER_CELL",
    "load_masts",
    "ship_frame",
    "coregister",
]

#: Physical index (0 = bow-most) -> (number, name)
MAST_NAMES = {
    0: (1, "Foremast"),
    1: (2, "Mainmast"),
    2: (3, "Middle Mast"),
    3: (4, "Mizzenmast"),
    4: (5, "Jiggermast"),
}

#: physical index -> cell index in the drone .mat
DRONE_CELL = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}
#: physical index -> cell index in the laser .mat
LASER_CELL = {0: 4, 1: 2, 2: 1, 3: 0, 4: 3}


def load_masts(path: str, sensor: str = "drone") -> list[np.ndarray]:
    """Load the five mast clouds in physical bow->stern order.

    Parameters
    ----------
    path : str
        Path to the segmented ``.mat`` file (variable ``jarbol``: 5x1 cells of Nx3).
    sensor : {"drone", "laser"}
    """
    cells = sio.loadmat(path)["jarbol"]
    mapping = DRONE_CELL if sensor == "drone" else LASER_CELL
    return [np.asarray(cells[mapping[i], 0], dtype=float) for i in range(5)]


def robust_clean(points: np.ndarray, n_iter: int = 3, k_mad: float = 2.5, n_bands: int = 30):
    """Strip off-surface points by a per-height-band radial MAD filter.

    Terrestrial laser clouds of a rigged mast carry substantially more
    off-surface return (rigging, spars, deck structure) than the segmented SfM
    clouds -- 4-7% of the points on the four masts with usable coverage. The per-slice trim inside the circle fit is
    not sufficient on its own, so laser clouds are pre-cleaned with this filter.
    """
    from .geometry import irls_axis, pca_axis  # local import to avoid cycles

    P = np.asarray(points, dtype=float)
    for _ in range(n_iter):
        c0, d0, _ = pca_axis(P)
        c, d = irls_axis(P, c0, d0)
        t = (P - c) @ d
        radial = np.linalg.norm((P - c) - np.outer(t, d), axis=1)
        edges = np.linspace(t.min(), t.max(), n_bands + 1)
        keep = np.zeros(len(P), dtype=bool)
        for k in range(n_bands):
            upper = t < edges[k + 1] if k < n_bands - 1 else t <= edges[k + 1]
            m = (t >= edges[k]) & upper
            if m.sum() < 20:
                continue
            r = radial[m]
            med = np.median(r)
            mad = 1.4826 * np.median(np.abs(r - med)) + 1e-6
            keep[m] = np.abs(r - med) < k_mad * mad
        if keep.sum() < 200:
            break
        P = P[keep]
    return P


@dataclass
class ShipFrame:
    """Rigid transform into the common ship frame."""

    centre: np.ndarray       # (2,) horizontal centroid of the mast positions
    longitudinal: np.ndarray  # (2,) unit vector, +Y toward the stern
    transverse: np.ndarray    # (2,) unit vector, +X to one side
    z_base: float

    def apply(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        xy = P[:, :2] - self.centre
        return np.column_stack([
            xy @ self.transverse,
            xy @ self.longitudinal,
            P[:, 2] - self.z_base,
        ])


def ship_frame(masts: list[np.ndarray]) -> ShipFrame:
    """Define a common ship frame intrinsically from the geometry.

    Origin  : horizontal centroid of the five mast positions (the median x, y of each
              mast's points); z = 0 at the 1st percentile of all mast points
              (approximately deck level).
    Z       : the vertical axis of the input cloud (gravity for the georeferenced
              UAV cloud; approximately level for the TLS cloud, whose inclinometer
              was off).
    Y       : the least-squares (SVD) line through the five mast positions in the
              horizontal plane of the cloud, signed so that +Y points toward the stern
              (the Jiggermast, physical index 4).
    X       : perpendicular to Y in that plane (Y rotated by +90 degrees). The mapping
              (x, y, z) -> (X, Y, z) has determinant -1, i.e. the ship frame is
              left-handed; both clouds are mapped the same way, so comparisons between
              them are unaffected, but azimuths read in this frame are mirrored.

    Each cloud keeps its own vertical, so the frames of two clouds share the mast
    positions and the heading but are tilted against each other by the angle between
    their verticals (Section 3.1 of the paper).

    The rake direction is *not* used to orient the frame: at 1-2 degrees it is
    too weak a signal and yields an unstable heading. The line through the mast
    positions, by contrast, spans ~100 m and is very well determined.
    """
    pos = np.array([[np.median(m[:, 0]), np.median(m[:, 1])] for m in masts])
    centre = pos.mean(axis=0)
    _, _, Vt = np.linalg.svd(pos - centre)
    lon = Vt[0]
    if np.dot(pos[4] - centre, lon) < 0:   # +Y toward the stern-most mast
        lon = -lon
    tran = np.array([-lon[1], lon[0]])
    z_base = float(np.percentile(np.concatenate([m[:, 2] for m in masts]), 1))
    return ShipFrame(centre=centre, longitudinal=lon, transverse=tran, z_base=z_base)


def coregister(drone: list[np.ndarray], laser: list[np.ndarray], mirror_laser: bool = False):
    """Bring both sensors into one common ship frame.

    Returns ``(drone_t, laser_t)``, each a list of five transformed clouds.

    Both input frames are right-handed with +z up, so no reflection is applied beyond
    the one built into :func:`ship_frame` (identical for both sensors). Set
    ``mirror_laser=True`` only if the laser frame is known to be left-handed.

    On the Golden Horizon data the mast positions from the two sensors then agree
    to within 0.1 m transversely and 0.7 m longitudinally.
    """
    fd, fl = ship_frame(drone), ship_frame(laser)
    dt = [fd.apply(m) for m in drone]
    lt = [fl.apply(m) for m in laser]
    if mirror_laser:
        lt = [np.column_stack([-t[:, 0], t[:, 1], t[:, 2]]) for t in lt]
    return dt, lt
