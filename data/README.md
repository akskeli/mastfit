# Data

The point clouds are not included in this repository. The TLS point clouds of the
Golden Horizon masts were provided by Brodosplit and the UAV point cloud by an external
surveyor; they are available from the corresponding author on reasonable request,
subject to permission from the data owners.

Place the two files here:

```
data/korig_dron_JARBOLI_SEGMENTED.mat    # UAV photogrammetric point cloud, segmented masts
data/laser_JARBOLI_SEGMENTED.mat         # terrestrial laser scan, segmented masts (comparison)
```

Format (MATLAB `.mat`, readable with `scipy.io.loadmat`):

- `jarbol`: 5 × 1 cell array, one cell per mast, each an N × 3 array of X, Y, Z
  coordinates in metres;
- `labels`: per-point mast index (1–5), not used by the code.

The two files store the masts in different cell orders; `DRONE_CELL` and `LASER_CELL` in
`mastfit/io.py` map them to the physical order, bow to stern (Foremast, Mainmast, Middle
Mast, Mizzenmast, Jiggermast). Other paths can be passed to every script with `--drone`
and `--laser`.

`*.mat` files in this folder are ignored by git.
