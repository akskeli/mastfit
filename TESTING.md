# Testing and reproducing

How to check an installation of `mastfit` and what each check should print. Run all
commands from the repository root.

---

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"            # mastfit + numpy/scipy/matplotlib + pytest
python -c "import mastfit; print(mastfit.__version__)"
```

**Expect:** `1.6.0`

## 2. Test suite (no data needed)

```bash
pytest -q
```

**Expect:** `45 passed`, in about one to two minutes. The tests run on synthetic
phantoms only.

| Test | What it protects |
|---|---|
| `test_pipeline_is_bit_identical_across_runs` | determinism: identical input gives bit-identical output |
| `test_recovers_known_bend`, `test_recovers_known_rake` | the estimator recovers a known bend and rake |
| `test_straight_mast_has_small_bend` | the pipeline does not produce bending from noise |
| `test_bend_is_insensitive_to_slice_count` | the fixed bandwidth decouples the bend from the slice thickness |
| `test_correlated_error_is_worse_than_white` | correlated error costs more than independent noise |
| `test_matched_phantom_reproduces_real_slice_statistics` | the phantom matches the measured per-slice statistics |
| `test_phantom_geometry_tracks_each_real_mast` | each phantom has the geometry of its mast |
| `test_quality_control_suppresses_flare_inflation` | flare rejection works on a phantom with known truth |
| `test_scoring_is_not_offset_by_removed_base_slices` | phantoms are scored at physical heights |
| `test_smoother_transfer_follows_the_gaussian_response` | the smoother passes exp(−2π²L²/λ²) of a sinusoidal bend |
| geometry, doming and validation-tool tests | circle and ellipse fits, ovality, axis seeding, shared-curvature decomposition, cross-validation pairing |

## 3. Data

Place the two `.mat` files in `data/` (see `data/README.md`) and check them:

```bash
python -c "
from mastfit.io import load_masts
m = load_masts('data/korig_dron_JARBOLI_SEGMENTED.mat', sensor='drone')
print([len(x) for x in m])
"
```

**Expect:** `[14752, 18069, 17152, 25834, 17586]` (bow to stern: Foremast, Mainmast,
Middle Mast, Mizzenmast, Jiggermast). A different set means that the cell order of the
file differs from `DRONE_CELL` / `LASER_CELL` in `mastfit/io.py`.

## 4. Expected outputs

### Table 1

```bash
python scripts/01_fit_all_masts.py
```

```
Mast            Points   Rake   Diam  Trans   Long  Total  sigma
-------------------------------------------------------------------
1 Foremast      14,752   1.38    678     61    148    160    6.4
2 Mainmast      18,069   1.86    704     82     48     89    6.1
3 Middle Mast   17,152   1.73    749     30     34     45    5.2
4 Mizzenmast    25,834   1.45   1030     35     28     36    8.2
5 Jiggermast    17,586   1.20    959     37     60     70   11.3
```

### Determinism

```bash
python -c "
from mastfit.io import load_masts
from mastfit.pipeline import fit_mast
import hashlib
m = load_masts('data/korig_dron_JARBOLI_SEGMENTED.mat', sensor='drone')
for _ in range(3):
    r = fit_mast(m[0], name='Foremast')
    print(round(r.max_bend_mm, 4), hashlib.md5(r.bend_total_mm.tobytes()).hexdigest()[:16])
"
```

**Expect** three identical lines, for example `159.5599 05d4964292364860`. The hash can
differ on another NumPy or BLAS build; the three lines must be identical to each other.

### Other scripts

| Command | Time | Expect |
|---|---|---|
| `python scripts/02_doming_test.py` | seconds | shared fraction 28% (transverse), 14% (longitudinal) |
| `python scripts/03_contamination_ablation.py` | seconds | unsmoothed baseline: Mizzenmast 410 → 87 mm, Middle Mast 324 → 104 mm |
| `python scripts/04_crossvalidation.py` | ~1 min | RMS difference 14 / 33 mm, 95% LoA ±28 / ±65 mm, r = +0.65 / +0.44; Foremast excluded |
| `python scripts/05_synthetic_validation.py` | ~1 min | white noise ≈ 6 mm, correlated error ≈ 8.5 mm (5–13 mm); doming test 49% injected, 51% recovered |
| `python scripts/06_sensitivity.py` | ~30 s | 40/100/200 slices: Foremast 137/160/160 mm; L = 2/3/4/5 m: Foremast 176/160/147/136 mm |
| `python scripts/08_phantom_match.py` | ~1 min | real σ 45.6 mm, generic phantom 9.0 mm, matched 46.2 mm; transferred RMSE 9 → 21 mm |
| `python scripts/10_per_mast_accuracy.py` | ~30 s | floors 20.0–27.1 mm (mean), 29.0–40.1 mm (90th percentile) |
| `python scripts/12_paper_numbers.py` | ~2 min | every number of the text → `results/paper_numbers.{txt,json}` |
| `python scripts/15_significance_and_ablations.py --sections B --null 200` | long | p < 0.005, 0.020, 0.61, 0.79, 0.085 (bow to stern); Holm-adjusted 0.025 for the Foremast |
| `python scripts/13_paper_figures.py` | ~2 min | Figures 1 and 3–16 in `figures/paper/` (needs the output of script 15) |

`bash reproduce_all.sh` runs the tests and every script in order.

## 5. Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: mastfit` | the package is not installed (`pip install -e .`) or the environment is not active |
| `ERROR: missing data file` | the `.mat` files are not in `data/`, or the script was started from inside `scripts/`; run it from the repository root |
| point counts differ from the list above | the cell order of the `.mat` file differs from `DRONE_CELL` / `LASER_CELL` |
| `ValueError: too sparse to fit` | expected only for the TLS Foremast (1,930 points) |
| bends differ by 1–2 mm | usually an old NumPy or SciPy (needs numpy ≥ 1.21, scipy ≥ 1.7) |
| Matplotlib backend error | headless machine: `export MPLBACKEND=Agg` |
