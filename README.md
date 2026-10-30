# mastfit

Deterministic generalized-cylinder fitting for measuring the bending of ship masts from UAV photogrammetric point clouds.

Reference implementation for the paper *"Self-Referential Generalized-Cylinder Fitting for Mast Deformation Measurement from UAV Point Clouds"* (Sensors, 2026), applied to the five masts of the sailing ship **Golden Horizon**.

The method measures how far each mast departs from a straight, raked axis fitted to the mast itself, so it needs no CAD model, no ground truth and no manual alignment. It is fully deterministic: there is no random sampling, and repeated runs on the same input give bit-identical results.

---

## Main results

Each mast is modelled as a tapered generalized cylinder: a robust axis is fitted, slices along it are fitted with trimmed circles, base-flare and rigging-band slices are rejected, and an uncertainty-weighted centreline is split into rake and bending.

On the Golden Horizon (Table 1 of the paper, kernel bandwidth L = 3 m):

| Mast | Points | Rake (°) | Diameter (mm) | Transverse (mm) | Longitudinal (mm) | **Total (mm)** | σ_stat (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 Foremast | 14,752 | 1.38 | 678 | 61 | 148 | **160** | 6.4 |
| 2 Mainmast | 18,069 | 1.86 | 704 | 82 | 48 | **89** | 6.1 |
| 3 Middle Mast | 17,152 | 1.73 | 749 | 30 | 34 | **45** | 5.2 |
| 4 Mizzenmast | 25,834 | 1.45 | 1030 | 35 | 28 | **36** | 8.2 |
| 5 Jiggermast | 17,586 | 1.20 | 959 | 37 | 60 | **70** | 11.3 |

- **Accuracy.** σ_stat is the random slice-centre uncertainty. The limiting term is spatially correlated reconstruction error, which barely shows in the per-slice residual. On synthetic point clouds matched to each mast, an assumed correlated error gives per-mast error floors of 20–27 mm (mean) and 29–40 mm (90th percentile), against about 7 mm under independent noise. The floors depend mainly on the assumed amplitude of the correlated error (Table 4).
- **Significance.** Each bend is tested against the null distribution of the total bend of straight synthetic masts of the same geometry. At the 5% level the Foremast (p < 0.005) and the Mainmast (p = 0.020) are significant; the Jiggermast (0.085), Middle Mast (0.61) and Mizzenmast (0.79) are not resolved. After a Holm correction for five masts, only the Foremast remains significant. Across the fourteen error models examined, the Foremast is significant under all, the Mainmast under eleven and the Jiggermast under seven.
- **Independent check.** Processed by the same pipeline, a terrestrial laser scanning (TLS) survey gives bending profiles that differ from the UAV profiles by 14 mm (transverse) and 33 mm (longitudinal) RMS. The Foremast could not be checked, because its TLS coverage is too sparse to fit.
- **Resolution.** The Gaussian kernel smoother passes half of the amplitude of a bend of wavelength 16 m (L = 3 m), so the reported bends describe deformation at wavelengths of about 16 m and longer.

---

## Install

```bash
git clone https://github.com/akskeli/mastfit.git
cd mastfit
pip install -e .            # or: pip install -e ".[dev]" to also install pytest
```

Requires Python ≥ 3.9, NumPy, SciPy and Matplotlib. Script `11` also needs Graphviz (`dot`).

## Data

The point clouds are **not included** in this repository. The TLS point clouds were provided by Brodosplit and the UAV point cloud by an external surveyor; they are available from the corresponding author on reasonable request, subject to permission from the data owners. See [`data/README.md`](data/README.md) for the expected files and format.

The tests and `scripts/05_synthetic_validation.py` need no data.

---

## Reproducing the paper

Run the scripts from the repository root.

| Command | Reproduces |
|---|---|
| `python scripts/01_fit_all_masts.py` | Table 1 |
| `python scripts/02_doming_test.py` | five-mast shared-curvature test (Section 4.4) |
| `python scripts/03_contamination_ablation.py` | base-flare mechanism in the unsmoothed baseline (Section 4.2) |
| `python scripts/04_crossvalidation.py` | UAV–TLS comparison, Bland–Altman, mast-block bootstrap (Section 4.8) |
| `python scripts/05_synthetic_validation.py` | error model, robustness, flare mechanism and doming test on phantoms (Section 4.5); no data needed |
| `python scripts/06_sensitivity.py` | slice thickness, bandwidth and threshold sensitivity (Section 4.2) |
| `python scripts/08_phantom_match.py` | phantom-to-data match (Section 4.6, Table 2) |
| `python scripts/09_all_masts_three_sources.py` | all masts from UAV, TLS and matched phantoms side by side |
| `python scripts/10_per_mast_accuracy.py` | per-mast error floors (Section 4.7, Table 3) |
| `python scripts/11_pipeline_flowchart.py` | pipeline diagram (Figure 2) from `assets/pipeline_flowchart.dot` |
| `python scripts/12_paper_numbers.py` | every number quoted in the text, section by section → `results/paper_numbers.{txt,json}` |
| `python scripts/13_paper_figures.py` | Figures 1 and 3–16 → `figures/paper/` (run script 15 first) |
| `python scripts/14_robustness_checks.py --seeds 10` | biweight constant, correlated-error model and its sensitivity (Table 4), TLS frame tilt, TLS filter |
| `python scripts/15_significance_and_ablations.py --sections B --null 200` | null distribution of the total bend, p-values (Section 4.11, Table 5, Figure 8) |
| `python scripts/15_significance_and_ablations.py --sections ACD --seeds 10` | axis refinement vs robust cone fit, smoother response (Figure 16), ablations (Table 6) |
| `python scripts/16_null_by_error_model.py --n 100` | significance under each of the 14 error models (Table 4) |
| `python scripts/17_null_holdout.py --n 100` | false-positive rate of the test on hold-out phantoms (Table 5) |
| `python scripts/18_bandwidth_sweep.py --null 200` | bends, floors, null distributions and verdicts for L = 2–5 m (Section 4.13, Table 7) |
| `python scripts/19_section_model.py --seeds 5` | circle versus ellipse slice centres (Section 4.14) |
| `python scripts/20_tls_comparison.py --seeds 5` | UAV–TLS comparison per mast, without re-referencing, TLS-side uncertainty (Section 4.15, Table 8) |
| `python scripts/21_segmentation_check.py --figure` | validation of the automatic UAV mast segmentation (Section 3.1) |

`bash reproduce_all.sh` runs the tests and all scripts in order. The scripts that use many synthetic realizations (14–18) take from several minutes to about an hour each.

Results are written to `results/` and figures to `figures/`; both are ignored by git.

---

## Library use

```python
from mastfit import load_masts, fit_mast

masts = load_masts("data/korig_dron_JARBOLI_SEGMENTED.mat", sensor="drone")
res = fit_mast(masts[0], name="Foremast", bandwidth_m=3.0)

print(res.rake_deg, res.max_bend_mm)          # 1.38, 159.6
print(res.n_slices_kept, "/", res.n_slices_total)
```

### Pipeline

```
Axis        PCA seed -> deterministic IRLS refinement (Tukey biweight)
Sections    taper-aware slicing -> trimmed Taubin circle fit per slice, with centre covariance
QC          base-flare removal (taper consistency) -> rigging-band suppression
Centreline  covariance-weighted Gaussian kernel smoother (L = 3 m)
            -> least-squares straight axis (rake) and bending profile b(z)
Validation  matched phantoms (error floor) -> null distribution of the total bend (significance)
            -> five-mast shared-curvature test -> comparison with TLS
```

| Module | Purpose |
|---|---|
| `mastfit.geometry` | Taubin and trimmed circle fits, ellipse fit, PCA axis, IRLS refinement, robust cone fit, ovality |
| `mastfit.quality` | base-flare and rigging-band slice tests |
| `mastfit.pipeline` | `fit_mast()`: the full pipeline |
| `mastfit.doming` | five-mast shared-curvature decomposition and positional test |
| `mastfit.crossval` | common-span pairing and Bland–Altman comparison with a second sensor |
| `mastfit.validate` | synthetic phantoms, correlated-error models, matched phantoms, ovality null floor |
| `mastfit.baseline` | the unsmoothed baseline variant used to expose the flare mechanism |
| `mastfit.io` | loading, mast naming, ship frame and co-registration |

Design notes and the reasons behind them: [`docs/METHOD.md`](docs/METHOD.md).

---

## Tests

```bash
pytest -q
```

45 tests, no data needed. They cover the geometric primitives, the determinism guarantee (bit-identical reruns), recovery of known bends and rakes on phantoms, insensitivity to the slice count, the correlated-versus-independent error result, the phantom-to-data match, the slice tests, the cross-validation tools and the smoother response. [`TESTING.md`](TESTING.md) lists the expected outputs of the scripts.

---

## Scope and limitations

- **Not a learned method** and **not a stochastic-model-based adjustment.** Correlated reconstruction error is bounded empirically, with matched phantoms and an independent sensor, rather than carried as an error covariance inside the estimator.
- **A single static capture** cannot separate permanent from elastic deformation; the results describe the as-built centreline at capture time.
- **The error floor rests on an assumed correlated error model**, whose amplitude cannot be identified from the UAV cloud; Table 4 of the paper shows how the floors and verdicts change with it. The acquisition and processing metadata of the UAV survey were not available.
- **The Foremast bend**, the largest and the most robust statistically, rests on the UAV data alone. The TLS comparison supports the Mainmast bend in its transverse component only.
- **The shared-curvature test for doming is inconclusive** on five masts; the shared component is a candidate artefact, not a proven one.
- **Ovality** of the reconstructed sections exceeds the round-ring null level, but TLS suggests it belongs mostly to the reconstruction; it is not reported as deformation. The UAV diameters describe the reconstructed surface, not the steel tube.
- **The rake** is measured from the vertical of the UAV cloud and includes any trim or heel of the moored hull.

## Citation

```bibtex
@article{kuzmanicskelin2026mastfit,
  title   = {Self-Referential Generalized-Cylinder Fitting for Mast Deformation
             Measurement from UAV Point Clouds},
  author  = {Kuzmanic Skelin, Ana and Bonkovic, Mirjana and Milicevic, Mario
             and Mimica, Ratko},
  journal = {Sensors},
  year    = {2026},
  volume  = {26},
  note    = {in press},
  doi     = {10.3390/xxxxx}
}
```

## License

MIT — see [LICENSE](LICENSE).
