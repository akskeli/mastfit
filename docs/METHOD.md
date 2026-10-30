# Method notes

A short companion to the paper: the main design decisions and the reasons behind them.
Section, table and figure numbers refer to the paper.

## The estimand

A mast is a tapered tube with a possibly curved axis, a generalized cylinder. Fitting
one global cylinder confounds taper with bending, so the cloud is sliced perpendicular
to a robust axis and the centre of each slice is estimated independently.

The reported quantity is not the centreline itself but its departure from a straight
reference axis fitted to it by least squares:

```
b(z) = c(z) - (c0 + z * d)
```

The reference is derived from the mast, so no CAD model and no external alignment are
needed. The direction `d` gives the as-built rake, which is compared with the design
value (about 2° aft) but not imposed. Two consequences follow: any part of the
deformation that is linear in height is reported as rake, not as bending; and the rake
is measured from the vertical of the cloud, so it includes any trim or heel of the hull.

## Deterministic estimation

RANSAC-type fitters are randomized, so two runs on the same input can give different
answers. Every estimator here is deterministic (PCA seed, IRLS with Tukey's biweight,
trimmed Taubin circle fits, a fixed-bandwidth smoother), so repeated runs are
bit-identical; a test enforces this. After segmentation the clouds are only lightly
contaminated, so the robust steps act as a safety net rather than as the primary
estimator: changing the biweight constant over 3–9 changes the bends by at most 2.8 mm,
and replacing the IRLS axis by a robust cone fit by at most 4.9 mm.

## A fixed smoothing bandwidth, and what it resolves

Without smoothing, fine slicing lets the centreline follow per-slice noise and the
apparent bend inflates (Foremast 218 → 276 → 591 mm as the slice thickness falls from
1.25 to 0.5 and 0.25 m). A Gaussian kernel smoother with a fixed bandwidth `L` makes
the estimate nearly independent of the slice thickness (Foremast 137 → 160 → 160 mm).

`L` is the standard deviation of the kernel. A sinusoidal bend of wavelength λ is
passed with the factor exp(−2π²L²/λ²): half of its amplitude at λ = 5.3 L = 16 m for
L = 3 m, and at most 7% at the 7–8 m spacing of the yards. The reported bends therefore
describe deformation at wavelengths of about 16 m and longer.

L = 3 m was fixed before the analysis: the kernel must bridge the gaps left by removed
band and flare slices (1.3–2.2 m), and it should describe the bending of the mast as a
whole. The error floor and the null distribution are always computed with the same L,
and the significance verdicts are the same for L = 2–5 m (Table 7).

## Contamination at the level of whole slices

Contamination means points that are not on the mast surface. It is distinct from
correlated reconstruction error, which affects the good points. Two kinds act on whole
slices, so no per-point trim can remove them:

1. **Base flare** (the deck collar). Left in, it anchors and tilts the reference line and
   injects false bending along the whole mast. It is rejected by a taper-consistency
   test on contiguous end slices. In an unsmoothed baseline variant of the pipeline,
   disabling the test raises one bend from 87 to 410 mm (Mizzenmast). In the final
   pipeline the flare test, the covariance weighting and the band test overlap:
   disabling the flare test alone changes the bends by at most 2.2 mm, but disabling all
   three raises the same bend from 36 to 114 mm (Table 6).
2. **Rigging bands**: attachments that appear as local peaks in the fit residual and the
   ovality. The flagged slices form three to eight bands per mast at a median interval of
   about 7 m, of the order of the yard spacing.

The TLS clouds carry many more rigging returns and are cleaned first by a band-wise
radial filter (`mastfit.io.robust_clean`); on simulated oval or dented sections it keeps
the true surface and changes the bend by at most 0.3 mm.

## Accuracy: correlated error is the limiting term

Structure-from-motion error is spatially correlated. A smooth error field displaces a
whole slice rather than scattering its points, so it hardly shows in the per-slice
residual, yet it is what limits the accuracy of the centreline.

| Error model | Bending-profile RMSE |
|---|---|
| independent noise, 20 mm RMS | ~6 mm |
| spatially correlated error, 20 mm RMS | ~8.5 mm (5–13 mm over seeds) |
| matched phantoms, correlated field 45 mm RMS | per mast 20–27 mm (mean), 29–40 mm (90th percentile) |
| the same phantoms, independent noise | ~7 mm |

The correlated field of the matched phantoms is deliberately as large as the whole
observed per-slice scatter, so the floors are pessimistic. They depend mainly on the
assumed amplitude (means of 10–29 mm for 15–60 mm) and less on the correlation length
(16–23 mm for 1–40 m); Table 4 lists fourteen error models.

### The phantom must resemble the data

An accuracy figure measured on a phantom is transferred to the real masts, which is only
valid if the phantom resembles the data in the statistics that govern fitting accuracy
(`scripts/08_phantom_match.py`, Table 2):

| Statistic (per slice) | Real clouds | Generic phantom | Matched phantom |
|---|---:|---:|---:|
| radial residual σ, median | 45.6 mm | 9.0 mm | 46.2 mm |
| radial residual σ, 90th percentile | 112.2 mm | 11.2 mm | 75.4 mm |
| points trimmed as outliers | 5.6% | 0.2% | 6.6% |
| angular coverage, 10th percentile | 91.7% | 100% | 100% |
| point density | 387 /m | 391 /m | 423 /m |

`matched_phantom` injects incoherent noise (which sets σ), a correlated field (which sets
the accuracy) and scattered off-surface points. `phantom_geometry(i)` adds the height,
base radius, taper and point density of mast `i`, so each mast gets its own floor:

```python
P, z, b = matched_phantom(seed=i, **phantom_geometry(i))
```

The real clouds keep a heavier residual tail and mild coverage gaps behind rigging;
accuracy was insensitive to both over the range tested.

## Significance: a null distribution, not a floor

The error floor is a profile-wide RMSE of one bending component; the reported bend is
the maximum over height of the two-component magnitude, which for a straight mast is
1.9–2.5 times larger. The floor is therefore the accuracy of the profile, not a
significance threshold.

A bend is significant at the 5% level if it exceeds the 95th percentile of the total
bends of 200 straight matched phantoms of the same mast, i.e. if its empirical p-value
(1 + n)/(N + 1) is below 0.05 (`scripts/15_significance_and_ablations.py`, section B).
The test holds its level on independent straight phantoms (5.6%), whereas a rule of
"twice the 90th-percentile floor" would call 18% of straight masts bent
(`scripts/17_null_holdout.py`). The verdicts depend on the assumed error model
(`scripts/16_null_by_error_model.py`): only the Foremast is significant under all
fourteen models.

## The five-mast shared-curvature test

Reconstruction doming is a property of the reconstruction volume, so it imprints a
common low-order curvature on all masts imaged together, whereas genuine bends are
independent. Decomposing the five profiles into a shared and a mast-specific component
therefore bounds the artefact from the redundancy of the scene (`mastfit.doming`).

A shared component is not proof of doming: a common structural or environmental cause
looks the same. With five masts the smallest attainable exact p-value is 2/120 ≈ 0.017,
and on these data the test is inconclusive (transverse loading against mast position
r = −0.29, p = 0.65; longitudinal r = −0.96, p = 0.017, carried by the Foremast and gone
without it). The shared component is reported as a candidate artefact. On synthetic data
an injected 49% shared fraction is recovered as 51%.

## Comparison with terrestrial laser scanning

Both clouds are transformed into ship frames built in the same way from each cloud: the
vertical of the cloud, the line through the mast positions, and the deck level
(`mastfit.io.ship_frame`). The TLS scanner's inclinometer was switched off, so its
vertical is tilted against the UAV vertical by 3.3°. Bending is measured against each
mast's own straight axis and is invariant to this rotation, so it does not affect the
comparison; absolute rake is reported from the UAV data alone.

Paired at the same heights over the span both sensors cover and re-referenced to one
straight line (`mastfit.crossval`), the profiles differ by 14 mm (transverse) and 33 mm
(longitudinal) RMS. Section 4.15 gives the comparison per mast and without
re-referencing (`scripts/20_tls_comparison.py`).

## Further checks

- **Section model** (`scripts/19_section_model.py`): replacing the circle centres by
  ellipse centres changes the bends by at most 3.8 mm. A one-sided radial bias of the
  reconstructed surface that varies along the height would move the centres with either
  model; only the comparison with TLS bounds it.
- **Segmentation** (`scripts/21_segmentation_check.py`): each segmented UAV mast is
  continuous along its height and around it, agrees in length with the independently
  segmented TLS mast to within 1.75 m, and has 0.9–3.0% of its points more than 0.3 m
  outside the fitted surface, mostly at the deck around the foot and in slices that the
  flare and band tests remove. Removing them changes the bends by at most 4.0 mm.
