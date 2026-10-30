# Changelog

## 1.6.0 (2026)

First public release, accompanying the paper *"Self-Referential Generalized-Cylinder
Fitting for Mast Deformation Measurement from UAV Point Clouds"* (Sensors, 2026).

* Deterministic pipeline for mast bending from UAV point clouds: PCA axis seed, IRLS
  refinement with Tukey's biweight, taper-aware slicing with trimmed Taubin circle fits
  and centre covariances, base-flare and rigging-band slice tests, covariance-weighted
  Gaussian kernel smoother, least-squares straight axis and rake–bend decomposition.
* Validation tools: matched synthetic phantoms with independent and spatially correlated
  error models, per-mast error floors, null distribution of the total bend for the
  significance test, five-mast shared-curvature test for photogrammetric doming,
  comparison with terrestrial laser scanning.
* Scripts that reproduce every table, figure and number of the paper.
