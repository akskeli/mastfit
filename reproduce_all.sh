#!/usr/bin/env bash
# Reproduce every number and figure in the paper (requires the data in ./data/).
set -euo pipefail

echo "=== tests ==="
pytest -q

for s in scripts/0[1-6]_*.py scripts/0[89]_*.py scripts/10_*.py scripts/12_*.py; do
  echo
  echo "=== $(basename "$s") ==="
  python "$s"
done

echo; echo "=== robustness, significance and further analyses (14-21) ==="
python scripts/14_robustness_checks.py --seeds 10
python scripts/15_significance_and_ablations.py --sections B --null 200
python scripts/15_significance_and_ablations.py --sections ACD --seeds 10
python scripts/16_null_by_error_model.py --n 100
python scripts/17_null_holdout.py --n 100
python scripts/18_bandwidth_sweep.py --null 200
python scripts/19_section_model.py --seeds 5
python scripts/20_tls_comparison.py --seeds 5
python scripts/21_segmentation_check.py --figure

echo; echo "=== figures ==="
python scripts/13_paper_figures.py

echo
echo "Done. Figures in ./figures/, tables in ./results/"
