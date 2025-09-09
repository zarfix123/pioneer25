# East Asian Influence Detection in Classical Music

Detects East Asian musical influence in Western piano works using 9 symbolic features and a tree-ensemble classifier. Repo includes frozen splits, predictions, figures, and analysis artifacts for full reproducibility without retraining.

## Data format

One CSV with all labeled segments:

Required columns (in order):
`piece,start,end,pentatonicism,parallel_motion,density,rhythm_reg,syncopation,melodic_intervals,register_usage,articulation,dynamics,influence`

Required files:

* `data/features_9x.csv` — full dataset
* `data/segments_manifest.csv` — `piece,start,end` (one row per segment; mirrors `features_9x.csv`)

## Repo layout (key files)

```
data/
  features_9x.csv
  segments_manifest.csv
models/
  FINAL_MODEL.joblib
  FINAL_MODEL_INFO.json
outputs/
  predictions_test.csv
  predictions_oof.csv
  metrics_summary.json
  piece_level_metrics.csv
  permutation_importance_test.csv
  feature_importances_cv.csv
  feature_ranks_cv.csv
  root_splits_per_fold.csv
  root_splits_overall.csv
  early_tree_splits_overall.csv
  feature_rank_spearman.csv
code/
  train_model.py
  analyze_features.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train (optional)


```bash
python code/train_model.py
```

This writes:

* `data/splits/SPLIT_PIECES.json` (piece-level frozen split)
* `models/FINAL_MODEL.joblib` (trained estimator)
* `models/FINAL_MODEL_INFO.json` (algo, hyperparams, versions, split hash, metrics)
* `outputs/predictions_test.csv` and `outputs/predictions_oof.csv`

## Results

Headline metrics are stored in:

* `outputs/metrics_summary.json` (hold-out + grouped CV)
* `code/images/confusion_matrix.png`
* Per-piece: `outputs/piece_level_metrics.csv`

## Notes on reproducibility

* Grouped, leak-free protocol (by `piece`) for split, CV, and OoF.
* `data/splits/SPLIT_PIECES.json` is frozen; predictions/figures are generated from committed CSVs, not in-sample fits.
* Random seed: `42`. Library versions in `models/FINAL_MODEL_INFO.json`.

## Citation

Please cite the repository release (see `CITATION.cff`).
Paper citation will be added upon publication.

## LicenseSPLIT_PIECES

MIT (see `LICENSE`).