# East Asian Influence Detection in Classical Music

Machine learning model that detects East Asian musical influence using symbolic features extracted from musical scores.

## Dataset

The model expects labeled CSV files with these columns in order:
- `piece`: Musical work name  
- `start`, `end`: Measure numbers for 4-bar segments
- `pentatonicism`: Pentatonic scale usage (0-2)
- `parallel_motion`: Parallel voice movement (0-2) 
- `density`: Harmonic/textural complexity (0-2)
- `rhythm_reg`: Rhythm regularity (0-2)
- `syncopation`: Off-beat emphasis (0-2)
- `melodic_intervals`: Size of melodic jumps (0-2)
- `register_usage`: Range of pitches used (0-2)
- `articulation`: Legato vs. detached playing (0-2)
- `dynamics`: Volume variation markings (0-2)
- `influence`: Binary label (0=Western, 1=East Asian influenced)

Required data files:
- `data/western data.csv`: Western classical segments
- `data/influenced data.csv`: East Asian influenced segments

## Usage

### Training
```bash
python train_model.py
```
Outputs:
- `FINAL_MODEL.joblib`: Trained Extra Trees classifier
- `FINAL_MODEL_INFO.json`: Model metadata and performance metrics

### Evaluation  
```bash
python evaluate_model.py
```
Prints accuracy, F1 score, classification report, and confusion matrix.
Outputs: `confusion_matrix.png`

### Feature Analysis
```bash
python analyze_features.py
```
Analyzes feature importance and stability across cross-validation folds.
Outputs:
- `feature_importances.png`: Global importance bar chart
- `importance_stability.png`: Cross-fold variation plot
- `feature_importance.csv`: Importance rankings
- `importance_by_fold.csv`: Fold-by-fold importance values

### Data Preprocessing
```bash
python parse_labels.py
```
Validates and cleans the dataset, ensures proper feature ordering.
Outputs: `data/cleaned_dataset.csv`

## Model Performance

The Extra Trees classifier achieves ~89% accuracy using grouped cross-validation to prevent data leakage between musical pieces. Key features for detecting East Asian influence include pentatonicism, parallel motion, and register usage patterns.

## Citation

*Paper citation will be added upon publication.*