#!/usr/bin/env python3
"""
Model Training Pipeline for East Asian Musical Influence Classification

Trains Extra Trees model with optimal hyperparameters and saves final model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def load_data():
    """Load and preprocess labeled CSV data."""
    western_data = pd.read_csv('/home/dennis/Projects/research/data/western data.csv')
    influenced_data = pd.read_csv('/home/dennis/Projects/research/data/influenced data.csv')
    combined_data = pd.concat([western_data, influenced_data], ignore_index=True)
    
    # Clean influence labels
    labels = combined_data['influence'].astype(str).str.extract(r'^\s*([01])')[0]
    combined_data['influence'] = pd.to_numeric(labels, errors='coerce')
    
    # Define feature columns
    feature_names = ['pentatonicism', 'parallel_motion', 'density', 'rhythm_reg', 
                    'syncopation', 'melodic_intervals', 'register_usage', 
                    'articulation', 'dynamics']
    
    X = combined_data[feature_names]
    y = combined_data['influence']
    groups = combined_data['piece']
    
    print(f"Dataset: {len(X)} samples")
    print(f"Classes: {sum(y == 0)} non-influenced, {sum(y == 1)} influenced")
    
    return X, y, feature_names, groups

def train_model(X, y, groups, feature_names):
    """Train model with grouped cross-validation and hyperparameter tuning."""
    # Split data by piece groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.values[train_idx], X.values[test_idx]
    y_train, y_test = y.values[train_idx], y.values[test_idx]
    groups_train = groups.values[train_idx]
    
    # Grouped cross-validation
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Hyperparameter search spaces
    rf_space = {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [None, 6, 10, 16, 24],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.7, None],
        'class_weight': ['balanced', None]
    }
    
    et_space = {
        'n_estimators': [200, 400, 800],
        'max_depth': [None, 10, 16, 24],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.7, None],
        'class_weight': ['balanced', None]
    }
    
    # Train and tune models
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    et = ExtraTreesClassifier(random_state=RANDOM_STATE)
    
    rf_search = RandomizedSearchCV(rf, rf_space, n_iter=25, scoring='f1', cv=cv, 
                                  random_state=RANDOM_STATE, n_jobs=-1)
    et_search = RandomizedSearchCV(et, et_space, n_iter=20, scoring='f1', cv=cv,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    
    print("Training RandomForest...")
    rf_search.fit(X_train, y_train, groups=groups_train)
    
    print("Training ExtraTrees...")
    et_search.fit(X_train, y_train, groups=groups_train)
    
    # Select best model
    if et_search.best_score_ > rf_search.best_score_:
        best_model = et_search.best_estimator_
        best_params = et_search.best_params_
        model_type = 'ExtraTrees'
        cv_score = et_search.best_score_
    else:
        best_model = rf_search.best_estimator_
        best_params = rf_search.best_params_
        model_type = 'RandomForest'
        cv_score = rf_search.best_score_
    
    # Evaluate on test set
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"Best model: {model_type}")
    print(f"CV F1: {cv_score:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Test F1: {test_f1:.3f}")
    
    return best_model, {
        'model_type': model_type,
        'cv_f1': cv_score,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'feature_names': feature_names,
        'hyperparameters': best_params
    }

def save_model(model, model_info):
    """Save trained model and metadata."""
    # Save model
    model_path = '/home/dennis/Projects/research/code/FINAL_MODEL.joblib'
    joblib.dump(model, model_path)
    
    # Save metadata
    info_path = '/home/dennis/Projects/research/code/FINAL_MODEL_INFO.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {info_path}")

def main():
    print("=== East Asian Influence Model Training ===")
    X, y, feature_names, groups = load_data()
    model, model_info = train_model(X, y, groups, feature_names)
    save_model(model, model_info)
    print("Training complete!")

if __name__ == "__main__":
    main()