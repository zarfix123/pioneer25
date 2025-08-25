#!/usr/bin/env python3
"""
Model Evaluation Script for East Asian Musical Influence Classification

Evaluates the trained model and prints detailed performance metrics.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def load_model_and_data():
    """Load trained model and test data."""
    # Load model
    model = joblib.load('/home/dennis/Projects/research/code/FINAL_MODEL.joblib')
    
    # Load metadata
    with open('/home/dennis/Projects/research/code/FINAL_MODEL_INFO.json', 'r') as f:
        model_info = json.load(f)
    
    # Load data
    western_data = pd.read_csv('/home/dennis/Projects/research/data/western data.csv')
    influenced_data = pd.read_csv('/home/dennis/Projects/research/data/influenced data.csv')
    combined_data = pd.concat([western_data, influenced_data], ignore_index=True)
    
    # Clean labels
    labels = combined_data['influence'].astype(str).str.extract(r'^\s*([01])')[0]
    combined_data['influence'] = pd.to_numeric(labels, errors='coerce')
    
    feature_names = model_info['feature_names']
    X = combined_data[feature_names]
    y = combined_data['influence']
    groups = combined_data['piece']
    
    return model, model_info, X, y, groups, feature_names

def evaluate_model(model, X, y, groups):
    """Evaluate model with same train/test split as training."""
    # Use same split as training
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_test, y_test = X.values[test_idx], y.values[test_idx]
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("=== Model Performance ===")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Test F1 Score: {f1:.3f}")
    print()
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Non-influenced', 'Influenced'],
                              digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Non-inf  Influenced")
    print(f"Non-inf      {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"Influenced   {cm[1,0]:3d}       {cm[1,1]:3d}")
    
    return accuracy, f1, cm

def cross_validate_model(model, X, y, groups):
    """Perform grouped cross-validation."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    cv_scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring='f1')
    cv_accuracy = cross_val_score(model, X, y, groups=groups, cv=cv, scoring='accuracy')
    
    print("=== Cross-Validation Results ===")
    print(f"CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"CV Accuracy: {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")
    print()

def save_confusion_matrix(cm, output_path='/home/dennis/Projects/research/code/confusion_matrix.png'):
    """Save confusion matrix as PNG."""
    try:
        plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ['Non-influenced', 'Influenced']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {output_path}")
    except ImportError:
        print("Matplotlib not available - confusion matrix not saved")

def main():
    print("=== Model Evaluation ===")
    
    try:
        model, model_info, X, y, groups, feature_names = load_model_and_data()
        
        print(f"Model type: {model_info['model_type']}")
        print(f"Training CV F1: {model_info['cv_f1']:.3f}")
        print()
        
        # Evaluate on test set
        accuracy, f1, cm = evaluate_model(model, X, y, groups)
        
        # Cross-validation
        cross_validate_model(model, X, y, groups)
        
        # Save confusion matrix
        save_confusion_matrix(cm)
        
        print("Evaluation complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_model.py first to create the model.")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()