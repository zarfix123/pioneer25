#!/usr/bin/env python3
"""
Feature Importance Analysis for East Asian Musical Influence Classification

Analyzes and visualizes feature importance, stability, and model behavior.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def load_model_and_data():
    """Load trained model and data."""
    model = joblib.load('/home/dennis/Projects/research/code/FINAL_MODEL.joblib')
    
    with open('/home/dennis/Projects/research/code/FINAL_MODEL_INFO.json', 'r') as f:
        model_info = json.load(f)
    
    # Load data
    western_data = pd.read_csv('/home/dennis/Projects/research/data/western data.csv')
    influenced_data = pd.read_csv('/home/dennis/Projects/research/data/influenced data.csv')
    combined_data = pd.concat([western_data, influenced_data], ignore_index=True)
    
    labels = combined_data['influence'].astype(str).str.extract(r'^\s*([01])')[0]
    combined_data['influence'] = pd.to_numeric(labels, errors='coerce')
    
    feature_names = model_info['feature_names']
    X = combined_data[feature_names]
    y = combined_data['influence']
    groups = combined_data['piece']
    
    return model, model_info, X, y, groups, feature_names

def analyze_global_importance(model, feature_names):
    """Analyze global feature importance."""
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("=== Global Feature Importance ===")
    for i, (feature, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {feature:<20} {importance:.3f}")
    print()
    
    return feature_importance

def analyze_stability_across_folds(X, y, groups, feature_names, model_type, model_params):
    """Analyze feature importance stability across cross-validation folds."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    fold_importances = []
    
    print("=== Analyzing Stability Across Folds ===")
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_fold = X.values[train_idx]
        y_fold = y.values[train_idx]
        
        # Create and train model for this fold
        if model_type == 'ExtraTrees':
            from sklearn.ensemble import ExtraTreesClassifier
            fold_model = ExtraTreesClassifier(random_state=RANDOM_STATE, **model_params)
        else:
            from sklearn.ensemble import RandomForestClassifier
            fold_model = RandomForestClassifier(random_state=RANDOM_STATE, **model_params)
        
        fold_model.fit(X_fold, y_fold)
        fold_importances.append(fold_model.feature_importances_)
        print(f"Fold {fold_idx + 1} complete")
    
    # Calculate stability metrics
    importance_matrix = np.array(fold_importances)
    mean_importance = np.mean(importance_matrix, axis=0)
    std_importance = np.std(importance_matrix, axis=0)
    
    print("\nFeature Stability (mean ± std):")
    for i, feature in enumerate(feature_names):
        print(f"{feature:<20} {mean_importance[i]:.3f} ± {std_importance[i]:.3f}")
    
    # Spearman correlation between folds
    correlations = []
    for i in range(len(fold_importances)):
        for j in range(i+1, len(fold_importances)):
            corr, _ = spearmanr(fold_importances[i], fold_importances[j])
            correlations.append(corr)
    
    mean_correlation = np.mean(correlations)
    print(f"\nMean Spearman correlation between folds: {mean_correlation:.3f}")
    
    return importance_matrix, mean_correlation

def create_importance_plots(feature_importance, importance_matrix, feature_names):
    """Create feature importance visualization."""
    try:
        # Global importance bar chart
        plt.figure(figsize=(10, 6))
        features, importances = zip(*feature_importance)
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(len(features)), importances)
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.title('Global Feature Importance')
        plt.tight_layout()
        
        # Stability plot
        plt.subplot(1, 2, 2)
        for i in range(importance_matrix.shape[1]):
            plt.plot(range(1, 6), importance_matrix[:, i], 'o-', 
                    label=feature_names[i], alpha=0.7)
        plt.xlabel('Fold')
        plt.ylabel('Importance')
        plt.title('Importance Stability Across Folds')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_path = '/home/dennis/Projects/research/code/feature_importances.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved: {output_path}")
        
        # Stability correlation plot
        plt.figure(figsize=(8, 6))
        mean_importance = np.mean(importance_matrix, axis=0)
        std_importance = np.std(importance_matrix, axis=0)
        
        plt.errorbar(range(len(feature_names)), mean_importance, yerr=std_importance,
                    fmt='o', capsize=5)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.title('Feature Importance with Cross-Fold Variation')
        plt.tight_layout()
        
        stability_path = '/home/dennis/Projects/research/code/importance_stability.png'
        plt.savefig(stability_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Stability plot saved: {stability_path}")
        
    except ImportError:
        print("Matplotlib not available - plots not created")

def save_importance_csv(feature_importance, importance_matrix, feature_names):
    """Save feature importance data as CSV."""
    # Global importance
    importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
    importance_df.to_csv('/home/dennis/Projects/research/code/feature_importance.csv', index=False)
    
    # Cross-fold importance
    fold_df = pd.DataFrame(importance_matrix.T, 
                          columns=[f'Fold_{i+1}' for i in range(importance_matrix.shape[0])],
                          index=feature_names)
    fold_df['Mean'] = fold_df.mean(axis=1)
    fold_df['Std'] = fold_df.std(axis=1)
    fold_df.to_csv('/home/dennis/Projects/research/code/importance_by_fold.csv')
    
    print("Importance data saved to CSV files")

def analyze_root_splits(model, feature_names):
    """Analyze which features are used for root splits."""
    try:
        root_features = []
        for tree in model.estimators_:
            root_feature_idx = tree.tree_.feature[0]  # Root node feature
            if root_feature_idx >= 0:  # Valid feature (not leaf)
                root_features.append(feature_names[root_feature_idx])
        
        from collections import Counter
        root_counts = Counter(root_features)
        
        print("=== Root Split Analysis ===")
        print("Features used as root splits:")
        for feature, count in root_counts.most_common():
            percentage = (count / len(root_features)) * 100
            print(f"{feature:<20} {count:3d} ({percentage:5.1f}%)")
        print()
        
        return root_counts
    except:
        print("Root split analysis not available for this model type")
        return {}

def main():
    print("=== Feature Importance Analysis ===")
    
    try:
        model, model_info, X, y, groups, feature_names = load_model_and_data()
        
        print(f"Model: {model_info['model_type']}")
        print(f"Features: {len(feature_names)}")
        print()
        
        # Global importance
        feature_importance = analyze_global_importance(model, feature_names)
        
        # Root splits
        root_counts = analyze_root_splits(model, feature_names)
        
        # Stability analysis
        importance_matrix, correlation = analyze_stability_across_folds(
            X, y, groups, feature_names, 
            model_info['model_type'], 
            model_info.get('hyperparameters', {})
        )
        
        # Create visualizations
        create_importance_plots(feature_importance, importance_matrix, feature_names)
        
        # Save data
        save_importance_csv(feature_importance, importance_matrix, feature_names)
        
        print("\nFeature analysis complete!")
        
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_model.py first.")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()