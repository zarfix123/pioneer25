#!/usr/bin/env python3
"""
SIMPLE FINAL MODEL: East Asian Musical Influence Classifier

Based on our optimization discoveries:
1. 25 trees works better than 200+ trees 
2. Fixed data separators so model learns real patterns
3. 89.4% accuracy is our best achievable performance

This script creates the final production model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def load_data():
    print("=== LOADING FINAL DATASET ===")
    western_data = pd.read_csv('/home/dennis/Projects/research/data/western data.csv')
    influenced_data = pd.read_csv('/home/dennis/Projects/research/data/influenced data.csv')
    combined_data = pd.concat([western_data, influenced_data], ignore_index=True)
    labels = combined_data['influence'].astype(str).str.extract(r'^\s*([01])')[0]
    combined_data['influence'] = pd.to_numeric(labels, errors='coerce')
    feature_names = ['pentatonicism', 'parallel_motion', 'density', 'rhythm_reg', 
                    'syncopation', 'melodic_intervals', 'register_usage', 
                    'articulation', 'dynamics']
    X = combined_data[feature_names]
    y = combined_data['influence']
    groups = combined_data['piece']
    print(f"Final dataset: {len(X)} samples")
    print(f"Non-influenced: {sum(y == 0)} | Influenced: {sum(y == 1)}")
    return X, y, feature_names, groups

def create_final_model():
    print("\n=== CREATING BASE MODEL ===")
    return RandomForestClassifier(
        n_estimators=25,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        bootstrap=True
    )

def strict_grouped_protocol(X, y, groups, feature_names):
    print("\n=== STRICT GROUPED PROTOCOL ===")
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, RandomizedSearchCV
    from sklearn.ensemble import ExtraTreesClassifier
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.values[train_idx], X.values[test_idx]
    y_train, y_test = y.values[train_idx], y.values[test_idx]
    groups_train = groups.values[train_idx]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_space = {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [None, 6, 10, 16, 24],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.7, None],
        'class_weight': ['balanced', None],
        'bootstrap': [True]
    }
    et = ExtraTreesClassifier(random_state=RANDOM_STATE)
    et_space = {
        'n_estimators': [200, 400, 800],
        'max_depth': [None, 10, 16, 24],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.7, None],
        'class_weight': ['balanced', None]
    }
    rf_search = RandomizedSearchCV(rf, rf_space, n_iter=25, scoring='f1', cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True)
    et_search = RandomizedSearchCV(et, et_space, n_iter=20, scoring='f1', cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True)
    print("Tuning RandomForest on train set with grouped CV")
    rf_search.fit(X_train, y_train, groups=groups_train)
    print(f"RF best F1 (CV): {rf_search.best_score_:.3f}")
    print(f"RF best params: {rf_search.best_params_}")
    print("Tuning ExtraTrees on train set with grouped CV")
    et_search.fit(X_train, y_train, groups=groups_train)
    print(f"ET best F1 (CV): {et_search.best_score_:.3f}")
    print(f"ET best params: {et_search.best_params_}")
    if et_search.best_score_ > rf_search.best_score_:
        best_est = 'ExtraTrees'
        best_model = et_search.best_estimator_
        best_cv = et_search.best_score_
    else:
        best_est = 'RandomForest'
        best_model = rf_search.best_estimator_
        best_cv = rf_search.best_score_
    print(f"Selected model: {best_est} (CV F1={best_cv:.3f})")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("\n=== STRICT GROUPED HOLD-OUT RESULTS ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\nDetailed Results:")
    print(classification_report(y_test, y_pred, target_names=['Non-influenced', 'Influenced']))
    return best_model, acc, f1, best_est, best_cv

def show_what_model_learned(model, feature_names):
    print("\n=== WHAT THE MODEL LEARNED ===")
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    print("Most important features for detecting East Asian influence:")
    for i, (feature, importance) in enumerate(feature_importance[:5]):
        print(f"  {i+1}. {feature}: {importance:.1%} importance")
    print(f"\nThis means the model primarily looks at:")
    top_features = [f[0] for f in feature_importance[:3]]
    print(f"  1. {top_features[0]} (most important)")
    print(f"  2. {top_features[1]} (second most)")
    print(f"  3. {top_features[2]} (third most)")

def save_production_model(model, feature_names, accuracy):
    print(f"\n=== SAVING PRODUCTION MODEL ===")
    model_path = '/home/dennis/Projects/research/code/FINAL_MODEL.joblib'
    joblib.dump(model, model_path)
    model_info = {
        'model': model,
        'feature_names': feature_names,
        'accuracy': accuracy,
        'description': 'Strict grouped protocol tuned model',
        'usage': 'Load with joblib.load() and use .predict() with 9 features in order'
    }
    joblib.dump(model_info, '/home/dennis/Projects/research/code/FINAL_MODEL_INFO.joblib')
    print(f"✅ PRODUCTION MODEL SAVED: {model_path}")
    print(f"✅ Accuracy: {accuracy:.1%}")
    print(f"✅ Ready for testing real pieces!")

def create_usage_example():
    print(f"\n=== HOW TO USE THE MODEL ===")
    print("```python")
    print("import joblib")
    print("model = joblib.load('FINAL_MODEL.joblib')")
    print("")
    print("# Test a piece: [pentatonicism, parallel_motion, density, rhythm_reg,")
    print("#                syncopation, melodic_intervals, register_usage, articulation, dynamics]")
    print("new_piece = [2, 1, 1, 2, 1, 0, 2, 1, 1]")
    print("prediction = model.predict([new_piece])[0]")
    print("confidence = max(model.predict_proba([new_piece])[0])")
    print("")
    print("if prediction == 1:")
    print("    print(f'East Asian influenced (confidence: {confidence:.1%})')")
    print("else:")
    print("    print(f'Not influenced (confidence: {confidence:.1%})')")
    print("```")

def main():
    print("FINAL MODEL CREATION")
    print("=" * 40)
    print("Goal: Create production-ready model for East Asian influence detection")
    X, y, feature_names, groups = load_data()
    model, acc, f1, est_name, cv_f1 = strict_grouped_protocol(X, y, groups, feature_names)
    show_what_model_learned(model, feature_names)
    save_production_model(model, feature_names, acc)
    create_usage_example()
    print("\n" + "=" * 40)
    print("🎵 FINAL MODEL COMPLETE! 🎵")
    print(f"Ready to test on real musical pieces!")

if __name__ == "__main__":
    main() 