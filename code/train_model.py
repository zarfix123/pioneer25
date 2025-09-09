#!/usr/bin/env python3
"""
Model Training Pipeline (Leak-Free, Option 1)

- Freeze grouped Train/Test split by piece
- Tune on Train only with StratifiedGroupKFold
- Evaluate once on frozen Test
- Save Test predictions and Train OoF predictions for visualization
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, average_precision_score, roc_auc_score
)
import joblib
import sklearn

RANDOM_STATE = 42
TEST_SIZE = 0.35  # use 0.35 if that's your final protocol; change to 0.30 if you prefer
BASE = "/home/dennis/Projects/research"
DATA_DIR = f"{BASE}/data"
CODE_DIR = f"{BASE}/code"
OUT_DIR  = f"{BASE}/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "pentatonicism", "parallel_motion", "density", "rhythm_reg",
    "syncopation", "melodic_intervals", "register_usage",
    "articulation", "dynamics"
]

def load_data():
    west = pd.read_csv(f"{DATA_DIR}/western data.csv")
    infl = pd.read_csv(f"{DATA_DIR}/influenced data.csv")
    df = pd.concat([west, infl], ignore_index=True)

    # clean binary labels from {0,1} strings
    labels = df["influence"].astype(str).str.extract(r"^\s*([01])")[0]
    df["influence"] = pd.to_numeric(labels, errors="coerce")

    # minimal sanity
    assert set(df["influence"].dropna().unique()) <= {0,1}, "Labels must be 0/1"

    X = df[FEATURES].copy()
    y = df["influence"].astype(int).values
    groups = df["piece"].astype(str).values

    # keep meta for saving preds
    meta_cols = ["piece", "start", "end"]
    for c in meta_cols:
        if c not in df.columns:
            df[c] = np.nan
    meta = df[["piece", "start", "end"]].copy()

    print(f"Dataset: {len(X)} segments | classes: {(y==0).sum()} non-inf, {(y==1).sum()} inf")
    print(f"Pieces: {df['piece'].nunique()}")

    return df, X, y, groups, meta

def freeze_split(groups):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(np.zeros_like(groups), np.zeros_like(groups), groups=groups))
    return train_idx, test_idx

def hyperparam_search(X_train, y_train, groups_train):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    et = ExtraTreesClassifier(random_state=RANDOM_STATE)

    rf_space = {
        'n_estimators': [100, 200, 400],
        'max_depth': [None, 10, 16, 24],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.7, 1.0],
        'class_weight': ['balanced', None]
    }

    et_space = {
        'n_estimators': [200, 400, 800],
        'max_depth': [None, 10, 16, 24],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.7, 1.0],  # 1.0 == all features
        'class_weight': ['balanced', None]
    }

    rf_search = RandomizedSearchCV(
        rf, rf_space, n_iter=20, scoring='f1', cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True
    )
    et_search = RandomizedSearchCV(
        et, et_space, n_iter=20, scoring='f1', cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True
    )

    print("Tuning RandomForest (train-only CV)...")
    rf_search.fit(X_train, y_train, groups=groups_train)

    print("Tuning ExtraTrees (train-only CV)...")
    et_search.fit(X_train, y_train, groups=groups_train)

    if et_search.best_score_ >= rf_search.best_score_:
        best = et_search
        model_type = "ExtraTrees"
    else:
        best = rf_search
        model_type = "RandomForest"

    print(f"Best model: {model_type}  | CV F1: {best.best_score_:.3f}")
    return model_type, best.best_estimator_, best.best_params_, best.best_score_

def eval_on_test(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)  # threshold policy: 0.5; adjust if you later calibrate

    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    p0,r0,f10,_ = precision_recall_fscore_support(y_test, y_pred, labels=[0], average=None, zero_division=0)
    p1,r1,f11,_ = precision_recall_fscore_support(y_test, y_pred, labels=[1], average=None, zero_division=0)
    ap   = average_precision_score(y_test, y_prob)  # AUPRC (positive=1)
    try:
        auc  = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = np.nan
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision_noninf": p0[0], "recall_noninf": r0[0], "f1_noninf": f10[0],
        "precision_inf": p1[0],    "recall_inf": r1[0],   "f1_inf":  f11[0],
        "auprc_inf": ap, "auroc": auc,
        "confusion_matrix": cm.tolist()
    }
    return metrics, y_pred, y_prob

def majority_vote_piece_level(meta, y_true, y_pred):
    df = meta.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    agg = df.groupby("piece").agg(
        true=('y_true', lambda v: int(np.round(v.mean()))),  # if piece is mixed this is coarse
        pred=('y_pred', lambda v: int(np.round(v.mean())))
    ).reset_index()
    piece_acc = (agg["true"] == agg["pred"]).mean()
    return piece_acc, agg

def main():
    print("=== Leak-free training (Option 1) ===")
    raw, X, y, groups, meta = load_data()

    # 1) Freeze grouped split and save it
    train_idx, test_idx = freeze_split(groups)
    X_train, X_test = X.values[train_idx], X.values[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    meta_train, meta_test = meta.iloc[train_idx].reset_index(drop=True), meta.iloc[test_idx].reset_index(drop=True)

    split_info = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_pieces": sorted(set(groups[train_idx].tolist())),
        "test_pieces": sorted(set(groups[test_idx].tolist()))
    }
    with open(f"{OUT_DIR}/SPLIT_PIECES.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split pieces to {OUT_DIR}/SPLIT_PIECES.json")

    # 2) Hyperparam tuning on Train only
    model_type, best_estimator, best_params, cv_f1 = hyperparam_search(X_train, y_train, groups_train)

    # 3) Fit final model on all Train
    best_estimator.fit(X_train, y_train)

    # 4) Evaluate once on frozen Test
    test_metrics, test_pred, test_prob = eval_on_test(best_estimator, X_test, y_test)
    print("=== TEST metrics (frozen) ===")
    for k,v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.3f}" if isinstance(v,float) else f"{k}: {v}")
    print(f"confusion_matrix (rows=[noninf,inf], cols=[pred0,pred1]): {test_metrics['confusion_matrix']}")

    # 5) Save Test predictions for figures
    test_out = meta_test.copy()
    test_out["y_true"] = y_test
    test_out["proba_infl"] = test_prob
    test_out["proba_noninfl"] = 1.0 - test_prob
    test_out["y_pred"] = test_pred
    test_out["mode"] = "test"
    test_out.to_csv(f"{OUT_DIR}/TEST_PREDICTIONS.csv", index=False)
    print(f"Saved test predictions to {OUT_DIR}/TEST_PREDICTIONS.csv")

    # (optional) piece-level majority vote on Test
    piece_acc, piece_table = majority_vote_piece_level(meta_test, y_test, test_pred)
    print(f"Piece-level accuracy (test, majority vote): {piece_acc:.3f}")
    piece_table.to_csv(f"{OUT_DIR}/TEST_PIECE_VOTE.csv", index=False)

    # 6) Train OoF predictions on Train for visualization
    #    (same hyperparams; grouped CV; out-of-fold predicted probabilities)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base = best_estimator.__class__(**best_estimator.get_params())
    oof_prob = cross_val_predict(
        base, X_train, y_train, groups=groups_train, cv=cv,
        method="predict_proba", n_jobs=-1
    )[:,1]
    oof_pred = (oof_prob >= 0.5).astype(int)
    train_oof = meta_train.copy()
    train_oof["y_true"] = y_train
    train_oof["proba_infl"] = oof_prob
    train_oof["proba_noninfl"] = 1.0 - oof_prob
    train_oof["y_pred"] = oof_pred
    train_oof["mode"] = "oof_train"
    train_oof.to_csv(f"{OUT_DIR}/TRAIN_OOF_PREDICTIONS.csv", index=False)
    print(f"Saved train OoF predictions to {OUT_DIR}/TRAIN_OOF_PREDICTIONS.csv")

    # 7) Save model + info (with exact hyperparams and versions)
    model_path = f"{CODE_DIR}/FINAL_MODEL.joblib"
    info_path  = f"{CODE_DIR}/FINAL_MODEL_INFO.json"

    joblib.dump(best_estimator, model_path)

    model_info = {
        "model_type": model_type,
        "feature_names": FEATURES,
        "hyperparameters": best_params,
        "cv_train_f1": cv_f1,
        "test_metrics": test_metrics,
        "split_info_path": f"{OUT_DIR}/SPLIT_PIECES.json",
        "predictions": {
            "test": f"{OUT_DIR}/TEST_PREDICTIONS.csv",
            "train_oof": f"{OUT_DIR}/TRAIN_OOF_PREDICTIONS.csv"
        },
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__
    }
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {info_path}")
    print("Done.")

if __name__ == "__main__":
    main()
