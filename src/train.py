"""Classical-ML baselines with leakage-free, subject-grouped cross-validation.

We deliberately start classical (this file) before any deep learning: logistic
regression, random forest, and an RBF-SVM, each on standardised features,
evaluated with GroupKFold so no subject appears in both train and test.
Reports ROC-AUC, sensitivity, specificity, F1 — the metrics that matter for a
seizure-prediction alarm.
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

from config import RANDOM_STATE, OUTPUTS


def _models():
    return {
        "logreg": Pipeline([
            ("imp", SimpleImputer()), ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=RANDOM_STATE))]),
        "rf": Pipeline([
            ("imp", SimpleImputer()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            n_jobs=-1, random_state=RANDOM_STATE))]),
        "svm_rbf": Pipeline([
            ("imp", SimpleImputer()), ("sc", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
                ensemble=False, cv=3))]),
    }


def _metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # recall of preictal
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"auc": auc, "sensitivity": sens, "specificity": spec,
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def cross_validate(X, y, groups, n_splits=None):
    """GroupKFold CV across models. Returns results dict + prints a table."""
    n_groups = len(set(groups))
    n_splits = n_splits or min(5, n_groups)
    if n_splits < 2:
        raise RuntimeError(f"Need >=2 subjects for grouped CV, got {n_groups}.")
    gkf = GroupKFold(n_splits=n_splits)
    results = {}
    for name, model in _models().items():
        fold_metrics = []
        for tr, te in gkf.split(X, y, groups):
            if len(set(y[tr])) < 2:
                continue
            model.fit(X[tr], y[tr])
            prob = model.predict_proba(X[te])[:, 1]
            fold_metrics.append(_metrics(y[te], prob))
        agg = {k: float(np.nanmean([m[k] for m in fold_metrics]))
               for k in ["auc", "sensitivity", "specificity", "f1"]} if fold_metrics else {}
        results[name] = {"folds": len(fold_metrics), **agg}

    print(f"\n{'model':10} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'F1':>6}  (GroupKFold, {n_splits} folds)")
    print("-" * 46)
    for name, r in results.items():
        if r.get("auc") is not None and "auc" in r:
            print(f"{name:10} {r['auc']:6.3f} {r['sensitivity']:6.3f} "
                  f"{r['specificity']:6.3f} {r['f1']:6.3f}")
    (OUTPUTS / "cv_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {OUTPUTS / 'cv_results.json'}")
    return results
