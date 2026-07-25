"""Patient-specific (personalized) seizure prediction.

The pooled cross-patient model does not generalize (see the window-level AUC of
~0.47 in the README): preictal signatures differ from patient to patient, so a
decision boundary learned on other people transfers poorly. The clinical
deployment for seizure prediction is therefore almost always *patient-specific*:
a model trained and tested within one person's own recordings.

I evaluate each patient with leave-one-seizure-out cross-validation. Positives
(preictal windows) are grouped by seizure, so the held-out seizure's windows
never appear in training; negatives (interictal windows) are partitioned across
the same folds. A patient needs at least two seizures and enough interictal data
to be evaluable; the rest are reported as skipped.

This is the honest comparison to the cross-patient baseline, and the expected
place for real signal to appear.
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from config import OUTPUTS, RANDOM_STATE


def _patient_folds(y_pat, crop_pat, crops, seed=RANDOM_STATE):
    """Leave-one-seizure-out folds for one patient.

    Returns a list of (train_idx, test_idx) into the patient-local arrays, or an
    empty list with a reason string if the patient can't be evaluated."""
    seiz_crops = [c for c in np.unique(crop_pat) if crops[c]["seizures"]]
    pos = np.where(y_pat == 1)[0]
    neg = np.where(y_pat == 0)[0]
    if len(seiz_crops) < 2:
        return [], f"only {len(seiz_crops)} seizure(s)"
    if len(neg) < 10 or len(pos) < 10:
        return [], f"too few windows (pos={len(pos)}, neg={len(neg)})"

    rng = np.random.default_rng(seed)
    neg_shuf = neg.copy()
    rng.shuffle(neg_shuf)
    neg_parts = np.array_split(neg_shuf, len(seiz_crops))     # disjoint negative folds

    folds = []
    for k, sc in enumerate(seiz_crops):
        test_pos = pos[crop_pat[pos] == sc]
        train_pos = pos[crop_pat[pos] != sc]
        test_neg = neg_parts[k]
        train_neg = np.setdiff1d(neg, test_neg, assume_unique=False)
        if len(test_pos) == 0 or len(test_neg) == 0:
            continue
        folds.append((np.concatenate([train_pos, train_neg]),
                      np.concatenate([test_pos, test_neg])))
    return folds, None


def _metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return auc, sens, spec


def personalized_cv(X, y, groups, meta, model_factory, save=True):
    """Leave-one-seizure-out CV within each patient. Prints a per-patient table
    and the mean over evaluable patients; returns a results dict."""
    crops = meta["crops"]
    per_patient, pooled_true, pooled_prob = {}, [], []

    for patient in sorted(set(groups)):
        pm = groups == patient
        Xp, yp, cp = X[pm], y[pm], meta["crop"][pm]
        folds, reason = _patient_folds(yp, cp, crops)
        if not folds:
            per_patient[patient] = {"skipped": reason}
            continue
        aucs, senss, specs = [], [], []
        for tr, te in folds:
            model = model_factory()
            model.fit(Xp[tr], yp[tr])
            prob = model.predict_proba(Xp[te])[:, 1]
            a, se, sp = _metrics(yp[te], prob)
            aucs.append(a); senss.append(se); specs.append(sp)
            pooled_true.append(yp[te]); pooled_prob.append(prob)
        per_patient[patient] = {
            "seizures": len(folds),
            "auc": float(np.nanmean(aucs)),
            "sensitivity": float(np.nanmean(senss)),
            "specificity": float(np.nanmean(specs))}

    evaluable = {p: r for p, r in per_patient.items() if "auc" in r}
    mean_auc = float(np.nanmean([r["auc"] for r in evaluable.values()])) if evaluable else float("nan")
    pooled_auc = (roc_auc_score(np.concatenate(pooled_true), np.concatenate(pooled_prob))
                  if pooled_true else float("nan"))

    results = {"n_patients_evaluable": len(evaluable),
               "n_patients_total": len(per_patient),
               "mean_patient_auc": mean_auc, "pooled_auc": pooled_auc,
               "per_patient": per_patient}
    _print(results)
    if save:
        (OUTPUTS / "personalized_results.json").write_text(json.dumps(results, indent=2))
        print(f"\nsaved -> {OUTPUTS / 'personalized_results.json'}")
    return results


def _print(r):
    print(f"\nPersonalized (patient-specific) leave-one-seizure-out CV")
    print("-" * 52)
    print(f"{'patient':10}{'seiz':>6}{'AUC':>8}{'Sens':>8}{'Spec':>8}")
    for p in sorted(r["per_patient"]):
        d = r["per_patient"][p]
        if "auc" in d:
            print(f"{p:10}{d['seizures']:>6}{d['auc']:>8.3f}"
                  f"{d['sensitivity']:>8.3f}{d['specificity']:>8.3f}")
        else:
            print(f"{p:10}{'--':>6}{'skip: ' + d['skipped']:>24}")
    print("-" * 52)
    print(f"evaluable patients : {r['n_patients_evaluable']}/{r['n_patients_total']}")
    print(f"mean patient AUC   : {r['mean_patient_auc']:.3f}")
    print(f"pooled AUC         : {r['pooled_auc']:.3f}")
