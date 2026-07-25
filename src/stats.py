"""Statistical rigor for the evaluation: AUPRC, confidence intervals, and a
label-permutation significance test.

Why each is here, with the reference a reviewer would expect:

* AUPRC (average precision). ROC-AUC can look optimistic under class imbalance
  because it rewards ranking the abundant negatives; the area under the
  precision-recall curve focuses on the rarer positive (preictal) class, which
  is what matters clinically (Saito and Rehmsmeier 2015; Davis and Goadrich 2006).

* Patient-clustered bootstrap confidence interval. Windows within a patient are
  correlated, so a naive per-window bootstrap understates uncertainty. I resample
  whole patients with replacement (a cluster bootstrap) and report the 2.5 to
  97.5 percentile interval of the pooled AUC.

* Label-permutation p-value. The non-parametric null for a decoding score:
  shuffle the labels many times, recompute the statistic, and read off the
  fraction of the null at or above the observed value (Combrisson and Jerbi
  2015; Ojala and Garriga 2010). This complements the analytical random-predictor
  test used in the alarm layer (Schelter 2006; Snyder 2008).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from config import RANDOM_STATE


def auprc(y, prob):
    y = np.asarray(y)
    return float(average_precision_score(y, prob)) if len(set(y)) > 1 else float("nan")


def bootstrap_auc_ci(y, prob, patient=None, n_boot=2000, alpha=0.05, seed=RANDOM_STATE):
    """Percentile bootstrap CI for ROC-AUC. If `patient` (one id per sample) is
    given, resample whole patients (cluster bootstrap) to respect within-patient
    correlation; otherwise resample samples."""
    y = np.asarray(y); prob = np.asarray(prob)
    rng = np.random.default_rng(seed)
    boots = []
    if patient is not None:
        patient = np.asarray(patient)
        groups = np.unique(patient)
        idx_by_group = {g: np.where(patient == g)[0] for g in groups}
        for _ in range(n_boot):
            pick = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([idx_by_group[g] for g in pick])
            if len(set(y[idx])) > 1:
                boots.append(roc_auc_score(y[idx], prob[idx]))
    else:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(set(y[idx])) > 1:
                boots.append(roc_auc_score(y[idx], prob[idx]))
    if not boots:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def permutation_pvalue(y, prob, n_perm=5000, seed=RANDOM_STATE):
    """Label-permutation p-value for ROC-AUC against the chance null (0.5).
    p = (1 + #{permuted AUC >= observed}) / (n_perm + 1)."""
    y = np.asarray(y); prob = np.asarray(prob)
    if len(set(y)) < 2:
        return float("nan")
    observed = roc_auc_score(y, prob)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        if roc_auc_score(rng.permutation(y), prob) >= observed:
            ge += 1
    return (1 + ge) / (n_perm + 1)
