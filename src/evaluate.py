"""Alarm-based, event-level evaluation of the seizure-prediction system.

Window-level ROC-AUC (see train.py) answers "can the model tell preictal from
interictal windows?". It does NOT answer the clinical question — "how many
seizures does the alarm anticipate, and how often does it cry wolf?" — because
one artefact can flip dozens of adjacent windows and inflate both sensitivity
and the apparent false-alarm count.

This module reports what the prediction literature actually reports:

  * Sensitivity   — fraction of seizures preceded by an alarm inside the
                    prediction window [onset - SPH - SOP, onset - SPH].
  * FPR/h         — false alarms per interictal hour (warning/ictal/postictal
                    time excluded from the denominator).
  * Warning time  — mean lead time from the (first) correct alarm to onset.
  * Chance level  — sensitivity an *unspecific random predictor* firing at the
                    SAME FPR/h would reach by luck (Schelter/Winterhalder), with
                    a binomial p-value. A model that doesn't beat this is noise.

CV is leave-one-patient-out (GroupKFold). The firing-power threshold is chosen
on the TRAINING patients only (never the held-out one), so the operating point
is picked honestly — the fix for a high-sensitivity/low-specificity classifier.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import binom
from sklearn.model_selection import GroupKFold

from config import (SPH_SEC, SOP_SEC, POSTICTAL_SEC, FP_WINDOW_SEC,
                    FP_THRESHOLD, FPR_BUDGET_PER_H, REFRACTORY_SEC, OUTPUTS)
from postprocess import firing_power, raise_alarms


def _merge(intervals):
    """Merge overlapping [lo, hi] intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def _score_crop(times, probs, seizures, duration, threshold):
    """Run the alarm layer on one recording's stream. Returns a dict of counts:
    seizures, predicted, false_alarms, warning_secs (list), interictal_sec."""
    t, fp = firing_power(times, probs, FP_WINDOW_SEC)
    alarms = raise_alarms(t, fp, threshold, REFRACTORY_SEC)

    # prediction window for each seizure: [onset - SPH - SOP, onset - SPH]
    windows = [(o - SPH_SEC - SOP_SEC, o - SPH_SEC, o) for o, _ in seizures]

    predicted, warnings = 0, []
    for lo, hi, onset in windows:
        hits = [a for a in alarms if lo <= a <= hi]
        if hits:
            predicted += 1
            warnings.append(onset - min(hits))     # lead time of the first correct alarm
    # an alarm is FALSE if it lands in no seizure's prediction window
    false_alarms = sum(
        1 for a in alarms
        if not any(lo <= a <= hi for lo, hi, _ in windows)
    )
    # interictal time = duration minus (prediction window + ictal + postictal)
    excluded = _merge([(o - SPH_SEC - SOP_SEC, (f if f is not None else o + 60) + POSTICTAL_SEC)
                       for o, f in seizures])
    excl_sec = sum(min(hi, duration) - max(lo, 0.0)
                   for lo, hi in excluded if hi > 0 and lo < duration)
    interictal_sec = max(0.0, duration - excl_sec)
    return {"seizures": len(seizures), "predicted": predicted,
            "false_alarms": false_alarms, "warnings": warnings,
            "interictal_sec": interictal_sec}


def _aggregate(idx, y_prob, meta, threshold):
    """Alarm metrics over the epochs in `idx` (grouped by crop). Returns
    (sensitivity, fpr_per_h, dict of raw totals)."""
    crop_ids = meta["crop"][idx]
    t0 = meta["t0"][idx]
    prob = y_prob[idx]
    tot = {"seizures": 0, "predicted": 0, "false_alarms": 0, "interictal_sec": 0.0}
    warnings = []
    for cid in np.unique(crop_ids):
        m = crop_ids == cid
        info = meta["crops"][cid]
        s = _score_crop(t0[m], prob[m], info["seizures"], info["duration"], threshold)
        for k in ("seizures", "predicted", "false_alarms", "interictal_sec"):
            tot[k] += s[k]
        warnings += s["warnings"]
    sens = tot["predicted"] / tot["seizures"] if tot["seizures"] else float("nan")
    hours = tot["interictal_sec"] / 3600.0
    fpr = tot["false_alarms"] / hours if hours > 0 else float("nan")
    tot["warning_min"] = float(np.mean(warnings) / 60.0) if warnings else float("nan")
    return sens, fpr, tot


def _tune_threshold(train_idx, y_prob, meta, grid=None):
    """Pick the firing-power threshold on the TRAINING patients: highest
    sensitivity whose FPR/h is within budget; if none qualify, the lowest FPR/h.
    Selection never sees the held-out test patient."""
    grid = grid if grid is not None else np.round(np.arange(0.30, 0.91, 0.05), 2)
    scored = []
    for thr in grid:
        sens, fpr, _ = _aggregate(train_idx, y_prob, meta, thr)
        scored.append((thr, sens, fpr))
    within = [s for s in scored if np.isfinite(s[2]) and s[2] <= FPR_BUDGET_PER_H]
    if within:
        return max(within, key=lambda s: (s[1], -s[2]))[0]
    return min(scored, key=lambda s: (np.inf if not np.isfinite(s[2]) else s[2]))[0]


def _chance(sens, fpr_per_h, n_seizures, n_predicted):
    """Unspecific-random-predictor sensitivity at the observed FPR/h, and the
    binomial p-value of doing at least as well as the model by luck.

    A random alarm process at rate `fpr` predicts a given seizure with
    probability p = 1 - exp(-fpr * SOP) (an alarm falling in its SOP window)."""
    if not np.isfinite(fpr_per_h) or n_seizures == 0:
        return float("nan"), float("nan")
    lam = fpr_per_h / 3600.0
    p = 1.0 - np.exp(-lam * SOP_SEC)
    p = min(max(p, 1e-9), 1 - 1e-9)
    pval = float(binom.sf(n_predicted - 1, n_seizures, p))   # P(>= observed by chance)
    return float(p), pval


def alarm_evaluate(X, y, groups, meta, model_factory, model_name="logreg",
                   n_splits=None, save=True):
    """Leave-one-patient-out alarm evaluation. `model_factory()` returns a fresh
    sklearn pipeline with predict_proba. Prints a table, returns a results dict."""
    n_groups = len(set(groups))
    n_splits = n_splits or min(5, n_groups)
    if n_splits < 2:
        raise RuntimeError(f"Need >=2 subjects for grouped CV, got {n_groups}.")
    gkf = GroupKFold(n_splits=n_splits)
    idx_all = np.arange(len(y))

    pooled = {op: {"seizures": 0, "predicted": 0, "false_alarms": 0,
                   "interictal_sec": 0.0, "warn": []}
              for op in ("default", "tuned")}
    per_fold = []

    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        model = model_factory()
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X)[:, 1]                 # score every epoch once

        thr_tuned = _tune_threshold(tr, prob, meta)
        row = {"patients_test": sorted(set(groups[te]))}
        for op, thr in (("default", FP_THRESHOLD), ("tuned", thr_tuned)):
            sens, fpr, tot = _aggregate(te, prob, meta, thr)
            row[op] = {"threshold": float(thr), "sensitivity": sens,
                       "fpr_per_h": fpr, "warning_min": tot["warning_min"],
                       **{k: tot[k] for k in ("seizures", "predicted", "false_alarms")},
                       "interictal_h": tot["interictal_sec"] / 3600.0}
            P = pooled[op]
            for k in ("seizures", "predicted", "false_alarms", "interictal_sec"):
                P[k] += tot[k]
            if np.isfinite(tot["warning_min"]):
                P["warn"].append(tot["warning_min"])
        per_fold.append(row)

    results = {"model": model_name, "sph_min": SPH_SEC / 60, "sop_min": SOP_SEC / 60,
               "fpr_budget_per_h": FPR_BUDGET_PER_H, "folds": len(per_fold),
               "per_fold": per_fold, "pooled": {}}
    for op, P in pooled.items():
        sens = P["predicted"] / P["seizures"] if P["seizures"] else float("nan")
        hours = P["interictal_sec"] / 3600.0
        fpr = P["false_alarms"] / hours if hours > 0 else float("nan")
        chance_sens, pval = _chance(sens, fpr, P["seizures"], P["predicted"])
        results["pooled"][op] = {
            "sensitivity": sens, "fpr_per_h": fpr,
            "warning_min": float(np.mean(P["warn"])) if P["warn"] else float("nan"),
            "seizures": P["seizures"], "predicted": P["predicted"],
            "false_alarms": P["false_alarms"], "interictal_h": hours,
            "chance_sensitivity": chance_sens, "p_value_vs_chance": pval}

    _print(results)
    if save:
        (OUTPUTS / "alarm_results.json").write_text(json.dumps(results, indent=2))
        print(f"\nsaved -> {OUTPUTS / 'alarm_results.json'}")
    return results


def _print(r):
    print(f"\nAlarm-based evaluation  ·  model={r['model']}  ·  "
          f"SPH={r['sph_min']:.0f} min  SOP={r['sop_min']:.0f} min  "
          f"·  leave-one-patient-out ({r['folds']} folds)")
    print("-" * 72)
    print(f"{'operating point':<16}{'Sens':>7}{'FPR/h':>8}{'warn(min)':>11}"
          f"{'seiz':>6}{'pred':>6}{'FA':>5}{'chance':>8}{'p':>8}")
    for op in ("default", "tuned"):
        p = r["pooled"][op]
        label = "default θ=0.5" if op == "default" else "tuned θ (train)"
        def f(x, w, d=3):
            return (f"{x:>{w}.{d}f}" if isinstance(x, float) and np.isfinite(x)
                    else f"{'--':>{w}}")
        print(f"{label:<16}{f(p['sensitivity'],7)}{f(p['fpr_per_h'],8,2)}"
              f"{f(p['warning_min'],11,1)}{p['seizures']:>6}{p['predicted']:>6}"
              f"{p['false_alarms']:>5}{f(p['chance_sensitivity'],8)}"
              f"{f(p['p_value_vs_chance'],8)}")
    tuned = r["pooled"]["tuned"]
    verdict = ("beats chance" if np.isfinite(tuned["p_value_vs_chance"])
               and tuned["p_value_vs_chance"] < 0.05 else "NOT distinguishable from chance")
    print(f"\n  read: tuned operating point is {verdict} "
          f"(p={tuned['p_value_vs_chance']:.3f}) on {tuned['seizures']} seizures.")
