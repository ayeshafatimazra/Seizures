"""Slice preprocessed recordings into labelled epochs and build the
feature matrix X, label vector y, and group vector (subject id, for
leakage-free cross-validation).

Labelling (seizure PREDICTION):
    preictal   window in [onset - PREICTAL - SPH, onset - SPH)  -> y = 1
    interictal window >INTERICTAL_GUARD from every seizure       -> y = 0
    ictal / postictal / SPH gap                                  -> dropped
"""
from __future__ import annotations
import numpy as np

from config import (WINDOW_SEC, STEP_SEC, PREICTAL_SEC, SPH_SEC,
                    POSTICTAL_SEC, INTERICTAL_GUARD_SEC)
from features import extract


def _label_epoch(t0, t1, seizures):
    """Return 1 (preictal), 0 (interictal), or None (drop) for [t0, t1)."""
    if not seizures:
        return 0                                   # seizure-free recording = interictal
    for onset, offset in seizures:
        offset = offset if offset is not None else onset + 60
        # inside ictal or SPH gap or postictal -> drop
        if onset - SPH_SEC <= t1 and t0 <= offset + POSTICTAL_SEC:
            if (onset - PREICTAL_SEC - SPH_SEC) <= t0 < (onset - SPH_SEC):
                return 1                           # preictal
            return None                            # ictal / gap / postictal
        # preictal window (before the SPH gap)
        if (onset - PREICTAL_SEC - SPH_SEC) <= t0 < (onset - SPH_SEC):
            return 1
    # far from every seizure?
    if all(abs(t0 - o) > INTERICTAL_GUARD_SEC and abs(t0 - (f or o)) > INTERICTAL_GUARD_SEC
           for o, f in seizures):
        return 0
    return None


def build_epochs(recordings, verbose=False):
    """Slice recordings into labelled epochs.

    Returns X (n_epochs, n_feat), y, groups (subjects), feat_names, and `meta` —
    the per-epoch timing needed to reconstruct each recording's probability
    stream for the alarm layer:

        meta["crop"] : int array, which recording (== continuous crop) each epoch belongs to
        meta["t0"]   : float array, epoch start time (s) within its crop
        meta["crops"]: list[dict] indexed by crop id, {subject, seizures, duration}

    Each Recording is one continuous crop (the loader already carves multi-hour
    EDFs into short preictal + interictal segments), so grouping epochs by
    meta["crop"] and sorting by t0 gives a gap-aware, time-ordered stream.
    """
    X, y, groups = [], [], []
    ep_crop, ep_t0 = [], []
    crops = []
    feat_names = None
    for cid, rec in enumerate(recordings):
        crops.append({"subject": rec.subject,
                      "seizures": list(rec.seizures),
                      "duration": float(rec.duration_sec)})
        w = int(WINDOW_SEC * rec.sfreq)
        step = int(STEP_SEC * rec.sfreq)
        n = rec.data.shape[1]
        kept = {0: 0, 1: 0}
        for start in range(0, n - w + 1, step):
            t0, t1 = start / rec.sfreq, (start + w) / rec.sfreq
            lab = _label_epoch(t0, t1, rec.seizures)
            if lab is None:
                continue
            vec, names = extract(rec.data[:, start:start + w], rec.sfreq, rec.ch_names)
            feat_names = feat_names or names
            X.append(vec)
            y.append(lab)
            groups.append(rec.subject)
            ep_crop.append(cid)
            ep_t0.append(t0)
            kept[lab] += 1
        if verbose:
            print(f"  [{rec.subject}] preictal={kept[1]:4d} interictal={kept[0]:4d}"
                  f"  seizures={len(rec.seizures)}")
    if not X:
        raise RuntimeError("No epochs built — check labelling windows vs recording length.")
    meta = {"crop": np.asarray(ep_crop), "t0": np.asarray(ep_t0), "crops": crops}
    return np.asarray(X), np.asarray(y), np.asarray(groups), feat_names, meta
