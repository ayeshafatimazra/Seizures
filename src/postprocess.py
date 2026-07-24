"""Turn a noisy stream of per-window preictal probabilities into discrete
alarms — the step that separates a *classifier* from a seizure-prediction
*system*.

Raw window-level predictions flicker: a single artefact flips a handful of
5 s windows to "preictal" and, counted individually, each looks like a false
alarm. The literature's fix is the **Firing Power** method (Teixeira et al.
2012; see also Nature Sci. Rep. 2023 "post-processing stage as a chronology"):

  1. smooth the preictal probability over a trailing window the length of the
     Seizure Occurrence Period (SOP) — the firing power in [0, 1];
  2. raise ONE alarm when firing power crosses a threshold;
  3. stay silent for a refractory period (the predicted window must elapse),
     so one sustained preictal build-up yields one alarm, not hundreds.

Everything here is time-based, not index-based, so it tolerates the gaps left
where ictal / SPH windows were dropped during labelling.
"""
from __future__ import annotations
import numpy as np


def firing_power(times, probs, window_sec):
    """Trailing time-average of the preictal probability.

    fp[i] = mean(probs[j] for j with times[i] - window_sec < times[j] <= times[i]).
    `times`/`probs` need not be pre-sorted; the returned array is aligned to the
    time-sorted order (see `sort_stream`).
    """
    times = np.asarray(times, float)
    probs = np.asarray(probs, float)
    order = np.argsort(times, kind="stable")
    t, p = times[order], probs[order]
    fp = np.empty_like(p)
    lo = 0
    for i in range(len(t)):
        while t[i] - t[lo] > window_sec:      # slide the left edge forward
            lo += 1
        fp[i] = p[lo:i + 1].mean()
    return t, fp


def raise_alarms(times, fp, threshold, refractory_sec):
    """Alarm times: fire when firing power first crosses `threshold`, then
    suppress new alarms for `refractory_sec`. `times`/`fp` must be sorted
    ascending (as returned by `firing_power`)."""
    alarms = []
    last = -np.inf
    for t, v in zip(times, fp):
        if v >= threshold and (t - last) >= refractory_sec:
            alarms.append(float(t))
            last = t
    return alarms
