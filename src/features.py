"""Turn a (n_channels, n_samples) window into a fixed-length feature vector.

Spectral (frequency domain): absolute + relative band power per channel,
spectral edge frequency (SEF95), and the theta/beta & (delta+theta)/(alpha+beta)
slowing ratios that show up in seizure and dementia EEG literature.

Temporal (time domain): line length, RMS, variance, zero-crossing rate, and the
three Hjorth parameters (activity, mobility, complexity).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch

from config import BANDS


# ---------------------------------------------------------------- spectral
def _psd(window, sfreq):
    nper = int(min(sfreq, window.shape[1]))       # ~1 s segments
    f, pxx = welch(window, fs=sfreq, nperseg=nper, axis=-1)
    return f, pxx


def spectral_features(window, sfreq):
    f, pxx = _psd(window, sfreq)
    total = pxx.sum(axis=1) + 1e-12               # per channel
    feats, names = [], []
    band_abs = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp = pxx[:, m].sum(axis=1)
        band_abs[name] = bp
        feats.append(bp)                          # absolute
        names += [f"{name}_abs_{i}" for i in range(bp.size)]
        feats.append(bp / total)                  # relative
        names += [f"{name}_rel_{i}" for i in range(bp.size)]

    # slowing / arousal ratios (per channel)
    tbr = band_abs["theta"] / (band_abs["beta"] + 1e-12)
    slow = ((band_abs["delta"] + band_abs["theta"]) /
            (band_abs["alpha"] + band_abs["beta"] + 1e-12))
    feats += [tbr, slow]
    names += [f"tbr_{i}" for i in range(tbr.size)]
    names += [f"slow_{i}" for i in range(slow.size)]

    # spectral edge frequency 95 %
    csum = np.cumsum(pxx, axis=1)
    sef = np.array([f[np.searchsorted(csum[c], 0.95 * csum[c, -1])]
                    for c in range(pxx.shape[0])])
    feats.append(sef)
    names += [f"sef95_{i}" for i in range(sef.size)]
    return np.concatenate(feats), names


# ---------------------------------------------------------------- temporal
def _hjorth(window):
    d1 = np.diff(window, axis=1)
    d2 = np.diff(d1, axis=1)
    var0 = window.var(axis=1) + 1e-12
    var1 = d1.var(axis=1) + 1e-12
    var2 = d2.var(axis=1) + 1e-12
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
    return activity, mobility, complexity


def temporal_features(window, sfreq):
    line_length = np.abs(np.diff(window, axis=1)).sum(axis=1)
    rms = np.sqrt((window ** 2).mean(axis=1))
    var = window.var(axis=1)
    zcr = (np.diff(np.signbit(window), axis=1) != 0).mean(axis=1)
    act, mob, cmp = _hjorth(window)
    feats = [line_length, rms, var, zcr, act, mob, cmp]
    names = []
    for tag, arr in zip(["ll", "rms", "var", "zcr", "hj_act", "hj_mob", "hj_cmp"], feats):
        names += [f"{tag}_{i}" for i in range(arr.size)]
    return np.concatenate(feats), names


def extract(window, sfreq):
    """Full feature vector for one epoch. Returns (vector, feature_names)."""
    sf, sn = spectral_features(window, sfreq)
    tf, tn = temporal_features(window, sfreq)
    return np.concatenate([sf, tf]), sn + tn
