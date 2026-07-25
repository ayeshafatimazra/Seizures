"""Turn a (n_channels, n_samples) window into a fixed-length feature vector.

Spectral (frequency domain): absolute + relative band power per channel; the
theta/beta, theta/alpha, beta/alpha, delta/theta and (delta+theta)/(alpha+beta)
ratios seen in seizure and dementia EEG; spectral edge frequency (SEF95),
spectral centroid, alpha peak frequency, and the 1/f power-spectral exponent;
plus one global frontal-alpha-asymmetry term.

Temporal (time domain): line length, RMS, variance, zero-crossing rate, the
three Hjorth parameters, and permutation entropy (an ordinal-pattern complexity
measure).

The ratio / index / complexity definitions follow the NeuroSkill EEG data
reference (band powers, tar/bar/dtr, faa, pse, spectral_centroid, apf,
permutation_entropy); permutation entropy stands in for the nonlinear-complexity
family, which is well established in the seizure-EEG literature.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.signal import welch

from config import BANDS


# ---------------------------------------------------------------- spectral
def _psd(window, sfreq):
    nper = int(min(sfreq, window.shape[1]))       # ~1 s segments
    f, pxx = welch(window, fs=sfreq, nperseg=nper, axis=-1)
    return f, pxx


def _pse(f, pxx):
    """1/f power-spectral exponent: slope of log10(power) vs log10(freq) over
    2-40 Hz, per channel (vectorised)."""
    m = (f >= 2.0) & (f <= 40.0)
    x = np.log10(f[m] + 1e-12)                    # (nb,)
    y = np.log10(pxx[:, m] + 1e-24)               # (n_ch, nb)
    xm = x.mean()
    xc = x - xm
    denom = (xc ** 2).sum() + 1e-12
    slope = (y - y.mean(axis=1, keepdims=True)) @ xc / denom
    return slope                                  # (n_ch,)


def spectral_features(window, sfreq, ch_names=None):
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

    # slowing / arousal ratios (per channel), NeuroSkill: tbr, tar, bar, dtr, slow
    eps = 1e-12
    ratios = {
        "tbr":  band_abs["theta"] / (band_abs["beta"] + eps),
        "tar":  band_abs["theta"] / (band_abs["alpha"] + eps),
        "bar":  band_abs["beta"] / (band_abs["alpha"] + eps),
        "dtr":  band_abs["delta"] / (band_abs["theta"] + eps),
        "slow": (band_abs["delta"] + band_abs["theta"]) /
                (band_abs["alpha"] + band_abs["beta"] + eps),
    }
    for tag, arr in ratios.items():
        feats.append(arr)
        names += [f"{tag}_{i}" for i in range(arr.size)]

    # spectral edge frequency 95 %
    csum = np.cumsum(pxx, axis=1)
    sef = np.array([f[np.searchsorted(csum[c], 0.95 * csum[c, -1])]
                    for c in range(pxx.shape[0])])
    feats.append(sef); names += [f"sef95_{i}" for i in range(sef.size)]

    # spectral centroid: power-weighted mean frequency (rises with cortical arousal)
    centroid = (pxx * f).sum(axis=1) / total
    feats.append(centroid); names += [f"centroid_{i}" for i in range(centroid.size)]

    # alpha peak frequency: argmax of power within the alpha band
    am = (f >= BANDS["alpha"][0]) & (f < BANDS["alpha"][1])
    fa = f[am]
    apf = fa[pxx[:, am].argmax(axis=1)]
    feats.append(apf); names += [f"apf_{i}" for i in range(apf.size)]

    # 1/f power-spectral exponent
    pse = _pse(f, pxx)
    feats.append(pse); names += [f"pse_{i}" for i in range(pse.size)]

    # frontal alpha asymmetry: ln(alpha F4) - ln(alpha F3), one global scalar
    if ch_names is not None and "F3" in ch_names and "F4" in ch_names:
        l = band_abs["alpha"][ch_names.index("F3")]
        r = band_abs["alpha"][ch_names.index("F4")]
        faa = np.log(r + eps) - np.log(l + eps)
        feats.append(np.array([faa])); names += ["faa"]

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


def _perm_entropy(window, order=3, delay=1):
    """Normalised permutation entropy per channel (Bandt-Pompe), vectorised over
    channels. Near 1 = complex/irregular; near 0 = highly ordered/periodic."""
    n = window.shape[1]
    L = n - delay * (order - 1)
    # ordinal embedding: (n_ch, L, order)
    emb = np.stack([window[:, i * delay: i * delay + L] for i in range(order)], axis=-1)
    ranks = emb.argsort(axis=-1)                       # permutation per point
    # encode each permutation as an integer code (base = order)
    codes = np.zeros(ranks.shape[:2], dtype=np.int64)
    for i in range(order):
        codes = codes * order + ranks[..., i]
    pe = np.empty(window.shape[0])
    norm = np.log(math.factorial(order))
    for c in range(window.shape[0]):
        counts = np.bincount(codes[c])
        p = counts[counts > 0] / L
        pe[c] = -(p * np.log(p)).sum() / norm
    return pe


def temporal_features(window, sfreq):
    line_length = np.abs(np.diff(window, axis=1)).sum(axis=1)
    rms = np.sqrt((window ** 2).mean(axis=1))
    var = window.var(axis=1)
    zcr = (np.diff(np.signbit(window), axis=1) != 0).mean(axis=1)
    act, mob, cmp = _hjorth(window)
    pe = _perm_entropy(window)
    feats = [line_length, rms, var, zcr, act, mob, cmp, pe]
    names = []
    for tag, arr in zip(["ll", "rms", "var", "zcr", "hj_act", "hj_mob", "hj_cmp", "pe"], feats):
        names += [f"{tag}_{i}" for i in range(arr.size)]
    return np.concatenate(feats), names


def extract(window, sfreq, ch_names=None):
    """Full feature vector for one epoch. Returns (vector, feature_names).
    `ch_names` (optional) enables the frontal-alpha-asymmetry term."""
    sf, sn = spectral_features(window, sfreq, ch_names)
    tf, tn = temporal_features(window, sfreq)
    return np.concatenate([sf, tf]), sn + tn
