"""Cleaning stage: notch -> bandpass -> bad-channel detect + interpolate ->
average re-reference. Uses MNE when available (proper spherical-spline
interpolation from 10-20 positions); falls back to scipy + neighbour-mean so
the synthetic pipeline still runs with no montage.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt

from config import (LINE_FREQ, NOTCH_FREQS, BANDPASS, FLAT_STD_UV, NOISE_Z,
                    STD_CHANNELS)


def _notch(x, sfreq, freqs=NOTCH_FREQS, q=30.0):
    y = x.copy()
    for f0 in freqs:
        if f0 >= sfreq / 2:
            continue
        b, a = iirnotch(f0, q, sfreq)
        y = filtfilt(b, a, y, axis=-1)
    return y


def _bandpass(x, sfreq, lo, hi, order=4):
    ny = sfreq / 2
    hi = min(hi, ny * 0.99)
    b, a = butter(order, [lo / ny, hi / ny], btype="band")
    return filtfilt(b, a, x, axis=-1)


def detect_bad_channels(data, ch_names):
    """Flag flat and high-amplitude channels. Returns list of bad names."""
    std = data.std(axis=1)
    bad = set(np.array(ch_names)[std < FLAT_STD_UV].tolist())      # flatlines
    # robust z-score of per-channel amplitude
    med, mad = np.median(std), np.median(np.abs(std - np.median(std))) + 1e-9
    z = 0.6745 * (std - med) / mad
    bad |= set(np.array(ch_names)[np.abs(z) > NOISE_Z].tolist())   # noisy/dead
    return sorted(bad)


def _interpolate_mne(data, ch_names, sfreq, bads):
    import mne
    info = mne.create_info(ch_names, sfreq, ch_types="eeg", verbose="ERROR")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")     # uV -> V
    try:
        raw.set_montage("standard_1020", match_case=False, on_missing="ignore",
                        verbose="ERROR")
        raw.info["bads"] = [b for b in bads if b in ch_names]
        if raw.info["bads"]:
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        return raw.get_data() * 1e6
    except Exception:
        return None


def _interpolate_meanfill(data, ch_names, bads):
    """Fallback: replace each bad channel with the mean of the good ones."""
    good_idx = [i for i, c in enumerate(ch_names) if c not in bads]
    if not good_idx:
        return data
    good_mean = data[good_idx].mean(axis=0)
    out = data.copy()
    for i, c in enumerate(ch_names):
        if c in bads:
            out[i] = good_mean
    return out


def preprocess(rec, verbose=False):
    """Run the full clean on a Recording (in place on a copy). Returns
    (Recording, meta) where meta records what happened for the QC report.
    """
    x = rec.data.astype(np.float64)
    x = _notch(x, rec.sfreq)
    x = _bandpass(x, rec.sfreq, *BANDPASS)

    bads = detect_bad_channels(x, rec.ch_names)
    interp = _interpolate_mne(x, rec.ch_names, rec.sfreq, bads) if bads else None
    method = "mne_spline"
    if interp is None:
        interp = _interpolate_meanfill(x, rec.ch_names, bads)
        method = "mean_fill" if bads else "none"
    x = interp

    x = x - x.mean(axis=0, keepdims=True)   # common average reference

    from data_loader import Recording
    out = Recording(data=x.astype(np.float32), ch_names=rec.ch_names,
                    sfreq=rec.sfreq, seizures=rec.seizures,
                    subject=rec.subject, synthetic=rec.synthetic)
    meta = {"n_bad": len(bads), "bad_channels": bads, "interp_method": method}
    if verbose:
        print(f"  [{rec.subject}] bads={bads or 'none'} interp={method}")
    return out, meta
