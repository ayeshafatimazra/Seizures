"""Compact time-frequency tensors for the deep-learning path.

The classical pipeline hands the model 514 hand-designed numbers per window. A
temporal CNN should instead see the signal's structure, so here I turn each 5 s,
19-channel window into a small per-channel spectrogram (channels x freq x time)
and let the network learn its own features.

Labelling and cropping are identical to `dataset.build_epochs`; only the
per-window representation differs. Spectrograms are resampled to a fixed
(F_BINS x T_BINS) grid so every window is the same shape regardless of sampling
rate, and stored as float16 to keep the per-patient cache small.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import spectrogram as _spec

from config import WINDOW_SEC, STEP_SEC, BANDS
from dataset import _label_epoch

F_BINS = 24            # frequency bins after resampling (0.5-70 Hz band)
T_BINS = 16            # time bins per 5 s window
FMAX = BANDS["gamma"][1]


def _window_spectrogram(win, sfreq):
    """(n_ch, n_samp) -> (n_ch, F_BINS, T_BINS) log-power, band-limited, resampled."""
    nper = int(min(sfreq * 0.5, win.shape[1]))          # ~0.5 s STFT segments
    f, t, Sxx = _spec(win, fs=sfreq, nperseg=nper,
                      noverlap=nper // 2, axis=-1)        # (n_ch, nf, nt)
    band = f <= FMAX
    f, Sxx = f[band], Sxx[:, band, :]
    Sxx = np.log1p(Sxx)                                  # compress dynamic range
    # resample freq and time onto fixed grids by binned averaging
    Sxx = _resample_axis(Sxx, axis=1, out=F_BINS)
    Sxx = _resample_axis(Sxx, axis=2, out=T_BINS)
    return Sxx.astype(np.float16)


def _resample_axis(a, axis, out):
    n = a.shape[axis]
    if n == out:
        return a
    idx = np.linspace(0, n, out + 1).astype(int)
    idx[-1] = n
    parts = [a.take(range(idx[i], max(idx[i] + 1, idx[i + 1])), axis=axis).mean(axis=axis, keepdims=True)
             for i in range(out)]
    return np.concatenate(parts, axis=axis)


def build_spectrograms(recordings, verbose=False):
    """Return S (n, n_ch, F_BINS, T_BINS) float16, y, groups, and meta
    (crop id + t0 + crops), mirroring dataset.build_epochs."""
    S, y, groups, ep_crop, ep_t0, crops = [], [], [], [], [], []
    for cid, rec in enumerate(recordings):
        crops.append({"subject": rec.subject, "seizures": list(rec.seizures),
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
            S.append(_window_spectrogram(rec.data[:, start:start + w], rec.sfreq))
            y.append(lab); groups.append(rec.subject)
            ep_crop.append(cid); ep_t0.append(t0); kept[lab] += 1
        if verbose:
            print(f"  [{rec.subject}] preictal={kept[1]:4d} interictal={kept[0]:4d}")
    if not S:
        raise RuntimeError("No spectrogram windows built.")
    meta = {"crop": np.asarray(ep_crop), "t0": np.asarray(ep_t0), "crops": crops}
    return np.asarray(S), np.asarray(y), np.asarray(groups), meta
