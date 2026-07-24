"""Signal-quality checks. The headline metric is a band-based SNR proxy:
in-band physiological power (1-40 Hz) vs high-frequency residual (>100 Hz),
reported in dB per channel. Also flags flatlines and clipping.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch

from config import QC_MIN_SNR_DB


def channel_snr_db(data, sfreq, sig_band=(1.0, 40.0), noise_band=(100.0, None)):
    """Per-channel SNR (dB) = 10*log10(in-band power / high-freq power)."""
    nper = int(min(4 * sfreq, data.shape[1]))
    f, pxx = welch(data, fs=sfreq, nperseg=nper, axis=-1)
    hi = noise_band[1] if noise_band[1] else sfreq / 2
    sig = (f >= sig_band[0]) & (f < sig_band[1])
    noise = (f >= noise_band[0]) & (f < hi)
    ps = pxx[:, sig].sum(axis=1)
    pn = pxx[:, noise].sum(axis=1) + 1e-12
    return 10 * np.log10(ps / pn)


def qc_report(rec) -> dict:
    """Compute a QC dict for one Recording."""
    snr = channel_snr_db(rec.data, rec.sfreq)
    std = rec.data.std(axis=1)
    peak = np.abs(rec.data).max(axis=1)
    flat = int((std < 0.5).sum())
    # crude clipping detector: many samples piled at the extreme value
    clip = 0
    for ch in rec.data:
        hi = np.percentile(np.abs(ch), 99.9)
        clip += int((np.abs(ch) >= hi * 0.999).mean() > 0.01)
    rep = {
        "subject": rec.subject,
        "n_channels": len(rec.ch_names),
        "duration_min": round(rec.duration_sec / 60, 1),
        "snr_db_median": round(float(np.median(snr)), 2),
        "snr_db_min": round(float(np.min(snr)), 2),
        "flat_channels": flat,
        "clipping_channels": clip,
        "peak_uv_max": round(float(peak.max()), 1),
        "pass": bool(np.median(snr) >= QC_MIN_SNR_DB and flat == 0),
    }
    return rep
