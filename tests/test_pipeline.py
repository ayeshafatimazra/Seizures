"""Unit tests for the parts of the pipeline where a silent bug would quietly
corrupt the science: label leakage, alarm post-processing, the chance-predictor
formula, montage harmonisation, and feature sanity.

    pip install pytest && pytest -q
"""
import numpy as np
import pytest

import config
from dataset import _label_epoch
from postprocess import firing_power, raise_alarms
from evaluate import _chance
from data_loader import harmonize_to, Recording
from features import _perm_entropy, extract
from stats import auprc, bootstrap_auc_ci, permutation_pvalue


# ---------------------------------------------------------------- labelling
def test_label_preictal_window():
    """A window inside [onset - PREICTAL - SPH, onset - SPH) is preictal (1)."""
    onset = 10_000.0
    t0 = onset - config.SPH_SEC - 100          # just inside the preictal band
    assert _label_epoch(t0, t0 + config.WINDOW_SEC, [(onset, onset + 60)]) == 1


def test_label_sph_gap_dropped():
    """A window inside the SPH gap (just before onset) is dropped (None)."""
    onset = 10_000.0
    t0 = onset - config.SPH_SEC / 2            # inside the horizon
    assert _label_epoch(t0, t0 + config.WINDOW_SEC, [(onset, onset + 60)]) is None


def test_label_far_interictal():
    """A window far from any seizure is interictal (0)."""
    onset = 10_000.0
    t0 = onset - config.INTERICTAL_GUARD_SEC - 5_000
    assert _label_epoch(t0, t0 + config.WINDOW_SEC, [(onset, onset + 60)]) == 0


def test_label_seizure_free_is_interictal():
    assert _label_epoch(0.0, config.WINDOW_SEC, []) == 0


# ---------------------------------------------------------------- firing power / alarms
def test_firing_power_is_trailing_mean():
    t = np.arange(10, dtype=float)             # 1 s apart
    p = np.ones(10)
    _, fp = firing_power(t, p, window_sec=3)
    assert np.allclose(fp, 1.0)                # mean of all-ones is one


def test_firing_power_sorts_by_time():
    t = np.array([2.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 0.0])
    ts, fp = firing_power(t, p, window_sec=10)
    assert list(ts) == [0.0, 1.0, 2.0]         # returned time-sorted
    assert fp[0] == 0.0 and fp[-1] == pytest.approx(1 / 3)


def test_alarm_refractory_suppresses_repeats():
    t = np.arange(0, 100, 1.0)
    fp = np.ones_like(t)                        # always above threshold
    alarms = raise_alarms(t, fp, threshold=0.5, refractory_sec=30)
    assert alarms[0] == 0.0
    assert all(b - a >= 30 for a, b in zip(alarms, alarms[1:]))


def test_no_alarm_below_threshold():
    t = np.arange(10, dtype=float)
    fp = np.full(10, 0.2)
    assert raise_alarms(t, fp, threshold=0.5, refractory_sec=5) == []


# ---------------------------------------------------------------- chance predictor
def test_chance_sensitivity_rises_with_fpr():
    lo, _ = _chance(sens=0.5, fpr_per_h=0.5, n_seizures=10, n_predicted=5)
    hi, _ = _chance(sens=0.5, fpr_per_h=5.0, n_seizures=10, n_predicted=5)
    assert 0 < lo < hi < 1                      # more false alarms -> easier to hit by luck


def test_chance_pvalue_small_when_beating_chance():
    # low FPR (small chance level) but all seizures predicted -> significant
    _, p = _chance(sens=1.0, fpr_per_h=0.1, n_seizures=20, n_predicted=20)
    assert p < 0.05


# ---------------------------------------------------------------- montage harmonisation
def test_harmonize_reorders_and_subsets():
    rec = Recording(data=np.arange(12).reshape(3, 4).astype(float),
                    ch_names=["C4", "F3", "Fz"], sfreq=256.0, subject="T")
    harmonize_to([rec], ["F3", "C4"])
    assert rec.ch_names == ["F3", "C4"]
    assert np.array_equal(rec.data[0], [4, 5, 6, 7])   # F3 row moved to front


def test_harmonize_raises_on_missing_channel():
    rec = Recording(data=np.zeros((1, 4)), ch_names=["F3"], sfreq=256.0, subject="T")
    with pytest.raises(ValueError):
        harmonize_to([rec], ["F3", "Cz"])


# ---------------------------------------------------------------- features
def test_perm_entropy_ordered_vs_random():
    ramp = np.arange(500.0)[None, :]            # monotonic -> low complexity
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((1, 500))       # random -> high complexity
    assert _perm_entropy(ramp)[0] < 0.2
    assert _perm_entropy(noise)[0] > 0.8


# ---------------------------------------------------------------- statistics
def test_auprc_perfect_and_prevalence():
    y = np.array([0, 0, 0, 1])
    assert auprc(y, np.array([0.1, 0.2, 0.3, 0.9])) == pytest.approx(1.0)   # perfect ranking
    # a random score should sit near the positive prevalence (0.25 here)
    rng = np.random.default_rng(0)
    yb = (rng.random(4000) < 0.25).astype(int)
    assert auprc(yb, rng.random(4000)) == pytest.approx(0.25, abs=0.05)


def test_permutation_pvalue_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    good = np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 100)]   # separable
    assert permutation_pvalue(y, good, n_perm=500) < 0.05
    noise = rng.random(200)                                      # no signal
    assert permutation_pvalue(y, noise, n_perm=500) > 0.05


def test_bootstrap_ci_brackets_auc():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    prob = np.r_[rng.normal(0, 1, 200), rng.normal(1.5, 1, 200)]
    lo, hi = bootstrap_auc_ci(y, prob, n_boot=500)
    from sklearn.metrics import roc_auc_score
    assert lo < roc_auc_score(y, prob) < hi and 0.5 < lo < hi <= 1.0


def test_extract_is_finite_and_named():
    rng = np.random.default_rng(1)
    win = rng.standard_normal((19, 2560))
    vec, names = extract(win, 512.0, ["Fp1", "Fp2", "F3", "F4", "C3", "C4",
                                      "P3", "P4", "O1", "O2", "F7", "F8", "T3",
                                      "T4", "T5", "T6", "Fz", "Cz", "Pz"])
    assert len(vec) == len(names)
    assert np.isfinite(vec).all()
    assert "faa" in names                       # FAA present when F3/F4 given
