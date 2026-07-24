"""Central configuration for the seizure-prediction pipeline.

All magic numbers live here so the science is auditable in one place.
Dataset target: PhysioNet Siena Scalp EEG Database v1.0.0 (512 Hz, 10-20, EDF).
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"          # drop Siena PNxx/*.edf here
OUTPUTS = ROOT / "outputs"                 # figures, metrics, trained models
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- signal
SFREQ = 512.0                              # Siena native sampling rate (Hz)
LINE_FREQ = 50.0                           # Italy mains -> notch 50 Hz + harmonics
BANDPASS = (0.5, 70.0)                     # Hz, after notch
NOTCH_FREQS = (50.0, 100.0, 150.0)         # mains + harmonics within band

# canonical 10-20 channels we standardise every recording onto
STD_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]

# frequency bands for spectral features (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 70.0),
}

# ---------------------------------------------------------------- windowing / labels
WINDOW_SEC = 5.0                           # epoch length fed to the model
STEP_SEC = 5.0                             # hop between epochs (== window -> no overlap)

# seizure-PREDICTION labelling (not detection):
#   preictal  = the run-up window BEFORE seizure onset  -> positive class (1)
#   interictal= baseline, far from any seizure           -> negative class (0)
#   ictal + guard bands are excluded from training.
PREICTAL_SEC = 30 * 60                     # 30 min pre-onset counted as preictal
SPH_SEC = 5 * 60                           # Seizure Prediction Horizon: drop 5 min just before onset
POSTICTAL_SEC = 15 * 60                    # exclude 15 min after seizure end
INTERICTAL_GUARD_SEC = 60 * 60            # interictal must be >1 h from any seizure
INTERICTAL_MAX_SEC = 40 * 60              # cap interictal pulled per patient (balance + memory)

# ---------------------------------------------------------------- alarm layer
# Window-level probabilities are noisy; the clinical object is an ALARM. We
# smooth the preictal probability with the Firing-Power method (Teixeira 2012),
# raise one alarm when it crosses a threshold, then stay silent for a
# refractory period. Metrics are then event-level: sensitivity + false
# predictions per hour (FPR/h), the pair the prediction literature reports.
SOP_SEC = PREICTAL_SEC                     # Seizure Occurrence Period == labelled preictal window
FP_WINDOW_SEC = SOP_SEC                     # firing-power smoothing window (== SOP)
FP_THRESHOLD = 0.5                          # default firing-power alarm threshold in [0,1]
FPR_BUDGET_PER_H = 0.5                      # threshold-tuning target: keep FPR/h at or below this
REFRACTORY_SEC = SPH_SEC + SOP_SEC          # post-alarm silence (the predicted window must pass)

# canonical Siena EEG montage (29 ch) — real EDFs are cropped/reordered onto the
# subset of these that each recording actually contains (case-insensitive match).
SIENA_CHANNELS = [
    "Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fc1", "Fc5",
    "Cp1", "Cp5", "F9", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4",
    "O2", "F8", "T4", "T6", "Fc2", "Fc6", "Cp2", "Cp6", "F10",
]

# ---------------------------------------------------------------- QC thresholds
QC_MIN_SNR_DB = 3.0                         # flag recording if median channel SNR < this
FLAT_STD_UV = 0.5                           # channel std below this (in scaled units) => flat
NOISE_Z = 4.0                               # channel whose amplitude z-score exceeds => bad

RANDOM_STATE = 42
