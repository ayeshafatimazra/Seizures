# Seizure Prediction from Scalp EEG

A classical-ML pipeline that predicts the **pre-ictal state** — the run-up
*before* a seizure — from raw scalp EEG, using the [PhysioNet Siena Scalp EEG
Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/) (14 patients,
512 Hz, 10–20 montage, 47 seizures over ~128 h).

This is **prediction, not detection**: the model learns to separate *preictal*
(30 min before onset, minus a 5 min horizon) from *interictal* (>1 h from any
seizure), so an alarm could fire before the event — not merely flag it as it
happens.

> **Project status:** runs end-to-end on real Siena data. Metrics below are a
> **3-patient** baseline (PN00, PN01, PN03; leave-one-patient-out CV). Scaling
> to all 14 patients: `[TODO — needs external drive]`.

---

## Results

Leave-one-patient-out cross-validation (GroupKFold, 3 folds, 580 features/epoch,
2149 preictal / 1440 interictal 5 s windows):

| Model | ROC-AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.65** | 0.86 | 0.23 | 0.74 |
| Random Forest | 0.64 | 0.88 | 0.12 | 0.74 |
| RBF SVM | 0.62 | 0.83 | 0.12 | 0.70 |

**Honest read.** AUC ≈ 0.65 is *above chance* for cross-patient seizure
prediction from classical scalp-EEG features — a legitimately hard task — but the
0.5-decision-threshold operating point over-calls preictal (high sensitivity,
low specificity). That gap is the target for the next steps: per-patient
calibration, threshold tuning on a validation fold, more patients, and the
Phase-2 temporal CNN. This number is the baseline those must beat, not a
finished result.

---

## Pipeline

```
load EDF  ─►  preprocess  ─►  QC  ─►  windowed features  ─►  classical ML
 (MNE)        notch 50 Hz      SNR    5 s epochs             GroupKFold CV
              bandpass 0.5–70  flat   spectral + temporal    LogReg / RF / SVM
              bad-chan interp  clip                          AUC · Sens · Spec
              avg reference
```

- **Preprocess** (`src/preprocess.py`) — 50 Hz notch + harmonics, 0.5–70 Hz
  bandpass, statistical bad-channel detection, **MNE spherical-spline
  interpolation** from 10–20 positions, common-average reference.
- **QC** (`src/qc.py`) — per-channel **SNR in dB** (in-band vs >100 Hz power),
  flatline and clipping detection, pass/fail gate.
- **Features** (`src/features.py`) — *spectral:* absolute + relative band power
  (δ θ α β γ), theta/beta & slowing ratios, spectral edge frequency (SEF95);
  *temporal:* line length, RMS, variance, zero-crossing rate, Hjorth
  (activity/mobility/complexity).
- **Labelling** (`src/dataset.py`) — preictal vs interictal with a seizure
  prediction horizon and postictal/guard exclusions to prevent leakage.
- **Model** (`src/train.py`) — standardised features, class-balanced LogReg /
  RandomForest / RBF-SVM, **GroupKFold** by patient.

---

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# real data: drop Siena patient folders under data/raw/PNxx/ then:
python run_pipeline.py

# no data yet? runs on a synthetic EEG fallback with an injected preictal signature:
python run_pipeline.py --synthetic
```

Get the data (no full 20 GB download needed — pull per patient):

```bash
aws s3 sync --no-sign-request \
  s3://physionet-open/siena-scalp-eeg/1.0.0/PN00/ data/raw/PN00/
```

---

## Roadmap

- [x] Preprocessing + QC + classical ML baseline (this repo)
- [x] Real-data metrics on 3-patient subset (AUC 0.65, LOPO-CV)
- [ ] Threshold/calibration tuning to fix specificity; scale to all 14 patients
- [ ] Per-patient (personalised) models vs pooled
- [ ] **Phase 2:** PyTorch temporal CNN on raw/spectrogram windows
- [ ] Sibling projects: Alzheimer's EEG classification · motor-imagery BCI

## Data & license

Siena Scalp EEG Database © its authors, released **CC-BY-4.0** via PhysioNet.
EEG data is **not** redistributed here (see `.gitignore`); download it from the
source above. Pipeline code in this repo is the author's own work.
