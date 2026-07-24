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

Two levels of evaluation. **Window-level** asks *can the model tell a preictal
5 s window from an interictal one?* **Alarm-level** asks the clinical question —
*how many seizures does the alarm anticipate, and how often does it cry wolf?* —
because a single artefact can flip dozens of adjacent windows and distort the
first number in both directions.

**Window-level** — leave-one-patient-out (GroupKFold, 3 folds, 580 features/epoch,
2149 preictal / 1440 interictal 5 s windows):

| Model | ROC-AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.65** | 0.86 | 0.23 | 0.74 |
| Random Forest | 0.64 | 0.88 | 0.12 | 0.74 |
| RBF SVM | 0.62 | 0.83 | 0.12 | 0.70 |

**Alarm-level** — Firing-Power post-processing (SPH 5 min, SOP 30 min), logistic
regression, same LOPO-CV, threshold selected on the training patients:

| Operating point | Event sensitivity | FPR/h | Warning time | vs. chance |
|---|---|---|---|---|
| Firing Power + tuned θ | **9/9 = 1.00** | 2.90 | 28.9 min | p = 0.09 |

**Honest read.** Window-level AUC ≈ 0.65 is *above chance* for cross-patient
prediction from classical features — a legitimately hard task — but the default
operating point over-calls preictal (high sensitivity, low specificity). The
alarm layer makes the real problem legible: the system *does* flag every one of
the 9 seizures, but at **2.9 false alarms per hour**, and at that alarm rate an
*unspecific random predictor* would already anticipate ~77% of seizures by luck
— so the result is **not yet statistically distinguishable from chance (p =
0.09)** on only 9 seizures. Beating chance, not maximising sensitivity, is the
bar. The levers are more patients (the p-value shrinks with seizure count),
per-patient calibration, and the Phase-2 temporal CNN. These numbers are the
baseline those must beat, not a finished result.

> Caveat on the sensitivity number: the loader crops each seizure's run-up into
> a tight preictal segment, so a preictal crop is dominated by preictal windows
> — firing power almost always crosses threshold there, making event sensitivity
> an optimistic upper bound. The FPR/h, measured on separate interictal crops,
> is the trustworthy burden figure. Continuous multi-hour recordings (Phase 2
> data handling) will tighten sensitivity.

---

## Pipeline

```
load EDF ─► preprocess ─► QC ─► windowed features ─► classical ML ─► alarm layer
 (MNE)      notch 50 Hz   SNR   5 s epochs            GroupKFold CV    Firing Power
            bandpass      flat  spectral + temporal   LogReg/RF/SVM    → event Sens
            bad-chan interp clip                       AUC·Sens·Spec    + FPR/h · chance
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
  prediction horizon and postictal/guard exclusions to prevent leakage; also
  emits per-epoch timing so the alarm layer can rebuild each recording's stream.
- **Model** (`src/train.py`) — standardised features, class-balanced LogReg /
  RandomForest / RBF-SVM, **GroupKFold** by patient.
- **Alarm layer** (`src/postprocess.py`, `src/evaluate.py`) — **Firing-Power**
  smoothing of the preictal probability over the SOP window, one alarm per
  threshold crossing, then a refractory silence. Event-level **sensitivity**,
  **false predictions per hour (FPR/h)**, and warning time — benchmarked against
  the sensitivity an **unspecific random predictor** reaches at the same FPR/h
  (binomial p-value). Threshold picked on training patients only.

### Prediction, precisely (SPH + SOP)

Two parameters define the task ([Winterhalder 2003](https://doi.org/10.1016/j.yebeh.2003.05.007);
[SzCORE / Dan 2024](https://onlinelibrary.wiley.com/doi/10.1111/epi.18113)):
the **Seizure Prediction Horizon (SPH)** is the intervention gap between the
alarm and the earliest the seizure may come (here 5 min); the **Seizure
Occurrence Period (SOP)** is the window it's then expected within (here 30 min).
A correct prediction is an alarm whose SOP contains the true onset. Longer SPH
and shorter SOP make a harder, more clinically useful predictor.

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
- [x] **Alarm layer:** Firing-Power post-processing + event-level evaluation
  (sensitivity, FPR/h, warning time) benchmarked against a random predictor
- [x] Honest threshold selection on training patients (no test-set peeking)
- [ ] Scale to all 14 patients — the fix for the not-yet-significant p-value
- [ ] Continuous multi-hour streams so event sensitivity isn't crop-optimistic
- [ ] Per-patient (personalised) models vs pooled
- [ ] **Phase 2:** PyTorch temporal CNN on raw/spectrogram windows
- [ ] Sibling projects: Alzheimer's EEG classification · motor-imagery BCI

## Key references

- Winterhalder et al. (2003), *The seizure-prediction characteristic* — SPH/SOP
  framework and the unspecific-random-predictor benchmark.
- Teixeira et al. (2012) / Nature Sci. Rep. (2023),
  [*post-processing as a chronology*](https://www.nature.com/articles/s41598-023-50609-z)
  — the Firing-Power alarm method used here.
- Dan et al. (2024), [*SzCORE*](https://onlinelibrary.wiley.com/doi/10.1111/epi.18113)
  — standard for event-based scoring and FPR-per-day reporting.
- [Review of seizure-prediction eval pitfalls](https://pmc.ncbi.nlm.nih.gov/articles/PMC9732735/)
  — why window-level metrics and random-split CV overstate performance.

## Data & license

Siena Scalp EEG Database © its authors, released **CC-BY-4.0** via PhysioNet.
EEG data is **not** redistributed here (see `.gitignore`); download it from the
source above. Pipeline code in this repo is the author's own work.
