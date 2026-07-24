# Seizure Prediction from Scalp EEG

I built a classical machine-learning pipeline that attempts to predict the
**preictal state**, the interval that precedes a seizure, from raw scalp EEG. I
use the [PhysioNet Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
(14 patients, 512 Hz, 10-20 montage, 47 documented seizures over roughly 128
hours).

This is prediction rather than detection. I train a model to separate
*preictal* windows (the 30 minutes before onset, excluding a 5-minute horizon)
from *interictal* windows (more than one hour from any seizure), so that an
alarm could in principle fire before an event rather than flag it as it occurs.

> **Project status.** I processed the complete 14-patient cohort. The headline
> result is a negative one, and I report it as such: pooled, cross-patient
> prediction from classical features does not generalize on this dataset. I
> treat the numbers below as a rigorous baseline that future work must beat, not
> as a working predictor.

## Results

I evaluate at two levels. The **window level** asks whether the model can tell a
preictal 5-second window from an interictal one. The **alarm level** asks the
clinical question, how many seizures the system anticipates and how often it
raises a false alarm, because a single artefact can flip dozens of adjacent
windows and distort the window-level numbers in both directions.

**Window level.** Patient-grouped 5-fold cross-validation (no patient appears in
both train and test), 19-channel 10-20 montage, 380 features per epoch, 28,936
windows (13,607 preictal and 15,329 interictal) across 14 patients:

| Model | ROC-AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| Logistic regression | 0.47 | 0.45 | 0.50 | 0.43 |
| Random forest | 0.42 | 0.43 | 0.46 | 0.41 |

**Alarm level.** Firing-Power post-processing (SPH 5 min, SOP 30 min), logistic
regression, same patient-grouped cross-validation, with the alarm threshold
selected on the training patients only:

| Operating point | Event sensitivity | FPR/h | Warning time | Random predictor | p |
|---|---|---|---|---|---|
| Default (theta = 0.5) | 32/42 = 0.76 | 2.10 | 33.2 min | 0.65 | 0.086 |
| Tuned threshold | 21/42 = 0.50 | 1.14 | 31.8 min | 0.44 | 0.244 |

### What I read from this

On an initial three-patient subset I had measured an ROC-AUC of 0.65, which
looked encouraging. On all 14 patients the cross-patient AUC falls to roughly
0.47, at or slightly below chance. This is not a defect in the pipeline. It is
the generalization gap that the seizure-prediction literature documents
repeatedly: a model fit on a few patients rarely transfers to unseen patients,
because preictal signatures are largely patient-specific, and small samples make
cross-patient estimates look far better than they are.

The alarm layer tells the same story in clinical terms. At its most sensitive
operating point the system flags 32 of 42 seizures, but it does so at 2.1 false
alarms per hour, and at that alarm rate an unspecific random predictor would
already anticipate about 65 percent of seizures by chance. The result is
therefore not statistically distinguishable from chance (p = 0.086). Tightening
the threshold lowers the false-alarm rate to 1.14 per hour but drops sensitivity
to chance level as well.

I regard this as the honest and useful outcome of the project: a correctly
evaluated cross-patient baseline, and clear evidence that the next step is
patient-specific modelling rather than more pooled features.

> **Caveat on the sensitivity figure.** My loader crops each seizure's run-up
> into a tight preictal segment, so a preictal crop is dominated by preictal
> windows and firing power almost always crosses the threshold there. Event
> sensitivity is therefore an optimistic upper bound. The false-alarm rate,
> which I measure on separate interictal crops, is the trustworthy burden
> figure. Continuous multi-hour recordings would tighten the sensitivity
> estimate.

## Pipeline

```
load EDF -> preprocess -> QC -> windowed features -> classical ML -> alarm layer
 (MNE)      notch 50 Hz   SNR   5 s epochs           grouped CV      Firing Power
            bandpass      flat  spectral + temporal  LogReg / RF     event Sens
            bad-chan interp clip                      AUC Sens Spec   FPR/h + chance
            avg reference
```

- **Preprocessing** (`src/preprocess.py`). I apply a 50 Hz notch filter and its
  harmonics, a 0.5-70 Hz bandpass, statistical bad-channel detection, MNE
  spherical-spline interpolation from 10-20 electrode positions, and a
  common-average reference.
- **Quality control** (`src/qc.py`). I compute per-channel signal-to-noise ratio
  in decibels (in-band power against power above 100 Hz), detect flatlines and
  clipping, and apply a pass/fail gate.
- **Features** (`src/features.py`). For each window I extract spectral features
  (absolute and relative band power in delta, theta, alpha, beta, and gamma; the
  theta/beta and slowing ratios; and the 95 percent spectral edge frequency) and
  temporal features (line length, RMS, variance, zero-crossing rate, and the
  three Hjorth parameters). I harmonize every patient onto a fixed 19-channel
  10-20 montage, giving 380 features per epoch.
- **Labelling** (`src/dataset.py`). I label preictal against interictal using a
  seizure prediction horizon and postictal and guard exclusions to prevent
  leakage, and I emit per-epoch timing so the alarm layer can rebuild each
  recording's probability stream.
- **Models** (`src/train.py`). I standardize the features and fit class-balanced
  logistic regression and random forest, with patient-grouped cross-validation
  so no patient appears in both train and test. I omit the RBF-SVM at cohort
  scale because it is computationally intractable on tens of thousands of epochs.
- **Alarm layer** (`src/postprocess.py`, `src/evaluate.py`). I smooth the
  preictal probability with the Firing-Power method over the Seizure Occurrence
  Period, raise one alarm per threshold crossing, and then hold a refractory
  silence. I report event-level sensitivity, false predictions per hour, and
  warning time, and I benchmark each result against the sensitivity that an
  unspecific random predictor would reach at the same false-alarm rate, with a
  binomial p-value.

### Defining prediction precisely (SPH and SOP)

Two parameters define the task (Winterhalder et al. 2003;
[SzCORE, Dan et al. 2024](https://onlinelibrary.wiley.com/doi/10.1111/epi.18113)).
The Seizure Prediction Horizon (SPH) is the intervention gap between the alarm
and the earliest the seizure may arrive, which I set to 5 minutes. The Seizure
Occurrence Period (SOP) is the window within which the seizure is then expected,
which I set to 30 minutes. I count a prediction as correct when an alarm's SOP
contains the true onset. A longer SPH and a shorter SOP define a harder and more
clinically useful predictor.

## Data handling on a constrained disk

The full database is about 20 GB and my machine had limited free space, so I
never hold more than one patient at a time (`build_all.py`). For each patient I
sync the recordings from PhysioNet's public S3 bucket, extract and cache the
feature matrix, and then delete the raw EDF files before moving on. Peak
additional disk use is a single patient, at most 3.4 GB. The step is resumable:
a re-run skips any patient whose features are already cached.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full 14-patient cohort, downloaded and cached one patient at a time:
python build_all.py            # extract (resumable) then evaluate
python build_all.py --eval     # re-aggregate caches and evaluate only

# single run on whatever is already under data/raw/PNxx/:
python run_pipeline.py

# no data yet? a synthetic EEG fallback with an injected preictal signature:
python run_pipeline.py --synthetic
```

To pull one patient by hand:

```bash
aws s3 sync --no-sign-request \
  s3://physionet-open/siena-scalp-eeg/1.0.0/PN00/ data/raw/PN00/
```

## Roadmap

- [x] Preprocessing, QC, and classical ML baseline
- [x] Alarm layer: Firing-Power post-processing and event-level evaluation
  (sensitivity, FPR/h, warning time) benchmarked against a random predictor
- [x] Honest threshold selection on training patients, with no test-set peeking
- [x] Full 14-patient cohort processed on a disk-constrained machine
- [x] Rigorous negative result: pooled cross-patient prediction does not beat chance
- [ ] Patient-specific (personalized) models, the likely source of real signal
- [ ] Continuous multi-hour streams so event sensitivity is not crop-optimistic
- [ ] Phase 2: a PyTorch temporal CNN on raw or spectrogram windows

## Key references

- Winterhalder et al. (2003), *The seizure-prediction characteristic*. Source of
  the SPH and SOP framework and the unspecific-random-predictor benchmark.
- Teixeira et al. (2012) and
  [Nature Sci. Rep. (2023)](https://www.nature.com/articles/s41598-023-50609-z),
  on post-processing as a chronology. Source of the Firing-Power alarm method I use.
- Dan et al. (2024), [SzCORE](https://onlinelibrary.wiley.com/doi/10.1111/epi.18113).
  Standard for event-based scoring and false-alarm reporting.
- [A review of seizure-prediction evaluation pitfalls](https://pmc.ncbi.nlm.nih.gov/articles/PMC9732735/),
  on why window-level metrics and random-split cross-validation overstate performance.

## Data and license

The Siena Scalp EEG Database is copyright its authors and released under
CC-BY-4.0 through PhysioNet. I do not redistribute the EEG data here (see
`.gitignore`); download it from the source above. The pipeline code in this
repository is my own work.
