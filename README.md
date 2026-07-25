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

> **Project status.** I processed the complete 14-patient cohort and evaluated
> two regimes. Pooled, cross-patient prediction does not generalize (AUC about
> 0.47, at or below chance), which is the honest negative baseline. Patient
> specific (personalized) models, trained and tested within one person, recover
> real signal (pooled AUC about 0.71). That contrast is the main result: preictal
> structure is largely patient-specific, so the personalized setting is the one
> worth pursuing.

## Results

I evaluate at two levels. The **window level** asks whether the model can tell a
preictal 5-second window from an interictal one. The **alarm level** asks the
clinical question, how many seizures the system anticipates and how often it
raises a false alarm, because a single artefact can flip dozens of adjacent
windows and distort the window-level numbers in both directions.

**Cross-patient, window level.** Patient-grouped 5-fold cross-validation (no
patient appears in both train and test), 19-channel 10-20 montage, 514 features
per epoch, 28,936 windows (13,607 preictal and 15,329 interictal) across 14
patients:

| Model | ROC-AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| Logistic regression | 0.47 | 0.46 | 0.49 | 0.43 |
| Random forest | 0.42 | 0.43 | 0.46 | 0.42 |

**Cross-patient, alarm level.** Firing-Power post-processing (SPH 5 min, SOP 30
min), logistic regression, same patient-grouped cross-validation, with the alarm
threshold selected on the training patients only:

| Operating point | Event sensitivity | FPR/h | Warning time | Random predictor | p |
|---|---|---|---|---|---|
| Default (theta = 0.5) | 32/42 = 0.76 | 2.01 | 32.5 min | 0.63 | 0.056 |
| Tuned threshold | 21/42 = 0.50 | 1.42 | 30.6 min | 0.51 | 0.600 |

**Personalized (patient-specific), window level.** Leave-one-seizure-out
cross-validation within each patient, logistic regression, 11 of 14 patients
evaluable (the other three lack a second seizure or any captured interictal
data):

| | ROC-AUC |
|---|---|
| Mean across patients | 0.69 |
| Pooled | 0.71 |
| Best patient (PN03) | 0.98 |
| Worst patient (PN06) | 0.40 |

![Cross-patient versus personalized AUC](figures/cross_vs_personalized_auc.png)

**Personalized, alarm level.** Per-patient Firing-Power alarms (leave-one-seizure-out
for sensitivity, leave-one-interictal-crop-out for the false-alarm rate). Alarm
prediction is viable for a subset of patients but not universally:

| Patient | Event sensitivity | FPR/h |
|---|---|---|
| PN03 | 1.00 | 0.75 |
| PN05 | 1.00 | 1.50 |
| PN10 | 0.67 | 0.91 |
| ... | ... | ... |
| Pooled (11 patients) | 0.51 | 1.40 |

The pooled alarm result sits at chance because the non-responders drag it down,
but individual patients such as PN03 (every seizure anticipated at 0.75 false
alarms per hour) are genuinely useful. This responder / non-responder split is a
well-documented feature of seizure prediction, not an artefact.

### What I read from this

On an initial three-patient subset I had measured an ROC-AUC of 0.65, which
looked encouraging. On all 14 patients the cross-patient AUC falls to roughly
0.47, at or slightly below chance, and the alarm layer does not beat an
unspecific random predictor (p = 0.056 at the default operating point, p = 0.60
tuned). This is not a defect in the pipeline. It is the generalization gap that
the seizure-prediction literature documents repeatedly: a model fit on some
patients transfers poorly to unseen patients, because preictal signatures are
largely patient-specific, and small samples make cross-patient estimates look
far better than they are.

The personalized evaluation confirms this directly. When I train and test within
a single patient, the pooled ROC-AUC rises to 0.71, and individual patients
range from highly predictable (PN03 at 0.98, PN05 at 0.98, PN10 at 0.81) to no
better than chance (PN06 at 0.40, PN09 at 0.47). The signal is real but it lives
inside each patient, not across the population.

![Per-patient personalized AUC](figures/per_patient_auc.png)

The same heterogeneity appears at the alarm level. Each point below is one
patient's operating point; the upper-left corner (high sensitivity, low
false-alarm rate) is where a predictor is clinically useful, and a handful of
patients reach it.

![Per-patient sensitivity versus false-alarm rate](figures/sensitivity_vs_fpr.png)

I regard this contrast as the honest and useful outcome of the project: a
correctly evaluated cross-patient baseline that does not beat chance, and a
patient-specific result that does, which is exactly the regime the clinical
literature treats as viable.

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
            bandpass      flat  spectral+temporal    LogReg / RF     event Sens
            bad-chan interp clip +nonlinear          cross-patient   FPR/h + chance
            avg reference                            + personalized
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
  theta/beta, theta/alpha, beta/alpha, delta/theta and slowing ratios; the 95
  percent spectral edge frequency, spectral centroid, alpha peak frequency, and
  the 1/f power-spectral exponent), temporal features (line length, RMS,
  variance, zero-crossing rate, and the three Hjorth parameters), one nonlinear
  complexity feature (permutation entropy), and a global frontal-alpha-asymmetry
  term. The ratio, index, and complexity definitions follow the NeuroSkill EEG
  data reference. I harmonize every patient onto a fixed 19-channel 10-20
  montage, giving 514 features per epoch.
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
- **Personalized models** (`src/personalized.py`). I run leave-one-seizure-out
  cross-validation within each patient, grouping preictal windows by seizure so
  the held-out seizure never leaks into training, and I report per-patient and
  pooled AUC. This is the regime that actually carries signal.

### Defining prediction precisely (SPH and SOP)

Two parameters define the task (Winterhalder et al. 2003;
[SzCORE, Dan et al. 2024](https://onlinelibrary.wiley.com/doi/10.1111/epi.18113)).
The Seizure Prediction Horizon (SPH) is the intervention gap between the alarm
and the earliest the seizure may arrive, which I set to 5 minutes. The Seizure
Occurrence Period (SOP) is the window within which the seizure is then expected,
which I set to 30 minutes. I count a prediction as correct when an alarm's SOP
contains the true onset. A longer SPH and a shorter SOP define a harder and more
clinically useful predictor.

The figure below is one worked example on a held-out seizure from PN03. The
firing power (the smoothed preictal probability) rises through the SOP window and
crosses the alarm threshold, raising a single alarm well before onset. It also
makes the crop-optimism caveat visible: because this preictal crop is dominated
by preictal windows, the firing power stays high throughout, which is why I trust
the false-alarm rate (measured on interictal crops) more than the raw sensitivity.

![Firing-power trace with alarm](figures/firing_power_trace.png)

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
- [x] Patient-specific (personalized) models: leave-one-seizure-out, pooled AUC 0.71
- [x] Expanded feature set (spectral ratios, 1/f exponent, permutation entropy, FAA)
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
- NeuroSkill EEG data reference. Source of the spectral ratio, index, and
  nonlinear-complexity feature definitions (band powers, tar/bar/dtr, spectral
  centroid, alpha peak frequency, 1/f exponent, permutation entropy, frontal
  alpha asymmetry).

## Data and license

The Siena Scalp EEG Database is copyright its authors and released under
CC-BY-4.0 through PhysioNet. I do not redistribute the EEG data here (see
`.gitignore`); download it from the source above. The pipeline code in this
repository is my own work.
