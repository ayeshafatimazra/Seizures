#!/usr/bin/env python
"""End-to-end seizure-PREDICTION pipeline on the Siena Scalp EEG database.

    load  ->  preprocess (notch/bandpass/bad-chan interp/CAR)  ->  QC (SNR)
          ->  windowed features (spectral + temporal)          ->  classical ML

Runs on the real Siena data if EDFs are present under data/raw/PNxx/, otherwise
on a synthetic fallback so the whole thing executes today.

    python run_pipeline.py            # auto: real if present, else synthetic
    python run_pipeline.py --synthetic
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd

from data_loader import load_dataset
from preprocess import preprocess
from qc import qc_report
from dataset import build_epochs
from train import cross_validate, make_model
from evaluate import alarm_evaluate
from config import OUTPUTS


def main(prefer_real=True):
    print("=" * 60)
    print(" SEIZURE-PREDICTION PIPELINE  ·  Siena Scalp EEG")
    print("=" * 60)

    recs = load_dataset(prefer_real=prefer_real)
    mode = "REAL Siena EDF" if not recs[0].synthetic else "SYNTHETIC fallback"
    print(f"\n[1] Loaded {len(recs)} recording(s), mode: {mode}")

    print("\n[2] Preprocess  (notch 50 Hz · bandpass 0.5-70 · bad-chan interp · CAR)")
    clean, qc_rows = [], []
    for rec in recs:
        c, meta = preprocess(rec, verbose=True)
        clean.append(c)
        qc_rows.append({**qc_report(c), **{"interp": meta["interp_method"]}})

    print("\n[3] Quality control")
    qc_df = pd.DataFrame(qc_rows)
    print(qc_df.to_string(index=False))
    qc_df.to_csv(OUTPUTS / "qc_report.csv", index=False)

    print("\n[4] Feature extraction + labelling (preictal vs interictal)")
    X, y, groups, names, meta = build_epochs(clean, verbose=True)
    print(f"\n    X = {X.shape}  ·  features/epoch = {X.shape[1]}  ·  "
          f"positives(preictal) = {int(y.sum())}/{len(y)}  ·  subjects = {len(set(groups))}")

    print("\n[5] Classical ML, window-level, subject-grouped cross-validation")
    cross_validate(X, y, groups)

    print("\n[6] Alarm layer, event-level evaluation (Firing Power + tuned threshold)")
    alarm_evaluate(X, y, groups, meta, model_factory=lambda: make_model("logreg"),
                   model_name="logreg")

    print("\nDONE. Artifacts in ./outputs/  "
          "(next phase: PyTorch temporal CNN, see README).")


if __name__ == "__main__":
    main(prefer_real="--synthetic" not in sys.argv)
