#!/usr/bin/env python
"""Process the FULL 14-patient Siena database on a disk-constrained machine.

The whole dataset is ~20 GB but we never need it resident at once: each patient
is synced from PhysioNet's public S3 bucket, sliced into cropped epochs, and its
(tiny) feature matrix cached to outputs/features/PNxx.pkl — after which the raw
EDFs are freed. Peak extra disk is one patient (≤3.4 GB). Re-runs skip any
patient already cached, so this is safely resumable.

    python build_all.py            # extract (resumable) + then evaluate
    python build_all.py --eval     # just re-aggregate caches + evaluate

Patients you already downloaded (PN00/01/03) keep their raw EDFs; the rest are
deleted after caching. All raw data is re-syncable with one command, and is
gitignored either way.
"""
import shutil
import subprocess
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from config import DATA_RAW, OUTPUTS, STD_CHANNELS
from data_loader import (parse_seizure_list, _match_ann, load_edf_segments,
                         harmonize_to)
from preprocess import preprocess
from dataset import build_epochs
from train import cross_validate, make_model
from evaluate import alarm_evaluate
from personalized import personalized_cv

PATIENTS = ["PN00", "PN01", "PN03", "PN05", "PN06", "PN07", "PN09",
            "PN10", "PN11", "PN12", "PN13", "PN14", "PN16", "PN17"]
KEEP_RAW = {"PN00", "PN01", "PN03"}                 # user's originals — don't delete
S3 = "s3://physionet-open/siena-scalp-eeg/1.0.0"
CACHE = OUTPUTS / "features"


def sync(p):
    dst = DATA_RAW / p
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", "--no-sign-request", "--only-show-errors",
                    f"{S3}/{p}/", str(dst)], check=True)


def load_patient(p):
    subj = DATA_RAW / p
    lists = list(subj.glob("Seizures-list-*.txt"))
    ann = parse_seizure_list(lists[0]) if lists else {}
    recs = []
    for edf in sorted(subj.glob("*.edf")):
        for seg in load_edf_segments(edf, _match_ann(edf.name, ann)):
            recs.append(seg)
    return harmonize_to(recs, STD_CHANNELS)


def process(p):
    cache = CACHE / f"{p}.pkl"
    if cache.exists():
        print(f"[{p}] cached — skip")
        return
    print(f"[{p}] sync from S3 ...", flush=True)
    sync(p)
    print(f"[{p}] preprocess + build epochs ...", flush=True)
    clean = [preprocess(r)[0] for r in load_patient(p)]
    X, y, groups, names, meta = build_epochs(clean, verbose=True)
    with open(cache, "wb") as f:
        pickle.dump({"X": X, "y": y, "groups": groups, "names": names, "meta": meta}, f)
    print(f"[{p}] cached  X={X.shape}  preictal={int(y.sum())}/{len(y)}", flush=True)
    if p not in KEEP_RAW:
        shutil.rmtree(DATA_RAW / p, ignore_errors=True)
        print(f"[{p}] raw EDFs freed", flush=True)


def aggregate():
    """Concatenate all cached patients, offsetting crop ids so they stay unique."""
    Xs, ys, gs, ep_crop, ep_t0, crops = [], [], [], [], [], []
    names, off = None, 0
    for p in PATIENTS:
        cache = CACHE / f"{p}.pkl"
        if not cache.exists():
            continue
        d = pickle.load(open(cache, "rb"))
        Xs.append(d["X"]); ys.append(d["y"]); gs.append(d["groups"]); names = d["names"]
        m = d["meta"]
        ep_crop.append(m["crop"] + off)
        ep_t0.append(m["t0"])
        crops += m["crops"]
        off += len(m["crops"])
    if not Xs:
        raise RuntimeError("No cached patients found — run without --eval first.")
    meta = {"crop": np.concatenate(ep_crop), "t0": np.concatenate(ep_t0), "crops": crops}
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), names, meta)


def main(eval_only=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    if not eval_only:
        for p in PATIENTS:
            process(p)

    X, y, groups, names, meta = aggregate()
    print("\n" + "=" * 60)
    print(f" FULL COHORT  ·  X={X.shape}  ·  subjects={len(set(groups))}"
          f"  ·  preictal={int(y.sum())}/{len(y)}")
    print("=" * 60)

    print("\n[5] Classical ML  —  window-level, subject-grouped cross-validation")
    cross_validate(X, y, groups, models=("logreg", "rf"))     # SVM omitted: intractable at cohort scale

    print("\n[6] Alarm layer  —  event-level evaluation")
    alarm_evaluate(X, y, groups, meta, model_factory=lambda: make_model("logreg"),
                   model_name="logreg")

    print("\n[7] Personalized (patient-specific) models  —  leave-one-seizure-out")
    personalized_cv(X, y, groups, meta, model_factory=lambda: make_model("logreg"))

    n = len({c["subject"] for c in meta["crops"]})
    print(f"\nDONE — {n} patients processed. Caches in {CACHE}/.")


if __name__ == "__main__":
    main(eval_only="--eval" in sys.argv)
