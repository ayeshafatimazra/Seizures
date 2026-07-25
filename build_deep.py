#!/usr/bin/env python
"""Phase 2 build: per-patient spectrogram extraction (disk-safe) + CNN eval.

Same streaming strategy as build_all.py: sync one patient from PhysioNet S3,
turn its windows into compact spectrogram tensors, cache them to
outputs/spectrograms/PNxx.npz, free the raw EDFs, move on. Then aggregate every
patient and run the personalized (leave-one-seizure-out) CNN.

    python build_deep.py            # extract (resumable) + evaluate
    python build_deep.py --eval     # re-aggregate caches + evaluate only
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from config import DATA_RAW, OUTPUTS
from preprocess import preprocess
from spectrogram import build_spectrograms
from deep import deep_personalized
from build_all import PATIENTS, KEEP_RAW, sync, load_patient   # reuse the loaders

SPCACHE = OUTPUTS / "spectrograms"


def process(p):
    cache = SPCACHE / f"{p}.npz"
    if cache.exists():
        print(f"[{p}] cached — skip"); return
    if not (DATA_RAW / p).exists() or not list((DATA_RAW / p).glob("*.edf")):
        print(f"[{p}] sync from S3 ...", flush=True); sync(p)
    print(f"[{p}] preprocess + spectrograms ...", flush=True)
    clean = [preprocess(r)[0] for r in load_patient(p)]
    S, y, groups, meta = build_spectrograms(clean, verbose=True)
    np.savez_compressed(cache, S=S, y=y, groups=groups,
                        crop=meta["crop"], t0=meta["t0"],
                        crops=json.dumps(meta["crops"]))
    print(f"[{p}] cached  S={S.shape}", flush=True)
    if p not in KEEP_RAW:
        shutil.rmtree(DATA_RAW / p, ignore_errors=True)
        print(f"[{p}] raw EDFs freed", flush=True)


def aggregate():
    Ss, ys, gs, ep_crop, ep_t0, crops = [], [], [], [], [], []
    off = 0
    for p in PATIENTS:
        cache = SPCACHE / f"{p}.npz"
        if not cache.exists():
            continue
        d = np.load(cache, allow_pickle=False)
        Ss.append(d["S"]); ys.append(d["y"]); gs.append(d["groups"])
        ep_crop.append(d["crop"] + off); ep_t0.append(d["t0"])
        pc = json.loads(str(d["crops"]))
        crops += pc; off += len(pc)
    if not Ss:
        raise RuntimeError("No spectrogram caches — run without --eval first.")
    meta = {"crop": np.concatenate(ep_crop), "t0": np.concatenate(ep_t0), "crops": crops}
    return np.concatenate(Ss), np.concatenate(ys), np.concatenate(gs), meta


def main(eval_only=False):
    SPCACHE.mkdir(parents=True, exist_ok=True)
    if not eval_only:
        for p in PATIENTS:
            process(p)
    S, y, groups, meta = aggregate()
    print("\n" + "=" * 60)
    print(f" DEEP COHORT  ·  S={S.shape}  ·  subjects={len(set(groups))}"
          f"  ·  preictal={int(y.sum())}/{len(y)}")
    print("=" * 60)
    deep_personalized(S, y, groups, meta, epochs=20)


if __name__ == "__main__":
    main(eval_only="--eval" in sys.argv)
