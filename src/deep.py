"""Phase 2: a small temporal CNN on per-channel spectrograms, evaluated in the
personalized regime (the only regime where the classical model beat chance).

This is a deliberately compact 2D CNN. On this dataset each patient offers only
a few hundred to a few thousand windows, which is far less than a deep model
wants, so the honest hypothesis is that the CNN will be competitive with, not
dramatically better than, the classical logistic regression. I report the
comparison as-is; a network that ties a linear model on this little data is the
expected and truthful result, and the scaffold is ready for more data.

CPU-only, fixed seed. Uses leave-one-seizure-out per patient, identical folds to
the classical personalized evaluation, so the two are directly comparable.
"""
from __future__ import annotations
import json
import numpy as np

from config import OUTPUTS, RANDOM_STATE


def _torch():
    import torch
    torch.manual_seed(RANDOM_STATE)
    return torch


class _CNN:
    """Lazily-built small CNN wrapper (keeps torch import inside the deep path)."""
    def __init__(self, n_ch, f_bins, t_bins):
        torch = _torch()
        import torch.nn as nn
        self.torch = torch
        self.net = nn.Sequential(
            nn.Conv2d(n_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, 1))

    def fit(self, X, y, epochs=20, lr=1e-3):
        torch, nn = self.torch, self.torch.nn
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        pos = max(1, int((y == 1).sum())); neg = max(1, int((y == 0).sum()))
        pos_weight = torch.tensor([neg / pos])          # rebalance the rarer class
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                out = self.net(Xt[b]).squeeze(1)
                loss_fn(out, yt[b]).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.tensor(X, dtype=torch.float32)).squeeze(1))
        return p.numpy()


def _standardize(train, *others):
    mu = train.mean(axis=(0, 3), keepdims=True)
    sd = train.std(axis=(0, 3), keepdims=True) + 1e-6
    return tuple((a - mu) / sd for a in (train, *others))


def deep_personalized(S, y, groups, meta, epochs=20, save=True):
    """Leave-one-seizure-out CNN per patient. Returns a results dict and prints a
    per-patient table with the classical AUC alongside for comparison."""
    from sklearn.metrics import roc_auc_score
    from personalized import _patient_folds

    S = S.astype(np.float32)
    classical = {}
    try:
        classical = json.load(open(OUTPUTS / "personalized_results.json"))["per_patient"]
    except Exception:
        pass

    per_patient, pooled_true, pooled_prob = {}, [], []
    for patient in sorted(set(groups)):
        pm = groups == patient
        Sp, yp, cp = S[pm], y[pm], meta["crop"][pm]
        folds, reason = _patient_folds(yp, cp, meta["crops"])
        if not folds:
            per_patient[patient] = {"skipped": reason}
            continue
        aucs = []
        for tr, te in folds:
            Xtr, Xte = _standardize(Sp[tr], Sp[te])
            model = _CNN(S.shape[1], S.shape[2], S.shape[3]).fit(Xtr, yp[tr], epochs=epochs)
            prob = model.predict_proba(Xte)
            if len(set(yp[te])) > 1:
                aucs.append(roc_auc_score(yp[te], prob))
            pooled_true.append(yp[te]); pooled_prob.append(prob)
        per_patient[patient] = {"seizures": len(folds),
                                "auc_cnn": float(np.nanmean(aucs)) if aucs else float("nan"),
                                "auc_logreg": classical.get(patient, {}).get("auc", float("nan"))}

    evaluable = {p: r for p, r in per_patient.items() if "auc_cnn" in r}
    mean_cnn = float(np.nanmean([r["auc_cnn"] for r in evaluable.values()])) if evaluable else float("nan")
    pooled_auc = (roc_auc_score(np.concatenate(pooled_true), np.concatenate(pooled_prob))
                  if pooled_true else float("nan"))
    results = {"model": "cnn", "epochs": epochs, "spectrogram_shape": list(S.shape[1:]),
               "n_patients_evaluable": len(evaluable), "mean_patient_auc_cnn": mean_cnn,
               "pooled_auc_cnn": pooled_auc, "per_patient": per_patient}
    _print(results)
    if save:
        (OUTPUTS / "deep_results.json").write_text(json.dumps(results, indent=2))
        print(f"\nsaved -> {OUTPUTS / 'deep_results.json'}")
    return results


def _print(r):
    print(f"\nPersonalized deep CNN (leave-one-seizure-out, spectrogram "
          f"{r['spectrogram_shape']}, {r['epochs']} epochs)")
    print("-" * 52)
    print(f"{'patient':10}{'seiz':>6}{'AUC CNN':>10}{'AUC logreg':>12}")
    for p in sorted(r["per_patient"]):
        d = r["per_patient"][p]
        if "auc_cnn" in d:
            lr = d["auc_logreg"]
            lrs = f"{lr:>12.3f}" if isinstance(lr, float) and np.isfinite(lr) else f"{'--':>12}"
            print(f"{p:10}{d['seizures']:>6}{d['auc_cnn']:>10.3f}{lrs}")
        else:
            print(f"{p:10}{'--':>6}{'skip: ' + d['skipped']:>22}")
    print("-" * 52)
    print(f"mean patient AUC  CNN={r['mean_patient_auc_cnn']:.3f}")
    print(f"pooled AUC        CNN={r['pooled_auc_cnn']:.3f}")
