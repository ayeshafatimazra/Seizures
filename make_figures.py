#!/usr/bin/env python
"""Generate the figures embedded in the README from the cached features and the
saved result JSONs. Writes PNGs to figures/.

    python make_figures.py

Figures:
  1. cross_vs_personalized_auc.png  — the headline contrast (pooled AUC).
  2. per_patient_auc.png            — per-patient personalized AUC, the heterogeneity.
  3. sensitivity_vs_fpr.png         — per-patient alarm operating points (responders).
  4. firing_power_trace.png         — a worked alarm example on a held-out seizure.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import build_all as B
from train import make_model
from config import (OUTPUTS, SPH_SEC, SOP_SEC, FP_THRESHOLD, FP_WINDOW_SEC,
                    REFRACTORY_SEC)
from postprocess import firing_power, raise_alarms

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
INK, ACCENT, MUTE, BAD = "#1b2a4a", "#2e6f9e", "#9bb4c4", "#b0413e"


def fig_cross_vs_personalized():
    cv = json.load(open(OUTPUTS / "cv_results.json"))
    pers = json.load(open(OUTPUTS / "personalized_results.json"))
    labels = ["Cross-patient\n(pooled)", "Personalized\n(pooled)"]
    vals = [cv["logreg"]["auc"], pers["pooled_auc"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=[MUTE, ACCENT], width=0.6)
    ax.axhline(0.5, ls="--", color=BAD, lw=1)
    ax.text(1.45, 0.51, "chance", color=BAD, fontsize=9, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold", color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Same features, same data: only the personalized\nregime beats chance",
                 fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "cross_vs_personalized_auc.png", dpi=150)
    plt.close(fig)


def fig_per_patient_auc():
    pers = json.load(open(OUTPUTS / "personalized_results.json"))
    items = [(p, d["auc"]) for p, d in pers["per_patient"].items() if "auc" in d]
    items.sort(key=lambda kv: kv[1], reverse=True)
    names = [p for p, _ in items]
    vals = [v for _, v in items]
    colors = [ACCENT if v >= 0.5 else BAD for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color=colors, width=0.7)
    ax.axhline(0.5, ls="--", color="#555", lw=1)
    ax.axhline(pers["pooled_auc"], ls=":", color=INK, lw=1.2)
    ax.text(len(names) - 0.5, pers["pooled_auc"] + 0.01,
            f"pooled {pers['pooled_auc']:.2f}", ha="right", fontsize=9, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ROC-AUC (leave-one-seizure-out)")
    ax.set_title("Personalized prediction is patient-specific:\nsome patients are highly predictable, others not",
                 fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "per_patient_auc.png", dpi=150)
    plt.close(fig)


def fig_sensitivity_vs_fpr():
    al = json.load(open(OUTPUTS / "personalized_alarm_results.json"))
    xs, ys, labs = [], [], []
    for p, d in al["per_patient"].items():
        if "sensitivity" in d and np.isfinite(d["fpr_per_h"]):
            xs.append(d["fpr_per_h"]); ys.append(d["sensitivity"]); labs.append(p)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(xs, ys, c=INK, s=45, zorder=3)
    for x, y, l in zip(xs, ys, labs):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.axhline(0.5, ls="--", color=MUTE, lw=1)
    ax.set_xlabel("False alarms per hour")
    ax.set_ylabel("Event sensitivity")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(left=0)
    ax.set_title("Per-patient alarm operating points\n(upper-left is a useful predictor)",
                 fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity_vs_fpr.png", dpi=150)
    plt.close(fig)


def fig_firing_power_trace(patient="PN03"):
    X, y, groups, names, meta = B.aggregate()
    gcrop = meta["crop"]
    pm = np.where(groups == patient)[0]
    pcrops = np.unique(gcrop[pm])
    seiz = [c for c in pcrops if meta["crops"][c]["seizures"]]
    inter = [c for c in pcrops if not meta["crops"][c]["seizures"]]
    sc = seiz[0]                                       # held-out seizure
    tr = np.where(np.isin(gcrop, [c for c in seiz if c != sc] + inter))[0]
    te = np.where(gcrop == sc)[0]
    m = make_model("logreg"); m.fit(X[tr], y[tr])
    prob = m.predict_proba(X[te])[:, 1]
    t, fp = firing_power(meta["t0"][te], prob, FP_WINDOW_SEC)
    alarms = raise_alarms(t, fp, FP_THRESHOLD, REFRACTORY_SEC)
    onset = meta["crops"][sc]["seizures"][0][0]

    tm = t / 60.0
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(tm, prob, color=MUTE, lw=0.8, alpha=0.8, label="window preictal probability")
    ax.plot(tm, fp, color=ACCENT, lw=2.0, label="firing power (SOP-smoothed)")
    ax.axhline(FP_THRESHOLD, ls="--", color="#555", lw=1, label=f"alarm threshold {FP_THRESHOLD}")
    ax.axvspan((onset - SPH_SEC - SOP_SEC) / 60, (onset - SPH_SEC) / 60,
               color=ACCENT, alpha=0.10, label="SOP (prediction window)")
    ax.axvline(onset / 60, color=BAD, lw=1.5, label="seizure onset")
    for i, a in enumerate(alarms):
        ax.axvline(a / 60, color="#e08a12", lw=1.2, ls=":",
                   label="alarm" if i == 0 else None)
    ax.set_xlabel("time (minutes into recording)")
    ax.set_ylabel("probability / firing power")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Worked alarm example, {patient} held-out seizure",
                 fontsize=11, color=INK)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "firing_power_trace.png", dpi=150)
    plt.close(fig)


def fig_cnn_vs_logreg():
    path = OUTPUTS / "deep_results.json"
    if not path.exists():
        return
    dr = json.load(open(path))
    items = [(p, d["auc_cnn"], d.get("auc_logreg"))
             for p, d in dr["per_patient"].items()
             if "auc_cnn" in d and np.isfinite(d.get("auc_logreg", np.nan))]
    items.sort(key=lambda t: t[2], reverse=True)
    names = [p for p, _, _ in items]
    cnn = [c for _, c, _ in items]
    lr = [l for _, _, l in items]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.2, lr, 0.4, label="classical logreg", color=MUTE)
    ax.bar(x + 0.2, cnn, 0.4, label="temporal CNN", color=ACCENT)
    ax.axhline(0.5, ls="--", color="#555", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0); ax.set_ylabel("ROC-AUC (leave-one-seizure-out)")
    ax.set_title("Deep CNN ties the classical model per patient\n(data-limited: hundreds of windows each)",
                 fontsize=11, color=INK)
    ax.legend(fontsize=9, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "cnn_vs_logreg.png", dpi=150)
    plt.close(fig)


def main():
    fig_cross_vs_personalized()
    fig_per_patient_auc()
    fig_sensitivity_vs_fpr()
    fig_firing_power_trace()
    fig_cnn_vs_logreg()
    print(f"wrote figures to {FIG}/:")
    for p in sorted(FIG.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
