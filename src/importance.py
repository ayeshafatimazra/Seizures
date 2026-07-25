"""Which features carry the personalized signal?

With 514 features it is not enough to report an AUC; I want to know what the
model leans on. For each evaluable patient I fit a standardized logistic
regression on all of that patient's windows, take the absolute coefficients
(comparable because the inputs are standardized), collapse them onto feature
*families* (a family is one feature type across all channels, e.g. every
`delta_abs_*`), and average each family's importance across patients.

This is a descriptive, model-based importance, not a causal claim. It answers
"what does the personalized classifier weight?", which is the honest question.
"""
from __future__ import annotations
import json
import re
import numpy as np

from config import OUTPUTS

# human-readable labels for the feature-name prefixes produced by features.py
FAMILY_LABELS = {
    "delta_abs": "delta power (abs)", "theta_abs": "theta power (abs)",
    "alpha_abs": "alpha power (abs)", "beta_abs": "beta power (abs)",
    "gamma_abs": "gamma power (abs)", "delta_rel": "delta power (rel)",
    "theta_rel": "theta power (rel)", "alpha_rel": "alpha power (rel)",
    "beta_rel": "beta power (rel)", "gamma_rel": "gamma power (rel)",
    "tbr": "theta/beta ratio", "tar": "theta/alpha ratio", "bar": "beta/alpha ratio",
    "dtr": "delta/theta ratio", "slow": "slowing ratio", "sef95": "spectral edge freq",
    "centroid": "spectral centroid", "apf": "alpha peak freq", "pse": "1/f exponent",
    "ll": "line length", "rms": "RMS", "var": "variance", "zcr": "zero-crossing rate",
    "hj_act": "Hjorth activity", "hj_mob": "Hjorth mobility", "hj_cmp": "Hjorth complexity",
    "pe": "permutation entropy", "faa": "frontal alpha asymmetry",
}


def _family(name):
    """delta_abs_7 -> delta_abs ; faa -> faa."""
    m = re.match(r"^(.*?)(?:_\d+)?$", name)
    return m.group(1) if m else name


def feature_importance(X, y, groups, meta, feat_names, model_factory, save=True):
    """Mean absolute standardized logistic-regression coefficient per feature
    family, averaged over evaluable patients (>=2 seizures and both classes)."""
    from personalized import _patient_folds       # reuse the evaluability check

    families = [_family(n) for n in feat_names]
    uniq = sorted(set(families))
    fam_idx = {f: np.array([i for i, g in enumerate(families) if g == f]) for f in uniq}

    per_patient = []
    for patient in sorted(set(groups)):
        pm = groups == patient
        folds, reason = _patient_folds(y[pm], meta["crop"][pm], meta["crops"])
        if not folds:
            continue
        model = model_factory()
        model.fit(X[pm], y[pm])
        coef = np.abs(model.named_steps["clf"].coef_.ravel())
        per_patient.append({f: float(coef[idx].mean()) for f, idx in fam_idx.items()})

    agg = {f: float(np.mean([p[f] for p in per_patient])) for f in uniq}
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    results = {"n_patients": len(per_patient),
               "family_importance": {f: v for f, v in ranked}}
    _print(ranked)
    if save:
        (OUTPUTS / "feature_importance.json").write_text(json.dumps(results, indent=2))
        _plot(ranked)
        print(f"\nsaved -> {OUTPUTS / 'feature_importance.json'} and figures/feature_importance.png")
    return results


def _print(ranked, top=12):
    print(f"\nFeature-family importance (mean |standardized coef|, personalized logreg)")
    print("-" * 52)
    for f, v in ranked[:top]:
        print(f"  {FAMILY_LABELS.get(f, f):24} {v:.3f}")


def _plot(ranked, top=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    top_items = ranked[:top][::-1]
    labels = [FAMILY_LABELS.get(f, f) for f, _ in top_items]
    vals = [v for _, v in top_items]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(labels, vals, color="#2e6f9e")
    ax.set_xlabel("mean |standardized coefficient|")
    ax.set_title("What the personalized model weights\n(top feature families)",
                 fontsize=11, color="#1b2a4a")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    figdir = Path(__file__).resolve().parents[1] / "figures"
    figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / "feature_importance.png", dpi=150)
    plt.close(fig)
