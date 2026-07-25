"""Load Siena EDF recordings + seizure annotations, OR synthesise a
realistic fallback so the whole pipeline runs before the 20 GB download.

A "Recording" is the common object the rest of the pipeline consumes:
    data      : np.ndarray  (n_channels, n_samples), microvolts
    ch_names  : list[str]
    sfreq     : float
    seizures  : list[(onset_sec, offset_sec)]
    subject   : str
    synthetic : bool
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np

from config import (SFREQ, STD_CHANNELS, SIENA_CHANNELS, DATA_RAW, RANDOM_STATE,
                    PREICTAL_SEC, SPH_SEC, POSTICTAL_SEC, INTERICTAL_GUARD_SEC,
                    INTERICTAL_MAX_SEC, PREICTAL_LEADIN_SEC)


@dataclass
class Recording:
    data: np.ndarray
    ch_names: list
    sfreq: float
    seizures: list = field(default_factory=list)   # [(onset_s, offset_s), ...]
    subject: str = "SYN"
    synthetic: bool = False

    @property
    def duration_sec(self) -> float:
        return self.data.shape[1] / self.sfreq


# ---------------------------------------------------------------- real data
_TIME_RE = re.compile(r"(\d{1,2}[.:]\d{2}[.:]\d{2})")


def _to_sec(s: str) -> int:
    """'21:51:02' or '9.35.15' -> seconds since midnight (handles . or : sep)."""
    h, m, sec = (int(p) for p in re.split(r"[.:]", s.strip())[:3])
    return h * 3600 + m * 60 + sec


def parse_seizure_list(txt_path: Path) -> dict:
    """Parse a Siena `Seizures-list-PNxx.txt` into {edf_filename: [(onset, offset)]}.

    Robust to the format drift across patients: 'Seizure start time' vs bare
    'Start time', '.' vs ':' separators, and overnight recordings where a
    seizure clock time is *earlier* than the registration start (midnight wrap).
    All times are returned in seconds relative to each file's registration start.
    """
    text = txt_path.read_text(errors="ignore")
    out: dict = {}
    cur_file, reg_start = None, None

    for line in text.splitlines():
        low = line.lower().strip()
        m = _TIME_RE.search(line)
        if low.startswith("file name"):
            cur_file = line.split(":", 1)[-1].strip()
            out.setdefault(cur_file, [])
            reg_start = None
        elif "registration start" in low and m:
            reg_start = _to_sec(m.group(1))
        elif "registration" in low:
            continue                                   # ignore registration end
        elif "start time" in low and m and reg_start is not None and cur_file:
            onset = _to_sec(m.group(1)) - reg_start
            if onset < 0:                              # crossed midnight
                onset += 86400
            out[cur_file].append([float(onset), None])
        elif "end time" in low and m and reg_start is not None and out.get(cur_file):
            off = _to_sec(m.group(1)) - reg_start
            if off < 0:
                off += 86400
            entry = out[cur_file][-1]
            if entry[1] is None:
                if off < entry[0]:                     # end wrapped past midnight
                    off += 86400
                entry[1] = float(off)
    return {k: [tuple(v) for v in vs] for k, vs in out.items()}


def _pick_eeg_channels(raw):
    """Return (picks, canonical_names) selecting only real EEG channels from the
    Siena montage, case-insensitively (handles 'EEG Fp1', 'FP2', drops SPO2/HR/MK).
    """
    norm = {}
    for c in raw.ch_names:
        n = re.sub(r"^\s*EEG\s*", "", c).strip()
        norm[n.upper()] = c                         # normalised -> original
    picks, names = [], []
    for target in SIENA_CHANNELS:
        orig = norm.get(target.upper())
        if orig is not None:
            picks.append(orig)
            names.append(target)
    return picks, names


def _plan_crops(seizures, duration):
    """Given seizure (onset, offset) times and recording duration, return
    [(tmin, tmax, seizures_rel, tag)], one continuous crop per seizure plus
    bounded interictal crops kept >guard from every seizure. Cropping happens
    BEFORE load so multi-hour recordings never hit RAM whole.

    Each seizure crop now reaches back past the interictal guard to include a
    genuine interictal lead-in (INTERICTAL_GUARD_SEC + PREICTAL_LEADIN_SEC before
    onset), so the alarm layer must lift firing power from a real baseline rather
    than sit on an all-preictal crop. The start is clipped at the previous
    seizure's postictal end, so each crop still holds exactly one seizure and a
    tightly-spaced pair simply yields whatever clean baseline actually exists.
    """
    seiz = sorted((o, (f if f is not None else o + 60)) for o, f in seizures)
    crops = []
    lead = INTERICTAL_GUARD_SEC + PREICTAL_LEADIN_SEC
    prev_end = -np.inf
    for o, f in seiz:
        tmin = max(0.0, o - lead, prev_end + POSTICTAL_SEC)
        tmax = min(duration, f + POSTICTAL_SEC)     # labeler drops ictal/postictal
        crops.append((tmin, tmax, [(o - tmin, f - tmin)], "preictal"))
        prev_end = f

    def clean(c0, c1):
        return all(c1 < o - INTERICTAL_GUARD_SEC or c0 > f + INTERICTAL_GUARD_SEC
                   for o, f in seiz)

    gathered, t, chunk = 0.0, 0.0, 30 * 60
    while t < duration and gathered < INTERICTAL_MAX_SEC:
        seg = min(chunk, duration - t, INTERICTAL_MAX_SEC - gathered)
        if seg < 5 * 60:
            break
        if clean(t, t + seg):
            crops.append((t, t + seg, [], "interictal"))
            gathered += seg
        t += seg
    return crops


def load_edf_segments(edf_path: Path, seizures=None):
    """Yield cropped, channel-harmonised Recording segments from one EDF."""
    import mne  # lazy: synthetic path needs no mne

    hdr = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    sfreq = float(hdr.info["sfreq"])
    max_time = (hdr.n_times - 1) / sfreq            # time of the last sample
    picks, names = _pick_eeg_channels(hdr)
    subject = edf_path.parent.name

    for tmin, tmax, sz_rel, tag in _plan_crops(list(seizures or []), max_time):
        if tmin >= max_time:
            continue
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        raw.pick(picks).reorder_channels(picks)
        raw.crop(tmin=tmin, tmax=min(tmax, max_time))
        raw.load_data(verbose="ERROR")                       # loads only the crop
        raw.rename_channels(dict(zip(picks, names)))
        data = (raw.get_data() * 1e6).astype(np.float32)     # V -> uV
        yield Recording(data=data, ch_names=names,
                        sfreq=float(raw.info["sfreq"]), seizures=sz_rel,
                        subject=subject, synthetic=False)


def _match_ann(edf_name, ann):
    """Look up seizures for edf_name, tolerating filename drift
    ('PN01.edf' in the list vs 'PN01-1.edf' on disk)."""
    if edf_name in ann:
        return ann[edf_name]
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().replace(".edf", ""))
    e = norm(edf_name)
    cands = [v for k, v in ann.items()
             if v and (norm(k).startswith(e) or e.startswith(norm(k)))]
    return cands[0] if len(cands) == 1 else []


def discover_recordings(root: Path = DATA_RAW):
    """Yield (edf_path, seizures) for every EDF under data/raw/PNxx/."""
    root = Path(root)
    for subj_dir in sorted(p for p in root.glob("PN*") if p.is_dir()):
        lists = list(subj_dir.glob("Seizures-list-*.txt"))
        ann = parse_seizure_list(lists[0]) if lists else {}
        for edf in sorted(subj_dir.glob("*.edf")):
            yield edf, _match_ann(edf.name, ann)


# ---------------------------------------------------------------- synthetic fallback
def synth_recording(minutes=120, sfreq=256.0, seed=RANDOM_STATE,
                    subject="SYN") -> Recording:
    """Generate physiologically-flavoured EEG for ONE subject.

    Mirrors real Siena structure: a long recording that contains BOTH a
    far-from-seizure interictal baseline (early in the record) AND a single
    seizure late in the record with a pre-ictal build-up (rising beta/gamma +
    rhythmic 3 Hz spike-wave at onset). So every synthetic subject contributes
    both classes, the condition grouped CV needs to score an AUC.

    Synthesised at 256 Hz (not 512) purely to keep the demo's memory/compute
    light; the pipeline is sample-rate-driven, so real 512 Hz EDFs run
    unchanged.
    """
    rng = np.random.default_rng(seed)
    ch = list(STD_CHANNELS)
    n_ch = len(ch)
    n = int(minutes * 60 * sfreq)
    t = np.arange(n) / sfreq
    x = np.zeros((n_ch, n), dtype=np.float32)

    # 1/f background via cumulative-sum noise + 10 Hz alpha rhythm
    for i in range(n_ch):
        pink = np.cumsum(rng.standard_normal(n))
        pink -= np.linspace(pink[0], pink[-1], n)          # detrend
        pink /= (pink.std() + 1e-9)
        alpha = 0.6 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        x[i] = 15 * pink + 8 * alpha + rng.standard_normal(n)

    # one seizure late in the record so early minutes are true interictal
    onset = minutes * 60 * rng.uniform(0.82, 0.90)
    offset = onset + rng.uniform(40, 90)
    seizures = [(onset, offset)]

    # ramp spans the full labelled preictal window [onset-PREICTAL-SPH, onset)
    # so the injected signature actually covers the epochs we tag as preictal.
    pre = PREICTAL_SEC + SPH_SEC
    ramp = np.clip((t - (onset - pre)) / pre, 0, 1)
    x += (ramp * (10 * np.sin(2 * np.pi * 22 * t) +       # growing beta
                  7 * np.sin(2 * np.pi * 40 * t))).astype(np.float32)  # + gamma
    ict = ((t >= onset) & (t <= offset)).astype(np.float32)
    x += (ict * 60 * np.sign(np.sin(2 * np.pi * 3 * t))).astype(np.float32)  # spike-wave

    # inject a bad channel in ~half the subjects so QC + interpolation have work
    if seed % 2 == 0:
        bad = ch.index("T4")
        x[bad] += (5 * x[bad].std() * rng.standard_normal(n)).astype(np.float32)
    return Recording(data=x, ch_names=ch, sfreq=sfreq,
                    seizures=seizures, subject=subject, synthetic=True)


def harmonize_to(recs, channels):
    """Force every recording onto an exact, fixed channel list (order preserved),
    so feature vectors line up across patients processed in separate passes.
    Raises if a recording is missing a required channel (keeps the montage honest
    rather than silently changing feature dimensionality)."""
    for r in recs:
        missing = [c for c in channels if c not in r.ch_names]
        if missing:
            raise ValueError(f"{r.subject}: missing required channels {missing}; "
                             f"has {r.ch_names}")
        idx = [r.ch_names.index(c) for c in channels]
        r.data = r.data[idx]
        r.ch_names = list(channels)
    return recs


def _harmonize_channels(recs):
    """Subset every recording to the channels common to all, in a fixed order,
    so the per-channel feature vectors line up across patients."""
    if not recs:
        return recs
    common = set(recs[0].ch_names)
    for r in recs[1:]:
        common &= set(r.ch_names)
    order = [c for c in SIENA_CHANNELS if c in common] or sorted(common)
    for r in recs:
        idx = [r.ch_names.index(c) for c in order]
        r.data = r.data[idx]
        r.ch_names = list(order)
    return recs


def load_dataset(prefer_real=True, n_synth_subjects=5, verbose=True):
    """Return a list of Recordings. Real Siena data if present, else synthetic.

    Real path: each patient's EDFs are cropped into short preictal + interictal
    segments (subject id preserved for grouped CV), then all segments are
    harmonised onto their common channel set.
    """
    if prefer_real and any(Path(DATA_RAW).glob("PN*")):
        recs = []
        for edf, sz in discover_recordings():
            for seg in load_edf_segments(edf, sz):
                recs.append(seg)
            if verbose:
                print(f"    loaded {edf.name}  (seizures={len(sz)})")
        recs = _harmonize_channels(recs)
        if recs:
            if verbose:
                print(f"    -> {len(recs)} segments · {len(recs[0].ch_names)} common channels")
            return recs
    return [synth_recording(minutes=120, seed=RANDOM_STATE + s,
                            subject=f"SYN{s:02d}")
            for s in range(n_synth_subjects)]
