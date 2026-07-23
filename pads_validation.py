"""Real-tremor validation on PADS (PhysioNet Parkinson's Disease Smartwatch).

469 patients, 100 Hz wrist accelerometer+gyroscope, condition-labelled
(Parkinson's 276, Healthy 79, Essential Tremor 28, ...). Rest tremor = 'Relaxed'
task; postural tremor = 'StretchHold'. Accelerometer is the clinical gold
standard for tremor frequency.

We run the project's OWN spectral estimator (features._peak_freq with the same
drift-removing high-pass used on video) on the raw sensor channels and check:
  (1) does it recover physiological tremor frequency (PD rest ~4-6 Hz; ET ~4-11 Hz)?
  (2) do our features separate PD and ET from Healthy controls (ROC-AUC)?
  (3) does the recovered frequency see the ET-vs-PD difference?

Scope: validates the spectral frequency/tremor core on real tremor across a large,
properly-controlled cohort (79 HC). It does NOT test the 21-landmark model or the
video path (accelerometer, not landmarks) -- but it is far stronger evidence than
UCI-395 for the core the model depends on.
"""
from __future__ import annotations

import zipfile
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from features import _peak_freq, _highpass

ZIP = Path(__file__).resolve().parent / "realdata" / "pads.zip"
R = "pads-parkinsons-disease-smartwatch-dataset-1.0.0/"
Z = zipfile.ZipFile(ZIP)
NAMES = set(Z.namelist())


def conditions():
    out = {}
    for n in NAMES:
        if "/patients/patient_" in n and n.endswith(".json"):
            j = json.loads(Z.read(n))
            out[j["id"]] = j["condition"]
    return out


def descriptor(pid, task):
    """Best-wrist tremor descriptor for a patient+task, or None."""
    best = None
    for wrist in ("Left", "Right"):
        name = f"{R}movement/timeseries/{pid}_{task}_{wrist}Wrist.txt"
        if name not in NAMES:
            continue
        rows = [l.split(",") for l in Z.read(name).decode("utf-8", "replace").splitlines() if l.strip()]
        try:
            a = np.array(rows, float)
        except ValueError:
            continue
        if a.shape[0] < 128 or a.shape[1] < 7:
            continue
        t = a[:, 0]
        fs = 1.0 / np.median(np.diff(t))
        if not (50 <= fs <= 200):
            continue
        specs = [_peak_freq(_highpass(a[:, c], fs), fs) for c in range(1, 7)]
        dom = max(specs, key=lambda s: s["peak_prom"])
        cand = dict(peak=dom["peak"], prom=dom["peak_prom"],
                    total=float(sum(s["total"] for s in specs)),
                    tremor_band=float(np.mean([sum(s["bandfrac"][1:4]) for s in specs])))
        if best is None or cand["prom"] > best["prom"]:
            best = cand
    return best


def auc(pos, neg, key):
    s = np.array([r[key] for r in pos] + [r[key] for r in neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    a = roc_auc_score(y, np.nan_to_num(s))
    return max(a, 1 - a)


def main():
    cond = conditions()
    rest, post = {}, {}
    for pid, c in cond.items():
        r = descriptor(pid, "Relaxed")
        p = descriptor(pid, "StretchHold")
        if r:
            rest.setdefault(c, []).append(r)
        if p:
            post.setdefault(c, []).append(p)

    PD, HC, ET = "Parkinson's", "Healthy", "Essential Tremor"
    n = {c: len(rest.get(c, [])) for c in (PD, HC, ET)}
    print(f"PADS parsed (rest task): PD={n[PD]}, HC={n[HC]}, ET={n[ET]}  "
          f"(postural: PD={len(post.get(PD,[]))}, HC={len(post.get(HC,[]))}, ET={len(post.get(ET,[]))})\n")

    print("(1) FACE VALIDITY — recovered dominant tremor frequency (Hz):")
    for label, c, task, dd in [("PD rest", PD, rest, "4-6"), ("PD postural", PD, post, "4-6"),
                               ("ET postural", ET, post, "4-11"), ("HC rest", HC, rest, "-")]:
        pk = np.array([r["peak"] for r in task.get(c, [])])
        if len(pk):
            print(f"    {label:12s} median={np.median(pk):.2f}  IQR=[{np.percentile(pk,25):.2f},"
                  f"{np.percentile(pk,75):.2f}]  (expected ~{dd} Hz)  n={len(pk)}")

    print("\n(2) SEPARATION vs Healthy controls (ROC-AUC using our spectral features):")
    for grp, task, gname, tname in [(PD, rest, "PD", "rest"), (ET, post, "ET", "postural")]:
        pos, neg = task.get(grp, []), task.get(HC, [])
        if pos and neg:
            row = "  ".join(f"{k}={auc(pos,neg,k):.3f}" for k in ("tremor_band", "prom", "total"))
            print(f"    {gname} vs HC ({tname}): {row}   (n={len(pos)} vs {len(neg)})")

    print("\n(3) ET-vs-PD frequency split (postural tremor):")
    et_f = [r["peak"] for r in post.get(ET, [])]
    pd_f = [r["peak"] for r in post.get(PD, [])]
    if et_f and pd_f:
        try:
            p = mannwhitneyu(et_f, pd_f).pvalue
        except ValueError:
            p = float("nan")
        print(f"    ET median={np.median(et_f):.2f} Hz vs PD median={np.median(pd_f):.2f} Hz  "
              f"(Mann-Whitney p={p:.3g})")

    print("\nScope: real-tremor validation of the spectral core across a controlled")
    print("cohort (79 HC). Not the 21-landmark model or the video path (accelerometer).")


if __name__ == "__main__":
    main()
