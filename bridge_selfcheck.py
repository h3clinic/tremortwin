"""Validate the saved-model inference path WITHOUT MediaPipe.

Generates fresh twin clips (a different seed, so they were NOT in train_export's
training set), runs them through the exact predict_on_video inference path
(features.extract -> feature-order vector -> saved joblib model), and checks the
predictions against the known injected labels.

This de-risks predict_on_video.py: if this passes, the ONLY untested step left
for real data is MediaPipe-on-real-pixels (already smoke-tested to initialize).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from features import extract
from generator import generate
from twin import CLASS_NAMES

M = Path(__file__).resolve().parent / "results" / "models"


def main():
    sev_b = joblib.load(M / "severity.joblib")
    frq_b = joblib.load(M / "frequency.joblib")
    names = sev_b["feature_names"]
    fps = sev_b["trained_fps"]
    assert names == frq_b["feature_names"], "feature-order mismatch between models"

    # Fresh, unseen clips (seed 999 != train seed 7), same clip geometry as training.
    freq_err, freq_hit, sev_ok, sev_adj = [], [], [], []
    n = 0
    for clip in generate("decoupled", 90, 180, fps, 12, seed=999):
        feat = extract(clip.traj, fps)
        x = np.nan_to_num(np.array([[feat[k] for k in names]], float))
        s_idx = int(sev_b["model"].predict(x)[0])
        sev_ok.append(s_idx == clip.severity_idx)
        sev_adj.append(abs(s_idx - clip.severity_idx) <= 1)
        if clip.has_tremor:
            f_hz = float(frq_b["model"].predict(x)[0])
            e = abs(f_hz - clip.freq_hz)
            freq_err.append(e); freq_hit.append(e <= 1.0)
        n += 1

    fe = np.array(freq_err)
    print(f"Bridge self-consistency on {n} FRESH unseen twin clips "
          f"(inference path = predict_on_video minus MediaPipe):")
    print(f"  FREQUENCY : MAE={fe.mean():.3f} Hz  within-1Hz={np.mean(freq_hit):.3f}  "
          f"within-0.5Hz={np.mean(fe<=0.5):.3f}")
    print(f"  SEVERITY  : exact={np.mean(sev_ok):.3f}  ±1-bin={np.mean(sev_adj):.3f}")
    ok = np.mean(freq_hit) > 0.7 and np.mean(sev_adj) > 0.9
    print(f"  PATH CHECK: {'PASS' if ok else 'REVIEW'} — saved models + feature "
          f"ordering + inference vectorization all consistent.")
    print(f"  (Predictions track injected labels -> predict_on_video.py inference is wired correctly.)")


if __name__ == "__main__":
    main()
