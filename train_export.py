"""Train the frequency + severity models on ALL synthetic (decoupled) data and
persist them, so they can be applied to REAL video for the sim-to-real test.

We train on the full synthetic set (no held-out needed here — the held-out set
is the *real* video). Saves model + the exact feature order so inference on real
MediaPipe landmarks uses identical features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from features import extract
from generator import generate
from twin import CLASS_NAMES

OUT = Path(__file__).resolve().parent / "results" / "models"
OUT.mkdir(parents=True, exist_ok=True)


def build(poses, frames, fps, subjects, seed):
    X, freq, sev, has = [], [], [], []
    names = None
    for i, clip in enumerate(generate("decoupled", poses, frames, fps, subjects, seed)):
        f = extract(clip.traj, fps)
        if names is None:
            names = sorted(f.keys())
        X.append([f[k] for k in names])
        freq.append(clip.freq_hz); sev.append(clip.severity_idx); has.append(clip.has_tremor)
        if (i + 1) % 400 == 0:
            print(f"  extracted {i+1}")
    X = np.nan_to_num(np.array(X, float))
    return X, np.array(freq, float), np.array(sev, int), np.array(has, bool), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=500)
    ap.add_argument("--frames", type=int, default=180)   # 3 s @ 60 fps
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--subjects", type=int, default=28)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print("Generating + extracting synthetic training set...")
    X, freq, sev, has, names = build(args.poses, args.frames, args.fps, args.subjects, args.seed)
    print(f"  X={X.shape}")

    sev_model = ExtraTreesClassifier(n_estimators=600, random_state=0, n_jobs=-1).fit(X, sev)
    freq_model = ExtraTreesRegressor(n_estimators=600, random_state=0, n_jobs=-1).fit(X[has], freq[has])

    joblib.dump({"model": sev_model, "feature_names": names, "class_names": CLASS_NAMES,
                 "kind": "severity", "trained_fps": args.fps}, OUT / "severity.joblib")
    joblib.dump({"model": freq_model, "feature_names": names, "kind": "frequency",
                 "trained_fps": args.fps}, OUT / "frequency.joblib")
    meta = {"trained_on": "synthetic decoupled twin", "n_clips": int(len(X)),
            "poses": args.poses, "frames": args.frames, "fps": args.fps,
            "note": "Apply to real MediaPipe landmark trajectories via predict_on_video.py"}
    (OUT / "models_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved severity.joblib + frequency.joblib to {OUT}")


if __name__ == "__main__":
    main()
