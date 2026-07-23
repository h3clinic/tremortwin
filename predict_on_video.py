"""Sim-to-real bridge: apply the synthetic-trained models to REAL hand videos.

Runs MediaPipe Hands on each video -> (T,21,2) landmark trajectory (MediaPipe's
native 21-landmark order, which matches this project's LANDMARK_ORDER: wrist=0,
fingertips at 4/8/12/16/20) -> the SAME features.extract() used in training ->
predicts tremor frequency (Hz) and severity class.

If a labels CSV is provided (columns: video, freq_hz [optional], severity
[optional; one of Absent/Slight/Mild/Moderate/Marked/Severe]) it also reports
sim-to-real metrics: frequency MAE / within-1 Hz, severity accuracy / +/-1-bin.

Usage:
  python predict_on_video.py --videos /path/to/videos --labels labels.csv \
      --models-dir results/models --out results/realvideo_predictions.csv

This is the honest sim-to-real test: model trained ONLY on the synthetic twin,
evaluated on real patients. Frequency is the clean cross-dataset comparison
(objective, matches accelerometer ground truth); severity needs scale
calibration to the dataset's rating (TETRAS/UPDRS) before hard comparison.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from features import extract
from twin import CLASS_NAMES, CLASS_IDX


def landmarks_from_video(path, model_path, max_frames=1800, min_detect=0.5):
    """Return (T,21,2) normalized landmark trajectory + detection rate, or None.

    Uses the MediaPipe Tasks API HandLandmarker (VIDEO mode) — mediapipe 0.10.x
    removed the legacy solutions.hands API. Landmark order is MediaPipe-native
    (wrist=0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20), which
    matches this project's LANDMARK_ORDER so features.extract() applies directly.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base = python.BaseOptions(model_asset_path=str(model_path))
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=1,
                                        running_mode=vision.RunningMode.VIDEO)
    lm = vision.HandLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, detected, last, n = [], 0, None, 0
    while n < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lm.detect_for_video(mp_img, int(n * 1000.0 / fps))
        n += 1
        if res.hand_landmarks:
            pts = np.array([[p.x, p.y] for p in res.hand_landmarks[0]], float)  # 21x2
            last = pts
            detected += 1
        elif last is not None:
            pts = last                      # carry-forward on a dropped frame
        else:
            pts = None
        if pts is not None:
            frames.append(pts)
    cap.release(); lm.close()
    if len(frames) < 16:
        return None, fps, 0.0
    rate = detected / max(1, n)
    if rate < min_detect:
        return None, fps, rate
    return np.stack(frames, 0), fps, rate


def load_labels(path):
    if not path:
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = Path(row.get("video", "")).name
            out[key] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--models-dir", default="results/models")
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent / "hand_landmarker.task"),
                    help="MediaPipe hand_landmarker.task path")
    ap.add_argument("--out", default="results/realvideo_predictions.csv")
    ap.add_argument("--min-detect", type=float, default=0.5)
    args = ap.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"missing MediaPipe model: {args.model}\n"
                         "download: https://storage.googleapis.com/mediapipe-models/"
                         "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

    import joblib
    sev_b = joblib.load(Path(args.models_dir) / "severity.joblib")
    frq_b = joblib.load(Path(args.models_dir) / "frequency.joblib")
    names = sev_b["feature_names"]

    labels = load_labels(args.labels)
    vids = sorted([p for p in Path(args.videos).rglob("*")
                   if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm")])
    if not vids:
        raise SystemExit(f"no videos found under {args.videos}")

    rows, freq_err, freq_hit = [], [], []
    sev_true, sev_pred = [], []
    for vp in vids:
        traj, fps, rate = landmarks_from_video(vp, args.model, min_detect=args.min_detect)
        if traj is None:
            print(f"  SKIP {vp.name}: hand not reliably detected (rate={rate:.2f})")
            rows.append({"video": vp.name, "detect_rate": round(rate, 3), "status": "undetected"})
            continue
        feat = extract(traj, int(round(fps)))
        x = np.nan_to_num(np.array([[feat[k] for k in names]], float))
        f_hz = float(frq_b["model"].predict(x)[0])
        s_idx = int(sev_b["model"].predict(x)[0])
        s_name = CLASS_NAMES[s_idx]
        rec = {"video": vp.name, "detect_rate": round(rate, 3), "fps": round(fps, 1),
               "pred_freq_hz": round(f_hz, 2), "pred_severity": s_name, "status": "ok"}
        lab = labels.get(vp.name, {})
        if lab.get("freq_hz"):
            gt = float(lab["freq_hz"]); e = abs(gt - f_hz)
            rec["gt_freq_hz"] = gt; rec["freq_abs_err"] = round(e, 2)
            freq_err.append(e); freq_hit.append(e <= 1.0)
        if lab.get("severity") in CLASS_IDX:
            rec["gt_severity"] = lab["severity"]
            sev_true.append(CLASS_IDX[lab["severity"]]); sev_pred.append(s_idx)
        rows.append(rec)
        print(f"  {vp.name}: freq={f_hz:.2f}Hz severity={s_name} (detect {rate:.2f})")

    # write predictions
    keys = ["video", "status", "detect_rate", "fps", "pred_freq_hz", "gt_freq_hz",
            "freq_abs_err", "pred_severity", "gt_severity"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out}")

    # sim-to-real metrics
    if freq_err:
        fe = np.array(freq_err)
        print(f"\nFREQUENCY sim->real: n={len(fe)}  MAE={fe.mean():.3f} Hz  "
              f"within-1Hz={np.mean(freq_hit):.3f}  within-0.5Hz={np.mean(fe<=0.5):.3f}")
    if sev_true:
        st, sp = np.array(sev_true), np.array(sev_pred)
        print(f"SEVERITY sim->real: n={len(st)}  exact={np.mean(st==sp):.3f}  "
              f"±1-bin={np.mean(np.abs(st-sp)<=1):.3f}  "
              f"(note: calibrate our 6-bin scale to the dataset's rating first)")
    if not freq_err and not sev_true:
        print("No ground-truth labels supplied — predictions only. Provide --labels "
              "with freq_hz and/or severity columns to compute sim-to-real accuracy.")


if __name__ == "__main__":
    main()
