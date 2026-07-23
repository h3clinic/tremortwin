"""Orchestrate the full experiment for one generator mode.

Pipeline: generate twin clips -> blind features -> subject-held-out split ->
self-improvement loop for (a) severity classification and (b) frequency
regression -> leakage audit -> write results JSON + console summary.

Run:
  python run.py --mode decoupled --poses 360 --subjects 24
  python run.py --mode coupled   --poses 360 --subjects 24
  python run.py --quick           # tiny smoke run of both modes
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import leakage
from features import extract, feature_groups
from generator import generate
from selfimprove import (clf_metrics, reg_metrics, run_self_improvement)
from twin import CLASS_NAMES

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def build_matrix(mode, poses, frames, fps, subjects, seed, p_no_tremor):
    rows, meta = [], []
    t0 = time.time()
    for i, clip in enumerate(generate(mode, poses, frames, fps, subjects, seed,
                                      p_no_tremor=p_no_tremor)):
        rows.append(extract(clip.traj, fps))
        meta.append((clip.pose_id, clip.subject_id, clip.view, clip.freq_hz,
                     clip.amp_rad, clip.severity_idx, clip.has_tremor, clip.tremor_type))
        if (i + 1) % 200 == 0:
            print(f"    extracted {i+1} clips ({time.time()-t0:.0f}s)")
    names = sorted(rows[0].keys())
    X = np.array([[r[k] for k in names] for r in rows], float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    pose_id = np.array([m[0] for m in meta])
    subject_id = np.array([m[1] for m in meta])
    freq = np.array([m[3] for m in meta], float)
    amp = np.array([m[4] for m in meta], float)
    sev = np.array([m[5] for m in meta], int)
    has_tremor = np.array([m[6] for m in meta], bool)
    print(f"    feature matrix: {X.shape}, {time.time()-t0:.0f}s total")
    return dict(X=X, names=names, pose_id=pose_id, subject_id=subject_id,
                freq=freq, amp=amp, sev=sev, has_tremor=has_tremor)


def subject_holdout(subject_id, test_frac, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    idx = np.arange(len(subject_id))
    tr, te = next(gss.split(idx, groups=subject_id))
    return tr, te


def run_mode(mode, poses, frames, fps, subjects, seed, test_frac, targets, p_no_tremor):
    print(f"\n=== MODE: {mode} ===")
    data = build_matrix(mode, poses, frames, fps, subjects, seed, p_no_tremor)
    X, names = data["X"], data["names"]
    fg = feature_groups(names)
    result = {"mode": mode, "n_rows": int(len(X)), "n_features": len(names),
              "config": dict(poses=poses, frames=frames, fps=fps, subjects=subjects,
                             seed=seed, test_frac=test_frac, p_no_tremor=p_no_tremor)}

    # ---- Leakage check 0: are the two targets collinear by construction? ----
    result["label_collinearity"] = leakage.label_collinearity(
        data["freq"], data["sev"], data["amp"])
    print("  label collinearity r(freq,severity) = "
          f"{result['label_collinearity'].get('pearson_freq_severity'):.3f} "
          f"-> {result['label_collinearity']['verdict']}")

    # ---- Severity classification (all clips) ----
    print("\n  [SEVERITY CLASSIFICATION] subject-held-out")
    tr, te = subject_holdout(data["subject_id"], test_frac, seed)
    gd = leakage.group_disjoint(data["subject_id"][tr], data["subject_id"][te])
    si = run_self_improvement(
        X[tr], data["sev"][tr], data["subject_id"][tr], names, fg,
        task="classification", primary="accuracy", target=targets["severity"], seed=seed)
    pred = si["model"].predict(X[te][:, si["keep"]])
    sev_test = clf_metrics(data["sev"][te], pred)
    # Per-class support on test
    classes, counts = np.unique(data["sev"][te], return_counts=True)
    shuffle_null = leakage.label_shuffle_null(X[tr][:, si["keep"]], data["sev"][tr],
                                              data["subject_id"][tr], "classification", seed)
    sep = leakage.train_test_separability(X[tr][:, si["keep"]], X[te][:, si["keep"]], seed)
    perm = leakage.permutation_top(si["model"], X[te][:, si["keep"]], data["sev"][te],
                                   [names[i] for i in si["keep"]], k=10)
    result["severity"] = {
        "group_disjoint": gd, "best": si["best"], "history": si["history"],
        "test_metrics": sev_test,
        "test_class_support": {CLASS_NAMES[c]: int(n) for c, n in zip(classes, counts)},
        "leakage": {"label_shuffle_null": shuffle_null, "train_test_separability": sep,
                    "top_features": perm},
    }
    print(f"    TEST: exact_acc={sev_test['accuracy']:.3f}  adjacent_acc={sev_test['adjacent_acc']:.3f}  "
          f"macro_f1={sev_test['macro_f1']:.3f}")
    print(f"    shuffle-null acc={shuffle_null['shuffled_score']:.3f} "
          f"(chance~{shuffle_null['majority_chance']:.3f}) -> {shuffle_null['verdict']}")

    # ---- Frequency regression (tremor-present clips only) ----
    print("\n  [FREQUENCY REGRESSION] subject-held-out, tremor-present only")
    mask = data["has_tremor"]
    Xf, freqf, subjf = X[mask], data["freq"][mask], data["subject_id"][mask]
    namef = names
    trf, tef = subject_holdout(subjf, test_frac, seed)
    sif = run_self_improvement(
        Xf[trf], freqf[trf], subjf[trf], namef, fg,
        task="regression", primary="within_1hz", target=targets["frequency"], seed=seed)
    predf = sif["model"].predict(Xf[tef][:, sif["keep"]])
    freq_test = reg_metrics(freqf[tef], predf)
    # Fair signal-processing-only baseline: per-axis PSD peak (at f, not the
    # rectified 2f of displacement magnitude), median across moving landmarks,
    # used directly as the frequency estimate (NO learning) -> shows what ML adds.
    pm_idx = names.index("peak_axis_median")
    naive = reg_metrics(freqf[tef], Xf[tef][:, pm_idx])
    result["frequency"] = {
        "best": sif["best"], "history": sif["history"], "test_metrics": freq_test,
        "naive_fft_baseline": naive,
    }
    print(f"    TEST ML : MAE={freq_test['mae_hz']:.3f}Hz  within1Hz={freq_test['within_1hz']:.3f}  "
          f"within0.5Hz={freq_test['within_0p5hz']:.3f}")
    print(f"    TEST FFT-only baseline (no ML): MAE={naive['mae_hz']:.3f}Hz  "
          f"within1Hz={naive['within_1hz']:.3f}")
    print(f"    -> ML uplift over plain FFT peak: "
          f"{freq_test['within_1hz']-naive['within_1hz']:+.3f} within-1Hz")

    (OUT / f"{mode}_results.json").write_text(json.dumps(result, indent=2))
    print(f"  saved -> results/{mode}_results.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["coupled", "decoupled", "both"], default="both")
    ap.add_argument("--poses", type=int, default=360)
    ap.add_argument("--frames", type=int, default=120)   # 2.0 s @ 60 fps
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--subjects", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.22)
    ap.add_argument("--p-no-tremor", type=float, default=0.12)
    ap.add_argument("--target-severity", type=float, default=0.95)
    ap.add_argument("--target-frequency", type=float, default=0.95)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.poses, args.subjects, args.frames = 60, 8, 90
    targets = {"severity": args.target_severity, "frequency": args.target_frequency}
    modes = ["coupled", "decoupled"] if args.mode == "both" else [args.mode]
    summary = {}
    for m in modes:
        r = run_mode(m, args.poses, args.frames, args.fps, args.subjects, args.seed,
                     args.test_frac, targets, args.p_no_tremor)
        summary[m] = {
            "severity_exact": r["severity"]["test_metrics"]["accuracy"],
            "severity_adjacent": r["severity"]["test_metrics"]["adjacent_acc"],
            "freq_within1hz": r["frequency"]["test_metrics"]["within_1hz"],
            "freq_mae": r["frequency"]["test_metrics"]["mae_hz"],
            "freq_naive_within1hz": r["frequency"]["naive_fft_baseline"]["within_1hz"],
            "targets_collinear": abs(r["label_collinearity"].get("pearson_freq_severity", 0)) > 0.9,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n================ SUMMARY ================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
