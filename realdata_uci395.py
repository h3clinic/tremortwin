"""Real-tremor validation of our spectral frequency core on UCI-395.

UCI-395 (Parkinson spiral/stability tablet data) TestID=2 is the "Stability Test
on a Certain Point": the subject holds the pen on a fixed point, so the X/Y
wobble is real hand tremor as a 2-D position time series -- the SAME modality a
MediaPipe landmark produces. There is no frequency ground truth (no accelerometer),
so we test two honest, checkable things:

  1. Do the PD subjects' recovered dominant frequencies land in the physiological
     rest/postural tremor band (~4-6 Hz)?  (face validity of the estimator)
  2. Does a tremor descriptor computed by OUR feature core separate PD from
     controls?  (ROC-AUC)  -- i.e. does the synthetic-tuned method detect REAL tremor?

Scope: this validates the spectral core (features._peak_freq), NOT the full
21-landmark model (a pen gives one point, not 21). Full-model validation needs
real hand VIDEO (MediaPipe -> 21 landmarks), which is what TIM-Tremor would have
provided.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from features import _peak_freq   # reuse the exact estimator used by the model


def highpass(sig, fs, cut=3.0):
    """Remove postural drift/sway (<cut Hz) so peak-picking sees the tremor, not
    the slow hold drift. This is the real-data preprocessing the clean synthetic
    signals never needed -- a concrete sim-to-real adjustment."""
    if len(sig) < 30:
        return sig
    b, a = butter(4, cut, btype="highpass", fs=fs)
    return filtfilt(b, a, sig)

ZIP = Path(__file__).resolve().parent / "realdata" / "uci395.zip"
STCP = "2"   # Stability Test on Certain Point


def parse_stcp(raw: str):
    """Return (t_seconds, X, Y) for the TestID=2 segment, or None."""
    xs, ys, ts = [], [], []
    for line in raw.splitlines():
        if not line.strip():
            continue
        c = line.split(";")
        if len(c) < 7 or c[6].strip() != STCP:
            continue
        try:
            xs.append(float(c[0])); ys.append(float(c[1])); ts.append(float(c[5]))
        except ValueError:
            continue
    if len(xs) < 64:
        return None
    t = np.array(ts); x = np.array(xs); y = np.array(ys)
    t = (t - t[0]) / 1000.0                      # timestamps are ms -> seconds
    return t, x, y


def analyze(t, x, y):
    """Resample to uniform fs, run the spectral core on X and Y, return descriptors."""
    dt = np.median(np.diff(t))
    if dt <= 0:
        return None
    fs = 1.0 / dt
    if not (30 <= fs <= 400):                    # sanity on tablet rate
        fs = float(np.clip(fs, 30, 400))
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    xu = np.interp(grid, t, x); yu = np.interp(grid, t, y)
    # Raw peak (drift-contaminated) vs high-passed peak (drift removed).
    sx, sy = _peak_freq(xu, fs), _peak_freq(yu, fs)
    hx, hy = _peak_freq(highpass(xu, fs), fs), _peak_freq(highpass(yu, fs), fs)
    dom = sx if sx["peak_prom"] >= sy["peak_prom"] else sy
    dom_hp = hx if hx["peak_prom"] >= hy["peak_prom"] else hy
    def band_frac(s):
        return sum(s["bandfrac"][1:4])           # ~4-10 Hz core of BANDS
    tb = 0.5 * (band_frac(sx) + band_frac(sy))
    return dict(fs=fs, n=len(grid), peak=dom["peak"], peak_hp=dom_hp["peak"],
                prom=dom["peak_prom"], entropy=dom["entropy"], tremor_band=tb,
                total=0.5 * (sx["total"] + sy["total"]))


def main():
    z = zipfile.ZipFile(ZIP)
    rows = []
    for name in z.namelist():
        if not name.endswith(".txt"):
            continue
        grp = "PD" if "/parkinson/" in name else ("HC" if "/control/" in name else None)
        if grp is None:
            continue
        seg = parse_stcp(z.read(name).decode("utf-8", "replace"))
        if seg is None:
            continue
        d = analyze(*seg)
        if d is None:
            continue
        d["group"] = grp; d["file"] = Path(name).name
        rows.append(d)

    pd_ = [r for r in rows if r["group"] == "PD"]
    hc = [r for r in rows if r["group"] == "HC"]
    print(f"Parsed STCP hold from {len(rows)} subjects: {len(pd_)} PD, {len(hc)} HC "
          f"(median fs ~{np.median([r['fs'] for r in rows]):.0f} Hz)\n")

    pk = np.array([r["peak"] for r in pd_])
    pkh = np.array([r["peak_hp"] for r in pd_])
    print("(1) FACE VALIDITY — PD recovered dominant tremor frequency:")
    print(f"    RAW peak (drift-contaminated): median={np.median(pk):.2f} Hz  "
          f"in 3.5-7.5 Hz = {np.mean((pk>=3.5)&(pk<=7.5)):.2f}")
    print(f"    HIGH-PASS >3Hz (drift removed): median={np.median(pkh):.2f} Hz  "
          f"IQR=[{np.percentile(pkh,25):.2f},{np.percentile(pkh,75):.2f}]  "
          f"in 3.5-7.5 Hz = {np.mean((pkh>=3.5)&(pkh<=7.5)):.2f}")

    print("\n(2) PD-vs-HC SEPARATION using OUR spectral features (ROC-AUC, PD=positive):")
    y = np.array([1] * len(pd_) + [0] * len(hc))
    for feat in ("tremor_band", "prom", "total"):
        s = np.array([r[feat] for r in pd_] + [r[feat] for r in hc])
        s = np.nan_to_num(s)
        auc = roc_auc_score(y, s)
        auc = max(auc, 1 - auc)                  # feature direction-agnostic
        try:
            p = mannwhitneyu([r[feat] for r in pd_], [r[feat] for r in hc]).pvalue
        except ValueError:
            p = float("nan")
        print(f"    {feat:12s} AUC={auc:.3f}  (Mann-Whitney p={p:.3g})")

    print("\nHonest scope: validates the spectral frequency/tremor core on REAL hand")
    print("tremor (2-D position). It does NOT test the full 21-landmark model or the")
    print("MediaPipe video path -- that needs real hand video (TIM-Tremor, now offline).")


if __name__ == "__main__":
    main()
