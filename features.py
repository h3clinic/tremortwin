"""Blind feature extraction from 2D landmark trajectories.

"Blind" = the only input is the 21x T x 2 MediaPipe-style landmark stream. The
extractor knows nothing about the injected frequency/amplitude; it must recover
tremor descriptors from the motion alone, exactly as a real tracker-fed model
would. Features are spectral (Welch PSD: peak frequency, band powers, spectral
entropy/centroid) and temporal (size-normalized displacement, jerk, zero
crossings, autocorrelation period).

Everything amplitude-related is normalized by an in-clip hand scale
(wrist->middle-base distance) so the descriptors are invariant to camera
distance / image resolution -- the same scale-normalization the render-side
`normalize_signals.py` performs.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from twin import LANDMARK_ORDER

# numpy>=2.0 renamed trapz -> trapezoid.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Drift/translation removal. Real MediaPipe trajectories carry low-frequency
# postural sway + whole-hand translation that swamps the tremor peak (validated
# on UCI-395: raw peak ~2 Hz drift vs 4 Hz true tremor). A gentle high-pass makes
# the estimator robust on real video; on the drift-free synthetic twin it barely
# changes anything (tremor is >=4 Hz, cutoff 2.5 Hz). Applied consistently in
# train + inference so synthetic and real go through the identical transform.
_HP_CUT = 2.5

def _highpass(sig, fs, cut=_HP_CUT):
    sig = np.asarray(sig, float)
    if len(sig) < 30 or fs < 2.2 * cut:
        return sig - sig.mean()
    from scipy.signal import butter, filtfilt
    b, a = butter(2, cut, btype="highpass", fs=fs)
    try:
        return filtfilt(b, a, sig)
    except ValueError:
        return sig - sig.mean()

WRIST_I = LANDMARK_ORDER.index("radius_ulna")
MID_BASE_I = LANDMARK_ORDER.index("midd_meta")
FINGERTIPS = [LANDMARK_ORDER.index(n) for n in
              ("thumb_dist", "index_dist", "midd_dist", "ring_dist", "pinky_dist")]
BANDS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 13)]


def _hand_scale(traj: np.ndarray) -> float:
    """Mean wrist->middle-metacarpal distance over the clip (normalized units)."""
    d = np.linalg.norm(traj[:, WRIST_I, :] - traj[:, MID_BASE_I, :], axis=1)
    return float(np.median(d) + 1e-6)


def _peak_freq(sig: np.ndarray, fps: int):
    """Welch PSD dominant frequency with parabolic sub-bin interpolation, plus
    spectral descriptors. Returns dict."""
    sig = sig - sig.mean()
    n = len(sig)
    if np.allclose(sig, 0) or n < 8:
        return dict(peak=0.0, centroid=0.0, entropy=0.0, total=0.0, bandfrac=[0.0] * len(BANDS),
                    peak_prom=0.0, hi_lo=0.0)
    nper = min(n, 64)
    f, p = welch(sig, fs=fps, nperseg=nper, nfft=max(256, nper), detrend="constant")
    p = p + 1e-12
    # Restrict to plausible tremor band for peak picking (1-14 Hz).
    band = (f >= 1.0) & (f <= 14.0)
    fb, pb = f[band], p[band]
    if len(pb) == 0:
        return dict(peak=0.0, centroid=0.0, entropy=0.0, total=0.0, bandfrac=[0.0] * len(BANDS),
                    peak_prom=0.0, hi_lo=0.0)
    ki = int(np.argmax(pb))
    peak = float(fb[ki])
    # Parabolic interpolation around the peak for sub-bin resolution.
    if 0 < ki < len(pb) - 1:
        a, b, c = np.log(pb[ki - 1]), np.log(pb[ki]), np.log(pb[ki + 1])
        denom = (a - 2 * b + c)
        if abs(denom) > 1e-9:
            shift = 0.5 * (a - c) / denom
            peak = float(fb[ki] + shift * (fb[1] - fb[0]))
    total = float(_trapz(pb, fb))
    centroid = float(np.sum(fb * pb) / np.sum(pb))
    pn = pb / np.sum(pb)
    entropy = float(-np.sum(pn * np.log(pn)) / np.log(len(pn)))
    bandfrac = []
    for lo, hi in BANDS:
        m = (fb >= lo) & (fb < hi)
        bandfrac.append(float(_trapz(pb[m], fb[m]) / (total + 1e-12)) if m.any() else 0.0)
    peak_prom = float(pb[ki] / (np.median(pb) + 1e-12))
    lo_pow = _trapz(pb[fb < 6], fb[fb < 6])
    hi_pow = _trapz(pb[fb >= 6], fb[fb >= 6])
    hi_lo = float(hi_pow / (lo_pow + 1e-12))
    return dict(peak=peak, centroid=centroid, entropy=entropy, total=total,
                bandfrac=bandfrac, peak_prom=peak_prom, hi_lo=hi_lo)


def _temporal(disp: np.ndarray, fps: int):
    """Temporal descriptors of a 1D displacement-magnitude signal."""
    d = disp - disp.mean()
    rms = float(np.sqrt(np.mean(d ** 2)))
    mx = float(np.max(np.abs(d)))
    jerk = float(np.sqrt(np.mean(np.diff(d, 2) ** 2))) if len(d) > 2 else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(d))) > 0))
    # Autocorrelation first non-zero-lag peak -> period -> frequency.
    ac = np.correlate(d, d, mode="full")[len(d) - 1:]
    ac = ac / (ac[0] + 1e-12)
    period_f = 0.0
    if len(ac) > 3:
        lo = 1
        while lo < len(ac) - 1 and ac[lo] > ac[lo + 1]:
            lo += 1
        seg = ac[lo:]
        if len(seg) > 1:
            lag = lo + int(np.argmax(seg))
            if lag > 0:
                period_f = fps / lag
    return dict(rms=rms, max=mx, jerk=jerk, zcr=zcr, acf_freq=period_f)


def extract(traj: np.ndarray, fps: int) -> dict:
    """traj (T,21,2) -> flat feature dict. Pure function of the landmark stream."""
    scale = _hand_scale(traj)
    per_lm = []          # spectral+temporal per landmark
    peak_list = []
    wrist = traj[:, WRIST_I, :]
    for li in range(traj.shape[1]):
        p = traj[:, li, :]
        # Lever arm (distance from wrist pivot) is a SPATIAL quantity -> raw coords.
        lever = float(np.linalg.norm(p - wrist, axis=1).mean()) + 1e-9
        # High-pass each axis to strip postural drift / whole-hand translation, so
        # every downstream feature reflects tremor, not slow motion (real-data fix).
        fx = _highpass(p[:, 0], fps)
        fy = _highpass(p[:, 1], fps)
        raw = np.hypot(fx, fy)                       # tremor displacement magnitude
        disp = raw / scale                            # size-normalized
        # Pose-invariant ANGULAR amplitude: tremor displacement / lever arm -> the
        # injected wrist rotation amplitude, independent of finger pose/view/scale.
        ang_amp = float(np.sqrt(np.mean((raw - raw.mean()) ** 2)) / lever)
        sx = _peak_freq(fx / scale, fps)
        sy = _peak_freq(fy / scale, fps)
        sd = _peak_freq(disp, fps)
        t = _temporal(disp, fps)
        peak_list.append(sd["peak"])
        per_lm.append(dict(
            disp_rms=t["rms"], disp_max=t["max"], jerk=t["jerk"], zcr=t["zcr"],
            acf_freq=t["acf_freq"], ang_amp=ang_amp, peak=sd["peak"], centroid=sd["centroid"],
            entropy=sd["entropy"], total=sd["total"], peak_prom=sd["peak_prom"],
            hi_lo=sd["hi_lo"],
            **{f"band{j}": sd["bandfrac"][j] for j in range(len(BANDS))},
            peakx=sx["peak"], peaky=sy["peak"],
        ))

    keys = list(per_lm[0].keys())
    arr = {k: np.array([d[k] for d in per_lm]) for k in keys}
    feats = {}
    # Aggregate each per-landmark feature across the 21 landmarks.
    for k in keys:
        feats[f"{k}__mean"] = float(np.mean(arr[k]))
        feats[f"{k}__max"] = float(np.max(arr[k]))
        feats[f"{k}__std"] = float(np.std(arr[k]))
    # Fingertip-focused aggregates (fingertips carry the largest lever arm).
    tip = [per_lm[i] for i in FINGERTIPS]
    for k in ("disp_rms", "disp_max", "peak", "total", "jerk", "ang_amp"):
        feats[f"tip_{k}__mean"] = float(np.mean([d[k] for d in tip]))
        feats[f"tip_{k}__max"] = float(np.max([d[k] for d in tip]))
    # Moving-landmark mask (which landmarks actually carry tremor signal).
    rms_arr = np.array([per_lm[i]["disp_rms"] for i in range(len(per_lm))])
    mv = rms_arr > 1e-4
    # Cross-landmark frequency agreement: how consistent is the dominant peak?
    peaks = np.array(peak_list)
    moving = peaks[mv]
    ref = np.median(moving) if len(moving) else 0.0
    feats["peak_median"] = float(ref)
    feats["peak_iqr"] = float(np.subtract(*np.percentile(moving, [75, 25]))) if len(moving) else 0.0
    feats["peak_agree"] = float(np.mean(np.abs(moving - ref) < 0.75)) if len(moving) else 0.0
    # Fair signal-processing-only frequency estimate: per-AXIS PSD peak (at f, not
    # the rectified 2f of displacement magnitude), median over moving landmarks.
    axis_peak = np.array([(per_lm[i]["peakx"] + per_lm[i]["peaky"]) / 2 for i in range(len(per_lm))])
    feats["peak_axis_median"] = float(np.median(axis_peak[mv])) if mv.any() else 0.0
    # Robust pose-invariant angular-amplitude summaries (severity signal).
    ang = np.array([per_lm[i]["ang_amp"] for i in range(len(per_lm))])
    feats["ang_amp_median"] = float(np.median(ang[mv])) if mv.any() else 0.0
    feats["ang_amp_p90"] = float(np.percentile(ang[mv], 90)) if mv.any() else 0.0
    feats["n_moving"] = float(len(moving))
    feats["hand_scale"] = float(scale)
    return feats


# Feature groups for the self-improvement loop's feature-escalation rounds.
def feature_groups(all_keys):
    temporal = [k for k in all_keys if any(s in k for s in
                ("disp_rms", "disp_max", "jerk", "zcr", "acf_freq", "tip_disp", "tip_jerk",
                 "n_moving", "ang_amp"))]
    spectral = [k for k in all_keys if any(s in k for s in
                ("peak", "centroid", "entropy", "total", "band", "hi_lo", "prom", "tip_peak", "tip_total"))]
    agreement = [k for k in all_keys if any(s in k for s in
                 ("peak_median", "peak_iqr", "peak_agree"))]
    return {"temporal": temporal, "spectral": spectral, "agreement": agreement}
