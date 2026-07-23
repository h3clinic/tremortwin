"""Generate the paper figures from the actual twin, models, and real datasets.
Saves vector PDFs into paper/figs/. Run: python make_figures.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

HERE = Path(__file__).resolve().parent
FIG = HERE / "paper" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

from twin import (HandRig, ObsNoise, LANDMARK_ORDER, FINGER_CHAINS, WRIST,
                  random_static_pose, render_clip, tremor_params_coupled)
from generator import generate
from features import extract, _peak_freq, _highpass

IDX = {n: i for i, n in enumerate(LANDMARK_ORDER)}


def fig_twin():
    """Panel A: projected 21-landmark hand skeleton. Panel B: fingertip tremor trace."""
    rng = np.random.default_rng(3)
    rig = HandRig()
    pose = random_static_pose(rig, rng)
    amp, freq, _ = tremor_params_coupled(55)
    clean = ObsNoise(jitter_std=0, global_jitter_std=0, dropout_prob=0)
    traj = render_clip(rig, pose, amp, freq, 120, 60, "equatorial_front", rng, clean)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    # Skeleton drawn in the hand's widest anatomical plane (finger spread z vs
    # length y), from the rest 3D joints, so the hand is recognizable.
    lm3d = rig.landmarks(pose)
    for finger, chain in FINGER_CHAINS.items():
        pts = np.array([lm3d[0]] + [lm3d[IDX[b]] for b in chain])
        ax[0].plot(-pts[:, 2], pts[:, 1], "-o", ms=3, lw=1.2, color="0.2")
    ax[0].plot(-lm3d[0, 2], lm3d[0, 1], "s", ms=6, color="crimson")
    ax[0].annotate("wrist", (-lm3d[0, 2], lm3d[0, 1]), textcoords="offset points",
                   xytext=(6, -2), fontsize=7, color="crimson")
    ax[0].set_title("(a) Twin hand skeleton (21 joints)")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("spread axis"); ax[0].set_ylabel("length axis")

    tip = traj[:, IDX["index_dist"], :]
    t = np.arange(traj.shape[0]) / 60.0
    ax[1].plot(t, (tip[:, 0] - tip[:, 0].mean()) * 1e3, lw=1.0, color="0.15")
    ax[1].set_title(f"(b) Index-tip oscillation  (f={freq:.1f} Hz)")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("x displ. (norm.$\\times10^{-3}$)")
    fig.tight_layout(); fig.savefig(FIG / "fig_twin.pdf"); plt.close(fig)
    print("wrote fig_twin.pdf")


def fig_collinearity():
    """Frequency vs amplitude for coupled (a line) vs decoupled (a cloud)."""
    def collect(mode):
        f, a = [], []
        for c in generate(mode, 400, 8, 60, 12, seed=11,
                          views=["equatorial_front"], p_no_tremor=0.0):
            f.append(c.freq_hz); a.append(c.amp_rad)
        return np.array(f), np.array(a)
    cf, ca = collect("coupled")
    df, da = collect("decoupled")
    rc = pearsonr(cf, ca)[0]
    rd = pearsonr(df, da)[0]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    ax[0].scatter(cf, ca * 1e3, s=8, color="crimson", alpha=0.6)
    ax[0].set_title(f"(a) Coupled labels  (r={rc:.2f})")
    ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("amplitude (mrad)")
    ax[1].scatter(df, da * 1e3, s=8, color="0.25", alpha=0.5)
    ax[1].set_title(f"(b) Decoupled labels  (r={rd:.2f})")
    ax[1].set_xlabel("frequency (Hz)")
    fig.tight_layout(); fig.savefig(FIG / "fig_collinearity.pdf"); plt.close(fig)
    print(f"wrote fig_collinearity.pdf  (r_coupled={rc:.2f}, r_decoupled={rd:.2f})")


def fig_freq_scatter():
    """Predicted vs true frequency on held-out synthetic clips (trained model)."""
    import joblib
    b = joblib.load(HERE / "results" / "models" / "frequency.joblib")
    names, fps = b["feature_names"], b["trained_fps"]
    X, y = [], []
    for c in generate("decoupled", 140, 180, fps, 14, seed=20240, p_no_tremor=0.0):
        fe = extract(c.traj, fps)
        X.append([fe[k] for k in names]); y.append(c.freq_hz)
    X = np.nan_to_num(np.array(X)); y = np.array(y)
    pred = b["model"].predict(X)
    within1 = np.mean(np.abs(pred - y) <= 1.0)
    mae = np.mean(np.abs(pred - y))

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    lim = [2, 13]
    ax.fill_between(lim, [lim[0] - 1, lim[1] - 1], [lim[0] + 1, lim[1] + 1],
                    color="0.85", label="$\\pm$1 Hz")
    ax.plot(lim, lim, "--", color="0.4", lw=1)
    ax.scatter(y, pred, s=7, color="crimson", alpha=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("true frequency (Hz)"); ax.set_ylabel("predicted frequency (Hz)")
    ax.set_title(f"Held-out subjects\nwithin-1Hz={within1*100:.1f}%, MAE={mae:.2f} Hz")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig_freq.pdf"); plt.close(fig)
    print(f"wrote fig_freq.pdf  (within1={within1:.3f}, mae={mae:.3f})")


def fig_realdata():
    """(a) UCI-395 recovered PD-rest freq raw vs high-pass. (b) PADS freq by group."""
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ok = False

    # UCI-395
    try:
        import zipfile
        from realdata_uci395 import parse_stcp, analyze
        z = zipfile.ZipFile(HERE / "realdata" / "uci395.zip")
        raw, hp = [], []
        for n in z.namelist():
            if "/parkinson/" in n and n.endswith(".txt"):
                seg = parse_stcp(z.read(n).decode("utf-8", "replace"))
                if seg:
                    d = analyze(*seg)
                    if d:
                        raw.append(d["peak"]); hp.append(d["peak_hp"])
        bins = np.arange(1, 12, 0.6)
        ax[0].hist(raw, bins=bins, alpha=0.6, color="0.6", label="raw peak")
        ax[0].hist(hp, bins=bins, alpha=0.7, color="crimson", label="+high-pass")
        ax[0].axvspan(3.5, 7.5, color="green", alpha=0.08)
        ax[0].set_title("(a) UCI-395 PD rest tremor")
        ax[0].set_xlabel("recovered frequency (Hz)"); ax[0].set_ylabel("patients")
        ax[0].legend()
        ok = True
    except Exception as e:
        ax[0].text(0.5, 0.5, f"UCI-395 unavailable", ha="center")
        print("UCI fig skipped:", repr(e))

    # PADS
    try:
        from pads_validation import conditions, descriptor
        cond = conditions()
        groups = {"Healthy": [], "Parkinson's": [], "Essential Tremor": []}
        for pid, c in cond.items():
            if c in groups:
                d = descriptor(pid, "Relaxed")
                if d:
                    groups[c].append(d["peak"])
        data = [groups["Healthy"], groups["Parkinson's"], groups["Essential Tremor"]]
        bp = ax[1].boxplot(data, labels=["HC", "PD", "ET"], showfliers=False,
                           patch_artist=True)
        for patch, col in zip(bp["boxes"], ["0.7", "crimson", "steelblue"]):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        ax[1].set_title("(b) PADS recovered frequency")
        ax[1].set_ylabel("frequency (Hz)"); ax[1].set_ylim(1, 12)
        ok = True
    except Exception as e:
        ax[1].text(0.5, 0.5, "PADS unavailable", ha="center")
        print("PADS fig skipped:", repr(e))

    if ok:
        fig.tight_layout(); fig.savefig(FIG / "fig_real.pdf")
        print("wrote fig_real.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_twin()
    fig_collinearity()
    fig_freq_scatter()
    fig_realdata()
    print("figures in", FIG)
