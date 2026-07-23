"""Emit a fully self-contained paper .tex in which every figure is native
pgfplots/TikZ, so the single file renders with no external image uploads.
Reuses the real model and data. Run: python make_pgf.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

from twin import (HandRig, ObsNoise, LANDMARK_ORDER, FINGER_CHAINS,
                  random_static_pose, render_clip, tremor_params_coupled)
from generator import generate
from features import extract, _peak_freq, _highpass
IDX = {n: i for i, n in enumerate(LANDMARK_ORDER)}


def coords(xs, ys, prec=3):
    return " ".join(f"({x:.{prec}f},{y:.{prec}f})" for x, y in zip(xs, ys))


# ---------------------------------------------------------------- twin
def twin_tikz():
    rng = np.random.default_rng(3)
    rig = HandRig()
    pose = random_static_pose(rig, rng)
    amp, freq, _ = tremor_params_coupled(55)
    clean = ObsNoise(0, 0, 0)
    traj = render_clip(rig, pose, amp, freq, 120, 60, "equatorial_front", rng, clean)
    lm = rig.landmarks(pose)

    chains = []
    for finger, chain in FINGER_CHAINS.items():
        pts = [lm[0]] + [lm[IDX[b]] for b in chain]
        pts = np.array(pts)
        chains.append(coords(-pts[:, 2], pts[:, 1]))
    wx, wy = -lm[0, 2], lm[0, 1]

    tip = traj[:, IDX["index_dist"], :]
    t = np.arange(traj.shape[0]) / 60.0
    osc = coords(t, (tip[:, 0] - tip[:, 0].mean()) * 1e3, prec=4)

    sk = "\n".join(
        f"\\addplot[gray!60,thick,mark=*,mark size=1pt] coordinates {{{c}}};"
        for c in chains)
    a = (f"\\begin{{tikzpicture}}\n\\begin{{axis}}[width=0.47\\textwidth,height=4.6cm,"
         f"axis equal image,xlabel={{spread axis}},ylabel={{length axis}},"
         f"title={{(a) Twin hand skeleton (21 joints)}},title style={{font=\\small}},"
         f"tick label style={{font=\\footnotesize}},label style={{font=\\footnotesize}}]\n"
         f"{sk}\n"
         f"\\addplot[only marks,mark=square*,red,mark size=2.5pt] coordinates {{({wx:.3f},{wy:.3f})}};\n"
         f"\\end{{axis}}\n\\end{{tikzpicture}}")
    b = (f"\\begin{{tikzpicture}}\n\\begin{{axis}}[width=0.5\\textwidth,height=4.6cm,"
         f"xlabel={{time (s)}},ylabel={{x displ. ($\\times10^{{-3}}$)}},"
         f"title={{(b) Index-tip oscillation ($f={freq:.1f}$ Hz)}},title style={{font=\\small}},"
         f"tick label style={{font=\\footnotesize}},label style={{font=\\footnotesize}}]\n"
         f"\\addplot[black,thick,no marks] coordinates {{{osc}}};\n"
         f"\\end{{axis}}\n\\end{{tikzpicture}}")
    return a + "\\hfill\n" + b


# ---------------------------------------------------------------- collinearity
def collinearity_tikz():
    def collect(mode, n):
        f, a = [], []
        for c in generate(mode, n, 8, 60, 12, seed=11,
                          views=["equatorial_front"], p_no_tremor=0.0):
            f.append(c.freq_hz); a.append(c.amp_rad * 1e3)
        return f, a
    cf, ca = collect("coupled", 150)
    df, da = collect("decoupled", 220)
    ax = lambda t, xmax: (f"width=0.47\\textwidth,height=4.6cm,xlabel={{frequency (Hz)}},"
                          f"ylabel={{amplitude (mrad)}},xmin=3,xmax={xmax},ymin=-5,ymax=160,"
                          f"title={{{t}}},title style={{font=\\small}},"
                          f"tick label style={{font=\\footnotesize}},label style={{font=\\footnotesize}}")
    a = (f"\\begin{{tikzpicture}}\n\\begin{{axis}}[{ax('(a) Coupled labels ($r=-1.00$)',10.5)}]\n"
         f"\\addplot[only marks,mark=*,mark size=0.9pt,red,opacity=0.6] coordinates {{{coords(cf,ca)}}};\n"
         f"\\end{{axis}}\n\\end{{tikzpicture}}")
    b = (f"\\begin{{tikzpicture}}\n\\begin{{axis}}[{ax('(b) Decoupled labels ($r=-0.23$)',12.5)}]\n"
         f"\\addplot[only marks,mark=*,mark size=0.9pt,black,opacity=0.45] coordinates {{{coords(df,da)}}};\n"
         f"\\end{{axis}}\n\\end{{tikzpicture}}")
    return a + "\\hfill\n" + b


# ---------------------------------------------------------------- freq scatter
def freq_tikz():
    import joblib
    b = joblib.load(HERE / "results" / "models" / "frequency.joblib")
    names, fps = b["feature_names"], b["trained_fps"]
    X, y = [], []
    for c in generate("decoupled", 90, 180, fps, 14, seed=20240, p_no_tremor=0.0):
        fe = extract(c.traj, fps)
        X.append([fe[k] for k in names]); y.append(c.freq_hz)
    X = np.nan_to_num(np.array(X)); y = np.array(y)
    pred = b["model"].predict(X)
    w1 = np.mean(np.abs(pred - y) <= 1) * 100
    mae = np.mean(np.abs(pred - y))
    return (f"\\begin{{tikzpicture}}\n\\begin{{axis}}[width=\\columnwidth,height=6.2cm,"
            f"axis equal image,xmin=2,xmax=13,ymin=2,ymax=13,xlabel={{true frequency (Hz)}},"
            f"ylabel={{predicted frequency (Hz)}},title={{held-out: within-1Hz$={w1:.1f}$\\%, "
            f"MAE$={mae:.2f}$ Hz}},title style={{font=\\small}},"
            f"tick label style={{font=\\footnotesize}},label style={{font=\\footnotesize}}]\n"
            f"\\addplot[gray!60,dotted] coordinates {{(2,1)(13,12)}};\n"
            f"\\addplot[gray!60,dotted] coordinates {{(2,3)(13,14)}};\n"
            f"\\addplot[gray,dashed] coordinates {{(2,2)(13,13)}};\n"
            f"\\addplot[only marks,mark=*,mark size=0.8pt,red,opacity=0.5] "
            f"coordinates {{{coords(y,pred)}}};\n"
            f"\\end{{axis}}\n\\end{{tikzpicture}}")


# ---------------------------------------------------------------- real data
def real_tikz():
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
    bins = np.arange(1, 12.1, 0.6)
    ctr = (bins[:-1] + bins[1:]) / 2
    hr, _ = np.histogram(raw, bins=bins)
    hh, _ = np.histogram(hp, bins=bins)
    ymax = int(max(hr.max(), hh.max()) + 3)
    a = (f"\\begin{{tikzpicture}}\n\\begin{{axis}}[width=0.5\\textwidth,height=4.8cm,"
         f"ybar,bar width=2.6pt,xmin=1,xmax=12,ymin=0,ymax={ymax},xlabel={{recovered frequency (Hz)}},"
         f"ylabel={{patients}},title={{(a) UCI-395 PD rest tremor}},title style={{font=\\small}},"
         f"legend style={{font=\\footnotesize,at={{(0.98,0.98)}},anchor=north east}},"
         f"tick label style={{font=\\footnotesize}},label style={{font=\\footnotesize}}]\n"
         f"\\draw[green!45,dashed] (axis cs:3.5,0)--(axis cs:3.5,{ymax});\n"
         f"\\draw[green!45,dashed] (axis cs:7.5,0)--(axis cs:7.5,{ymax});\n"
         f"\\addplot[fill=gray!55,draw=gray!55] coordinates {{{coords(ctr,hr)}}};\n"
         f"\\addplot[fill=red!60,draw=red!60,fill opacity=0.65] coordinates {{{coords(ctr,hh)}}};\n"
         f"\\legend{{raw peak,+high-pass}}\n"
         f"\\end{{axis}}\n\\end{{tikzpicture}}")

    from pads_validation import conditions, descriptor
    cond = conditions()
    stats = {}
    for label, name in [("HC", "Healthy"), ("PD", "Parkinson's"), ("ET", "Essential Tremor")]:
        vals = []
        for pid, c in cond.items():
            if c == name:
                d = descriptor(pid, "Relaxed")
                if d:
                    vals.append(d["peak"])
        v = np.array(vals)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        wlo = v[v >= q1 - 1.5 * iqr].min()
        whi = v[v <= q3 + 1.5 * iqr].max()
        stats[label] = (q1, med, q3, wlo, whi)

    box = ""
    for i, lab in enumerate(["HC", "PD", "ET"], start=1):
        q1, med, q3, wlo, whi = stats[lab]
        col = {"HC": "gray!40", "PD": "red!45", "ET": "blue!30"}[lab]
        box += (f"\\draw[thick,fill={col}] (axis cs:{i-0.22:.2f},{q1:.2f}) rectangle "
                f"(axis cs:{i+0.22:.2f},{q3:.2f});\n"
                f"\\draw[very thick,orange] (axis cs:{i-0.22:.2f},{med:.2f})--"
                f"(axis cs:{i+0.22:.2f},{med:.2f});\n"
                f"\\draw (axis cs:{i:.2f},{q3:.2f})--(axis cs:{i:.2f},{whi:.2f});\n"
                f"\\draw (axis cs:{i:.2f},{q1:.2f})--(axis cs:{i:.2f},{wlo:.2f});\n")
    b = (f"\\begin{{tikzpicture}}\n\\begin{{axis}}[width=0.47\\textwidth,height=4.8cm,"
         f"xmin=0.5,xmax=3.5,ymin=1,ymax=12,xtick={{1,2,3}},xticklabels={{HC,PD,ET}},"
         f"ylabel={{frequency (Hz)}},title={{(b) PADS recovered frequency}},title style={{font=\\small}},"
         f"tick label style={{font=\\footnotesize}},label style={{font=\\footnotesize}}]\n"
         f"{box}\\end{{axis}}\n\\end{{tikzpicture}}")
    return a + "\\hfill\n" + b


def main():
    src = (HERE / "paper" / "tremor_twin_healthcom.tex").read_text(encoding="utf-8")
    src = src.replace("\\usepackage{graphicx}",
                      "\\usepackage{graphicx}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.17}")
    repl = {
        "\\includegraphics[width=\\textwidth]{figs/fig_twin.pdf}": twin_tikz(),
        "\\includegraphics[width=\\textwidth]{figs/fig_collinearity.pdf}": collinearity_tikz(),
        "\\includegraphics[width=\\columnwidth]{figs/fig_freq.pdf}": freq_tikz(),
        "\\includegraphics[width=\\textwidth]{figs/fig_real.pdf}": real_tikz(),
    }
    for k, v in repl.items():
        assert k in src, f"anchor not found: {k}"
        src = src.replace(k, v)
    out = HERE / "paper" / "tremor_twin_selfcontained.tex"
    out.write_text(src, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
