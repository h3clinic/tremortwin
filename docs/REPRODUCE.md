# Reproducing every number

Each result in the README maps to one command below. Runtimes are from a laptop
CPU (no GPU is used anywhere).

## 0. Setup

```bash
pip install -r requirements.txt
export HANDHARNESS_DIR=/path/to/handharness      # holds the rigged-hand .glb
python twin.py
```

Expected: a per-joint comparison table ending in `MAX ERROR = 0.000005 -> PASS`,
then fingertip displacement rising monotonically with tremor intensity. If it
raises `FileNotFoundError`, the asset path is wrong; see [DATA.md](DATA.md).

## 1. Synthetic results and the leakage audit

The 2-second rows of the results table, both label models:

```bash
python run.py --mode both --poses 360 --subjects 24 --frames 120 --fps 60
```

Writes `results/coupled_results.json` and `results/decoupled_results.json`.
Runtime about 6 minutes. Console prints the collinearity verdict, the
self-improvement trajectory for both targets, the held-out metrics, and the
shuffle-null result.

Expected: coupled reports `r(freq,severity) = -0.97` with the verdict
`DEGENERATE`; decoupled reports `r ~ 0.3` with `OK`.

The 4-second headline row:

```bash
python run.py --mode decoupled --poses 600 --subjects 30 --frames 240 --fps 60 \
  --target-frequency 0.95 --target-severity 0.95
```

Runtime about 10 minutes. This is the row that yields 95.8% within 1 Hz and
0.18 Hz mean absolute error.

Generate the report from whatever JSON is present:

```bash
python make_report.py          # writes ../RESULTS_REPORT.md
```

To run the whole sweep (both modes, long clips, retrain, report) in one pass:

```bash
bash run_selfeval.sh           # about 20 minutes
```

## 2. Train and export the models

```bash
python train_export.py --poses 500 --frames 180 --fps 60 --subjects 28
```

Writes `results/models/frequency.joblib` and `results/models/severity.joblib`
(about 180 MB total, gitignored). Runtime about 5 minutes.

Verify the saved-model inference path on fresh unseen clips:

```bash
python bridge_selfcheck.py
```

Expected: `PATH CHECK: PASS`, with frequency within 1 Hz around 0.89 and severity
within one grade at 1.000. This exercises the exact path `predict_on_video.py` uses,
minus MediaPipe.

## 3. Real-tremor validation

Fetch the datasets first ([DATA.md](DATA.md)).

```bash
python realdata_uci395.py
```

Expected: about 47 Parkinson's and 7 control hold-task recordings parsed; raw peak
median 2.08 Hz with 2% in the physiological band; after the high-pass, median
4.01 Hz with 81% in band; patient-vs-control AUC up to 0.94.

```bash
python pads_validation.py
```

Expected: 276 PD, 79 healthy, 28 essential tremor; recovered frequencies near
5.7 Hz; essential tremor vs healthy AUC about 0.90; Parkinsonian rest tremor vs
healthy about 0.69. Runtime about 3 minutes (it reads roughly 10,000 recordings).

## 4. Real video, when you have it

```bash
python predict_on_video.py --videos /path/to/clips --labels labels.csv \
  --models-dir results/models
```

`labels.csv` takes a `video` column plus optional `freq_hz` and `severity`
(`Absent`, `Slight`, `Mild`, `Moderate`, `Marked`, `Severe`). Without labels it
prints predictions only. Requires `hand_landmarker.task`; see [DATA.md](DATA.md).

This is the one path in the repository that has **not** been run on real footage.
Its inference half is verified by `bridge_selfcheck.py`; the MediaPipe half is
verified only to initialise.

## 5. Figures and paper

```bash
python make_figures.py     # vector PDFs into paper/figs/
python make_pgf.py         # paper/tremor_twin_selfcontained.tex, figures inline
python gen_codemap.py      # docs/CODE_MAP.md
```

`make_figures.py` needs the trained models (step 2) and both datasets (step 3),
because three of the four figures are drawn from real model output and real data.

## Determinism

Generators take an explicit `--seed`, and the seeds used for the reported numbers
are the defaults in each command above. Held-out evaluations in
`bridge_selfcheck.py` and the figure scripts deliberately use different seeds from
training so the clips are genuinely unseen. Exact figures may shift in the last
decimal across scikit-learn versions.
