# TremorTwin

Training a markerless hand-tremor estimator on a rigged digital twin, and checking
honestly whether it transfers to real patients.

A rigged hand is oscillated at a known frequency and amplitude, its 21 joints are
projected to image coordinates, and a model that sees nothing but those landmarks
is trained to recover the tremor frequency and a severity grade. Every experiment
runs a data-leakage audit. The trained spectral core is then applied to two public
real-patient datasets to test transfer.

The repository contains the twin, the generators, the feature extractor, the
self-improving search, the leakage audit, the real-data validations, the figure
and report generators, and the IEEE Healthcom paper built from the results.

---

## Headline results

Subject-held-out (no subject, pose, or camera view appears in two splits).

| Setting | r(freq, severity) | Freq within 1 Hz | Freq MAE | Severity exact | Severity ±1 grade |
|---|---|---|---|---|---|
| Coupled labels, 2 s | **−0.97** (degenerate) | 99.7% | 0.14 Hz | 96.9% | 100% |
| Decoupled, 2 s | 0.32 | 84.3% | 0.49 Hz | 81.9% | 100% |
| Decoupled, 4 s | 0.23 | 91.5% | 0.27 Hz | 90.5% | 100% |
| **Decoupled, 4 s + high-pass** | 0.27 | **95.8%** | **0.18 Hz** | 86.5% | **100%** |

A signal-processing baseline (per-axis spectral peak, no learning) reaches 86.8%
within 1 Hz, so learning adds roughly nine points on frequency. The larger benefit
of learning is on severity, where amplitude must be read through pose-dependent
lever arms.

### The finding that matters most

The upstream pipeline derives frequency **and** amplitude from a single `intensity`
scalar:

```python
frequency_hz  = 10.0 - (intensity / 100.0) * 6.0
amplitude_rad = (intensity / 100.0) * 0.15
```

That makes the two prediction targets collinear (measured r = −0.97, and r = −1.00
between frequency and amplitude). A model that appears to predict both is
predicting one number and reporting it twice. Any "we hit 95–100% on frequency and
severity" claim built on those labels is degenerate.

`generator.py` therefore provides a **decoupled** mode: tremor type fixes the
frequency band, amplitude is sampled independently, and severity follows amplitude.
This also matches the physiology, where frequency and amplitude arise from distinct
neural mechanisms (Pan 2025). Every defensible number in this repository comes from
the decoupled mode.

### Real-patient transfer (partial)

| Dataset | What it shows |
|---|---|
| **UCI-395** (pen "hold" = rest tremor as 2-D position) | Naive peak lands at 2.08 Hz (postural drift). After the same high-pass used in training, median 4.01 Hz with 81% inside the physiological 3.5–7.5 Hz band. Patients vs controls AUC 0.94 (small control group, n=7). |
| **PADS** (469 people, 79 healthy controls) | Recovered frequencies ~5.7 Hz for PD and ET, both physiological. Essential tremor vs controls **AUC 0.90**. Parkinsonian rest tremor vs controls **AUC 0.69** — low because a large share of PD patients are not tremor-dominant, which dilutes the group. |

Both datasets exercise the **spectral core** the model depends on, not the full
21-landmark model, because neither is hand video. See
[REALDATA_VALIDATION.md](REALDATA_VALIDATION.md).

---

## What is and is not established

**Established.**
- The twin's forward kinematics reproduce the source asset's rest geometry to
  within 5×10⁻⁶ of the model size, so the geometry is the real rig.
- Frequency is recoverable to 0.18 Hz on leakage-controlled synthetic data, and the
  same estimator recovers physiological tremor frequency on two real datasets.
- The leakage controls pass everywhere: shuffling labels collapses accuracy to
  chance, and no group crosses a split.
- The frequency/severity collinearity is real, quantified, and fixed.

**Not established.**
- The full 21-landmark model has **never been run on real hand video**. UCI-395
  gives one landmark; PADS gives accelerometer channels. The one public dataset
  suited to the test (TIM-Tremor) is offline as of this writing.
- The twin oscillates the wrist with fingers held still. Real tremor includes
  finger and pill-rolling components.
- Severity on real data is largely untested, and camera-based amplitude is known to
  be less reliable than camera-based frequency (Wolke et al. 2025).

Treat the synthetic numbers as an upper bound under ideal tracking.
`predict_on_video.py` is written and its inference path is verified; it is waiting
on real footage.

---

## Repository layout

Every file, and what it is for.

### Core pipeline (import order: each depends only on the ones above it)

| File | Purpose |
|---|---|
| [`twin.py`](twin.py) | The digital twin. Parses the rigged-hand glTF, runs forward kinematics, applies the wrist tremor model, projects joints to image coordinates, and adds a MediaPipe-style tracker-noise model. Running it directly performs the FK self-test against the asset's own scene graph. |
| [`generator.py`](generator.py) | Builds datasets from the twin. `coupled` reproduces the upstream labelling (used only to expose the collinearity); `decoupled` samples frequency and amplitude independently. Also simulates subjects with distinct hand shapes and noise levels so splits can be subject-wise. |
| [`features.py`](features.py) | Blind feature extraction. Input is only the 21×T×2 landmark stream. Welch spectra, band powers, entropy, displacement, jerk, autocorrelation period, and the pose-invariant angular-amplitude estimate. Includes the drift high-pass that real data forced on us. |
| [`selfimprove.py`](selfimprove.py) | The self-improving search. Escalates the feature set and model family in rounds, scores by subject-grouped cross-validation, stops on target or saturation. Also holds the metric definitions. |
| [`leakage.py`](leakage.py) | The audit suite: group disjointness, target collinearity, label-shuffle null, train/test domain classifier, permutation importance. |
| [`run.py`](run.py) | Orchestrator. Generates a mode, extracts features, holds out subjects, runs the search for both targets, runs the audit, writes `results/<mode>_results.json`. |

### Training, export, and inference

| File | Purpose |
|---|---|
| [`train_export.py`](train_export.py) | Trains frequency and severity models on all synthetic data and pickles them with their exact feature order into `results/models/`. |
| [`bridge_selfcheck.py`](bridge_selfcheck.py) | Runs fresh unseen twin clips through the saved models via the exact inference path used on video, minus MediaPipe. Confirms the model, feature ordering, and vectorization are consistent. |
| [`predict_on_video.py`](predict_on_video.py) | The sim-to-real bridge. Runs the MediaPipe Tasks HandLandmarker over real videos, feeds the landmarks to the same `features.extract`, predicts frequency and severity, and reports sim-to-real metrics if given a labels CSV. |

### Real-data validation

| File | Purpose |
|---|---|
| [`realdata_uci395.py`](realdata_uci395.py) | Parses the UCI-395 stability-hold task, runs the project's own spectral estimator, and reports recovered frequency (raw vs drift-removed) and patient/control separation. |
| [`pads_validation.py`](pads_validation.py) | Parses the PADS cohort, recovers tremor frequency on the rest and postural tasks, and reports AUCs against the healthy control group. |

### Reporting and figures

| File | Purpose |
|---|---|
| [`make_report.py`](make_report.py) | Turns the result JSON into [`RESULTS_REPORT.md`](RESULTS_REPORT.md) and fills the numeric placeholders in the paper notes. |
| [`make_figures.py`](make_figures.py) | Renders the four paper figures as vector PDFs into `paper/figs/`, from the real model and real datasets. |
| [`make_pgf.py`](make_pgf.py) | Emits `paper/tremor_twin_selfcontained.tex`, in which every figure is inline pgfplots so the paper compiles with no image files. |
| [`gen_codemap.py`](gen_codemap.py) | Regenerates [`docs/CODE_MAP.md`](docs/CODE_MAP.md) from the AST, so the code map cannot drift from the code. |
| [`probe_glb.py`](probe_glb.py) | One-off inspection of the rigged asset: node hierarchy, skins, joints, mesh size. How the anisotropic root scale was found. |
| [`run_selfeval.sh`](run_selfeval.sh) | The self-evaluation sweep: both modes, the long-clip configuration, retrain, and report, in one pass. |

### Documentation

| File | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the pieces fit, the data shapes that flow between them, and the design decisions with reasons. |
| [`docs/CODE_MAP.md`](docs/CODE_MAP.md) | Auto-generated. Every module, class and function with signature, purpose, and line number. |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | Exact commands to reproduce every number in this README. |
| [`docs/DATA.md`](docs/DATA.md) | The assets and datasets, how to obtain each, and their licences. Nothing licence-restricted is committed here. |
| [`RESULTS_REPORT.md`](RESULTS_REPORT.md) | Generated results report with the leakage audit and the self-improvement trajectory. |
| [`REALDATA_VALIDATION.md`](REALDATA_VALIDATION.md) | The UCI-395 and PADS validations in full, with scope limits. |
| [`DATASETS.md`](DATASETS.md) | Survey of external tremor datasets, access status verified, with a recommended validation ladder. |
| [`REFERENCES.md`](REFERENCES.md) | Verified citations: video-tremor prior art, clinical scales, sim-to-real, and useful repositories. |
| [`IEEE_HEALTHCOM_NOTES.md`](IEEE_HEALTHCOM_NOTES.md) | Paper working notes: framing, related-work differentiation, novelty positioning, limitations. |
| [`RENDER_SIDE.md`](RENDER_SIDE.md) | The Blender rendering half: the black-triangle fix, skin-tone randomisation, and the mini50 gate before scaling up. |
| [`paper/README.md`](paper/README.md) | How to build the paper and which of the two `.tex` files to use. |
| [`results/README.md`](results/README.md) | What each result JSON and log contains. |
| [`realdata/README.md`](realdata/README.md) | How to fetch the two real datasets. |

---

## Quickstart

```bash
pip install -r requirements.txt
export HANDHARNESS_DIR=/path/to/handharness    # the rigged-hand asset lives there
python twin.py                                  # FK self-test, should print PASS
python run.py --mode both --poses 360           # full pipeline + leakage audit
```

`twin.py` finds the asset via `$HANDHARNESS_GLB`, then `$HANDHARNESS_DIR`, then a
sibling `../handharness/` checkout. See [`docs/DATA.md`](docs/DATA.md).

Full command sequences for every result are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

---

## The paper

`paper/` holds the IEEE Healthcom submission, built from the numbers in
`results/`. Use `tremor_twin_selfcontained.tex` — every figure is inline pgfplots,
so it compiles anywhere with no image uploads. See [`paper/README.md`](paper/README.md).

---

## Provenance and attribution

- The rigged-hand asset and the Blender tremor pipeline this twin mirrors come from
  the upstream **handharness** project. The asset is **not** redistributed here.
- **UCI-395**: Isenkul, Sakar & Kursun, UCI Machine Learning Repository dataset 395
  (CC BY 4.0).
- **PADS**: Varghese et al., PhysioNet (CC BY-NC-SA 4.0). Not redistributed;
  non-commercial terms apply.
- **MediaPipe HandLandmarker** model: Google, downloaded at runtime.
- Prior art and clinical references are listed in [`REFERENCES.md`](REFERENCES.md).

This is research code. It is not a medical device and must not be used for
diagnosis.
