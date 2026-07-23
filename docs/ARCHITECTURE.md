# Architecture

How the pieces fit, what flows between them, and why each design decision was made.

## The premise the whole design rests on

The downstream model never sees pixels. It sees 21 hand landmarks per frame.

That single fact determines the architecture. Rendering quality, skin tone, and
lighting decide only whether MediaPipe *detects* the hand. Once landmarks exist,
the entire tremor signal — frequency and amplitude — lives in the landmark
*trajectory*. So a twin that produces faithful landmark trajectories is a faithful
stand-in for the render-and-track pipeline for everything except detection, and it
runs without Blender, without rendering, and without a GPU.

The cost of this choice is stated plainly: the twin cannot answer detectability
questions. That half lives in [`../RENDER_SIDE.md`](../RENDER_SIDE.md).

## Data flow and shapes

```
rigged .glb  (21 joints, hierarchy + local transforms)
   |
   |  twin.HandRig.fk()            forward kinematics
   v
(21, 3)  joint positions in model space
   |
   |  wrist rotation theta(t) applied to the whole rigid posed hand
   |  twin.project()               view rotation + weak perspective
   v
(21, 2)  normalized image coordinates, one frame
   |
   |  twin.render_clip()           T frames
   |  twin.add_observation_noise() jitter, wobble, dropped frames
   v
(T, 21, 2)  one clip   <-- this is exactly what MediaPipe outputs for a real video
   |
   |  features.extract()           high-pass, Welch spectra, temporal descriptors
   v
dict of 77 scalar features
   |
   |  run.build_matrix()
   v
X (n_clips, 77) + labels + group ids
   |
   |  selfimprove.run_self_improvement()   subject-grouped CV, escalating search
   |  leakage.*                            audit at every step
   v
results/<mode>_results.json  ->  make_report.py  ->  RESULTS_REPORT.md
                             ->  make_figures.py ->  paper/figs/*.pdf
```

## Why the landmark ordering matters

`twin.LANDMARK_ORDER` places the wrist at index 0 and the five fingertips at
indices 4, 8, 12, 16, 20. That is MediaPipe's native ordering. It is deliberate:
real MediaPipe output can be fed straight into `features.extract` with **no
remapping**, which is what makes `predict_on_video.py` a thin bridge rather than a
translation layer. Change `LANDMARK_ORDER` and the real-video path silently breaks.

## Design decisions, with reasons

### The rig is parsed, not approximated
`twin.HandRig` reads the glTF node hierarchy and local transforms directly and runs
its own forward kinematics. Running `python twin.py` checks the result against the
asset's own scene graph and reports the maximum error (5×10⁻⁶ of model size).
An anisotropic scale on the root node must be applied or the hand comes out at
roughly half size with the wrong proportions; that bug is why the self-test exists.

### Only the wrist oscillates
The upstream pipeline holds the fingers fixed for a clip and oscillates the wrist,
so the posed hand is a rigid body rotating about the wrist. The twin reproduces
that exactly, including the second harmonic and the inter-cycle noise term. This is
a faithfulness decision, not a simplification we chose — and it is also a stated
limitation, because real tremor has finger components.

### Two label models, one of which exists only to be criticised
`coupled` reproduces the upstream `intensity`-driven labelling. It is kept so the
collinearity can be measured and reported rather than asserted. `decoupled` samples
a tremor type (which fixes a frequency band) and an amplitude independently, and
derives severity from amplitude. All defensible numbers come from `decoupled`.

### Severity is read as an angle, not a displacement
Observed landmark displacement is `amp_rad × lever_arm`, and the lever arm changes
with finger pose, camera view, and hand size. Dividing a landmark's displacement by
its distance from the wrist recovers the wrist rotation amplitude directly, which is
pose-, view-, and scale-invariant. That feature (`ang_amp`) is the physically
correct severity signal and is why severity is learnable at all.

### The high-pass came from real data, not from tuning
Clean synthetic signals have no postural drift, so the estimator originally did
without a high-pass. On UCI-395 the naive spectral peak landed at 2 Hz, tracking
slow postural drift rather than the 4 Hz tremor. Adding a 2.5 Hz high-pass to the
per-landmark coordinate signals moved the recovered frequency into the
physiological band on real data, and on synthetic data it raised frequency accuracy
from 91.5% to 95.8%. It costs about four points of exact-grade severity because it
slightly attenuates the slowest rest tremor near 4 Hz. Both sides of that trade are
reported.

### Splits are grouped by subject, always
`generator.py` simulates subjects with distinct hand shapes and tracker-noise
levels. Every split — the held-out test set and the inner cross-validation folds —
is grouped by `subject_id`. `leakage.group_disjoint` asserts that no subject, pose,
or camera view appears in two splits. The four views of one pose share a tremor
state, so splitting them across train and test would leak directly.

### The audit runs on every experiment, not on demand
`run.py` calls the full audit each time: target collinearity, label-shuffle null,
train/test domain classifier, and permutation importance. A result that has not
been audited is not reported.

## Module dependency graph

```
twin.py                     no internal dependencies
  generator.py              imports twin
  features.py               imports twin
selfimprove.py              sklearn/numpy only
leakage.py                  sklearn/scipy only
  run.py                    imports generator, features, selfimprove, leakage
  train_export.py           imports generator, features, twin
    bridge_selfcheck.py     imports generator, features, twin + saved models
    predict_on_video.py     imports features, twin + saved models + MediaPipe
  realdata_uci395.py        imports features only
  pads_validation.py        imports features only
  make_figures.py           imports twin, generator, features (read-only)
  make_pgf.py               imports the above, rewrites the paper
  make_report.py            reads results JSON only
```

Per-function detail is in [CODE_MAP.md](CODE_MAP.md), generated from the AST.
