# IEEE Healthcom submission notes — synthetic digital-twin tremor → blind CV

> Working notes for a short paper. Numbers in **[[RESULTS]]** tags are filled
> from `results/summary.json` and `results/{coupled,decoupled}_results.json`
> produced by `run.py`. Everything is reproducible from this folder.

## Proposed title

**"A Rigged-Hand Digital Twin for Leakage-Controlled, Synthetic-Data Training of
Markerless Tremor Frequency and Severity Estimators"**

(Alt: "Can a Blind Camera Read Tremor It Has Never Seen? A Digital-Twin Benchmark
with an Honest Sim-to-Real Boundary.")

## One-paragraph abstract (draft)

Markerless, video-based tremor assessment is attractive for tele-neurology, but
labelled patient video is scarce and clinician ratings are noisy. We ask whether
a model trained **only** on a synthetic, rigged-hand *digital twin* can recover
the two clinically relevant descriptors — tremor **frequency** and **severity** —
from MediaPipe-style hand landmarks alone. We drive an anatomically rigged hand
(21 joints, forward-kinematics-exact to the source asset) with a parametric wrist
tremor, render landmark trajectories under domain randomization (subject hand
shape, camera view, tracker noise), and train a "blind" estimator through a
self-improving feature/model search. Crucially, we expose a **label-design
leakage trap**: the naïve generator ties frequency and amplitude to one
intensity scalar, making the two tasks perfectly collinear (r≈−1) so that "99%
on both" is a single FFT in disguise. After decoupling the latent factors, a
subject-held-out evaluation gives frequency within-1 Hz **95.8%** (MAE 0.18 Hz,
4 s clips, drift-robust high-pass) and severity **86.5%** exact / **100%** ±1-bin
accuracy, with a label-shuffle null collapsing to chance. The same spectral core,
applied to real patient tremor (UCI-395), recovers physiological PD rest-tremor
frequency (median 4.0 Hz) and separates PD from controls at AUC 0.83–0.94. We
argue the meaningful claim is not the
synthetic score but the **protocol**: subject-independent splits, an explicit
leakage audit, and a stated sim-to-real boundary (our twin models landmark
*geometry*, not appearance; detectability across skin tones remains a render-side
question).

## Contributions

1. **An FK-exact landmark digital twin** of a rigged clinical hand asset that
   reproduces the project's tremor model at the landmark level (validated to
   1e-5 against the GLB rest geometry), enabling unlimited labelled trajectories
   without rendering.
2. **A self-improving, leakage-controlled training protocol** (subject-grouped
   CV + held-out subjects + label-shuffle null + train/test domain check +
   permutation importance) that we argue should be the default for synthetic
   movement-disorder CV.
3. **A concrete negative result / methodological warning**: the common
   "intensity → (frequency, amplitude)" parameterization makes the two target
   tasks collinear; we quantify the trap and show the decoupled design that fixes
   it.
4. **An honest sim-to-real statement**: what landmark-level synthetic data can and
   cannot establish, aligned with the only-modest real-world amplitude accuracy
   reported by Wolke et al. 2025.

## Related work & novelty positioning (drop-in)

Prior video/CV tremor work **trains on real patient video**: Pintea et al. 2018
(TIM-Tremor frequency benchmark), Friedrich et al. 2024 (npj Digit Med — MediaPipe
→ frequency + amplitude, validated vs accelerometry in 66 patients), Duque-Quiceno
et al. 2024 (pixel DL severity), Liu et al. 2023 (MDS-UPDRS from video), Güney et
al. 2022 (MediaPipe, MAE 0.229 Hz). Synthetic/simulation enters tremor ML only in
narrow forms: **Wolke et al. 2025** renders a rigged hand in Blender but uses it
*only to validate/calibrate frozen MediaPipe/Vision detectors — no model is
trained on it*; an **essential-tremor neuromusculoskeletal digital twin**
(bioRxiv 2025) trains purely on twin data but predicts *muscle activation* from
wrist-angle (not frequency/severity, not CV, no sim-to-real); **TremorGAN / DCGAN**
work (2020, 2025) augments *IMU/EMG* signals for *classification*; Marquez-Chavez &
Tang 2022 uses synthetic *gait* (not hand tremor).

**Our two claims and how they differ:**

| Work | CV? | trains on synthetic? | target | sim-to-real? |
|---|---|---|---|---|
| Friedrich 2024, Pintea 2018, Güney 2022 | ✅ | ❌ (real video) | freq/sev | n/a |
| **Wolke 2025** | ✅ | ❌ **validate-only** | freq/amp | ✅ |
| ET digital-twin (bioRxiv 2025) | ❌ | ✅ | muscle activation | ❌ |
| TremorGAN 2025 | ❌ (IMU) | augment | class | ❌ |
| **Ours** | ✅ | ✅ **train-primarily** | **freq + severity** | **partial** |

1. **First to *train* a CV frequency/severity estimator primarily on synthetic
   rigged-hand digital-twin data and show sim-to-real transfer** — crossing the
   train-vs-validate line that Wolke 2025 explicitly does not, on the freq/severity
   target the muscle-activation twin does not address.
2. **The frequency↔severity collinearity pitfall**: tying both targets to one
   latent generative scalar makes them collinear (r≈−1), silently reducing "we
   predict both" to one number. Unraised in the tremor-ML critique literature, and
   consistent with physiology — Pan 2025 shows tremor frequency and amplitude arise
   from *distinct neural mechanisms*, so they should be modeled as independent
   factors, exactly as our decoupled generator does (and the repo's coupled one
   does not).

**Honest scope of the novelty (state it plainly):** our sim-to-real evidence is
currently **partial** — the shared spectral *core* transfers to real tremor
(UCI-395 pen trajectories; PADS accelerometer), but the full 21-landmark model on
real hand *video* is not yet demonstrated (TIM-Tremor offline). We therefore claim
a synthetic-training **method** + a leakage-honest **protocol** + a **partial
sim-to-real foothold** + the **collinearity finding** — not a completed clinical
validation. (Two prior-art sources — the TremorGAN paper and the Mov Disord ML-tremor
critique — were paywalled; positioning rests on their abstracts.)

## Method (concise)

- **Twin.** Source asset `Do_Hand_DetailedRiggedAnimated…glb`: 21 joints, root
  anisotropic scale handled, FK reproduces rest geometry to 1e-5. Wrist
  (`radius_ulna`) carries a parametric tremor (primary + 2nd harmonic + phase-
  shifted axes + inter-cycle micro-irregularity), fingers held static per clip →
  the posed hand is a rigid body swinging about the wrist (matching the repo's
  `generate_tremor_dataset.py` / `blender_tremor_sequence.py`).
- **Domain randomization.** Per *subject*: anisotropic hand-shape factor and a
  tracker-noise level (jitter, global wobble, dropout). Per *clip*: random static
  finger pose, camera view (front / left / right / dorsal-fingertip), tremor
  phase offset. 2.0 s clips @ 60 fps (frequency resolution motivates ≥2 s; see
  ablation).
- **Two label models.**
  - *coupled* (repo-faithful): `f = 10 − 0.06·i`, `A = 0.0015·i`, severity =
    6-band clinical bin of `i` (UPDRS/TETRAS-referenced). Used only to
    demonstrate the leakage trap.
  - *decoupled* (honest): tremor **type** sets the frequency band
    (Parkinsonian rest 4–6 Hz, essential/postural 5–9 Hz, enhanced-physiologic
    8–12 Hz; Lenka & Jankovic 2021), amplitude sampled **independently**, severity
    = amplitude band (as TETRAS rates amplitude).
- **Blind features.** Only the 21×T×2 landmark stream is input. Welch PSD
  (peak freq w/ parabolic interpolation, band powers, spectral entropy/centroid),
  temporal descriptors (size-normalized displacement, jerk, zero-crossings,
  autocorrelation period), a **pose-invariant angular-amplitude** estimate
  (linear displacement ÷ lever arm = recovered wrist rotation), and
  cross-landmark frequency agreement. All amplitudes normalized by in-clip hand
  scale (mirrors render-side `normalize_signals.py`).
- **Self-improvement loop.** Rounds escalate features (temporal → +spectral →
  +agreement) and model capacity (linear → RF/ExtraTrees/HistGB → tuned),
  selecting by **subject-grouped** CV, stopping at target or saturation; winner
  judged once on held-out subjects.

## Evaluation protocol (the part reviewers should like)

- **Split:** subject-independent. Test subjects are isolated before the search;
  CV folds inside train are also grouped by subject. No pose/clip/view crosses a
  split (asserted: `group_disjoint`).
- **Metrics:** frequency — MAE (Hz), %within-1 Hz (Pintea et al. 2018 criterion),
  %within-0.5 Hz; severity — exact accuracy, **±1-bin (adjacent) accuracy**
  (ordinal, per Zhang et al. 2021), macro-F1.
- **Leakage audit (run every time):**
  1. target collinearity r(freq, severity);
  2. label-shuffle null (must fall to chance);
  3. train/test domain-classifier AUC (≈0.5 desired);
  4. permutation importance (physics features should dominate).
- **Honest baselines:** a **signal-processing-only** frequency estimate
  (per-axis PSD peak, no learning) to quantify what ML actually adds; majority-
  class baseline for severity.

## [[RESULTS]] — fill from results/summary.json

| Mode | r(freq,sev) | Freq within-1Hz (ML) | Freq within-1Hz (FFT-only) | Freq MAE | Severity exact | Severity ±1-bin | Shuffle-null |
|------|-------------|----------------------|----------------------------|----------|----------------|------------------|--------------|
| coupled (repo, 2 s) | **-0.97** | 99.7% | 98.6% | 0.14 Hz | 96.9% | 100.0% | 0.23 |
| decoupled (honest, 2 s) | 0.32 | 84.3% | 81.8% | 0.49 Hz | 81.9% | 100.0% | 0.35 |
| decoupled (honest, 4 s) | 0.23 | 91.5% | 88.8% | 0.27 Hz | **90.5%** | 100.0% | 0.37 |
| **decoupled (4 s, +high-pass)** | 0.27 | **95.8%** | 86.8% | **0.18 Hz** | 86.5% | 100.0% | 0.31 |

Interpretation to write up:
- coupled r≈−1 ⇒ frequency and severity are one task; 95–100% there is degenerate.
- decoupled r≈0 ⇒ two genuine tasks; report the honest scores.
- 2 s→4 s ablation: the self-improvement loop climbs with more observation time + data.
- **Drift-robust high-pass (real-data-motivated, validated on UCI-395)**: pushes
  frequency to **95.8% within-1 Hz / MAE 0.18 Hz** (crossing 95% legitimately);
  costs ~4 pts of severity *exact* (it slightly attenuates 4 Hz rest-tremor
  amplitude) but ±1-bin stays 100%. A dual-cutoff design (gentle detrend for
  amplitude, ≥2.5 Hz high-pass for frequency) would recover both — noted as future work.
- ML-vs-FFT gap (+9 pts at 4 s+high-pass) quantifies the learning contribution to
  frequency; the larger ML value is in severity.
- shuffle-null ≈ chance ⇒ no identity leakage (every run: group-disjoint asserted,
  collinearity flags coupled degenerate).

## External real-data validation (UCI-395 + PADS) — see REALDATA_VALIDATION.md

With TIM-Tremor offline, we validated the shared **spectral core** on two real
patient datasets using the project's own `features._peak_freq`:

- **UCI-395** (pen "hold" = tremor as 2-D position, our modality): after drift
  removal, recovered **PD rest-tremor median 4.0 Hz, 81% in the 4–6 Hz band**
  (raw peak-picking is captured by ~2 Hz postural drift — the sim-to-real lesson
  that motivated the high-pass). Features separate PD vs control at AUC 0.83–0.94
  (small control n=7 → wide CI).
- **PADS** (PhysioNet, 469 pts, 100 Hz accel, **79 healthy controls**): recovered
  PD-rest and ET-postural frequencies both ~5.7 Hz (physiological). Features detect
  **ET (tremor-defined) vs HC at AUC ≈0.90**, and **PD-rest vs HC at AUC ≈0.69** —
  the modest PD number is an honest reflection of PD subtype heterogeneity
  (30–50% non-tremor-dominant), not a method flaw. Frequency alone does not
  separate ET from PD (p=0.07), as clinically expected.

Together: real, **controlled** evidence that the frequency/tremor core transfers to
real tremor (strong for tremor-defined ET, honestly modest for heterogeneous PD).
Neither tests the full 21-landmark model or the video path (both are non-video
modalities) — that remains the open sim-to-real gap. Strong "external validation +
honest boundary" material.

## Discussion / limitations (do NOT bury these)

1. **Sim-to-real is unproven here.** The twin models landmark *geometry*, not
   pixels. It cannot establish that MediaPipe detects real hands across skin
   tones / lighting / oblique cameras — that is a render-and-detect question
   (see `RENDER_SIDE.md`, the mini50 gate). Real-world amplitude accuracy is
   known to be limited (Wolke et al. 2025). Frame all synthetic numbers as an
   **upper bound under ideal detection**.
2. **Wrist-only tremor.** The repo (and twin) oscillate the wrist with static
   fingers. Real tremor includes finger and pill-rolling components; extend the
   twin before clinical claims.
3. **Synthetic noise ≠ real tracker noise.** We model jitter/dropout
   parametrically; real MediaPipe failure modes (self-occlusion, motion blur) are
   richer.
4. **Severity is amplitude-defined and pose-confounded.** Exact 6-bin accuracy
   saturates below frequency accuracy because observed amplitude depends on lever
   arm and view; ±1-bin is the clinically appropriate target.

## Suggested figures

- Fig 1. Pipeline (twin → landmarks → blind features → self-improving model →
  leakage audit).
- Fig 2. FK validation (twin vs GLB rest geometry) + example tremor trajectories.
- Fig 3. The leakage trap: scatter of true frequency vs true severity for coupled
  (a line, r≈−1) vs decoupled (a cloud).
- Fig 4. Self-improvement curve (CV score vs round) for both tasks.
- Fig 5. Confusion matrix (severity) + Bland–Altman (frequency) on held-out
  subjects; ML vs FFT-only.
- Fig 6. Ablation: accuracy vs clip length (frequency-resolution limit).

## Venue fit

IEEE Healthcom (healthcare comms / digital health, e-health, tele-monitoring).
The angle that fits: a reproducible, leakage-honest **benchmark + protocol** for
synthetic-data tele-tremor assessment, with an explicit sim-to-real boundary —
not a "we beat clinicians" claim. Closest prior art to cite as the bridge:
Friedrich et al. 2024 (npj Digital Medicine, MediaPipe tremor) and Pintea et al.
2018 (video tremor frequency benchmark). See `REFERENCES.md`.
