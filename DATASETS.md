# External tremor datasets for sim-to-real validation (verified 2026-06-30)

Every entry verified against a live URL. Access status is explicit — most
clinical tremor **video** is IRB-gated (identifiable faces); the open ones are
flagged. Our pipeline consumes **landmark trajectories**, so datasets that
release MediaPipe keypoints are usable even when raw video is withheld.

## Recommended validation ladder (best-first)

1. **TIM-Tremor** — the real sim-to-real test. Open video + synchronized wrist
   accelerometer (frequency ground truth). Run our `predict_on_video.py` on it;
   compare predicted frequency to the accelerometer (within-1 Hz / MAE). This is
   the single closest external benchmark and the de-facto standard for the task.
2. **PADS** — open accelerometer (469 subjects incl. an essential-tremor group).
   No video, but the gold-standard **frequency reference**: derive tremor peak
   frequency (PSD) from the 100 Hz signal to check our estimator's targets on
   real tremor and get an ET-vs-PD frequency spread.
3. **UCI-395 STCP / HandPD online** — structured tremor as **pen trajectories**;
   the "stability test" hold is effectively rest tremor → FFT it as a 1-D signal
   sanity check for the frequency head. Small N, clean, open.
4. **Frontiers FoG turning-in-place** — fully-open RGB video + IMU + PSD frequency,
   validates a video→frequency method end-to-end (caveat: lower-limb gait, not
   hand — method validation / negative control, not hand transfer).

## A. Video + ground truth

| Dataset | N | Modalities | Labels | License / access | Fit |
|---|---|---|---|---|---|
| **TIM-Tremor** (4TU, DOI [10.4121/uuid:522d14ed-…](https://doi.org/10.4121/uuid:522d14ed-3019-4206-b49e-a4e674b6440a)) | 55 patients × ~21 tasks | RGB 1080p + Kinect depth + **wrist accel @1 kHz** | frequency (Hz) from accel + severity 0–3 | **open** (per-item DL); 4TU CC-family (confirm on page) | **primary** — used by Pintea 2018 |
| **Frontiers FoG turning-in-place** (Figshare [10.6084/m9.figshare.14984667](https://doi.org/10.6084/m9.figshare.14984667)) | 35 PD | RGB video (lower limbs) + IMU @128 Hz | FoG episodes, UPDRS, PSD FoG-ratio | **CC0, open** | method validation (gait, not hand) |

## B. Landmarks/features released, raw video gated (still usable for us)

| Dataset | N | What's released | Labels | Notes |
|---|---|---|---|---|
| **PARK / UFNet** (Rochester, [ROC-HCI GitHub](https://github.com/ROC-HCI)) | 845 subj / 3306 videos | **MediaPipe hand landmarks + features** (raw video withheld) | MDS-UPDRS finger-tap 0–4 | webcam analogue; but finger-**tap** = bradykinesia, not tremor |
| **PoET** (King's, Friedrich 2024; model on [Harvard Dataverse 10.7910/DVN/CRJRJF](https://doi.org/10.7910/DVN/CRJRJF), code [peach-lucien/PoET](https://github.com/peach-lucien/PoET)) | 66 ET (DBS) | **trained DLC-RCNN model + pipeline** (video on request) | mocap + accel + **FTM** | best methodological baseline to compare against |
| **FastEval Parkinsonism** ([yuyuan871111/fast_eval_Parkinsonism](https://github.com/yuyuan871111/fast_eval_Parkinsonism)) | 840 videos / 186 | code (video on request) | hand-movement severity + FI metric | reference pipeline |
| **TREMAN** ([tesar-tech/treman_algorithms](https://github.com/tesar-tech/treman_algorithms)) | 30 ET/dystonic | code (video on request) | IMU @100 Hz | ET/dystonic baseline |

## C. Accelerometer / IMU (frequency reference, no video)

| Dataset | N | Labels | License / access |
|---|---|---|---|
| **PADS** (PhysioNet [10.13026/m0w9-zx22](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)) | 469 (PD/ET/atypical/HC) | dx class (derive Hz from 100 Hz accel+gyro) | CC BY-NC-SA 4.0, open |
| **ALAMEDA** (Zenodo [10.5281/zenodo.10782573](https://zenodo.org/records/10782573)) | feature rows | 4 binary MDS-UPDRS tremor labels | CC BY 4.0, open (features only) |
| **mPower** (Synapse syn4993293) | 9,520 | self-report MDS-UPDRS subset | **DUA-gated**, no video |
| IEEE DataPort "simulate hand tremor" / MPU9250 sets | 10 / — | simulated / binary shake | mixed, some subscription-gated |

## D. Spiral / handwriting (structured tremor as images or pen trajectories)

| Dataset | N | Modality | License |
|---|---|---|---|
| **UCI-395 Spiral Drawings** ([archive.ics.uci.edu/dataset/395](https://archive.ics.uci.edu/dataset/395), DOI 10.24432/C5Q01S) | 77 (62 PD/15 HC) | **pen trajectory** (X,Y,pressure,t) + images; **STCP hold = rest-tremor signal** | CC BY 4.0, open |
| **HandPD / NewHandPD** (UNESP) | 66 | images **+ online pen-trajectory** (BiSP) | academic-use, open |
| **Kaggle Parkinson's Drawings** (kmader) | 204 imgs | images only | license reported CC BY 4.0 — *verify on page* |

## Practical notes for plugging into our pipeline
- **Frequency** is the clean, objective sim-to-real metric everywhere (accel/IMU
  ground truth ↔ our prediction). Lead with it.
- **Severity** scales differ (TIM-Tremor 0–3; UPDRS 0–4; ours 6-bin Absent→Severe).
  Fit a monotone map or report rank correlation / ±1 agreement — never raw
  cross-scale accuracy.
- **Domain gap to flag**: our twin is **wrist-only**; TIM-Tremor/PARK include
  finger tremor and finger-tapping. Expect a gap on finger-dominant tasks — a
  reason to extend the twin with finger tremor before strong claims.
- Datasets that release **MediaPipe landmarks** (PARK) can be fed to a thin
  variant of `predict_on_video.py` that skips the video-decode step and reads
  keypoints directly.

### Do-not-cite-as-datasets (paper-only, verified)
- "Wearable sensors during drawing tasks … essential tremor" (Sci Rep 2022 /
  IEEE 2023) — methods paper, **no public data**.
- A standalone "Zenodo ET-IMU dataset with frequency labels" — **could not be
  verified to exist**; the open ET signals live inside PADS (DD group) and TIM-Tremor.
