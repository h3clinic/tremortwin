# Data and assets

Nothing in this list is committed to the repository. Two of the four items carry
licences that prohibit or restrict redistribution, one is 735 MB, and one belongs
to a different project. All are gitignored and fetched locally.

## 1. Rigged-hand asset (required)

`Do_Hand_DetailedRiggedAnimated_shared_16022026.glb` — 21 joints, 5248-vertex mesh,
armature `Do_HandRigged`, wrist bone `radius_ulna`.

Belongs to the upstream **handharness** project and is **not** redistributed here.
Point the twin at your checkout:

```bash
export HANDHARNESS_DIR=/path/to/handharness
# or, if the file sits elsewhere:
export HANDHARNESS_GLB=/path/to/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb
```

Resolution order in `twin._find_glb()`: `$HANDHARNESS_GLB`, then
`$HANDHARNESS_DIR/input/hand_base/extracted/source/<asset>`, then a sibling
`../handharness/` checkout, then the same relative path next to `twin.py`.

Two properties of this asset matter and are easy to get wrong:

- The root node carries an **anisotropic scale** of about `[2.37, 1.93, 2.05]`.
  Ignore it and the hand comes out at roughly half size with wrong proportions.
  `twin.py`'s self-test catches this.
- The five `*_meta` joints share one world position at the carpus, so the rig has
  21 joints but only 16 distinct positions at rest.

Inspect any replacement asset with:

```bash
python probe_glb.py
```

## 2. MediaPipe HandLandmarker (required only for `predict_on_video.py`)

```bash
curl -L -o hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

About 7.5 MB, from Google. Note that `mediapipe` 0.10.x removed the legacy
`mp.solutions.hands` API, so this repository uses the Tasks API
(`mediapipe.tasks.python.vision.HandLandmarker`), which requires this file.

## 3. UCI-395 — Parkinson spiral/stability tablet data

Used by `realdata_uci395.py`. Licence **CC BY 4.0**, open download, about 3.6 MB.

```bash
mkdir -p realdata
curl -L -o realdata/uci395.zip \
  "https://archive.ics.uci.edu/static/public/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet.zip"
```

Format: `hw_dataset/{control,parkinson}/*.txt`, semicolon-separated columns
`X; Y; Z; Pressure; GripAngle; Timestamp; TestID`. `TestID = 2` is the
Stability-on-a-Point hold, which is the rest tremor this project uses. Timestamps
are milliseconds; sampling works out to roughly 140 Hz.

Cite: Isenkul, Sakar & Kursun, "Improved spiral test using digitized graphics
tablet for monitoring Parkinson's disease", UCI ML Repository dataset 395.

## 4. PADS — Parkinson's Disease Smartwatch (PhysioNet)

Used by `pads_validation.py`. Licence **CC BY-NC-SA 4.0** — non-commercial, share-alike.
**Not redistributed.** About 735 MB.

```bash
curl -L -o realdata/pads.zip \
  "https://physionet.org/content/parkinsons-disease-smartwatch/get-zip/1.0.0/"
```

Contents used here:

- `patients/patient_XXX.json` — the `condition` field gives the diagnosis
  (`Healthy` 79, `Parkinson's` 276, `Essential Tremor` 28, plus other groups).
- `movement/timeseries/{id}_{Task}_{Left|Right}Wrist.txt` — 1024 samples at about
  100 Hz, columns `time, ax, ay, az, gx, gy, gz`.
- Tasks used: `Relaxed` for rest tremor, `StretchHold` for postural tremor.

Cite: Varghese et al., PhysioNet, 2023.

## 5. Real hand video (not yet obtained)

The outstanding gap. `predict_on_video.py` needs hand video with a frequency
reference. **TIM-Tremor** (55 patients, RGB video plus synchronised wrist
accelerometry) is the natural benchmark and is what Pintea et al. 2018 used, but
its 4TU record returns HTTP 410 as of this writing following a repository
migration.

Alternatives, with their limitations, are surveyed in
[`../DATASETS.md`](../DATASETS.md). Briefly: PARK/UFNet releases MediaPipe
landmarks but its task is finger tapping, which measures bradykinesia rather than
tremor; PoET and TREMAN release code and models but gate the video.

## Why none of this is committed

| Item | Reason |
|---|---|
| `*.glb` | Belongs to the upstream handharness project. |
| `realdata/pads.zip` | 735 MB, and CC BY-NC-SA forbids unrestricted redistribution. |
| `realdata/uci395.zip` | Small and openly licensed, but kept out so all data is fetched the same way. |
| `hand_landmarker.task` | Third-party Google binary, trivially re-downloadable. |
| `results/models/*.joblib` | About 180 MB, rebuilt in five minutes by `train_export.py`. |
