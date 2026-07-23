# Real datasets

Empty on a fresh clone. The archives are gitignored, one because of its licence and
one because of its size. Full details, formats, and citations are in
[`../docs/DATA.md`](../docs/DATA.md).

## Fetch

```bash
# UCI-395 — Parkinson tablet drawings, CC BY 4.0, ~3.6 MB
curl -L -o uci395.zip \
  "https://archive.ics.uci.edu/static/public/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet.zip"

# PADS — PhysioNet smartwatch cohort, CC BY-NC-SA 4.0, ~735 MB
curl -L -o pads.zip \
  "https://physionet.org/content/parkinsons-disease-smartwatch/get-zip/1.0.0/"
```

Both scripts read straight from the `.zip`; do not unpack them.

```bash
cd .. && python realdata_uci395.py
cd .. && python pads_validation.py
```

## What each one tests

| Dataset | Signal used | What it establishes |
|---|---|---|
| UCI-395 | Pen X/Y during the Stability-on-a-Point hold (`TestID = 2`) — rest tremor as a 2-D position series, the same modality a landmark produces | That the spectral estimator recovers real Parkinsonian rest-tremor frequency once postural drift is removed |
| PADS | Wrist accelerometer, `Relaxed` (rest) and `StretchHold` (postural) tasks | That the tremor-band features separate real patients from a proper control group (n = 79) |

Neither is hand video, so neither tests the full 21-landmark model or the MediaPipe
path. They test the spectral core that the model depends on. That boundary is
stated in [`../REALDATA_VALIDATION.md`](../REALDATA_VALIDATION.md) and in the paper.

## Licence note

PADS is **CC BY-NC-SA 4.0**: non-commercial, share-alike, attribution required. Do
not commit it, redistribute it, or use it commercially. Cite Varghese et al.,
PhysioNet, 2023.
