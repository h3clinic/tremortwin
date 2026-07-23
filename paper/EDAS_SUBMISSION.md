# IEEE Healthcom 2026 — EDAS submission fields

Paste these into the EDAS "Register a paper" form. Plain text (LaTeX stripped) so
the fields render correctly.

## Title
TremorTwin: Training a Markerless Tremor-Frequency and Severity Estimator on a Rigged-Hand Digital Twin, with a Leakage-Controlled Evaluation

## Keywords (up to 6)
1. Tremor quantification
2. Synthetic training data
3. Hand landmark tracking
4. Digital twin
5. Sim-to-real transfer
6. Parkinson's disease

## Topics (choose 1–3)
Primary: Signal/Data Processing and Computing For Health Systems
Second:  Medical Studies
Third:   Service & Applications

(Alternative for the third slot: "Privacy Enhancing Technology in eHealth" —
defensible because synthetic training data avoids sharing identifiable patient
video, but only if you add a sentence or two developing that angle in the paper.
As written, the three above match the content more closely.)

## Abstract (plain text, ~260 words; limit is 50–1000)
Video-based tremor assessment is a promising modality for tele-neurology, but annotated patient video is scarce and clinician ratings are subject to inter-rater variability. This study investigates whether a model trained solely on synthetic data can recover the two clinically relevant tremor descriptors, oscillation frequency and severity, from hand landmarks alone. We present TremorTwin, a landmark-level digital twin derived from an anatomically rigged hand asset. Forward kinematics reproduce the asset's rest geometry to within 5x10^-6 of the model size. The wrist is driven by a parametric tremor, the moving joints are projected to image coordinates under a tracker-noise model, and a blind estimator is trained through an iterative feature and model search. We further identify a labelling pitfall that inflates reported accuracy: when frequency and amplitude are both derived from a single intensity scalar, the two prediction targets become collinear (r = -0.97), so an apparent joint prediction reduces to one measurement. After the two factors are sampled independently, a subject-held-out evaluation yields a frequency error below 1 Hz on 95.8% of clips (mean absolute error 0.18 Hz) and a severity grade within one grade on all clips (exact-grade accuracy 86.5%). A label-shuffle control returns to chance and no subject appears in two splits. Applied to two public datasets, the same spectral estimator recovers Parkinsonian rest-tremor frequency near 4-6 Hz and separates patients from controls, although the separation is strong only for essential tremor (AUC 0.90). The full landmark model has not been evaluated on real video, because the one public dataset suited to that test is presently offline. We therefore report a synthetic-training method, a leakage-controlled evaluation protocol, and a partial sim-to-real result, rather than a clinical validation.

## Notes before you submit
- Page limit is 6 IEEE pages (a 7th is allowed at USD 100). The current draft is
  around 4–5 pages; a method figure and a Bland-Altman plot would fill it.
- Final upload is a PDF built from the IEEE conference template (the .tex already
  uses IEEEtran). No page numbers, headers, or footers in the PDF.
- Confirm whether review is double-blind. If it is, keep the author block
  anonymised in the PDF (it currently reads "[Author names withheld for review]").
  If single-blind, put the real author names in before uploading. EDAS records
  authors separately from the PDF either way, so "Add yourself as author" in EDAS
  is required regardless.
- Verify the Wolke 2025 and the other paywalled reference against full text before
  camera-ready.
