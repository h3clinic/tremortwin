# Results — synthetic digital-twin tremor → blind CV self-improvement

Twin FK validated to ~1e-5 vs the source GLB rest geometry. Each mode: 1440 clips (coupled) / 2400 clips (decoupled), 77 blind features, subject-held-out evaluation.

## Headline comparison (held-out subjects)

| | **coupled** (repo label model) | **decoupled** (honest) |
|---|---|---|
| r(freq, severity) | **-0.97** → DEGENERATE | 0.27 → independent |
| Frequency MAE | 0.144 Hz | 0.182 Hz |
| Frequency within-1 Hz (ML) | 99.7% | 95.8% |
| Frequency within-1 Hz (FFT-only, no ML) | 98.6% | 86.8% |
| Severity exact acc | 96.9% | 86.5% |
| Severity ±1-bin acc | 100.0% | 100.0% |
| Severity macro-F1 | 0.963 | 0.805 |
| Shuffle-null acc (chance) | 0.229 (0.285) | 0.313 (0.349) |

## The data-leakage finding (headline)

- **coupled**: DEGENERATE: targets collinear (|r|>0.9) -> predicting one trivially gives the other (Pearson r(freq,severity)=-0.972, r(freq,amp)=-1.000). Because the repo sets `freq=10-0.06*i` and `amp=0.0015*i`, predicting frequency and predicting severity are the SAME problem. Hitting 95-100% on both is one FFT, not two skills.
- **decoupled**: OK: targets are not collinear -> two independent problems (r=0.266). Frequency (set by tremor type) and severity (set by amplitude) are now genuinely independent targets.
- **Shuffle-null control**: with labels shuffled, accuracy collapses to chance (coupled 0.229 vs 0.285; decoupled 0.313 vs 0.349) → the models learn tremor signal, not identity artifacts.
- **Top severity features (decoupled, permutation importance)**: disp_rms__std, peaky__std, peak_axis_median, peakx__std, disp_rms__mean, acf_freq__std. The pose-invariant angular-amplitude feature dominating is the physically correct signal.

## What ML adds over plain signal processing (frequency)

- decoupled: ML within-1 Hz 95.8% vs fair FFT-only baseline 86.8% (uplift +9.0 pts). Frequency is *largely* recoverable by a periodogram; ML mainly fixes harmonic/2f confusions and noisy clips. This matches the literature (Güney 2022 MAE 0.229 Hz; Pintea 2018).

## Self-improvement trajectory (best CV score per improvement)

**coupled — severity (CV accuracy):** logreg(0.903) → rf(0.955) → histgb(0.966)
**coupled — frequency (CV within-1Hz):** ridge(0.993) → rf(0.999)

**decoupled — severity (CV accuracy):** logreg(0.775) → rf(0.782) → rf(0.833) → histgb(0.855) → histgb(0.859)
**decoupled — frequency (CV within-1Hz):** ridge(0.496) → rf(0.633) → rf(0.962) → extra(0.969)

## Honest read

1. The target (95–100%) is trivially reachable on the *coupled* labels, but the audit proves that is degenerate — a single recovered number.
2. On the *decoupled*, leakage-controlled, subject-held-out setup the numbers are the defensible ones. Severity ±1-bin is the clinically appropriate metric (Zhang 2021).
3. **This is synthetic landmark data.** It bounds performance *under ideal detection*; it does not prove real-patient accuracy. Real-world amplitude accuracy is known to be limited (Wolke et al. 2025). Detectability across skin tones/cameras is a render-side question (see RENDER_SIDE.md, mini50 gate).

See `IEEE_HEALTHCOM_NOTES.md` for the paper skeleton and `REFERENCES.md` for citations.