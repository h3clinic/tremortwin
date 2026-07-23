# Results

Written by `run.py`, `train_export.py`, and `make_report.py`. Everything here is
regenerable; see [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

## Files

| File | Contents |
|---|---|
| `coupled_results.json` | Current coupled-label run. Kept only to demonstrate the collinearity, not as a result. |
| `decoupled_results.json` | Current decoupled run, 4-second clips with the drift high-pass. **The headline numbers come from here.** |
| `coupled_prefix.json`, `decoupled_prefix.json` | The same two runs *before* the high-pass was added, preserved so the change can be audited rather than asserted. |
| `coupled_2s_results.json`, `decoupled_2s_results.json` | The earlier 2-second-clip runs, kept for the clip-length ablation. |
| `summary.json` | Condensed metrics across modes, written at the end of a `run.py` invocation. |
| `full_run.log`, `improve_4s.log`, `selfeval.log`, `train_export.log` | Console transcripts, including the round-by-round self-improvement trajectory. |
| `models/` | Pickled models from `train_export.py`. **Gitignored** (~180 MB); rebuild in about five minutes. |

## Result JSON schema

```
mode                        "coupled" | "decoupled"
n_rows, n_features          matrix dimensions
config                      poses, frames, fps, subjects, seed, test_frac, ...
label_collinearity
  pearson_freq_severity     the headline leakage number
  pearson_freq_amp          -1.0 exactly in coupled mode, by construction
  verdict                   "DEGENERATE: ..." | "OK: ..."
severity
  group_disjoint            {disjoint: bool, overlap_count, group_counts}
  best                      winning {score, config, feature set}
  history                   every candidate tried, with its CV score
  test_metrics              accuracy, adjacent_acc, macro_f1
  test_class_support        per-class counts in the held-out set
  leakage
    label_shuffle_null      shuffled score vs majority-class chance, plus verdict
    train_test_separability domain-classifier AUC (0.5 is ideal)
    top_features            permutation importance, top 10
frequency
  best, history             as above
  test_metrics              mae_hz, within_1hz, within_0p5hz, r2
  naive_fft_baseline        same metrics for the no-learning spectral peak
```

## Reading these honestly

- Check `label_collinearity.verdict` **before** any accuracy figure. If it says
  `DEGENERATE`, the frequency and severity numbers in that file describe one
  recovered quantity, not two.
- Check `leakage.label_shuffle_null`. The shuffled score must sit near
  `majority_chance`. If it does not, the features encode something label-correlated
  that is not tremor.
- `group_disjoint.disjoint` must be `true`.
- `train_test_separability.domain_auc` above about 0.65 is expected here and is not
  leakage: subject-independent splits genuinely shift the hand-shape distribution.
