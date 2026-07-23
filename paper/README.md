# Paper

The IEEE Healthcom submission, built from the numbers in `../results/`.

## Which file to use

| File | Figures | Use when |
|---|---|---|
| **`tremor_twin_selfcontained.tex`** | Inline `pgfplots`, no external files | **Default.** Paste into Overleaf and compile. Nothing else to upload. |
| `tremor_twin_healthcom.tex` | `\includegraphics` from `figs/*.pdf` | You want vector image files, e.g. to restyle a figure separately. Requires the `figs/` folder next to the `.tex`. |

Both carry identical text and identical numbers. The self-contained version exists
because the external-figure version renders blank figures if the `figs/` folder is
not uploaded alongside it.

## Building

```bash
pdflatex tremor_twin_selfcontained.tex
pdflatex tremor_twin_selfcontained.tex     # twice, so figure and table refs resolve
```

Or upload to Overleaf and compile. The document class is `IEEEtran` with the
`conference` option, which already omits page numbers, headers, and footers as
Healthcom requires. Do not add any.

## Figures

`figs/*.pdf` are generated from the real model and real datasets, never mocked up.
Regenerate with:

```bash
cd .. && python make_figures.py     # rewrites paper/figs/*.pdf
cd .. && python make_pgf.py         # rewrites tremor_twin_selfcontained.tex
```

| Figure | Content | Source |
|---|---|---|
| `fig_twin.pdf` | The 21-joint twin hand, and an index-fingertip tremor trace | `twin.py` forward kinematics |
| `fig_collinearity.pdf` | Coupled labels as a line (r = −1.00) beside decoupled labels as a cloud (r = −0.23) | `generator.py`, both modes |
| `fig_freq.pdf` | Predicted vs true frequency on held-out subjects with the ±1 Hz band | the exported model in `results/models/` |
| `fig_real.pdf` | UCI-395 drift correction, and PADS frequency by diagnostic group | `realdata_uci395.py`, `pads_validation.py` |

Note that `fig_freq` reports about 93% within 1 Hz because it uses the exported
3-second model, while Table I reports 95.8% for the 4-second configuration. The
caption says so; this is not an inconsistency.

## Known layout pitfalls

Both were hit and fixed; if you edit the paper, watch for them returning.

- **The results table overflowing the column.** Six columns do not fit IEEE's
  ~3.5 inch column, and the overflow prints on top of the right-hand column. The
  table is wrapped in `\resizebox{\columnwidth}{!}{...}`. Keep that.
- **A wide figure splitting the reference list.** `figure*` floats can drift to the
  top of the references page and cut the bibliography in half. `stfloats` is loaded
  and the real-data figure is placed `[!b]`. If it still happens, put `\clearpage`
  immediately before `\begin{thebibliography}`.

## Submission

`EDAS_SUBMISSION.md` holds the exact field values for the EDAS form: title,
keywords, the plain-text abstract, and the topic selections. Six-page limit; a
seventh page costs USD 100.
