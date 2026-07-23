# Documentation index

Start at the [top-level README](../README.md) for the results and the claim
boundaries. This folder holds the connective documentation.

## In this folder

| Document | Read it when |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | You want to know how the modules fit, what data shapes flow between them, and why each design decision was made. |
| [CODE_MAP.md](CODE_MAP.md) | You want every module, class, and function with its signature, purpose, and line number. Auto-generated from the AST by `gen_codemap.py`, so it cannot drift from the code. |
| [REPRODUCE.md](REPRODUCE.md) | You want the exact command behind a specific number, with its expected output and runtime. |
| [DATA.md](DATA.md) | You need the rigged-hand asset, the MediaPipe model, or either real dataset, and their licences. |

## Elsewhere in the repository

| Document | Subject |
|---|---|
| [../RESULTS_REPORT.md](../RESULTS_REPORT.md) | Generated results with the leakage audit and the self-improvement trajectory. |
| [../REALDATA_VALIDATION.md](../REALDATA_VALIDATION.md) | The UCI-395 and PADS validations in full, including what they do not establish. |
| [../DATASETS.md](../DATASETS.md) | Survey of external tremor datasets with verified access status, and a recommended validation ladder. |
| [../REFERENCES.md](../REFERENCES.md) | Verified citations: prior art, clinical scales, sim-to-real, useful repositories. |
| [../IEEE_HEALTHCOM_NOTES.md](../IEEE_HEALTHCOM_NOTES.md) | Paper working notes: framing, related-work differentiation, novelty positioning. |
| [../RENDER_SIDE.md](../RENDER_SIDE.md) | The Blender half: the black-triangle fix, skin-tone randomisation, and the mini50 gate. |
| [../paper/README.md](../paper/README.md) | Building the paper, the figures, and the two layout pitfalls already fixed. |
| [../results/README.md](../results/README.md) | The result JSON schema and how to read it honestly. |
| [../realdata/README.md](../realdata/README.md) | Fetching the two real datasets. |

## Suggested reading order

1. [../README.md](../README.md) — what was built, what it scores, what it does not prove.
2. [ARCHITECTURE.md](ARCHITECTURE.md) — the premise the design rests on, and the data flow.
3. [../RESULTS_REPORT.md](../RESULTS_REPORT.md) and [../REALDATA_VALIDATION.md](../REALDATA_VALIDATION.md) — the evidence.
4. [CODE_MAP.md](CODE_MAP.md) — the code itself.
5. [REPRODUCE.md](REPRODUCE.md) — run it yourself.
