# Supplementary figure reproduction

Each folder is a frozen plot bundle: `metadata/` (pickle + yaml), `plots/`
(current PDFs), `replot.py` (drawing code), and `figure_S*.py` (the public
entry point). Scripts read only files in that folder. They do **not** open
recording blocks or Jupyter notebooks.

```bash
cd src/eye_tracking_system_tools/figures/reproduction/supplementary/S3
python figure_S3.py
```

PDFs are written into `plots/` (`--overwrite` is the default for `figure_S*.py`).

| Folder | Script | Regenerates from pickle | Static (shipped, not replotted) |
|---|---|---|---|
| S1 | `figure_S1.py` / `figure_S1a.py` | S1a simulation curve, S1b/S1c Kerr bars | — |
| S3 | `figure_S3.py` | S3a–c | — |
| S8 | `figure_S8.py` | traces, mouse 2c/2d/2e, S8j | S8a/c/e eye stills |
| S9 | `figure_S9.py` | S9a unified jitter | — |
| S10 | `figure_S10.py` | S10a–c Rayleigh | — |
| S11 | `figure_S11.py` | epoch triptychs, ISI, raw epochs | — |
| S12 | `figure_S12.py` | S12a overall | S12b example trace |
| S13 | `figure_S13.py` | S13a–f | — |

S2 and S4–S7 are not in this tree yet (no frozen pickle bundles).
