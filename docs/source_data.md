# S1 Data

Download `S1_Data.xlsx` from the associated dataset ([record](https://zenodo.org/records/23016364), DOI [10.5281/zenodo.23016364](https://doi.org/10.5281/zenodo.23016364)) and place it next to [`figures/plot_s1_main.py`](../figures/plot_s1_main.py). That DOI identifies the dataset, not this software. The same record also contains `Table_S1.xlsx` and `PV_106.zip` (the example block `PV_106 / 2025_09_04 / block_015`). This page indexes `S1_Data.xlsx` only. See [figures/README.md](../figures/README.md).

Column A is a 0-based row index. Titles start in column B. The plotter drops column A and redraws each panel from these sheets.

| Sheet | One row | Panel |
|---|---|---|
| Figure 1E | one camera-jitter displacement (µm) | 1E |
| Figure 2B | one eye-position sample in the 240–320 s window | 2B |
| Figure 2C | one aligned speed or position sample | 2C and 2D |
| Figure 2E | one saccade: animal, amplitude, peak velocity | 2E |
| Figure 2F | one angle pair | 2F |
| Figure 2G | one saccade (`animal`, `pairing`, `magnitude_raw_angular`, `net_angular_disp`) | 2G |
| Figure 2I | one rotated angle | 2I |
| Figure 2J | one animal | 2J |
| Figure 3A | one eye-position sample, 210–240 s | 3A |
| Figure 3B | one eye-position sample, 310–340 s | 3B |
| Figure 3C | one time sample, including saccade, head, and state columns | 3C |
| Figure 3D | one pupil diameter | 3D |
| Figure 3E | one z-scored pupil sample. Columns continue in `*_part2` and `*_part3` | 3E |
| Figure 3F | one inter-saccade interval (`animal`, `isi_ms`) | 3F |
| Figure S1A | one calibration sample | S1A |
| Figure S1B | one block’s pooled leftover error | S1B |
| Figure S2 | one eye’s pooled span (`animal`, `eye`, `main_span_deg`, `perp_span_deg`) | S2 |
| Figure S3 | one event | S3 |
| Figure S4 | one 10 s rolling L/R pupil correlation | S4 |
| Figure S5 | one species’ horizontal and vertical span, plus a `source` column | S5 |
| Figure S8A | one frame of φ or θ for lizard, turtle, or mouse | S8A |
| Figure S8B | one aligned sample of one isolated mouse saccade | S8B |
| Figure S8C | the same rows as Figure S8B; the plot uses `position_deg` | S8C |
| Figure S8D | one mouse saccade | S8D |
| Figure S8E | one joint-density sample | S8E |
| Figure S9 | one jitter sample. Long columns continue in `*_part` columns | S9 |
| Figure S10A, S10B, S10C | one noise or Rayleigh sample | S10A–C |
| Figure S11A | one same-epoch interval with a behavioral state | S11A |
| Figure S11B | one epoch duration | S11B |
| Figure S12A | one φ/θ sample in the short 3C window | S12A |
| Figure S12B | one large saccade, including `has_back_and_forth` | S12B |
| Figure S13A | one labelled saccade | S13A, and S13B, S13C, S13E derived from it |
| Figure S13D | one unlabelled pooled saccade | S13D |
