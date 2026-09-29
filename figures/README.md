# Figure reproduction

Redraw every deposited panel from `S1_Data.xlsx`. This folder is the only place you need for paper figures.

1. Install the environment for your platform as in [docs/install.md](../docs/install.md).
2. Download `S1_Data.xlsx` from the associated dataset ([record](https://zenodo.org/records/23016364), DOI [10.5281/zenodo.23016364](https://doi.org/10.5281/zenodo.23016364)) and place it next to this script. That DOI identifies the dataset, not this software. The same record also contains `Table_S1.xlsx` and `PV_106.zip`; this script uses only `S1_Data.xlsx`.
3. Run:

```bash
python plot_s1_main.py
```

PDFs are written into this folder. Optional: `python plot_s1_main.py PATH/S1_Data.xlsx [OUTDIR]`.

The workbook is the only data file. This script does not import the PETS package and does not read YAML, pickle, or CSV sidecars. Runtime packages: numpy, pandas, matplotlib, scipy, openpyxl.
