# PETS: Personalized Eye Tracking System

Release **V1.0.0** (Python package version `1.0.0`).

PETS is open-source eye-tracking software for small freely moving animals, demonstrated in lizards, mice, and turtles. This repository is the software.

The associated manuscript is *An open-source eye-tracking system adaptable to diverse terrestrial vertebrates reveals state-dependent visual acquisition in lizards*.

Source repository: <https://github.com/EvolutionaryNeuralCodingLab/PETS>

## Start here

1. **Install** — [docs/install.md](docs/install.md) (Windows / macOS / Linux conda envs).
2. **Reproduce paper plots** — [figures/README.md](figures/README.md). Download `S1_Data.xlsx` from the associated dataset, place it next to `figures/plot_s1_main.py`, and run `python plot_s1_main.py`. Sheet index: [docs/source_data.md](docs/source_data.md).
3. **Preprocess acquired data** — [docs/preprocessing.md](docs/preprocessing.md). The Preprocessing GUI is the go-to tool (`python -m eye_tracking_system_tools.annotation.preprocessing_gui`). The worked example is `PV_106 / 2025_09_04 / block_015` inside `PV_106.zip` in the associated dataset.
4. **Review a preprocessed block** — [docs/annotation.md](docs/annotation.md) (Block Annotator and Event Explorer).
5. **Acquire video on a Raspberry Pi** — [docs/acquisition.md](docs/acquisition.md).
6. **Manuscript statistics** — `statistics/` runs the paper's permutation tests. They need a pickle of at least two already-processed blocks. They do not read `S1_Data.xlsx` or the example block. Settings: [statistics/params.yaml](statistics/params.yaml).

## Associated dataset

The dataset is archived separately from this software:

<https://zenodo.org/records/23016364>

DOI: <https://doi.org/10.5281/zenodo.23016364>

That DOI identifies the **dataset**. It is not the DOI of this software. Zenodo assigns a software DOI when a Git release of this repository is archived. Until then, cite the software with [CITATION.cff](CITATION.cff).

The dataset contains:

- `S1_Data.xlsx` — numerical source data for the figure panels
- `Table_S1.xlsx` — comparison with previously published eye-tracking systems
- `README.md` — dataset README
- `PV_106.zip` — the example recording `PV_106 / 2025_09_04 / block_015`, including videos and Open Ephys files

## License and citation

Software is [MIT](LICENSE). Printable hardware files in this repository, and the associated dataset, are [CC BY 4.0](LICENSE-DATA.md).
