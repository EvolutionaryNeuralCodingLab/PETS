# PETS

Open-source eye tracking for small freely moving animals: personalized 3D-printed headmounts, Raspberry Pi cameras, and analysis code for lizards, mice, and turtles.

This repository accompanies *An open-source, adaptable eye-tracking system enables studies of visual acquisition across diverse terrestrial vertebrates*.

## Start here

1. **Install** — [docs/install.md](docs/install.md) (Windows / macOS / Linux conda envs).
2. **Reproduce paper plots** — [figures/README.md](figures/README.md). Download `S1_Data.xlsx`, place it next to `figures/plot_s1_main.py`, run `python plot_s1_main.py`. Sheet index: [docs/source_data.md](docs/source_data.md).
3. **Preprocess acquired data** — [docs/preprocessing.md](docs/preprocessing.md). The Preprocessing GUI is the go-to tool (`python -m eye_tracking_system_tools.annotation.preprocessing_gui`). Worked example: `PV_106 / 2025_09_04 / block_015`.
4. **Review a preprocessed block** — [docs/annotation.md](docs/annotation.md) (Block Annotator and Event Explorer).
5. **Acquire video on a Raspberry Pi** — [docs/acquisition.md](docs/acquisition.md).

Software is [MIT](LICENSE). Data and printable hardware files are [CC BY 4.0](LICENSE-DATA.md). Cite this archive with [CITATION.cff](CITATION.cff).
