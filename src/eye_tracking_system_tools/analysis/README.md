# Analysis

Saccade detection runs inside the Preprocessing GUI Saccades tab. That tab and the Calibration tab import `saccade_export`, `param_tune`, `eye_trace_io`, and `pixel_calibration`. Defaults are the `saccade` and `binocular` sections of [`configs/analysis_params.yaml`](../../../configs/analysis_params.yaml).

Paper figure PDFs are not built from this package. Redraw them with [`figures/plot_s1_main.py`](../../../figures/plot_s1_main.py) and `S1_Data.xlsx` from the associated dataset ([record](https://zenodo.org/records/23016364), DOI [10.5281/zenodo.23016364](https://doi.org/10.5281/zenodo.23016364); see [figures/README.md](../../../figures/README.md)). That DOI identifies the dataset, not this software. The example recording is `PV_106.zip` in the same dataset.

Review detected events with the saccade viewer. Pass a YAML registry of block folders:

```bash
python -m eye_tracking_system_tools.analysis.saccade_viewer --registry PATH.yaml
```

```yaml
animals:
  animal_id:
    - /path/to/animal_id/YYYY_MM_DD/block_001
```

Each entry is a block path, or a mapping `{path, left_eye_csv, right_eye_csv}` when a specific trace file should be used.
