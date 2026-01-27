# Figure 2d: Mean angular-displacement traces

## Data File Required

Place the `pos_vel_by_amp_bins_bundle.pkl` file in this directory. This file should be exported from the `main_sequence_analysis.ipynb` notebook.

## Expected Bundle Structure

Same as Figure 2c, but using position data instead of velocity:
- `animals`: Dictionary keyed by animal ID, each containing:
  - `series`: List of dictionaries, each containing:
    - `pos_center`: Array of position values (mean)
    - `pos_lo`: Array of position values (lower bound, optional)
    - `pos_hi`: Array of position values (upper bound, optional)

## Usage

```bash
python figure_2d.py
```

The script will generate `figure_2d.pdf` in this directory.
