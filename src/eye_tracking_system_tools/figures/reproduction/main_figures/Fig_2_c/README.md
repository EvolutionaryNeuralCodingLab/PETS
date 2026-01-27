# Figure 2c: Mean saccade angular speed profiles

## Data File Required

Place the `pos_vel_by_amp_bins_bundle.pkl` file in this directory. This file should be exported from the `main_sequence_analysis.ipynb` notebook.

## Expected Bundle Structure

The pickle file should contain a dictionary with:
- `params`: Dictionary with plotting parameters (fig_size, dpi, lw, etc.)
- `animals`: Dictionary keyed by animal ID, each containing:
  - `series`: List of dictionaries, each containing:
    - `label`: String label for the amplitude bin
    - `color_rgba`: Tuple of (r, g, b, a) color values
    - `vel_center`: Array of velocity values (mean)
    - `vel_lo`: Array of velocity values (lower bound, optional)
    - `vel_hi`: Array of velocity values (upper bound, optional)
    - `time_axis` or `mov_axis`: Array of time values (may need to be reconstructed)

## Usage

```bash
python figure_2c.py
```

The script will generate `figure_2c.pdf` in this directory.
