# Figure 2e: Relationship between saccade amplitude and peak angular speed

## Data File Required

Place the `amplitude_velocity_linear_fit_bundle.pkl` file in this directory. This file should be exported from the `main_sequence_analysis.ipynb` notebook.

## Expected Bundle Structure

The pickle file should contain a dictionary with:
- `params`: Dictionary with plotting parameters
- `edges`: Array of bin edges for amplitude bins
- `per_animal_stats`: Dictionary keyed by animal ID, each containing a DataFrame with columns:
  - `amp_lo`, `amp_hi`: Amplitude bin boundaries
  - `n`: Number of saccades in bin
  - `mean_peak_v`: Mean peak velocity
- `global_fit`: Dictionary with linear fit parameters (slope, intercept, etc.)
- `linear_stats_df`: DataFrame with linear fit statistics

## Usage

```bash
python figure_2e.py
```

The script will generate `figure_2e.pdf` in this directory.
