Binning trials for smoothed epoch durations (same smoothing as epoch_durations_smoothed). Each PDF is quiet full-range | quiet zoomed to ceil(max active) | active on the same zoom. Integer-aligned fixed-width or coarser n_bins schemes; optional KDE overlay.

To redraw without the PETS package:
  python replot.py
PDFs land in plots/replot/ (originals in plots/ are not overwritten).
  python replot.py --overwrite   # replace plots/ in place
