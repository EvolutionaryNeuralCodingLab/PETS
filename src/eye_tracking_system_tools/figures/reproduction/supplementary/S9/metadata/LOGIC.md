Four-panel jitter histograms (rigid lizard, modular lizard, modular mouse, modular turtle) sharing one x-limit: the 99.5th percentile of the most jittery mount type. Bar width is per panel: k × the coarsest 1-pixel step in that mount type (integer-pixel correlation peak × µm/px). k is a scalar or a mount dict (lizard covers rigid+modular).

To redraw without the PETS package:
  python replot.py
PDFs land in plots/replot/ (originals in plots/ are not overwritten).
  python replot.py --overwrite   # replace plots/ in place
