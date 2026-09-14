Quiet-segment pooled inter-frame Δangle histogram from one block. Overlay: Rayleigh(B_med), vertical 2B (2 axis-σ noise marker), and detector threshold when set. SNR = threshold / (2 * B_med).
Replot reads rayleigh_noise_core.pkl; it does not reload Kerr traces.

To redraw without the PETS package:
  python replot.py
PDFs land in plots/replot/ (originals in plots/ are not overwritten).
  python replot.py --overwrite   # replace plots/ in place
