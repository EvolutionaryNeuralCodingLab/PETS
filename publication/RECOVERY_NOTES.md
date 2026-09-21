# Source-data recovery notes (requested panels)

## Resolved mappings (submitted PDFs)

- Fig 3D = pupil histograms (repo Fig_3_e); Fig 3E = z-score diffs (repo Fig_3_f, n=4); Fig 3F = ISI (repo Fig_3_d).
- S8E = lizard vs mouse coupling (repo S8j).
- S11A = ISI by state; S11B = epoch durations n=174/191.
- S12A = example traces; S12B = 21.3% event table.
- Fig 1E (median ~31 µm, n=454300) is not the S9 rigid pool (median ~42 µm, n=1.47e6).
- S10: manuscript/PDF SNR = threshold/median(D); pickle stored threshold/(2B). Turtle pickle has no threshold; manuscript 2.0°/frame.
- Fig 2E n=34876 vs S13 n=33294 vs Fig 2J n=28146 are different event definitions.

## Recovered

- S2 per-animal spans from Paper_Figures pickle (35.0±7.3 / 26.1±6.4).
- S4 n=5 pupil-correlation bars (time-weighted 66.3 / 23.4 / 10.3).
- S5 literature + Pogona bars (Human 90/75, Pogona 35/26, Mouse 20/20, Cormorant 18/16, Barn owl 4/3).
- S11B individual epoch durations (match PDF n).
- Fig 2I individual saccade angles.
- S12A example window from the Fig 3C vignette around event PV_126_007_297867_R.

## Not frozen (documented on the matching S1_Data sheets)

- Fig 2C–D per-saccade aligned traces (means + n stored).
- Fig 2E n=34876 points (OLS + bin means stored; closest event-cache table n=34336 does not match OLS).
- Fig 2G per-saccade amplitudes (mean±SEM + five animal difference curves stored).
- Fig S3 L/R pairs (99×99 counts stored).
- Fig 3F / S11A raw ISI lists (plotted densities stored).
- Fig 3D pupil animal IDs on pooled samples.

Re-running `revisions_messy` `main_sequence_analysis.ipynb` on the lab experiment mount is the only identified path to the exact 2C/2E individuals (PV_106 n=6954, global n=34876). That was not executed here (would re-detect events; analysis must not be altered).
