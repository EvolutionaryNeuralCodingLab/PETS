Reviewer Q6: subsequent *detected* eye movements after large unique gaze events that start while the head is still. Concurrent L/R onsets within binocular.sync_diff_ms collapse to one event (amplitude = max net_angular_disp). Head-still at onset = no lizMov sample in the lookback window before t_on (not the during-saccade head_movement flag). Large = cohort P75 of those amplitudes, rounded down to 0.5°, floored at 5°. Follow-ups are existing saccade-table rows with onset in (t0, t0+post_window], excluding the primary L/R constituents. Reverse = circular direction difference > 90°. Head association uses the next lizMov bout onset in the same window. Final position is primary-eye displacement along the primary axis at t0+post_window, divided by primary amplitude. Does not label events as nystagmus, VOR, or corrective saccades.
Replot redraws summary PDFs from secondary_after_saccade.pkl; the event browser needs the live export (eye CSVs + lizMov).

To redraw without the PETS package:
  python replot.py
PDFs land in plots/replot/ (originals in plots/ are not overwritten).
  python replot.py --overwrite   # replace plots/ in place
