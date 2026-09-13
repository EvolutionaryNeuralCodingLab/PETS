# Captions — reverse follow-up after large saccades (head still vs moving)

Operational rule for the video-S1 ‘back-and-forth’: a **large** unique gaze event 
(amplitude ≥ 7.0°) is followed within **100 ms** by another 
**detected** saccade whose direction differs by **> 90°** 
(approximately opposite). Concurrent L/R onsets within the paper pairing window count as one event. 
Head stationary vs moving is the existing `lizMov.mat` annotation: a movement sample overlapping 
the primary saccade on–off span.

**amplitude_distribution.pdf.** Amplitude of unique gaze events on blocks with `lizMov.mat`. 
Large-event threshold 7.0° = cohort P75 rounded down to 0.5°, not below 5°.

**reverse_followup_overall.pdf.** Frequency of the reverse-follow-up rule among large saccades: 
**41.5%** (1834/4416).

**reverse_followup_by_head.pdf.** Same rule, split by lizard-movement annotation. Head stationary: 
**27.9%** (n=1419). Head moving: 
**48.0%** (n=2997). Two-sided Fisher p=1.93e-37. 
The intended reading is that the back-and-forth is nystagmus-like stabilization during head movement.

**reverse_followup_by_head_animal.pdf.** Per-animal rates, same split.

## Quantitative answer

- Large saccades with a reverse follow-up within 100 ms: **41.5%** (1834 of 4416)
- Head stationary: **27.9%** (396/1419)
- Head moving: **48.0%** (1438/2997)
- Median reverse follow-up latency / amplitude: **50 ms** / **4.2°**
