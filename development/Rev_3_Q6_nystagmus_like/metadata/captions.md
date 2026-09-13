# Captions — secondary eye movements after large still-head saccades

These panels quantify follow-up *detected* saccades after large unique gaze events 
that start while the head is still. Labels such as nystagmus, VOR, or corrective 
saccade are not applied.

**Cohort filter:** unique L/R events (paper pairing window) with no `lizMov` movement 
sample in the 50 ms before onset; 
amplitude ≥ 5.5° (P75 
of 7006 still-head events, floored at 5°). 
Follow-ups are existing detections with onset in (t0, t0+250 ms], 
excluding the L/R constituents of the primary event. Reverse = circular direction difference > 90°.

**amplitude_distribution.pdf.** Amplitude distribution of unique gaze events whose head is still at onset (n=7006). The dashed orange line is the large-event threshold (5.5°), set to the cohort 75th percentile rounded down to 0.5° and not below one main-sequence bin (5°).

**event_browser.pdf.** Multi-page event browser. Each page is one qualifying large saccade. Top two traces: left and right eye φ (solid) and θ (dashed), recentered to the pre-onset median. Shading marks primary on–off. Dotted vertical lines mark subsequent detected saccades (green = same-direction, magenta = reverse). Bottom: `lizMov` movement samples and the next head-bout onset (orange). Title carries animal, block, event id, and t_on so the source video can be recovered.

**example_gallery.pdf.** Stratified examples of the same peri-event layout as the browser (reverse / same / none × with / without later head onset).

**secondary_fractions.pdf.** Fraction of large still-head saccades (n=1903) with no subsequent detected eye movement, with a first follow-up in the same direction as the primary, or with a first follow-up that is counter-directed. Inset: ≥2 follow-ups (37.5%) and both directions in the window (21.7%).

**secondary_latency_amp.pdf.** Latency from primary onset and amplitude of the first subsequent detected saccade, split by same-direction vs reverse.

**head_association.pdf.** Fraction of events in each follow-up class that have a head-bout onset in the same post-saccadic window. This is an association, not a mechanistic claim.

**final_position.pdf.** Final primary-eye position at the end of the post window, projected on the primary movement axis and divided by primary amplitude (left). 1 = remains at the new position; 0 = back at the pre-saccadic position. Right: signed displacement vs pre-saccade (degrees). Median remaining fraction = 0.92; 84.2% remain ≥ 0.5 of the primary displacement.

**population_traces.pdf.** Individual peri-event traces aligned to primary onset (t=0), split by first follow-up class. Top: signed eye position along the primary axis. Bottom: head-movement occupancy from `lizMov` samples. Black = median. Traces are not averaged away; at most 250 events per panel are drawn if the class is larger.

**population_traces_by_head.pdf.** Same individual-trace layout, split by whether a head-bout onset occurs in the post-saccadic window.

## Quantitative summary

- Still-head unique events: **7006**
- Qualifying large events: **1903** (threshold 5.5°)
- Any subsequent detected eye movement: **59.0%** (1123/1903)
- First follow-up same-direction: **20.8%**; reverse: **35.9%**; unclassified direction: **2.3%**
- ≥2 follow-ups: **37.5%**; both directions in window: **21.7%**
- Subsequent head-bout onset: **16.4%**; among events with a follow-up, **17.2%** also have later head onset
- Median first-follow-up latency: **68 ms**; amplitude **2.3°**
- Median remaining fraction of primary displacement at window end: **0.92** (84.2% stay ≥ 0.5)
