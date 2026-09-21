# Manuscript / submission edits (authors apply to the .docx)

Do not copy the manuscript into the git repository.

## Required

1. **Title** (manuscript + submission system), editor-requested:
   An open-source, adaptable eye-tracking system enables studies of visual acquisition across diverse terrestrial vertebrates

2. **Pupil/state n:** change “5 animals” / n=5 to **n=4** wherever it refers to the state-dependent pupil analysis (Fig 3D–E). Animals: PV_106, PV_143, PV_62, PV_126. Do not add PV_57. Fig S4 remains n=5.

3. **Figure legends** for editor-requested panels: add “Numerical values in S1 Data, sheet &lt;Fig…&gt;.”

4. **S1 Data SI legend** (Supporting information):
   S1 Data. Excel workbook with one worksheet per requested panel (Fig1E, Fig2B–G, Fig2I–J, Fig3A–F, FigS1A–B, FigS2–S5, FigS8E, FigS9, FigS10A–C, FigS11A–B, FigS12A–B, FigS13A–E). The README sheet maps repo folder letters where they differ from the published panels and notes any summary-only tables.

5. **Data Availability** (replace GitHub-only + “upon request” as the primary statement):
   Source data for the requested figure panels are in Supporting Information file S1 Data. Analysis code, hardware files, and the tagged software release are archived at Zenodo (DOI: &lt;to be inserted after deposit&gt;) and developed at https://github.com/EvolutionaryNeuralCodingLab/PETS.

6. **Caption clarifications already confirmed against the PDFs** (no panel-letter swap needed in the manuscript legends):
   - Fig 3 D/E/F in the legend already match the submitted PDF (D pupil, E z-score, F ISI). Repo folders use a different order.
   - S8E is the coupling plot (not a still).
   - S11 A/B and S12 A/B in Document S2 already match the submitted PDFs.
   - Fig 1E and Fig S9 are different jitter samples (medians ~31 µm vs rigid ~42 µm); say so if a reader could conflate them.
   - S10 SNR in the manuscript/PDF is threshold/median(D). Keep that definition; the repo pickle used threshold/(2B).

7. **Editor point F (n≤5, plot individual points):** S2 already shows per-animal points; S4 is already per-animal stacked bars; S13E already shows per-animal slopes. Remaining bar-only n≤5 artwork (if any, e.g. pooled S1B) is a figure-design change, not a source-data gap. Depositing S1 Data does not by itself restyle the PDFs.

## Optional / not done here

- ORCIDs on the Zenodo record.
- Confirm MIT + CC-BY-4.0 if the institution requires a different license.
