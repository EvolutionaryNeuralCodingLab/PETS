# Tag, GitHub release, and Zenodo

The publication snapshot is commit `f6d2187` on branch `plos-biology`.
Creating the annotated tag was not run from this session (release tagging
needs an explicit local approval). From the repo root:

```bash
git checkout plos-biology
git tag -a v1.0.0 -m "PLOS Biology publication snapshot"
git push -u origin plos-biology
git push origin v1.0.0
gh release create v1.0.0 --title "v1.0.0 — PLOS Biology source code and S1 Data" \
  --notes "Software, hardware files, and S1_Data.xlsx for the PLOS Biology article." \
  publication/S1_Data.xlsx
```

Then archive the GitHub release to Zenodo (GitHub–Zenodo webhook, or upload
the tagged zip at https://zenodo.org). Insert the DOI into:

- manuscript Data Availability (copy: `~/Desktop/PLOS_tech_rev/PLOS_biology_Manuscript_PLOS_edits.docx`)
- `publication/MANUSCRIPT_EDITS.md`
- README Zenodo sentence

`.zenodo.json` and `CITATION.cff` are already in the tree. Confirm MIT + CC BY 4.0
if the institution requires a different license. Add ORCIDs to CITATION.cff
before the freeze if desired.
