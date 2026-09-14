#!/usr/bin/env python3
"""S11 epoch-duration triptychs, ISI histograms, and raw epoch hists from pickles.

Does not read recording blocks.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from _run import run_replot

PLOTS = ROOT / "plots"


def _copy_if(src: Path, dest: Path) -> None:
    if src.is_file():
        shutil.copy2(src, dest)


if __name__ == "__main__":
    run_replot(ROOT, "epoch_duration_bin_trials", "epoch_duration_bin_trials.pkl")
    _copy_if(PLOTS / "epoch_duration_triptych_width5s_quiet50s.pdf", PLOTS / "S11a_triptych.pdf")
    for p in sorted(PLOTS.glob("epoch_duration_triptych_*kde*.pdf")):
        shutil.copy2(p, PLOTS / "S11a_triptych_kde.pdf")

    run_replot(ROOT, "isi_plotdata", "ISI_ISI_histogram_plotdata.pickle")
    _copy_if(PLOTS / "ISI_histogram.pdf", PLOTS / "S11b_ISI_histogram.pdf")
    _copy_if(PLOTS / "legend_ISI_histogram.pdf", PLOTS / "S11b_animal_legend.pdf")

    run_replot(ROOT, "epoch_durations", "epoch_durations_raw.pkl")
