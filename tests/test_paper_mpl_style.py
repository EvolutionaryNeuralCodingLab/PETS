"""Paper matplotlib style: Type-42 embedding for Illustrator."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eye_tracking_system_tools.analysis.paper_mpl_style import apply_paper_style
from eye_tracking_system_tools.analysis.plot_bundle import write_replot_script


def test_apply_paper_style_sets_type42():
    apply_paper_style()
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42
    assert "Arial" in plt.rcParams["font.sans-serif"]


def test_replot_script_embeds_type42(tmp_path: Path):
    path = write_replot_script(tmp_path, "figure_s3")
    text = path.read_text(encoding="utf-8")
    assert 'plt.rcParams["pdf.fonttype"] = 42' in text
    assert 'plt.rcParams["ps.fonttype"] = 42' in text
    assert "Arial" in text
    assert "figsize=(3.0, 1.7)" in text or "figsize=(3, 1.7)" in text
    compile(text, str(path), "exec")


def test_2e_export_accepts_s13_preonset_flag():
    import inspect

    from eye_tracking_system_tools.analysis.figures_2c_2e import export_amplitude_velocity_fit

    assert "use_s13_preonset" in inspect.signature(export_amplitude_velocity_fit).parameters
