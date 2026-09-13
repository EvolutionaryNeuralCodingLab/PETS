"""Copy AGG PDFs to material_agg_pdf_corrected_FINAL with paper Type-42 style.

Does not touch outputs/material_aggregation_for_PDF_final. Does not run the
aggregator main() (that rmtree's AGG). Replots from existing pickles only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
AGG = REPO / "outputs" / "material_aggregation_for_PDF_final"
DEST = REPO / "outputs" / "material_agg_pdf_corrected_FINAL"
PYTHON = sys.executable


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    return env


def _copy_if(src: Path, dest: Path) -> None:
    if src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _rename_plots(plots: Path, mapping: dict[str, str]) -> None:
    plots.mkdir(parents=True, exist_ok=True)
    for src_name, dest_name in mapping.items():
        src = plots / src_name
        if src.is_file():
            shutil.copy2(src, plots / dest_name)


def run_replot(bundle: Path, kind: str, pickle_name: str | None = None) -> None:
    from eye_tracking_system_tools.analysis.plot_bundle import write_replot_script

    write_replot_script(bundle, kind)
    cmd = [PYTHON, "replot.py", "--overwrite"]
    if pickle_name:
        cmd.extend(["--pickle", pickle_name])
    subprocess.run(cmd, cwd=bundle, check=True, env=_env())


RAW_EPOCHS = REPO / "outputs" / "replotting_standalone" / "epoch_durations_raw"


def export_s11_raw(s11: Path) -> None:
    """Unsmoothed epoch-duration histograms into ``S11/plots/`` (does not touch S11a)."""
    import pickle

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from eye_tracking_system_tools.analysis.paper_mpl_style import apply_paper_style

    src_pkl = RAW_EPOCHS / "metadata" / "epoch_durations.pkl"
    if not src_pkl.is_file():
        raise FileNotFoundError(src_pkl)
    plots = s11 / "plots"
    meta = s11 / "metadata"
    plots.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    dest_pkl = meta / "epoch_durations_raw.pkl"
    shutil.copy2(src_pkl, dest_pkl)
    src_yaml = RAW_EPOCHS / "metadata" / "epoch_durations.pkl.meta.yaml"
    if src_yaml.is_file():
        shutil.copy2(src_yaml, meta / "epoch_durations_raw.pkl.meta.yaml")

    data = pickle.loads(dest_pkl.read_bytes())
    apply_paper_style()
    stored = data.get("bins") or {}
    n_bins = int(data.get("n_bins") or 30)
    xmin = data.get("linear_xmin_s")
    xmin_f = 0.0 if xmin is None else float(xmin)

    def _bins(lab: str, scale: str, vals: np.ndarray) -> np.ndarray:
        body = stored.get(lab) if isinstance(stored, dict) else None
        if isinstance(body, dict) and scale in body:
            edges = np.asarray(body[scale], dtype=float)
            if edges.size >= 2:
                return edges
        pos = vals[np.isfinite(vals) & (vals > 0)]
        n_edges = n_bins + 1
        if pos.size == 0:
            return np.logspace(-2, 1, n_edges) if scale == "log" else np.linspace(0.0, 1.0, n_edges)
        hi = float(np.nanmax(pos))
        if scale == "log":
            lo = max(float(np.nanmin(pos)), 1e-2)
            if hi <= lo:
                hi = lo * 10.0
            return np.logspace(np.log10(lo), np.log10(hi), n_edges)
        return np.linspace(xmin_f, hi if hi > xmin_f else xmin_f + 1.0, n_edges)

    for lab, color in (("active", "#D55E00"), ("quiet", "#0072B2")):
        vals = np.asarray(data.get(lab, []), dtype=float)
        for scale in ("log", "linear"):
            fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
            edges = _bins(lab, scale, vals)
            if vals.size:
                ax.hist(vals, bins=edges, color=color, edgecolor="black", linewidth=0.4, alpha=1.0)
                if scale == "log":
                    ax.set_xscale("log")
                else:
                    ax.set_xlim(xmin_f, float(edges[-1]))
            ax.set_xlabel("Epoch duration [s]")
            ax.set_ylabel("Count")
            ax.set_title(f"{lab} n={vals.size} ({scale}, raw)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(plots / f"S11_raw_{lab}_{scale}.pdf", format="pdf", bbox_inches="tight")
            plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2), dpi=300, sharey=True)
    for ax, lab, color in (
        (axes[0], "active", "#D55E00"),
        (axes[1], "quiet", "#0072B2"),
    ):
        vals = np.asarray(data.get(lab, []), dtype=float)
        edges = _bins(lab, "log", vals)
        if vals.size:
            ax.hist(vals, bins=edges, color=color, edgecolor="black", linewidth=0.4, alpha=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("Epoch duration [s]")
        ax.set_title(f"{lab} n={vals.size}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plots / "S11_raw_log.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    leftover = s11 / "raw"
    if leftover.is_dir():
        shutil.rmtree(leftover)


def _prune_plots(dest: Path, keep: list[str]) -> None:
    """Keep only Illustrator display files in dest/plots/."""
    _prune_dir(dest / "plots", keep)


def _prune_dir(folder: Path, keep: list[str]) -> None:
    if not folder.is_dir():
        return
    replot_dir = folder / "replot"
    if replot_dir.is_dir():
        shutil.rmtree(replot_dir)
    allowed = set(keep)
    for p in list(folder.iterdir()):
        if p.is_dir():
            continue
        if p.name not in allowed:
            p.unlink()


def _clear_plots(bundle: Path) -> None:
    plots = bundle / "plots"
    if plots.is_dir():
        shutil.rmtree(plots)
        plots.mkdir()


def load_mouse_323_tables(*, keep_traces: bool = True, force: bool = False):
    from eye_tracking_system_tools.analysis.block_registry import load_registry
    from eye_tracking_system_tools.analysis.event_cache import build_or_load_event_tables
    from eye_tracking_system_tools.analysis.export_meta import load_params_yaml

    params = load_params_yaml(REPO / "configs" / "analysis_params_mouse.yaml")
    specs = load_registry(REPO / "configs" / "mouse_M_002_blocks.yaml")
    cache = REPO / "outputs" / "review_answers_latest" / "metadata_mouse_323"
    cache.mkdir(parents=True, exist_ok=True)
    return build_or_load_event_tables(
        specs,
        params,
        cache,
        keep_traces=keep_traces,
        force=force,
        prefer_finalized=False,
    )


def export_mouse_s8i_preonset(s8: Path) -> str:
    """S8i: mouse Fig 2e with S13 mixed amplitude (length-1 uses pre-onset)."""
    from eye_tracking_system_tools.analysis.figures_2c_2e import export_amplitude_velocity_fit
    from eye_tracking_system_tools.analysis.paper_mpl_style import apply_paper_style
    from eye_tracking_system_tools.analysis.plot_bundle import write_replot_script

    apply_paper_style()
    tables, cache, from_cache = load_mouse_323_tables(keep_traces=True)
    e2 = s8 / "mouse_figure_2e"
    e2.mkdir(parents=True, exist_ok=True)
    pkl = export_amplitude_velocity_fit(tables, e2, show=False, use_s13_preonset=True)
    write_replot_script(e2, "figure_2e")
    src = e2 / "plots" / "figure_2e.pdf"
    if not src.is_file():
        raise FileNotFoundError(f"mouse 2e PDF missing: {src}")
    shutil.copy2(src, s8 / "plots" / "S8i_figure_2e.pdf")
    return (
        f"S8i pre-onset A from {pkl} "
        f"(cache {getattr(cache, 'name', cache)} from_cache={from_cache})"
    )


def export_corrected() -> int:
    if not AGG.is_dir():
        raise SystemExit(f"AGG not found: {AGG}")
    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(AGG, DEST)

    sys.path.insert(0, str(SRC))
    from eye_tracking_system_tools.analysis.paper_mpl_style import apply_paper_style

    apply_paper_style()

    # S1: Kerr bars; keep S1a crop
    s1 = DEST / "S1"
    run_replot(s1, "kerr_component_error")
    _rename_plots(
        s1 / "plots",
        {
            "cohort_component_error_across_animals.pdf": "S1b.pdf",
            "cohort_component_error_overall_phi_theta.pdf": "S1c.pdf",
        },
    )

    # S3
    s3 = DEST / "S3"
    run_replot(s3, "figure_s3")
    _rename_plots(
        s3 / "plots",
        {
            "figure_S3_head_still.pdf": "S3a_head_still.pdf",
            "figure_S3_head_moving.pdf": "S3b_head_moving.pdf",
            "figure_S3_colorbar.pdf": "S3c_colorbar.pdf",
        },
    )

    # S8 traces (three pickles in one metadata folder)
    s8 = DEST / "S8"
    for species, letter in (("lizard", "b"), ("mouse", "d"), ("turtle", "f")):
        pkl = f"trace_{species}_series.pkl"
        run_replot(s8, "species_trace", pickle_name=pkl)
        _rename_plots(
            s8 / "plots",
            {f"trace_{species}.pdf": f"S8{letter}_trace_{species}.pdf"},
        )

    iso = s8 / "mouse_2c_2d_isolated"
    if (iso / "metadata").is_dir():
        run_replot(iso, "pos_vel")
        for src, dest_name in (
            ("figure_2c_isolated.pdf", "S8g_figure_2c_isolated.pdf"),
            ("figure_2d_isolated.pdf", "S8h_figure_2d_isolated.pdf"),
        ):
            src_pdf = iso / "plots" / src
            if not src_pdf.is_file():
                src_pdf = iso / "plots" / src.replace("_isolated", "")
            _copy_if(src_pdf, s8 / "plots" / dest_name)
        _clear_plots(iso)

    s8i_note = export_mouse_s8i_preonset(s8)
    _clear_plots(s8 / "mouse_figure_2e")
    src_txt = s8 / "SOURCE.txt"
    prev = src_txt.read_text(encoding="utf-8") if src_txt.is_file() else ""
    src_txt.write_text(prev.rstrip() + f"\nS8i: {s8i_note}\n", encoding="utf-8")
    cap = s8 / "captions.md"
    if cap.is_file():
        text = cap.read_text(encoding="utf-8")
        old = (
            "**(i)** Mouse amplitude–peak-speed (Fig. 2e analogue), same 3.23 °/frame events."
        )
        new = (
            "**(i)** Mouse amplitude–peak-speed (Fig. 2e analogue), same 3.23 °/frame events. "
            "Amplitude is onset→offset except length-1 events (pre-onset→offset), matching S13."
        )
        if old in text:
            cap.write_text(text.replace(old, new), encoding="utf-8")

    s8j = s8 / "S8j"
    run_replot(s8j, "figure_2f_lizard_mouse")
    joint = s8j / "joint colorbar"
    separate = s8j / "separate colorbar"
    joint.mkdir(parents=True, exist_ok=True)
    separate.mkdir(parents=True, exist_ok=True)
    plots = s8 / "plots"
    s8j_plots = s8j / "plots"
    for name in (
        "S8j_lizard_mouse_all_2f.pdf",
        "S8j_lizard_mouse_zoom_2f.pdf",
        "S8j_colorbar.pdf",
        "S8j_colorbar_zoom.pdf",
    ):
        if (s8j_plots / name).is_file():
            shutil.copy2(s8j_plots / name, joint / name)
    for name in (
        "S8j_lizard_mouse_all_2f_separate.pdf",
        "S8j_lizard_mouse_zoom_2f_separate.pdf",
        "S8j_colorbar_lizard.pdf",
        "S8j_colorbar_mouse.pdf",
        "S8j_colorbar_lizard_zoom.pdf",
        "S8j_colorbar_mouse_zoom.pdf",
    ):
        if (s8j_plots / name).is_file():
            shutil.copy2(s8j_plots / name, separate / name)
    _clear_plots(s8j)

    # S9
    s9 = DEST / "S9"
    run_replot(s9, "unified_jitter")
    _rename_plots(
        s9 / "plots",
        {"unified_jitter_quantification.pdf": "S9a_unified_jitter.pdf"},
    )

    # S10: one pickle per species
    s10 = DEST / "S10"
    for pkl_name, dest_pdf in (
        ("S10a_lizard_rayleigh_noise_core.pkl", "S10a_lizard.pdf"),
        ("S10b_mouse_rayleigh_noise_core.pkl", "S10b_mouse.pdf"),
        ("S10c_turtle_rayleigh_noise_core.pkl", "S10c_turtle.pdf"),
    ):
        run_replot(s10, "rayleigh_noise_core", pickle_name=pkl_name)
        _rename_plots(s10 / "plots", {"rayleigh_noise_core.pdf": dest_pdf})

    # S11 epoch triptychs + ISI histogram/legend from plotdata
    s11 = DEST / "S11"
    run_replot(s11, "epoch_duration_bin_trials", pickle_name="epoch_duration_bin_trials.pkl")
    _rename_plots(
        s11 / "plots",
        {
            "epoch_duration_triptych_width5s_quiet50s.pdf": "S11a_triptych.pdf",
            "epoch_duration_triptych_kde_width5s_quiet50s.pdf": "S11a_triptych_kde.pdf",
        },
    )
    # trial name may be width5s_quiet50s with kde in a different trial
    for p in (s11 / "plots").glob("epoch_duration_triptych_*kde*.pdf"):
        shutil.copy2(p, s11 / "plots" / "S11a_triptych_kde.pdf")
    hist_pkl = s11 / "metadata" / "ISI_ISI_histogram_plotdata.pickle"
    if hist_pkl.is_file():
        run_replot(s11, "isi_plotdata", pickle_name=hist_pkl.name)
        _rename_plots(
            s11 / "plots",
            {
                "ISI_histogram.pdf": "S11b_ISI_histogram.pdf",
                "legend_ISI_histogram.pdf": "S11b_animal_legend.pdf",
            },
        )
    export_s11_raw(s11)

    # S12 overall bar; S12b example has no pickle
    s12 = DEST / "S12"
    run_replot(s12, "double_steps_move_only")
    _rename_plots(s12 / "plots", {"back_and_forth_overall.pdf": "S12a_overall.pdf"})

    # S13
    s13 = DEST / "S13"
    run_replot(s13, "s13_preonset_2e")

    _prune_plots(DEST / "S1", ["S1a.pdf", "S1b.pdf", "S1c.pdf"])
    _prune_plots(DEST / "S3", ["S3a_head_still.pdf", "S3b_head_moving.pdf", "S3c_colorbar.pdf"])
    _prune_plots(
        DEST / "S8",
        [
            "S8a_lizard_eye_still.png",
            "S8b_trace_lizard.pdf",
            "S8c_mouse_eye_still.png",
            "S8d_trace_mouse.pdf",
            "S8e_turtle_eye_still.png",
            "S8f_trace_turtle.pdf",
            "S8g_figure_2c_isolated.pdf",
            "S8h_figure_2d_isolated.pdf",
            "S8i_figure_2e.pdf",
        ],
    )
    _prune_dir(
        DEST / "S8" / "S8j" / "joint colorbar",
        [
            "S8j_lizard_mouse_all_2f.pdf",
            "S8j_lizard_mouse_zoom_2f.pdf",
            "S8j_colorbar.pdf",
            "S8j_colorbar_zoom.pdf",
        ],
    )
    _prune_dir(
        DEST / "S8" / "S8j" / "separate colorbar",
        [
            "S8j_lizard_mouse_all_2f_separate.pdf",
            "S8j_lizard_mouse_zoom_2f_separate.pdf",
            "S8j_colorbar_lizard.pdf",
            "S8j_colorbar_mouse.pdf",
            "S8j_colorbar_lizard_zoom.pdf",
            "S8j_colorbar_mouse_zoom.pdf",
        ],
    )
    _prune_plots(DEST / "S9", ["S9a_unified_jitter.pdf"])
    _prune_plots(DEST / "S10", ["S10a_lizard.pdf", "S10b_mouse.pdf", "S10c_turtle.pdf"])
    _prune_plots(
        DEST / "S11",
        [
            "S11a_triptych.pdf",
            "S11a_triptych_kde.pdf",
            "S11b_ISI_histogram.pdf",
            "S11b_log_active.pdf",
            "S11b_log_quiet.pdf",
            "S11b_linear_active.pdf",
            "S11b_linear_quiet.pdf",
            "S11b_animal_legend.pdf",
            "S11_raw_log.pdf",
            "S11_raw_active_log.pdf",
            "S11_raw_quiet_log.pdf",
            "S11_raw_active_linear.pdf",
            "S11_raw_quiet_linear.pdf",
        ],
    )
    _prune_plots(DEST / "S12", ["S12a_overall.pdf", "S12b_example.pdf"])
    _prune_plots(
        DEST / "S13",
        [
            "S13a_all.pdf",
            "S13b_concurrent.pdf",
            "S13c_monocular.pdf",
            "S13d_preonset_scatter.pdf",
            "S13e_slope_test.pdf",
            "S13f_animal_legend.pdf",
        ],
    )

    (DEST / "MANIFEST.md").write_text(
        "\n".join(
            [
                "# material_agg_pdf_corrected_FINAL",
                "",
                "One Illustrator PDF per panel (S8j has joint vs separate colorbars).",
                "Type 42 Arial. Data unchanged except S8i (mouse 2e uses S13 mixed A:",
                "onset→offset, length-1 events pre-onset→offset).",
                "",
                "- S1  S1a crop, S1b/S1c Kerr bars",
                "- S3  S3a still / S3b moving / S3c colorbar",
                "- S8  a–i in plots/; S8j in S8j/joint colorbar/ and S8j/separate colorbar/",
                "- S9  S9a unified jitter",
                "- S10 S10a–c Rayleigh cores",
                "- S11 S11a triptychs + S11b ISI + raw epoch hists in plots/",
                "- S12 S12a overall + S12b example",
                "- S13 S13a–f pre-onset A main sequence",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {DEST}")
    n_t3 = 0
    n_pdf = 0
    for pdf in DEST.rglob("*.pdf"):
        n_pdf += 1
        raw = pdf.read_bytes()
        if b"/Subtype /Type3" in raw or b"/Subtype/Type3" in raw:
            n_t3 += 1
            print(f"WARNING Type 3 font: {pdf.relative_to(DEST)}")
    print(f"PDFs={n_pdf} Type3={n_t3}")
    return 0 if n_t3 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(export_corrected())
