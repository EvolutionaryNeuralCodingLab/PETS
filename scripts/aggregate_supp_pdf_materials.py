#!/usr/bin/env python3
"""Sparse copy of supplementary panels into ``outputs/material_aggregation_for_PDF_final``.

One folder per figure. ``plots/`` holds only the Illustrator display files.
Copy, never move/delete sources.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
AGG = REPO / "outputs" / "material_aggregation_for_PDF_final"

DOC_S2 = Path(
    "/Users/nimi/.cursor/projects/Users-nimi-Projects-PETS/attachments/"
    "02dd1d93-cdaa-4d6d-8b6f-1b5c1b595716/Document_S2_-_supplementary_figures-1.pdf"
)
DOC_S2_ALT = Path("/Users/nimi/Downloads/Document S2 - supplementary figures-1.pdf")

MOUSE_323_CACHE = REPO / "outputs" / "review_answers_latest" / "metadata_mouse_323"
S1_KERR_BUNDLE = REPO / "outputs" / "replotting_standalone" / "kerr_component_error"
S3_STANDALONE = REPO / "outputs" / "replotting_standalone" / "figure_S3"
MOUSE_ISOLATED_BUNDLE = REPO / "outputs" / "replotting_standalone" / "mouse_figure_2c_2d_isolated"
MOUSE_2E_BUNDLE = REPO / "outputs" / "replotting_standalone" / "mouse_figure_2e"
S8J_BUNDLE = REPO / "outputs" / "replotting_standalone" / "s8_2f_lizard_mouse_all"
SPECIES_TRACES = REPO / "outputs" / "review_answers_finely_tuned_species_traces_20260818_10_21"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _fig_dir(name: str) -> Path:
    dest = AGG / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "plots").mkdir(exist_ok=True)
    (dest / "metadata").mkdir(exist_ok=True)
    return dest


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _copy_named_pdfs(src_plots: Path, dest_plots: Path, mapping: dict[str, str]) -> list[str]:
    missing: list[str] = []
    dest_plots.mkdir(parents=True, exist_ok=True)
    for src_name, dest_name in mapping.items():
        src = src_plots / src_name
        if src.is_file():
            shutil.copy2(src, dest_plots / dest_name)
        else:
            missing.append(src_name)
    return missing


def _copy_metadata_once(src: Path, dest: Path) -> None:
    meta = src / "metadata"
    if meta.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(meta, dest)
    replot = src / "replot.py"
    if replot.is_file():
        shutil.copy2(replot, dest.parent / "replot.py")


def _replot_figure(dest: Path) -> None:
    rp = dest / "replot.py"
    if not rp.is_file():
        return
    env = os.environ.copy()
    src = str(REPO / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "replot.py", "--overwrite"], cwd=dest, check=False, env=env)


def _prune_plots(
    dest: Path,
    keep: list[str],
    rename: dict[str, str] | None = None,
) -> None:
    """Keep only Illustrator display files in plots/; drop replot leftovers."""
    plots = dest / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    replot_dir = plots / "replot"
    if replot_dir.is_dir():
        shutil.rmtree(replot_dir)
    if rename:
        for src_name, dest_name in rename.items():
            src = plots / src_name
            if src.is_file():
                shutil.copy2(src, plots / dest_name)
    allowed = set(keep)
    for p in list(plots.iterdir()):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.name not in allowed:
            p.unlink()


def extract_s1a_plot(dest_plots: Path) -> str:
    pdf = DOC_S2 if DOC_S2.exists() else DOC_S2_ALT
    if not pdf.exists():
        return f"Document S2 PDF not found ({DOC_S2})"
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf))
    page = reader.pages[0]
    # Crop A4 page to the Blender plot (drop Word title + caption chrome).
    # Coordinates: origin bottom-left, points. Measured from a 2000-px render of page 1.
    page.mediabox.lower_left = (150, 288)
    page.mediabox.upper_right = (430, 508)
    for box_name in ("cropbox", "trimbox", "artbox", "bleedbox"):
        box = getattr(page, box_name, None)
        if box is not None:
            box.lower_left = page.mediabox.lower_left
            box.upper_right = page.mediabox.upper_right
    writer = PdfWriter()
    writer.add_page(page)
    out = dest_plots / "S1a.pdf"
    with open(out, "wb") as fh:
        writer.write(fh)
    try:
        subprocess.run(
            ["qlmanage", "-t", "-s", "1800", "-o", str(dest_plots), str(out)],
            check=False,
            capture_output=True,
        )
    except OSError:
        pass
    return f"Cropped Document S2 page 1 plot → {out.name} from {pdf}"


def try_extract_stills(dest: Path) -> dict[str, str]:
    sys.path.insert(0, str(REPO / "src"))
    from eye_tracking_system_tools.analysis.eye_size_on_sensor import (
        default_species_blocks,
    )
    from eye_tracking_system_tools.analysis.species_traces import DEFAULTS

    notes: dict[str, str] = {}
    try:
        import cv2
    except ImportError:
        return {sp: "opencv not available" for sp in ("lizard", "mouse", "turtle")}

    letter = {"lizard": "a", "mouse": "c", "turtle": "e"}
    for spec in default_species_blocks():
        block = Path(spec.block_path)
        candidates = [block]
        text = str(block)
        for a, b in (
            ("/Volumes/Data-1/Nimrod/experiments/", "/Volumes/Data-2/Nimrod/experiments/"),
            ("/Volumes/Data-1/Nimrod/experiments/", "/Volumes/Data/Nimrod/experiments/"),
            ("/Volumes/Data/Nimrod/experiments/", "/Volumes/Data-1/Nimrod/experiments/"),
            ("/Volumes/Data/Nimrod/experiments/", "/Volumes/Samsung_T5/experiments/"),
        ):
            if a in text:
                candidates.append(Path(text.replace(a, b)))
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            notes[spec.species] = f"block not mounted ({block})"
            continue
        le_dir = found / "eye_videos" / "LE"
        videos = [
            p
            for p in sorted(le_dir.glob("**/*.mp4"))
            if "DLC" not in p.name and "labeled" not in p.name.lower()
        ]
        if not videos:
            notes[spec.species] = f"no eye_videos/LE mp4 under {found}"
            continue
        video = videos[0]
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 60.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = int(float(DEFAULTS[spec.species]["start_s"]) * fps)
        idx = min(max(idx, 0), max(nframes - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            notes[spec.species] = f"failed to read frame {idx} from {video}"
            continue
        out = dest / "plots" / f"S8{letter[spec.species]}_{spec.species}_eye_still.png"
        cv2.imwrite(str(out), frame)
        notes[spec.species] = (
            f"{out.name} from {video} frame {idx} "
            f"(trace start {DEFAULTS[spec.species]['start_s']}s)"
        )
    return notes


def export_s13(dest: Path) -> str:
    sys.path.insert(0, str(REPO / "src"))
    from eye_tracking_system_tools.analysis.block_registry import load_registry
    from eye_tracking_system_tools.analysis.diagnostics_2e import export_s13_preonset_main_sequence
    from eye_tracking_system_tools.analysis.event_cache import build_or_load_event_tables
    from eye_tracking_system_tools.analysis.export_meta import load_params_yaml

    registry = REPO / "configs" / "paper_blocks.yaml"
    params = load_params_yaml(REPO / "configs" / "analysis_params.yaml")
    specs = load_registry(registry)
    cache_dir = REPO / "outputs" / "review_answers_latest" / "metadata"
    try:
        tables, cache_path, from_cache = build_or_load_event_tables(
            specs, params, cache_dir, keep_traces=True
        )
    except Exception as exc:
        return f"S13 event tables failed: {exc}"
    tmp = REPO / "outputs" / "_tmp_s13_preonset"
    try:
        written = export_s13_preonset_main_sequence(tables, tmp, show=False)
    except Exception as exc:
        return (
            f"S13 pre-onset export failed ({exc}). "
            f"cache={cache_path} from_cache={from_cache}. Lab traces required."
        )
    bundle = written["_bundle_dir"]
    for name in (
        "S13a_all.pdf",
        "S13b_concurrent.pdf",
        "S13c_monocular.pdf",
        "S13d_preonset_scatter.pdf",
        "S13e_slope_test.pdf",
        "S13f_animal_legend.pdf",
    ):
        src = bundle / "plots" / name
        if src.is_file():
            shutil.copy2(src, dest / "plots" / name)
    if (bundle / "metadata").is_dir():
        if (dest / "metadata").exists():
            shutil.rmtree(dest / "metadata")
        shutil.copytree(bundle / "metadata", dest / "metadata")
    if (bundle / "replot.py").is_file():
        shutil.copy2(bundle / "replot.py", dest / "replot.py")
    shutil.rmtree(tmp, ignore_errors=True)
    return f"S13 mixed A (length-1 pre-onset) from paper_blocks (cache {cache_path.name})"


def _copy_bundle_sidecars(src: Path, dest: Path) -> None:
    """Copy metadata/ + replot.py from a plot bundle into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    src_meta = src / "metadata"
    if src_meta.is_dir():
        if (dest / "metadata").exists():
            shutil.rmtree(dest / "metadata")
        shutil.copytree(src_meta, dest / "metadata")
    rp = src / "replot.py"
    if rp.is_file():
        shutil.copy2(rp, dest / "replot.py")


def load_lizard_paper_tables(*, keep_traces: bool = True):
    from eye_tracking_system_tools.analysis.block_registry import load_registry
    from eye_tracking_system_tools.analysis.event_cache import build_or_load_event_tables
    from eye_tracking_system_tools.analysis.export_meta import load_params_yaml

    params = load_params_yaml(REPO / "configs" / "analysis_params.yaml")
    specs = load_registry(REPO / "configs" / "paper_blocks.yaml")
    cache_dir = REPO / "outputs" / "review_answers_latest" / "metadata"
    return build_or_load_event_tables(
        specs, params, cache_dir, keep_traces=keep_traces, prefer_finalized=True
    )


def load_mouse_323_tables(*, keep_traces: bool = True, force: bool = False):
    """M_002 events at YAML 3.23 °/frame; never let a 0.8 GUI finalize override."""
    from eye_tracking_system_tools.analysis.block_registry import load_registry
    from eye_tracking_system_tools.analysis.event_cache import build_or_load_event_tables
    from eye_tracking_system_tools.analysis.export_meta import load_params_yaml

    params = load_params_yaml(REPO / "configs" / "analysis_params_mouse.yaml")
    specs = load_registry(REPO / "configs" / "mouse_M_002_blocks.yaml")
    MOUSE_323_CACHE.mkdir(parents=True, exist_ok=True)
    return build_or_load_event_tables(
        specs,
        params,
        MOUSE_323_CACHE,
        keep_traces=keep_traces,
        force=force,
        prefer_finalized=False,
    )


def export_s1_kerr_into(dest: Path) -> str:
    sys.path.insert(0, str(REPO / "src"))
    from eye_tracking_system_tools.analysis.kerr_component_error_export import export_s1_kerr_bars
    from eye_tracking_system_tools.analysis.plot_bundle import write_replot_script

    try:
        written = export_s1_kerr_bars(S1_KERR_BUNDLE, show=False)
    except Exception as exc:
        return f"S1 Kerr export failed: {exc}"
    bundle = written["_bundle_dir"]
    plots = dest / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    bpdf = bundle / "plots" / "cohort_component_error_across_animals.pdf"
    cpdf = bundle / "plots" / "cohort_component_error_overall_phi_theta.pdf"
    if bpdf.is_file():
        shutil.copy2(bpdf, plots / "S1b.pdf")
    if cpdf.is_file():
        shutil.copy2(cpdf, plots / "S1c.pdf")
    _copy_bundle_sidecars(bundle, dest)
    write_replot_script(dest, "kerr_component_error")
    _replot_figure(dest)
    bpdf2 = dest / "plots" / "cohort_component_error_across_animals.pdf"
    cpdf2 = dest / "plots" / "cohort_component_error_overall_phi_theta.pdf"
    if bpdf2.is_file():
        shutil.copy2(bpdf2, plots / "S1b.pdf")
    if cpdf2.is_file():
        shutil.copy2(cpdf2, plots / "S1c.pdf")
    return f"S1b/S1c from pickle {bundle / 'metadata' / 'kerr_component_error.pkl'}"


def export_s3_live(dest: Path) -> str:
    sys.path.insert(0, str(REPO / "src"))
    from eye_tracking_system_tools.analysis.figures_2f_2h_2i import export_figure_s3
    from eye_tracking_system_tools.analysis.plot_bundle import write_replot_script

    note = f"S3 copy from {S3_STANDALONE}"
    try:
        tables, cache_path, from_cache = load_lizard_paper_tables(keep_traces=True)
        written = export_figure_s3(tables, S3_STANDALONE, show=False)
        note = (
            f"S3 live from paper_blocks (cache {cache_path.name} from_cache={from_cache}); "
            f"pickle {written.get('figure_S3.pickle')}"
        )
    except Exception as exc:
        note = f"S3 live export failed ({exc}); copying existing standalone if present"
        if not (S3_STANDALONE / "metadata").is_dir():
            return note
    miss = _copy_named_pdfs(
        S3_STANDALONE / "plots",
        dest / "plots",
        {
            "figure_S3_head_still.pdf": "S3a_head_still.pdf",
            "figure_S3_head_moving.pdf": "S3b_head_moving.pdf",
            "figure_S3_colorbar.pdf": "S3c_colorbar.pdf",
        },
    )
    _copy_bundle_sidecars(S3_STANDALONE, dest)
    write_replot_script(dest, "figure_s3")
    _replot_figure(dest)
    _prune_plots(
        dest,
        ["S3a_head_still.pdf", "S3b_head_moving.pdf", "S3c_colorbar.pdf"],
        {
            "figure_S3_head_still.pdf": "S3a_head_still.pdf",
            "figure_S3_head_moving.pdf": "S3b_head_moving.pdf",
            "figure_S3_colorbar.pdf": "S3c_colorbar.pdf",
        },
    )
    return f"{note}; missing={miss}"


def export_s8_2f_compare(dest: Path, mou_tables=None, liz_tables=None) -> str:
    sys.path.insert(0, str(REPO / "src"))
    from eye_tracking_system_tools.analysis.figures_2f_2h_2i import export_figure_2f_lizard_mouse_all
    from eye_tracking_system_tools.analysis.plot_bundle import write_replot_script

    try:
        if liz_tables is None:
            liz_tables, liz_cache, _ = load_lizard_paper_tables(keep_traces=True)
        else:
            liz_cache = Path("in-memory")
        if mou_tables is None:
            mou_tables, mou_cache, _ = load_mouse_323_tables(keep_traces=True)
        else:
            mou_cache = Path("in-memory")
        written = export_figure_2f_lizard_mouse_all(liz_tables, mou_tables, S8J_BUNDLE, show=False)
    except Exception as exc:
        return f"S8 2f compare export failed: {exc}"
    bundle = written["_bundle_dir"]
    s8j = dest / "S8j"
    joint = s8j / "joint colorbar"
    separate = s8j / "separate colorbar"
    joint.mkdir(parents=True, exist_ok=True)
    separate.mkdir(parents=True, exist_ok=True)
    plots = dest / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    mapping_joint = (
        ("S8j_lizard_mouse_all_2f.pdf", joint, plots),
        ("S8j_colorbar.pdf", joint, plots),
    )
    for name, folder, extra in mapping_joint:
        src = bundle / "plots" / name
        if src.is_file():
            shutil.copy2(src, folder / name)
            shutil.copy2(src, extra / name)
    for name in (
        "S8j_lizard_mouse_all_2f_separate.pdf",
        "S8j_lizard_mouse_zoom_2f_separate.pdf",
        "S8j_colorbar_lizard.pdf",
        "S8j_colorbar_mouse.pdf",
        "S8j_colorbar_lizard_zoom.pdf",
        "S8j_colorbar_mouse_zoom.pdf",
    ):
        src = bundle / "plots" / name
        if src.is_file():
            shutil.copy2(src, separate / name)
    _copy_bundle_sidecars(bundle, s8j)
    write_replot_script(s8j, "figure_2f_lizard_mouse")
    _replot_figure(s8j)
    s8j_plots = s8j / "plots"
    if (s8j_plots / "S8j_lizard_mouse_all_2f.pdf").is_file():
        shutil.copy2(s8j_plots / "S8j_lizard_mouse_all_2f.pdf", joint / "S8j_lizard_mouse_all_2f.pdf")
        shutil.copy2(s8j_plots / "S8j_lizard_mouse_all_2f.pdf", plots / "S8j_lizard_mouse_all_2f.pdf")
    if (s8j_plots / "S8j_lizard_mouse_zoom_2f.pdf").is_file():
        shutil.copy2(s8j_plots / "S8j_lizard_mouse_zoom_2f.pdf", joint / "S8j_lizard_mouse_zoom_2f.pdf")
        shutil.copy2(s8j_plots / "S8j_lizard_mouse_zoom_2f.pdf", plots / "S8j_lizard_mouse_zoom_2f.pdf")
    if (s8j_plots / "S8j_colorbar.pdf").is_file():
        shutil.copy2(s8j_plots / "S8j_colorbar.pdf", joint / "S8j_colorbar.pdf")
        shutil.copy2(s8j_plots / "S8j_colorbar.pdf", plots / "S8j_colorbar.pdf")
    if (s8j_plots / "S8j_colorbar_zoom.pdf").is_file():
        shutil.copy2(s8j_plots / "S8j_colorbar_zoom.pdf", joint / "S8j_colorbar_zoom.pdf")
    if (s8j_plots / "S8j_lizard_mouse_all_2f_separate.pdf").is_file():
        shutil.copy2(
            s8j_plots / "S8j_lizard_mouse_all_2f_separate.pdf",
            separate / "S8j_lizard_mouse_all_2f_separate.pdf",
        )
    if (s8j_plots / "S8j_lizard_mouse_zoom_2f_separate.pdf").is_file():
        shutil.copy2(
            s8j_plots / "S8j_lizard_mouse_zoom_2f_separate.pdf",
            separate / "S8j_lizard_mouse_zoom_2f_separate.pdf",
        )
    for name in (
        "S8j_colorbar_lizard.pdf",
        "S8j_colorbar_mouse.pdf",
        "S8j_colorbar_lizard_zoom.pdf",
        "S8j_colorbar_mouse_zoom.pdf",
    ):
        if (s8j_plots / name).is_file():
            shutil.copy2(s8j_plots / name, separate / name)
    thr_m = (getattr(mou_tables, "params", {}) or {}).get("saccade", {}).get(
        "speed_threshold_deg_per_frame"
    )
    return (
        f"lizard vs mouse all-events 2f (liz {getattr(liz_cache, 'name', liz_cache)}, "
        f"mou {getattr(mou_cache, 'name', mou_cache)}; mouse thr={thr_m})"
    )


def export_s8_mouse_isolated(dest: Path, mou_tables) -> str:
    sys.path.insert(0, str(REPO / "src"))
    from eye_tracking_system_tools.analysis.figures_2c_2e import (
        export_amplitude_velocity_fit,
        export_pos_vel_bundle,
    )
    from eye_tracking_system_tools.analysis.plot_bundle import write_replot_script

    try:
        pkl = export_pos_vel_bundle(
            mou_tables,
            MOUSE_ISOLATED_BUNDLE,
            show=False,
            isolation="nan_mask_neighbors",
        )
        export_amplitude_velocity_fit(mou_tables, MOUSE_2E_BUNDLE, show=False)
    except Exception as exc:
        return f"S8 mouse isolated 2c/2d/2e failed: {exc}"
    iso_dest = dest / "mouse_2c_2d_isolated"
    if iso_dest.exists():
        shutil.rmtree(iso_dest)
    shutil.copytree(MOUSE_ISOLATED_BUNDLE, iso_dest)
    write_replot_script(iso_dest, "pos_vel")
    _replot_figure(iso_dest)
    plots = dest / "plots"
    g = iso_dest / "plots" / "figure_2c_isolated.pdf"
    h = iso_dest / "plots" / "figure_2d_isolated.pdf"
    if not g.is_file():
        g = iso_dest / "plots" / "figure_2c.pdf"
    if not h.is_file():
        h = iso_dest / "plots" / "figure_2d.pdf"
    if g.is_file():
        shutil.copy2(g, plots / "S8g_figure_2c_isolated.pdf")
    if h.is_file():
        shutil.copy2(h, plots / "S8h_figure_2d_isolated.pdf")
    e2 = dest / "mouse_figure_2e"
    if e2.exists():
        shutil.rmtree(e2)
    shutil.copytree(MOUSE_2E_BUNDLE, e2)
    write_replot_script(e2, "figure_2e")
    _replot_figure(e2)
    e_pdf = e2 / "plots" / "figure_2e.pdf"
    if e_pdf.is_file():
        shutil.copy2(e_pdf, plots / "S8i_figure_2e.pdf")
    thr = (getattr(mou_tables, "params", {}) or {}).get("saccade", {}).get(
        "speed_threshold_deg_per_frame"
    )
    return (
        f"isolated 2c/2d pickle {pkl}; 2e from same tables; "
        f"speed_threshold_deg_per_frame={thr}"
    )


def copy_s8_trace_pickles(dest: Path) -> list[str]:
    notes: list[str] = []
    meta = dest / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    for letter, species in (("b", "lizard"), ("d", "mouse"), ("f", "turtle")):
        src_dir = SPECIES_TRACES / f"trace_{species}"
        src_pdf = src_dir / "plots" / f"trace_{species}.pdf"
        if src_pdf.is_file():
            shutil.copy2(src_pdf, dest / "plots" / f"S8{letter}_trace_{species}.pdf")
            notes.append(f"S8{letter} pdf {src_pdf}")
        pkl = src_dir / "metadata" / "trace_series.pkl"
        if pkl.is_file():
            shutil.copy2(pkl, meta / f"trace_{species}_series.pkl")
            notes.append(f"S8{letter} pickle {pkl}")
    return notes


PREFERRED_S12_EXAMPLE = "PV_126_007_297867_R"


def assemble_s12(s12: Path) -> str:
    """S12a: thinner overall bar (head-moving, ≥7°). S12b: keep the chosen example."""
    from eye_tracking_system_tools.analysis.block_registry import load_registry
    from eye_tracking_system_tools.analysis.double_steps_rev_verbatim import (
        export_back_and_forth_example_from_block,
        export_back_and_forth_example_from_trace,
        rank_binocular_back_and_forth_examples,
    )
    from eye_tracking_system_tools.analysis.plot_bundle import write_replot_script

    move = (
        REPO
        / "development"
        / "movement_associated_nystagmus"
        / "double_steps_rev_verbatim"
        / "move_only"
    )
    ds = REPO / "development" / "movement_associated_nystagmus" / "double_steps_rev_verbatim"
    events_csv = ds / "metadata" / "large_saccades.csv"
    overall = move / "plots" / "back_and_forth_overall.pdf"
    if overall.is_file():
        shutil.copy2(overall, s12 / "plots" / "S12a_overall.pdf")
    _copy_metadata_once(move, s12 / "metadata")

    ranked = rank_binocular_back_and_forth_examples(events_csv)
    pick = None
    if not ranked.empty:
        hit = ranked.loc[ranked["event_id"].astype(str) == PREFERRED_S12_EXAMPLE]
        pick = hit.iloc[0] if not hit.empty else ranked.iloc[0]
    s12_example_id = str(pick["event_id"]) if pick is not None else PREFERRED_S12_EXAMPLE
    specs12 = load_registry(REPO / "configs" / "paper_blocks.yaml")
    spec12 = None
    if pick is not None:
        want_key = str(pick["block_key"])
        want_animal = str(pick["animal"])
        want_block = str(pick["block"]).zfill(3)
        for sp in specs12:
            if sp.block_key == want_key or (
                sp.animal == want_animal and sp.block_num == want_block
            ):
                spec12 = sp
                break
    tmp12 = REPO / "outputs" / "_tmp_s12_example"
    shutil.rmtree(tmp12, ignore_errors=True)
    if spec12 is not None:
        export_back_and_forth_example_from_block(
            spec=spec12,
            events_csv=events_csv,
            event_id=s12_example_id,
            out_dir=tmp12,
            pre_s=0.25,
            post_s=0.55,
        )
    else:
        traces = REPO / "outputs" / "review_answers_finely_tuned_species_traces_20260818_10_21"
        export_back_and_forth_example_from_trace(
            trace_pkl=traces / "trace_lizard" / "metadata" / "trace_series.pkl",
            events_csv=events_csv,
            event_id=s12_example_id,
            out_dir=tmp12,
            pre_s=0.25,
            post_s=0.55,
        )
    ex = tmp12 / "back_and_forth_example" / "plots" / "back_and_forth_example.pdf"
    if ex.is_file():
        shutil.copy2(ex, s12 / "plots" / "S12b_example.pdf")
    shutil.rmtree(tmp12, ignore_errors=True)

    write_replot_script(s12, "double_steps_move_only")
    _replot_figure(s12)
    _prune_plots(
        s12,
        ["S12a_overall.pdf", "S12b_example.pdf"],
        {"back_and_forth_overall.pdf": "S12a_overall.pdf"},
    )

    summary = {}
    sy = s12 / "metadata" / "summary.yaml"
    if sy.is_file():
        summary = yaml.safe_load(sy.read_text()) or {}
    overall_s = summary.get("overall") or {}
    pct = overall_s.get("pct")
    n_hit = overall_s.get("n_hit", overall_s.get("n_reverse", 0))
    n = overall_s.get("n", 0)
    thr = summary.get("amp_threshold_deg", 7)
    pct_s = f"{float(pct):.1f}" if pct is not None and np.isfinite(float(pct)) else "n/a"
    both = None if pick is None else bool(pick["both_eyes_extras"])
    _write(
        s12 / "SOURCE.txt",
        f"{move / 'plots' / 'back_and_forth_overall.pdf'}\n"
        f"example {s12_example_id} from full-block Kerr traces "
        f"(both_eyes_extras={both}; window −0.25/+0.55 s). "
        f"Head-moving only, large ≥ {thr}°; {pct_s}% ({n_hit}/{n}).",
    )
    _write(
        s12 / "captions.md",
        f"""\
Figure S12. Post-saccadic back-and-forth sequences during head movement.

Large unique saccades (≥{float(thr):.0f}°) during annotated head-moving periods, followed within
150 ms by ≥2 smaller extras with a direction reversal. Frequency: {pct_s}% ({n_hit}/{n}).
These events are interpreted as nystagmus-like gaze-stabilizing movements during head motion.

**(a)** Rate among large saccades during head movement.
**(b)** Example sequence ({s12_example_id}); both eyes show the reverse extras. Window −0.25 to
+0.55 s around primary onset. Solid: primary onset; dashed: primary offset; green dotted: extras
within 150 ms. Blue = left eye, orange = right eye.
""",
    )
    cap = s12 / "captions.md"
    if cap.is_file():
        shutil.copy2(cap, s12 / "metadata" / "captions.md")
    return s12_example_id


def build_s1_s3_s8(manifest: list[str]) -> None:
    """Write S1, S3, and S8 into AGG in place (does not wipe other figures)."""
    sys.path.insert(0, str(REPO / "src"))

    s1 = _fig_dir("S1")
    note_a = extract_s1a_plot(s1 / "plots")
    kerr_note = export_s1_kerr_into(s1)
    lookup = (
        REPO
        / "development"
        / "kerr_relative_error"
        / "ellipse_angle_mapping_correct_diameter_08mm_distance_13mm.csv"
    )
    if lookup.is_file():
        shutil.copy2(lookup, s1 / "metadata" / lookup.name)
    _prune_plots(s1, ["S1a.pdf", "S1b.pdf", "S1c.pdf"])
    _write(s1 / "SOURCE.txt", note_a + "\n" + kerr_note)
    _write(
        s1 / "captions.md",
        """\
Figure S1. Angular reconstruction validation.

**(a)** Simulation-based accuracy assessment. A 3D Blender animation of a rendered eye
monotonously spanning (−60, −60) to (60, 60) degrees (1° step) was compared with ground truth.
Mean total error sqrt((φ − X)² + (θ − Y)²) versus eccentricity (bottom) and ellipse ratio
(minor/major; top). Dashed line: P. vitticeps extended span (±35° Euclidean; max radial 50°).
This panel is the published Document S2 plot (page 1 cropped to the axes; not a Kerr heatmap).

**(b)** Relative rest-centered reprojection error, per animal. Raw and Kerr-corrected
estimates were independently zeroed to rest; residual |Δ| on the measurements used in the study.

**(c)** Eye-pooled leftover error (animal as unit, n = 5): φ mean|Δ| = 0.84 ± 0.48°
(range 0.42–1.67°); θ mean|Δ| = 1.19 ± 0.48° (range 0.65–1.93°). Error bars are SD across animals.
""",
    )
    manifest.append("S1  S1a crop + S1b/S1c from kerr pickle")

    s3 = _fig_dir("S3")
    s3_note = export_s3_live(s3)
    _write(s3 / "SOURCE.txt", s3_note)
    _write(
        s3 / "captions.md",
        """\
Figure S3. Coupling of peak saccade speed between eyes, with and without head movements.

**(a)** Head-stationary 2D histogram of right- vs left-eye peak saccade speed (0–0.2 °/ms,
100 × 100 bins / 100 histogram edges). **(b)** Same for saccades during head movements. **(c)** Colour scale shared
by (a) and (b) only (not with Fig. 2f). Each histogram is independently normalised to sum to 1.
Dashed line: equality between eyes. Legacy contra ±51 ms sampling; PV_62 excluded; animals pooled equally.
""",
    )
    manifest.append("S3  still / moving / colorbar (legacy pairing, pickle replot)")

    s8 = _fig_dir("S8")
    trace_notes = copy_s8_trace_pickles(s8)
    meta = s8 / "metadata"
    if meta.is_dir():
        for stale in meta.glob("s8j_*"):
            stale.unlink()
    still_notes = try_extract_stills(s8)
    iso_note = "mouse isolated 2c/2d/2e skipped"
    s8j_note = "S8j skipped"
    try:
        mou_tables, mou_cache, mou_from_cache = load_mouse_323_tables(keep_traces=True)
        iso_note = export_s8_mouse_isolated(s8, mou_tables)
        iso_note = f"{iso_note}; cache {mou_cache.name} from_cache={mou_from_cache}"
        liz_tables = None
        try:
            liz_tables, _, _ = load_lizard_paper_tables(keep_traces=True)
        except Exception as exc:
            s8j_note = f"lizard tables failed: {exc}"
        if liz_tables is not None:
            s8j_note = export_s8_2f_compare(s8, mou_tables=mou_tables, liz_tables=liz_tables)
    except Exception as exc:
        iso_note = f"mouse 3.23 tables failed: {exc}"
    _prune_plots(
        s8,
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
            "S8j_lizard_mouse_all_2f.pdf",
            "S8j_lizard_mouse_zoom_2f.pdf",
            "S8j_colorbar.pdf",
        ],
    )
    _write(
        s8 / "SOURCE.txt",
        f"traces: {SPECIES_TRACES}\n"
        f"trace pickles: {trace_notes}\n"
        f"mouse isolated 2c/2d/2e: {iso_note}\n"
        f"2f lizard vs mouse all-events: {s8j_note}\n"
        f"stills: {still_notes}\n"
        "S8g/S8h/S8i/S8j share M_002 events at 3.23 °/frame (prefer_finalized=False).\n"
        "S8h is isolated mean position from the same series as S8g (nan_mask_neighbors).\n"
        "S8j: full view at matched 99.5th percentile; zoom = slower 50% of events, same 59×59 bins;\n"
        "joint vs separate colorbars in S8/S8j/.\n",
    )
    _write(
        s8 / "captions.md",
        """\
Figure S8. Cross-species validation.

**(a,c,e)** Representative eye-video frames (lizard PV_126/007, mouse M_002/012, turtle T_18/001)
when lab volumes are mounted. **(b,d,f)** Matching φ/θ traces (independent y-limits).
**(g)** Mouse saccade speed profiles (Fig. 2c analogue) after masking other same-eye events in the
±100 ms window (`nan_mask_neighbors`). Detector floor 3.23 °/frame (jitter-catalog mouse threshold).
**(h)** Mouse position profiles from the same isolated series as (g).
**(i)** Mouse amplitude–peak-speed (Fig. 2e analogue), same 3.23 °/frame events.
**(j)** Interocular peak-speed coupling for **all saccades** in lizard (left; 0.8 °/frame) and
mouse (right; 3.23 °/frame). Two zooms, same 60-edge (59 × 59) grid: the full view uses
independent square limits at the 99.5th percentile of each species; the inner-50% view crops
the faster half of events (per-event max of the two eyes) so the threshold region is shown at
higher resolution, still with the same fraction of each dataset. Joint colour scales in
`S8/S8j/joint colorbar/`; per-panel colour scales in `S8/S8j/separate colorbar/`. Mouse has no
head-movement tags, so lizard is shown without the head-stationary filter used in main Fig. 2f
(same lizard animals as Fig. 2f: PV_57 and PV_62 excluded).
""",
    )
    manifest.append("S8  traces+pickles + isolated 2c/2d + 2e + lizard/mouse 2f (3.23 mouse)")


def update_s1_s3_s8() -> int:
    """Patch S1/S3/S8 in place. Does not delete the rest of AGG."""
    AGG.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = ["# material_aggregation_for_PDF_final (S1/S3/S8 patch)"]
    build_s1_s3_s8(manifest)
    man = AGG / "MANIFEST.md"
    if man.is_file():
        mapped = {
            "S1": "S1  S1a crop + S1b/S1c from kerr pickle",
            "S3": "S3  still / moving / colorbar (legacy pairing, pickle replot)",
            "S8": "S8  traces+pickles + isolated 2c/2d + 2e + lizard/mouse 2f (3.23 mouse)",
        }
        out = []
        for line in man.read_text(encoding="utf-8").splitlines():
            body = line.lstrip("- ").strip()
            key = body.split()[0] if body else ""
            if key in mapped:
                out.append("- " + mapped[key])
            else:
                out.append(line)
        man.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(f"patched {AGG}")
    print("\n".join(manifest))
    return 0


def main() -> int:
    if AGG.exists():
        shutil.rmtree(AGG)
    AGG.mkdir(parents=True)
    manifest: list[str] = ["# material_aggregation_for_PDF_final (sparse)"]
    sys.path.insert(0, str(REPO / "src"))
    from eye_tracking_system_tools.analysis.plot_bundle import write_replot_script

    build_s1_s3_s8(manifest)

    # --- S9 ---
    s9 = _fig_dir("S9")
    jsrc = REPO / "outputs" / "USE_THIS_VERSION" / "unified jitter quantification across animals"
    src_pdf = jsrc / "plots" / "unified_jitter_quantification.pdf"
    if src_pdf.is_file():
        shutil.copy2(src_pdf, s9 / "plots" / "S9a_unified_jitter.pdf")
    _copy_metadata_once(jsrc, s9 / "metadata")
    jcsv = REPO / "outputs" / "jitter_csv_values" / "jitter_values.csv"
    if jcsv.is_file():
        shutil.copy2(jcsv, s9 / "metadata" / "jitter_values.csv")
    write_replot_script(s9, "unified_jitter")
    _replot_figure(s9)
    _prune_plots(
        s9,
        ["S9a_unified_jitter.pdf"],
        {"unified_jitter_quantification.pdf": "S9a_unified_jitter.pdf"},
    )
    _write(s9 / "SOURCE.txt", f"{jsrc}\nMedian (solid) and P95 (dotted) annotated via replot --overwrite.")
    _write(
        s9 / "captions.md",
        """\
Figure S9. Camera jitter at the imaged eye plane.

**(a)** Displacement histograms for rigid (single-piece) lizard and modular lizard / mouse / turtle mounts.
Solid line = median; dotted = 95th percentile.
Medians: mouse = 28 µm (2.2 px), turtle = 80 µm (2.0 px), modular lizard = 90 µm (2.0 px), rigid lizard = 42 µm (1.0 px).
95th percentiles: mouse = 90 µm (7.1 px), turtle = 179 µm (4.5 px), modular lizard = 211 µm (5.0 px), rigid lizard = 173 µm (5.4 px).
Turtle: one animal / one block. Mouse histograms use pixel_bin_k = 4. Stats are pooled samples per mount type.
Interframe jitter gating removed 0.21% of frames.
""",
    )
    manifest.append("S9  unified jitter with median/P95")

    # --- S10 ---
    s10 = _fig_dir("S10")
    liz = REPO / "outputs" / "review_answers_latest" / "rayleigh_noise_core"
    mou = REPO / "outputs" / "review_answers_latest" / "rayleigh_noise_core_mouse"
    tur = REPO / "outputs" / "review_answers_latest" / "rayleigh_noise_core_turtle"
    mapping_src = (
        (liz, "S10a_lizard.pdf", "lizard"),
        (mou, "S10b_mouse.pdf", "mouse"),
        (tur, "S10c_turtle.pdf", "turtle"),
    )
    s10_notes = []
    s10_pkls: dict[str, Path] = {}
    s10_summ: dict[str, dict] = {}
    for src, dest_name, species in mapping_src:
        pdf = src / "plots" / "rayleigh_noise_core.pdf"
        if pdf.is_file():
            shutil.copy2(pdf, s10 / "plots" / dest_name)
        for extra in (src / "metadata").glob("*"):
            if extra.is_file():
                dest = s10 / "metadata" / f"{dest_name.replace('.pdf', '')}_{extra.name}"
                shutil.copy2(extra, dest)
                if extra.name == "rayleigh_noise_core.pkl":
                    s10_pkls[species] = dest
        sy = src / "metadata" / "rayleigh_noise_core_summary.yaml"
        if sy.is_file():
            body = yaml.safe_load(sy.read_text()) or {}
            s10_summ[species] = body
            s10_notes.append(
                f"{dest_name}: n={body.get('n')} median_D={body.get('median_D')} "
                f"SNR={body.get('snr_thr_over_2B')} from {src}"
            )
    ks_csv = None
    ks_rows: dict[str, dict] = {}
    if s10_pkls:
        from eye_tracking_system_tools.analysis.rayleigh_noise_core_export import (
            write_s10_ks_gof_csv,
        )

        ks_csv = write_s10_ks_gof_csv(
            s10_pkls,
            s10 / "ks_rayleigh_gof.csv",
            n_sim=1000,
            seed=0,
        )
        shutil.copy2(ks_csv, s10 / "metadata" / "ks_rayleigh_gof.csv")
        ks_df = pd.read_csv(ks_csv)
        ks_rows = {str(r["species"]): r for r in ks_df.to_dict(orient="records")}
        s10_notes.append(f"KS GOF: {ks_csv}")
    _prune_plots(s10, ["S10a_lizard.pdf", "S10b_mouse.pdf", "S10c_turtle.pdf"])
    _write(s10 / "SOURCE.txt", "\n".join(s10_notes) or "S10 sources missing")

    def _s10_ks_d(species: str) -> str:
        row = ks_rows.get(species) or {}
        d = row.get("ks_statistic")
        if d is None or not np.isfinite(float(d)):
            return ""
        return f"{float(d):.3f}"

    liz_s = s10_summ.get("lizard") or {}
    mou_s = s10_summ.get("mouse") or {}
    tur_s = s10_summ.get("turtle") or {}
    _write(
        s10 / "captions.md",
        f"""\
Figure S10. Frame-to-frame tracking noise in quiet intervals is close to Rayleigh.

Histograms of D = hypot(Δφ, Δθ) in GUI-audited stationary windows, with a Rayleigh overlay
scaled to the sample median (B = median(D)/√(2 ln 2)). The dashed 2B line is the 2 axis-σ
noise radius. Where a detector threshold is defined it is shown; SNR = threshold / 2B.

**(a)** Lizard (PV_228, block_016; n = {int(liz_s.get('n') or 0)}). median(D) = {float(liz_s.get('median_D') or float('nan')):.3f}°/frame,
B = {float(liz_s.get('B_med') or float('nan')):.3f}, 2B = {float(liz_s.get('floor_2B') or float('nan')):.3f},
threshold = 0.8°/frame, SNR = {float(liz_s.get('snr_thr_over_2B') or float('nan')):.2f}. KS D = {_s10_ks_d('lizard')}.

**(b)** Mouse (M_002, block_012; n = {int(mou_s.get('n') or 0)}). median(D) = {float(mou_s.get('median_D') or float('nan')):.3f}°/frame,
B = {float(mou_s.get('B_med') or float('nan')):.3f}, 2B = {float(mou_s.get('floor_2B') or float('nan')):.3f},
threshold = 3.23°/frame, SNR = {float(mou_s.get('snr_thr_over_2B') or float('nan')):.2f}. KS D = {_s10_ks_d('mouse')}.

**(c)** Turtle (T_18, block_001; n = {int(tur_s.get('n') or 0)}). median(D) = {float(tur_s.get('median_D') or float('nan')):.3f}°/frame,
B = {float(tur_s.get('B_med') or float('nan')):.3f}, 2B = {float(tur_s.get('floor_2B') or float('nan')):.3f}.
No detection threshold. KS D = {_s10_ks_d('turtle')}.

The body of each histogram follows the median-matched Rayleigh overlay. KS D is the largest
vertical gap between the empirical and Rayleigh CDFs. With several thousand samples those
gaps exceed the Lilliefors 5% critical D, so a formal test rejects *exact* Rayleigh; the
residual is a modest heavy tail rather than a different family. Lizard and mouse are closest
(D = {_s10_ks_d('lizard')} and {_s10_ks_d('mouse')}); turtle has a thicker tail (D = {_s10_ks_d('turtle')}).
""",
    )
    manifest.append("S10  GUI quiet cores + KS Rayleigh GOF csv")

    # --- S11 ---
    s11 = _fig_dir("S11")
    ep = REPO / "outputs" / "USE_THIS_VERSION" / "epoch_duration_bin_trials"
    isi = REPO / "outputs" / "USE_THIS_VERSION" / "ISI_by_state"
    _copy_named_pdfs(
        ep / "plots",
        s11 / "plots",
        {
            "epoch_duration_triptych_width5s_quiet50s.pdf": "S11a_triptych.pdf",
            "epoch_duration_triptych_kde_width5s_quiet50s.pdf": "S11a_triptych_kde.pdf",
        },
    )
    _copy_named_pdfs(
        isi / "plots",
        s11 / "plots",
        {
            "ISI_histogram.pdf": "S11b_ISI_histogram.pdf",
            "ISI_log_active.pdf": "S11b_log_active.pdf",
            "ISI_log_quiet.pdf": "S11b_log_quiet.pdf",
            "ISI_linear_active.pdf": "S11b_linear_active.pdf",
            "ISI_linear_quiet.pdf": "S11b_linear_quiet.pdf",
            "legend_ISI_histogram.pdf": "S11b_animal_legend.pdf",
        },
    )
    for extra in (ep / "metadata").glob("*"):
        if extra.is_file():
            shutil.copy2(extra, s11 / "metadata" / extra.name)
    for extra in (isi / "metadata").glob("*"):
        if extra.is_file():
            shutil.copy2(extra, s11 / "metadata" / f"ISI_{extra.name}")
    _prune_plots(
        s11,
        [
            "S11a_triptych.pdf",
            "S11a_triptych_kde.pdf",
            "S11b_ISI_histogram.pdf",
            "S11b_log_active.pdf",
            "S11b_log_quiet.pdf",
            "S11b_linear_active.pdf",
            "S11b_linear_quiet.pdf",
            "S11b_animal_legend.pdf",
        ],
    )
    _write(s11 / "SOURCE.txt", f"S11a epochs: {ep}\nS11b ISI: {isi}\nAnimal legend from ISI_histogram.")
    _write(
        s11 / "captions.md",
        """\
Figure S11. Active/Quiet epoch durations and inter-saccadic intervals.

**(a)** Durations of sustained Active and Quiet epochs after bridging opposite-state interruptions
shorter than 3 s and retaining epochs ≥ 5 s (quiet full-range, quiet zoomed, and active on the same
zoom; 5 s bins). This smoothing is used only here, not for Fig. 3.
**(b)** ISI distributions split by Active and Quiet (intervals that cross a state boundary excluded).
Per-animal colours are in the separate legend panel.
""",
    )
    manifest.append("S11  USE_THIS_VERSION epoch triptych + ISI + animal legend")

    # --- S12 ---
    s12 = _fig_dir("S12")
    assemble_s12(s12)
    manifest.append("S12  head-moving overall rate + example")

    # --- S13 ---
    s13 = _fig_dir("S13")
    s13_note = export_s13(s13)
    _replot_figure(s13)
    _prune_plots(
        s13,
        [
            "S13a_all.pdf",
            "S13b_concurrent.pdf",
            "S13c_monocular.pdf",
            "S13d_preonset_scatter.pdf",
            "S13e_slope_test.pdf",
            "S13f_animal_legend.pdf",
        ],
    )
    slope_txt = ""
    sy = s13 / "metadata" / "slope_comparison.yaml"
    if sy.is_file():
        sl = yaml.safe_load(sy.read_text()) or {}
        p = sl.get("paired_p")
        p_s = f"{float(p):.3g}" if p is not None and np.isfinite(float(p)) else "nan"
        t_s = f"{float(sl.get('paired_t')):.2f}" if sl.get("paired_t") is not None and np.isfinite(float(sl.get("paired_t"))) else "nan"
        slope_txt = (
            f"Across animals (n = {sl.get('n_animals')}), OLS slope was "
            f"{float(sl.get('concurrent_mean_slope', float('nan'))):.4f} ± "
            f"{float(sl.get('concurrent_sem_slope', float('nan'))):.4f} °/ms per degree "
            f"for concurrent saccades versus "
            f"{float(sl.get('monocular_mean_slope', float('nan'))):.4f} ± "
            f"{float(sl.get('monocular_sem_slope', float('nan'))):.4f} for monocular "
            f"(mean ± SEM of per-animal fits). Paired t = {t_s}, "
            f"p = {p_s}, df = {sl.get('df')}. Amplitude is onset→offset except "
            f"length-1 events (pre-onset→offset)."
        )
    _write(s13 / "SOURCE.txt", s13_note)
    _write(
        s13 / "captions.md",
        """\
Figure S13. Main sequence for all, concurrent, and monocular saccades.

Paper-style per-animal mean peak speed vs amplitude (each animal a coloured line; identity in
**(f)**). Amplitude is onset→offset except for detector length = 1, which uses pre-onset→offset.
Length is counted in inter-frame intervals, so a length-1 event is only two samples. Peak velocity
can then be the step *into* the event (the frame before onset → onset), while onset→offset
amplitude starts at the onset sample and misses that step. Longer saccades have interior samples,
so peak velocity already sits inside the marked interval and onset→offset amplitude already spans
the movement; they do not need the pre-onset correction. Length-1 events are kept.

**(a)** All saccades. **(b)** Concurrent. **(c)** Monocular. **(d)** Pooled scatter (mixed
amplitude). **(e)** Per-animal OLS slopes concurrent vs monocular and paired t-test. **(f)** Animal
colour legend.

"""
        + slope_txt,
    )
    manifest.append("S13  pre-onset A means + scatter + paired slope test")

    _write(AGG / "MANIFEST.md", "\n".join(["- " + m if not m.startswith("#") else m for m in manifest]))
    print(f"wrote {AGG}")
    print("\n".join(manifest))
    return 0


if __name__ == "__main__":
    if "--s1-s3-s8-only" in sys.argv:
        raise SystemExit(update_s1_s3_s8())
    raise SystemExit(main())
