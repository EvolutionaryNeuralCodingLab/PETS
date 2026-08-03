"""Figures 3a–3c: single-block eye-angle / pupil / saccade-rate vignettes.

Fig 3a/3b are short quiet-epoch φ/θ exemplars; Fig 3c is the longer combined
panel (φ, θ, pupil diameter, saccade+head rate) with the behavior-state strip
overlaid. All three drive
:func:`eye_tracking_system_tools.figures.plotting_functions.plot_zoomed_in_with_head_rate`
off a single :class:`~eye_tracking_system_tools.analysis.pipeline.BlockBundle`,
following the ``EventTables``-driven pattern of ``figures_2g_2j.py``
(:func:`resolve_figure_dirs`, :func:`write_pickle_with_meta`, :func:`show_and_close`).

Requires per-frame eye traces (``BlockBundle.left`` / ``.right``) — build
``EventTables`` with ``keep_traces=True`` before calling into this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.behavior_state import (
    has_behavior_state,
    read_behavior_state,
)
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.pipeline import BlockBundle, EventTables
from eye_tracking_system_tools.analysis.pixel_calibration import read_pixel_size
from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs
from eye_tracking_system_tools.figures.plotting_functions import (
    plot_zoomed_in_with_head_rate,
)

logger = logging.getLogger(__name__)

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

# Paper vignette source block (present in configs/paper_blocks.yaml).
PREFERRED_BLOCK_KEY = "PV_126_block_007"

_PLOT_KWARG_KEYS = (
    "std_multiplier",
    "head_merge_ms",
    "head_merge_strategy",
    "plot_state_y",
    "x_zero_origin",
    "pupil_ticks",
    "phi_theta_ticks",
    "phi_ticks",
    "theta_ticks",
    "window_size",
    "bin_size",
)

_DEFAULTS: dict[str, dict[str, Any]] = {
    "figure_3a": {
        "start_s": 210.0,
        "end_s": 240.0,
        "traces": ["center_x", "center_y"],
        "figsize": (2.3, 1.8),
        "std_multiplier": 3,
        "with_state": False,
    },
    "figure_3b": {
        "start_s": 310.0,
        "end_s": 340.0,
        "traces": ["center_x", "center_y"],
        "figsize": (2.3, 1.8),
        "std_multiplier": 3,
        "with_state": False,
    },
    "figure_3c": {
        "start_s": 200.0,
        "end_s": 415.0,
        "traces": ["center_x", "center_y", "pupil_diameter", "saccade_frequency"],
        "figsize": (4.7, 3.7),
        "std_multiplier": 3.5,
        "with_state": True,
        "head_merge_ms": 20,
        "plot_state_y": 20,
        "x_zero_origin": True,
        "pupil_ticks": [1.5, 2.0, 2.5],
    },
}


def _pick_block_key(tables: EventTables, block_key: str | None) -> str:
    """Explicit ``block_key`` wins; else prefer :data:`PREFERRED_BLOCK_KEY`; else first block."""
    available = {b.spec.block_key: b for b in tables.blocks}
    if not available:
        raise ValueError("EventTables has no blocks to draw a vignette from.")
    if block_key is not None:
        if block_key not in available:
            raise KeyError(f"block_key {block_key!r} not found; available={sorted(available)}")
        return block_key
    if PREFERRED_BLOCK_KEY in available:
        return PREFERRED_BLOCK_KEY
    return next(iter(available))


def _recenter_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract each column's own nanmedian so φ/θ vignettes are centered at 0."""
    out = df.copy()
    for col in ("k_phi", "k_theta"):
        if col in out.columns:
            vals = out[col].to_numpy(dtype=float)
            med = np.nanmedian(vals)
            if np.isfinite(med):
                out[col] = vals - med
    return out


def _ensure_pupil_mm(df: pd.DataFrame, mm_per_px: float | None) -> pd.DataFrame:
    """Overwrite ``pupil_diameter`` with the mm-converted ``major_ax`` when calibration is known.

    Mirrors ``figures_3e_3f_pupil._pupil_diameter_mm``: ``major_ax`` is always
    pixels, so it takes priority over any pre-existing ``pupil_diameter``
    column (which ``eye_trace_io`` only aliases from ``major_ax`` verbatim).
    """
    out = df.copy()
    if "major_ax" in out.columns and mm_per_px is not None:
        out["pupil_diameter"] = out["major_ax"].to_numpy(dtype=float) * float(mm_per_px)
    return out


def _head_movement_ms(bundle: BlockBundle) -> np.ndarray:
    """Best-effort head-movement onset times (ms); empty array when unavailable.

    Prefers ``lizMov.mat`` (see ``head_labels.find_lizmov_mat``) so the
    saccade/head-rate panel matches the paper notebook; falls back to an empty
    array (flat head-rate trace) rather than failing the whole figure.
    """
    from eye_tracking_system_tools.analysis.head_labels import (
        find_lizmov_mat,
        load_lizmov_times_ms,
    )

    mat_path = find_lizmov_mat(bundle.spec)
    if mat_path is None:
        logger.info(
            "[%s] no lizMov.mat under oe_files/; head-movement rate will be all-zero",
            bundle.spec.block_key,
        )
        return np.array([], dtype=float)
    try:
        return load_lizmov_times_ms(mat_path)
    except Exception as exc:
        logger.warning(
            "[%s] failed reading %s (%s); head-movement rate will be all-zero",
            bundle.spec.block_key,
            mat_path,
            exc,
        )
        return np.array([], dtype=float)


def _export_vignette(
    tables: EventTables,
    out_dir: Path,
    *,
    figure_name: str,
    show: bool,
    block_key: str | None,
    start_s: float | None,
    end_s: float | None,
) -> dict[str, Path]:
    top_cfg = dict(tables.params.get("figure_3a_3c", {}))
    cfg = {**_DEFAULTS[figure_name], **dict(top_cfg.get(figure_name, {}))}
    start_time = float(start_s) if start_s is not None else float(cfg["start_s"])
    end_time = float(end_s) if end_s is not None else float(cfg["end_s"])
    traces = list(cfg["traces"])

    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)

    resolved_key = block_key or top_cfg.get("block_key")
    key = _pick_block_key(tables, resolved_key)
    bundle = tables.block_dict[key]
    spec = bundle.spec

    if bundle.left is None or bundle.right is None or bundle.left.empty or bundle.right.empty:
        raise ValueError(
            f"[{key}] no eye traces loaded; build EventTables with keep_traces=True"
        )

    left_df = _recenter_angles(bundle.left)
    right_df = _recenter_angles(bundle.right)

    pix = read_pixel_size(spec.block_path)
    if pix is None:
        logger.warning("[%s] no LR_pix_size.csv; pupil_diameter left in raw units", key)
    left_df = _ensure_pupil_mm(left_df, pix.l_mm_per_px if pix else None)
    right_df = _ensure_pupil_mm(right_df, pix.r_mm_per_px if pix else None)

    left_ms = (
        bundle.l_saccades["saccade_on_ms"].to_numpy(dtype=float)
        if bundle.l_saccades is not None and not bundle.l_saccades.empty
        else np.array([], dtype=float)
    )
    right_ms = (
        bundle.r_saccades["saccade_on_ms"].to_numpy(dtype=float)
        if bundle.r_saccades is not None and not bundle.r_saccades.empty
        else np.array([], dtype=float)
    )

    behavior_state_df = None
    if bool(cfg.get("with_state", False)) and has_behavior_state(spec):
        behavior_state_df = read_behavior_state(spec)

    head_ms = _head_movement_ms(bundle) if "saccade_frequency" in traces else np.array([], dtype=float)

    plot_kwargs = {k: cfg[k] for k in _PLOT_KWARG_KEYS if k in cfg}

    out_pdf = figures_dir / f"{figure_name}.pdf"
    plot_zoomed_in_with_head_rate(
        start_time,
        end_time,
        traces=traces,
        left_df=left_df,
        right_df=right_df,
        left_ms=left_ms,
        right_ms=right_ms,
        head_movements_ms=head_ms,
        behavior_state_df=behavior_state_df,
        behavior_time_unit="ms",
        figure_size=tuple(cfg.get("figsize", (2.3, 1.8))),
        export_path=out_pdf,
        show=show,
        **plot_kwargs,
    )

    pkl = metadata_dir / f"{figure_name}_data.pickle"
    write_pickle_with_meta(
        {
            "figure_name": figure_name,
            "block_key": key,
            "start_s": start_time,
            "end_s": end_time,
            "traces": traces,
            "params": cfg,
        },
        pkl,
        meta={
            "csv_choices": tables.csv_meta,
            "params": cfg,
            "figure": figure_name,
            "block_key": key,
        },
        entrypoint=f"eye_tracking_system_tools.analysis.figures_3a_3c_vignettes.export_{figure_name}",
    )

    return {f"{figure_name}.pdf": out_pdf, f"{figure_name}_data.pickle": pkl}


def export_figure_3a(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
    block_key: str | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> dict[str, Path]:
    """Fig 3a: quiet-epoch φ/θ vignette (paper default: PV_126_block_007, 210–240 s)."""
    return _export_vignette(
        tables, out_dir, figure_name="figure_3a", show=show,
        block_key=block_key, start_s=start_s, end_s=end_s,
    )


def export_figure_3b(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
    block_key: str | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> dict[str, Path]:
    """Fig 3b: second quiet-epoch φ/θ vignette (paper default: PV_126_block_007, 310–340 s)."""
    return _export_vignette(
        tables, out_dir, figure_name="figure_3b", show=show,
        block_key=block_key, start_s=start_s, end_s=end_s,
    )


def export_figure_3c(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
    block_key: str | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> dict[str, Path]:
    """Fig 3c: φ/θ/pupil/saccade-rate vignette with behavior-state strip (200–415 s)."""
    return _export_vignette(
        tables, out_dir, figure_name="figure_3c", show=show,
        block_key=block_key, start_s=start_s, end_s=end_s,
    )
