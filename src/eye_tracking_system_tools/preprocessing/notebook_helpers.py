"""
Notebook helpers promoted out of the preprocessing notebooks.

This module is the one stop import for helpers that used to be defined inline
inside the following Jupyter notebooks under
``src/eye_tracking_system_tools/preprocessing/``:

* ``block_synchronization.ipynb`` (Cell 1, the big "simple approach" cell)
* ``add_accelerometer_state_annotations.ipynb`` (Cells 1, 5, 6)

The Preprocessing GUI (under ``annotation/preprocessing_gui``) cannot import
from a notebook, so these helpers had to live in a real Python module before
the GUI could wrap them.

Wherever a helper already exists in :mod:`block_sync_core`, this module
simply re-exports it so callers (both the notebooks and the GUI) have a
single, stable import path::

    from eye_tracking_system_tools.preprocessing.notebook_helpers import (
        simple_sync_build, build_arena_grid_df, build_final_sync_df_merge_nearest,
        verify_final_df_against_sources, export_final_sync_df, load_final_sync_df,
        plot_simple_sync_bokeh, sanity_plot_final_df,
        insert_dup_by_pos, insert_dup_by_oe_sample,
        find_jittery_frames, export_eye_data_2d,
        rolling_window_analysis, create_behavior_df,
    )

The helpers below are byte-for-byte copies of the original notebook code so the
notebooks continue to produce identical outputs after they are switched over to
importing from this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence, Union
from itertools import cycle

import numpy as np
import pandas as pd

# Re-exports from block_sync_core. These were originally inline in
# block_synchronization.ipynb Cell 1 and have since been promoted into
# block_sync_core.py for downstream code; we forward them here so the
# notebook only needs one import line.
from eye_tracking_system_tools.preprocessing.block_sync_core import (
    _normalize_to_seconds,
    _read_eye_internal_seconds,
    _locate_eye_timestamps_csv,
    _delta_analysis,
    _first_ttl_sample,
    _get_fs,
    _assert_strictly_increasing,
    _nearest_with_tol,
    _shift_eye_df_by_index,
    build_eye_df_simple,
    simple_sync_build,
    describe_eye_tick,
    shift_eye_df_by_index,
    build_arena_grid_df,
    ArenaGridInfo,
    _infer_ttl_fps,
    _build_arena_grid,
    build_final_sync_df_merge_nearest,
    verify_final_df_against_sources,
    export_final_sync_df,
    load_final_sync_df,
    find_jittery_frames,
    add_intermediate_elements,
    export_eye_data_2d,
    create_distance_plot,
)


# ============================================================================
# Bokeh visualization helpers (originally inline in block_synchronization.ipynb)
# ============================================================================
#
# These are kept separate from block_sync_core.py on purpose: they require
# Bokeh and are only used by the notebooks and by the Preprocessing GUI's
# shift-correction tab. block_sync_core stays import-light for headless use.


def plot_simple_sync_bokeh(
    block,
    df_left=None,
    df_right=None,
    shift_range: int = 200,
    show_led: bool = True,
    to_browser: bool = True,
):
    """Open the slider-based shift-correction Bokeh plot.

    Plots both eyes' brightness against Open Ephys time (seconds) with two
    JS sliders that index-shift the traces in place. Used in Stage 1.8 of
    the synchronization pipeline (and by the Preprocessing GUI's
    "Open shift plot" button).

    Parameters mirror the notebook implementation; see
    ``block_synchronization.ipynb`` Cell 1 for the original.
    """
    from bokeh.io import output_notebook, reset_output, show
    from bokeh.layouts import column, row
    from bokeh.models import ColumnDataSource, CustomJS, Span
    from bokeh.plotting import figure
    from bokeh.resources import INLINE

    try:
        from bokeh.models import Slider
    except ImportError:
        from bokeh.models.widgets import Slider

    if df_left is None or df_right is None:
        df_left, df_right = simple_sync_build(block, export=False)

    fs = _get_fs(block)

    def _nan2none(a):
        return [None if (not np.isfinite(v)) else float(v) for v in a]

    xL = df_left["oe_time_s"].to_numpy(dtype=float)
    yL = df_left["brightness"].to_numpy(dtype=float)
    xR = df_right["oe_time_s"].to_numpy(dtype=float)
    yR = df_right["brightness"].to_numpy(dtype=float)

    stepL_ms = float(np.median(np.diff(xL)) * 1000.0) if len(xL) > 1 else float("nan")
    stepR_ms = float(np.median(np.diff(xR)) * 1000.0) if len(xR) > 1 else float("nan")
    print(
        f"[INFO] Slider tick ~ {stepL_ms:.3f} ms (Left), {stepR_ms:.3f} ms (Right)"
    )

    src_le = ColumnDataSource(dict(x=xL.tolist(), y=_nan2none(yL), y0=_nan2none(yL)))
    src_re = ColumnDataSource(dict(x=xR.tolist(), y=_nan2none(yR), y0=_nan2none(yR)))

    p = figure(
        title=(
            "Simple synchronization - brightness vs OE time (s)  "
            "(zoom/pan; use sliders to shift)"
        ),
        x_axis_label="OE time (s)",
        y_axis_label="Brightness (a.u.)",
        width=1200,
        height=450,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    p.line("x", "y", source=src_le, line_width=1.5, color="#1f77b4", legend_label="Left eye")
    p.line("x", "y", source=src_re, line_width=1.5, color="#d62728", legend_label="Right eye")
    p.legend.click_policy = "hide"

    if show_led and ("LED_driver" in getattr(block, "oe_events", {}).columns):
        led = block.oe_events["LED_driver"].dropna().astype(int).to_numpy()
        if led.size:
            led_s = led / fs
            for x in led_s:
                p.add_layout(
                    Span(
                        location=float(x),
                        dimension="height",
                        line_color="#2ca02c",
                        line_alpha=0.5,
                        line_width=1.5,
                    )
                )

    sL = Slider(
        title="Left Eye Shift (indices)",
        start=-shift_range,
        end=shift_range,
        value=0,
        step=1,
        width=350,
    )
    sR = Slider(
        title="Right Eye Shift (indices)",
        start=-shift_range,
        end=shift_range,
        value=0,
        step=1,
        width=350,
    )

    cb = CustomJS(
        args=dict(le=src_le, re=src_re),
        code="""
        const sL = sL_slider.value|0;
        const sR = sR_slider.value|0;

        // LEFT
        const yL  = le.data['y'];
        const yL0 = le.data['y0'];
        const NL  = yL.length;
        for (let i=0; i<NL; i++) {
            const j = i + sL;
            yL[i] = (j>=0 && j<NL) ? yL0[j] : null;
        }
        le.change.emit();

        // RIGHT
        const yR  = re.data['y'];
        const yR0 = re.data['y0'];
        const NR  = yR.length;
        for (let i=0; i<NR; i++) {
            const j = i + sR;
            yR[i] = (j>=0 && j<NR) ? yR0[j] : null;
        }
        re.change.emit();
        """,
    )
    cb.args["sL_slider"] = sL
    cb.args["sR_slider"] = sR
    sL.js_on_change("value", cb)
    sR.js_on_change("value", cb)

    layout = column(p, row(sL, sR))

    reset_output()
    if to_browser:
        show(layout)
    else:
        output_notebook(resources=INLINE, hide_banner=True)
        show(layout)


def hover_inspect_eyes_bokeh(
    df_left,
    df_right,
    title: str = "Hover inspect - per-eye stream on OE time",
):
    """Hover-tooltip Bokeh plot for inspecting per-frame eye stream values.

    Inspect ``oe_sample``, ``oe_time_s``, ``frame_idx`` and ``brightness`` at
    each point. Works on dataframes produced by :func:`simple_sync_build` or
    shifted versions thereof.
    """
    from bokeh.io import output_notebook, show
    from bokeh.layouts import column
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    output_notebook()

    def _prep(df, label):
        d = df.sort_index().copy()
        return ColumnDataSource(
            dict(
                oe_time_s=d["oe_time_s"].to_numpy(dtype=float),
                brightness=d["brightness"].to_numpy(dtype=float),
                frame_idx=d["frame_idx"].to_numpy(dtype=float),
                oe_sample=d.index.to_numpy(dtype=np.int64),
                label=np.array([label] * len(d), dtype=object),
            )
        )

    srcL = _prep(df_left, "LEFT")
    srcR = _prep(df_right, "RIGHT")

    p = figure(
        title=title,
        x_axis_label="OE time (s)",
        y_axis_label="Brightness (a.u.)",
        width=1200,
        height=450,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    rL = p.line(
        "oe_time_s",
        "brightness",
        source=srcL,
        line_width=1.5,
        color="#1f77b4",
        legend_label="Left eye",
    )
    rR = p.line(
        "oe_time_s",
        "brightness",
        source=srcR,
        line_width=1.5,
        color="#d62728",
        legend_label="Right eye",
    )
    p.legend.click_policy = "hide"

    htL = HoverTool(
        renderers=[rL],
        mode="vline",
        tooltips=[
            ("eye", "@label"),
            ("oe_time_s", "@oe_time_s{0.000}"),
            ("oe_sample", "@oe_sample{0}"),
            ("frame_idx", "@frame_idx{0}"),
            ("brightness", "@brightness{0.00}"),
        ],
    )
    htR = HoverTool(
        renderers=[rR],
        mode="vline",
        tooltips=[
            ("eye", "@label"),
            ("oe_time_s", "@oe_time_s{0.000}"),
            ("oe_sample", "@oe_sample{0}"),
            ("frame_idx", "@frame_idx{0}"),
            ("brightness", "@brightness{0.00}"),
        ],
    )
    p.add_tools(htL, htR)
    show(column(p))


def sanity_plot_final_df(
    final_df,
    fs: float,
    show_led_off: bool = False,
    led_off_samples: Optional[np.ndarray] = None,
    title: str = "final_df sanity",
):
    """Bokeh sanity plot of the final synchronized dataframe.

    The Preprocessing GUI re-implements an equivalent plot natively with
    pyqtgraph; this function is preserved so notebook users get the same
    Bokeh-in-notebook experience as before.
    """
    from bokeh.io import output_notebook, show
    from bokeh.models import ColumnDataSource, Span
    from bokeh.plotting import figure

    x_s = np.asarray(final_df["Arena_TTL"], dtype=float) / fs
    yL = np.asarray(final_df["L_values"], dtype=float)
    yR = np.asarray(final_df["R_values"], dtype=float)

    output_notebook()
    p = figure(
        title=title,
        x_axis_label="OE time (s)",
        y_axis_label="Brightness (a.u.)",
        width=1200,
        height=450,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    src = ColumnDataSource(dict(x=x_s, yL=yL, yR=yR))
    p.line("x", "yL", source=src, line_width=1.5, color="#1f77b4", legend_label="Left eye")
    p.line("x", "yR", source=src, line_width=1.5, color="#d62728", legend_label="Right eye")
    p.legend.click_policy = "hide"

    if show_led_off and led_off_samples is not None and len(led_off_samples):
        for s in led_off_samples:
            p.add_layout(
                Span(
                    location=float(s) / fs,
                    dimension="height",
                    line_color="#2ca02c",
                    line_alpha=0.5,
                    line_width=1.5,
                )
            )
    show(p)


def bokeh_plotter(
    data_list,
    label_list,
    plot_name: str = "default",
    x_axis: str = "X",
    y_axis: str = "Y",
    peaks=None,
    peaks_list: bool = False,
    export_path=False,
):
    """Lightweight multi-trace Bokeh line plot with optional peak markers.

    Original lives in ``add_accelerometer_state_annotations.ipynb`` Cell 1
    and is also referenced (without import) by the jitter review step in
    ``block_synchronization.ipynb``. Promoted here so both notebooks (and
    the GUI) can share it.
    """
    import bokeh
    import bokeh.io
    import bokeh.plotting

    color_cycle = cycle(bokeh.palettes.Category10_10)
    fig = bokeh.plotting.figure(
        title=f"bokeh explorer: {plot_name}",
        x_axis_label=x_axis,
        y_axis_label=y_axis,
        width=1500,
        height=700,
    )

    data_vector = None
    for i, vec in enumerate(range(len(data_list))):
        color = next(color_cycle)
        data_vector = data_list[vec]
        if label_list is None:
            fig.line(
                range(len(data_vector)),
                data_vector,
                line_color=color,
                legend_label=f"Line {len(fig.renderers)}",
            )
        elif len(label_list) == len(data_list):
            fig.line(
                range(len(data_vector)),
                data_vector,
                line_color=color,
                legend_label=f"{label_list[i]}",
            )
        if peaks is not None and peaks_list is True:
            fig.circle(peaks[i], data_vector[peaks[i]], size=10, color=color)

    if peaks is not None and peaks_list is False and data_vector is not None:
        fig.circle(peaks, data_vector[peaks], size=10, color="red")

    if export_path is not False:
        print(f"exporting to {export_path}")
        bokeh.io.output.output_file(
            filename=str(Path(export_path) / f"{plot_name}.html"),
            title=f"{plot_name}",
        )
    bokeh.plotting.show(fig)


# ============================================================================
# Dropped-frame correction (insertion) helpers
# ============================================================================
# Originally inline in block_synchronization.ipynb Cell 1. These do not depend
# on Bokeh and are usable from any context (notebook, GUI, batch script).

InsertMode = Literal["prev", "current"]


def _normalize_insert_positions(
    df: pd.DataFrame,
    insert_at: Sequence[Union[int, np.integer]],
    *,
    mode: Literal["pos", "oe_sample"] = "pos",
) -> np.ndarray:
    """Map user-supplied insertion points to 0..N-1 row positions in df."""
    n = len(df)
    if n == 0:
        return np.array([], dtype=int)

    insert_at = np.asarray(insert_at, dtype=np.int64)

    if mode == "pos":
        pos = np.clip(insert_at, 0, n - 1)
        return pos.astype(int)

    if mode == "oe_sample":
        idx = df.sort_index().index.to_numpy(dtype=np.int64)
        p = np.searchsorted(idx, insert_at, side="left")
        p0 = np.clip(p - 1, 0, n - 1)
        p1 = np.clip(p, 0, n - 1)
        d0 = np.abs(idx[p0] - insert_at)
        d1 = np.abs(idx[p1] - insert_at)
        pos = np.where(d0 <= d1, p0, p1)
        return pos.astype(int)

    raise ValueError("mode must be 'pos' or 'oe_sample'.")


def insert_duplicate_frames_slide(
    df: pd.DataFrame,
    insert_at: Sequence[Union[int, np.integer]],
    *,
    mode: Literal["pos", "oe_sample"] = "pos",
    duplicate: InsertMode = "prev",
    cols: Sequence[str] = ("frame_idx", "brightness"),
    leave_trailing_nan: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Insert duplicated frames at given positions, sliding content forward.

    The OE time axis (``df.index`` and ``oe_time_s``) is preserved; only the
    ``cols`` are shifted by one row at each insertion. This is the function
    invoked by the notebook's "frame insertion (advanced)" cell when manual
    correction of dropped frames is required.
    """
    if len(df) == 0:
        return df.copy()

    d = df.sort_index().copy()

    pos = _normalize_insert_positions(d, insert_at, mode=mode)
    pos = np.sort(pos)

    if verbose:
        print(
            f"[INFO] Will insert {len(pos)} duplicated frame(s) at positions: "
            f"{pos.tolist()} (mode={mode}, duplicate={duplicate})"
        )

    out = d.copy()
    for c in cols:
        if c not in out.columns:
            raise ValueError(f"Column '{c}' not in df.")
        out[c] = out[c].to_numpy(dtype=float)

    for k in pos:
        for c in cols:
            arr = out[c].to_numpy(dtype=float)

            if duplicate == "prev":
                src_val = arr[k - 1] if k > 0 else np.nan
            else:
                src_val = arr[k]

            arr[k + 1:] = arr[k:-1]
            arr[k] = src_val

            if leave_trailing_nan:
                arr[-1] = np.nan

            out[c] = arr

    return out


def insert_dup_by_pos(df, positions, **kwargs):
    """Convenience wrapper: insert duplicates by row position."""
    return insert_duplicate_frames_slide(df, positions, mode="pos", **kwargs)


def insert_dup_by_oe_sample(df, oe_samples, **kwargs):
    """Convenience wrapper: insert duplicates by OE sample timestamp."""
    return insert_duplicate_frames_slide(df, oe_samples, mode="oe_sample", **kwargs)


# ============================================================================
# Accelerometer / behavior state helpers
# ============================================================================
# Originally inline in add_accelerometer_state_annotations.ipynb Cells 5 and 6.


def rolling_window_analysis(
    df: pd.DataFrame,
    window_size: int = 10000,
    step_size: int = 1000,
) -> pd.DataFrame:
    """Compute a rolling mean of ``movAll`` over fixed-size time windows.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``t_mov_ms`` (timestamps in ms) and ``movAll`` columns,
        as produced by ``block.block_get_lizard_movement()``.
    window_size : int
        Window length in milliseconds (default 10 s).
    step_size : int
        Step between successive window starts in milliseconds (default 1 s).

    Returns
    -------
    pd.DataFrame
        Columns: ``window_start`` (ms) and ``average_movAll``.
    """
    df = df.sort_values("t_mov_ms").reset_index(drop=True)

    t_min = df["t_mov_ms"].min()
    t_max = df["t_mov_ms"].max()

    window_starts = np.arange(t_min, t_max + step_size, step_size)

    results = {"window_start": [], "average_movAll": []}

    for start in window_starts:
        end = start + window_size
        window_data = df[(df["t_mov_ms"] >= start) & (df["t_mov_ms"] < end)]
        avg_movAll = window_data["movAll"].mean() if not window_data.empty else 0
        results["window_start"].append(start)
        results["average_movAll"].append(avg_movAll)

    return pd.DataFrame(results)


def create_behavior_df(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-window ``behavior`` annotation into start/end segments.

    The input must have ``window_start`` and ``behavior`` columns (typically
    produced by thresholding the output of :func:`rolling_window_analysis`).

    Returns a tidy dataframe with ``start_time``, ``end_time`` and
    ``annotation`` (in milliseconds; final segment extends 1000 ms past the
    last window start to capture the trailing window).
    """
    behavior_df = []
    current_behavior = df["behavior"].iloc[0]
    start_time = df["window_start"].iloc[0]

    for i in range(1, len(df)):
        if df["behavior"].iloc[i] != current_behavior:
            end_time = df["window_start"].iloc[i]
            behavior_df.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "annotation": current_behavior,
                }
            )
            current_behavior = df["behavior"].iloc[i]
            start_time = df["window_start"].iloc[i]

    end_time = df["window_start"].iloc[-1] + 1000
    behavior_df.append(
        {
            "start_time": start_time,
            "end_time": end_time,
            "annotation": current_behavior,
        }
    )

    return pd.DataFrame(behavior_df)


__all__ = [
    # re-exports from block_sync_core
    "_normalize_to_seconds",
    "_read_eye_internal_seconds",
    "_locate_eye_timestamps_csv",
    "_delta_analysis",
    "_first_ttl_sample",
    "_get_fs",
    "_assert_strictly_increasing",
    "_nearest_with_tol",
    "_shift_eye_df_by_index",
    "build_eye_df_simple",
    "simple_sync_build",
    "describe_eye_tick",
    "shift_eye_df_by_index",
    "build_arena_grid_df",
    "ArenaGridInfo",
    "_infer_ttl_fps",
    "_build_arena_grid",
    "build_final_sync_df_merge_nearest",
    "verify_final_df_against_sources",
    "export_final_sync_df",
    "load_final_sync_df",
    "find_jittery_frames",
    "add_intermediate_elements",
    "export_eye_data_2d",
    "create_distance_plot",
    # bokeh helpers
    "plot_simple_sync_bokeh",
    "hover_inspect_eyes_bokeh",
    "sanity_plot_final_df",
    "bokeh_plotter",
    # insertion helpers
    "_normalize_insert_positions",
    "insert_duplicate_frames_slide",
    "insert_dup_by_pos",
    "insert_dup_by_oe_sample",
    "InsertMode",
    # accelerometer / behavior helpers
    "rolling_window_analysis",
    "create_behavior_df",
]
