# ============================================================================
# Block synchronization visualization and interactive sync correction tool
# ============================================================================
# Moved from block_synchronization.ipynb for maintainability.
# Import: from eye_tracking_system_tools.preprocessing.block_sync_visualization import (
#     plot_simple_sync_bokeh, hover_inspect_eyes_bokeh, sanity_plot_final_df,
#     insert_dup_by_pos, insert_dup_by_oe_sample, remove_frame_at_pos,
#     interactive_sync_tool_bokeh, plot_sync_verification_with_electrophys,
#     plot_led_off_events_viewer, ...)
# ============================================================================

from __future__ import annotations

from typing import Optional, Literal, Sequence, Union

import numpy as np
import pandas as pd

from bokeh.io import output_notebook, show, reset_output
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, Span, HoverTool
try:
    from bokeh.models import Slider
    from bokeh.resources import INLINE
except Exception:
    from bokeh.models.widgets import Slider
    from bokeh.resources import INLINE
from bokeh.layouts import column, row


def _get_fs(block) -> float:
    """Return Open Ephys sample rate (Hz)."""
    fs = getattr(block, "sample_rate", None)
    if fs is None:
        fs = float(block.get_sample_rate())
        block.sample_rate = fs
    return float(fs)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_simple_sync_bokeh(
    block,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    shift_range: int = 200,
    show_led: bool = True,
    to_browser: bool = True,
):
    """
    Bokeh plot of both eyes' brightness vs Open Ephys time (seconds), with manual shift sliders.
    df_left and df_right must be from simple_sync_build() or shifted versions (index=oe_sample, cols: oe_time_s, brightness, frame_idx).
    """
    fs = _get_fs(block)

    def _nan2none(a):
        return [None if (not np.isfinite(v)) else float(v) for v in a]

    xL = df_left["oe_time_s"].to_numpy(dtype=float)
    yL = df_left["brightness"].to_numpy(dtype=float)
    xR = df_right["oe_time_s"].to_numpy(dtype=float)
    yR = df_right["brightness"].to_numpy(dtype=float)

    stepL_ms = float(np.median(np.diff(xL)) * 1000.0) if len(xL) > 1 else float("nan")
    stepR_ms = float(np.median(np.diff(xR)) * 1000.0) if len(xR) > 1 else float("nan")
    print(f"[INFO] Slider tick ≈ {stepL_ms:.3f} ms (Left), {stepR_ms:.3f} ms (Right)")

    src_le = ColumnDataSource(dict(x=xL.tolist(), y=_nan2none(yL), y0=_nan2none(yL)))
    src_re = ColumnDataSource(dict(x=xR.tolist(), y=_nan2none(yR), y0=_nan2none(yR)))

    p = figure(
        title="Simple synchronization — brightness vs OE time (s)  (zoom/pan; use sliders to shift)",
        x_axis_label="OE time (s)",
        y_axis_label="Brightness (a.u.)",
        width=1200,
        height=450,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p.line("x", "y", source=src_le, line_width=1.5, color="#1f77b4", legend_label="Left eye")
    p.line("x", "y", source=src_re, line_width=1.5, color="#d62728", legend_label="Right eye")
    p.legend.click_policy = "hide"

    if show_led and hasattr(block, "oe_events") and block.oe_events is not None:
        if "LED_driver" in block.oe_events.columns:
            led_rising = block.oe_events["LED_driver"].dropna().astype(int).to_numpy()
            if led_rising.size:
                led_rising_s = led_rising / fs
                for x in led_rising_s:
                    p.add_layout(
                        Span(
                            location=float(x),
                            dimension="height",
                            line_color="#2ca02c",
                            line_alpha=0.7,
                            line_width=2,
                            line_dash="solid",
                        )
                    )
        if "LED_driver_fall" in block.oe_events.columns:
            led_falling = block.oe_events["LED_driver_fall"].dropna().astype(int).to_numpy()
            if led_falling.size:
                led_falling_s = led_falling / fs
                for x in led_falling_s:
                    p.add_layout(
                        Span(
                            location=float(x),
                            dimension="height",
                            line_color="#d62728",
                            line_alpha=0.7,
                            line_width=2,
                            line_dash="dashed",
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
        const yL  = le.data['y'];
        const yL0 = le.data['y0'];
        const NL  = yL.length;
        for (let i=0; i<NL; i++) {
            const j = i + sL;
            yL[i] = (j>=0 && j<NL) ? yL0[j] : null;
        }
        le.change.emit();
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
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    title: str = "Hover inspect — per-eye stream on OE time",
):
    """Hover over each trace to see oe_sample, oe_time_s, frame_idx, brightness."""
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
        "oe_time_s", "brightness", source=srcL, line_width=1.5, color="#1f77b4", legend_label="Left eye"
    )
    rR = p.line(
        "oe_time_s", "brightness", source=srcR, line_width=1.5, color="#d62728", legend_label="Right eye"
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
    final_df: pd.DataFrame,
    fs: float,
    show_led: bool = True,
    block=None,
    title: str = "final_df sanity",
):
    """Plot final_df L_values/R_values vs OE time with optional LED verticals."""
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

    if show_led and block is not None and hasattr(block, "oe_events") and block.oe_events is not None:
        if "LED_driver" in block.oe_events.columns:
            led_rising = block.oe_events["LED_driver"].dropna().astype(int).to_numpy()
            if len(led_rising) > 0:
                led_rising_s = led_rising / fs
                for x in led_rising_s:
                    p.add_layout(
                        Span(
                            location=float(x),
                            dimension="height",
                            line_color="#2ca02c",
                            line_alpha=0.7,
                            line_width=2,
                            line_dash="solid",
                        )
                    )
        if "LED_driver_fall" in block.oe_events.columns:
            led_falling = block.oe_events["LED_driver_fall"].dropna().astype(int).to_numpy()
            if len(led_falling) > 0:
                led_falling_s = led_falling / fs
                for x in led_falling_s:
                    p.add_layout(
                        Span(
                            location=float(x),
                            dimension="height",
                            line_color="#d62728",
                            line_alpha=0.7,
                            line_width=2,
                            line_dash="dashed",
                        )
                    )
    show(p)


# ============================================================================
# FRAME INSERTION / REMOVAL (for sync correction)
# ============================================================================

InsertMode = Literal["prev", "current"]


def _normalize_insert_positions(
    df: pd.DataFrame,
    insert_at: Sequence[Union[int, np.integer]],
    *,
    mode: Literal["pos", "oe_sample"] = "pos",
) -> np.ndarray:
    """Convert insert points into 0..N-1 row positions. mode='pos' or 'oe_sample'."""
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
    """Insert duplicated frame(s) at given positions; OE time axis unchanged."""
    if len(df) == 0:
        return df.copy()
    d = df.sort_index().copy()
    n = len(d)
    pos = _normalize_insert_positions(d, insert_at, mode=mode)
    pos = np.sort(pos)
    if verbose:
        print(
            f"[INFO] Will insert {len(pos)} duplicated frame(s) at positions: {pos.tolist()} (mode={mode}, duplicate={duplicate})"
        )
    out = d.copy()
    for c in cols:
        if c not in out.columns:
            raise ValueError(f"Column '{c}' not in df.")
        out[c] = out[c].to_numpy(dtype=float)
    for k in pos:
        for c in cols:
            arr = out[c].to_numpy(dtype=float)
            src_val = arr[k - 1] if (duplicate == "prev" and k > 0) else (arr[k] if duplicate == "current" else np.nan)
            if duplicate == "prev" and k == 0:
                src_val = np.nan
            arr[k + 1 :] = arr[k:-1]
            arr[k] = src_val
            if leave_trailing_nan:
                arr[-1] = np.nan
            out[c] = arr
    return out


def insert_dup_by_pos(df: pd.DataFrame, positions, **kwargs) -> pd.DataFrame:
    """Convenience: insert duplicate by row position."""
    return insert_duplicate_frames_slide(df, positions, mode="pos", **kwargs)


def insert_dup_by_oe_sample(df: pd.DataFrame, oe_samples, **kwargs) -> pd.DataFrame:
    """Convenience: insert duplicate by OE sample."""
    return insert_duplicate_frames_slide(df, oe_samples, mode="oe_sample", **kwargs)


def remove_frame_at_pos(
    df: pd.DataFrame,
    positions: Sequence[Union[int, np.integer]],
    *,
    mode: Literal["pos", "oe_sample"] = "pos",
    cols: Sequence[str] = ("frame_idx", "brightness"),
    leave_trailing_nan: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove frame(s) by sliding content backward (relative correction, inverse of duplicate).

    Keeps the same oe_sample grid (same rows, same oe_time_s / ms_axis). For each
    position k, the *content* (frame_idx, brightness) is shifted backward: what
    was at k+1 moves to k, k+2 to k+1, ..., and the last row gets NaN. So the
    relative synchronization changes (like duplicate, but in the opposite
    direction). Use when eye data has shifted forward relative to OE events.
    """
    if len(df) == 0:
        return df.copy()
    d = df.sort_index().copy()
    n = len(d)
    pos = _normalize_insert_positions(d, positions, mode=mode)
    pos = np.unique(np.clip(pos, 0, n - 1))
    if len(pos) == 0:
        return d
    if verbose:
        print(
            f"[INFO] Will remove {len(pos)} frame(s) at positions: {pos.tolist()} "
            f"(slide content backward, mode={mode})"
        )
    out = d.copy()
    for c in cols:
        if c not in out.columns:
            raise ValueError(f"Column '{c}' not in df.")
        out[c] = out[c].to_numpy(dtype=float)
    # Process from lowest position first so each slide sees current state
    for k in sorted(pos):
        for c in cols:
            arr = out[c].to_numpy(dtype=float)
            # Slide backward: arr[k:-1] = arr[k+1:], last gets NaN
            arr[k:-1] = arr[k + 1 :]
            if leave_trailing_nan:
                arr[-1] = np.nan
            out[c] = arr
    return out


# ============================================================================
# INTERACTIVE SYNCHRONIZATION TOOL (duplicate/remove frames, then export)
# ============================================================================

def _make_sync_plot_bokeh(block, dfL: pd.DataFrame, dfR: pd.DataFrame, show_led: bool = True):
    """Build a Bokeh figure of left/right brightness with LED verticals (green dotted rising, red dotted falling) and hover."""
    fs = _get_fs(block)

    def _nan2none(a):
        return [None if (not np.isfinite(v)) else float(v) for v in a]

    xL = dfL["oe_time_s"].to_numpy(dtype=float)
    yL = dfL["brightness"].to_numpy(dtype=float)
    xR = dfR["oe_time_s"].to_numpy(dtype=float)
    yR = dfR["brightness"].to_numpy(dtype=float)

    srcL = ColumnDataSource(
        dict(
            x=xL.tolist(),
            y=_nan2none(yL),
            oe_time_s=dfL["oe_time_s"].to_numpy(dtype=float),
            frame_idx=dfL["frame_idx"].to_numpy(dtype=float),
            oe_sample=dfL.index.to_numpy(dtype=np.int64),
            brightness=dfL["brightness"].to_numpy(dtype=float),
        )
    )
    srcR = ColumnDataSource(
        dict(
            x=xR.tolist(),
            y=_nan2none(yR),
            oe_time_s=dfR["oe_time_s"].to_numpy(dtype=float),
            frame_idx=dfR["frame_idx"].to_numpy(dtype=float),
            oe_sample=dfR.index.to_numpy(dtype=np.int64),
            brightness=dfR["brightness"].to_numpy(dtype=float),
        )
    )

    p = figure(
        title="Sync correction — brightness vs OE time (s). Duplicate/Remove frames below, then Export.",
        x_axis_label="OE time (s)",
        y_axis_label="Brightness (a.u.)",
        width=1200,
        height=450,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    rL = p.line("x", "y", source=srcL, line_width=1.5, color="#1f77b4", legend_label="Left eye")
    rR = p.line("x", "y", source=srcR, line_width=1.5, color="#d62728", legend_label="Right eye")
    p.legend.click_policy = "hide"

    htL = HoverTool(
        renderers=[rL],
        mode="vline",
        tooltips=[
            ("eye", "Left"),
            ("oe_time_s", "@oe_time_s{0.000}"),
            ("frame_idx", "@frame_idx{0}"),
            ("oe_sample", "@oe_sample{0}"),
            ("brightness", "@brightness{0.00}"),
        ],
    )
    htR = HoverTool(
        renderers=[rR],
        mode="vline",
        tooltips=[
            ("eye", "Right"),
            ("oe_time_s", "@oe_time_s{0.000}"),
            ("frame_idx", "@frame_idx{0}"),
            ("oe_sample", "@oe_sample{0}"),
            ("brightness", "@brightness{0.00}"),
        ],
    )
    p.add_tools(htL, htR)

    if show_led and hasattr(block, "oe_events") and block.oe_events is not None:
        if "LED_driver" in block.oe_events.columns:
            led_rising = block.oe_events["LED_driver"].dropna().astype(int).to_numpy()
            if led_rising.size:
                led_rising_s = led_rising / fs
                for x in led_rising_s:
                    p.add_layout(
                        Span(
                            location=float(x),
                            dimension="height",
                            line_color="#2ca02c",
                            line_alpha=0.8,
                            line_width=1.5,
                            line_dash="dotted",
                        )
                    )
        if "LED_driver_fall" in block.oe_events.columns:
            led_falling = block.oe_events["LED_driver_fall"].dropna().astype(int).to_numpy()
            if led_falling.size:
                led_falling_s = led_falling / fs
                for x in led_falling_s:
                    p.add_layout(
                        Span(
                            location=float(x),
                            dimension="height",
                            line_color="#d62728",
                            line_alpha=0.8,
                            line_width=1.5,
                            line_dash="dotted",
                        )
                    )
    return p


def interactive_sync_tool_bokeh(
    block,
    dfL_shifted: pd.DataFrame,
    dfR_shifted: pd.DataFrame,
    show_led: bool = True,
    result_container: Optional[dict] = None,
):
    """
    Interactive Bokeh sync tool: view left/right brightness with LED_driver TTL verticals
    (green dotted = rising, red dotted = falling), hover for frame/oe_sample/brightness.
    Use text box + buttons to Duplicate or Remove a frame at a given eye and frame index (position).
    Plot auto-refreshes after each edit. When satisfied, click Export to write corrected
    dfL_shifted and dfR_shifted into result_container; then call build_final_sync_df_merge_nearest(block, result_container['dfL'], result_container['dfR']).

    Parameters
    ----------
    block : BlockSync-like
    dfL_shifted, dfR_shifted : pd.DataFrame
        Eye dataframes (index=oe_sample, cols: oe_time_s, frame_idx, brightness).
    show_led : bool
        Draw LED_driver rising (green dotted) and falling (red dotted) verticals.
    result_container : dict or None
        If provided, on Export this dict will be updated with keys 'dfL' and 'dfR' (corrected copies).
        So you can do: result = {}; interactive_sync_tool_bokeh(..., result_container=result);
        then after Export: build_final_sync_df_merge_nearest(block, result['dfL'], result['dfR']).

    Returns
    -------
    The ipywidgets VBox (control row + output) so you can display() it in the notebook.
    """
    try:
        from ipywidgets import Button, Dropdown, IntText, HBox, VBox, Output
        from IPython.display import display, clear_output
    except ImportError as e:
        raise ImportError(
            "interactive_sync_tool_bokeh requires ipywidgets and IPython. Install with: pip install ipywidgets"
        ) from e

    if result_container is None:
        result_container = {}

    state = {"dfL": dfL_shifted.copy(), "dfR": dfR_shifted.copy()}

    eye_choices = [("Left", "L"), ("Right", "R")]
    eye_dropdown = Dropdown(options=eye_choices, value="L", description="Eye:")
    frame_index = IntText(
        value=0,
        description="Row position:",
        min=0,
        tooltip="0-based position. Duplicate: insert copy here, content slides forward. Remove: content at this position onward slides backward (last row → NaN), oe_sample grid unchanged.",
    )
    btn_duplicate = Button(description="Duplicate frame")
    btn_remove = Button(description="Remove frame")
    btn_export = Button(description="Export corrected dfs")
    out = Output()

    def _do_refresh():
        p = _make_sync_plot_bokeh(block, state["dfL"], state["dfR"], show_led=show_led)
        show(p)

    try:
        refresh_plot = out.capture(clear_output=True)(_do_refresh)
    except (AttributeError, TypeError):
        def refresh_plot():
            with out:
                clear_output(wait=True)
                _do_refresh()

    def on_duplicate(_):
        eye = "L" if eye_dropdown.value == "L" else "R"
        pos = int(frame_index.value)
        df = state["dfL"] if eye == "L" else state["dfR"]
        n = len(df.sort_index())
        if pos < 0 or pos >= n:
            print(f"[WARN] Frame index {pos} out of range [0, {n-1}]. No change.")
            return
        new_df = insert_dup_by_pos(df, [pos], duplicate="prev", leave_trailing_nan=True, verbose=True)
        if eye == "L":
            state["dfL"] = new_df
        else:
            state["dfR"] = new_df
        refresh_plot()

    def on_remove(_):
        try:
            eye = "L" if eye_dropdown.value == "L" else "R"
            raw_val = frame_index.value
            if raw_val is None or raw_val == "":
                print("[WARN] Frame index is empty. Enter a non-negative integer.")
                return
            pos = int(raw_val)
            df = state["dfL"] if eye == "L" else state["dfR"]
            n = len(df.sort_index())
            if n == 0:
                print("[WARN] No frames to remove.")
                return
            if pos < 0 or pos >= n:
                print(f"[WARN] Frame index {pos} out of range [0, {n-1}]. No change.")
                return
            new_df = remove_frame_at_pos(df, [pos], verbose=True)
            if eye == "L":
                state["dfL"] = new_df
            else:
                state["dfR"] = new_df
            refresh_plot()
        except Exception as e:
            import traceback
            print(f"[ERROR] Remove frame failed: {e}")
            traceback.print_exc()

    def on_export(_):
        result_container["dfL"] = state["dfL"].copy()
        result_container["dfR"] = state["dfR"].copy()
        print("[OK] Exported corrected dfL and dfR to result_container. Use build_final_sync_df_merge_nearest(block, result_container['dfL'], result_container['dfR']) to build the final sync.")

    btn_duplicate.on_click(on_duplicate)
    btn_remove.on_click(on_remove)
    btn_export.on_click(on_export)

    controls = HBox([eye_dropdown, frame_index, btn_duplicate, btn_remove, btn_export])
    layout = VBox([controls, out])
    with out:
        p = _make_sync_plot_bokeh(block, state["dfL"], state["dfR"], show_led=show_led)
        show(p)
    return layout


# ============================================================================
# VERIFICATION PLOTS (electrophys + LED)
# ============================================================================

def plot_sync_verification_with_electrophys(block, channel: int = 1, to_browser: bool = True):
    """
    Bokeh: ephys trace (top) and eye brightness (bottom) with LED TTL verticals (rising green, falling red).
    Requires block.oe_rec and block.final_sync_df.
    """
    from bokeh.layouts import column as bokeh_column
    from bokeh.models import Span as BokehSpan

    fs = block.sample_rate if hasattr(block, "sample_rate") else block.get_sample_rate()
    if not hasattr(block, "oe_rec") or block.oe_rec is None:
        raise ValueError("block.oe_rec is not available. BlockSync should create this automatically.")
    if not hasattr(block, "final_sync_df") or block.final_sync_df is None:
        raise ValueError("final_sync_df is not available. Load it first.")

    df = block.final_sync_df
    arena_ttl_samples = df["Arena_TTL"].dropna().astype(np.int64).values
    arena_ttl_seconds = arena_ttl_samples / fs
    l_brightness = df["L_values"].values
    r_brightness = df["R_values"].values

    led_rising_samples = np.array([])
    led_falling_samples = np.array([])
    if hasattr(block, "oe_events") and block.oe_events is not None:
        if "LED_driver" in block.oe_events.columns:
            led_rising_samples = block.oe_events["LED_driver"].dropna().astype(np.int64).values
        if "LED_driver_fall" in block.oe_events.columns:
            led_falling_samples = block.oe_events["LED_driver_fall"].dropna().astype(np.int64).values

    led_rising_seconds = led_rising_samples / fs if len(led_rising_samples) > 0 else np.array([])
    led_falling_seconds = led_falling_samples / fs if len(led_falling_samples) > 0 else np.array([])

    print(f"LED TTL events: {len(led_rising_samples)} rising edges, {len(led_falling_samples)} falling edges")

    rec_duration_ms = block.oe_rec.recordingDuration_ms
    chunk_duration_ms = 10000.0
    n_chunks = int(np.ceil(rec_duration_ms / chunk_duration_ms))
    ephys_traces = []
    ephys_times_ms = []

    for i in range(n_chunks):
        start_ms = i * chunk_duration_ms
        window_ms = min(chunk_duration_ms, rec_duration_ms - start_ms)
        if window_ms <= 0:
            break
        try:
            start_time_ms = np.array([[start_ms]])
            chunk_data, chunk_timestamps = block.oe_rec.get_data(
                channels=[channel],
                start_time_ms=start_time_ms,
                window_ms=window_ms,
                convert_microvolts=True,
                return_timestamps=True,
                repress_output=True,
            )
            if chunk_data is not None and chunk_timestamps is not None:
                ephys_traces.append(chunk_data[0, 0, :])
                ephys_times_ms.append(chunk_timestamps[0, :])
                if (i + 1) % 10 == 0:
                    print(f"  Extracted chunk {i+1}/{n_chunks}...")
        except Exception as e:
            print(f"  Warning: Error extracting chunk {i+1}: {e}")

    if len(ephys_traces) == 0:
        raise ValueError("Failed to extract any electrophysiology data")

    ephys_trace = np.concatenate(ephys_traces)
    ephys_time_ms = np.concatenate(ephys_times_ms)
    ephys_time_s = ephys_time_ms / 1000.0
    print(f"Extracted {len(ephys_trace)} samples from electrophysiology recording")

    max_downsample = 10
    downsample_factor = min(max_downsample, max(1, len(ephys_trace) // 500000))
    if downsample_factor > 1:
        ephys_trace_ds = ephys_trace[::downsample_factor]
        ephys_time_s_ds = ephys_time_s[::downsample_factor]
        print(f"Downsampled by factor {downsample_factor} for visualization")
    else:
        ephys_trace_ds = ephys_trace
        ephys_time_s_ds = ephys_time_s

    reset_output()
    p_ephys = figure(
        width=1200,
        height=400,
        title=f"Electrophysiology Trace (Channel {channel}) with LED TTL Events",
        x_axis_label="Time (seconds)",
        y_axis_label="Voltage (mV)",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p_ephys.line(
        ephys_time_s_ds, ephys_trace_ds, line_width=1, alpha=0.7, color="black", legend_label="Ephys"
    )
    for x in led_rising_seconds:
        p_ephys.add_layout(
            BokehSpan(
                location=float(x), dimension="height",
                line_color="#2ca02c", line_alpha=0.7, line_width=2, line_dash="solid"
            )
        )
    for x in led_falling_seconds:
        p_ephys.add_layout(
            BokehSpan(
                location=float(x), dimension="height",
                line_color="#d62728", line_alpha=0.7, line_width=2, line_dash="dashed"
            )
        )

    p_eyes = figure(
        width=1200,
        height=400,
        title="Eye Brightness Vectors with LED TTL Events",
        x_axis_label="Time (seconds)",
        y_axis_label="Brightness (arbitrary units)",
        x_range=p_ephys.x_range,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p_eyes.line(arena_ttl_seconds, l_brightness, line_width=2, alpha=0.8, color="blue", legend_label="Left Eye")
    p_eyes.line(arena_ttl_seconds, r_brightness, line_width=2, alpha=0.8, color="red", legend_label="Right Eye")
    for x in led_rising_seconds:
        p_eyes.add_layout(
            BokehSpan(
                location=float(x), dimension="height",
                line_color="#2ca02c", line_alpha=0.7, line_width=2, line_dash="solid"
            )
        )
    for x in led_falling_seconds:
        p_eyes.add_layout(
            BokehSpan(
                location=float(x), dimension="height",
                line_color="#d62728", line_alpha=0.7, line_width=2, line_dash="dashed"
            )
        )

    layout = bokeh_column(p_ephys, p_eyes)
    if to_browser:
        show(layout)
    else:
        return layout


def plot_led_off_events_viewer(
    block, channel: int = 1, window_half_s: float = 0.25, to_browser: bool = True
):
    """
    Interactive viewer: windows around each LED_driver falling edge; Prev/Next to step.
    Full-resolution ephys + eye brightness. Requires block.oe_rec, block.final_sync_df, block.oe_events.
    """
    from bokeh.layouts import column as bokeh_column, row as bokeh_row
    from bokeh.models import Button, CustomJS, Div, Span as BokehSpan

    fs = block.sample_rate if hasattr(block, "sample_rate") else block.get_sample_rate()
    if not hasattr(block, "oe_rec") or block.oe_rec is None:
        raise ValueError("block.oe_rec is not available.")
    if not hasattr(block, "final_sync_df") or block.final_sync_df is None:
        raise ValueError("final_sync_df is not available. Load it first.")
    if not hasattr(block, "oe_events") or block.oe_events is None or "LED_driver_fall" not in block.oe_events.columns:
        raise ValueError("oe_events with 'LED_driver_fall' is required.")

    df = block.final_sync_df
    arena_ttl_samples = df["Arena_TTL"].dropna().astype(np.int64).values
    arena_ttl_seconds = arena_ttl_samples / fs
    l_brightness = df["L_values"].values
    r_brightness = df["R_values"].values

    led_falling_samples = block.oe_events["LED_driver_fall"].dropna().astype(np.int64).values
    led_falling_seconds = led_falling_samples / fs
    led_rising_seconds = np.array([])
    if "LED_driver" in block.oe_events.columns:
        led_rising_seconds = block.oe_events["LED_driver"].dropna().astype(np.int64).values / fs
    n_events = len(led_falling_seconds)
    if n_events == 0:
        raise ValueError("No LED_driver falling edges found in oe_events.")

    window_ms = 2 * window_half_s * 1000
    print(f"Extracting {n_events} windows of {window_ms:.0f} ms around LED_driver falling edges...")

    all_ephys_t = []
    all_ephys_y = []
    all_eyes_t = []
    all_eyes_L = []
    all_eyes_R = []
    event_center_t = []
    event_rising_t = []

    for i, t_center_s in enumerate(led_falling_seconds):
        start_ms = (t_center_s - window_half_s) * 1000
        start_time_ms = np.array([[start_ms]])
        try:
            chunk_data, chunk_timestamps = block.oe_rec.get_data(
                channels=[channel],
                start_time_ms=start_time_ms,
                window_ms=window_ms,
                convert_microvolts=True,
                return_timestamps=True,
                repress_output=True,
            )
        except Exception as e:
            print(f"  Event {i+1}: get_data failed ({e}), skipping")
            continue
        if chunk_data is None or chunk_timestamps is None:
            continue
        ephys_s = (chunk_timestamps[0, :] / 1000.0).tolist()
        ephys_y = chunk_data[0, 0, :].tolist()
        all_ephys_t.append(ephys_s)
        all_ephys_y.append(ephys_y)
        event_center_t.append(float(t_center_s))
        rising_in_window = [t for t in led_rising_seconds if (t_center_s - window_half_s) < t <= t_center_s]
        event_rising_t.append(max(rising_in_window) if rising_in_window else None)

        mask = (arena_ttl_seconds >= t_center_s - window_half_s) & (arena_ttl_seconds <= t_center_s + window_half_s)
        all_eyes_t.append(arena_ttl_seconds[mask].tolist())
        all_eyes_L.append(l_brightness[mask].tolist())
        all_eyes_R.append(r_brightness[mask].tolist())

    n_events = len(all_ephys_t)
    if n_events == 0:
        raise ValueError("Failed to extract any event windows.")

    cds_ix = ColumnDataSource(dict(ix=[0]))
    cds_ephys = ColumnDataSource(dict(x=all_ephys_t[0], y=all_ephys_y[0]))
    cds_eyes = ColumnDataSource(dict(x=all_eyes_t[0], L=all_eyes_L[0], R=all_eyes_R[0]))

    p_ephys = figure(
        width=900, height=320,
        title="Electrophysiology (full resolution) around LED off",
        x_axis_label="Time (s)", y_axis_label="Voltage (µV)",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p_ephys.line("x", "y", source=cds_ephys, line_width=1, color="black")
    span_ephys = BokehSpan(
        location=event_center_t[0], dimension="height",
        line_color="#d62728", line_dash="dashed", line_width=2
    )
    p_ephys.add_layout(span_ephys)
    rising_0 = event_rising_t[0] if event_rising_t[0] is not None else 0.0
    span_rising_ephys = BokehSpan(
        location=rising_0, dimension="height",
        line_color="#2ca02c", line_dash="dotted", line_width=2
    )
    if event_rising_t[0] is None:
        span_rising_ephys.visible = False
    p_ephys.add_layout(span_rising_ephys)

    p_eyes = figure(
        width=900, height=280,
        title="Eye brightness around LED off",
        x_axis_label="Time (s)", y_axis_label="Brightness",
        x_range=p_ephys.x_range, tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p_eyes.line("x", "L", source=cds_eyes, line_width=1.5, color="blue", legend_label="L")
    p_eyes.line("x", "R", source=cds_eyes, line_width=1.5, color="red", legend_label="R")
    span_rising_eyes = BokehSpan(
        location=rising_0, dimension="height",
        line_color="#2ca02c", line_dash="dotted", line_width=2
    )
    if event_rising_t[0] is None:
        span_rising_eyes.visible = False
    p_eyes.add_layout(span_rising_eyes)
    span_eyes_fall = BokehSpan(
        location=event_center_t[0], dimension="height",
        line_color="#d62728", line_dash="dashed", line_width=2
    )
    p_eyes.add_layout(span_eyes_fall)

    title_div = Div(text=f"<b>Event 1 / {n_events}</b> (LED driver falling edge)", styles={"font-size": "14px"})

    def make_callback(delta):
        return CustomJS(
            args=dict(
                cds_ix=cds_ix, cds_ephys=cds_ephys, cds_eyes=cds_eyes,
                all_ephys_t=all_ephys_t, all_ephys_y=all_ephys_y,
                all_eyes_t=all_eyes_t, all_eyes_L=all_eyes_L, all_eyes_R=all_eyes_R,
                n_events=n_events, title_div=title_div,
                event_center_t=event_center_t, event_rising_t=event_rising_t,
                span_ephys=span_ephys, span_rising_ephys=span_rising_ephys,
                span_rising_eyes=span_rising_eyes, span_eyes_fall=span_eyes_fall,
            ),
            code="""
                var cur = cds_ix.data.ix[0];
                cur = (cur + """ + str(delta) + """ + n_events) % n_events;
                cds_ix.data = { ix: [cur] };
                cds_ephys.data = { x: all_ephys_t[cur], y: all_ephys_y[cur] };
                cds_eyes.data = { x: all_eyes_t[cur], L: all_eyes_L[cur], R: all_eyes_R[cur] };
                title_div.text = "<b>Event " + (cur+1) + " / " + n_events + "</b> (LED driver falling edge)";
                span_ephys.location = event_center_t[cur];
                span_eyes_fall.location = event_center_t[cur];
                if (event_rising_t[cur] != null) {
                    span_rising_ephys.location = event_rising_t[cur];
                    span_rising_ephys.visible = true;
                    span_rising_eyes.location = event_rising_t[cur];
                    span_rising_eyes.visible = true;
                } else {
                    span_rising_ephys.visible = false;
                    span_rising_eyes.visible = false;
                }
            """,
        )

    btn_prev = Button(label="Previous event")
    btn_prev.js_on_click(make_callback(-1))
    btn_next = Button(label="Next event")
    btn_next.js_on_click(make_callback(1))

    reset_output()
    layout = bokeh_column(
        bokeh_row(btn_prev, btn_next, title_div),
        p_ephys,
        p_eyes,
    )
    if to_browser:
        show(layout)
        print(f"Viewer ready: {n_events} events, full-resolution ephys in ±{window_half_s}s around each LED off.")
    else:
        return layout
