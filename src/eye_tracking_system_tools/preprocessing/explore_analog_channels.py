#!/usr/bin/env python
"""
Standalone script to explore Open Ephys analog channels and label them by pattern.

Usage:
    python explore_analog_channels.py
    python explore_analog_channels.py "D:\\path\\to\\block_019"   # skip folder dialog

Flow:
1. A folder dialog asks you to select a block folder (e.g. .../PV_208/2025_12_14/block_019),
   or pass the block path as the first argument to skip the dialog.
2. Analog data for all channels is loaded (in µV via MicrovoltsPerADAnalog), downsampled for display.
3. An explorable Bokeh plot is saved to the block's analysis folder and opened in your browser.
4. A small GUI lets you assign a label to each channel number; labels are saved to
   analysis_path / "analog_channel_labels.json".

Run from the repo root or with PYTHONPATH including the package so imports resolve.
"""

from __future__ import annotations

import json
import os
import sys
import webbrowser
from pathlib import Path

import numpy as np

# Ensure we can import the package (script lives in .../src/eye_tracking_system_tools/preprocessing/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent.parent  # src/eye_tracking_system_tools -> src
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    tk = None

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.models import ColumnDataSource
    from bokeh.palettes import Category10
except ImportError:
    figure = output_file = save = ColumnDataSource = Category10 = None

try:
    from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
    from eye_tracking_system_tools.preprocessing.OERecording import OERecording
except ImportError as e:
    print(
        f"Import error: {e}\n"
        f"Make sure the package is on PYTHONPATH. Added: {_PKG_ROOT}\n"
        "Run from repo root: python src/eye_tracking_system_tools/preprocessing/explore_analog_channels.py\n"
        "Or: PYTHONPATH=src python -m eye_tracking_system_tools.preprocessing.explore_analog_channels",
        file=sys.stderr,
    )
    sys.exit(1)


LABELS_FILENAME = "analog_channel_labels.json"
PLOT_FILENAME = "analog_channels_explorer.html"
DOWNSAMPLE_FACTOR = 1000


def select_block_folder() -> Path | None:
    """Open a folder dialog to select the block directory (e.g. .../block_019). Returns None if cancelled."""
    if tk is None:
        raise RuntimeError("tkinter is required for folder selection")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(
        title="Select block folder (e.g. .../animal/date/block_019)",
        mustexist=True,
    )
    root.destroy()
    if not path:
        return None
    return Path(path)


def build_block_from_path(block_path: Path) -> BlockSync:
    """
    Build a BlockSync instance from a block folder path.
    Expects block_path to be .../animal/date/block_XXX (e.g. .../PV_208/2025_12_14/block_019).
    """
    block_path = block_path.resolve()
    if not block_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {block_path}")
    # .../experiment_path/animal/date/block_019
    animal_call = block_path.parent.parent.name
    experiment_date = block_path.parent.name
    block_num = block_path.name.replace("block_", "")
    path_to_animal_folder = str(block_path.parent.parent.parent)
    return BlockSync(
        animal_call=animal_call,
        experiment_date=experiment_date,
        block_num=block_num,
        path_to_animal_folder=path_to_animal_folder,
    )


def load_analog_data(block: BlockSync, downsample: int = DOWNSAMPLE_FACTOR):
    """
    Load all analog channels for the full recording in µV (get_analog_data uses MicrovoltsPerADAnalog).
    Returns (time_s_ds, values_by_channel, channel_numbers) with downsampled arrays.
    """
    oe_rec = block.oe_rec
    if oe_rec is None:
        raise RuntimeError("Block has no Open Ephys recording (oe_rec).")
    channels = list(oe_rec.analogChannelNumbers)
    start_time_ms = np.array([[0.0]])
    window_ms = oe_rec.recordingDuration_ms
    data_matrix, timestamps = oe_rec.get_analog_data(
        channels, start_time_ms, window_ms, convert_microvolts=True, return_timestamps=True
    )
    # data_matrix: (n_channels, n_windows, nSamples); timestamps: (n_windows, nSamples) in ms
    time_ms = np.asarray(timestamps[0]).flatten()
    time_s = time_ms / 1000.0
    time_s_ds = time_s[::downsample]
    values_by_channel = {}
    for i, ch in enumerate(channels):
        values_by_channel[int(ch)] = np.asarray(data_matrix[i, 0, :]).flatten()[::downsample]
    return time_s_ds, values_by_channel, channels


def build_and_save_plot(
    block: BlockSync,
    time_s: np.ndarray,
    values_by_channel: dict,
    channel_numbers: list,
    labels: dict | None = None,
) -> Path:
    """Create Bokeh figure with one line per channel, save to block analysis_path, return path."""
    if figure is None or output_file is None or save is None or ColumnDataSource is None:
        raise RuntimeError("Bokeh is required for plotting (pip install bokeh).")
    analysis_path = Path(block.analysis_path)
    analysis_path.mkdir(parents=True, exist_ok=True)
    out_path = analysis_path / PLOT_FILENAME
    labels = labels or {}

    sources = {}
    colors = list(Category10[10])  # cycle if more than 10 channels
    for i, ch in enumerate(channel_numbers):
        ch_int = int(ch)
        label = labels.get(str(ch_int), "").strip()
        legend = f"Ch {ch_int}" + (f": {label}" if label else "")
        sources[legend] = ColumnDataSource(
            data={"time_s": time_s, "values": values_by_channel[ch_int]}
        )

    output_file(str(out_path), title=f"Analog channels — Block {block.block_num}")
    p = figure(
        width=1000,
        height=600,
        title=f"Analog channels (µV) — Block {block.block_num}",
        x_axis_label="Time (s)",
        y_axis_label="Analog (µV)",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        active_drag="pan",
        active_scroll="wheel_zoom",
    )
    for i, legend in enumerate(sources):
        p.line(
            "time_s",
            "values",
            source=sources[legend],
            line_width=1.5,
            color=colors[i % len(colors)],
            legend_label=legend,
        )
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    save(p)
    return out_path


def open_in_browser(file_path: Path) -> None:
    """Open the HTML file in the default browser."""
    webbrowser.open(file_path.as_uri())


def labels_path(block: BlockSync) -> Path:
    return Path(block.analysis_path) / LABELS_FILENAME


def load_labels(block: BlockSync) -> dict:
    """Load channel number -> label from analysis_path/analog_channel_labels.json."""
    path = labels_path(block)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def save_labels(block: BlockSync, labels: dict) -> None:
    """Save channel number -> label to analysis_path/analog_channel_labels.json."""
    path = labels_path(block)
    Path(block.analysis_path).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    messagebox.showinfo("Saved", f"Labels saved to:\n{path}")


def run_labeler_gui(block: BlockSync, channel_numbers: list) -> None:
    """Open a small tkinter window to edit and save channel labels."""
    if tk is None:
        print("tkinter not available; skipping labeler GUI.")
        return
    labels = load_labels(block)
    win = tk.Tk()
    win.title(f"Analog channel labels — Block {block.block_num}")
    win.geometry("420x400")
    main = ttk.Frame(win, padding=10)
    main.pack(fill=tk.BOTH, expand=True)
    ttk.Label(main, text="Label each channel (e.g. Photodiode, LED_driver):").pack(anchor=tk.W)
    entries = {}
    for ch in channel_numbers:
        row = ttk.Frame(main)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=f"Channel {ch}:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        e = ttk.Entry(row, width=35)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True)
        e.insert(0, labels.get(str(int(ch)), ""))
        entries[int(ch)] = e

    def do_save():
        new_labels = {str(ch): e.get().strip() for ch, e in entries.items()}
        save_labels(block, new_labels)

    ttk.Button(main, text="Save labels", command=do_save).pack(pady=10)
    ttk.Label(main, text=f"Saved to: {labels_path(block)}", font=("", 8), foreground="gray").pack(anchor=tk.W)
    ttk.Label(main, text="Re-run this script to regenerate the plot with updated labels.", font=("", 8), foreground="gray").pack(anchor=tk.W, pady=4)
    win.mainloop()


def _show_error(title: str, msg: str) -> None:
    if tk is not None:
        try:
            messagebox.showerror(title, msg)
        except Exception:
            pass
    print(f"[{title}] {msg}", file=sys.stderr)


def main() -> None:
    if len(sys.argv) > 1:
        block_path = Path(sys.argv[1]).resolve()
        if not block_path.is_dir():
            print(f"Not a directory: {block_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Using block path from argument: {block_path}")
    else:
        print("Select the block folder (e.g. .../animal/date/block_019)...")
        block_path = select_block_folder()
        if block_path is None:
            print("Cancelled.")
            return
        print(f"Block path: {block_path}")
    try:
        block = build_block_from_path(block_path)
    except Exception as e:
        _show_error("Error", f"Could not load block:\n{e}")
        raise SystemExit(1)
    if block.oe_rec is None:
        _show_error("Error", "No Open Ephys recording found in this block.")
        raise SystemExit(1)
    analysis_path = Path(block.analysis_path)
    analysis_path.mkdir(parents=True, exist_ok=True)
    print("Loading analog data (full recording, downsampled for display)...")
    try:
        time_s_ds, values_by_channel, channel_numbers = load_analog_data(block)
    except Exception as e:
        _show_error("Error", f"Could not load analog data:\n{e}")
        raise SystemExit(1)
    labels = load_labels(block)
    print("Building plot...")
    try:
        out_path = build_and_save_plot(
            block, time_s_ds, values_by_channel, channel_numbers, labels
        )
    except Exception as e:
        _show_error("Error", f"Could not build plot:\n{e}")
        raise SystemExit(1)
    print(f"Plot saved to: {out_path}")
    open_in_browser(out_path)
    print("Opening labeler GUI...")
    run_labeler_gui(block, channel_numbers)
    print("Done.")


if __name__ == "__main__":
    main()
