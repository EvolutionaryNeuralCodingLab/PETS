"""
GUI for saccade-aligned LFP average plots.

Launch:
    python pipelines/saccade_lfp_average/saccade_lfp_gui.py
"""

from __future__ import annotations

import contextlib
import io
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from saccade_lfp_core import (  # noqa: E402
    EyeKey,
    PlotConfig,
    PipelineResult,
    build_figure,
    discover_block_numbers,
    export_display_figures,
    normalize_block_numbers,
    parse_electrodes,
    recompute_averages,
    resolve_export_paths,
    run_pipeline,
)


class GuiLoadLog:
    def __init__(self, text: tk.Text, root: tk.Misc) -> None:
        self._text = text
        self._root = root
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()

    def poll(self) -> None:
        while True:
            try:
                level, msg = self._queue.get_nowait()
            except queue.Empty:
                break
            self._text.insert(tk.END, f"{level}: {msg}\n")
            self._text.see(tk.END)
        self._root.after(100, self.poll)

    def info(self, msg: str) -> None:
        self._queue.put(("INFO", msg))

    def warn(self, msg: str) -> None:
        self._queue.put(("WARN", msg))


class SaccadeLfpApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Saccade-aligned LFP average")
        self.minsize(1100, 700)

        self._experiment_path = tk.StringVar()
        self._animal = tk.StringVar(value="PV_106")
        self._analysis_folder = tk.StringVar()
        self._electrodes = tk.StringVar(value="1")
        self._query = tk.StringVar()
        self._plot_label = tk.StringVar()
        self._export_dir = tk.StringVar()
        self._main_out = tk.StringVar()
        self._legend_out = tk.StringVar()
        self._readonly_entries: dict[int, ttk.Entry] = {}

        self._half_window = tk.DoubleVar(value=500.0)
        self._saccade_threshold = tk.DoubleVar(value=2.0)
        self._min_saccade_frames = tk.IntVar(value=1)
        self._batch_size = tk.IntVar(value=200)
        self._dpi = tk.IntVar(value=300)
        self._fig_w = tk.DoubleVar(value=2.4)
        self._fig_h = tk.DoubleVar(value=1.8)
        self._label_fs = tk.DoubleVar(value=9.0)
        self._tick_fs = tk.DoubleVar(value=8.0)
        self._sem_alpha = tk.DoubleVar(value=0.15)
        self._ep_noise_std_k = tk.DoubleVar(value=0.0)

        self._detection_mode = tk.StringVar(value="velocity")
        self._eye_data_source = tk.StringVar(value="auto")
        self._legacy_speed_threshold = tk.DoubleVar(value=2.0)
        self._legacy_magnitude_calib = tk.DoubleVar(value=1.0)
        self._legacy_sync_diff_ms = tk.DoubleVar(value=680.0)

        self._show_concurrent = tk.BooleanVar(value=True)
        self._show_monocular = tk.BooleanVar(value=True)
        self._show_left = tk.BooleanVar(value=True)
        self._show_right = tk.BooleanVar(value=True)
        self._show_all = tk.BooleanVar(value=True)

        self._block_vars: dict[str, tk.BooleanVar] = {}
        self._result: PipelineResult | None = None
        self._export_main_path: Path | None = None
        self._export_legend_path: Path | None = None
        self._current_fig: Figure | None = None
        self._legend_handles: list = []
        self._legend_labels: list[str] = []
        self._busy = False
        self._canvas: FigureCanvasTkAgg | None = None
        self._toolbar: NavigationToolbar2Tk | None = None
        self._plot_frame: ttk.Frame | None = None
        self._left_canvas: tk.Canvas | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left_outer = ttk.Frame(paned, padding=4)
        paned.add(left_outer, weight=1)

        self._left_canvas = tk.Canvas(left_outer, highlightthickness=0, width=420)
        left_scroll = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=self._left_canvas.yview)
        left = ttk.Frame(self._left_canvas)
        left.bind(
            "<Configure>",
            lambda e: self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all")),
        )
        self._left_window_id = self._left_canvas.create_window((0, 0), window=left, anchor=tk.NW)

        def _fit_left_panel(event) -> None:
            self._left_canvas.itemconfigure(self._left_window_id, width=event.width)

        self._left_canvas.bind("<Configure>", _fit_left_panel)
        self._left_canvas.configure(yscrollcommand=left_scroll.set)
        self._left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event) -> None:
            if self._left_canvas is not None:
                self._left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self._left_canvas.bind("<Enter>", lambda _e: self._left_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self._left_canvas.bind("<Leave>", lambda _e: self._left_canvas.unbind_all("<MouseWheel>"))

        paths = ttk.LabelFrame(left, text="Paths (type or browse)", padding=6)
        paths.pack(fill=tk.X, pady=4)
        ttk.Label(
            paths,
            text="Paths can be typed directly or chosen with Browse.",
            font=("", 8),
            wraplength=380,
        ).pack(anchor=tk.W, pady=(0, 4))

        self._path_row(
            paths,
            "Experiment folder:",
            self._experiment_path,
            self._browse_experiment,
        )
        self._path_row(paths, "Animal:", self._animal, None)
        self._path_row(
            paths,
            "Analysis folder:",
            self._analysis_folder,
            self._browse_analysis_folder,
        )
        scan_row = ttk.Frame(paths)
        scan_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(scan_row, text="Scan blocks", command=self._scan_blocks).pack(side=tk.LEFT)

        blocks_frame = ttk.LabelFrame(left, text="Blocks", padding=6)
        blocks_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        btns = ttk.Frame(blocks_frame)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Select all", command=self._select_all_blocks).pack(side=tk.LEFT)
        ttk.Button(btns, text="Clear", command=self._clear_blocks).pack(side=tk.LEFT, padx=4)

        self._blocks_canvas = tk.Canvas(blocks_frame, height=120, highlightthickness=0)
        blocks_scroll = ttk.Scrollbar(
            blocks_frame, orient=tk.VERTICAL, command=self._blocks_canvas.yview
        )
        self._blocks_inner = ttk.Frame(self._blocks_canvas)
        self._blocks_inner.bind(
            "<Configure>",
            lambda e: self._blocks_canvas.configure(scrollregion=self._blocks_canvas.bbox("all")),
        )
        self._blocks_canvas.create_window((0, 0), window=self._blocks_inner, anchor=tk.NW)
        self._blocks_canvas.configure(yscrollcommand=blocks_scroll.set)
        self._blocks_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        blocks_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ep_frame = ttk.LabelFrame(left, text="Electrophysiology", padding=6)
        ep_frame.pack(fill=tk.X, pady=4)
        row = ttk.Frame(ep_frame)
        row.pack(fill=tk.X)
        ttk.Label(row, text="HS channels:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._electrodes, width=20).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="(comma-separated)").pack(side=tk.LEFT)
        noise_row = ttk.Frame(ep_frame)
        noise_row.pack(fill=tk.X, pady=(6, 0))
        noise_row.columnconfigure(1, weight=1)
        ttk.Label(noise_row, text="Noise filter k:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        noise_spin = ttk.Spinbox(
            noise_row,
            textvariable=self._ep_noise_std_k,
            from_=0,
            to=50,
            increment=0.5,
            width=8,
        )
        noise_spin.grid(row=0, column=1, sticky=tk.W)
        ttk.Button(
            noise_row,
            text="Apply",
            width=8,
            command=self._reapply_noise_filter,
        ).grid(row=0, column=2, sticky=tk.W, padx=(6, 0))
        ttk.Label(
            ep_frame,
            text="0 = off. Drop trials with snippet std > k × median(trial std). "
            "Apply re-averages cached trials (no re-extract).",
            font=("", 8),
            wraplength=380,
        ).pack(anchor=tk.W, pady=(4, 0))

        filt = ttk.LabelFrame(left, text="Saccade filter (pandas query)", padding=6)
        filt.pack(fill=tk.X, pady=4)
        ttk.Entry(filt, textvariable=self._query).pack(fill=tk.X)
        ttk.Label(
            filt,
            text="Columns: block, eye, saccade_start_ms, peak_velocity, accel, behavior, "
            "sync_status (synced/non_synced), sync_pair_id, magnitude, angle",
            font=("", 8),
            wraplength=340,
        ).pack(anchor=tk.W, pady=(4, 0))

        detect = ttk.LabelFrame(left, text="Saccade detection", padding=6)
        detect.pack(fill=tk.X, pady=4)
        mode_row = ttk.Frame(detect)
        mode_row.pack(fill=tk.X, pady=2)
        ttk.Label(mode_row, text="Mode:", width=24).pack(side=tk.LEFT)
        ttk.Combobox(
            mode_row,
            textvariable=self._detection_mode,
            values=("velocity", "legacy"),
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT)
        src_row = ttk.Frame(detect)
        src_row.pack(fill=tk.X, pady=2)
        ttk.Label(src_row, text="Eye CSV source:", width=24).pack(side=tk.LEFT)
        ttk.Combobox(
            src_row,
            textvariable=self._eye_data_source,
            values=("auto", "eye_data_csv", "le_re_df"),
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT)
        ttk.Label(
            detect,
            text="auto prefers left/right_eye_data.csv, then le_df/re_df.csv.",
            font=("", 8),
            wraplength=380,
        ).pack(anchor=tk.W, pady=(2, 0))

        params = ttk.LabelFrame(left, text="Parameters", padding=6)
        params.pack(fill=tk.X, pady=4)
        self._spin(params, "± window (ms)", self._half_window, 10, 60000, 10)
        self._spin(params, "Velocity k×std", self._saccade_threshold, 0.1, 20, 0.1)
        self._spin(params, "Min saccade frames", self._min_saccade_frames, 1, 100, 1)
        self._spin(params, "Legacy speed threshold", self._legacy_speed_threshold, 0.1, 50, 0.1)
        self._spin(params, "Binocular sync window (ms)", self._legacy_sync_diff_ms, 10, 5000, 10)
        self._spin(params, "Legacy magnitude calib", self._legacy_magnitude_calib, 0.01, 10, 0.01)
        self._spin(params, "Batch size (get_data)", self._batch_size, 1, 2000, 10)
        self._spin(params, "Figure DPI", self._dpi, 72, 600, 1)
        self._spin(params, "Figure width (in)", self._fig_w, 0.5, 20, 0.1)
        self._spin(params, "Figure height / panel (in)", self._fig_h, 0.5, 20, 0.1)
        self._spin(params, "Axis label font size", self._label_fs, 4, 24, 0.5)
        self._spin(params, "Tick font size", self._tick_fs, 4, 24, 0.5)
        self._spin(params, "SEM alpha (0–1)", self._sem_alpha, 0.02, 1.0, 0.02)
        ttk.Label(
            params,
            text="Velocity: k×std + min frames. Legacy: fixed speed_r threshold. "
            "Sync window applies to both modes.",
            font=("", 8),
            wraplength=380,
        ).pack(anchor=tk.W, pady=(4, 0))

        traces = ttk.LabelFrame(left, text="Traces (toggle without recompute)", padding=6)
        traces.pack(fill=tk.X, pady=4)
        for text, var in (
            ("Concurrent (binocular pairs)", self._show_concurrent),
            ("Monocular (L + R unpaired)", self._show_monocular),
            ("Left (monocular only)", self._show_left),
            ("Right (monocular only)", self._show_right),
            ("All (concurrent + monocular)", self._show_all),
        ):
            ttk.Checkbutton(
                traces,
                text=text,
                variable=var,
                command=self._redraw_from_cache,
            ).pack(anchor=tk.W)

        out = ttk.LabelFrame(left, text="Export (saved under analysis folder / date)", padding=6)
        out.pack(fill=tk.X, pady=4)
        ttk.Label(
            out,
            text="Preview matches exported PDF exactly. Export saves the figure shown.",
            font=("", 8),
            wraplength=380,
        ).pack(anchor=tk.W, pady=(0, 4))
        label_row = ttk.Frame(out)
        label_row.pack(fill=tk.X, pady=2)
        label_row.columnconfigure(1, weight=1)
        ttk.Label(label_row, text="Plot label:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Entry(label_row, textvariable=self._plot_label).grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(
            out,
            text="Appended to filename, e.g. …_blocks_015_quiet.pdf",
            font=("", 8),
            wraplength=380,
        ).pack(anchor=tk.W, pady=(0, 4))
        self._readonly_path_row(out, "Output folder:", self._export_dir)
        self._readonly_path_row(out, "Main PDF:", self._main_out)
        self._readonly_path_row(out, "Legend PDF:", self._legend_out)

        actions = ttk.Frame(left)
        actions.pack(fill=tk.X, pady=8)
        self._refresh_btn = ttk.Button(actions, text="Refresh plot", command=self._refresh_plot)
        self._refresh_btn.pack(side=tk.LEFT)
        self._export_btn = ttk.Button(actions, text="Export PDFs", command=self._export)
        self._export_btn.pack(side=tk.LEFT, padx=8)

        right = ttk.Frame(paned, padding=4)
        paned.add(right, weight=3)

        self._plot_frame = ttk.Frame(right)
        self._plot_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            self._plot_frame,
            text="Set paths and blocks, then click Refresh plot.",
            foreground="#666666",
        ).pack(expand=True)

        log_frame = ttk.LabelFrame(left, text="Log", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=False, pady=4)
        self._log = tk.Text(log_frame, height=7, wrap=tk.WORD, font=("Consolas", 9))
        self._log.pack(fill=tk.BOTH, expand=True)
        self._gui_log = GuiLoadLog(self._log, self)
        self._gui_log.poll()

    def _path_row(
        self,
        parent,
        label: str,
        var: tk.StringVar,
        browse_cmd,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        row.columnconfigure(1, weight=1)

        ttk.Label(row, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Entry(row, textvariable=var).grid(row=0, column=1, sticky=tk.EW)
        if browse_cmd is not None:
            ttk.Button(row, text="Browse", width=9, command=browse_cmd).grid(
                row=0, column=2, sticky=tk.E, padx=(6, 0)
            )

    def _readonly_path_row(self, parent, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT, anchor=tk.NW)
        entry = ttk.Entry(row, textvariable=var, state="readonly")
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self._readonly_entries[id(var)] = entry

    def _set_readonly_text(self, var: tk.StringVar, value: str) -> None:
        entry = self._readonly_entries.get(id(var))
        if entry is not None:
            entry.configure(state="normal")
        var.set(value)
        if entry is not None:
            entry.configure(state="readonly")

    def _spin(self, parent, label, var, from_, to, inc) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=24).pack(side=tk.LEFT)
        ttk.Spinbox(row, textvariable=var, from_=from_, to=to, increment=inc, width=10).pack(
            side=tk.LEFT
        )

    def _analysis_root(self) -> Path | None:
        folder = self._analysis_folder.get().strip()
        if not folder:
            return None
        return Path(folder)

    def _browse_experiment(self) -> None:
        path = filedialog.askdirectory(title="Experiment folder (parent of animal folders)")
        if path:
            self._experiment_path.set(path)
            self._scan_blocks()

    def _browse_analysis_folder(self) -> None:
        path = filedialog.askdirectory(title="Analysis output folder")
        if path:
            folder = Path(path)
            folder.mkdir(parents=True, exist_ok=True)
            self._analysis_folder.set(str(folder))

    def _ensure_analysis_folder(self) -> Path:
        folder = self._analysis_root()
        if folder is not None and folder.is_dir():
            return folder
        exp = Path(self._experiment_path.get().strip())
        animal = self._animal.get().strip()
        if exp.is_dir() and animal:
            folder = exp / "_saccade_lfp_analysis" / animal
            folder.mkdir(parents=True, exist_ok=True)
            self._analysis_folder.set(str(folder))
            return folder
        raise ValueError("Set a valid analysis folder (type a path or use Browse).")

    def _assign_export_paths(self, result: PipelineResult, *, loud: bool = False) -> tuple[Path, Path]:
        analysis = self._ensure_analysis_folder()
        plot_label = self._plot_label.get().strip() or None
        resolved = resolve_export_paths(
            analysis,
            result.config.animal,
            result.block_labels,
            plot_label=plot_label,
        )
        self._export_main_path = resolved.main_path
        self._export_legend_path = resolved.legend_path
        self._set_readonly_text(self._export_dir, str(resolved.output_dir))
        self._set_readonly_text(self._main_out, str(resolved.main_path))
        self._set_readonly_text(self._legend_out, str(resolved.legend_path))
        if resolved.collision_warning:
            self._gui_log.warn(resolved.collision_warning)
            if loud:
                messagebox.showwarning("Filename adjusted", resolved.collision_warning)
        return resolved.main_path, resolved.legend_path

    def _scan_blocks(self) -> None:
        exp = Path(self._experiment_path.get().strip())
        animal = self._animal.get().strip()
        if not exp.is_dir():
            messagebox.showerror("Error", "Enter or browse to a valid experiment folder.")
            return
        if not animal:
            messagebox.showerror("Error", "Enter an animal name.")
            return

        for w in self._blocks_inner.winfo_children():
            w.destroy()
        self._block_vars.clear()

        blocks = discover_block_numbers(exp, animal)
        if not blocks:
            self._gui_log.warn(f"No blocks found under {exp / animal}")
            return

        for b in blocks:
            var = tk.BooleanVar(value=False)
            self._block_vars[b] = var
            ttk.Checkbutton(self._blocks_inner, text=f"block_{b}", variable=var).pack(
                anchor=tk.W
            )
        self._gui_log.info(f"Found {len(blocks)} block(s) for {animal}")

        try:
            self._ensure_analysis_folder()
        except ValueError:
            pass

    def _selected_blocks(self) -> list[int]:
        return normalize_block_numbers([b for b, v in self._block_vars.items() if v.get()])

    def _select_all_blocks(self) -> None:
        for v in self._block_vars.values():
            v.set(True)

    def _clear_blocks(self) -> None:
        for v in self._block_vars.values():
            v.set(False)

    def _visible_eyes(self) -> frozenset[EyeKey]:
        visible: set[EyeKey] = set()
        if self._show_concurrent.get():
            visible.add("Concurrent")
        if self._show_monocular.get():
            visible.add("Monocular")
        if self._show_left.get():
            visible.add("L")
        if self._show_right.get():
            visible.add("R")
        if self._show_all.get():
            visible.add("All")
        return frozenset(visible)

    def _build_config(self) -> PlotConfig:
        exp = Path(self._experiment_path.get().strip())
        if not exp.is_dir():
            raise ValueError("Enter or browse to a valid experiment folder.")
        animal = self._animal.get().strip()
        if not animal:
            raise ValueError("Enter an animal name.")
        blocks = self._selected_blocks()
        if not blocks:
            raise ValueError("Select at least one block.")
        electrodes = parse_electrodes(self._electrodes.get())
        query = self._query.get().strip() or None
        self._ensure_analysis_folder()

        return PlotConfig(
            experiment_path=exp,
            animal=animal,
            blocks=blocks,
            electrodes=electrodes,
            query=query,
            half_window_ms=float(self._half_window.get()),
            saccade_threshold=float(self._saccade_threshold.get()),
            min_saccade_frames=int(self._min_saccade_frames.get()),
            batch_size=int(self._batch_size.get()),
            dpi=int(self._dpi.get()),
            figsize=(float(self._fig_w.get()), float(self._fig_h.get())),
            label_fontsize=float(self._label_fs.get()),
            tick_fontsize=float(self._tick_fs.get()),
            sem_alpha=float(self._sem_alpha.get()),
            ep_noise_std_k=float(self._ep_noise_std_k.get()),
            visible_eyes=self._visible_eyes(),
            plot_label=self._plot_label.get().strip() or None,
            analysis_folder=self._ensure_analysis_folder(),
            detection_mode=self._detection_mode.get().strip() or "velocity",
            eye_data_source=self._eye_data_source.get().strip() or "auto",
            legacy_speed_threshold=float(self._legacy_speed_threshold.get()),
            legacy_magnitude_calib=float(self._legacy_magnitude_calib.get()),
            legacy_sync_diff_ms=float(self._legacy_sync_diff_ms.get()),
        )

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self._refresh_btn.configure(state=state)
        self._export_btn.configure(state=state)

    def _close_current_figure(self) -> None:
        self._current_fig = None

    def _clear_plot_area(self) -> None:
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        for w in self._plot_frame.winfo_children():
            w.destroy()

    def _show_figure(self, fig: Figure) -> None:
        self._clear_plot_area()
        self._close_current_figure()
        self._current_fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._toolbar = NavigationToolbar2Tk(self._canvas, self._plot_frame)
        self._toolbar.update()
        self._toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _sync_plot_config_from_gui(self) -> None:
        if self._result is None:
            return
        c = self._result.config
        c.dpi = int(self._dpi.get())
        c.figsize = (float(self._fig_w.get()), float(self._fig_h.get()))
        c.label_fontsize = float(self._label_fs.get())
        c.tick_fontsize = float(self._tick_fs.get())
        c.sem_alpha = float(self._sem_alpha.get())
        c.visible_eyes = self._visible_eyes()

    def _render_preview(self) -> None:
        if self._result is None:
            return
        visible = self._visible_eyes()
        if not visible:
            messagebox.showwarning("No traces", "Enable at least one trace to plot.")
            return
        self._sync_plot_config_from_gui()
        fig, self._legend_handles, self._legend_labels = build_figure(
            self._result, visible_eyes=visible
        )
        self._show_figure(fig)

    def _reapply_noise_filter(self) -> None:
        if self._result is None or self._busy:
            if self._result is None:
                messagebox.showinfo("Noise filter", "Refresh the plot first to load trials.")
            return
        try:
            k = float(self._ep_noise_std_k.get())
        except tk.TclError:
            messagebox.showerror("Noise filter", "Enter a numeric k value (0 = off).")
            return
        if k < 0:
            messagebox.showerror("Noise filter", "k must be >= 0.")
            return
        self._result.config.ep_noise_std_k = k
        try:
            recompute_averages(self._result, self._gui_log)
        except ValueError as exc:
            messagebox.showwarning("Noise filter", str(exc))
            return
        try:
            self._render_preview()
            self._gui_log.info(f"Noise filter applied (k={k:g}).")
        except Exception as exc:
            messagebox.showerror("Plot error", str(exc))

    def _redraw_from_cache(self) -> None:
        if self._result is None or self._busy:
            return
        try:
            self._render_preview()
        except Exception as exc:
            messagebox.showerror("Plot error", str(exc))

    def _refresh_plot(self) -> None:
        if self._busy:
            return
        try:
            config = self._build_config()
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return

        visible = self._visible_eyes()
        if not visible:
            messagebox.showwarning("No traces", "Enable at least one trace to plot.")
            return

        def work() -> None:
            captured = io.StringIO()
            err_msg: str | None = None
            result: PipelineResult | None = None
            try:
                with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                    result = run_pipeline(config, self._gui_log)
            except Exception as exc:
                err_msg = str(exc)
                tail = captured.getvalue().strip().splitlines()
                if tail:
                    self._gui_log.warn(tail[-1])

            def finish() -> None:
                if err_msg is not None:
                    messagebox.showerror("Error", err_msg)
                    self._gui_log.warn(err_msg)
                    self._set_busy(False)
                    return
                try:
                    assert result is not None
                    self._result = result
                    self._assign_export_paths(result)
                    self._render_preview()
                    self._gui_log.info("Plot refreshed.")
                    if self._export_main_path is not None:
                        self._gui_log.info(f"Export folder: {self._export_main_path.parent}")
                except Exception as exc:
                    finish_err = str(exc)
                    messagebox.showerror("Error", finish_err)
                    self._gui_log.warn(finish_err)
                finally:
                    self._set_busy(False)

            self.after(0, finish)

        self._set_busy(True)
        threading.Thread(target=work, daemon=True).start()

    def _export(self) -> None:
        if self._busy:
            return
        if self._result is None or self._current_fig is None:
            messagebox.showinfo("Export", "Refresh the plot first.")
            return
        visible = self._visible_eyes()
        if not visible:
            messagebox.showwarning("No traces", "Enable at least one trace to plot.")
            return

        try:
            main_path, legend_path = self._assign_export_paths(self._result, loud=True)
        except ValueError as exc:
            messagebox.showerror("Export error", str(exc))
            return

        try:
            export_display_figures(
                self._current_fig,
                main_path,
                legend_path,
                self._legend_handles,
                self._legend_labels,
            )
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            return

        messagebox.showinfo("Export complete", f"Saved:\n{main_path}\n{legend_path}")
        self._gui_log.info(f"Exported {main_path}")
        self._gui_log.info(f"Exported {legend_path}")


def main() -> None:
    app = SaccadeLfpApp()
    app.mainloop()


if __name__ == "__main__":
    main()
