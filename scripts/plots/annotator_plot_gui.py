"""
Minimal GUI for annotator event plots (pupil, degrees, EP).

Launch:
    python scripts/plots/annotator_plot_gui.py
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from annotator_plot_core import (  # noqa: E402
    AnnotationSummary,
    Normalization,
    PlotMode,
    PlotRequest,
    PlotStyle,
    STREAM_EP,
    STREAM_L_DEG,
    STREAM_L_PUPIL,
    STREAM_R_DEG,
    STREAM_R_PUPIL,
    annotation_label,
    build_plot_data,
    event_types_from_summaries,
    export_figures,
    scan_annotations,
)


class GuiLoadLog:
    def __init__(self, text: tk.Text) -> None:
        self._text = text

    def _write(self, level: str, msg: str) -> None:
        line = f"{level}: {msg}\n"

        def append() -> None:
            self._text.insert(tk.END, line)
            self._text.see(tk.END)

        self._text.after(0, append)

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def warn(self, msg: str) -> None:
        self._write("WARN", msg)


class AnnotatorPlotApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Annotator event plots")
        self.minsize(920, 640)

        self._output_folder = tk.StringVar()
        self._summaries: list[AnnotationSummary] = []
        self._annotation_vars: dict[Path, tk.BooleanVar] = {}
        self._event_type_vars: dict[str, tk.BooleanVar] = {}
        self._busy = False

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Annotator output folder:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self._output_folder, width=55).pack(
            side=tk.LEFT, padx=6, fill=tk.X, expand=True
        )
        ttk.Button(top, text="Browse…", command=self._browse_folder).pack(side=tk.LEFT)
        ttk.Button(top, text="Load annotations", command=self._load_annotations).pack(
            side=tk.LEFT, padx=4
        )

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(paned, padding=4)
        paned.add(left, weight=1)

        ttk.Label(left, text="Blocks (annotation files)", font=("", 9, "bold")).pack(
            anchor=tk.W
        )
        ann_btns = ttk.Frame(left)
        ann_btns.pack(fill=tk.X, pady=2)
        ttk.Button(ann_btns, text="Select all", command=self._select_all_blocks).pack(
            side=tk.LEFT
        )
        ttk.Button(ann_btns, text="Clear", command=self._clear_blocks).pack(
            side=tk.LEFT, padx=4
        )

        self._ann_canvas = tk.Canvas(left, highlightthickness=0)
        ann_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._ann_canvas.yview)
        self._ann_inner = ttk.Frame(self._ann_canvas)
        self._ann_inner.bind(
            "<Configure>",
            lambda e: self._ann_canvas.configure(scrollregion=self._ann_canvas.bbox("all")),
        )
        self._ann_canvas.create_window((0, 0), window=self._ann_inner, anchor=tk.NW)
        self._ann_canvas.configure(yscrollcommand=ann_scroll.set)
        self._ann_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ann_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(left, text="Event types", font=("", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        et_btns = ttk.Frame(left)
        et_btns.pack(fill=tk.X, pady=2)
        ttk.Button(et_btns, text="Select all", command=self._select_all_types).pack(
            side=tk.LEFT
        )
        ttk.Button(et_btns, text="Clear", command=self._clear_types).pack(side=tk.LEFT, padx=4)

        self._types_frame = ttk.Frame(left)
        self._types_frame.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(paned, padding=4)
        paned.add(right, weight=2)

        streams = ttk.LabelFrame(right, text="Streams", padding=6)
        streams.pack(fill=tk.X, pady=4)
        self._var_lpupil = tk.BooleanVar(value=True)
        self._var_rpupil = tk.BooleanVar(value=True)
        self._var_ldeg = tk.BooleanVar(value=False)
        self._var_rdeg = tk.BooleanVar(value=False)
        self._var_ep = tk.BooleanVar(value=False)
        for text, var in (
            ("L pupil diameter", self._var_lpupil),
            ("R pupil diameter", self._var_rpupil),
            ("L eye position (deg)", self._var_ldeg),
            ("R eye position (deg)", self._var_rdeg),
            ("Electrophysiology (HS)", self._var_ep),
        ):
            ttk.Checkbutton(streams, text=text, variable=var).pack(anchor=tk.W)

        ep_row = ttk.Frame(streams)
        ep_row.pack(fill=tk.X, pady=4)
        ttk.Label(ep_row, text="HS channels:").pack(side=tk.LEFT)
        self._ep_channels = tk.StringVar(value="1")
        ttk.Entry(ep_row, textvariable=self._ep_channels, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(ep_row, text="(comma-separated, e.g. 1,2)").pack(side=tk.LEFT)

        plot_row = ttk.LabelFrame(right, text="Plot", padding=6)
        plot_row.pack(fill=tk.X, pady=4)
        self._plot_mode = tk.StringVar(value="both")
        for val, text in (
            ("average", "Average ± SEM"),
            ("individual", "Individual trials only"),
            ("both", "Individuals + average"),
        ):
            ttk.Radiobutton(plot_row, text=text, variable=self._plot_mode, value=val).pack(
                anchor=tk.W
            )

        self._normalization = tk.StringVar(value="within_block_zscore")
        norm_row = ttk.Frame(plot_row)
        norm_row.pack(fill=tk.X, pady=4)
        ttk.Label(norm_row, text="Normalization:").pack(side=tk.LEFT)
        ttk.Combobox(
            norm_row,
            textvariable=self._normalization,
            values=("within_block_zscore", "per_trial_zscore", "none"),
            state="readonly",
            width=22,
        ).pack(side=tk.LEFT, padx=4)

        params = ttk.LabelFrame(right, text="Parameters", padding=6)
        params.pack(fill=tk.X, pady=4)

        def spin(parent, label, var, from_, to, inc=1, width=8):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=22).pack(side=tk.LEFT)
            ttk.Spinbox(
                row, textvariable=var, from_=from_, to=to, increment=inc, width=width
            ).pack(side=tk.LEFT)
            return var

        self._half_window = tk.DoubleVar(value=100.0)
        self._fig_w = tk.DoubleVar(value=2.2)
        self._fig_h = tk.DoubleVar(value=1.8)
        self._dpi = tk.IntVar(value=300)
        self._label_fs = tk.DoubleVar(value=9.0)
        self._tick_fs = tk.DoubleVar(value=8.0)
        self._line_w = tk.DoubleVar(value=1.2)
        self._mean_lw = tk.DoubleVar(value=1.8)
        self._trial_alpha = tk.DoubleVar(value=0.35)

        spin(params, "± window (ms)", self._half_window, 1, 60000, 10)
        spin(params, "Figure width (in)", self._fig_w, 0.5, 20, 0.1)
        spin(params, "Figure height / panel (in)", self._fig_h, 0.5, 20, 0.1)
        spin(params, "DPI", self._dpi, 72, 600, 1)
        spin(params, "Axis label font size", self._label_fs, 4, 24, 0.5)
        spin(params, "Tick font size", self._tick_fs, 4, 24, 0.5)
        spin(params, "Mean line width", self._mean_lw, 0.2, 6, 0.1)
        spin(params, "Trial alpha (0–1)", self._trial_alpha, 0.05, 1.0, 0.05)

        out = ttk.LabelFrame(right, text="Output", padding=6)
        out.pack(fill=tk.X, pady=4)
        self._main_out = tk.StringVar()
        self._legend_out = tk.StringVar()
        out_row = ttk.Frame(out)
        out_row.pack(fill=tk.X)
        ttk.Entry(out_row, textvariable=self._main_out, width=50).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(out_row, text="Main PDF…", command=self._pick_main_out).pack(side=tk.LEFT, padx=2)
        leg_row = ttk.Frame(out)
        leg_row.pack(fill=tk.X, pady=4)
        ttk.Entry(leg_row, textvariable=self._legend_out, width=50).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(leg_row, text="Legend PDF…", command=self._pick_legend_out).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Label(
            out,
            text="Legend is exported as a separate PDF (main figure has no legend).",
            font=("", 8),
        ).pack(anchor=tk.W)

        actions = ttk.Frame(right)
        actions.pack(fill=tk.X, pady=8)
        self._gen_btn = ttk.Button(actions, text="Generate PDFs", command=self._generate)
        self._gen_btn.pack(side=tk.LEFT)
        ttk.Button(actions, text="Preview", command=self._preview).pack(side=tk.LEFT, padx=8)

        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=4)
        self._log = tk.Text(log_frame, height=8, wrap=tk.WORD, font=("Consolas", 9))
        self._log.pack(fill=tk.BOTH, expand=True)
        self._gui_log = GuiLoadLog(self._log)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Annotator output folder")
        if path:
            self._output_folder.set(path)
            self._load_annotations()

    def _load_annotations(self) -> None:
        folder = Path(self._output_folder.get().strip())
        if not folder.is_dir():
            messagebox.showerror("Error", "Choose a valid output folder.")
            return
        self._summaries = scan_annotations(folder)
        for w in self._ann_inner.winfo_children():
            w.destroy()
        self._annotation_vars.clear()
        for w in self._types_frame.winfo_children():
            w.destroy()
        self._event_type_vars.clear()

        if not self._summaries:
            self._gui_log.warn(f"No *_annotations.json in {folder}")
            return

        for summary in self._summaries:
            var = tk.BooleanVar(value=True)
            self._annotation_vars[summary.path] = var
            ttk.Checkbutton(
                self._ann_inner,
                text=annotation_label(summary),
                variable=var,
                command=self._refresh_event_types,
            ).pack(anchor=tk.W)

        self._refresh_event_types()
        default_main = folder / "event_plot.pdf"
        default_leg = folder / "event_plot_legend.pdf"
        if not self._main_out.get():
            self._main_out.set(str(default_main))
        if not self._legend_out.get():
            self._legend_out.set(str(default_leg))
        self._gui_log.info(f"Loaded {len(self._summaries)} annotation file(s)")

    def _refresh_event_types(self) -> None:
        for w in self._types_frame.winfo_children():
            w.destroy()
        self._event_type_vars.clear()
        selected = self._selected_annotation_paths()
        types = event_types_from_summaries(self._summaries, selected)
        for et in types:
            var = tk.BooleanVar(value=True)
            self._event_type_vars[et] = var
            ttk.Checkbutton(self._types_frame, text=et, variable=var).pack(anchor=tk.W)

    def _selected_annotation_paths(self) -> list[Path]:
        return [p for p, v in self._annotation_vars.items() if v.get()]

    def _selected_event_types(self) -> list[str]:
        return [t for t, v in self._event_type_vars.items() if v.get()]

    def _select_all_blocks(self) -> None:
        for v in self._annotation_vars.values():
            v.set(True)
        self._refresh_event_types()

    def _clear_blocks(self) -> None:
        for v in self._annotation_vars.values():
            v.set(False)
        self._refresh_event_types()

    def _select_all_types(self) -> None:
        for v in self._event_type_vars.values():
            v.set(True)

    def _clear_types(self) -> None:
        for v in self._event_type_vars.values():
            v.set(False)

    def _parse_ep_channels(self) -> list[int]:
        text = self._ep_channels.get().strip()
        if not text:
            return [1]
        return [int(x.strip()) for x in text.replace(" ", "").split(",") if x.strip()]

    def _build_request(self) -> PlotRequest | None:
        paths = self._selected_annotation_paths()
        if not paths:
            messagebox.showerror("Error", "Select at least one block.")
            return None
        types = self._selected_event_types()
        if not types:
            messagebox.showerror("Error", "Select at least one event type.")
            return None

        stream_ids: list[str] = []
        if self._var_lpupil.get():
            stream_ids.append(STREAM_L_PUPIL)
        if self._var_rpupil.get():
            stream_ids.append(STREAM_R_PUPIL)
        if self._var_ldeg.get():
            stream_ids.append(STREAM_L_DEG)
        if self._var_rdeg.get():
            stream_ids.append(STREAM_R_DEG)
        if self._var_ep.get():
            stream_ids.append(STREAM_EP)
        if not stream_ids:
            messagebox.showerror("Error", "Select at least one stream to plot.")
            return None

        ep_channels = self._parse_ep_channels() if self._var_ep.get() else [1]

        style = PlotStyle(
            figsize=(float(self._fig_w.get()), float(self._fig_h.get())),
            dpi=int(self._dpi.get()),
            label_fontsize=float(self._label_fs.get()),
            tick_fontsize=float(self._tick_fs.get()),
            line_width=float(self._line_w.get()),
            mean_line_width=float(self._mean_lw.get()),
            trial_alpha=float(self._trial_alpha.get()),
        )

        return PlotRequest(
            annotation_paths=paths,
            event_types=types,
            stream_ids=stream_ids,
            ep_channels=ep_channels,
            half_window_ms=float(self._half_window.get()),
            mode=self._plot_mode.get(),  # type: ignore[arg-type]
            normalization=self._normalization.get(),  # type: ignore[arg-type]
            style=style,
        )

    def _pick_main_out(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF", "*.pdf")]
        )
        if path:
            self._main_out.set(path)

    def _pick_legend_out(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF", "*.pdf")]
        )
        if path:
            self._legend_out.set(path)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self._gen_btn.configure(state=state)

    def _run_plot_job(self, preview: bool) -> None:
        request = self._build_request()
        if request is None:
            return
        main_path = Path(self._main_out.get().strip())
        legend_path = Path(self._legend_out.get().strip())
        if not main_path.suffix:
            main_path = main_path.with_suffix(".pdf")
        if not legend_path.suffix:
            legend_path = legend_path.with_suffix(".pdf")

        def work() -> None:
            try:
                series = build_plot_data(request, self._gui_log)
                if not series:
                    self.after(
                        0,
                        lambda: messagebox.showwarning(
                            "No data", "No plottable snippets for this selection."
                        ),
                    )
                    return
                if preview:
                    import matplotlib.pyplot as plt

                    plt.rcParams["font.family"] = "sans-serif"
                    plt.rcParams["font.sans-serif"] = [
                        request.style.font_family,
                        "DejaVu Sans",
                    ]
                    n = len(series)
                    fig_h = request.style.figsize[1] * n if n > 1 else request.style.figsize[1]
                    fig, axes = plt.subplots(
                        n,
                        1,
                        figsize=(request.style.figsize[0], fig_h),
                        sharex=True,
                        squeeze=False,
                    )
                    from annotator_plot_core import (  # noqa: E402
                        _style_axes,
                        render_stream_axis,
                        ylabel_for_normalization,
                    )

                    ylabel = ylabel_for_normalization(request.normalization)
                    for ax, data in zip(axes.ravel(), series):
                        render_stream_axis(
                            ax, data, mode=request.mode, style=request.style
                        )
                        _style_axes(ax, request.style, ylabel)
                    fig.subplots_adjust(hspace=0.28 if n > 1 else 0.05)
                    self.after(0, lambda: plt.show())
                    self._gui_log.info("Preview window opened")
                else:
                    export_figures(
                        series,
                        mode=request.mode,
                        normalization=request.normalization,
                        style=request.style,
                        main_path=main_path,
                        legend_path=legend_path,
                    )
                    self.after(
                        0,
                        lambda: messagebox.showinfo(
                            "Done",
                            f"Saved:\n{main_path}\n{legend_path}",
                        ),
                    )
                    self._gui_log.info(f"Saved {main_path}")
                    self._gui_log.info(f"Saved {legend_path}")
            except Exception as exc:
                self.after(
                    0,
                    lambda: messagebox.showerror("Error", str(exc)),
                )
                self._gui_log.warn(str(exc))
            finally:
                self.after(0, lambda: self._set_busy(False))

        self._set_busy(True)
        threading.Thread(target=work, daemon=True).start()

    def _generate(self) -> None:
        if self._busy:
            return
        self._run_plot_job(preview=False)

    def _preview(self) -> None:
        if self._busy:
            return
        self._run_plot_job(preview=True)


def main() -> None:
    app = AnnotatorPlotApp()
    app.mainloop()


if __name__ == "__main__":
    main()
