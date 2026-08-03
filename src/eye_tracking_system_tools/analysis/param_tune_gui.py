"""
Simple desktop GUI: both-eye velocity traces + interactive speed threshold.

Opens a native window (Tk + Matplotlib). No full detector re-runs on drag —
only the threshold line moves (and above-threshold shading updates).

Launch::

    PYTHONPATH=src python -m eye_tracking_system_tools.analysis.param_tune_gui \\
        --registry configs/mouse_M_002_blocks.yaml \\
        --params configs/analysis_params_mouse.yaml

Or from a notebook: ``launch_param_tune_app(registry, params)``.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import matplotlib

matplotlib.use("TkAgg")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from eye_tracking_system_tools.analysis.block_registry import BlockSpec, load_registry  # noqa: E402
from eye_tracking_system_tools.analysis.export_meta import load_params_yaml  # noqa: E402
from eye_tracking_system_tools.analysis.param_tune import (  # noqa: E402
    TraceBundle,
    load_trace_bundle,
)

# Cap drawn points so pan/zoom stays responsive on long mouse blocks.
MAX_DRAW = 25_000
L_COLOR = "#1f77b4"
R_COLOR = "#d62728"
THR_COLOR = "#222222"


def _downsample(t: np.ndarray, y: np.ndarray, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(t)
    if n <= max_n:
        return t, y
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return t[idx], y[idx]


def _finite_ylim(*arrays: np.ndarray, pad: float = 0.08) -> tuple[float, float] | None:
    parts = [a[np.isfinite(a)] for a in arrays if a is not None and len(a)]
    if not parts:
        return None
    v = np.concatenate(parts)
    if v.size == 0:
        return None
    lo, hi = float(np.nanpercentile(v, 0.5)), float(np.nanpercentile(v, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return lo - pad * span, hi + pad * span


class ParamTuneDesktopApp:
    """Both-eye speed traces with a draggable horizontal threshold."""

    def __init__(
        self,
        registry_path: Path | str,
        params_path: Path | str,
        *,
        block_key: str | None = None,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.params_path = Path(params_path)
        self.specs: list[BlockSpec] = load_registry(self.registry_path)
        if not self.specs:
            raise ValueError(f"No blocks in {self.registry_path}")
        self.params: dict[str, Any] = load_params_yaml(self.params_path)

        self._bundles: dict[str, TraceBundle] = {}
        self._bundle: TraceBundle | None = None
        # Cached full-resolution arrays for current mode
        self._t_l = np.array([])
        self._t_r = np.array([])
        self._spd_l = np.array([])
        self._spd_r = np.array([])
        self._t0_ms = 0.0

        self._dragging = False
        self._press_y: float | None = None
        self._updating = False

        s = self.params.get("saccade", {}) or {}
        self._thr_deg = float(s.get("speed_threshold_deg_per_frame", 0.8))
        self._thr_px = 2.0  # probe only; not in YAML by default

        self.root = tk.Tk()
        self.root.title("PETS — velocity / threshold")
        self.root.geometry("1100x640")
        self.root.minsize(800, 480)

        self._build()
        keys = [s.block_key for s in self.specs]
        start = block_key if block_key in keys else keys[0]
        self.block_var.set(start)
        self._load_block()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Block").pack(side=tk.LEFT)
        self.block_var = tk.StringVar()
        self.block_combo = ttk.Combobox(
            top,
            textvariable=self.block_var,
            values=[s.block_key for s in self.specs],
            state="readonly",
            width=28,
        )
        self.block_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.block_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_block())

        self.mode_var = tk.StringVar(value="deg")
        ttk.Radiobutton(
            top, text="Degrees", variable=self.mode_var, value="deg", command=self._on_mode
        ).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(
            top, text="Pixels", variable=self.mode_var, value="px", command=self._on_mode
        ).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Button(top, text="Auto-zoom", command=self._auto_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Full span", command=self._full_span).pack(side=tk.LEFT, padx=2)

        ttk.Label(top, text="Threshold").pack(side=tk.LEFT, padx=(16, 4))
        self.thr_var = tk.DoubleVar(value=self._thr_deg)
        self.thr_spin = ttk.Spinbox(
            top,
            textvariable=self.thr_var,
            from_=0.0,
            to=200.0,
            increment=0.05,
            width=8,
            command=self._on_thr_spin,
        )
        self.thr_spin.pack(side=tk.LEFT)
        self.thr_spin.bind("<Return>", lambda _e: self._on_thr_spin())
        self.thr_spin.bind("<FocusOut>", lambda _e: self._on_thr_spin())
        self.unit_lbl = ttk.Label(top, text="deg/frame")
        self.unit_lbl.pack(side=tk.LEFT, padx=4)

        ttk.Button(top, text="Save thr → YAML", command=self._save_thr).pack(side=tk.RIGHT, padx=4)

        self.status_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(8, 2)).pack(
            side=tk.BOTTOM, fill=tk.X
        )

        fig_frame = ttk.Frame(self.root)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, fig_frame)
        toolbar.update()

        self._line_l: Line2D | None = None
        self._line_r: Line2D | None = None
        self._thr_line: Line2D | None = None
        self._fill_l = None
        self._fill_r = None

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        hint = ttk.Label(
            self.root,
            text="Drag the black threshold line (or edit the spinbox). Toolbar: pan / zoom.",
            padding=(8, 0),
        )
        hint.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------ data
    def _spec(self, key: str) -> BlockSpec:
        for s in self.specs:
            if s.block_key == key:
                return s
        raise KeyError(key)

    def _load_block(self) -> None:
        key = self.block_var.get()
        self._set_status(f"Loading {key}…")
        self.root.update_idletasks()
        if key not in self._bundles:
            self._bundles[key] = load_trace_bundle(self._spec(key), log=True)
        self._bundle = self._bundles[key]
        self._cache_arrays()
        self._draw(reset_view=True)
        self._set_status(self._status_text())

    def _cache_arrays(self) -> None:
        assert self._bundle is not None
        mode = self.mode_var.get()
        col = "angular_speed_r" if mode == "deg" else "speed_r"
        left, right = self._bundle.left, self._bundle.right

        def _xy(df):
            if col not in df.columns:
                return np.array([]), np.array([])
            if "ms_axis" in df.columns:
                t = df["ms_axis"].to_numpy(dtype=float)
            elif "timestamp" in df.columns:
                t = df["timestamp"].to_numpy(dtype=float)
            elif "OE_timestamp" in df.columns:
                t = df["OE_timestamp"].to_numpy(dtype=float)
            else:
                t = np.arange(len(df), dtype=float)
            y = df[col].to_numpy(dtype=float)
            return t, y

        self._t_l, self._spd_l = _xy(left)
        self._t_r, self._spd_r = _xy(right)
        t0_candidates = [t[0] for t in (self._t_l, self._t_r) if len(t)]
        self._t0_ms = float(min(t0_candidates)) if t0_candidates else 0.0

    def _thr(self) -> float:
        return float(self.thr_var.get())

    def _set_thr(self, value: float, *, remember: bool = True) -> None:
        value = max(0.0, float(value))
        self._updating = True
        try:
            self.thr_var.set(round(value, 4))
        finally:
            self._updating = False
        if remember:
            if self.mode_var.get() == "deg":
                self._thr_deg = value
            else:
                self._thr_px = value

    def _on_mode(self) -> None:
        # Persist current thr into the mode we're leaving, then restore other mode's thr
        if self.mode_var.get() == "deg":
            # switching TO deg — previous was px
            try:
                self._thr_px = float(self.thr_var.get())
            except Exception:
                pass
            self._set_thr(self._thr_deg, remember=False)
            self.unit_lbl.configure(text="deg/frame")
        else:
            try:
                self._thr_deg = float(self.thr_var.get())
            except Exception:
                pass
            self._set_thr(self._thr_px, remember=False)
            self.unit_lbl.configure(text="px/frame")
        self._cache_arrays()
        self._draw(reset_view=True)
        self._set_status(self._status_text())

    def _on_thr_spin(self) -> None:
        if self._updating:
            return
        try:
            v = float(self.thr_var.get())
        except Exception:
            return
        self._set_thr(v)
        self._update_threshold_artists()
        self.canvas.draw_idle()
        self._set_status(self._status_text())

    # ------------------------------------------------------------------ plot
    def _draw(self, *, reset_view: bool = False) -> None:
        xlim = self.ax.get_xlim() if not reset_view else None
        ylim = self.ax.get_ylim() if not reset_view else None

        self.ax.clear()
        mode = self.mode_var.get()
        ylab = "Angular speed (deg/frame)" if mode == "deg" else "Speed (px/frame)"
        thr = self._thr()

        # Relative time in seconds for readability
        def _plot_eye(t_ms, spd, color, label):
            if len(t_ms) == 0:
                return None, None
            t_s = (t_ms - self._t0_ms) / 1000.0
            td, yd = _downsample(t_s, spd, MAX_DRAW)
            (line,) = self.ax.plot(td, yd, color=color, lw=0.7, alpha=0.85, label=label)
            # Cheap above-threshold shade on downsampled data
            above = np.isfinite(yd) & (yd > thr)
            fill = None
            if np.any(above):
                fill = self.ax.fill_between(
                    td, 0, yd, where=above, color=color, alpha=0.15, linewidth=0
                )
            return line, fill

        self._line_l, self._fill_l = _plot_eye(self._t_l, self._spd_l, L_COLOR, "Left")
        self._line_r, self._fill_r = _plot_eye(self._t_r, self._spd_r, R_COLOR, "Right")
        self._thr_line = self.ax.axhline(
            thr, color=THR_COLOR, lw=1.8, ls="--", label=f"threshold={thr:g}"
        )
        self._thr_line.set_picker(8)

        self.ax.set_xlabel("Time (s, relative)")
        self.ax.set_ylabel(ylab)
        self.ax.legend(loc="upper right", fontsize=8)
        self.ax.grid(True, alpha=0.25)
        self.fig.tight_layout()

        if reset_view:
            self._full_span(draw=False)
            self._auto_zoom(draw=False)
        else:
            if xlim is not None:
                self.ax.set_xlim(xlim)
            if ylim is not None:
                self.ax.set_ylim(ylim)

        self.canvas.draw_idle()

    def _update_threshold_artists(self) -> None:
        """Move threshold line + refresh shading without rebuilding speed traces."""
        thr = self._thr()
        if self._thr_line is not None:
            self._thr_line.set_ydata([thr, thr])
            self._thr_line.set_label(f"threshold={thr:g}")
            self.ax.legend(loc="upper right", fontsize=8)

        # Rebuild fills only (cheap on downsampled arrays)
        for coll in list(self.ax.collections):
            coll.remove()
        for t_ms, spd, color in (
            (self._t_l, self._spd_l, L_COLOR),
            (self._t_r, self._spd_r, R_COLOR),
        ):
            if len(t_ms) == 0:
                continue
            t_s = (t_ms - self._t0_ms) / 1000.0
            td, yd = _downsample(t_s, spd, MAX_DRAW)
            above = np.isfinite(yd) & (yd > thr)
            if np.any(above):
                self.ax.fill_between(td, 0, yd, where=above, color=color, alpha=0.15, linewidth=0)

    def _auto_zoom(self, *, draw: bool = True) -> None:
        thr = self._thr()
        yl = _finite_ylim(self._spd_l, self._spd_r, np.array([0.0, thr * 1.2]))
        if yl is not None:
            self.ax.set_ylim(0.0, max(yl[1], thr * 1.15, 0.1))
        if draw:
            self.canvas.draw_idle()

    def _full_span(self, *, draw: bool = True) -> None:
        ends = []
        for t in (self._t_l, self._t_r):
            if len(t):
                ends.append(((t[0] - self._t0_ms) / 1000.0, (t[-1] - self._t0_ms) / 1000.0))
        if ends:
            lo = min(a[0] for a in ends)
            hi = max(a[1] for a in ends)
            if hi <= lo:
                hi = lo + 1.0
            self.ax.set_xlim(lo, hi)
        if draw:
            self.canvas.draw_idle()

    # ------------------------------------------------------------------ mouse drag on threshold
    def _near_thr(self, event) -> bool:
        if event.inaxes != self.ax or event.ydata is None or self._thr_line is None:
            return False
        y0, y1 = self.ax.get_ylim()
        tol = 0.03 * (y1 - y0)
        return abs(event.ydata - self._thr()) <= max(tol, 1e-6)

    def _on_press(self, event) -> None:
        if event.button != 1 or event.inaxes != self.ax:
            return
        if self._near_thr(event):
            self._dragging = True
            self._press_y = event.ydata

    def _on_release(self, event) -> None:
        if self._dragging:
            self._dragging = False
            self._press_y = None
            self._set_status(self._status_text())

    def _on_motion(self, event) -> None:
        if not self._dragging or event.inaxes != self.ax or event.ydata is None:
            return
        self._set_thr(event.ydata)
        self._update_threshold_artists()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ save / status
    def _status_text(self) -> str:
        if self._bundle is None:
            return ""
        mode = self.mode_var.get()
        thr = self._thr()
        unit = "deg/frame" if mode == "deg" else "px/frame"
        n_l = int(np.sum(np.isfinite(self._spd_l) & (self._spd_l > thr))) if len(self._spd_l) else 0
        n_r = int(np.sum(np.isfinite(self._spd_r) & (self._spd_r > thr))) if len(self._spd_r) else 0
        n_frames = max(len(self._spd_l), len(self._spd_r))
        return (
            f"{self._bundle.spec.block_key}  |  {mode}  |  thr={thr:g} {unit}  |  "
            f"frames≈{n_frames}  |  above thr: L={n_l} R={n_r} samples"
            + ("  |  (px thr is probe-only; Save writes deg thr)" if mode == "px" else "")
        )

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def _save_thr(self) -> None:
        if self.mode_var.get() != "deg":
            messagebox.showinfo(
                "Pixels mode",
                "YAML only stores speed_threshold_deg_per_frame.\n"
                "Switch to Degrees, set the threshold, then Save.",
            )
            return
        thr = self._thr()
        self.params.setdefault("saccade", {})
        self.params["saccade"]["speed_threshold_deg_per_frame"] = float(thr)
        # Preserve other keys; rewrite file (comments may be lost — same as before)
        text = yaml.safe_dump(self.params, sort_keys=False, default_flow_style=False)
        self.params_path.write_text(text)
        self._set_status(f"Wrote speed_threshold_deg_per_frame={thr:g} → {self.params_path}")

    def run(self) -> None:
        self.root.mainloop()


# Back-compat alias
ParamTuneApp = ParamTuneDesktopApp


def launch_param_tune_app(
    registry_path: Path | str,
    params_path: Path | str,
    *,
    block_key: str | None = None,
) -> None:
    """Open the desktop window and block until it is closed."""
    app = ParamTuneDesktopApp(registry_path, params_path, block_key=block_key)
    app.run()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Both-eye velocity + threshold desktop GUI")
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--block", type=str, default=None)
    args = p.parse_args(argv)
    launch_param_tune_app(args.registry, args.params, block_key=args.block)


if __name__ == "__main__":
    main()


__all__ = ["ParamTuneDesktopApp", "ParamTuneApp", "launch_param_tune_app", "main"]
