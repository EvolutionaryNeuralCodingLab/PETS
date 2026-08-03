"""
Jitter epoch picking and pooled histograms.

Paradigm
--------
1. Load existing ``block/analysis/jitter_report_dict.pkl`` (no video recompute).
2. Extract amplitude trace (``top_correlation_dist``) for manual epoch selection.
3. Persist epochs to ``metadata/jitter_epochs/*.yaml``.
4. Pool samples inside epochs → modular-vs-rigid hist + dedicated mouse hist.

Displacements are reported in µm by default, scaling each eye's pixel trace by that
block's ``analysis/LR_pix_size.csv`` (see :mod:`..pixel_calibration`); pass
``units="px"`` to stay in raw camera pixels.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Literal

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import rcParams
from matplotlib.widgets import SpanSelector

from eye_tracking_system_tools.analysis.run_layout import resolve_run_dir

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

MountType = Literal["modular", "rigid", "mouse"]
EyeName = Literal["left_eye", "right_eye", "both"]


@dataclass
class JitterBlockSpec:
    animal: str
    block_path: Path
    mount_type: MountType

    @property
    def block_key(self) -> str:
        return f"{self.animal}_{self.block_path.name}"

    @property
    def report_path(self) -> Path:
        return self.block_path / "analysis" / "jitter_report_dict.pkl"


@dataclass
class Epoch:
    start_frame: int
    end_frame: int
    eye: str = "both"
    notes: str = ""


@dataclass
class BlockEpochs:
    spec: JitterBlockSpec
    epochs: list[Epoch] = field(default_factory=list)
    amplitude_key: str = "top_correlation_dist"


_DATE_DIR = re.compile(r"^\d{4}[-_]\d{2}[-_]\d{2}$")


def has_jitter_report(block_path: Path | str) -> bool:
    return (Path(block_path) / "analysis" / "jitter_report_dict.pkl").is_file()


def infer_animal(block_path: Path | str) -> str:
    """``.../PV_143/2025_11_08/block_001`` → ``PV_143`` (skips date folders)."""
    for parent in Path(block_path).resolve().parents:
        if not parent.name or _DATE_DIR.match(parent.name):
            continue
        return parent.name
    return "UNKNOWN"


def infer_date(block_path: Path | str) -> str:
    """``.../PV_143/2025_11_08/block_001`` → ``2025_11_08`` (empty if absent)."""
    for parent in Path(block_path).resolve().parents:
        if _DATE_DIR.match(parent.name):
            return parent.name
    return ""


def guess_mount_type(animal: str) -> MountType:
    """Mouse animals (``M_002``) default to the dedicated mouse group."""
    return "mouse" if re.match(r"^M_\d", str(animal)) else "modular"


def find_blocks_with_reports(
    root: Path | str,
    *,
    max_depth: int = 4,
    limit: int = 500,
) -> list[Path]:
    """
    Bounded walk for block folders holding ``analysis/jitter_report_dict.pkl``.

    Depth-limited so scanning a network volume root stays responsive.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    base_depth = len(root.resolve().parts)
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if len(here.resolve().parts) - base_depth >= max_depth:
            dirnames[:] = []
        if here.name == "analysis" and "jitter_report_dict.pkl" in filenames:
            found.append(here.parent)
            dirnames[:] = []
        if len(found) >= limit:
            break
    return found


REGISTRY_HEADER = """# Jitter mount comparison registry.
# Each block_path must already contain
#   <block_path>/analysis/jitter_report_dict.pkl
# (from the preprocessing GUI jitter step — this tool never recomputes from video).
#
# mount_type:
#   modular | rigid  -> pooled into figures/jitter_modular_vs_rigid.pdf
#   mouse            -> dedicated figures/jitter_mouse.pdf
#
# Edit by hand, or populate with the notebook browser:
#   development/jitter_mount_pipeline.ipynb
"""


def read_registry_blocks(path: Path | str) -> list[JitterBlockSpec]:
    """Tolerant read: returns ``[]`` for a missing / placeholder-only registry."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: list[JitterBlockSpec] = []
    for row in data.get("blocks") or []:
        if not isinstance(row, dict) or "block_path" not in row:
            continue
        mt = str(row.get("mount_type", "modular")).strip().lower()
        if mt not in {"modular", "rigid", "mouse"}:
            raise ValueError(f"Invalid mount_type={mt!r} (expected modular|rigid|mouse)")
        out.append(
            JitterBlockSpec(
                animal=str(row.get("animal") or infer_animal(row["block_path"])),
                block_path=Path(row["block_path"]),
                mount_type=mt,  # type: ignore[arg-type]
            )
        )
    return out


def load_jitter_registry(path: Path | str) -> list[JitterBlockSpec]:
    """Strict read used by the CLI: raises when no usable blocks are listed."""
    specs = read_registry_blocks(path)
    if not specs:
        raise ValueError(
            f"{path}: no blocks listed. Fill mount_type + block_path entries "
            "before epoch picking (see development/jitter_mount_pipeline.ipynb)."
        )
    return specs


def write_jitter_registry(
    path: Path | str,
    specs: list[JitterBlockSpec],
    *,
    header: bool = True,
) -> Path:
    """Write the registry YAML, de-duplicating on ``block_path``."""
    path = Path(path)
    seen: set[str] = set()
    rows = []
    for spec in specs:
        key = str(Path(spec.block_path))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "animal": str(spec.animal),
                "block_path": key,
                "mount_type": str(spec.mount_type),
            }
        )
    body = yaml.safe_dump({"blocks": rows}, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(REGISTRY_HEADER)
        f.write(body)
    return path


def load_jitter_report(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing jitter report: {path}\n"
            "Run the preprocessing GUI jitter step for this block "
            "(do not recompute from video in this tool)."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not {"left_eye", "right_eye"} <= set(data):
        raise ValueError(f"{path}: expected dict with left_eye/right_eye keys")
    return data


def extract_amplitude_trace(
    report: dict[str, Any],
    *,
    eye: EyeName = "both",
    key: str = "top_correlation_dist",
) -> dict[str, np.ndarray]:
    """Return ``{eye_name: amplitude_array}`` frame-length traces."""
    eyes = ("left_eye", "right_eye") if eye == "both" else (eye,)
    out: dict[str, np.ndarray] = {}
    for e in eyes:
        eye_dict = report.get(e) or {}
        if key not in eye_dict:
            raise KeyError(
                f"{e}.{key} missing in jitter_report_dict "
                f"(available: {sorted(eye_dict)})"
            )
        out[e] = np.asarray(eye_dict[key], dtype=float)
    return out


def epochs_yaml_path(metadata_dir: Path, spec: JitterBlockSpec) -> Path:
    return metadata_dir / "jitter_epochs" / f"{spec.block_key}_epochs.yaml"


def save_epochs(block_epochs: BlockEpochs, metadata_dir: Path) -> Path:
    path = epochs_yaml_path(metadata_dir, block_epochs.spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "animal": block_epochs.spec.animal,
        "block_path": str(block_epochs.spec.block_path),
        "mount_type": block_epochs.spec.mount_type,
        "amplitude_key": block_epochs.amplitude_key,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": [
            {
                "start_frame": int(e.start_frame),
                "end_frame": int(e.end_frame),
                "eye": e.eye,
                "notes": e.notes,
            }
            for e in block_epochs.epochs
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def load_epochs(path: Path, spec: JitterBlockSpec | None = None) -> BlockEpochs:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if spec is None:
        spec = JitterBlockSpec(
            animal=str(data["animal"]),
            block_path=Path(data["block_path"]),
            mount_type=str(data["mount_type"]),  # type: ignore[arg-type]
        )
    epochs = [
        Epoch(
            start_frame=int(e["start_frame"]),
            end_frame=int(e["end_frame"]),
            eye=str(e.get("eye", "both")),
            notes=str(e.get("notes", "")),
        )
        for e in (data.get("epochs") or [])
    ]
    return BlockEpochs(
        spec=spec,
        epochs=epochs,
        amplitude_key=str(data.get("amplitude_key", "top_correlation_dist")),
    )


def pick_epochs_interactive(
    spec: JitterBlockSpec,
    *,
    eye: EyeName = "both",
    key: str = "top_correlation_dist",
    existing: list[Epoch] | None = None,
) -> BlockEpochs:
    """
    Matplotlib SpanSelector UI: drag to add epochs on the amplitude trace.

    Keys: ``u`` undo last, ``c`` clear, ``enter``/close window to finish.
    """
    report = load_jitter_report(spec.report_path)
    traces = extract_amplitude_trace(report, eye=eye, key=key)
    epochs: list[Epoch] = list(existing or [])

    n_panels = len(traces)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(10, 2.2 * n_panels),
        sharex=True,
        squeeze=False,
    )
    axes_flat = list(axes.ravel())
    selectors: list[SpanSelector] = []

    def _redraw_spans() -> None:
        for ax, (ename, amp) in zip(axes_flat, traces.items()):
            ax.cla()
            ax.plot(np.arange(len(amp)), amp, color="0.2", lw=0.6)
            for ep in epochs:
                if not _epoch_matches_eye(ep, ename):
                    continue
                ax.axvspan(ep.start_frame, ep.end_frame, color="C0", alpha=0.25)
            ax.set_ylabel(f"{ename}\n{key}", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes_flat[-1].set_xlabel("Frame", fontsize=9)
        fig.suptitle(
            f"{spec.block_key} [{spec.mount_type}] — drag to add epoch; "
            "u=undo c=clear Enter/close=done",
            fontsize=10,
        )
        fig.canvas.draw_idle()

    def _on_select(ename: str):
        def handler(xmin, xmax):
            lo, hi = int(min(xmin, xmax)), int(max(xmin, xmax))
            if hi - lo < 2:
                return
            epochs.append(Epoch(start_frame=lo, end_frame=hi, eye=ename))
            _redraw_spans()

        return handler

    for ax, ename in zip(axes_flat, traces):
        sel = SpanSelector(
            ax,
            _on_select(ename),
            "horizontal",
            useblit=True,
            interactive=False,
            drag_from_anywhere=True,
        )
        selectors.append(sel)

    def _on_key(event):
        if event.key == "u" and epochs:
            epochs.pop()
            _redraw_spans()
        elif event.key == "c":
            epochs.clear()
            _redraw_spans()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _redraw_spans()
    plt.show()

    return BlockEpochs(spec=spec, epochs=epochs, amplitude_key=key)


def _epoch_matches_eye(ep: Epoch, eye_name: str) -> bool:
    if ep.eye in ("both", eye_name):
        return True
    alias = eye_name.replace("_eye", "")
    return ep.eye in (alias, f"{alias}_eye")


def samples_in_epochs(
    amp: np.ndarray,
    epochs: list[Epoch],
    *,
    eye_name: str,
) -> np.ndarray:
    parts = []
    for ep in epochs:
        if not _epoch_matches_eye(ep, eye_name):
            continue
        lo = max(0, int(ep.start_frame))
        hi = min(len(amp), int(ep.end_frame))
        if hi > lo:
            parts.append(amp[lo:hi])
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def seed_middle_epochs(
    specs: list[JitterBlockSpec],
    metadata_dir: Path,
    *,
    frac: float = 0.5,
    key: str = "top_correlation_dist",
) -> list[Path]:
    """Non-interactive: keep the middle ``frac`` of frames as one epoch per eye."""
    frac = float(np.clip(frac, 0.05, 1.0))
    written = []
    for spec in specs:
        report = load_jitter_report(spec.report_path)
        traces = extract_amplitude_trace(report, eye="both", key=key)
        epochs: list[Epoch] = []
        for ename, amp in traces.items():
            n = len(amp)
            span = int(n * frac)
            start = max(0, (n - span) // 2)
            end = min(n, start + span)
            epochs.append(Epoch(start_frame=start, end_frame=end, eye=ename, notes="seed_middle"))
        be = BlockEpochs(spec=spec, epochs=epochs, amplitude_key=key)
        written.append(save_epochs(be, metadata_dir))
        print(f"[seed] {spec.block_key}: {len(epochs)} epochs → {written[-1]}")
    return written


@dataclass
class BlockSamples:
    """Epoch samples of one block, already scaled to ``units``."""

    spec: JitterBlockSpec
    samples: dict[str, np.ndarray]  # eye name -> samples inside the epochs
    units: str = "um"

    @property
    def block_key(self) -> str:
        return self.spec.block_key

    @property
    def mount_type(self) -> str:
        return self.spec.mount_type

    @property
    def values(self) -> np.ndarray:
        """All eyes concatenated, non-finite dropped."""
        parts = [a for a in self.samples.values() if a.size]
        if not parts:
            return np.array([], dtype=float)
        arr = np.concatenate(parts)
        return arr[np.isfinite(arr)]

    def stats(self) -> dict[str, Any]:
        arr = self.values
        return {
            "block_key": self.block_key,
            "animal": self.spec.animal,
            "block": self.spec.block_path.name,
            "mount_type": self.mount_type,
            "units": self.units,
            "n_samples": int(arr.size),
            "max": float(np.max(arr)) if arr.size else None,
            "p95": float(np.percentile(arr, 95)) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
            "mean": float(np.mean(arr)) if arr.size else None,
        }

    def label(self) -> str:
        """One-line checklist caption: identity, mean / p95 / max, sample count."""
        arr = self.values
        if not arr.size:
            return f"[{self.mount_type}] {self.block_key} — no epoch samples"
        unit = unit_label(self.units)
        return (
            f"[{self.mount_type}] {self.block_key} — "
            f"mean {np.mean(arr):,.1f} | p95 {np.percentile(arr, 95):,.1f} | "
            f"max {np.max(arr):,.1f} {unit} (n={arr.size:,})"
        )


@dataclass
class EpochRecord:
    """
    One saved epoch of one eye: the displacement values plus the video frame index
    behind each of them, which is what makes an exported pool reproducible.
    """

    spec: JitterBlockSpec
    eye: str
    epoch_index: int
    start_frame: int
    end_frame: int
    frames: np.ndarray  # int frame index per value
    values: np.ndarray  # displacement, scaled to ``units``
    units: str = "um"
    scale_um_per_px: float | None = None  # None when units == "px"
    notes: str = ""

    @property
    def block_key(self) -> str:
        return self.spec.block_key

    @property
    def mount_type(self) -> str:
        return self.spec.mount_type

    @property
    def epoch_key(self) -> str:
        """Stable id across exports: block, eye and the frame span it came from."""
        return f"{self.block_key}_{self.eye}_{self.start_frame}_{self.end_frame}"


def collect_epoch_records(
    specs: list[JitterBlockSpec],
    metadata_dir: Path,
    *,
    key: str = "top_correlation_dist",
    units: str = "um",
    verbose: bool = True,
) -> list[EpochRecord]:
    """
    Flatten every saved epoch into :class:`EpochRecord`s, scaled to ``units``.

    Same extraction as :func:`collect_block_samples` (and in the same order), but
    keeping each epoch separate and carrying its frame indices.
    """
    units = str(units).lower()
    if units not in {"um", "px"}:
        raise ValueError(f"units must be 'um' or 'px', got {units!r}")

    scales: dict[str, Any] = {}
    if units == "um":
        from eye_tracking_system_tools.analysis.pixel_calibration import require_pixel_sizes

        scales = require_pixel_sizes([s.block_path for s in specs])

    records: list[EpochRecord] = []
    for spec in specs:
        ep_path = epochs_yaml_path(metadata_dir, spec)
        if not ep_path.exists():
            if verbose:
                print(f"[skip] no epochs yet for {spec.block_key}: {ep_path}")
            continue
        block_epochs = load_epochs(ep_path, spec)
        report = load_jitter_report(spec.report_path)
        traces = extract_amplitude_trace(report, eye="both", key=key)
        for ename, amp in traces.items():
            scale = scales[str(spec.block_path)].um_per_px(ename) if units == "um" else None
            for i, ep in enumerate(block_epochs.epochs):
                if not _epoch_matches_eye(ep, ename):
                    continue
                lo = max(0, int(ep.start_frame))
                hi = min(len(amp), int(ep.end_frame))
                if hi <= lo:
                    continue
                values = amp[lo:hi]
                records.append(
                    EpochRecord(
                        spec=spec,
                        eye=ename,
                        epoch_index=i,
                        start_frame=lo,
                        end_frame=hi,
                        frames=np.arange(lo, hi, dtype=np.int64),
                        values=values * scale if scale is not None else values,
                        units=units,
                        scale_um_per_px=scale,
                        notes=ep.notes,
                    )
                )
    return records


def collect_block_samples(
    specs: list[JitterBlockSpec],
    metadata_dir: Path,
    *,
    key: str = "top_correlation_dist",
    units: str = "um",
    verbose: bool = True,
) -> list[BlockSamples]:
    """
    Load each block's epoch samples once, scaled to ``units``.

    ``units="um"`` (default) scales each eye by that block's
    ``analysis/LR_pix_size.csv`` calibration; ``units="px"`` keeps raw pixels.
    Blocks without a saved epoch YAML come back with no samples.
    """
    records = collect_epoch_records(
        specs, metadata_dir, key=key, units=units, verbose=verbose
    )
    by_block: dict[str, dict[str, list[np.ndarray]]] = {}
    for rec in records:
        by_block.setdefault(rec.block_key, {}).setdefault(rec.eye, []).append(rec.values)
    return [
        BlockSamples(
            spec=spec,
            samples={
                eye: np.concatenate(parts)
                for eye, parts in by_block.get(spec.block_key, {}).items()
            },
            units=str(units).lower(),
        )
        for spec in specs
    ]


def pool_block_samples(
    block_samples: list[BlockSamples],
    *,
    include: Collection[str] | None = None,
) -> dict[str, np.ndarray]:
    """Concatenate pre-loaded samples by mount type, keeping only ``include`` keys."""
    keys = None if include is None else {str(k) for k in include}
    if keys is not None:
        known = {bs.block_key for bs in block_samples}
        unknown = keys - known
        if unknown:
            raise KeyError(f"Unknown block key(s): {sorted(unknown)}")

    pools: dict[str, list[np.ndarray]] = {"modular": [], "rigid": [], "mouse": []}
    for bs in block_samples:
        if keys is not None and bs.block_key not in keys:
            continue
        for samp in bs.samples.values():
            if samp.size:
                pools[bs.mount_type].append(samp)
    return {
        k: (np.concatenate(v) if v else np.array([], dtype=float))
        for k, v in pools.items()
    }


def pool_by_mount(
    specs: list[JitterBlockSpec],
    metadata_dir: Path,
    *,
    key: str = "top_correlation_dist",
    units: str = "um",
    include: Collection[str] | None = None,
) -> dict[str, np.ndarray]:
    """Concatenate epoch samples keyed by mount_type (``include`` = block keys)."""
    collected = collect_block_samples(specs, metadata_dir, key=key, units=units)
    return pool_block_samples(collected, include=include)


def unit_label(units: str) -> str:
    """Axis label for a units code (``um`` → ``µm``)."""
    return "µm" if str(units).lower() == "um" else str(units)


def _hist_percent_frames(
    distances: np.ndarray,
    *,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return bins[:-1], np.zeros(len(bins) - 1)
    hist, edges = np.histogram(distances, bins=bins)
    pct = (hist / distances.size) * 100.0
    return edges[:-1], pct


def figure_modular_vs_rigid(
    pools: dict[str, np.ndarray],
    *,
    xmax: float | None = None,
    n_bins: int = 15,
    units: str = "um",
):
    """Fig 1e–style overlaid histograms for modular vs rigid. Returns the figure."""
    mod = pools.get("modular", np.array([]))
    rig = pools.get("rigid", np.array([]))
    finite = np.concatenate([a[np.isfinite(a)] for a in (mod, rig) if a.size])
    if finite.size == 0:
        raise ValueError("No modular/rigid epoch samples to plot")
    xmax = float(xmax) if xmax is not None else float(np.nanpercentile(finite, 99.5))
    xmax = max(xmax, 1e-6)
    bins = np.linspace(0, xmax, n_bins + 1)

    fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.7), dpi=150)
    width = np.diff(bins)
    x0, y0 = _hist_percent_frames(mod, bins=bins)
    x1, y1 = _hist_percent_frames(rig, bins=bins)
    ax.bar(
        x0,
        y0,
        width=width,
        align="edge",
        color="#0072B2",
        edgecolor="black",
        alpha=0.55,
        label=f"modular (n={mod.size})",
    )
    ax.bar(
        x1,
        y1,
        width=width,
        align="edge",
        color="#D55E00",
        edgecolor="black",
        alpha=0.55,
        label=f"rigid (n={rig.size})",
    )
    ax.set_xlabel(f"Displacement [{unit_label(units)}]", fontsize=10)
    ax.set_ylabel("% frames", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, xmax)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return fig


def figure_mouse_histogram(
    pools: dict[str, np.ndarray],
    *,
    xmax: float | None = None,
    n_bins: int = 15,
    units: str = "um",
):
    """Dedicated mouse jitter histogram. Returns the figure."""
    mouse = pools.get("mouse", np.array([]))
    mouse = mouse[np.isfinite(mouse)]
    if mouse.size == 0:
        raise ValueError("No mouse epoch samples to plot")
    xmax = float(xmax) if xmax is not None else float(np.nanpercentile(mouse, 99.5))
    xmax = max(xmax, 1e-6)
    bins = np.linspace(0, xmax, n_bins + 1)
    x, y = _hist_percent_frames(mouse, bins=bins)

    fig, ax = plt.subplots(1, 1, figsize=(2, 1.6), dpi=150)
    ax.bar(
        x,
        y,
        width=np.diff(bins),
        align="edge",
        color="gray",
        edgecolor="black",
        label=f"mouse (n={mouse.size})",
    )
    ax.set_xlabel(f"Displacement [{unit_label(units)}]", fontsize=10)
    ax.set_ylabel("% frames", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, xmax)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return fig


def _save_figure(fig, out_pdf: Path, *, show: bool = False) -> Path:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    if show:
        from IPython.display import display

        display(fig)
    plt.close(fig)
    return out_pdf


def plot_modular_vs_rigid(
    pools: dict[str, np.ndarray],
    out_pdf: Path,
    *,
    xmax: float | None = None,
    n_bins: int = 15,
    units: str = "um",
    show: bool = False,
) -> Path:
    fig = figure_modular_vs_rigid(pools, xmax=xmax, n_bins=n_bins, units=units)
    return _save_figure(fig, out_pdf, show=show)


def plot_mouse_histogram(
    pools: dict[str, np.ndarray],
    out_pdf: Path,
    *,
    xmax: float | None = None,
    n_bins: int = 15,
    units: str = "um",
    show: bool = False,
) -> Path:
    fig = figure_mouse_histogram(pools, xmax=xmax, n_bins=n_bins, units=units)
    return _save_figure(fig, out_pdf, show=show)


def write_pool_summary(
    pools: dict[str, np.ndarray],
    metadata_dir: Path,
    *,
    units: str = "um",
    blocks: list[BlockSamples] | None = None,
) -> Path:
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "units": units,
        "note": (
            "Amplitude from jitter_report_dict top_correlation_dist, scaled per eye "
            "by analysis/LR_pix_size.csv (mm/px)."
            if units == "um"
            else "Amplitude from jitter_report_dict top_correlation_dist, raw camera pixels."
        ),
        "groups": {},
    }
    for name, arr in pools.items():
        arr = arr[np.isfinite(arr)]
        summary["groups"][name] = {
            "n_samples": int(arr.size),
            "mean": float(np.mean(arr)) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
            "p95": float(np.percentile(arr, 95)) if arr.size else None,
            "max": float(np.max(arr)) if arr.size else None,
        }
    if blocks is not None:
        summary["blocks_used"] = [bs.stats() for bs in blocks]
    path = metadata_dir / "jitter_pool_summary.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return path


def run_pick_all(
    specs: list[JitterBlockSpec],
    metadata_dir: Path,
    *,
    eye: EyeName = "both",
) -> list[Path]:
    written = []
    for spec in specs:
        existing_path = epochs_yaml_path(metadata_dir, spec)
        existing = load_epochs(existing_path, spec).epochs if existing_path.exists() else []
        print(f"Picking epochs for {spec.block_key} ({spec.mount_type}) …")
        block_epochs = pick_epochs_interactive(spec, eye=eye, existing=existing)
        written.append(save_epochs(block_epochs, metadata_dir))
        print(f"  saved {len(block_epochs.epochs)} epochs → {written[-1]}")
    return written


def prompt_block_selection(block_samples: list[BlockSamples]) -> list[str]:
    """
    Text checklist: print every block with its mean / p95 / max displacement and
    read the numbers to keep (blank/``all`` keeps everything, ``-3`` drops block 3).
    """
    print("\nBlocks available for pooling:")
    for i, bs in enumerate(block_samples, start=1):
        print(f"  {i:>2}. {bs.label()}")
    print(
        "Enter the numbers to pool (e.g. '1 3 5', ranges '1-4', "
        "'-2' to drop one, blank = all):"
    )
    try:
        raw = input("> ").strip()
    except EOFError:
        raw = ""
    if not raw or raw.lower() == "all":
        return [bs.block_key for bs in block_samples]

    keep: set[int] = set()
    drop: set[int] = set()
    for token in raw.replace(",", " ").split():
        if re.fullmatch(r"-\d+", token):  # exclusion, e.g. -3
            drop.add(int(token[1:]))
        elif re.fullmatch(r"\d+", token):
            keep.add(int(token))
        elif re.fullmatch(r"\d+-\d+", token):  # inclusive range, e.g. 1-4
            lo, hi = (int(v) for v in token.split("-"))
            keep.update(range(lo, hi + 1))
        else:
            raise ValueError(f"Cannot parse selection token {token!r}")
    if not keep:  # only exclusions given → start from everything
        keep = set(range(1, len(block_samples) + 1))
    chosen = sorted(keep - drop)
    bad = [i for i in chosen if not 1 <= i <= len(block_samples)]
    if bad:
        raise ValueError(f"Out-of-range selection: {bad}")
    return [block_samples[i - 1].block_key for i in chosen]


def run_plot_pooled(
    specs: list[JitterBlockSpec],
    figures_dir: Path,
    metadata_dir: Path,
    *,
    units: str = "um",
    n_bins: int = 15,
    show: bool = False,
    include: Collection[str] | None = None,
    select: bool = False,
    block_samples: list[BlockSamples] | None = None,
) -> dict[str, Path]:
    """
    Pool epoch samples and write the histogram PDFs (``show`` renders inline).

    ``include`` restricts pooling to those ``block_key``s; ``select=True`` asks for
    them on stdin, listing each block's mean / p95 / max displacement (the notebook
    uses ``jitter_gui.JitterPoolSelector`` for the same choice as checkboxes).
    ``block_samples`` reuses an already-loaded collection instead of re-reading the
    reports. ``units="um"`` requires ``analysis/LR_pix_size.csv`` in every block.
    """
    collected = (
        block_samples
        if block_samples is not None
        else collect_block_samples(specs, metadata_dir, units=units)
    )
    if select and include is None:
        include = prompt_block_selection(collected)
    pools = pool_block_samples(collected, include=include)

    used = [
        bs
        for bs in collected
        if bs.values.size and (include is None or bs.block_key in set(include))
    ]
    print(f"pooling {len(used)}/{len(collected)} block(s): {[bs.block_key for bs in used]}")

    out: dict[str, Path] = {}
    out["summary"] = write_pool_summary(pools, metadata_dir, units=units, blocks=used)
    if pools["modular"].size or pools["rigid"].size:
        out["modular_vs_rigid"] = plot_modular_vs_rigid(
            pools,
            figures_dir / "jitter_modular_vs_rigid.pdf",
            n_bins=n_bins,
            units=units,
            show=show,
        )
    else:
        print("[warn] no modular/rigid samples — skip comparison hist")
    if pools["mouse"].size:
        out["mouse"] = plot_mouse_histogram(
            pools,
            figures_dir / "jitter_mouse.pdf",
            n_bins=n_bins,
            units=units,
            show=show,
        )
    else:
        print("[warn] no mouse samples — skip mouse hist")
    return out


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Jitter epoch picker + pooled modular/rigid/mouse histograms."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo / "configs" / "jitter_mount_blocks.yaml",
    )
    parser.add_argument("--out-root", type=Path, default=repo / "outputs")
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Empty → jitter_latest overwrite; else jitter_<tag>",
    )
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Interactive SpanSelector epoch picking (requires GUI)",
    )
    parser.add_argument(
        "--seed-middle-frac",
        type=float,
        default=None,
        help="Non-interactive: seed middle fraction of each trace as an epoch",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Pool existing epoch YAMLs and write histograms",
    )
    parser.add_argument(
        "--units",
        type=str,
        choices=("um", "px"),
        default="um",
        help="µm uses each block's analysis/LR_pix_size.csv; px keeps raw pixels",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        default=None,
        metavar="BLOCK_KEY",
        help="Pool only these blocks (e.g. PV_143_block_001)",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Choose the blocks to pool from a printed checklist of peak displacements",
    )
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args(argv)

    if not args.pick and not args.plot and args.seed_middle_frac is None:
        args.plot = True  # default: plot if epochs exist

    run = resolve_run_dir(
        args.out_root,
        args.tag or None,
        prefix="jitter",
        default_name="jitter_latest",
    )
    print(f"run: {run.run_dir}")
    specs = load_jitter_registry(args.registry)
    print(f"registry: {len(specs)} blocks")

    if args.seed_middle_frac is not None:
        seed_middle_epochs(specs, run.metadata_dir, frac=args.seed_middle_frac)
    if args.pick:
        run_pick_all(specs, run.metadata_dir)
    if args.plot:
        written = run_plot_pooled(
            specs,
            run.figures_dir,
            run.metadata_dir,
            units=args.units,
            n_bins=args.n_bins,
            include=args.include,
            select=args.select,
        )
        print("Wrote:")
        for k, p in written.items():
            print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
