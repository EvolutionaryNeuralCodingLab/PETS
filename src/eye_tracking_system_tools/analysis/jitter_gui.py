"""
Notebook (ipywidgets) front-ends for the jitter workflow.

* :class:`JitterBlockBrowser` — browse the filesystem, multi-select block folders,
  tag them modular / rigid / mouse and write ``configs/jitter_mount_blocks.yaml``.
  Pass ``require="eye"`` to stage any block with eye CSVs (span tool 1) instead of
  requiring a jitter report.
* :class:`JitterEpochPicker` — per-block amplitude trace with range-slider epoch
  selection, saved to ``metadata/jitter_epochs/*.yaml``.
* :class:`PixelCalibrationPanel` — create the missing ``analysis/LR_pix_size.csv``.
* :class:`JitterPoolSelector` — checklist of blocks with their mean / p95 / max
  displacement; pools and plots only the ticked ones.

Both render with the inline matplotlib backend (no ipympl / Qt needed).
Used by ``development/jitter_mount_pipeline.ipynb`` and
``development/eye_span_pipeline.ipynb``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from eye_tracking_system_tools.analysis.eye_movement_span import (
    find_blocks_with_eye_data,
    has_eye_span_data,
)
from eye_tracking_system_tools.analysis.jitter_epochs import (
    BlockEpochs,
    Epoch,
    JitterBlockSpec,
    MountType,
    collect_block_samples,
    epochs_yaml_path,
    extract_amplitude_trace,
    find_blocks_with_reports,
    guess_mount_type,
    has_jitter_report,
    infer_animal,
    load_epochs,
    load_jitter_report,
    read_registry_blocks,
    run_plot_pooled,
    save_epochs,
    write_jitter_registry,
)
from eye_tracking_system_tools.analysis.pixel_calibration import (
    DEFAULT_KNOWN_DIST_MM,
    calibrate_block,
    find_eye_videos,
    manual_calibration,
    read_pixel_size,
)

MOUNT_TYPES: tuple[str, ...] = ("modular", "rigid", "mouse")
RequireKind = Literal["jitter", "eye"]


def default_roots(repo: Path | None = None) -> list[Path]:
    """Quick-jump roots: mounted experiment volumes, sample data, home, repo."""
    candidates = [
        Path("/Volumes/Data-2/Nimrod/experiments"),
        Path("/Volumes/Data/Nimrod/experiments"),
        Path("/Volumes/Data-1/Nimrod/experiments"),
        Path.home() / "sample_data",
        Path.home(),
    ]
    if repo is not None:
        candidates.append(Path(repo))
    seen: set[str] = set()
    roots = []
    for c in candidates:
        if c.is_dir() and str(c) not in seen:
            seen.add(str(c))
            roots.append(c)
    return roots


class JitterBlockBrowser:
    """
    Filesystem browser for staging blocks into a registry YAML.

    ``browser = JitterBlockBrowser(registry_path); browser`` renders the UI.
    ``browser.specs`` returns the staged :class:`JitterBlockSpec` list.

    ``registry_format``:
      * ``"jitter"`` (default) — flat ``blocks:`` list with ``mount_type``
      * ``"paper"`` — ``animals: {name: [paths]}`` for the flexible paper / span tools

    ``require``:
      * ``"jitter"`` (default) — only stage folders with ``jitter_report_dict.pkl``
      * ``"eye"`` — stage folders with eye CSVs (``center_x``/``center_y`` or Kerr angles)
    """

    def __init__(
        self,
        registry_path: Path | str,
        *,
        roots: list[Path] | None = None,
        start: Path | str | None = None,
        repo: Path | None = None,
        load_existing: bool = True,
        registry_format: str = "jitter",
        require: RequireKind = "jitter",
    ) -> None:
        fmt = str(registry_format).strip().lower()
        if fmt not in {"jitter", "paper"}:
            raise ValueError(f"registry_format must be 'jitter' or 'paper', got {registry_format!r}")
        req = str(require).strip().lower()
        if req not in {"jitter", "eye"}:
            raise ValueError(f"require must be 'jitter' or 'eye', got {require!r}")
        self.registry_format = fmt
        self.require: RequireKind = req  # type: ignore[assignment]
        self.registry_path = Path(registry_path)
        self.roots = [Path(r) for r in (roots or default_roots(repo))]
        self._staged: dict[str, JitterBlockSpec] = {}
        if load_existing:
            for spec in self._read_existing():
                self._staged[str(spec.block_path)] = spec
        start_path = Path(start) if start else (self.roots[0] if self.roots else Path.home())
        self.cwd = start_path if start_path.is_dir() else Path.home()
        self._build()
        self._refresh_listing()
        self._refresh_staged()

    def _is_eligible(self, block_path: Path) -> bool:
        if self.require == "eye":
            return has_eye_span_data(block_path)
        return has_jitter_report(block_path)

    def _eligible_mark(self) -> str:
        return "eye data" if self.require == "eye" else "jitter report"

    def _eligibility_fail_reason(self, block_path: Path) -> str:
        if self.require == "eye":
            return f"no usable eye CSV in {block_path.name}/analysis"
        return f"no analysis/jitter_report_dict.pkl in {block_path.name}"

    def _read_existing(self) -> list[JitterBlockSpec]:
        """Load staged specs from the registry path (format-aware, tolerant)."""
        if self.registry_format == "paper":
            from eye_tracking_system_tools.analysis.block_registry import read_paper_registry

            return [
                JitterBlockSpec(
                    animal=s.animal,
                    block_path=s.block_path,
                    mount_type="modular",
                )
                for s in read_paper_registry(self.registry_path)
            ]
        return read_registry_blocks(self.registry_path)

    # ---------------------------------------------------------------- widgets
    def _build(self) -> None:
        self.path_label = widgets.HTML()
        self.status = widgets.HTML()

        self.root_dd = widgets.Dropdown(
            options=[(str(r), str(r)) for r in self.roots] or [("(none)", "")],
            description="Root:",
            layout=widgets.Layout(width="520px"),
            style={"description_width": "60px"},
        )
        self.path_box = widgets.Text(
            value=str(self.cwd),
            placeholder="/path/to/animal/date",
            description="Path:",
            layout=widgets.Layout(width="520px"),
            style={"description_width": "60px"},
        )
        self.go_btn = widgets.Button(description="Go", layout=widgets.Layout(width="60px"))
        self.up_btn = widgets.Button(description="Up", icon="level-up", layout=widgets.Layout(width="90px"))
        self.open_btn = widgets.Button(description="Open", icon="folder-open", layout=widgets.Layout(width="100px"))
        self.refresh_btn = widgets.Button(description="Refresh", icon="refresh", layout=widgets.Layout(width="100px"))

        self.listing = widgets.SelectMultiple(
            options=[],
            rows=14,
            layout=widgets.Layout(width="560px"),
        )
        self.mount_dd = widgets.Dropdown(
            options=MOUNT_TYPES,
            value="modular",
            description="Mount:",
            layout=widgets.Layout(width="200px"),
            style={"description_width": "55px"},
        )
        self.auto_mouse = widgets.Checkbox(
            value=True,
            description="auto-tag M_* as mouse",
            indent=False,
            layout=widgets.Layout(width="200px"),
        )
        self.add_btn = widgets.Button(
            description="Add selected", icon="plus", button_style="success",
            layout=widgets.Layout(width="150px"),
        )
        scan_tip = (
            "Recursively find block folders with eye CSVs"
            if self.require == "eye"
            else "Recursively find block folders that already have a jitter report"
        )
        self.scan_btn = widgets.Button(
            description="Scan & add below", icon="search",
            tooltip=scan_tip,
            layout=widgets.Layout(width="170px"),
        )
        self.scan_depth = widgets.IntSlider(
            value=4, min=1, max=6, description="depth:",
            layout=widgets.Layout(width="220px"), style={"description_width": "45px"},
        )

        self.staged_list = widgets.SelectMultiple(
            options=[], rows=10, layout=widgets.Layout(width="560px")
        )
        self.remove_btn = widgets.Button(
            description="Remove", icon="trash", button_style="warning",
            layout=widgets.Layout(width="110px"),
        )
        self.retag_btn = widgets.Button(
            description="Re-tag", icon="tags",
            tooltip="Set the Mount dropdown value on the selected staged blocks",
            layout=widgets.Layout(width="110px"),
        )
        self.clear_btn = widgets.Button(description="Clear all", layout=widgets.Layout(width="110px"))
        self.save_btn = widgets.Button(
            description="Save registry", icon="save", button_style="primary",
            layout=widgets.Layout(width="150px"),
        )
        self.reload_btn = widgets.Button(description="Reload from YAML", layout=widgets.Layout(width="160px"))

        self.go_btn.on_click(lambda _: self._navigate(Path(self.path_box.value)))
        self.up_btn.on_click(lambda _: self._navigate(self.cwd.parent))
        self.open_btn.on_click(lambda _: self._open_selected())
        self.refresh_btn.on_click(lambda _: self._refresh_listing())
        self.root_dd.observe(self._on_root, names="value")
        self.listing.observe(self._on_listing_click, names="value")
        self.add_btn.on_click(lambda _: self._add_selected())
        self.scan_btn.on_click(lambda _: self._scan_and_add())
        self.remove_btn.on_click(lambda _: self._remove_selected())
        self.retag_btn.on_click(lambda _: self._retag_selected())
        self.clear_btn.on_click(lambda _: self._clear())
        self.save_btn.on_click(lambda _: self.save())
        self.reload_btn.on_click(lambda _: self._reload())

        check_label = (
            "✓ = eye CSV present" if self.require == "eye" else "✓ = jitter report present"
        )
        self.widget = widgets.VBox(
            [
                widgets.HTML(f"<b>1. Browse to block folders</b> ({check_label})"),
                widgets.HBox([self.root_dd, self.up_btn, self.refresh_btn]),
                widgets.HBox([self.path_box, self.go_btn, self.open_btn]),
                self.path_label,
                self.listing,
                widgets.HBox([self.mount_dd, self.auto_mouse, self.add_btn, self.scan_btn, self.scan_depth]),
                widgets.HTML("<b>2. Staged blocks</b>"),
                self.staged_list,
                widgets.HBox([self.remove_btn, self.retag_btn, self.clear_btn, self.save_btn, self.reload_btn]),
                self.status,
            ]
        )

    def _ipython_display_(self) -> None:
        display(self.widget)

    # ---------------------------------------------------------------- helpers
    def _set_status(self, msg: str, *, level: str = "info") -> None:
        color = {"info": "#333", "ok": "#177245", "warn": "#b35c00", "err": "#a11"}[level]
        self.status.value = f"<span style='color:{color}'>{msg}</span>"

    def _navigate(self, path: Path) -> None:
        path = Path(path).expanduser()
        if not path.is_dir():
            self._set_status(f"Not a directory: {path}", level="err")
            return
        self.cwd = path
        self.path_box.value = str(path)
        self._refresh_listing()

    def _on_root(self, change) -> None:
        if change["new"]:
            self._navigate(Path(change["new"]))

    def _on_listing_click(self, change) -> None:
        sel = change["new"]
        if len(sel) == 1:
            self.path_box.value = sel[0]

    def _open_selected(self) -> None:
        sel = self.listing.value
        if len(sel) != 1:
            self._set_status("Select exactly one folder to open.", level="warn")
            return
        self._navigate(Path(sel[0]))

    def _refresh_listing(self) -> None:
        try:
            entries = sorted(
                (p for p in self.cwd.iterdir() if p.is_dir() and not p.name.startswith(".")),
                key=lambda p: p.name,
            )
        except (PermissionError, OSError) as exc:
            self._set_status(f"Cannot list {self.cwd}: {exc}", level="err")
            entries = []
        options = []
        n_blocks = 0
        for p in entries:
            if self._is_eligible(p):
                n_blocks += 1
                mark = "✓"
            else:
                mark = "📁"
            staged = " [staged]" if str(p) in self._staged else ""
            options.append((f"{mark} {p.name}{staged}", str(p)))
        self.listing.options = options
        self.path_label.value = (
            f"<code>{self.cwd}</code> — {len(options)} folders, "
            f"{n_blocks} with {self._eligible_mark()}"
        )

    def _mount_for(self, animal: str) -> MountType:
        if self.auto_mouse.value and guess_mount_type(animal) == "mouse":
            return "mouse"
        return self.mount_dd.value  # type: ignore[return-value]

    def _stage(self, block_path: Path) -> tuple[bool, str]:
        block_path = Path(block_path)
        if not self._is_eligible(block_path):
            return False, self._eligibility_fail_reason(block_path)
        animal = infer_animal(block_path)
        self._staged[str(block_path)] = JitterBlockSpec(
            animal=animal,
            block_path=block_path,
            mount_type=self._mount_for(animal),
        )
        return True, ""

    def _add_selected(self) -> None:
        sel = list(self.listing.value)
        if not sel:
            self._set_status("Nothing selected (ctrl/shift-click for multiple).", level="warn")
            return
        added, skipped = 0, []
        for s in sel:
            ok, why = self._stage(Path(s))
            added += int(ok)
            if not ok:
                skipped.append(why)
        self._refresh_staged()
        self._refresh_listing()
        msg = f"Staged {added} block(s)."
        if skipped:
            msg += f" Skipped {len(skipped)}: {skipped[0]}" + (" …" if len(skipped) > 1 else "")
        self._set_status(msg, level="ok" if added else "warn")

    def _scan_and_add(self) -> None:
        targets = [Path(s) for s in self.listing.value] or [self.cwd]
        self._set_status(f"Scanning {len(targets)} folder(s) …")
        found: list[Path] = []
        finder = find_blocks_with_eye_data if self.require == "eye" else find_blocks_with_reports
        for t in targets:
            found.extend(finder(t, max_depth=int(self.scan_depth.value)))
        for block in found:
            self._stage(block)
        self._refresh_staged()
        self._refresh_listing()
        self._set_status(
            f"Scan found {len(found)} block(s) with {self._eligible_mark()}.",
            level="ok" if found else "warn",
        )

    def _refresh_staged(self) -> None:
        options = []
        for key in sorted(self._staged):
            spec = self._staged[key]
            options.append((f"[{spec.mount_type}] {spec.animal} — {spec.block_path}", key))
        self.staged_list.options = options

    def _remove_selected(self) -> None:
        for key in list(self.staged_list.value):
            self._staged.pop(key, None)
        self._refresh_staged()
        self._refresh_listing()
        self._set_status("Removed selected staged block(s).", level="ok")

    def _retag_selected(self) -> None:
        keys = list(self.staged_list.value)
        if not keys:
            self._set_status("Select staged rows to re-tag.", level="warn")
            return
        for key in keys:
            spec = self._staged[key]
            self._staged[key] = JitterBlockSpec(
                animal=spec.animal,
                block_path=spec.block_path,
                mount_type=self.mount_dd.value,  # type: ignore[arg-type]
            )
        self._refresh_staged()
        self._set_status(f"Re-tagged {len(keys)} block(s) as {self.mount_dd.value}.", level="ok")

    def _clear(self) -> None:
        self._staged.clear()
        self._refresh_staged()
        self._refresh_listing()
        self._set_status("Cleared staged list (registry file untouched).", level="ok")

    def _reload(self) -> None:
        self._staged = {
            str(s.block_path): s for s in self._read_existing()
        }
        self._refresh_staged()
        self._refresh_listing()
        self._set_status(f"Reloaded {len(self._staged)} block(s) from {self.registry_path}", level="ok")

    # ----------------------------------------------------------------- public
    @property
    def specs(self) -> list[JitterBlockSpec]:
        return [self._staged[k] for k in sorted(self._staged)]

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path else self.registry_path
        specs = self.specs
        if not specs:
            self._set_status("Nothing staged — registry not written.", level="warn")
            return target
        if self.registry_format == "paper":
            from eye_tracking_system_tools.analysis.block_registry import write_paper_registry

            write_paper_registry(target, specs)
            animals = sorted({s.animal for s in specs})
            self._set_status(
                f"Saved {len(specs)} block(s) → {target} "
                f"(paper animals=[{', '.join(animals)}])",
                level="ok",
            )
            return target
        write_jitter_registry(target, specs)
        counts = {m: sum(1 for s in specs if s.mount_type == m) for m in MOUNT_TYPES}
        self._set_status(
            f"Saved {len(specs)} block(s) → {target} "
            f"(modular={counts['modular']}, rigid={counts['rigid']}, mouse={counts['mouse']})",
            level="ok",
        )
        return target


class JitterEpochPicker:
    """
    Per-block epoch selection on the extracted amplitude trace.

    Zoom with the *View* slider, set bounds with *Select*, then **Add epoch**.
    Epochs are written to ``metadata/jitter_epochs/<animal>_<block>_epochs.yaml``.
    """

    def __init__(
        self,
        specs: list[JitterBlockSpec],
        metadata_dir: Path | str,
        *,
        key: str = "top_correlation_dist",
        max_points: int = 4000,
    ) -> None:
        if not specs:
            raise ValueError("No blocks given — stage and save a registry first.")
        self.specs = list(specs)
        self.metadata_dir = Path(metadata_dir)
        self.key = key
        self.max_points = int(max_points)
        self._traces: dict[str, dict[str, np.ndarray]] = {}
        self._epochs: dict[str, list[Epoch]] = {}
        self._suspend = False
        self._build()
        self._load_block()

    # ---------------------------------------------------------------- widgets
    def _build(self) -> None:
        self.block_dd = widgets.Dropdown(
            options=[(f"[{s.mount_type}] {s.block_key}", i) for i, s in enumerate(self.specs)],
            value=0,
            description="Block:",
            layout=widgets.Layout(width="520px"),
            style={"description_width": "60px"},
        )
        self.eye_dd = widgets.Dropdown(
            options=[("both", "both"), ("left_eye", "left_eye"), ("right_eye", "right_eye")],
            value="both",
            description="Eye:",
            layout=widgets.Layout(width="220px"),
            style={"description_width": "45px"},
        )
        self.view = widgets.IntRangeSlider(
            value=(0, 1), min=0, max=1, description="View:", continuous_update=False,
            layout=widgets.Layout(width="760px"), style={"description_width": "55px"},
        )
        self.sel = widgets.IntRangeSlider(
            value=(0, 1), min=0, max=1, description="Select:", continuous_update=False,
            layout=widgets.Layout(width="760px"), style={"description_width": "55px"},
        )
        self.add_btn = widgets.Button(
            description="Add epoch", icon="plus", button_style="success",
            layout=widgets.Layout(width="130px"),
        )
        self.fullview_btn = widgets.Button(description="Reset view", layout=widgets.Layout(width="120px"))
        self.middle_btn = widgets.Button(
            description="Middle 50%", tooltip="Quick epoch over the central half of the trace",
            layout=widgets.Layout(width="130px"),
        )
        self.epoch_list = widgets.SelectMultiple(options=[], rows=6, layout=widgets.Layout(width="560px"))
        self.del_btn = widgets.Button(description="Remove", icon="trash", button_style="warning",
                                      layout=widgets.Layout(width="110px"))
        self.clear_btn = widgets.Button(description="Clear", layout=widgets.Layout(width="110px"))
        self.save_btn = widgets.Button(description="Save epochs", icon="save", button_style="primary",
                                       layout=widgets.Layout(width="140px"))
        self.save_all_btn = widgets.Button(description="Save all blocks", layout=widgets.Layout(width="150px"))
        self.plot_out = widgets.Output()
        self.status = widgets.HTML()

        self.block_dd.observe(lambda c: self._load_block(), names="value")
        self.eye_dd.observe(lambda c: self._redraw(), names="value")
        self.view.observe(self._on_view, names="value")
        self.sel.observe(lambda c: self._redraw(), names="value")
        self.add_btn.on_click(lambda _: self._add_epoch())
        self.fullview_btn.on_click(lambda _: self._reset_view())
        self.middle_btn.on_click(lambda _: self._middle_epoch())
        self.del_btn.on_click(lambda _: self._remove_epochs())
        self.clear_btn.on_click(lambda _: self._clear_epochs())
        self.save_btn.on_click(lambda _: self.save_current())
        self.save_all_btn.on_click(lambda _: self.save_all())

        self.widget = widgets.VBox(
            [
                widgets.HBox([self.block_dd, self.eye_dd]),
                self.plot_out,
                self.view,
                widgets.HBox([self.sel, self.add_btn]),
                widgets.HBox([self.fullview_btn, self.middle_btn]),
                widgets.HTML("<b>Epochs for this block</b>"),
                self.epoch_list,
                widgets.HBox([self.del_btn, self.clear_btn, self.save_btn, self.save_all_btn]),
                self.status,
            ]
        )

    def _ipython_display_(self) -> None:
        display(self.widget)

    # ---------------------------------------------------------------- helpers
    def _set_status(self, msg: str, *, level: str = "info") -> None:
        color = {"info": "#333", "ok": "#177245", "warn": "#b35c00", "err": "#a11"}[level]
        self.status.value = f"<span style='color:{color}'>{msg}</span>"

    @property
    def spec(self) -> JitterBlockSpec:
        return self.specs[self.block_dd.value]

    def _traces_for(self, spec: JitterBlockSpec) -> dict[str, np.ndarray]:
        cached = self._traces.get(spec.block_key)
        if cached is None:
            report = load_jitter_report(spec.report_path)
            cached = extract_amplitude_trace(report, eye="both", key=self.key)
            self._traces[spec.block_key] = cached
        return cached

    def _scales_for(self, spec: JitterBlockSpec) -> tuple[dict[str, float], str]:
        """Per-eye µm/px factors when the block is calibrated, else 1.0 in px."""
        try:
            ps = read_pixel_size(spec.block_path)
        except ValueError:
            ps = None
        if ps is None:
            return {}, "px"
        return {"left_eye": ps.l_um_per_px, "right_eye": ps.r_um_per_px}, "µm"

    def _load_block(self) -> None:
        spec = self.spec
        try:
            traces = self._traces_for(spec)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            self.plot_out.clear_output()
            self._set_status(str(exc), level="err")
            return
        n = max(len(a) for a in traces.values())
        last = max(n - 1, 1)
        if spec.block_key not in self._epochs:
            existing = epochs_yaml_path(self.metadata_dir, spec)
            self._epochs[spec.block_key] = (
                load_epochs(existing, spec).epochs if existing.exists() else []
            )
        self._suspend = True
        try:
            for slider in (self.view, self.sel):
                slider.min = 0
                slider.max = last
            self.view.value = (0, last)
            self.sel.value = (0, max(n // 10, 1))
        finally:
            self._suspend = False
        self._refresh_epoch_list()
        self._redraw()
        _, unit = self._scales_for(spec)
        cal = "calibrated (µm)" if unit == "µm" else "no pixel calibration — showing px"
        self._set_status(
            f"{spec.block_key}: {n} frames, "
            f"{len(self._epochs[spec.block_key])} epoch(s) loaded, {cal}.",
            level="info" if unit == "µm" else "warn",
        )

    def _on_view(self, change) -> None:
        """Zooming the view also narrows the selection slider, for frame-level precision."""
        lo, hi = (int(v) for v in change["new"])
        cur_lo, cur_hi = self.sel.value
        suspended = self._suspend
        self._suspend = True
        try:
            self.sel.min = 0
            self.sel.max = max(hi, 1)
            new_lo = min(max(int(cur_lo), lo), hi)
            new_hi = min(max(int(cur_hi), new_lo), hi)
            self.sel.value = (new_lo, new_hi)
            self.sel.min = lo
        finally:
            self._suspend = suspended
        if not self._suspend:
            self._redraw()

    def _reset_view(self) -> None:
        self.view.value = (self.view.min, self.view.max)

    def _epochs_here(self) -> list[Epoch]:
        return self._epochs.setdefault(self.spec.block_key, [])

    def _eyes(self) -> list[str]:
        traces = self._traces_for(self.spec)
        if self.eye_dd.value == "both":
            return list(traces)
        return [self.eye_dd.value]

    def _redraw(self) -> None:
        if self._suspend:
            return
        spec = self.spec
        try:
            traces = self._traces_for(spec)
        except Exception as exc:  # noqa: BLE001 - surfaced in the status line
            self._set_status(str(exc), level="err")
            return
        eyes = self._eyes()
        vlo, vhi = self.view.value
        slo, shi = self.sel.value
        scales, unit = self._scales_for(spec)

        with self.plot_out:
            self.plot_out.clear_output(wait=True)
            fig, axes = plt.subplots(
                len(eyes), 1, figsize=(11, 2.1 * len(eyes)), sharex=True, squeeze=False, dpi=110
            )
            for ax, ename in zip(axes.ravel(), eyes):
                amp = traces[ename] * scales.get(ename, 1.0)
                x = np.arange(len(amp))
                lo, hi = int(vlo), int(min(vhi + 1, len(amp)))
                xs, ys = x[lo:hi], amp[lo:hi]
                if xs.size > self.max_points:
                    stride = int(np.ceil(xs.size / self.max_points))
                    xs, ys = xs[::stride], ys[::stride]
                ax.plot(xs, ys, color="0.25", lw=0.6)
                for ep in self._epochs_here():
                    if ep.eye in ("both", ename):
                        ax.axvspan(ep.start_frame, ep.end_frame, color="#0072B2", alpha=0.22)
                ax.axvspan(slo, shi, color="#D55E00", alpha=0.18)
                ax.set_ylabel(f"{ename}\ndisplacement [{unit}]", fontsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xlim(vlo, vhi)
            axes.ravel()[-1].set_xlabel("Frame", fontsize=9)
            fig.suptitle(
                f"{spec.block_key} [{spec.mount_type}] — selection {slo}–{shi} "
                f"({shi - slo} frames)",
                fontsize=10,
            )
            fig.tight_layout()
            display(fig)
            plt.close(fig)

    def _refresh_epoch_list(self) -> None:
        options = []
        for i, ep in enumerate(self._epochs_here()):
            options.append(
                (f"{i}: {ep.eye} {ep.start_frame}–{ep.end_frame} ({ep.end_frame - ep.start_frame} fr)", i)
            )
        self.epoch_list.options = options

    def _add_epoch(self) -> None:
        lo, hi = self.sel.value
        if hi - lo < 2:
            self._set_status("Selection too short.", level="warn")
            return
        for ename in self._eyes():
            self._epochs_here().append(Epoch(start_frame=int(lo), end_frame=int(hi), eye=ename))
        self._refresh_epoch_list()
        self._redraw()
        self._set_status(f"Added epoch {lo}–{hi} for {', '.join(self._eyes())}.", level="ok")

    def _middle_epoch(self) -> None:
        self._reset_view()  # selection slider is clamped to the view window
        n = self.view.max + 1
        span = n // 2
        start = (n - span) // 2
        self.sel.value = (start, start + span)
        self._add_epoch()

    def _remove_epochs(self) -> None:
        keep = [ep for i, ep in enumerate(self._epochs_here()) if i not in set(self.epoch_list.value)]
        self._epochs[self.spec.block_key] = keep
        self._refresh_epoch_list()
        self._redraw()
        self._set_status("Removed selected epoch(s).", level="ok")

    def _clear_epochs(self) -> None:
        self._epochs[self.spec.block_key] = []
        self._refresh_epoch_list()
        self._redraw()
        self._set_status("Cleared epochs for this block.", level="ok")

    # ----------------------------------------------------------------- public
    def save_current(self) -> Path:
        spec = self.spec
        path = save_epochs(
            BlockEpochs(spec=spec, epochs=self._epochs_here(), amplitude_key=self.key),
            self.metadata_dir,
        )
        self._set_status(f"Saved {len(self._epochs_here())} epoch(s) → {path}", level="ok")
        return path

    def save_all(self) -> list[Path]:
        written = []
        for spec in self.specs:
            eps = self._epochs.get(spec.block_key)
            if not eps:
                continue
            written.append(
                save_epochs(
                    BlockEpochs(spec=spec, epochs=eps, amplitude_key=self.key), self.metadata_dir
                )
            )
        self._set_status(f"Saved epochs for {len(written)} block(s).", level="ok")
        return written


def registry_table(
    specs: list[JitterBlockSpec],
    metadata_dir: Path | str | None = None,
    *,
    key: str = "top_correlation_dist",
):
    """QC table: report status, frame counts, pixel calibration and epoch counts."""
    import pandas as pd

    rows = []
    for spec in specs:
        row: dict[str, object] = {
            "animal": spec.animal,
            "block": spec.block_path.name,
            "mount_type": spec.mount_type,
            "report": has_jitter_report(spec.block_path),
            "left_frames": None,
            "right_frames": None,
            "calibrated": False,
            "L_um_per_px": None,
            "R_um_per_px": None,
            "n_epochs": 0,
            "block_path": str(spec.block_path),
        }
        if row["report"]:
            try:
                traces = extract_amplitude_trace(
                    load_jitter_report(spec.report_path), eye="both", key=key
                )
                row["left_frames"] = int(len(traces.get("left_eye", [])))
                row["right_frames"] = int(len(traces.get("right_eye", [])))
            except (KeyError, ValueError) as exc:
                row["report"] = f"invalid: {exc}"
        try:
            ps = read_pixel_size(spec.block_path)
        except ValueError as exc:
            ps = None
            row["calibrated"] = f"invalid: {exc}"
        if ps is not None:
            row["calibrated"] = True
            row["L_um_per_px"] = round(ps.l_um_per_px, 3)
            row["R_um_per_px"] = round(ps.r_um_per_px, 3)
        if metadata_dir is not None:
            ep_path = epochs_yaml_path(Path(metadata_dir), spec)
            if ep_path.exists():
                row["n_epochs"] = len(load_epochs(ep_path, spec).epochs)
        rows.append(row)
    return pd.DataFrame(rows)


class PixelCalibrationPanel:
    """
    Review per-block ``analysis/LR_pix_size.csv`` and create the missing ones.

    **Calibrate selected** opens an OpenCV window per eye (drag the ROI diagonal
    across the known distance, Enter to accept) and writes the CSV that BlockSync
    and the jitter histograms both read. When the raw videos are offline, the
    manual row converts a measured landmark length in pixels instead.
    """

    def __init__(
        self,
        specs: list[JitterBlockSpec],
        *,
        known_dist_mm: float = DEFAULT_KNOWN_DIST_MM,
    ) -> None:
        if not specs:
            raise ValueError("No blocks given — save a registry first.")
        self.specs = list(specs)
        self._build(known_dist_mm)
        self.refresh()

    # ---------------------------------------------------------------- widgets
    def _build(self, known_dist_mm: float) -> None:
        self.table_html = widgets.HTML()
        self.block_list = widgets.SelectMultiple(options=[], rows=8,
                                                 layout=widgets.Layout(width="720px"))
        self.known_dist = widgets.FloatText(
            value=float(known_dist_mm), description="Known dist [mm]:", step=0.5,
            layout=widgets.Layout(width="230px"), style={"description_width": "125px"},
        )
        self.overwrite = widgets.Checkbox(value=False, description="overwrite existing",
                                          indent=False, layout=widgets.Layout(width="170px"))
        self.calib_btn = widgets.Button(
            description="Calibrate selected", icon="crop", button_style="primary",
            tooltip="Opens an OpenCV ROI window per eye (kernel must run on this machine)",
            layout=widgets.Layout(width="180px"),
        )
        self.show_missing = widgets.Button(description="Select uncalibrated",
                                           layout=widgets.Layout(width="170px"))
        self.left_px = widgets.FloatText(value=0.0, description="L px:",
                                         layout=widgets.Layout(width="150px"),
                                         style={"description_width": "45px"})
        self.right_px = widgets.FloatText(value=0.0, description="R px:",
                                          layout=widgets.Layout(width="150px"),
                                          style={"description_width": "45px"})
        self.manual_btn = widgets.Button(
            description="Manual entry", icon="keyboard-o",
            tooltip="Landmark length in pixels per eye, for blocks whose videos are offline",
            layout=widgets.Layout(width="150px"),
        )
        self.refresh_btn = widgets.Button(description="Refresh", icon="refresh",
                                          layout=widgets.Layout(width="110px"))
        self.status = widgets.HTML()

        self.calib_btn.on_click(lambda _: self._calibrate_selected())
        self.manual_btn.on_click(lambda _: self._manual_selected())
        self.show_missing.on_click(lambda _: self._select_missing())
        self.refresh_btn.on_click(lambda _: self.refresh())

        self.widget = widgets.VBox(
            [
                self.table_html,
                self.block_list,
                widgets.HBox([self.known_dist, self.overwrite, self.calib_btn,
                              self.show_missing, self.refresh_btn]),
                widgets.HTML("<i>No video? Enter the landmark length in pixels per eye:</i>"),
                widgets.HBox([self.left_px, self.right_px, self.manual_btn]),
                self.status,
            ]
        )

    def _ipython_display_(self) -> None:
        display(self.widget)

    # ---------------------------------------------------------------- helpers
    def _set_status(self, msg: str, *, level: str = "info") -> None:
        color = {"info": "#333", "ok": "#177245", "warn": "#b35c00", "err": "#a11"}[level]
        self.status.value = f"<span style='color:{color}'>{msg}</span>"

    def _selected_specs(self) -> list[JitterBlockSpec]:
        chosen = set(self.block_list.value)
        return [s for s in self.specs if str(s.block_path) in chosen]

    @property
    def missing(self) -> list[JitterBlockSpec]:
        return [s for s in self.specs if read_pixel_size(s.block_path) is None]

    def refresh(self) -> None:
        options = []
        n_ok = 0
        for spec in self.specs:
            try:
                ps = read_pixel_size(spec.block_path)
            except ValueError as exc:
                options.append((f"✗ {spec.block_key} — {exc}", str(spec.block_path)))
                continue
            if ps is None:
                vids = find_eye_videos(spec.block_path)
                have = "video ok" if all(vids.values()) else "no raw mp4"
                options.append((f"✗ {spec.block_key} — needs calibration ({have})", str(spec.block_path)))
            else:
                n_ok += 1
                options.append(
                    (
                        f"✓ {spec.block_key} — L {ps.l_um_per_px:.2f} / R {ps.r_um_per_px:.2f} µm/px",
                        str(spec.block_path),
                    )
                )
        self.block_list.options = options
        n_missing = len(self.specs) - n_ok
        self.table_html.value = (
            f"<b>Pixel calibration</b> — {n_ok}/{len(self.specs)} block(s) calibrated"
            + (f", <span style='color:#b35c00'>{n_missing} missing</span>" if n_missing else "")
        )

    def _select_missing(self) -> None:
        missing = {str(s.block_path) for s in self.missing}
        self.block_list.value = tuple(v for _, v in self.block_list.options if v in missing)
        self._set_status(f"Selected {len(self.block_list.value)} uncalibrated block(s).")

    def _calibrate_selected(self) -> None:
        specs = self._selected_specs()
        if not specs:
            self._set_status("Select block(s) to calibrate.", level="warn")
            return
        done, failed = 0, []
        for spec in specs:
            self._set_status(f"Calibrating {spec.block_key} — use the OpenCV window …")
            try:
                calibrate_block(
                    spec.block_path,
                    known_dist_mm=float(self.known_dist.value),
                    overwrite=bool(self.overwrite.value),
                )
                done += 1
            except Exception as exc:  # noqa: BLE001 - surfaced in the status line
                failed.append(f"{spec.block_key}: {exc}")
        self.refresh()
        msg = f"Calibrated {done} block(s)."
        if failed:
            msg += " Failed — " + " | ".join(failed)
        self._set_status(msg, level="ok" if done and not failed else "warn")

    def _manual_selected(self) -> None:
        specs = self._selected_specs()
        if not specs:
            self._set_status("Select block(s) for manual entry.", level="warn")
            return
        if self.left_px.value <= 0 or self.right_px.value <= 0:
            self._set_status("Enter positive L/R pixel distances.", level="warn")
            return
        for spec in specs:
            manual_calibration(
                spec.block_path,
                left_px=float(self.left_px.value),
                right_px=float(self.right_px.value),
                known_dist_mm=float(self.known_dist.value),
            )
        self.refresh()
        self._set_status(f"Wrote manual calibration for {len(specs)} block(s).", level="ok")


class JitterPoolSelector:
    """
    Checklist of registry blocks — each row shows that block's mean, 95th-percentile
    and raw maximum displacement — driving :func:`run_plot_pooled` over the ticked
    blocks only.

    Samples are loaded once on construction (reload with ``reload()`` after new
    epochs are saved), so ticking boxes and re-plotting is instant.
    """

    def __init__(
        self,
        specs: list[JitterBlockSpec],
        figures_dir: Path | str,
        metadata_dir: Path | str,
        *,
        units: str = "um",
        n_bins: int = 15,
        key: str = "top_correlation_dist",
    ) -> None:
        if not specs:
            raise ValueError("No blocks given — save a registry first.")
        self.specs = list(specs)
        self.figures_dir = Path(figures_dir)
        self.metadata_dir = Path(metadata_dir)
        self.units = units
        self.key = key
        self.block_samples: list = []
        self.written: dict[str, Path] = {}
        self._build(n_bins)
        self.reload()

    # ---------------------------------------------------------------- widgets
    def _build(self, n_bins: int) -> None:
        self.header = widgets.HTML()
        self.checks_box = widgets.VBox(layout=widgets.Layout(
            max_height="320px", overflow_y="auto", border="1px solid #ddd", padding="4px"
        ))
        self.checks: dict[str, widgets.Checkbox] = {}
        self.sort_by = widgets.Dropdown(
            options=[("registry order", "registry"), ("mean", "mean"), ("p95", "p95"),
                     ("max", "max"), ("mount type", "mount"), ("n samples", "n")],
            value="registry", description="Sort:",
            layout=widgets.Layout(width="230px"), style={"description_width": "40px"},
        )
        self.n_bins = widgets.IntText(value=int(n_bins), description="Bins:",
                                      layout=widgets.Layout(width="130px"),
                                      style={"description_width": "40px"})
        self.units_dd = widgets.Dropdown(
            options=[("µm", "um"), ("px", "px")], value=self.units, description="Units:",
            layout=widgets.Layout(width="150px"), style={"description_width": "45px"},
        )
        self.all_btn = widgets.Button(description="All", layout=widgets.Layout(width="70px"))
        self.none_btn = widgets.Button(description="None", layout=widgets.Layout(width="70px"))
        self.reload_btn = widgets.Button(description="Reload", icon="refresh",
                                         tooltip="Re-read epoch YAMLs and calibrations",
                                         layout=widgets.Layout(width="110px"))
        self.plot_btn = widgets.Button(description="Pool & plot selected", icon="bar-chart",
                                       button_style="primary",
                                       layout=widgets.Layout(width="200px"))
        self.status = widgets.HTML()
        self.out = widgets.Output()

        self.all_btn.on_click(lambda _: self._set_all(True))
        self.none_btn.on_click(lambda _: self._set_all(False))
        self.reload_btn.on_click(lambda _: self.reload())
        self.plot_btn.on_click(lambda _: self.run())
        self.sort_by.observe(lambda ch: self._render_checks(), names="value")
        self.units_dd.observe(lambda ch: self.reload(), names="value")

        self.widget = widgets.VBox([
            self.header,
            self.checks_box,
            widgets.HBox([self.all_btn, self.none_btn, self.sort_by, self.reload_btn]),
            widgets.HBox([self.units_dd, self.n_bins, self.plot_btn]),
            self.status,
            self.out,
        ])

    def _ipython_display_(self) -> None:
        display(self.widget)

    # ---------------------------------------------------------------- helpers
    def _set_status(self, msg: str, *, level: str = "info") -> None:
        color = {"info": "#333", "ok": "#177245", "warn": "#b35c00", "err": "#a11"}[level]
        self.status.value = f"<span style='color:{color}'>{msg}</span>"

    def reload(self) -> None:
        """Re-read epoch YAMLs and calibrations, keeping the current ticks."""
        previous = {k: c.value for k, c in self.checks.items()}
        self.units = str(self.units_dd.value)
        with self.out:
            self.out.clear_output(wait=True)
            self.block_samples = collect_block_samples(
                self.specs, self.metadata_dir, key=self.key, units=self.units, verbose=True
            )
        self.checks = {}
        for bs in self.block_samples:
            box = widgets.Checkbox(
                value=previous.get(bs.block_key, bool(bs.values.size)),
                description=bs.label(),
                indent=False,
                disabled=not bs.values.size,
                layout=widgets.Layout(width="98%"),
                style={"description_width": "0px"},
            )
            box.observe(lambda _: self._update_header(), names="value")
            self.checks[bs.block_key] = box
        self._render_checks()

    def _sorted_samples(self) -> list:
        """Descending on the chosen statistic; blocks without samples sink."""
        mode = self.sort_by.value
        metrics = {
            "mean": np.mean,
            "p95": lambda a: np.percentile(a, 95),
            "max": np.max,
            "n": lambda a: a.size,
        }
        if mode in metrics:
            fn = metrics[mode]
            return sorted(
                self.block_samples,
                key=lambda bs: (float(fn(bs.values)) if bs.values.size else -np.inf),
                reverse=True,
            )
        if mode == "mount":
            return sorted(self.block_samples, key=lambda bs: (bs.mount_type, bs.block_key))
        return list(self.block_samples)

    def _render_checks(self) -> None:
        self.checks_box.children = tuple(
            self.checks[bs.block_key] for bs in self._sorted_samples()
        )
        self._update_header()

    def _update_header(self) -> None:
        usable = sum(1 for bs in self.block_samples if bs.values.size)
        skipped = len(self.block_samples) - usable
        self.header.value = (
            f"<b>Blocks to pool</b> — {len(self.selected)}/{usable} ticked"
            + (f", {skipped} without epochs (greyed out)" if skipped else "")
            + f". Displacement mean / p95 / max in {'µm' if self.units == 'um' else 'px'}."
        )

    @property
    def selected(self) -> list[str]:
        """Ticked ``block_key``s."""
        return [k for k, c in self.checks.items() if c.value and not c.disabled]

    def _set_all(self, value: bool) -> None:
        for c in self.checks.values():
            if not c.disabled:
                c.value = value
        self._update_header()

    # ------------------------------------------------------------------- run
    def run(self, *, show: bool = True) -> dict[str, Path]:
        """Pool the ticked blocks and write/show the histograms."""
        chosen = self.selected
        if not chosen:
            self._set_status("Tick at least one block.", level="warn")
            return {}
        self._set_status(f"Pooling {len(chosen)} block(s) …")
        with self.out:
            self.out.clear_output(wait=True)
            self.written = run_plot_pooled(
                self.specs,
                self.figures_dir,
                self.metadata_dir,
                units=self.units,
                n_bins=int(self.n_bins.value),
                show=show,
                include=chosen,
                block_samples=self.block_samples,
            )
            for name, path in self.written.items():
                print(f"{name}: {path}")
        self._set_status(f"Pooled {len(chosen)} block(s).", level="ok")
        return self.written
