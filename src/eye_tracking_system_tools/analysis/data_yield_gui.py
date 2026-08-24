"""
Notebook (ipywidgets) front-ends for the data yield report.

* :func:`yield_registry_table` — QC from a loaded :class:`YieldDataset` (or presence-only).
* :class:`YieldPoolSelector` — tick blocks, then run pooled figures from RAM cache.
* :func:`load_yield_dataset` — one network pull into memory; reuse everywhere after.
"""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from eye_tracking_system_tools.analysis.data_yield import (
    DEFAULT_LIKELIHOOD_THR,
    MANUAL_ANNOTATIONS_NAME,
    YieldDataset,
    finalize_yield_export,
    load_yield_dataset,
    resolve_block_dlc_csvs,
    resolve_yield_eye_csv,
    run_yield_report,
    seed_yield_registry_from_jitter,
    yield_registry_table_from_dataset,
)
from eye_tracking_system_tools.analysis.jitter_epochs import (
    JitterBlockSpec,
    read_registry_blocks,
)
from eye_tracking_system_tools.preprocessing.noise_epochs import noise_epochs_path


def yield_registry_table(
    specs: list[JitterBlockSpec],
    *,
    dataset: YieldDataset | None = None,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THR,
) -> pd.DataFrame:
    """
    QC table for yield prerequisites.

    Prefer passing ``dataset=DATA`` from :func:`load_yield_dataset` — that path is
    RAM-only and includes post-mask yield %. Without a dataset, only cheap path /
    presence checks run (no eye/DLC CSV reads).
    """
    if dataset is not None:
        return yield_registry_table_from_dataset(
            dataset, likelihood_threshold=likelihood_threshold
        )

    rows: list[dict[str, object]] = []
    for spec in specs:
        analysis = Path(spec.block_path) / "analysis"
        dlc = resolve_block_dlc_csvs(spec.block_path)
        row: dict[str, object] = {
            "animal": spec.animal,
            "block": Path(spec.block_path).name,
            "mount_type": spec.mount_type,
            "dlc_left": dlc["left"].name if dlc["left"] is not None else None,
            "dlc_right": dlc["right"].name if dlc["right"] is not None else None,
            "eye_left": None,
            "eye_left_rule": None,
            "eye_right": None,
            "eye_right_rule": None,
            "noise_left": noise_epochs_path(spec.block_path, "left").is_file(),
            "noise_right": noise_epochs_path(spec.block_path, "right").is_file(),
            "manual_annotations": (analysis / MANUAL_ANNOTATIONS_NAME).is_file(),
            "yield_left_pct": None,
            "yield_right_pct": None,
            "block_path": str(spec.block_path),
            "yield_note": "load DATA first for yield %",
        }
        for eye, key_csv, key_rule in (
            ("left", "eye_left", "eye_left_rule"),
            ("right", "eye_right", "eye_right_rule"),
        ):
            try:
                choice = resolve_yield_eye_csv(analysis, eye)
                row[key_csv] = choice.path.name
                row[key_rule] = choice.rule
            except FileNotFoundError:
                continue
        rows.append(row)
    return pd.DataFrame(rows)


class YieldPoolSelector:
    """Checklist of registered blocks; pools DLC + ellipse yield figures."""

    def __init__(
        self,
        specs: list[JitterBlockSpec],
        figures_dir: Path | str,
        metadata_dir: Path | str,
        *,
        dataset: YieldDataset | None = None,
        likelihood_threshold: float = DEFAULT_LIKELIHOOD_THR,
        n_bins: int = 50,
    ) -> None:
        if not specs:
            raise ValueError("No blocks given — stage and save a registry first.")
        self.specs = list(specs)
        self.dataset = dataset
        self.figures_dir = Path(figures_dir)
        self.metadata_dir = Path(metadata_dir)
        self.likelihood_threshold = float(likelihood_threshold)
        self.written: dict[str, Path] = {}
        self._build(n_bins)
        self._render_checks()

    def _build(self, n_bins: int) -> None:
        self.header = widgets.HTML()
        self.checks: dict[str, widgets.Checkbox] = {}
        for spec in self.specs:
            box = widgets.Checkbox(
                value=True,
                description=f"[{spec.mount_type}] {spec.block_key}",
                indent=False,
                layout=widgets.Layout(width="98%"),
                style={"description_width": "0px"},
            )
            box.observe(lambda _: self._update_header(), names="value")
            self.checks[spec.block_key] = box
        self.checks_box = widgets.VBox()
        self.n_bins = widgets.IntText(
            value=int(n_bins),
            description="DLC bins:",
            layout=widgets.Layout(width="160px"),
            style={"description_width": "70px"},
        )
        self.thr = widgets.FloatText(
            value=self.likelihood_threshold,
            description="Lik. thr:",
            step=0.01,
            layout=widgets.Layout(width="160px"),
            style={"description_width": "70px"},
        )
        self.all_btn = widgets.Button(description="All", layout=widgets.Layout(width="70px"))
        self.none_btn = widgets.Button(description="None", layout=widgets.Layout(width="70px"))
        self.run_btn = widgets.Button(
            description="Pool + plot",
            icon="play",
            button_style="success",
            layout=widgets.Layout(width="140px"),
        )
        self.status = widgets.HTML()
        self.out = widgets.Output()

        self.all_btn.on_click(lambda _: self._set_all(True))
        self.none_btn.on_click(lambda _: self._set_all(False))
        self.run_btn.on_click(lambda _: self.run(show=True))

        self.ui = widgets.VBox(
            [
                self.header,
                self.checks_box,
                widgets.HBox([self.n_bins, self.thr, self.all_btn, self.none_btn, self.run_btn]),
                self.status,
                self.out,
            ]
        )

    def _render_checks(self) -> None:
        order = sorted(self.specs, key=lambda s: (s.mount_type, s.block_key))
        self.checks_box.children = tuple(self.checks[s.block_key] for s in order)
        self._update_header()

    def _update_header(self) -> None:
        cache = "RAM dataset" if self.dataset is not None else "disk (slow)"
        self.header.value = (
            f"<b>Blocks to pool</b> — {len(self.selected)}/{len(self.specs)} ticked "
            f"[{cache}]."
        )

    @property
    def selected(self) -> list[str]:
        return [k for k, c in self.checks.items() if c.value]

    def _set_all(self, value: bool) -> None:
        for c in self.checks.values():
            c.value = value
        self._update_header()

    def _set_status(self, msg: str, *, level: str = "info") -> None:
        color = {"info": "#333", "ok": "#177245", "warn": "#b35c00", "err": "#a11"}[level]
        self.status.value = f"<span style='color:{color}'>{msg}</span>"

    def run(self, *, show: bool = True) -> dict[str, Path]:
        chosen = self.selected
        if not chosen:
            self._set_status("Tick at least one block.", level="warn")
            return {}
        if self.dataset is None:
            self._set_status(
                "No DATA cache — reading from disk/network (slow). "
                "Run load_yield_dataset first.",
                level="warn",
            )
        self.likelihood_threshold = float(self.thr.value)
        self._set_status(f"Pooling {len(chosen)} block(s) …")
        with self.out:
            self.out.clear_output(wait=True)
            self.written = run_yield_report(
                self.specs,
                self.figures_dir,
                self.metadata_dir,
                include=chosen,
                likelihood_threshold=self.likelihood_threshold,
                n_bins=int(self.n_bins.value),
                show=show,
                verbose=True,
                dataset=self.dataset,
            )
            for name, path in self.written.items():
                print(f"{name}: {path}")
        self._set_status(f"Pooled {len(chosen)} block(s).", level="ok")
        return self.written

    def _ipython_display_(self) -> None:
        display(self.ui)


def load_yield_specs(registry_path: Path | str) -> list[JitterBlockSpec]:
    """Load yield registry blocks (same schema as jitter)."""
    return read_registry_blocks(registry_path)


__all__ = [
    "YieldPoolSelector",
    "finalize_yield_export",
    "load_yield_dataset",
    "load_yield_specs",
    "seed_yield_registry_from_jitter",
    "yield_registry_table",
]
