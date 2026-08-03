"""
Notebook widgets for eye-movement span tool 1.

* Collect blocks with :class:`~eye_tracking_system_tools.analysis.jitter_gui.JitterBlockBrowser`
  (``require="eye"``).
* :class:`SpanCharacteristicsPanel` — compute full-range + p5–p95 spans in
  degrees and pixels for every staged block/eye and write a CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from eye_tracking_system_tools.analysis.eye_movement_span import (
    DISPLAY_COLS,
    collect_span_characteristics,
    format_characteristics_report,
)


class SpanCharacteristicsPanel:
    """Compute and display full / p95 eye-movement spans (deg + px) for many blocks."""

    def __init__(
        self,
        specs: list[Any],
        metadata_dir: Path | str,
        *,
        lo: float = 5.0,
        hi: float = 95.0,
    ) -> None:
        if not specs:
            raise ValueError("No blocks given — stage and save a registry first.")
        self.specs = list(specs)
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.lo = float(lo)
        self.hi = float(hi)
        self.table: pd.DataFrame = pd.DataFrame()
        self.out_path: Path | None = None
        self._build()

    def _build(self) -> None:
        self.lo_box = widgets.FloatText(
            value=self.lo, description="lo %:",
            layout=widgets.Layout(width="140px"), style={"description_width": "45px"},
        )
        self.hi_box = widgets.FloatText(
            value=self.hi, description="hi %:",
            layout=widgets.Layout(width="140px"), style={"description_width": "45px"},
        )
        self.run_btn = widgets.Button(
            description="Compute spans", icon="calculator", button_style="primary",
            layout=widgets.Layout(width="160px"),
        )
        self.status = widgets.HTML("")
        self.out = widgets.Output()
        self.run_btn.on_click(lambda _: self.compute())
        self.widget = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Span characteristics</b> — full range (max−min) and "
                    f"percentile clip (default p{self.lo:g}–p{self.hi:g}) in "
                    "<b>degrees</b> and <b>pixels</b>."
                ),
                widgets.HBox([self.lo_box, self.hi_box, self.run_btn]),
                self.status,
                self.out,
            ]
        )

    def _ipython_display_(self) -> None:
        display(self.widget)

    def _set_status(self, msg: str, *, level: str = "info") -> None:
        color = {"info": "#333", "ok": "#177245", "warn": "#b35c00", "err": "#a11"}[level]
        self.status.value = f"<span style='color:{color}'>{msg}</span>"

    def compute(self) -> pd.DataFrame:
        self.lo = float(self.lo_box.value)
        self.hi = float(self.hi_box.value)
        self._set_status(f"Computing spans for {len(self.specs)} block(s) …")
        self.out.clear_output(wait=True)
        try:
            self.table = collect_span_characteristics(
                self.specs, lo=self.lo, hi=self.hi
            )
            self.out_path = self.metadata_dir / "eye_span_characteristics.csv"
            self.table.to_csv(self.out_path, index=False)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed: {exc}", level="err")
            with self.out:
                print(exc)
            raise

        n_ok = int(self.table["ok"].sum()) if not self.table.empty else 0
        n_blocks = int(self.table["block_key"].nunique()) if not self.table.empty else 0
        self._set_status(
            f"{n_ok}/{len(self.table)} eyes OK across {n_blocks} blocks → {self.out_path}",
            level="ok" if n_ok else "warn",
        )
        with self.out:
            print(format_characteristics_report(self.table))
            cols = [c for c in DISPLAY_COLS if c in self.table.columns]
            display(self.table[cols])
        return self.table
