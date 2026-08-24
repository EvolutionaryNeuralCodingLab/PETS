"""ipywidgets panel for threshold-based event subset selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from eye_tracking_system_tools.analysis.pipeline import (
    EventTables,
    SaccadeFilter,
    apply_saccade_filter,
    filter_event_tables,
)
from eye_tracking_system_tools.analysis.saccade_viewer.launch import launch_saccade_viewer
from eye_tracking_system_tools.analysis.saccade_viewer.selectors.common import (
    enrich_events_for_viewer,
    numeric_event_columns,
)


@dataclass
class ThresholdRule:
    column: str
    min_val: float | None = None
    max_val: float | None = None

    def describe(self) -> str:
        parts = []
        if self.min_val is not None:
            parts.append(f">= {self.min_val:g}")
        if self.max_val is not None:
            parts.append(f"<= {self.max_val:g}")
        return f"{self.column}: " + (" & ".join(parts) if parts else "any")


def apply_threshold_rules(df: pd.DataFrame, rules: list[ThresholdRule]) -> pd.DataFrame:
    if df is None or df.empty or not rules:
        return df.copy() if df is not None else pd.DataFrame()
    out = df.copy()
    for rule in rules:
        if rule.column not in out.columns:
            raise KeyError(f"Column not in events table: {rule.column!r}")
        col = pd.to_numeric(out[rule.column], errors="coerce")
        mask = col.notna()
        if rule.min_val is not None:
            mask &= col >= float(rule.min_val)
        if rule.max_val is not None:
            mask &= col <= float(rule.max_val)
        out = out.loc[mask]
    return out.reset_index(drop=True)


class ThresholdSelectorPanel:
    """
    Notebook widget: filter ``ctx.tables`` by block, event kind, and numeric thresholds.

    ``panel.selected_events`` holds the latest filtered table (viewer-ready).
    """

    def __init__(
        self,
        tables: EventTables,
        *,
        registry_path: Path | str | None = None,
        default_block_keys: list[str] | None = None,
        on_change: Callable[[pd.DataFrame], None] | None = None,
    ) -> None:
        self.tables = tables
        self.registry_path = Path(registry_path) if registry_path else None
        self.on_change = on_change
        self.selected_events = pd.DataFrame()
        self._rules: list[ThresholdRule] = []

        all_keys = sorted(tables.block_dict.keys())
        default_block_keys = default_block_keys or all_keys
        self._block_checks = {
            k: widgets.Checkbox(value=k in default_block_keys, description=k, indent=False)
            for k in all_keys
        }
        self.event_kind = widgets.Dropdown(
            options=[
                ("All events", "all"),
                ("Monocular only", "monocular"),
                ("Concurrent only", "concurrent"),
            ],
            value="all",
            description="Kind:",
        )
        self.head_movement = widgets.Dropdown(
            options=[
                ("Any", "any"),
                ("Head-stationary", "without"),
                ("Head-coupled", "with"),
            ],
            value="any",
            description="Head:",
        )
        numeric = numeric_event_columns(tables.all_saccades)
        self.column_pick = widgets.Dropdown(
            options=numeric or ["net_angular_disp"],
            description="Column:",
        )
        self.min_box = widgets.FloatText(value=0.0, description="Min:", layout=widgets.Layout(width="160px"))
        self.max_box = widgets.FloatText(value=0.0, description="Max:", layout=widgets.Layout(width="160px"))
        self.use_min = widgets.Checkbox(value=False, description="Use min")
        self.use_max = widgets.Checkbox(value=False, description="Use max")
        self.btn_add_rule = widgets.Button(description="Add threshold", icon="plus")
        self.btn_clear_rules = widgets.Button(description="Clear thresholds", icon="trash")
        self.rules_html = widgets.HTML("No thresholds yet.")
        self.btn_preview = widgets.Button(description="Preview subset", button_style="info", icon="filter")
        self.btn_launch = widgets.Button(description="Launch viewer", button_style="primary", icon="eye")
        self.status = widgets.HTML("")
        self.out = widgets.Output()

        self.btn_add_rule.on_click(lambda _: self._add_rule())
        self.btn_clear_rules.on_click(lambda _: self._clear_rules())
        self.btn_preview.on_click(lambda _: self._preview())
        self.btn_launch.on_click(lambda _: self._launch())

        block_box = widgets.VBox(
            [widgets.HTML("<b>Blocks</b>")] + list(self._block_checks.values())
        )
        rule_row = widgets.HBox(
            [
                self.column_pick,
                self.use_min,
                self.min_box,
                self.use_max,
                self.max_box,
                self.btn_add_rule,
                self.btn_clear_rules,
            ]
        )
        self.widget = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Threshold selector</b> — build a verification subset from scalar event columns."
                ),
                widgets.HBox([block_box, widgets.VBox([self.event_kind, self.head_movement])]),
                rule_row,
                self.rules_html,
                widgets.HBox([self.btn_preview, self.btn_launch]),
                self.status,
                self.out,
            ]
        )

    def _ipython_display_(self) -> None:
        display(self.widget)

    def _selected_block_keys(self) -> list[str]:
        return [k for k, w in self._block_checks.items() if w.value]

    def _add_rule(self) -> None:
        col = str(self.column_pick.value)
        rule = ThresholdRule(
            column=col,
            min_val=float(self.min_box.value) if self.use_min.value else None,
            max_val=float(self.max_box.value) if self.use_max.value else None,
        )
        self._rules.append(rule)
        self._refresh_rules_html()

    def _clear_rules(self) -> None:
        self._rules.clear()
        self._refresh_rules_html()

    def _refresh_rules_html(self) -> None:
        if not self._rules:
            self.rules_html.value = "<i>No thresholds yet.</i>"
            return
        items = "".join(f"<li>{r.describe()}</li>" for r in self._rules)
        self.rules_html.value = f"<ul>{items}</ul>"

    def _build_subset(self) -> pd.DataFrame:
        keys = self._selected_block_keys()
        if not keys:
            raise ValueError("Select at least one block.")

        work = filter_event_tables(self.tables, block_keys=keys)
        filt = SaccadeFilter(
            event_kind=str(self.event_kind.value),
            head_movement=None if self.head_movement.value == "any" else str(self.head_movement.value),
        )
        work = apply_saccade_filter(work, filt)
        subset = apply_threshold_rules(work.all_saccades, self._rules)
        return enrich_events_for_viewer(work, subset)

    def _preview(self) -> None:
        with self.out:
            self.out.clear_output(wait=True)
            try:
                subset = self._build_subset()
            except Exception as exc:  # noqa: BLE001
                self.status.value = f"<span style='color:#b03030'>{exc}</span>"
                return
            self.selected_events = subset
            self.status.value = f"<b>{len(subset)}</b> events selected."
            if self.on_change:
                self.on_change(subset)
            display(subset.head(20))
            if len(subset) > 20:
                print(f"... ({len(subset) - 20} more rows)")

    def _launch(self) -> None:
        if self.selected_events.empty:
            self._preview()
        if self.selected_events.empty:
            self.status.value = "<span style='color:#b03030'>No events to verify.</span>"
            return
        launch_saccade_viewer(
            self.selected_events,
            registry_path=self.registry_path,
            block=True,
        )
