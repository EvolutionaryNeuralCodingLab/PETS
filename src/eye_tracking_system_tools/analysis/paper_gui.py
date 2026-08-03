"""
Notebook (ipywidgets) front-ends for the flexible paper-figures tool.

* :func:`block_qc_table` — per-block overview DataFrame for the QC cell.
* :class:`PaperFigureSelector` — per-animal grouped checkboxes with eligibility
  greying, a flexible saccade filter (kind / head movement / query / column
  truth values), a params-override box, and a Build button that runs the
  catalogued exporter for one figure.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import ipywidgets as widgets
import pandas as pd
import yaml
from IPython.display import display

from eye_tracking_system_tools.analysis.behavior_state import has_behavior_state
from eye_tracking_system_tools.analysis.event_cache import ensure_traces_for_blocks
from eye_tracking_system_tools.analysis.figure_catalog import CATALOG, get_spec, run_figure
from eye_tracking_system_tools.analysis.paper_export import FigureBuildResult
from eye_tracking_system_tools.analysis.pipeline import (
    EventTables,
    SaccadeFilter,
    apply_saccade_filter,
    boolish_event_columns,
    filter_event_tables,
)
from eye_tracking_system_tools.analysis.pixel_calibration import read_pixel_size


def block_qc_table(tables: EventTables) -> pd.DataFrame:
    """Per-block inventory: saccade counts, synced pairs, eligibility flags."""
    rows = []
    for b in tables.blocks:
        spec = b.spec
        n_l = int(len(b.l_saccades)) if b.l_saccades is not None else 0
        n_r = int(len(b.r_saccades)) if b.r_saccades is not None else 0
        if b.all_saccades is not None and not b.all_saccades.empty and "Main" in getattr(
            tables.synced, "columns", []
        ):
            # Count synced pairs belonging to this block.
            syn = tables.synced
            if not syn.empty and "animal" in syn.columns and "block" in syn.columns:
                mask = (syn["animal"].astype(str) == spec.animal) & (
                    syn["block"].astype(str).map(
                        lambda x: "".join(c for c in str(x) if c.isdigit()).zfill(3)
                    )
                    == spec.block_num
                )
                n_synced = int(syn.loc[mask, "Main"].nunique()) if mask.any() else 0
            else:
                n_synced = 0
        else:
            n_synced = 0
        has_traces = (
            b.left is not None
            and not getattr(b.left, "empty", True)
            and b.right is not None
            and not getattr(b.right, "empty", True)
        )
        rows.append(
            {
                "block_key": spec.block_key,
                "animal": spec.animal,
                "block": spec.block_path.name,
                "n_saccades_L": n_l,
                "n_saccades_R": n_r,
                "n_synced_pairs": n_synced,
                "has_traces": has_traces,
                "has_behavior_state": has_behavior_state(spec),
                "has_pix_size": read_pixel_size(spec.block_path) is not None,
                "block_path": str(spec.block_path),
            }
        )
    if not rows and not tables.all_saccades.empty:
        # Event-only mode: summarize from all_saccades columns.
        for (animal, block), g in tables.all_saccades.groupby(["animal", "block"]):
            digits = "".join(c for c in str(block) if c.isdigit()).zfill(3)
            key = f"{animal}_block_{digits}"
            n_l = int((g["eye"] == "L").sum()) if "eye" in g.columns else 0
            n_r = int((g["eye"] == "R").sum()) if "eye" in g.columns else 0
            rows.append(
                {
                    "block_key": key,
                    "animal": str(animal),
                    "block": f"block_{digits}",
                    "n_saccades_L": n_l,
                    "n_saccades_R": n_r,
                    "n_synced_pairs": 0,
                    "has_traces": False,
                    "has_behavior_state": False,
                    "has_pix_size": False,
                    "block_path": "",
                }
            )
    return pd.DataFrame(rows)


def _eligibility(tables: EventTables, fig_id: str, block_key: str) -> tuple[bool, str]:
    """Return (ok, reason) for whether ``block_key`` can contribute to ``fig_id``."""
    spec = get_spec(fig_id)
    needs = set(spec.needs)
    if "jitter_bundle" in needs:
        return True, "uses jitter export (block selection N/A)"
    bundle = tables.block_dict.get(block_key)
    if bundle is None:
        if "events" in needs and not tables.all_saccades.empty:
            return True, ""
        return False, "not in event tables"
    if "events" in needs:
        n = len(bundle.all_saccades) if bundle.all_saccades is not None else 0
        if n == 0:
            return False, "no saccades"
    if "behavior_state" in needs and not has_behavior_state(bundle.spec):
        return False, "no behavior_state.csv"
    if "pix_size" in needs and read_pixel_size(bundle.spec.block_path) is None:
        return False, "no LR_pix_size.csv"
    # traces: missing traces are OK — we can reload them on Build
    if "traces" in needs:
        pass
    return True, ""


def _seed_params_yaml(tables: EventTables, fig_id: str) -> str:
    section = get_spec(fig_id).params_section
    if not section:
        return "# (no params section for this figure)\n"
    body = deepcopy(tables.params.get(section, {})) or {}
    return yaml.safe_dump({section: body}, sort_keys=False, default_flow_style=False)


_TRUTH_OPTIONS = (
    ("Any", "any"),
    ("True", "true"),
    ("False", "false"),
    ("NaN", "na"),
)


class PaperContext:
    """Shared notebook state: tables, run dirs, and per-figure build results."""

    def __init__(
        self,
        tables: EventTables,
        out_dir: Path | str,
        *,
        registry_path: Path | str | None = None,
        params_path: Path | str | None = None,
        on_build: Callable[[FigureBuildResult], None] | None = None,
    ) -> None:
        self.tables = tables
        self.out_dir = Path(out_dir)
        self.registry_path = Path(registry_path) if registry_path else None
        self.params_path = Path(params_path) if params_path else None
        self.builds: dict[str, FigureBuildResult] = {}
        self.on_build = on_build
        self.qc = block_qc_table(tables)

    def record(self, result: FigureBuildResult) -> None:
        self.builds[result.fig_id] = result
        if self.on_build is not None:
            self.on_build(result)


class PaperFigureSelector:
    """
    Per-figure block checklist + saccade filter + params override + Build.

    ``selector = PaperFigureSelector("2g", ctx); selector`` renders the UI.
    After Build, ``selector.result`` / ``ctx.builds[fig_id]`` hold the outputs.

    The saccade filter section supports:

    * event kind — all / concurrent (synced) / monocular
    * head movement — any / without / with / labeled-only
    * optional pandas ``query`` string
    * per-column truth filters for other boolean-like flags
    """

    def __init__(
        self,
        fig_id: str,
        ctx: PaperContext,
        *,
        single_block: bool = False,
        extra_widgets: list | None = None,
        default_runner_kwargs: dict[str, Any] | None = None,
        default_saccade_filter: SaccadeFilter | dict[str, Any] | None = None,
        enable_saccade_filter: bool = True,
    ) -> None:
        self.fig_id = fig_id
        self.spec = get_spec(fig_id)
        self.ctx = ctx
        self.single_block = single_block or fig_id in {"3a", "3b", "3c"}
        self.default_runner_kwargs = dict(default_runner_kwargs or {})
        self.enable_saccade_filter = bool(enable_saccade_filter)
        if "jitter_bundle" in self.spec.needs:
            self.enable_saccade_filter = False
        self._default_filter = SaccadeFilter.from_mapping(default_saccade_filter)
        self.result: FigureBuildResult | None = None
        self._extra = list(extra_widgets or [])
        self.column_truth: dict[str, widgets.Dropdown] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        self.header = widgets.HTML(
            f"<b>{self.spec.label}</b> "
            f"<span style='color:#666'>(needs: {', '.join(self.spec.needs) or '—'})</span>"
        )
        self.checks_box = widgets.VBox(
            layout=widgets.Layout(
                max_height="280px", overflow_y="auto", border="1px solid #ddd", padding="4px"
            )
        )
        self.animal_toggles: dict[str, widgets.Checkbox] = {}
        self.checks: dict[str, widgets.Checkbox] = {}
        self._populate_checks()

        self.params_box = widgets.Textarea(
            value=_seed_params_yaml(self.ctx.tables, self.fig_id),
            description="Params:",
            layout=widgets.Layout(width="98%", height="120px"),
            style={"description_width": "60px"},
        )
        self.all_btn = widgets.Button(description="All", layout=widgets.Layout(width="70px"))
        self.none_btn = widgets.Button(description="None", layout=widgets.Layout(width="70px"))
        self.build_btn = widgets.Button(
            description="Build",
            icon="play",
            button_style="primary",
            layout=widgets.Layout(width="120px"),
        )
        self.status = widgets.HTML()
        self.out = widgets.Output()

        self.all_btn.on_click(lambda _: self._set_all(True))
        self.none_btn.on_click(lambda _: self._set_all(False))
        self.build_btn.on_click(lambda _: self.build())

        children: list = [
            self.header,
            self.checks_box,
            widgets.HBox([self.all_btn, self.none_btn, self.build_btn]),
        ]
        children.extend(self._extra)
        if self.enable_saccade_filter:
            children.append(self._build_filter_section())
        children.extend([self.params_box, self.status, self.out])
        self.widget = widgets.VBox(children)
        if self.enable_saccade_filter:
            self._refresh_filter_preview()

    def _seed_filter_widgets(self) -> dict[str, Any]:
        seed = self._default_filter or SaccadeFilter()
        kind = seed.normalized_kind()
        hm = seed.normalized_head()
        if hm is True:
            head_val = "with"
        elif hm is False:
            head_val = "without"
        elif hm == "labeled":
            head_val = "labeled"
        else:
            head_val = "any"
        return {
            "event_kind": kind,
            "head_movement": head_val,
            "query": (seed.query or "") if seed.query else "",
            "column_equals": dict(seed.column_equals or {}),
        }

    def _build_filter_section(self) -> widgets.Widget:
        seed = self._seed_filter_widgets()
        self.event_kind = widgets.ToggleButtons(
            options=[
                ("All", "all"),
                ("Concurrent", "concurrent"),
                ("Monocular", "monocular"),
            ],
            value=seed["event_kind"],
            description="Events:",
            style={"description_width": "70px", "button_width": "110px"},
            layout=widgets.Layout(width="auto"),
        )
        self.head_movement = widgets.ToggleButtons(
            options=[
                ("Any", "any"),
                ("Without head", "without"),
                ("With head", "with"),
                ("Labeled only", "labeled"),
            ],
            value=seed["head_movement"],
            description="Head:",
            style={"description_width": "70px", "button_width": "110px"},
            layout=widgets.Layout(width="auto"),
        )
        self.query_box = widgets.Textarea(
            value=seed["query"],
            description="Query:",
            placeholder='optional pandas query, e.g. head_movement==False and animal!="PV_57"',
            layout=widgets.Layout(width="98%", height="54px"),
            style={"description_width": "70px"},
        )
        self.filter_preview = widgets.HTML()
        self.filter_reset = widgets.Button(
            description="Reset filter",
            layout=widgets.Layout(width="120px"),
        )
        self.filter_reset.on_click(lambda _: self._reset_filter())

        # Extra boolean-like columns (head_movement already has a preset).
        self.column_truth = {}
        truth_rows = []
        cols = boolish_event_columns(
            self.ctx.tables.all_saccades, exclude=("head_movement",)
        )
        seeded_eq = seed["column_equals"]
        for col in cols:
            raw = seeded_eq.get(col, "any")
            if raw is True:
                val = "true"
            elif raw is False:
                val = "false"
            elif raw is None:
                val = "na"
            else:
                val = "any"
            dd = widgets.Dropdown(
                options=list(_TRUTH_OPTIONS),
                value=val,
                description=f"{col}:",
                layout=widgets.Layout(width="220px"),
                style={"description_width": "110px"},
            )
            self.column_truth[col] = dd
            truth_rows.append(dd)

        advanced_children: list = [
            widgets.HTML(
                "<span style='color:#666;font-size:12px'>"
                "Pandas <code>DataFrame.query</code> over the selected events "
                "(applied after the toggles / column truth filters)."
                "</span>"
            ),
            self.query_box,
        ]
        if truth_rows:
            advanced_children.insert(
                0,
                widgets.HTML("<b>Column truth values</b>"),
            )
            advanced_children.insert(1, widgets.HBox(truth_rows))

        advanced = widgets.Accordion(
            children=[widgets.VBox(advanced_children)],
            selected_index=None,
        )
        advanced.set_title(0, "Advanced: query & column truth filters")

        for w in (self.event_kind, self.head_movement, self.query_box, *self.column_truth.values()):
            w.observe(lambda _c: self._refresh_filter_preview(), names="value")
        # Also refresh when block ticks change.
        for box in self.checks.values():
            box.observe(lambda _c: self._refresh_filter_preview(), names="value")

        return widgets.VBox(
            [
                widgets.HTML("<b>Saccade filter</b>"),
                self.event_kind,
                self.head_movement,
                advanced,
                widgets.HBox([self.filter_reset, self.filter_preview]),
            ],
            layout=widgets.Layout(
                border="1px solid #ddd", padding="6px", margin="4px 0"
            ),
        )

    def _reset_filter(self) -> None:
        if not self.enable_saccade_filter:
            return
        self.event_kind.value = "all"
        self.head_movement.value = "any"
        self.query_box.value = ""
        for dd in self.column_truth.values():
            dd.value = "any"
        self._refresh_filter_preview()

    def current_saccade_filter(self) -> SaccadeFilter | None:
        """Return the GUI filter, or ``None`` when inactive / disabled."""
        if not self.enable_saccade_filter:
            return None
        column_equals: dict[str, Any] = {}
        for col, dd in self.column_truth.items():
            if dd.value == "true":
                column_equals[col] = True
            elif dd.value == "false":
                column_equals[col] = False
            elif dd.value == "na":
                column_equals[col] = None
        filt = SaccadeFilter(
            event_kind=str(self.event_kind.value),
            head_movement=str(self.head_movement.value),
            query=str(self.query_box.value).strip() or None,
            column_equals=column_equals or None,
        )
        return filt if filt.is_active() else None

    def _refresh_filter_preview(self) -> None:
        if not self.enable_saccade_filter:
            return
        try:
            keys = self.selected
            base = filter_event_tables(
                self.ctx.tables, block_keys=keys or None
            )
            n_base = int(len(base.all_saccades))
            filt = self.current_saccade_filter()
            if filt is None:
                self.filter_preview.value = (
                    f"<span style='color:#666'>kept {n_base:,} / {n_base:,} events "
                    f"(no saccade filter)</span>"
                )
                return
            filtered = apply_saccade_filter(base, filt)
            n_keep = int(len(filtered.all_saccades))
            n_syn = int(len(filtered.synced))
            n_mono = int(len(filtered.non_synced))
            self.filter_preview.value = (
                f"<span style='color:#177245'>"
                f"kept {n_keep:,} / {n_base:,} events "
                f"(synced rows={n_syn:,}, monocular={n_mono:,}) — {filt.describe()}"
                f"</span>"
            )
        except Exception as exc:  # noqa: BLE001
            self.filter_preview.value = (
                f"<span style='color:#a11'>filter preview error: {exc}</span>"
            )

    def _populate_checks(self) -> None:
        qc = self.ctx.qc
        if qc.empty and self.ctx.tables.blocks:
            qc = block_qc_table(self.ctx.tables)
            self.ctx.qc = qc
        by_animal: dict[str, list[str]] = {}
        labels: dict[str, str] = {}
        enabled: dict[str, bool] = {}
        reasons: dict[str, str] = {}

        if not qc.empty:
            for _, row in qc.iterrows():
                key = str(row["block_key"])
                animal = str(row["animal"])
                ok, reason = _eligibility(self.ctx.tables, self.fig_id, key)
                enabled[key] = ok
                reasons[key] = reason
                label = (
                    f"{key}  L={row['n_saccades_L']} R={row['n_saccades_R']} "
                    f"synced={row['n_synced_pairs']}"
                )
                if not ok:
                    label += f"  [{reason}]"
                labels[key] = label
                by_animal.setdefault(animal, []).append(key)
        elif "jitter_bundle" in self.spec.needs:
            labels["__jitter__"] = "Fig 1e uses a jitter export path (set below / in runner kwargs)"
            enabled["__jitter__"] = True
            reasons["__jitter__"] = ""
            by_animal["(jitter)"] = ["__jitter__"]

        sections = []
        for animal, keys in sorted(by_animal.items()):
            toggle = widgets.Checkbox(
                value=True,
                description=f"{animal} ({len(keys)})",
                indent=False,
                layout=widgets.Layout(width="200px"),
            )
            self.animal_toggles[animal] = toggle

            def _make_handler(keys_local, toggle_w):
                def _handler(_change=None):
                    for k in keys_local:
                        box = self.checks.get(k)
                        if box is not None and not box.disabled:
                            box.value = bool(toggle_w.value)

                return _handler

            toggle.observe(_make_handler(keys, toggle), names="value")
            row_boxes = []
            for key in keys:
                if self.single_block:
                    # Radio-like: use checkboxes but Build will take the first ticked.
                    pass
                box = widgets.Checkbox(
                    value=enabled.get(key, False),
                    description=labels.get(key, key),
                    indent=False,
                    disabled=not enabled.get(key, False),
                    layout=widgets.Layout(width="98%"),
                    style={"description_width": "0px"},
                )
                self.checks[key] = box
                row_boxes.append(box)
            sections.append(widgets.VBox([toggle, *row_boxes]))
        self.checks_box.children = tuple(sections)

    def _ipython_display_(self) -> None:
        display(self.widget)

    def _set_status(self, msg: str, *, level: str = "info") -> None:
        color = {"info": "#333", "ok": "#177245", "warn": "#b35c00", "err": "#a11"}[level]
        self.status.value = f"<span style='color:{color}'>{msg}</span>"

    def _set_all(self, value: bool) -> None:
        for c in self.checks.values():
            if not c.disabled:
                c.value = value
        for t in self.animal_toggles.values():
            t.value = value
        self._refresh_filter_preview()

    @property
    def selected(self) -> list[str]:
        keys = [k for k, c in self.checks.items() if c.value and not c.disabled and not k.startswith("__")]
        if self.single_block and keys:
            return [keys[0]]
        return keys

    def _parse_overrides(self) -> dict[str, Any]:
        text = self.params_box.value.strip()
        if not text or text.startswith("# (no params"):
            return {}
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("Params override must be a YAML mapping")
        return data

    def build(self, *, show: bool = True) -> FigureBuildResult | None:
        """Filter, merge params, run the figure, record the result on ``ctx``."""
        if "jitter_bundle" in self.spec.needs:
            kwargs = dict(self.default_runner_kwargs)
            if "jitter_bundle" not in kwargs or kwargs["jitter_bundle"] is None:
                self._set_status(
                    "Set jitter_bundle= path (finalized jitter export) before building Fig 1e.",
                    level="warn",
                )
                return None
            block_keys: list[str] = []
        else:
            block_keys = self.selected
            if not block_keys:
                self._set_status("Tick at least one eligible block.", level="warn")
                return None
            kwargs = dict(self.default_runner_kwargs)

        try:
            overrides = self._parse_overrides()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Params YAML error: {exc}", level="err")
            return None

        try:
            saccade_filter = self.current_saccade_filter()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Saccade filter error: {exc}", level="err")
            return None
        filter_dict = saccade_filter.to_dict() if saccade_filter is not None else {}

        filt_note = f", filter=[{saccade_filter.describe()}]" if saccade_filter else ""
        self._set_status(
            f"Building {self.fig_id} from {len(block_keys) or 'bundle'}{filt_note} …"
        )
        with self.out:
            self.out.clear_output(wait=True)
            tables = self.ctx.tables
            if "traces" in self.spec.needs and block_keys:
                tables = ensure_traces_for_blocks(tables, block_keys)
                self.ctx.tables = tables  # keep traces for subsequent builds
            try:
                # Forward vignette window kwargs from extra widgets if present.
                for w in self._extra:
                    if getattr(w, "description", "") == "start_s":
                        kwargs["start_s"] = float(w.value)
                    elif getattr(w, "description", "") == "end_s":
                        kwargs["end_s"] = float(w.value)
                    elif getattr(w, "description", "") == "jitter_bundle":
                        kwargs["jitter_bundle"] = Path(str(w.value)).expanduser()
                if self.single_block and block_keys:
                    kwargs.setdefault("block_key", block_keys[0])

                outputs = run_figure(
                    self.fig_id,
                    tables,
                    self.ctx.out_dir,
                    block_keys=block_keys or None,
                    saccade_filter=saccade_filter,
                    params_overrides=overrides,
                    show=show,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Build failed: {exc}", level="err")
                import traceback

                traceback.print_exc()
                return None

            filtered = filter_event_tables(tables, block_keys=block_keys or None)
            filtered = apply_saccade_filter(filtered, saccade_filter)
            section = self.spec.params_section
            if section and overrides:
                params_used = overrides if section in overrides else {section: overrides}
            elif section:
                params_used = {section: deepcopy(tables.params.get(section, {}))}
            else:
                params_used = overrides

            result = FigureBuildResult(
                fig_id=self.fig_id,
                block_keys=list(block_keys),
                params_used=params_used,
                outputs=outputs,
                n_events=int(len(filtered.all_saccades)),
                n_synced_rows=int(len(filtered.synced)),
                runner_kwargs={
                    k: (str(v) if isinstance(v, Path) else v) for k, v in kwargs.items()
                },
                saccade_filter=filter_dict,
                notes=(saccade_filter.describe() if saccade_filter else ""),
            )
            self.result = result
            self.ctx.record(result)
            for name, path in outputs.items():
                if str(path).lower().endswith(".pdf"):
                    print(f"  {name}: {path}")
        self._set_status(
            f"Built {self.fig_id} — {len(block_keys)} block(s), "
            f"{result.n_events} events → {len(outputs)} output(s)"
            f"{filt_note}.",
            level="ok",
        )
        return result


def make_vignette_selector(fig_id: str, ctx: PaperContext, *, start_s: float, end_s: float) -> PaperFigureSelector:
    """PaperFigureSelector with start/end time widgets for Fig 3a–3c."""
    start_w = widgets.FloatText(value=float(start_s), description="start_s",
                                layout=widgets.Layout(width="180px"))
    end_w = widgets.FloatText(value=float(end_s), description="end_s",
                              layout=widgets.Layout(width="180px"))
    return PaperFigureSelector(
        fig_id,
        ctx,
        single_block=True,
        extra_widgets=[widgets.HBox([start_w, end_w])],
    )


def make_fig1e_selector(ctx: PaperContext, jitter_bundle: Path | str | None = None) -> PaperFigureSelector:
    """PaperFigureSelector for Fig 1e with a jitter-export path text box."""
    path_w = widgets.Text(
        value=str(jitter_bundle or ""),
        description="jitter_bundle",
        layout=widgets.Layout(width="98%"),
        style={"description_width": "100px"},
        placeholder="path/to/jitter_comparison_figures_…/",
    )
    return PaperFigureSelector(
        "1e",
        ctx,
        extra_widgets=[path_w],
        default_runner_kwargs={"jitter_bundle": Path(jitter_bundle)} if jitter_bundle else {},
    )


__all__ = [
    "CATALOG",
    "PaperContext",
    "PaperFigureSelector",
    "SaccadeFilter",
    "block_qc_table",
    "make_fig1e_selector",
    "make_vignette_selector",
]
