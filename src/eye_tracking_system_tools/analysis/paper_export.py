"""
Finalize flexible paper-figure builds into a dated, rebuildable export folder.

``finalize_paper_export`` writes

    paper_figures_<tag>_<YYYYmmdd>_<HH>_<MM>/
        figure_2c.pdf ...                  (only what was built)
        figure_specs.pickle                (selections + params + exporter ids)
        selections.csv                     (flat figure → block inventory)
        manifest.yaml                      (human-readable provenance)

The pickle stores selections + params + registry pointers only — rebuilding
re-runs exporters from the source blocks (data volumes must be mounted).
"""

from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection

import pandas as pd
import yaml

from eye_tracking_system_tools.analysis.export_meta import git_hash
from eye_tracking_system_tools.analysis.figure_catalog import CATALOG, run_figure
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.run_layout import assert_not_reproduction

BUNDLE_VERSION = 1
BUNDLE_NAME = "figure_specs.pickle"
SELECTIONS_CSV = "selections.csv"
MANIFEST_NAME = "manifest.yaml"
FOLDER_PREFIX = "paper_figures"


@dataclass
class FigureBuildResult:
    """One figure built in the notebook session."""

    fig_id: str
    block_keys: list[str]
    params_used: dict[str, Any]
    outputs: dict[str, Path]
    n_events: int = 0
    n_synced_rows: int = 0
    runner_kwargs: dict[str, Any] = field(default_factory=dict)
    saccade_filter: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class PaperExportResult:
    export_dir: Path
    bundle_path: Path
    selections_csv: Path
    manifest_path: Path
    figures: dict[str, Path] = field(default_factory=dict)
    bundle: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"export: {self.export_dir}"]
        for name, path in sorted(self.figures.items()):
            lines.append(f"  {name}: {path.name}")
        lines.append(f"  data: {self.bundle_path.name}")
        return "\n".join(lines)


def export_folder_name(tag: str = "", *, when: datetime | None = None) -> str:
    when = when or datetime.now()
    safe = "".join(c if (c.isalnum() or c in "-.") else "_" for c in str(tag).strip())
    safe = safe.strip("_")
    parts = [FOLDER_PREFIX, safe, when.strftime("%Y%m%d"), when.strftime("%H"), when.strftime("%M")]
    return "_".join(p for p in parts if p)


def _unique_dir(parent: Path, name: str) -> Path:
    candidate = parent / name
    n = 2
    while candidate.exists():
        candidate = parent / f"{name}_{n}"
        n += 1
    return candidate


def _pdf_outputs(outputs: dict[str, Path]) -> dict[str, Path]:
    return {k: Path(v) for k, v in outputs.items() if str(v).lower().endswith(".pdf")}


def selections_table(builds: list[FigureBuildResult]) -> pd.DataFrame:
    rows = []
    for b in builds:
        if b.block_keys:
            for key in b.block_keys:
                animal = key.split("_block_")[0] if "_block_" in key else ""
                rows.append(
                    {
                        "fig_id": b.fig_id,
                        "block_key": key,
                        "animal": animal,
                        "n_events": b.n_events,
                        "n_synced_rows": b.n_synced_rows,
                        "notes": b.notes,
                    }
                )
        else:
            rows.append(
                {
                    "fig_id": b.fig_id,
                    "block_key": "",
                    "animal": "",
                    "n_events": b.n_events,
                    "n_synced_rows": b.n_synced_rows,
                    "notes": b.notes or "no block selection (e.g. jitter bundle)",
                }
            )
    return pd.DataFrame(rows)


def finalize_paper_export(
    builds: list[FigureBuildResult] | dict[str, FigureBuildResult],
    out_root: Path | str,
    *,
    tag: str = "",
    registry_path: Path | str | None = None,
    params_path: Path | str | None = None,
    notes: str = "",
    when: datetime | None = None,
    scratch_figures_dir: Path | str | None = None,
) -> PaperExportResult:
    """
    Copy built PDFs into a dated folder and write selection / params provenance.

    ``builds`` may be a list or a ``{fig_id: FigureBuildResult}`` mapping from
    the notebook context.
    """
    if isinstance(builds, dict):
        build_list = list(builds.values())
    else:
        build_list = list(builds)
    if not build_list:
        raise ValueError("Nothing to export — build at least one figure first.")

    out_root = Path(out_root)
    assert_not_reproduction(out_root)
    when = when or datetime.now()
    export_dir = _unique_dir(out_root, export_folder_name(tag, when=when))
    export_dir.mkdir(parents=True)

    figures: dict[str, Path] = {}
    for b in build_list:
        for name, src in _pdf_outputs(b.outputs).items():
            src = Path(src)
            if not src.is_file():
                # Try scratch figures dir as fallback (by output basename).
                if scratch_figures_dir is not None:
                    alt = Path(scratch_figures_dir) / src.name
                    if not alt.is_file():
                        alt = Path(scratch_figures_dir) / Path(name).name
                    if alt.is_file():
                        src = alt
            if not src.is_file():
                continue
            dst = export_dir / src.name
            # Avoid collisions when two figures emit same basename (unlikely).
            if dst.exists():
                dst = export_dir / f"{b.fig_id}_{src.name}"
            shutil.copy2(src, dst)
            figures[f"{b.fig_id}:{src.name}"] = dst

    sel = selections_table(build_list)
    sel_path = export_dir / SELECTIONS_CSV
    sel.to_csv(sel_path, index=False)

    per_figure = []
    for b in build_list:
        per_figure.append(
            {
                "fig_id": b.fig_id,
                "label": CATALOG[b.fig_id].label if b.fig_id in CATALOG else b.fig_id,
                "block_keys": list(b.block_keys),
                "params_used": b.params_used,
                "runner_kwargs": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in (b.runner_kwargs or {}).items()
                },
                "saccade_filter": dict(b.saccade_filter or {}),
                "n_events": int(b.n_events),
                "n_synced_rows": int(b.n_synced_rows),
                "outputs": [Path(p).name for p in _pdf_outputs(b.outputs).values()],
                "notes": b.notes,
            }
        )

    bundle: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_local": when.isoformat(timespec="seconds"),
        "tag": str(tag),
        "notes": str(notes),
        "registry_path": str(registry_path) if registry_path else None,
        "params_path": str(params_path) if params_path else None,
        "git_commit": git_hash(),
        "figures": per_figure,
    }
    bundle_path = export_dir / BUNDLE_NAME
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "export_dir": str(export_dir),
        "tag": str(tag),
        "created_local": bundle["created_local"],
        "registry_path": bundle["registry_path"],
        "params_path": bundle["params_path"],
        "git_commit": bundle["git_commit"],
        "n_figures": len(per_figure),
        "figures": [
            {
                "fig_id": p["fig_id"],
                "n_blocks": len(p["block_keys"]),
                "n_events": p["n_events"],
                "outputs": p["outputs"],
            }
            for p in per_figure
        ],
        "notes": notes,
    }
    manifest_path = export_dir / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return PaperExportResult(
        export_dir=export_dir,
        bundle_path=bundle_path,
        selections_csv=sel_path,
        manifest_path=manifest_path,
        figures=figures,
        bundle=bundle,
    )


def load_paper_export(path: Path | str) -> dict[str, Any]:
    """Load ``figure_specs.pickle`` from a file or an export folder."""
    path = Path(path)
    if path.is_dir():
        path = path / BUNDLE_NAME
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or "figures" not in bundle:
        raise ValueError(f"{path}: not a paper figure export bundle")
    return bundle


def describe_export(path: Path | str) -> pd.DataFrame:
    """Flat inventory of figures × blocks from an export."""
    bundle = load_paper_export(path)
    rows = []
    for fig in bundle["figures"]:
        keys = fig.get("block_keys") or [""]
        for key in keys:
            rows.append(
                {
                    "fig_id": fig["fig_id"],
                    "label": fig.get("label", ""),
                    "block_key": key,
                    "n_events": fig.get("n_events", 0),
                    "n_synced_rows": fig.get("n_synced_rows", 0),
                    "outputs": ", ".join(fig.get("outputs") or []),
                }
            )
    return pd.DataFrame(rows)


def rebuild_from_export(
    path: Path | str,
    tables: EventTables,
    out_dir: Path | str,
    *,
    fig_ids: Collection[str] | None = None,
    show: bool = False,
) -> dict[str, dict[str, Path]]:
    """
    Re-run each recorded figure from its stored block selection and params.

    ``tables`` must cover every block referenced in the export (typically rebuilt
    from the recorded registry path). Data volumes must be mounted.
    """
    bundle = load_paper_export(path)
    out_dir = Path(out_dir)
    wanted = {str(f) for f in fig_ids} if fig_ids is not None else None
    written: dict[str, dict[str, Path]] = {}
    for fig in bundle["figures"]:
        fig_id = fig["fig_id"]
        if wanted is not None and fig_id not in wanted:
            continue
        if fig_id not in CATALOG:
            print(f"[skip] unknown fig_id {fig_id}")
            continue
        params_used = fig.get("params_used") or {}
        # params_used may be the full params dict or a single section body.
        overrides = params_used
        runner_kwargs = dict(fig.get("runner_kwargs") or {})
        # Paths stored as strings.
        for k, v in list(runner_kwargs.items()):
            if isinstance(v, str) and (k.endswith("_path") or k.endswith("_bundle") or "path" in k):
                runner_kwargs[k] = Path(v)
        result = run_figure(
            fig_id,
            tables,
            out_dir,
            block_keys=fig.get("block_keys") or None,
            saccade_filter=fig.get("saccade_filter") or None,
            params_overrides=overrides,
            show=show,
            **runner_kwargs,
        )
        written[fig_id] = result
    return written
