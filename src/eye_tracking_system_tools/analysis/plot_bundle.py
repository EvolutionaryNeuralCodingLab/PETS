"""Output folders for analysis plots: ``<run>/<plot_id>/{plots,metadata}``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from eye_tracking_system_tools.analysis.export_meta import git_hash
from eye_tracking_system_tools.analysis.run_layout import assert_not_reproduction

PLOTS_DIRNAME = "plots"
METADATA_DIRNAME = "metadata"


@dataclass
class PlotBundle:
    bundle_dir: Path
    plots_dir: Path
    metadata_dir: Path
    plot_id: str
    kind: str = "generic_pickle"
    cohort: dict[str, Any] = field(default_factory=dict)
    logic_key: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def bundle_root_for(out_dir: Path | str, plot_id: str) -> Path:
    out_dir = Path(out_dir)
    if out_dir.name == str(plot_id):
        return out_dir
    return out_dir / str(plot_id)


def begin_plot_bundle(
    out_dir: Path | str,
    plot_id: str,
    *,
    kind: str = "generic_pickle",
    tables: Any | None = None,
    cohort: dict[str, Any] | None = None,
    logic_key: str = "",
    params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> PlotBundle:
    del tables  # kept for call-site compatibility
    bundle_dir = bundle_root_for(out_dir, plot_id)
    plots_dir = bundle_dir / PLOTS_DIRNAME
    metadata_dir = bundle_dir / METADATA_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    assert_not_reproduction(bundle_dir)
    return PlotBundle(
        bundle_dir=bundle_dir,
        plots_dir=plots_dir,
        metadata_dir=metadata_dir,
        plot_id=str(plot_id),
        kind=str(kind),
        cohort=dict(cohort or {}),
        logic_key=logic_key or str(plot_id),
        params=dict(params or {}),
        extra=dict(extra or {}),
    )


def finish_plot_bundle(bundle: PlotBundle) -> None:
    meta = bundle.metadata_dir
    meta.mkdir(parents=True, exist_ok=True)
    params_out = {
        "kind": bundle.kind,
        "plot_id": bundle.plot_id,
        "git_hash": git_hash(),
        "params": bundle.params,
        **bundle.extra,
    }
    with open(meta / "params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params_out, f, sort_keys=False)
    with open(meta / "cohort.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(bundle.cohort or {"cohort": "unspecified"}, f, sort_keys=False)


__all__ = [
    "PlotBundle",
    "begin_plot_bundle",
    "finish_plot_bundle",
]
