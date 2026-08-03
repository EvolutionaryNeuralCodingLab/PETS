"""
Finalize a jitter mount comparison into a self-contained, dated deliverable.

``finalize_jitter_export`` writes

    jitter_comparison_figures_<tag>_<YYYYmmdd>_<HH>_<MM>/
        jitter_modular_vs_rigid.pdf
        jitter_mouse.pdf                 (when mouse blocks were pooled)
        jitter_comparison_data.pickle    (everything needed to redraw / re-edit)
        epochs_table.csv                 (the same inventory, minus the arrays)
        manifest.yaml                    (human-readable provenance)

The pickle holds one row per pooled epoch — animal, date, block, mount, eye, the
frame span it was cut from, the µm/px factor applied — together with that epoch's
frame indices and displacement values. Every plotted number therefore maps back to
a block and a frame range, and :func:`plot_from_bundle` redraws the figures from
the pickle alone (optionally from a filtered subset of the table).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection

import numpy as np
import pandas as pd
import yaml

from eye_tracking_system_tools.analysis.jitter_epochs import (
    EpochRecord,
    JitterBlockSpec,
    collect_epoch_records,
    figure_modular_vs_rigid,
    figure_mouse_histogram,
    infer_date,
    unit_label,
)
from eye_tracking_system_tools.analysis.run_layout import assert_not_reproduction

BUNDLE_VERSION = 1
BUNDLE_NAME = "jitter_comparison_data.pickle"
EPOCHS_CSV = "epochs_table.csv"
MANIFEST_NAME = "manifest.yaml"
FOLDER_PREFIX = "jitter_comparison_figures"
MOUNT_TYPES = ("modular", "rigid", "mouse")

# Identity + provenance columns; ``frames`` / ``values`` carry the data itself.
EPOCH_COLUMNS = [
    "epoch_key",
    "block_key",
    "animal",
    "date",
    "block",
    "mount_type",
    "eye",
    "epoch_index",
    "start_frame",
    "end_frame",
    "n_samples",
    "mean",
    "p95",
    "max",
    "scale_um_per_px",
    "notes",
    "block_path",
]


@dataclass
class ExportResult:
    """Paths written by :func:`finalize_jitter_export`."""

    export_dir: Path
    bundle_path: Path
    epochs_csv: Path
    manifest_path: Path
    figures: dict[str, Path] = field(default_factory=dict)
    bundle: dict[str, Any] = field(default_factory=dict)

    @property
    def epochs(self) -> pd.DataFrame:
        return self.bundle["epochs"]

    def __str__(self) -> str:
        lines = [f"export: {self.export_dir}"]
        for name, path in self.figures.items():
            lines.append(f"  {name}: {path.name}")
        lines.append(f"  data: {self.bundle_path.name}")
        return "\n".join(lines)


def export_folder_name(tag: str = "", *, when: datetime | None = None) -> str:
    """``jitter_comparison_figures_<tag>_<YYYYmmdd>_<HH>_<MM>`` (local clock)."""
    when = when or datetime.now()
    safe = "".join(c if (c.isalnum() or c in "-.") else "_" for c in str(tag).strip())
    safe = safe.strip("_")
    parts = [FOLDER_PREFIX, safe, when.strftime("%Y%m%d"), when.strftime("%H"), when.strftime("%M")]
    return "_".join(p for p in parts if p)


def _unique_dir(parent: Path, name: str) -> Path:
    """Never overwrite a finalized export: append ``_2``, ``_3`` … if needed."""
    candidate = parent / name
    n = 2
    while candidate.exists():
        candidate = parent / f"{name}_{n}"
        n += 1
    return candidate


def epochs_table(records: list[EpochRecord]) -> pd.DataFrame:
    """One row per pooled epoch: identity, stats, frame indices and values."""
    rows = []
    for rec in records:
        finite = rec.values[np.isfinite(rec.values)]
        rows.append(
            {
                "epoch_key": rec.epoch_key,
                "block_key": rec.block_key,
                "animal": rec.spec.animal,
                "date": infer_date(rec.spec.block_path),
                "block": rec.spec.block_path.name,
                "mount_type": rec.mount_type,
                "eye": rec.eye,
                "epoch_index": int(rec.epoch_index),
                "start_frame": int(rec.start_frame),
                "end_frame": int(rec.end_frame),
                "n_samples": int(rec.values.size),
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "p95": float(np.percentile(finite, 95)) if finite.size else np.nan,
                "max": float(np.max(finite)) if finite.size else np.nan,
                "scale_um_per_px": rec.scale_um_per_px,
                "notes": rec.notes,
                "block_path": str(rec.spec.block_path),
                "frames": rec.frames,
                "displacement": rec.values,
            }
        )
    columns = EPOCH_COLUMNS + ["frames", "displacement"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def bundle_pools(
    bundle: dict[str, Any],
    *,
    epochs: pd.DataFrame | None = None,
    include: Collection[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Rebuild the ``{mount_type: samples}`` pools the figures are drawn from.

    ``epochs`` accepts a filtered copy of ``bundle["epochs"]`` (drop rows, then
    re-plot); ``include`` is the shorthand for keeping whole blocks by key.
    """
    table = bundle["epochs"] if epochs is None else epochs
    if include is not None:
        keys = {str(k) for k in include}
        unknown = keys - set(table["block_key"])
        if unknown:
            raise KeyError(f"Unknown block key(s): {sorted(unknown)}")
        table = table[table["block_key"].isin(keys)]
    pools: dict[str, np.ndarray] = {}
    for mount in MOUNT_TYPES:
        parts = [
            np.asarray(v, dtype=float)
            for v in table.loc[table["mount_type"] == mount, "displacement"]
            if np.size(v)
        ]
        pools[mount] = np.concatenate(parts) if parts else np.array([], dtype=float)
    return pools


def bundle_samples_long(
    bundle: dict[str, Any],
    *,
    epochs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Explode the epoch table into one row per sample.

    Columns: identity (``animal``/``date``/``block``/``mount_type``/``eye``/
    ``epoch_key``), ``frame`` and ``displacement``. Kept out of the pickle because
    it is a few million rows; build it on demand for stats or CSV export.
    """
    table = bundle["epochs"] if epochs is None else epochs
    id_cols = ["epoch_key", "block_key", "animal", "date", "block", "mount_type", "eye"]
    frames: list[pd.DataFrame] = []
    for row in table.itertuples(index=False):
        n = int(np.size(row.displacement))
        if not n:
            continue
        block = {c: [getattr(row, c)] * n for c in id_cols}
        block["frame"] = np.asarray(row.frames, dtype=np.int64)
        block["displacement"] = np.asarray(row.displacement, dtype=float)
        frames.append(pd.DataFrame(block))
    if not frames:
        return pd.DataFrame(columns=[*id_cols, "frame", "displacement"])
    out = pd.concat(frames, ignore_index=True)
    out.attrs["units"] = bundle.get("units", "um")
    return out


def _figure_paths(out_dir: Path, suffix: str = "") -> dict[str, Path]:
    return {
        "modular_vs_rigid": out_dir / f"jitter_modular_vs_rigid{suffix}.pdf",
        "mouse": out_dir / f"jitter_mouse{suffix}.pdf",
    }


def _draw(
    pools: dict[str, np.ndarray],
    out_dir: Path | None,
    *,
    units: str,
    n_bins: int,
    xmax: float | None,
    show: bool,
    suffix: str = "",
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Draw both histograms, saving when ``out_dir`` is given."""
    paths = _figure_paths(out_dir, suffix) if out_dir is not None else {}
    written: dict[str, Path] = {}
    figures: dict[str, Any] = {}
    specs = [
        ("modular_vs_rigid", figure_modular_vs_rigid, pools["modular"].size or pools["rigid"].size),
        ("mouse", figure_mouse_histogram, pools["mouse"].size),
    ]
    for name, builder, has_data in specs:
        if not has_data:
            continue
        fig = builder(pools, xmax=xmax, n_bins=n_bins, units=units)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(paths[name], format="pdf", bbox_inches="tight")
            written[name] = paths[name]
        figures[name] = fig
    if show:
        from IPython.display import display

        for fig in figures.values():
            display(fig)
    import matplotlib.pyplot as plt

    for fig in figures.values():
        plt.close(fig)
    return written, figures


def finalize_jitter_export(
    specs: list[JitterBlockSpec],
    metadata_dir: Path | str,
    out_root: Path | str,
    *,
    tag: str = "",
    units: str = "um",
    n_bins: int = 15,
    xmax: float | None = None,
    include: Collection[str] | None = None,
    key: str = "top_correlation_dist",
    notes: str = "",
    registry: Path | str | None = None,
    show: bool = True,
    when: datetime | None = None,
) -> ExportResult:
    """
    Freeze the current pooling choice into a dated folder of figures + data.

    ``include`` is the block-key list from the pool selector (``pool.selected``);
    omit it to pool every block that has saved epochs.
    """
    metadata_dir = Path(metadata_dir)
    out_root = Path(out_root)
    assert_not_reproduction(out_root)

    records = collect_epoch_records(specs, metadata_dir, key=key, units=units, verbose=True)
    if include is not None:
        keys = {str(k) for k in include}
        unknown = keys - {r.block_key for r in records}
        if unknown:
            raise KeyError(
                f"No epochs collected for: {sorted(unknown)} "
                "(pick epochs for them, or drop them from the selection)"
            )
        records = [r for r in records if r.block_key in keys]
    if not records:
        raise ValueError(
            "Nothing to export — no saved epochs for the selected blocks. "
            "Pick epochs in section 4 first."
        )

    table = epochs_table(records)
    pools = bundle_pools({"epochs": table})
    finite = np.concatenate([a[np.isfinite(a)] for a in pools.values() if a.size])
    resolved_xmax = float(xmax) if xmax is not None else float(np.nanpercentile(finite, 99.5))

    export_dir = _unique_dir(out_root, export_folder_name(tag, when=when))
    export_dir.mkdir(parents=True)

    written, _ = _draw(
        pools,
        export_dir,
        units=units,
        n_bins=n_bins,
        xmax=resolved_xmax,
        show=show,
    )

    bundle: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_local": (when or datetime.now()).isoformat(timespec="seconds"),
        "tag": str(tag),
        "notes": str(notes),
        "units": units,
        "unit_label": unit_label(units),
        "amplitude_key": key,
        "n_bins": int(n_bins),
        "xmax": resolved_xmax,
        "epochs": table,
        "group_stats": {m: _stats(a) for m, a in pools.items()},
        "sources": {
            "registry": str(registry) if registry else None,
            "epochs_metadata_dir": str(metadata_dir),
            "blocks": sorted({r.block_key for r in records}),
        },
        "figures": {k: v.name for k, v in written.items()},
        "note": (
            "epochs: one row per pooled epoch; 'displacement' holds the values in "
            f"{unit_label(units)} and 'frames' the matching video frame indices. "
            "Redraw with analysis.jitter_export.plot_from_bundle()."
        ),
    }

    bundle_path = export_dir / BUNDLE_NAME
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    epochs_csv = export_dir / EPOCHS_CSV
    table[EPOCH_COLUMNS].to_csv(epochs_csv, index=False)

    manifest_path = export_dir / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_manifest(bundle, table, export_dir), f, sort_keys=False, allow_unicode=True)

    result = ExportResult(
        export_dir=export_dir,
        bundle_path=bundle_path,
        epochs_csv=epochs_csv,
        manifest_path=manifest_path,
        figures=written,
        bundle=bundle,
    )
    print(result)
    return result


def _stats(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "n_samples": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else None,
        "median": float(np.median(arr)) if arr.size else None,
        "p95": float(np.percentile(arr, 95)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
    }


def _manifest(bundle: dict[str, Any], table: pd.DataFrame, export_dir: Path) -> dict[str, Any]:
    grouped = (
        table.groupby(["block_key", "animal", "date", "block", "mount_type"], dropna=False)
        .agg(n_epochs=("epoch_key", "count"), n_samples=("n_samples", "sum"), max=("max", "max"))
        .reset_index()
    )
    per_block = [
        {
            "block_key": str(r.block_key),
            "animal": str(r.animal),
            "date": str(r.date),
            "block": str(r.block),
            "mount_type": str(r.mount_type),
            "n_epochs": int(r.n_epochs),
            "n_samples": int(r.n_samples),
            "max": float(r.max) if np.isfinite(r.max) else None,
        }
        for r in grouped.itertuples(index=False)
    ]
    return {
        "created_local": bundle["created_local"],
        "created_utc": bundle["created_utc"],
        "tag": bundle["tag"],
        "notes": bundle["notes"],
        "folder": export_dir.name,
        "units": bundle["units"],
        "amplitude_key": bundle["amplitude_key"],
        "n_bins": bundle["n_bins"],
        "xmax": bundle["xmax"],
        "figures": bundle["figures"],
        "data_file": BUNDLE_NAME,
        "epochs_csv": EPOCHS_CSV,
        "group_stats": bundle["group_stats"],
        "sources": bundle["sources"],
        "blocks": per_block,
        "redraw": (
            "from eye_tracking_system_tools.analysis.jitter_export import plot_from_bundle; "
            f"plot_from_bundle('{export_dir / BUNDLE_NAME}')"
        ),
    }


def load_jitter_bundle(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    """Load a bundle from the pickle, or from the export folder holding it."""
    if isinstance(source, dict):
        return source
    path = Path(source)
    if path.is_dir():
        path = path / BUNDLE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No jitter bundle at {path}")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or "epochs" not in bundle:
        raise ValueError(f"{path}: not a jitter comparison bundle")
    return bundle


def plot_from_bundle(
    source: Path | str | dict[str, Any],
    *,
    out_dir: Path | str | None = None,
    epochs: pd.DataFrame | None = None,
    include: Collection[str] | None = None,
    n_bins: int | None = None,
    xmax: float | None = None,
    units: str | None = None,
    show: bool = True,
    suffix: str = "",
) -> dict[str, Path]:
    """
    Redraw the histograms from an exported bundle.

    Pass ``epochs=bundle["epochs"].query(...)`` or ``include=[block_key, …]`` to
    re-plot a subset, ``out_dir`` to write PDFs (``suffix`` keeps the originals
    intact), and ``n_bins`` / ``xmax`` to restyle without touching the data.
    """
    bundle = load_jitter_bundle(source)
    pools = bundle_pools(bundle, epochs=epochs, include=include)
    if not any(a.size for a in pools.values()):
        raise ValueError("No samples left after filtering — nothing to plot")
    written, _ = _draw(
        pools,
        Path(out_dir) if out_dir is not None else None,
        units=units or bundle.get("units", "um"),
        n_bins=int(n_bins if n_bins is not None else bundle.get("n_bins", 15)),
        xmax=xmax if xmax is not None else bundle.get("xmax"),
        show=show,
        suffix=suffix,
    )
    return written


def describe_bundle(source: Path | str | dict[str, Any]) -> pd.DataFrame:
    """
    Per-block inventory of an exported bundle (no arrays).

    Stats are pooled over the block's samples, not averaged over its epochs.
    """
    bundle = load_jitter_bundle(source)
    table = bundle["epochs"]
    group_cols = ["mount_type", "animal", "date", "block", "block_key"]
    rows = []
    for keys, grp in table.groupby(group_cols, dropna=False, sort=False):
        vals = np.concatenate([np.asarray(v, dtype=float) for v in grp["displacement"]])
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_epochs": int(len(grp)),
                "n_samples": int(grp["n_samples"].sum()),
                "first_frame": int(grp["start_frame"].min()),
                "last_frame": int(grp["end_frame"].max()),
                "mean": float(np.mean(vals)) if vals.size else np.nan,
                "p95": float(np.percentile(vals, 95)) if vals.size else np.nan,
                "max": float(np.max(vals)) if vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["mount_type", "animal", "block"]).reset_index(drop=True)
