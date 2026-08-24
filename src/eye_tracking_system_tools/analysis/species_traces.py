"""Per-species φ/θ example traces (independent y-limits)."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.eye_trace_io import load_block_eyes
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle
from eye_tracking_system_tools.figures.plotting_functions import TRACE_L, TRACE_R

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

DEFAULT_DURATION_S = 80.0
MIN_FINITE_FRAC = 0.80

# start_s is a preferred wall-clock start (seconds). If that window is mostly NaN
# (common at recording onset), the exporter slides forward to the first stretch
# with enough finite k_phi. duration_s is the plotted length.
DEFAULTS = {
    "lizard": {
        "block_path": "/Volumes/Data-1/Nimrod/experiments/PV_126/2024_07_18/block_007",
        "animal": "PV_126",
        "block_num": "007",
        "start_s": 210.0,
        "end_s": 290.0,
        "duration_s": DEFAULT_DURATION_S,
        "filename": "trace_lizard.pdf",
    },
    "mouse": {
        "block_path": "/Volumes/Data/Nimrod/experiments/M_002/2026_07_28/block_012",
        "animal": "M_002",
        "block_num": "012",
        "start_s": 30.0,
        "end_s": 110.0,
        "duration_s": DEFAULT_DURATION_S,
        "filename": "trace_mouse.pdf",
    },
    "turtle": {
        "block_path": "/Volumes/Data/Nimrod/experiments/T_18/block_001",
        "animal": "T_18",
        "block_num": "001",
        "start_s": 50.0,
        "end_s": 130.0,
        "duration_s": DEFAULT_DURATION_S,
        "filename": "trace_turtle.pdf",
    },
}


def time_seconds_from_ms_axis(ms_axis) -> np.ndarray:
    """Convert ``ms_axis`` to seconds. Median dt ≳ 0.5 → values are milliseconds."""
    ms = np.asarray(ms_axis, dtype=float)
    finite = ms[np.isfinite(ms)]
    if finite.size < 2:
        return ms / 1000.0
    dt = float(np.nanmedian(np.diff(finite)))
    if dt > 0.5:
        return ms / 1000.0
    return ms


def pick_trace_window(
    t_s: np.ndarray,
    finite: np.ndarray,
    *,
    start_s: float | None,
    duration_s: float = DEFAULT_DURATION_S,
    min_finite_frac: float = MIN_FINITE_FRAC,
    step_s: float = 1.0,
) -> tuple[float, float]:
    """Return ``(lo, hi)`` seconds covering ``duration_s`` with enough finite samples."""
    t_s = np.asarray(t_s, dtype=float)
    finite = np.asarray(finite, dtype=bool)
    duration_s = float(duration_s)
    t_min = float(np.nanmin(t_s))
    t_max = float(np.nanmax(t_s))
    if not np.isfinite(t_min) or t_max - t_min < duration_s * 0.5:
        raise ValueError(
            f"trace shorter than requested window ({t_max - t_min:.3f}s < {duration_s}s)"
        )

    def _ok(lo: float, hi: float) -> bool:
        m = (t_s >= lo) & (t_s <= hi)
        if not bool(m.sum()) or float(finite[m].mean()) < min_finite_frac:
            return False
        # Do not start inside a NaN gap (that clips the visible trace).
        i0 = int(np.argmin(np.abs(t_s - lo)))
        head = (t_s >= lo) & (t_s < lo + min(1.0, duration_s * 0.05))
        if not bool(finite[i0]):
            return False
        if bool(head.sum()) and float(finite[head].mean()) < min_finite_frac:
            return False
        return True

    preferred = float(start_s) if start_s is not None else t_min
    hi0 = preferred + duration_s
    if _ok(preferred, hi0):
        return preferred, hi0

    search_from = t_min
    start = search_from
    while start + duration_s <= t_max + 1e-9:
        if _ok(start, start + duration_s):
            print(
                f"[species_traces] window {preferred:.1f}-{hi0:.1f}s has too many NaNs; "
                f"using {start:.1f}-{start + duration_s:.1f}s",
                flush=True,
            )
            return start, start + duration_s
        start += step_s
    raise ValueError(
        f"no {duration_s:.0f}s window with ≥{min_finite_frac:.0%} finite k_phi "
        f"(t={t_min:.1f}–{t_max:.1f}s)"
    )


def _recenter(series) -> np.ndarray:
    vals = np.asarray(series, dtype=float)
    med = np.nanmedian(vals)
    if np.isfinite(med):
        return vals - med
    return vals


def plot_species_trace(
    spec: BlockSpec,
    *,
    start_s: float | None,
    end_s: float | None = None,
    duration_s: float = DEFAULT_DURATION_S,
    out_pdf: Path,
    show: bool = False,
) -> tuple[Path, dict]:
    loaded = load_block_eyes(spec)
    left, right = loaded.left, loaded.right
    if left is None or right is None or left.empty or right.empty:
        raise ValueError(f"{spec.block_key}: missing eye traces")
    if "ms_axis" not in left.columns:
        raise ValueError(f"{spec.block_key}: no ms_axis column")
    _ = end_s  # window length comes from duration_s; start_s is preferred origin
    t = time_seconds_from_ms_axis(left["ms_axis"].to_numpy(dtype=float))
    phi_l_all = pd.to_numeric(left["k_phi"], errors="coerce").to_numpy()
    phi_r_all = pd.to_numeric(right["k_phi"], errors="coerce").to_numpy()
    n = min(t.size, phi_l_all.size, phi_r_all.size)
    t = t[:n]
    finite = np.isfinite(phi_l_all[:n]) & np.isfinite(phi_r_all[:n])
    lo, hi = pick_trace_window(
        t,
        finite,
        start_s=start_s,
        duration_s=float(duration_s),
    )
    mask = (t >= lo) & (t <= hi)
    if not np.any(mask) or not np.any(np.isfinite(phi_l_all[:n][mask])):
        raise ValueError(f"{spec.block_key}: no finite k_phi in {lo:.1f}–{hi:.1f}s")
    t_rel = t[mask] - lo
    phi_l = _recenter(phi_l_all[:n][mask])
    phi_r = _recenter(phi_r_all[:n][mask])
    th_l = _recenter(pd.to_numeric(left["k_theta"], errors="coerce").to_numpy()[:n][mask])
    th_r = _recenter(pd.to_numeric(right["k_theta"], errors="coerce").to_numpy()[:n][mask])
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 2.0), dpi=300, sharex=True)
    for ax, y_l, y_r, ylab in (
        (axes[0], phi_l, phi_r, "φ [deg]"),
        (axes[1], th_l, th_r, "θ [deg]"),
    ):
        ax.plot(t_rel, y_l, color=TRACE_L, lw=0.6, label="Left")
        ax.plot(t_rel, y_r, color=TRACE_R, lw=0.6, label="Right")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlim(0.0, float(duration_s))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("[s]", fontsize=8)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    payload = {
        "t_s": t_rel,
        "phi": {"L": phi_l, "R": phi_r},
        "theta": {"L": th_l, "R": th_r},
        "pdf_name": out_pdf.name,
        "animal": spec.animal,
        "block": spec.block_num,
        "window_start_s": lo,
        "window_end_s": hi,
        "duration_s": float(hi - lo),
        "time_unit": "seconds",
    }
    return out_pdf, payload


def export_species_traces(
    out_dir: Path,
    *,
    overrides: dict | None = None,
    show: bool = False,
    export: bool = True,
    show_animal: list[str] | None = None,
) -> dict[str, Path]:
    """
    Compute and optionally export species traces.

    Args:
        out_dir: Output directory.
        overrides: Optional override configuration.
        show: Whether to show the plot.
        export: If False, do not write any files or output directories, just compute and show.
        show_animal: list of species names to process; if not None, only include these.

    Returns:
        Dictionary mapping output filename to file path (empty if export is False).
    """
    written: dict[str, Path] = {}
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    if overrides:
        for species, body in overrides.items():
            cfg.setdefault(species, {}).update(body)
    for species, body in cfg.items():
        # Only compute/plot for requested species, if filtering
        if show_animal is not None and species not in show_animal:
            continue
        spec = BlockSpec(
            animal=str(body["animal"]),
            block_path=Path(body["block_path"]),
            block_num=str(body["block_num"]),
        )
        if not spec.block_path.is_dir():
            continue
        plot_id = f"trace_{species}"
        # If not exporting, we do not create any output directories/files
        if export:
            bundle = begin_plot_bundle(
                out_dir,
                plot_id,
                kind="species_trace",
                logic_key="species_trace",
                cohort={
                    "cohort": species,
                    "animals": [spec.animal],
                    "block_keys": [spec.block_key],
                    "rule": "species_traces.DEFAULTS",
                },
                extra={"species": species, "block_path": str(spec.block_path)},
            )
            out = bundle.plots_dir / str(body["filename"])
        else:
            bundle = None
            # Dummy output path just for plotting; not written to disk
            out = Path("/dev/null")
        duration_s = float(body.get("duration_s") or DEFAULT_DURATION_S)
        start_s = body.get("start_s")
        end_s = body.get("end_s")
        if start_s is not None and end_s is not None:
            duration_s = max(duration_s, float(end_s) - float(start_s))
        try:
            _, payload = plot_species_trace(
                spec,
                start_s=None if start_s is None else float(start_s),
                end_s=None if end_s is None else float(end_s),
                duration_s=duration_s,
                out_pdf=out,
                show=show,
            )
        except Exception as exc:
            print(f"[species_traces] skip {species}: {exc}")
            if export and bundle is not None:
                shutil.rmtree(bundle.bundle_dir, ignore_errors=True)
            continue
        if export and bundle is not None:
            write_pickle_with_meta(
                payload,
                bundle.metadata_dir / "trace_series.pkl",
                meta={
                    "species": species,
                    "animal": spec.animal,
                    "window_start_s": payload.get("window_start_s"),
                    "window_end_s": payload.get("window_end_s"),
                },
                entrypoint="eye_tracking_system_tools.analysis.species_traces.export_species_traces",
            )
            finish_plot_bundle(bundle)
            written[out.name] = out
    return written
