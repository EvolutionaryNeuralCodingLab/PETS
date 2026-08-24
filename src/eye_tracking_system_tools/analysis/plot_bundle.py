"""Self-contained per-plot export bundles: plots/ + metadata/ + replot.py."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from eye_tracking_system_tools.analysis.export_meta import git_hash
from eye_tracking_system_tools.analysis.run_layout import assert_not_reproduction

PLOTS_DIRNAME = "plots"
METADATA_DIRNAME = "metadata"

_MOUSE_ANIMAL = re.compile(r"^M_\d", re.IGNORECASE)
_LIZARD_ANIMAL = re.compile(r"^PV_", re.IGNORECASE)
LIZARD_PAPER_ANIMALS = frozenset({"PV_106", "PV_143", "PV_62", "PV_126", "PV_57"})

CATALOG_FOLDER = {
    "2c_2d": "figure_2c_2d",
    "2e": "figure_2e",
    "2f": "figure_2f",
    "s3": "figure_S3",
    "2g": "figure_2g",
    "2h": "figure_2h",
    "2i": "figure_2i",
    "2j": "figure_2j",
    "2b": "figure_2b",
    "2b_examples": "figure_2b_examples",
    "3a": "figure_3a",
    "3b": "figure_3b",
    "3c": "figure_3c",
    "3d": "figure_3d",
    "3e": "figure_3e",
    "3f": "figure_3f",
    "1e": "figure_1e",
}

KIND_FOR_FIG = {
    "2c_2d": "pos_vel",
    "2e": "figure_2e",
    "2f": "figure_2f",
    "s3": "figure_s3",
    "2g": "figure_2g",
    "2h": "generic_pickle",
    "2i": "generic_pickle",
    "2j": "generic_pickle",
    "2b": "generic_pickle",
    "2b_examples": "generic_pickle",
    "3a": "generic_pickle",
    "3b": "generic_pickle",
    "3c": "generic_pickle",
    "3d": "isi_hist",
    "3e": "generic_pickle",
    "3f": "generic_pickle",
    "1e": "jitter_histogram",
}

LOGIC_TEXT: dict[str, str] = {
    "figure_2f": (
        "Inter-ocular peak-speed 2D histogram.\n"
        "Concurrent L/R pairs (binocular.sync_diff_ms, 34 ms) contribute one point "
        "from both detected event peaks. Unpaired events keep the ipsilateral event "
        "peak vs contralateral ±51 ms window sample. Head-stationary; equal-animal weights.\n"
        "Replot reads the histogram pickle (and re-bins from stored speeds if "
        "present); it does not re-detect saccades.\n"
    ),
    "figure_S3": (
        "Same collector as Fig 2f (one point per concurrent pair; unpaired keep the "
        "contra-window sample), all saccades, still vs moving with a shared colorbar.\n"
        "Replot reads figure_S3.pickle; it does not re-detect saccades.\n"
    ),
    "figure_2c_2d": (
        "Amplitude-binned mean position (2d) and unsigned angular speed (2c). "
        "Align to peak; Gaussian-smooth numerator/occupancy with bandwidth_ms / dt_ms.\n"
        "Replot reads pos_vel_by_amp_bins_bundle.pkl.\n"
    ),
    "figure_2e": (
        "Amplitude–peak-speed linear fit (Fig 2e). Per-animal amp-bin means for all "
        "saccades plus the same panel filtered to concurrent and monocular events "
        "(pairing window 34 ms). Shared xlim/ylim across the three means plots. "
        "The dashed line is OLS (scipy.stats.linregress) on individual events; "
        "panels annotate slope and Pearson r. Pooled Pogona (PV_*) events are "
        "drawn as an overcrowded all-animal scatter and as a Gaussian-KDE "
        "likelihood map (Fig 2h settings: Scott bandwidth, 200 bins, turbo, "
        "normalized so the grid sums to 1). Single-animal class scatters still "
        "use scatter_animal.\n"
        "Replot reads amplitude_velocity_linear_fit_bundle.pkl.\n"
    ),
    "diagnostics_2e": (
        "Diagnostic scatter of Fig 2e (amp vs peak V) for pooled Pogona events, "
        "colored by detector length (end_index − start_index). Same amp/V filters "
        "as Fig 2e. Overlay: 1-frame identity V=amp/(1000/fps), detection floor "
        "(speed_threshold / frame_ms), and OLS fit. The steep ridge is short "
        "(length 1–2) events, not a plot cutoff.\n"
        "Replot reads diagnostics_2e.pkl.\n"
    ),
    "robustness": (
        "Monocular-fraction vs speed threshold and pairing window; Engbert–Kliegl "
        "noise floor from non-saccade samples. Shuffle (if present) is a binocular "
        "coincidence null (circular-shift R onsets), not a tracking-noise test.\n"
    ),
    "robustness_noise_floor": (
        "Engbert–Kliegl noise floor: inter-frame Δangle outside saccade windows. "
        "The orange line is the mean GUI speed threshold (deg/frame); the gray "
        "band is the min–max of per-block GUI thresholds used at finalize.\n"
    ),
    "ISI_by_state": (
        "Same-epoch ISIs only (drop intervals that cross a behavior-state boundary). "
        "Log and linear histograms for active and quiet.\n"
    ),
    "epoch_durations": (
        "Active/quiet epoch length histograms. Each state is drawn twice: log-x "
        "with log-spaced bins, and linear-x with linear bins from 0 to max.\n"
    ),
    "epoch_durations_raw": (
        "Raw (unsmoothed) active/quiet epoch lengths from behavior_state.csv. "
        "Log-x and linear-x (0 to max) with 30 bins each.\n"
    ),
    "epoch_durations_smoothed": (
        "Same epoch lengths after smooth_behavior_state (bridge A–B–A ≤3 s, "
        "then min 5 s per state). Epochs shorter than 1 s are dropped; linear "
        "x-axis starts at 1 s.\n"
    ),
    "figure_3e_raw": (
        "Fig 3e pupil-diameter quiet vs active histogram (PV_62), raw behavior state.\n"
    ),
    "figure_3e_smoothed": (
        "Fig 3e with the same bins/animals, but behavior state smoothed in memory "
        "(bridge ≤3 s, min 5 s). CSVs are not rewritten.\n"
    ),
    "figure_3f_raw": (
        "Fig 3f per-animal z-scored (active−quiet) pupil difference, raw state.\n"
    ),
    "figure_3f_smoothed": (
        "Fig 3f with the same bins/exclusions, smoothed behavior state in memory.\n"
    ),
    "qc_reversals_noise_blinks": (
        "Reversal = ≥50% return toward start within 80 ms after offset. Noise traces "
        "and nictitating peri from pupil-perimeter epochs or DLC likelihood fallback.\n"
    ),
    "corrective_head": (
        "Corrective = contralateral monocular partner in (0, 80] ms and ≤45°. "
        "Head peri from lizMov.mat rising edges vs saccade onset.\n"
    ),
    "s1_occupancy": (
        "Empirical eccentricity occupancy hypot(k_phi, k_theta); vertical line at ±35°.\n"
    ),
    "jitter_histogram": (
        "Residual camera displacement at the eye plane from jitter_report_dict "
        "top_correlation_dist, scaled by LR_pix_size.csv when units=um. "
        "Pooling is by registry mount_type (not animal-name guessing).\n"
    ),
    "unified_jitter": (
        "Four-panel jitter histograms (rigid lizard, modular lizard, modular "
        "mouse, modular turtle) sharing one x-limit: the 99.5th percentile of "
        "the most jittery mount type, with identical bins.\n"
    ),
    "unified_noise": (
        "Four-panel non-saccade inter-frame Δangle histograms (rigid lizard, "
        "modular lizard, turtle, mouse). Shared x-limit is the 99.5th percentile "
        "of the noisiest condition. Saccade windows are excluded when detections "
        "exist; turtle uses all finite samples. No detector threshold is drawn.\n"
    ),
    "raw_interframe_delta": (
        "Four-panel inter-frame Δangle (hypot Δφ, Δθ) with no saccade mask and "
        "no percentile trim. Linear PDF: histogram + MLE Rayleigh overlay. "
        "Scale B is the noise measure (σ of each Gaussian axis if the model holds). "
        "Rayleigh shape checks: skew≈0.631, excess kurtosis≈0.245. "
        "The pickle stores every finite sample; trim in the notebook if needed.\n"
    ),
    "yield_histogram": (
        "DLC pupil likelihood and ellipse completeness after noise/manual removals, "
        "pooled by registry mount_type.\n"
    ),
    "figure_3c": (
        "Single-block vignette (PV_126 block 007, 200–415 s) with φ/θ, pupil, "
        "and saccade/head rate. The overhead active/quiet strip is drawn twice: "
        "raw behavior_state.csv, and smooth_behavior_state (bridge ≤3 s, min 5 s). "
        "figure_3c.pdf is the smoothed strip; figure_3c_raw.pdf keeps the unsmoothed one.\n"
    ),
    "species_trace": (
        "Recentred φ/θ traces from one hardcoded block per species, 80 s each. "
        "ms_axis is converted to seconds; if the requested window is mostly NaN "
        "(typical at recording onset), the exporter slides to the first valid stretch. "
        "Independent y-limits.\n"
    ),
}


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


def catalog_folder_id(fig_id: str, cohort: str = "lizard") -> str:
    base = CATALOG_FOLDER.get(str(fig_id), f"figure_{fig_id}")
    if str(cohort).lower() == "mouse":
        return f"mouse_{base}"
    return base


def animals_from_tables(tables: Any) -> list[str]:
    seen: list[str] = []
    blocks = getattr(tables, "blocks", None) or []
    for b in blocks:
        spec = getattr(b, "spec", None)
        animal = str(getattr(spec, "animal", "") or "")
        if animal and animal not in seen:
            seen.append(animal)
    df = getattr(tables, "all_saccades", None)
    if df is not None and getattr(df, "empty", True) is False and "animal" in df.columns:
        for a in df["animal"].astype(str).unique().tolist():
            if a and a not in seen:
                seen.append(a)
    return seen


def block_keys_from_tables(tables: Any) -> list[str]:
    keys: list[str] = []
    for b in getattr(tables, "blocks", None) or []:
        spec = getattr(b, "spec", None)
        key = str(getattr(spec, "block_key", "") or "")
        if key:
            keys.append(key)
    return keys


def infer_catalog_cohort(
    tables: Any,
    *,
    override: str | None = None,
) -> dict[str, Any]:
    """Return cohort metadata. Mixed lizard/mouse animals raise unless override is set."""
    animals = animals_from_tables(tables)
    keys = block_keys_from_tables(tables)
    if override:
        cohort = str(override).strip().lower()
        if cohort not in {"mouse", "lizard"}:
            raise ValueError(f"cohort override must be 'mouse' or 'lizard', got {override!r}")
        return {
            "cohort": cohort,
            "animals": animals,
            "block_keys": keys,
            "rule": "explicit",
        }
    if not animals:
        return {
            "cohort": "lizard",
            "animals": animals,
            "block_keys": keys,
            "rule": "empty_default_lizard",
        }
    mouse = [_MOUSE_ANIMAL.match(a) is not None for a in animals]
    lizard = [
        _LIZARD_ANIMAL.match(a) is not None or a in LIZARD_PAPER_ANIMALS
        for a in animals
    ]
    if all(mouse):
        return {
            "cohort": "mouse",
            "animals": animals,
            "block_keys": keys,
            "rule": "animal_ids",
        }
    if all(lizard):
        return {
            "cohort": "lizard",
            "animals": animals,
            "block_keys": keys,
            "rule": "animal_ids",
        }
    raise ValueError(
        "Mixed or unknown catalog animals "
        f"{animals}; pass cohort='mouse' or cohort='lizard' to override."
    )


def resolve_plot_bundle(run_dir: Path | str, plot_id: str) -> tuple[Path, Path, Path]:
    """Create ``run_dir / plot_id / {plots, metadata}``."""
    bundle = Path(run_dir) / str(plot_id)
    plots_dir = bundle / PLOTS_DIRNAME
    metadata_dir = bundle / METADATA_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    assert_not_reproduction(bundle)
    return bundle, plots_dir, metadata_dir


def bundle_root_for(out_dir: Path | str, plot_id: str) -> Path:
    """If ``out_dir`` is already this plot folder, keep it; else nest ``plot_id``."""
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
    bundle_dir = bundle_root_for(out_dir, plot_id)
    plots_dir = bundle_dir / PLOTS_DIRNAME
    metadata_dir = bundle_dir / METADATA_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    assert_not_reproduction(bundle_dir)
    info = dict(cohort or {})
    if not info and tables is not None:
        try:
            info = infer_catalog_cohort(tables)
        except ValueError:
            info = {
                "cohort": "unspecified",
                "animals": animals_from_tables(tables),
                "block_keys": block_keys_from_tables(tables),
                "rule": "not_catalog",
            }
    return PlotBundle(
        bundle_dir=bundle_dir,
        plots_dir=plots_dir,
        metadata_dir=metadata_dir,
        plot_id=str(plot_id),
        kind=str(kind),
        cohort=info,
        logic_key=logic_key or str(plot_id),
        params=dict(params or {}),
        extra=dict(extra or {}),
    )


def finish_plot_bundle(bundle: PlotBundle) -> None:
    """Write params.yaml, cohort.yaml, LOGIC.md, and replot.py into the bundle."""
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
    logic = LOGIC_TEXT.get(bundle.logic_key) or LOGIC_TEXT.get(bundle.plot_id) or (
        f"Plot bundle {bundle.plot_id} (kind={bundle.kind}).\n"
        "Run python replot.py to redraw PDFs from metadata/ into plots/replot/.\n"
        "Replot does not re-detect saccades or reload lab volumes.\n"
    )
    logic = logic.rstrip() + (
        "\n\nTo redraw without the PETS package:\n"
        "  python replot.py\n"
        "PDFs land in plots/replot/ (originals in plots/ are not overwritten).\n"
        "  python replot.py --overwrite   # replace plots/ in place\n"
    )
    (meta / "LOGIC.md").write_text(logic, encoding="utf-8")
    write_replot_script(bundle.bundle_dir, bundle.kind)


def write_replot_script(bundle_dir: Path, kind: str) -> Path:
    path = Path(bundle_dir) / "replot.py"
    path.write_text(_REPLOT_SCRIPT.replace("__KIND__", str(kind)), encoding="utf-8")
    return path


def is_plot_bundle(path: Path | str) -> bool:
    p = Path(path)
    return (p / PLOTS_DIRNAME).is_dir() and (p / METADATA_DIRNAME).is_dir()


def iter_plot_bundles(run_dir: Path | str) -> list[Path]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    found = [p for p in sorted(run_dir.iterdir()) if p.is_dir() and is_plot_bundle(p)]
    return found


# Standalone reviewer script (no PETS import). Kind is patched in.
_REPLOT_SCRIPT = r'''#!/usr/bin/env python3
"""Redraw PDFs from this folder's metadata/ into plots/replot/. No PETS install required."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
META = ROOT / "metadata"
PLOTS = ROOT / "plots"
OUT = PLOTS if "--overwrite" in sys.argv else (PLOTS / "replot")
KIND = "__KIND__"


def _load_pickle():
    for p in sorted(META.glob("*.pickle")) + sorted(META.glob("*.pkl")):
        with open(p, "rb") as f:
            return pickle.load(f), p
    return None, None


def _savefig(fig, name: str) -> None:
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _nice_axis_max(value, *, step=None):
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        return 0.5
    if step is None:
        step = 0.05 if x < 2.0 else 0.1
    return float(np.ceil((x / step) - 1e-12) * step)


def _ticks_for_span(lo, hi):
    lo, hi = float(lo), float(hi)
    if hi <= 0.3:
        step = 0.1
    elif hi <= 0.6:
        step = 0.25
    elif hi <= 1.5:
        step = 0.5
    else:
        step = 1.0
    ticks = []
    t = lo
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    if ticks[-1] < hi - 1e-9:
        ticks.append(round(hi, 10))
    return ticks


def _hist2d(x, y, w, rng, bins):
    n_edge = max(int(bins), 2)
    edges = np.linspace(float(rng[0]), float(rng[1]), n_edge)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        z = np.zeros((n_edge - 1, n_edge - 1), dtype=float)
        return {"xedges": edges, "yedges": edges, "norm_counts": z}
    counts, xe, ye = np.histogram2d(x, y, bins=[edges, edges], weights=w)
    norm = counts / counts.sum() if counts.sum() > 0 else counts
    return {"xedges": xe, "yedges": ye, "norm_counts": norm}


def _rebin_figure_2f(data: dict) -> dict:
    """Rebuild macro/micro hists from stored speeds so axis-limit edits take effect."""
    right = np.asarray(data.get("right_eye_speeds", []), dtype=float)
    left = np.asarray(data.get("left_eye_speeds", []), dtype=float)
    if right.size == 0 or left.size == 0:
        return data
    try:
        from eye_tracking_system_tools.analysis.figures_2f_2h_2i import rebin_figure_2f_data

        return rebin_figure_2f_data(data)
    except Exception:
        pass
    auto = bool(data.get("auto_view_limits"))
    pct = float(data.get("macro_pct", 99.5))
    bins = int(data.get("bins", 60))
    weights = data.get("weights")
    data = dict(data)
    if auto:
        finite_r = right[np.isfinite(right)]
        finite_l = left[np.isfinite(left)]
        hi = _nice_axis_max(
            max(float(np.nanpercentile(finite_r, pct)), float(np.nanpercentile(finite_l, pct)))
        )
        frac = float(data.get("micro_frac_of_macro", 0.2))
        micro_hi = _nice_axis_max(hi * frac, step=0.05)
        data["macro_range"] = (0.0, hi)
        data["micro_range"] = (0.0, micro_hi)
        data["macro_tick_list"] = _ticks_for_span(0.0, hi)
        data["micro_tick_list"] = _ticks_for_span(0.0, micro_hi)
    macro_range = tuple(data["macro_range"])
    micro_range = tuple(data["micro_range"])
    data["macro"] = _hist2d(right, left, weights, macro_range, bins)
    data["micro"] = _hist2d(right, left, weights, micro_range, bins)
    data["vmax_all"] = float(
        max(
            np.nanmax(data["macro"]["norm_counts"]),
            np.nanmax(data["micro"]["norm_counts"]),
            1e-12,
        )
    )
    return data


def replot_figure_2f(data: dict) -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    data = _rebin_figure_2f(data)
    turbo = plt.get_cmap("turbo", 256)
    colors = turbo(np.linspace(0, 1, 256))
    colors[0] = np.array([1, 1, 1, 1])
    cmap = mcolors.ListedColormap(colors)
    vmax = float(data.get("vmax_all", 1.0) or 1.0)
    fig, axs = plt.subplots(1, 2, figsize=(3, 1.7), dpi=300, constrained_layout=True)
    for ax, key, rng, title, ticks in (
        (axs[0], "macro", tuple(data["macro_range"]), "Macro", data.get("macro_tick_list", [0, 0.25, 0.5])),
        (axs[1], "micro", tuple(data["micro_range"]), "Micro", data.get("micro_tick_list", [0, 0.05, 0.1])),
    ):
        hist = data[key]
        ax.pcolormesh(
            hist["xedges"], hist["yedges"], hist["norm_counts"].T,
            cmap=cmap, vmin=0, vmax=vmax, shading="flat",
        )
        ax.set_xlim(*rng)
        ax.set_ylim(*rng)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.plot([rng[0], rng[1]], [rng[0], rng[1]], ls="--", color="gray", lw=1)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Right max V [deg/ms]", fontsize=9)
        ax.set_ylabel("Left max V [deg/ms]", fontsize=9)
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _savefig(fig, "figure_2f.pdf")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
    cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
    plt.colorbar(sm, cax=cax, orientation="vertical")
    _savefig(fig_cbar, "figure_2f_colorbar.pdf")


def replot_figure_s3(data: dict) -> None:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("turbo")
    vmax = float(data.get("vmax", data.get("vmax_all", 1.0)) or 1.0)
    rng = tuple(data.get("range", data.get("macro_range", (0.0, 0.5))))
    ticks = data.get("ticks", [0.0, 0.25, 0.5])
    for name, hist in (
        ("figure_S3_head_still.pdf", data.get("hist_still") or data.get("still")),
        ("figure_S3_head_moving.pdf", data.get("hist_moving") or data.get("moving")),
    ):
        if not hist:
            continue
        fig, ax = plt.subplots(figsize=(1.7, 1.7), dpi=300)
        ax.pcolormesh(
            hist["xedges"], hist["yedges"], hist["norm_counts"].T,
            cmap=cmap, vmin=0, vmax=vmax, shading="flat",
        )
        ax.set_xlim(*rng)
        ax.set_ylim(*rng)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_box_aspect(1)
        _savefig(fig, name)
    fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
    cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, cax=cax, orientation="vertical")
    _savefig(fig_cbar, "figure_S3_colorbar.pdf")


def replot_jitter_histogram(data: dict) -> None:
    import matplotlib.pyplot as plt

    values = np.asarray(data.get("values", []), dtype=float)
    values = values[np.isfinite(values)]
    n_bins = int(data.get("n_bins", 15))
    xmax = float(data.get("xmax") or (np.nanpercentile(values, 99.5) if values.size else 1.0))
    bins = np.linspace(0, max(xmax, 1e-6), n_bins + 1)
    fig, ax = plt.subplots(figsize=(2, 1.6), dpi=150)
    if values.size:
        counts, edges = np.histogram(values, bins=bins)
        y = 100.0 * counts / counts.sum() if counts.sum() else counts
        ax.bar(edges[:-1], y, width=np.diff(edges), align="edge", color="gray", edgecolor="black")
    ax.set_xlim(0, xmax)
    ax.set_xlabel(f"Displacement [{data.get('units', 'um')}] (eye plane)", fontsize=10)
    ax.set_ylabel("% frames", fontsize=10)
    name = str(data.get("pdf_name", "histogram.pdf"))
    _savefig(fig, name)


def replot_unified_jitter(data: dict) -> None:
    import matplotlib.pyplot as plt

    pools = data.get("pools") or {}
    n_bins = int(data.get("n_bins", 15))
    xmax = float(data.get("xmax") or 1.0)
    units = str(data.get("units", "um"))
    panels = data.get("panels") or [
        ["rigid", "rigid lizard", "#D55E00"],
        ["modular", "modular lizard", "#0072B2"],
        ["mouse", "modular mouse", "#009E73"],
        ["turtle", "modular turtle", "#CC79A7"],
    ]
    bins = np.linspace(0, max(xmax, 1e-6), n_bins + 1)
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.8), dpi=150, sharex=True)
    for ax, panel in zip(axes.ravel(), panels):
        mount, label, color = panel[0], panel[1], panel[2]
        values = np.asarray(pools.get(mount, []), dtype=float)
        values = values[np.isfinite(values)]
        n = int(values.size)
        if values.size:
            counts, edges = np.histogram(values, bins=bins)
            y = 100.0 * counts / counts.sum() if counts.sum() else counts
            ax.bar(edges[:-1], y, width=np.diff(edges), align="edge", color=color, edgecolor="black", alpha=0.7)
        ax.set_xlim(0, xmax)
        ax.set_title(f"{label} (n={n})", fontsize=8)
        ax.set_ylabel("% frames", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ulabel = "µm" if units.lower() == "um" else units
    for ax in axes[1]:
        ax.set_xlabel(f"Displacement [{ulabel}] (eye plane)", fontsize=8)
    name = str(data.get("pdf_name", "unified_jitter_quantification.pdf"))
    _savefig(fig, name)


def replot_unified_noise(data: dict) -> None:
    import matplotlib.pyplot as plt

    pools = data.get("pools") or {}
    n_bins = int(data.get("n_bins", 60))
    xmax = float(data.get("xmax") or 1.0)
    xlabel = str(data.get("xlabel") or "Inter-frame Δangle [deg/frame]")
    ylabel = str(data.get("ylabel") or "% samples")
    panels = data.get("panels") or [
        ["rigid", "rigid lizard", "#D55E00"],
        ["modular", "modular lizard", "#0072B2"],
        ["turtle", "turtle", "#CC79A7"],
        ["mouse", "mouse", "#009E73"],
    ]
    bins = np.linspace(0, max(xmax, 1e-6), n_bins + 1)
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.8), dpi=150, sharex=True)
    for ax, panel in zip(axes.ravel(), panels):
        mount, label, color = panel[0], panel[1], panel[2]
        values = np.asarray(pools.get(mount, []), dtype=float)
        values = values[np.isfinite(values)]
        n = int(values.size)
        if values.size:
            counts, edges = np.histogram(values, bins=bins)
            y = 100.0 * counts / counts.sum() if counts.sum() else counts
            ax.bar(edges[:-1], y, width=np.diff(edges), align="edge", color=color, edgecolor="black", alpha=0.7)
        ax.set_xlim(0, xmax)
        ax.set_title(f"{label} (n={n})", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel(xlabel, fontsize=8)
    name = str(data.get("pdf_name", "unified_noise_measure.pdf"))
    _savefig(fig, name)


def replot_csv_lines() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    for csv in sorted(META.glob("*.csv")):
        df = pd.read_csv(csv)
        num = df.select_dtypes(include=["number"])
        if num.shape[1] < 2:
            continue
        x, y = num.iloc[:, 0], num.iloc[:, 1]
        fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
        ax.plot(x, y, color="#1f77b4", lw=1.2)
        ax.set_xlabel(num.columns[0], fontsize=8)
        ax.set_ylabel(num.columns[1], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _savefig(fig, csv.stem + ".pdf")


def replot_pos_vel(data: dict) -> None:
    import matplotlib.pyplot as plt

    t = np.asarray(data.get("t_grid", data.get("t_ms", [])), dtype=float)
    series = data.get("animals") or data.get("curves") or {}
    if not series and "vel_center" in data:
        series = {"all": data}
    fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
    for name, body in series.items():
        vel = np.asarray(body.get("vel_center", body.get("velocity", [])), dtype=float)
        if t.size and vel.size:
            n = min(t.size, vel.size)
            ax.plot(t[:n], vel[:n], lw=0.8, label=str(name))
    ax.set_xlabel("Time [ms]", fontsize=8)
    ax.set_ylabel("Speed", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(fig, "figure_2c.pdf")
    fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
    for name, body in series.items():
        pos = np.asarray(body.get("pos_center", body.get("position", [])), dtype=float)
        if t.size and pos.size:
            n = min(t.size, pos.size)
            ax.plot(t[:n], pos[:n], lw=0.8, label=str(name))
    ax.set_xlabel("Time [ms]", fontsize=8)
    ax.set_ylabel("Position [deg]", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(fig, "figure_2d.pdf")


def replot_species_trace(data: dict) -> None:
    import matplotlib.pyplot as plt

    t = np.asarray(data.get("t_s", []), dtype=float)
    duration = float(data.get("duration_s") or (float(t[-1]) if t.size else 80.0))
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 2.0), dpi=300, sharex=True)
    for ax, key, ylab in (
        (axes[0], "phi", "φ [deg]"),
        (axes[1], "theta", "θ [deg]"),
    ):
        body = data.get(key, {})
        if "L" in body:
            ax.plot(t, np.asarray(body["L"], dtype=float), color="#1f77b4", lw=0.8, label="Left")
        if "R" in body:
            ax.plot(t, np.asarray(body["R"], dtype=float), color="#d62728", lw=0.8, label="Right")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlim(0.0, duration)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("[s]", fontsize=8)
    _savefig(fig, str(data.get("pdf_name", "trace.pdf")))


def _epoch_duration_bins(vals: np.ndarray, scale: str, n_bins: int = 30) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    n_edges = int(n_bins) + 1
    if vals.size == 0:
        return np.logspace(-2, 1, n_edges) if scale == "log" else np.linspace(0.0, 1.0, n_edges)
    hi = float(np.nanmax(vals))
    if scale == "log":
        lo = max(float(np.nanmin(vals)), 1e-2)
        if hi <= lo:
            hi = lo * 10.0
        return np.logspace(np.log10(lo), np.log10(hi), n_edges)
    return np.linspace(0.0, hi if hi > 0 else 1.0, n_edges)


def replot_epoch_durations(data: dict) -> None:
    import matplotlib.pyplot as plt

    stored = data.get("bins") or {}
    n_bins = int(data.get("n_bins") or 30)
    xmin = data.get("linear_xmin_s")
    xmin_f = 0.0 if xmin is None else float(xmin)
    for lab, color in (("active", "#D55E00"), ("quiet", "#0072B2")):
        vals = np.asarray(data.get(lab, []), dtype=float)
        for scale in ("log", "linear"):
            fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
            body = stored.get(lab) if isinstance(stored, dict) else None
            bins = None
            if isinstance(body, dict) and scale in body:
                bins = np.asarray(body[scale], dtype=float)
            if bins is None or bins.size < 2:
                bins = _epoch_duration_bins(vals, scale, n_bins=n_bins)
            if vals.size:
                ax.hist(vals, bins=bins, color=color, edgecolor="black", alpha=0.8)
                if scale == "log":
                    ax.set_xscale("log")
                else:
                    ax.set_xlim(xmin_f, float(bins[-1]))
            ax.set_xlabel("Epoch duration [s]", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_title(f"{lab} n={vals.size} ({scale})", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _savefig(fig, f"epoch_durations_{lab}_{scale}.pdf")


def _replot_2e_axes(ax, xlim, ylim):
    ax.set_xlabel("Amplitude [deg]", fontsize=8)
    ax.set_ylabel("Peak V [deg/ms]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _replot_2e_means(per_animal_stats, global_fit, color_map, animal_order, params, xlim, ylim, title, name):
    import matplotlib.pyplot as plt

    figsize = tuple(params.get("figsize", (1.5, 1.7)))
    dpi = int(params.get("dpi", 300))
    lw = float(params.get("lw", 1.0))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for animal in animal_order:
        sdf = per_animal_stats.get(animal) if isinstance(per_animal_stats, dict) else None
        if sdf is None:
            continue
        try:
            empty = sdf.empty
        except AttributeError:
            continue
        if empty:
            continue
        x = (sdf["amp_lo"] + sdf["amp_hi"]) / 2
        ax.plot(x, sdf["mean_peak_v"], "o-", color=color_map.get(animal, "0.3"), ms=3, lw=lw, label=str(animal))
    slope = (global_fit or {}).get("slope")
    intercept = (global_fit or {}).get("intercept")
    if slope is not None and intercept is not None and np.isfinite(slope) and np.isfinite(intercept) and xlim is not None:
        xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
        ax.plot(xs, float(slope) * xs + float(intercept), "k--", lw=lw)
    if title:
        ax.set_title(title, fontsize=7)
    _replot_2e_axes(ax, xlim, ylim)
    slope = (global_fit or {}).get("slope")
    r = (global_fit or {}).get("r")
    if r is None or not np.isfinite(r):
        r2 = (global_fit or {}).get("R2")
        r = float(np.sqrt(r2)) if r2 is not None and np.isfinite(r2) else float("nan")
    if slope is not None and np.isfinite(slope) and np.isfinite(r):
        ax.text(
            0.04, 0.97, f"slope = {float(slope):.3f}\nPearson r = {float(r):.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6, color="0.1",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
        )
    fig.tight_layout()
    _savefig(fig, name)


def replot_figure_2e(data: dict) -> None:
    import matplotlib.pyplot as plt

    params = data.get("params") or {}
    xlim = data.get("xlim")
    ylim = data.get("ylim")
    if xlim is not None:
        xlim = (float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ylim = (float(ylim[0]), float(ylim[1]))
    color_map = data.get("color_map") or {}
    animal_order = list(data.get("animal_order") or (data.get("per_animal_stats") or {}).keys())
    classes = data.get("classes") or {}
    all_body = classes.get("all") or {
        "per_animal_stats": data.get("per_animal_stats") or {},
        "global_fit": data.get("global_fit") or {},
        "n": (data.get("global_fit") or {}).get("n"),
    }
    _replot_2e_means(
        all_body.get("per_animal_stats") or {},
        all_body.get("global_fit") or {},
        color_map,
        animal_order,
        params,
        xlim,
        ylim,
        None,
        "figure_2e_per_animal_means.pdf",
    )
    _replot_2e_means(
        all_body.get("per_animal_stats") or {},
        all_body.get("global_fit") or {},
        color_map,
        animal_order,
        params,
        xlim,
        ylim,
        None,
        "figure_2e.pdf",
    )
    for label, name in (
        ("concurrent", "figure_2e_per_animal_means_concurrent.pdf"),
        ("monocular", "figure_2e_per_animal_means_monocular.pdf"),
    ):
        body = classes.get(label) or {}
        n = body.get("n")
        title = f"{label} n={n}" if n is not None else label
        _replot_2e_means(
            body.get("per_animal_stats") or {},
            body.get("global_fit") or {},
            color_map,
            animal_order,
            params,
            xlim,
            ylim,
            title,
            name,
        )
    pooled = data.get("pooled") or {}
    amp = np.asarray(pooled.get("amp", data.get("amp", data.get("x", []))), dtype=float)
    vel = np.asarray(pooled.get("vel", data.get("vel", data.get("y", []))), dtype=float)
    n = int(min(amp.size, vel.size))
    amp, vel = amp[:n], vel[:n]
    figsize = tuple(params.get("figsize", (1.5, 1.7)))
    dens_figsize = tuple(params.get("density_figsize", (3.8, 3.2)))
    dpi = int(params.get("dpi", 300))
    lw = float(params.get("lw", 1.0))
    fit = all_body.get("global_fit") or data.get("global_fit") or {}
    cohort = pooled.get("cohort", "all")

    def _annot(ax, body, fs=9):
        sl = (body or {}).get("slope")
        rr = (body or {}).get("r")
        if rr is None or not np.isfinite(rr):
            r2 = (body or {}).get("R2")
            rr = float(np.sqrt(r2)) if r2 is not None and np.isfinite(r2) else float("nan")
        if sl is None or not np.isfinite(sl) or not np.isfinite(rr):
            return
        ax.text(
            0.04, 0.97, f"slope = {float(sl):.3f}\nPearson r = {float(rr):.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=fs, color="0.1",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
        )

    if n:
        fig, ax = plt.subplots(figsize=dens_figsize, dpi=dpi)
        ax.scatter(amp, vel, s=4, alpha=0.12, color="0.45", edgecolors="none", rasterized=True)
        if np.isfinite(fit.get("slope", np.nan)) and xlim is not None:
            xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
            ax.plot(xs, float(fit["slope"]) * xs + float(fit["intercept"]), "k--", lw=lw)
        ax.set_title(f"all animals ({cohort}) n={n}", fontsize=11)
        _replot_2e_axes(ax, xlim, ylim)
        ax.set_xlabel("Amplitude [deg]", fontsize=11)
        ax.set_ylabel("Peak V [deg/ms]", fontsize=11)
        ax.tick_params(labelsize=9)
        _annot(ax, fit, 9)
        fig.tight_layout()
        _savefig(fig, "figure_2e_all_animals_scatter.pdf")
        fig, ax = plt.subplots(figsize=dens_figsize, dpi=dpi)
        dens = data.get("density") or {}
        zi = dens.get("zi")
        extent = dens.get("extent")
        if extent is None and xlim and ylim:
            extent = (float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
        cmap = str(dens.get("cmap") or params.get("density_cmap", "turbo"))
        if zi is not None:
            im = ax.imshow(
                np.asarray(zi).T, extent=extent, origin="lower", cmap=cmap,
                aspect="auto", interpolation="bilinear",
            )
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Likelihood", fontsize=10)
            cb.ax.tick_params(labelsize=9)
        if np.isfinite(fit.get("slope", np.nan)) and xlim is not None:
            xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
            ax.plot(xs, float(fit["slope"]) * xs + float(fit["intercept"]), "k--", lw=max(lw, 1.2))
        ax.set_title(f"all animals ({cohort}) n={n}", fontsize=11)
        _replot_2e_axes(ax, xlim, ylim)
        ax.set_xlabel("Amplitude [deg]", fontsize=11)
        ax.set_ylabel("Peak V [deg/ms]", fontsize=11)
        ax.tick_params(labelsize=9)
        _annot(ax, fit, 9)
        fig.tight_layout()
        _savefig(fig, "figure_2e_all_animals_density.pdf")
    animals_p = np.asarray(pooled.get("animal", []), dtype=str)
    pairing = np.asarray(pooled.get("pairing", []), dtype=str)
    scatter_animal = params.get("scatter_animal")
    if scatter_animal is None and animals_p.size:
        scatter_animal = str(animals_p[0])
    if scatter_animal is not None and amp.size and animals_p.size == amp.size:
        animal_mask = animals_p == str(scatter_animal)
        class_specs = (
            ("all_events", animal_mask, "figure_2e_all_events_scatter.pdf"),
            ("concurrent", animal_mask & (pairing == "concurrent") if pairing.size == amp.size else animal_mask, "figure_2e_concurrent_scatter.pdf"),
            ("monocular", animal_mask & (pairing == "monocular") if pairing.size == amp.size else animal_mask, "figure_2e_monocular_scatter.pdf"),
        )
        for label, mask, name in class_specs:
            xa, ya = amp[mask], vel[mask]
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            if xa.size:
                ax.scatter(xa, ya, s=4, alpha=0.12, color="0.45", edgecolors="none", rasterized=True)
            title = f"{label} {scatter_animal} (empty)" if xa.size == 0 else f"{label} {scatter_animal} n={int(xa.size)}"
            if xa.size >= 3 and xlim is not None:
                sl = np.polyfit(xa, ya, 1)
                xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
                ax.plot(xs, sl[0] * xs + sl[1], "k--", lw=lw)
                r = np.corrcoef(xa, ya)[0, 1]
                title = f"{label} {scatter_animal} n={int(xa.size)}  R²={float(r*r):.2f}"
            ax.set_title(title, fontsize=7)
            _replot_2e_axes(ax, xlim, ylim)
            fig.tight_layout()
            _savefig(fig, name)


def replot_diagnostics_2e(data: dict) -> None:
    import matplotlib.pyplot as plt

    params = data.get("params") or {}
    amp = np.asarray(data.get("amp", []), dtype=float)
    vel = np.asarray(data.get("vel", []), dtype=float)
    length_bin = np.asarray(data.get("length_bin", []), dtype=object).astype(str)
    n = int(min(amp.size, vel.size, length_bin.size))
    amp, vel, length_bin = amp[:n], vel[:n], length_bin[:n]
    counts = data.get("counts") or {}
    colors = data.get("length_colors") or {
        "1": "#D55E00", "2": "#E69F00", "3": "#F0E442", "4": "#009E73",
        "5": "#0072B2", "6–8": "#56B4E9", "9+": "#000000",
    }
    order = list(data.get("length_draw_order") or ["9+", "6–8", "5", "4", "3", "2", "1"])
    xlim = data.get("xlim") or (0.0, 1.0)
    ylim = data.get("ylim") or (0.0, 0.5)
    figsize = tuple(params.get("figsize", (5.2, 4.2)))
    dpi = int(params.get("dpi", 300))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for label in order:
        mask = length_bin == str(label)
        if not np.any(mask):
            continue
        ax.scatter(
            amp[mask], vel[mask], s=6, alpha=0.35, color=colors.get(label, "0.4"),
            edgecolors="none", rasterized=True,
            label=f"{label}  n={counts.get(label, int(mask.sum()))}",
        )
    x0, x1 = float(xlim[0]), float(xlim[1])
    xs = np.linspace(max(x0, 0.0), x1, 80)
    one_slope = float(params.get("one_frame_slope") or (60.0 / 1000.0))
    thr = float(params.get("speed_threshold_deg_per_ms") or 0.048)
    ax.plot(xs, one_slope * xs, color="0.15", lw=1.4, ls="-", label="1-frame  V=amp/16.7 ms")
    ax.axhline(thr, color="0.15", lw=1.2, ls=":", label=f"detect floor  {thr:.3f} deg/ms")
    slope, intercept = data.get("slope"), data.get("intercept")
    if slope is not None and intercept is not None and np.isfinite(slope) and np.isfinite(intercept):
        ax.plot(xs, float(slope) * xs + float(intercept), "k--", lw=1.2, label=f"OLS  slope={float(slope):.3f}")
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_xlabel("Amplitude [deg]", fontsize=11)
    ax.set_ylabel("Peak V [deg/ms]", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cohort = params.get("cohort", "all")
    ax.set_title(f"all animals ({cohort}) n={n}  colored by length", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=1.6)
    fig.tight_layout()
    _savefig(fig, "figure_2e_scatter_by_length.pdf")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data, pkl = _load_pickle()
    kind = KIND
    if kind == "figure_2f" and data is not None:
        replot_figure_2f(data)
    elif kind == "figure_s3" and data is not None:
        replot_figure_s3(data)
    elif kind == "jitter_histogram" and data is not None:
        replot_jitter_histogram(data)
    elif kind == "yield_histogram" and data is not None:
        replot_jitter_histogram(data)
    elif kind == "pos_vel" and data is not None:
        replot_pos_vel(data)
    elif kind == "species_trace" and data is not None:
        replot_species_trace(data)
    elif kind == "unified_jitter" and data is not None:
        replot_unified_jitter(data)
    elif kind == "raw_interframe_delta" and data is not None:
        import matplotlib.pyplot as plt

        pools = data.get("pools") or {}
        n_bins = int(data.get("n_bins", 50))
        fits = data.get("fits") or {}
        panels = data.get("panels") or [
            ["rigid", "rigid lizard", "#D55E00"],
            ["modular", "modular lizard", "#0072B2"],
            ["turtle", "turtle", "#CC79A7"],
            ["mouse", "mouse", "#009E73"],
        ]
        fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.0), dpi=150)
        for ax, panel in zip(axes.ravel(), panels):
            mount, label, color = panel[0], panel[1], panel[2]
            values = np.asarray(pools.get(mount, []), dtype=float)
            values = values[np.isfinite(values) & (values >= 0)]
            n = int(values.size)
            hi = float(values.max()) if n else 1.0
            if n:
                ax.hist(values, bins=n_bins, range=(0, hi), density=True, color=color, edgecolor="0.3", alpha=0.7)
                scale = (fits.get(mount) or {}).get("B")
                if scale is not None and np.isfinite(float(scale)) and float(scale) > 0:
                    xs = np.linspace(0, hi, 400)
                    b = float(scale)
                    ax.plot(xs, (xs / (b * b)) * np.exp(-0.5 * (xs / b) ** 2), color="0.15", lw=1.5)
                    ax.set_title(f"{label}  B={b:.3f}  n={n}", fontsize=8)
                else:
                    ax.set_title(f"{label} (n={n})", fontsize=8)
            ax.set_xlim(0, hi)
            ax.set_xlabel("Inter-frame Δangle [deg/frame]", fontsize=8)
            ax.set_ylabel("Probability density", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        _savefig(fig, str(data.get("pdf_name", "raw_interframe_delta.pdf")))
        logx_name = data.get("logx_pdf_name")
        if logx_name:
            xmax = float(data.get("logx_xmax") or 1.0)
            lo = 1e-3
            fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.0), dpi=150, sharex=True)
            bins = np.logspace(np.log10(lo), np.log10(max(xmax, lo * 10)), n_bins + 1)
            for ax, panel in zip(axes.ravel(), panels):
                mount, label, color = panel[0], panel[1], panel[2]
                values = np.asarray(pools.get(mount, []), dtype=float)
                values = values[np.isfinite(values)]
                n = int(values.size)
                if values.size:
                    counts, edges = np.histogram(values, bins=bins)
                    y = 100.0 * counts / counts.sum() if counts.sum() else counts
                    ax.bar(edges[:-1], y, width=np.diff(edges), align="edge", color=color, edgecolor="black", alpha=0.7)
                ax.set_xscale("log")
                ax.set_xlim(lo, xmax)
                ax.set_title(f"{label} (n={n})", fontsize=8)
                ax.set_ylabel("% samples", fontsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            for ax in axes[1]:
                ax.set_xlabel("Inter-frame Δangle [deg/frame]", fontsize=8)
            _savefig(fig, str(logx_name))
    elif kind == "unified_noise" and data is not None:
        replot_unified_noise(data)
    elif kind == "epoch_durations" and data is not None:
        replot_epoch_durations(data)
    elif kind == "figure_2e" and data is not None:
        replot_figure_2e(data)
    elif kind == "diagnostics_2e" and data is not None:
        replot_diagnostics_2e(data)
    elif kind in {"robustness", "s1_occupancy", "corrective_head", "ISI_by_state", "qc_reversals_noise_blinks", "figure_2g", "isi_hist"}:
        replot_csv_lines()
        if data is None:
            print(f"replot kind={kind}: drew CSV panels (pickle optional)")
    elif data is None:
        print("No pickle/CSV found in metadata/; nothing to replot.")
        return 1
    else:
        print(f"kind={kind} pickle={None if pkl is None else pkl.name}: generic CSV/line fallback")
        replot_csv_lines()
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
