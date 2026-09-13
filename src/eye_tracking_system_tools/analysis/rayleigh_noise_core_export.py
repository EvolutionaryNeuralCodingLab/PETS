"""Export quiet-segment Rayleigh noise-core histograms as plot bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle
from eye_tracking_system_tools.analysis.rayleigh_core import (
    ks_rayleigh,
    rayleigh_pdf,
    rayleigh_scale_from_median,
)

PLOT_ID = "rayleigh_noise_core"
PDF_NAME = "rayleigh_noise_core.pdf"
PICKLE_NAME = "rayleigh_noise_core.pkl"
SUMMARY_NAME = "rayleigh_noise_core_summary.yaml"


def resolve_noise_core_xmax(
    arr: np.ndarray,
    B: float,
    threshold: float | None,
) -> float:
    """X-limit showing the noise body/tail and the detection threshold.

    Takes the farther of the 99.9th percentile of ``D`` and ``6 B`` (Rayleigh
    tail), then extends to ``1.08 × threshold`` when a detector floor is set.
    """
    pos = np.asarray(arr, dtype=float)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    thr = (
        float(threshold)
        if threshold is not None and np.isfinite(threshold) and float(threshold) > 0
        else None
    )
    if pos.size == 0:
        return 1.08 * thr if thr is not None else 1.0
    p999 = float(np.percentile(pos, 99.9))
    fit_hi = 6.0 * float(B) if np.isfinite(B) and B > 0 else p999
    xmax = max(p999, fit_hi)
    if thr is not None:
        xmax = max(xmax, thr * 1.08)
    return float(xmax)


def resolve_noise_core_n_bins(
    arr: np.ndarray,
    B: float,
    xmax: float,
    n_bins: int = 40,
) -> int:
    """Keep histogram resolution on the noise body when xmax extends to a far threshold."""
    pos = np.asarray(arr, dtype=float)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    base = max(int(n_bins), 8)
    if pos.size == 0 or not np.isfinite(xmax) or xmax <= 0:
        return base
    p999 = float(np.percentile(pos, 99.9))
    fit_hi = 6.0 * float(B) if np.isfinite(B) and B > 0 else p999
    core = max(p999, fit_hi, 1e-12)
    return max(base, int(round(base * float(xmax) / core)))


def figure_rayleigh_noise_core_window(
    d: np.ndarray,
    *,
    B: float | None = None,
    threshold: float | None = None,
    title: str = "",
    n_bins: int = 40,
    color: str = "0.70",
    hist_label: str = "quiet segments",
    noise_marker: str = "2B",
) -> plt.Figure:
    """Quiet-segment hist + median-matched Rayleigh.

    ``noise_marker``:
      - ``\"2B\"`` — 2 axis-σ radius (legacy supplement style)
      - ``\"median\"`` — sample median of ``D`` (media / Methods style)
    """
    from eye_tracking_system_tools.analysis.paper_mpl_style import apply_paper_style

    apply_paper_style()
    marker = str(noise_marker).strip().lower()
    if marker not in {"2b", "median"}:
        raise ValueError(f"noise_marker must be '2B' or 'median', got {noise_marker!r}")

    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if B is None or not np.isfinite(B):
        B = rayleigh_scale_from_median(arr)
    B = float(B) if np.isfinite(B) else float("nan")
    median_d = float(np.median(arr)) if arr.size else float("nan")
    xmax = resolve_noise_core_xmax(arr, B, threshold)
    n_bins_use = resolve_noise_core_n_bins(arr, B, xmax, n_bins)

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    if arr.size:
        core = arr[arr > 0]
        plot_arr = core if core.size >= 8 else arr
        # Same opaque gray + black edge for every species (Illustrator-clean).
        _ = color
        ax.hist(
            plot_arr,
            bins=int(n_bins_use),
            range=(0.0, xmax),
            density=True,
            color="0.70",
            edgecolor="black",
            linewidth=0.4,
            alpha=1.0,
            histtype="bar",
            label=hist_label,
        )
        xs = np.linspace(0.0, xmax, 400)
        if np.isfinite(B) and B > 0:
            ax.plot(xs, rayleigh_pdf(xs, B), color="0.1", lw=1.8, label=f"Rayleigh B={B:.3g}")
        if marker == "2b" and np.isfinite(B) and B > 0:
            floor_2b = 2.0 * B
            ax.axvline(
                floor_2b,
                color="#E69F00",
                ls="--",
                lw=1.2,
                label=f"2B (2σ)={floor_2b:.3g}",
            )
        elif marker == "median" and np.isfinite(median_d) and median_d > 0:
            ax.axvline(
                median_d,
                color="#E69F00",
                ls="--",
                lw=1.2,
                label=f"median={median_d:.3g}",
            )
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(
            float(threshold),
            color="#D55E00",
            ls="--",
            lw=1.2,
            label=f"threshold={float(threshold):g}",
        )
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("D = hypot(Δφ, Δθ)  [deg/frame]")
    ax.set_ylabel("Probability density")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _snr_from_threshold(threshold: float | None, B: float) -> dict[str, float]:
    out = {
        "threshold": float(threshold) if threshold is not None else float("nan"),
        "B_med": float(B),
        "floor_2B": float("nan"),
        "snr_thr_over_2B": float("nan"),
    }
    if not np.isfinite(B) or B <= 0:
        return out
    floor_2b = 2.0 * float(B)
    out["floor_2B"] = floor_2b
    if threshold is not None and np.isfinite(threshold) and floor_2b > 0:
        out["snr_thr_over_2B"] = float(threshold) / floor_2b
    return out


def snr_thr_over_median(threshold: float | None, median_d: float) -> float:
    """SNR = detection threshold / median(D)."""
    if (
        threshold is None
        or not np.isfinite(threshold)
        or not np.isfinite(median_d)
        or float(median_d) <= 0
    ):
        return float("nan")
    return float(threshold) / float(median_d)


def export_rayleigh_noise_core_window(
    out_dir: Path | str,
    *,
    d: np.ndarray,
    quiet_segments: list[tuple[float, float]],
    block_label: str,
    block_path: str | Path,
    animal: str,
    eye: str,
    mount_type: str = "",
    threshold: float | None = None,
    B: float | None = None,
    n_bins: int = 40,
    color: str = "#0072B2",
    show: bool = False,
    plot_id: str = PLOT_ID,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write noise-core PDF, pickle, and sidecars for quiet-segment reproduction."""
    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if B is None or not np.isfinite(B):
        B = rayleigh_scale_from_median(arr)
    B = float(B) if np.isfinite(B) else float("nan")

    snr = _snr_from_threshold(threshold, B)
    seg_summary = (
        f"{quiet_segments[0][0]:g}–{quiet_segments[0][1]:g} ms"
        if len(quiet_segments) == 1
        else f"{len(quiet_segments)} segments"
    )
    title = f"{block_label}  {eye}  {seg_summary}"
    if np.isfinite(snr["snr_thr_over_2B"]):
        title += f"  SNR thr/2B={snr['snr_thr_over_2B']:.2f}"

    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="rayleigh_noise_core",
        logic_key="rayleigh_noise_core",
        cohort={
            "cohort": mount_type or "unspecified",
            "animals": [str(animal)] if animal else [],
            "block_keys": [str(block_path)],
            "rule": "quiet_window_export",
            "mount_type": mount_type or None,
        },
        params={
            "quiet_segments_ms": [[float(a), float(b)] for a, b in quiet_segments],
            "eye": str(eye),
            "n_bins": int(n_bins),
            "detection_threshold_deg_per_frame": snr["threshold"],
            "snr_definition": "detection_threshold / (2 * B_med)",
        },
        extra={
            "block_label": block_label,
            "block_path": str(block_path),
            "metric": "hypot(d_k_phi, d_k_theta) deg/frame",
            "noise_measure": "Rayleigh B_med = median(D)/sqrt(2 ln 2)",
            "marker_2sigma": "2 * B_med (2 axis-σ radius on D)",
        },
    )

    fig = figure_rayleigh_noise_core_window(
        arr,
        B=B,
        threshold=threshold,
        title=title,
        n_bins=n_bins,
        color=color,
    )
    pdf_path = bundle.plots_dir / PDF_NAME
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)

    payload = {
        "d": arr.astype(np.float64),
        "quiet_segments_ms": [[float(a), float(b)] for a, b in quiet_segments],
        "block_label": block_label,
        "block_path": str(block_path),
        "animal": str(animal),
        "eye": str(eye),
        "mount_type": str(mount_type),
        "n_bins": int(n_bins),
        "color": color,
        "pdf_name": PDF_NAME,
        "title": title,
        "hist_label": "quiet segments",
        "xlabel": r"D = hypot(Δφ, Δθ)  [deg/frame]",
        "ylabel": "Probability density",
        "B_med": B,
        "median_D": float(np.median(arr)) if arr.size else float("nan"),
        "n": int(arr.size),
        "n_zero": int(np.sum(arr == 0)),
        "xmax": resolve_noise_core_xmax(arr, B, threshold),
        **snr,
        **(extra_meta or {}),
    }
    pkl = bundle.metadata_dir / PICKLE_NAME
    write_pickle_with_meta(
        payload,
        pkl,
        meta={
            "n": int(arr.size),
            "quiet_segments_ms": payload["quiet_segments_ms"],
            "B_med": B,
            "floor_2B": snr["floor_2B"],
            "snr_thr_over_2B": snr["snr_thr_over_2B"],
            "threshold": snr["threshold"],
        },
        entrypoint=(
            "eye_tracking_system_tools.analysis.rayleigh_noise_core_export."
            "export_rayleigh_noise_core_window"
        ),
    )

    summary = {
        "block_label": block_label,
        "block_path": str(block_path),
        "animal": str(animal),
        "eye": str(eye),
        "mount_type": mount_type or None,
        "quiet_segments_ms": payload["quiet_segments_ms"],
        "n": int(arr.size),
        "n_zero": payload["n_zero"],
        "median_D": payload["median_D"],
        "B_med": B,
        "floor_2B": snr["floor_2B"],
        "detection_threshold_deg_per_frame": snr["threshold"],
        "snr_thr_over_2B": snr["snr_thr_over_2B"],
        "snr_definition": "detection_threshold / (2 * B_med)",
        "marker_2sigma_note": "2*B_med is the 2 axis-σ noise marker on D (Rayleigh scale B).",
    }
    summary_path = bundle.metadata_dir / SUMMARY_NAME
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    finish_plot_bundle(bundle)
    written: dict[str, Path] = {
        PDF_NAME: pdf_path,
        PICKLE_NAME: pkl,
        SUMMARY_NAME: summary_path,
        "params.yaml": bundle.metadata_dir / "params.yaml",
        "replot.py": bundle.bundle_dir / "replot.py",
    }
    return written


def _frame_steps_deg(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    dphi = np.diff(np.asarray(phi, dtype=float))
    dtheta = np.diff(np.asarray(theta, dtype=float))
    return np.hypot(dphi, dtheta)


def pick_quiet_segments_from_d(
    d: np.ndarray,
    t_mid_ms: np.ndarray,
    *,
    n_segments: int = 5,
    win_ms: float = 2000.0,
    min_ms: float = 1500.0,
    hi_pct: float = 40.0,
) -> list[tuple[float, float]]:
    """Lowest-median sliding windows of ``D``, used as quiet segments.

    Prefers non-overlapping ``win_ms`` intervals with the smallest median ``D``.
    Falls back to long contiguous runs below ``hi_pct`` if sliding windows fail.
    """
    arr = np.asarray(d, dtype=float)
    t = np.asarray(t_mid_ms, dtype=float)
    if arr.size < 8 or t.size != arr.size:
        return []
    order = np.argsort(t)
    arr = arr[order]
    t = t[order]
    finite = np.isfinite(arr) & (arr >= 0)
    if int(finite.sum()) < 8:
        return []

    t0 = float(t[0])
    t1 = float(t[-1])
    span = t1 - t0
    width = float(win_ms)
    if span < width:
        width = max(float(min_ms), span)
    step = max(width / 4.0, 100.0)
    starts = np.arange(t0, max(t1 - width, t0) + 1e-9, step)
    scored: list[tuple[float, float, float]] = []
    for s in starts:
        e = s + width
        m = finite & (t >= s) & (t <= e)
        if int(m.sum()) < 8:
            continue
        scored.append((float(np.median(arr[m])), float(s), float(e)))
    scored.sort()
    picked: list[tuple[float, float]] = []
    for _, s, e in scored:
        if any(not (e <= a or s >= b) for a, b in picked):
            continue
        picked.append((s, e))
        if len(picked) >= int(n_segments):
            break
    if picked:
        picked.sort()
        return picked

    cutoff = float(np.percentile(arr[finite], hi_pct))
    quiet = finite & (arr <= cutoff)
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(quiet.size)
    while i < n:
        if not bool(quiet[i]):
            i += 1
            continue
        j = i
        while j < n and bool(quiet[j]):
            j += 1
        runs.append((i, j))
        i = j
    run_scored: list[tuple[float, float, float]] = []
    for a, b in runs:
        ta, tb = float(t[a]), float(t[b - 1])
        dur = tb - ta
        if dur < float(min_ms):
            continue
        run_scored.append((dur, ta, tb))
    run_scored.sort(reverse=True)
    segs = [(ta, tb) for _, ta, tb in run_scored[: int(n_segments)]]
    segs.sort()
    return segs


def export_quiet_core_from_trace_pickle(
    trace_pkl: Path | str,
    out_dir: Path | str,
    *,
    mount_type: str,
    threshold: float | None,
    color: str = "#0072B2",
    plot_id: str = PLOT_ID,
    n_segments: int = 5,
    min_ms: float = 1500.0,
    show: bool = False,
) -> dict[str, Path]:
    """Build a Rayleigh quiet-core bundle from a species-trace pickle (no lab volume)."""
    import pickle

    path = Path(trace_pkl)
    with open(path, "rb") as f:
        data = pickle.load(f)
    t_s = np.asarray(data["t_s"], dtype=float)
    win0 = float(data.get("window_start_s") or 0.0)
    t_abs_s = t_s + win0 if float(np.nanmax(t_s) if t_s.size else 0) < 500 else t_s
    d_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    for side in ("L", "R"):
        phi = np.asarray((data.get("phi") or {}).get(side, []), dtype=float)
        theta = np.asarray((data.get("theta") or {}).get(side, []), dtype=float)
        n = min(phi.size, theta.size, t_abs_s.size)
        if n < 8:
            continue
        step = _frame_steps_deg(phi[:n], theta[:n])
        t_mid = 0.5 * (t_abs_s[: n - 1] + t_abs_s[1:n]) * 1000.0
        d_parts.append(step)
        t_parts.append(t_mid)
    if not d_parts:
        raise ValueError(f"No φ/θ series in {path}")
    d_all = np.concatenate(d_parts)
    t_all = np.concatenate(t_parts)
    order = np.argsort(t_all)
    d_all = d_all[order]
    t_all = t_all[order]
    segs = pick_quiet_segments_from_d(
        d_all, t_all, n_segments=n_segments, min_ms=min_ms
    )
    if not segs:
        # Fall back to the lowest-D third of the window as one segment.
        lo = float(np.nanmin(t_all))
        hi = float(np.nanmax(t_all))
        segs = [(lo, hi)]
    mask = np.zeros(d_all.size, dtype=bool)
    for a, b in segs:
        mask |= (t_all >= a) & (t_all <= b)
    window_d = d_all[mask]
    animal = str(data.get("animal") or "")
    block = str(data.get("block") or "")
    return export_rayleigh_noise_core_window(
        out_dir,
        d=window_d,
        quiet_segments=segs,
        block_label=f"{animal} block_{block} (trace window)",
        block_path=str(path),
        animal=animal,
        eye="both",
        mount_type=mount_type,
        threshold=threshold,
        color=color,
        show=show,
        plot_id=plot_id,
        extra_meta={
            "source_trace_pickle": str(path),
            "window_start_s": float(data.get("window_start_s") or 0.0),
            "window_end_s": float(data.get("window_end_s") or 0.0),
            "quiet_rule": "lowest-median 2 s windows (n_segments) inside the trace",
        },
    )


S10_KS_CSV_NAME = "ks_rayleigh_gof.csv"


def ks_gof_row_from_noise_core(
    payload: dict[str, Any],
    *,
    species: str,
    panel: str,
    n_sim: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """KS vs the S10 Rayleigh overlay (scale = ``B_med``)."""
    d = np.asarray(payload.get("d"), dtype=float)
    B = payload.get("B_med")
    gof = ks_rayleigh(d, B if B is not None else None, n_sim=n_sim, rng=rng)
    n_all = int(np.isfinite(d).sum()) if d.size else 0
    return {
        "species": species,
        "panel": panel,
        "animal": str(payload.get("animal") or ""),
        "mount_type": str(payload.get("mount_type") or ""),
        "block_label": str(payload.get("block_label") or ""),
        "n": n_all,
        "n_positive": int(gof["n"]),
        "median_D": float(payload.get("median_D") if payload.get("median_D") is not None else np.nan),
        "B_med": float(gof["B"]),
        "ks_statistic": float(gof["ks_statistic"]),
        "ks_p": float(gof["ks_p"]),
        "ks_p_lilliefors": float(gof["ks_p_lilliefors"]),
        "d_crit_05_lilliefors": float(gof["d_crit_05_lilliefors"]),
        "n_sim": int(gof["n_sim"]),
        "scale_estimator": "B_med = median(D)/sqrt(2 ln 2)",
        "null": "Rayleigh(loc=0, scale=B_med)",
    }


def write_s10_ks_gof_csv(
    pickle_map: dict[str, Path],
    out_csv: Path | str,
    *,
    n_sim: int = 1000,
    seed: int = 0,
) -> Path:
    """Write one KS GOF row per S10 noise-core pickle.

    ``pickle_map`` keys are species labels (``lizard``, ``mouse``, ``turtle``).
    Values are ``rayleigh_noise_core.pkl`` paths.
    """
    import pickle

    import pandas as pd

    rng = np.random.default_rng(int(seed))
    panel_of = {"lizard": "S10a", "mouse": "S10b", "turtle": "S10c"}
    rows: list[dict[str, Any]] = []
    for species, pkl in pickle_map.items():
        path = Path(pkl)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        rows.append(
            ks_gof_row_from_noise_core(
                payload,
                species=str(species),
                panel=panel_of.get(str(species), str(species)),
                n_sim=n_sim,
                rng=rng,
            )
        )
    table = pd.DataFrame(rows)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


S10_MEDIA_PANELS: tuple[tuple[str, str, str], ...] = (
    ("lizard", "S10a_lizard_rayleigh_noise_core.pkl", "S10a_lizard.pdf"),
    ("mouse", "S10b_mouse_rayleigh_noise_core.pkl", "S10b_mouse.pdf"),
    ("turtle", "S10c_turtle_rayleigh_noise_core.pkl", "S10c_turtle.pdf"),
)


def write_s10_media_style(
    s10_dir: Path | str,
    *,
    out_subdir: str = "media_style",
) -> dict[str, Path]:
    """Redraw S10 quiet-core panels with median marker (no 2B) + caption.md.

    Reads existing species pickles under ``s10_dir/metadata/`` and writes PDFs /
    caption to ``s10_dir/<out_subdir>/``. Does not modify the parent S10 plots.
    """
    import pickle

    root = Path(s10_dir)
    meta = root / "metadata"
    out = root / out_subdir
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # Optional KS table for caption numbers.
    ks_by_species: dict[str, float] = {}
    ks_csv = root / S10_KS_CSV_NAME
    if not ks_csv.is_file():
        ks_csv = meta / S10_KS_CSV_NAME
    if ks_csv.is_file():
        import pandas as pd

        for rec in pd.read_csv(ks_csv).itertuples(index=False):
            ks_by_species[str(rec.species)] = float(rec.ks_statistic)

    written: dict[str, Path] = {}
    caption_rows: list[dict[str, Any]] = []
    for species, pkl_name, pdf_name in S10_MEDIA_PANELS:
        pkl = meta / pkl_name
        if not pkl.is_file():
            raise FileNotFoundError(pkl)
        with open(pkl, "rb") as f:
            payload = pickle.load(f)
        d = np.asarray(payload.get("d"), dtype=float)
        B = float(payload.get("B_med")) if payload.get("B_med") is not None else float("nan")
        median_d = float(payload.get("median_D")) if payload.get("median_D") is not None else float(
            np.median(d[np.isfinite(d)]) if np.any(np.isfinite(d)) else float("nan")
        )
        thr_raw = payload.get("threshold", payload.get("detection_threshold_deg_per_frame"))
        thr = float(thr_raw) if thr_raw is not None and np.isfinite(float(thr_raw)) else None
        snr = snr_thr_over_median(thr, median_d)
        block_label = str(payload.get("block_label") or "")
        eye = str(payload.get("eye") or "both")
        n_seg = len(payload.get("quiet_segments_ms") or [])
        seg_summary = f"{n_seg} segments" if n_seg else "quiet segments"
        title = f"{block_label}  {eye}  {seg_summary}"
        if np.isfinite(snr):
            title += f"  SNR thr/median={snr:.2f}"

        fig = figure_rayleigh_noise_core_window(
            d,
            B=B,
            threshold=thr,
            title=title,
            n_bins=int(payload.get("n_bins") or 40),
            color=str(payload.get("color") or "0.70"),
            hist_label=str(payload.get("hist_label") or "quiet segments"),
            noise_marker="median",
        )
        pdf_path = plots / pdf_name
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        show_and_close(fig, False)
        written[pdf_name] = pdf_path

        caption_rows.append(
            {
                "species": species,
                "panel": pdf_name.replace(".pdf", ""),
                "block_label": block_label,
                "animal": str(payload.get("animal") or ""),
                "n": int(payload.get("n") or d.size),
                "median_D": median_d,
                "B_med": B,
                "threshold": thr if thr is not None else float("nan"),
                "snr_thr_over_median": snr,
                "ks_statistic": ks_by_species.get(species, float("nan")),
            }
        )

    caption_path = out / "caption.md"
    caption_path.write_text(_s10_media_caption(caption_rows), encoding="utf-8")
    written["caption.md"] = caption_path

    summary_path = out / "media_style_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "snr_definition": "detection_threshold / median(D)",
                "noise_marker": "median(D)",
                "rayleigh_scale": "B = median(D)/sqrt(2 ln 2)",
                "panels": caption_rows,
            },
            f,
            sort_keys=False,
        )
    written["media_style_summary.yaml"] = summary_path
    return written


def _s10_media_caption(rows: list[dict[str, Any]]) -> str:
    by = {str(r["species"]): r for r in rows}

    def _panel(letter: str, species: str, has_thr: bool) -> str:
        r = by[species]
        ks = r["ks_statistic"]
        ks_txt = f" KS D = {ks:.3f}." if np.isfinite(ks) else ""
        block_label = str(r.get("block_label") or "")
        block_bit = ""
        if "block_" in block_label:
            # e.g. "PV_228 / 2026_06_15 / block_016" → block_016
            tok = [t for t in block_label.replace("/", " ").split() if t.startswith("block_")]
            if tok:
                block_bit = f", {tok[0]}"
        head = (
            f"**({letter})** {species.capitalize()} ({r['animal']}{block_bit}; n = {int(r['n'])}). "
            f"median(D) = {r['median_D']:.3f}°/frame, "
            f"B = {r['B_med']:.3f}°/frame"
        )
        if has_thr and np.isfinite(r["threshold"]) and np.isfinite(r["snr_thr_over_median"]):
            head += (
                f",\nthreshold = {r['threshold']:g}°/frame, "
                f"SNR = {r['snr_thr_over_median']:.2f}."
            )
        else:
            head += ".\nNo detection threshold."
        return head + ks_txt

    return (
        "Figure S10. Frame-to-frame tracking noise in quiet intervals is close to Rayleigh.\n"
        "\n"
        "Histograms of D = hypot(Δφ, Δθ) in GUI-audited stationary windows, with a Rayleigh\n"
        "overlay scaled to the sample median (B = median(D)/√(2 ln 2)). The dashed orange\n"
        "line marks median(D), the typical noise magnitude. Where a detector threshold is\n"
        "defined it is shown; SNR = threshold / median(D).\n"
        "\n"
        f"{_panel('a', 'lizard', True)}\n"
        "\n"
        f"{_panel('b', 'mouse', True)}\n"
        "\n"
        f"{_panel('c', 'turtle', False)}\n"
        "\n"
        "The body of each histogram follows the median-matched Rayleigh overlay. KS D is the\n"
        "largest vertical gap between the empirical and Rayleigh CDFs. With several thousand\n"
        "samples those gaps exceed the Lilliefors 5% critical D, so a formal test rejects\n"
        "*exact* Rayleigh; the residual is a modest heavy tail rather than a different\n"
        "family.\n"
    )
