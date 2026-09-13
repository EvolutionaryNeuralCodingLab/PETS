"""Back-and-forth sequences during head movement only.

Reviewer Q6 overhaul: report how often the Video S1 double-step (small
back-and-forth after a large saccade) occurs **during annotated head movement**,
and treat that rate as nystagmus-like gaze stabilization. Head-stationary
saccades are dropped before the large-saccade threshold is set; there is no
still-vs-moving comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.double_steps_rev_verbatim import (
    POST_WINDOW_MS,
    collect_double_steps,
)
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.movement_associated_nystagmus import (
    AMP_FLOOR_DEG,
    AMP_PERCENTILE,
    MOVING_COLOR,
    OVERALL_COLOR,
    REVERSE_MIN_DEG,
    _bar_with_n,
    _drop_constituents,
    _rate,
    _sync_diff_ms,
    figure_amplitude_distribution,
)
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle
from eye_tracking_system_tools.analysis.secondary_after_saccade import propose_amplitude_threshold

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

PLOT_ID = "move_only"
KIND = "double_steps_move_only"
DEFAULT_AMP_THRESHOLD_DEG = 7.0

PDF_AMP = "amplitude_distribution.pdf"
PDF_OVERALL = "back_and_forth_overall.pdf"
PDF_BY_ANIMAL = "back_and_forth_by_animal.pdf"
EVENTS_CSV = "large_saccades.csv"
DIAG_CSV = "block_diagnostics.csv"
SUMMARY_YAML = "summary.yaml"
CAPTIONS_MD = "captions.md"
PICKLE_NAME = "double_steps_move_only.pkl"


def collect_double_steps_move_only(
    tables: EventTables,
    *,
    post_window_ms: float = POST_WINDOW_MS,
    reverse_min_deg: float = REVERSE_MIN_DEG,
    amp_percentile: float = AMP_PERCENTILE,
    amp_floor_deg: float = AMP_FLOOR_DEG,
    amp_threshold_deg: float | None = DEFAULT_AMP_THRESHOLD_DEG,
    sync_diff_ms: float | None = None,
) -> dict[str, Any]:
    payload = collect_double_steps(
        tables,
        post_window_ms=post_window_ms,
        reverse_min_deg=reverse_min_deg,
        amp_percentile=amp_percentile,
        amp_floor_deg=amp_floor_deg,
        amp_threshold_deg=amp_threshold_deg,
        sync_diff_ms=sync_diff_ms,
    )
    all_unique = payload.get("all_unique")
    if all_unique is None or all_unique.empty or "head_movement" not in all_unique.columns:
        moving = all_unique.copy() if all_unique is not None else pd.DataFrame()
        n_dropped = 0
    else:
        flag = all_unique["head_movement"].astype(bool)
        n_dropped = int((~flag).sum())
        moving = all_unique.loc[flag].copy().reset_index(drop=True)
    amps = (
        pd.to_numeric(moving["amp_deg"], errors="coerce").to_numpy(float)
        if not moving.empty
        else np.asarray([], dtype=float)
    )
    proposed, amp_stats = propose_amplitude_threshold(
        amps, percentile=amp_percentile, floor_deg=amp_floor_deg
    )
    threshold = float(amp_threshold_deg) if amp_threshold_deg is not None else proposed
    amp_stats["threshold_deg"] = float(threshold)
    amp_stats["threshold_source"] = "user" if amp_threshold_deg is not None else "proposed_head_moving"
    if moving.empty:
        large = moving.copy()
    else:
        large = moving.loc[pd.to_numeric(moving["amp_deg"], errors="coerce") >= threshold].copy()
        large = large.reset_index(drop=True)
    print(
        f"[double_steps_move_only] unique_moving={len(moving)}  "
        f"dropped_still={n_dropped}  threshold={threshold:.1f}°  large={len(large)}"
    )
    params = dict(payload.get("params") or {})
    params["head_filter"] = "primary_span_overlaps_lizMov"
    return {
        "all_unique": moving,
        "all_unique_unfiltered": payload.get("all_unique"),
        "large": large,
        "diag": payload.get("diag"),
        "amp_stats": amp_stats,
        "threshold_deg": float(threshold),
        "n_unique_dropped_still": n_dropped,
        "params": params,
    }


def summarize_double_steps_move_only(payload: dict[str, Any]) -> dict[str, Any]:
    large = payload.get("large")
    if large is None or large.empty:
        large = pd.DataFrame()
        n = 0
    else:
        n = int(len(large))
    params = dict(payload.get("params") or {})
    amp_stats = dict(payload.get("amp_stats") or {})
    hits = (
        large["has_back_and_forth"].astype(bool)
        if n and "has_back_and_forth" in large.columns
        else pd.Series(dtype=bool)
    )
    overall = _rate(n, int(hits.sum()) if n else 0)
    overall["n_hit"] = overall.pop("n_reverse") if "n_reverse" in overall else overall.get("n_hit", 0)
    overall["n_reverse"] = overall.get("n_hit", 0)

    per_animal: dict[str, Any] = {}
    if n and "animal" in large.columns:
        for animal, g in large.groupby("animal", dropna=False):
            gh = g["has_back_and_forth"].astype(bool)
            d = _rate(int(len(g)), int(gh.sum()))
            d["n_hit"] = d.get("n_reverse", 0)
            per_animal[str(animal)] = d

    n_moving_unique = int(len(payload["all_unique"])) if payload.get("all_unique") is not None else 0
    return {
        "question": (
            "Among large saccades that occur during annotated head movement, how often "
            "is there an immediate small-amplitude back-and-forth sequence, interpreted "
            "as nystagmus-like gaze stabilization?"
        ),
        "rule": (
            f"head-moving unique events only (lizMov overlap with primary on–off); "
            f"large = net_angular_disp ≥ {payload.get('threshold_deg')}°; "
            f"back-and-forth = ≥2 smaller extras in (t0, t0+{params.get('post_window_ms', POST_WINDOW_MS):.0f} ms], "
            f"≥1 counter-directed vs primary (> {params.get('reverse_min_deg', REVERSE_MIN_DEG):.0f}°), "
            "and ≥1 consecutive pair of extras that reverse relative to each other"
        ),
        "n_unique_head_moving": n_moving_unique,
        "n_unique_dropped_still": int(payload.get("n_unique_dropped_still") or 0),
        "amp_threshold_deg": float(payload.get("threshold_deg", amp_stats.get("threshold_deg", float("nan")))),
        "amp_stats": amp_stats,
        "params": params,
        "overall": overall,
        "per_animal": per_animal,
    }


def figure_overall(summary: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2.15, 3.6), dpi=150)
    overall = summary.get("overall") or {}
    pct = float(overall.get("pct") or 0.0)
    n = int(overall.get("n") or 0)
    n_hit = int(overall.get("n_hit") or overall.get("n_reverse") or 0)
    ax.bar([0], [pct], color=OVERALL_COLOR, edgecolor="0.2", width=0.42)
    if np.isfinite(pct):
        ax.text(0, pct + 1.4, f"{pct:.1f}%\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(["back-and-forth"])
    ax.set_ylabel("% of large saccades")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.7, 0.7)
    ax.set_title(f"{n_hit} / {n}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def figure_by_animal(summary: dict[str, Any]) -> plt.Figure:
    per = summary.get("per_animal") or {}
    animals = sorted(per)
    if not animals:
        fig, ax = plt.subplots(figsize=(4.0, 2.4), dpi=150)
        ax.set_title("No animals")
        ax.axis("off")
        return fig
    x = np.arange(len(animals), dtype=float)
    pcts = [float((per[a] or {}).get("pct") or 0.0) for a in animals]
    ns = [int((per[a] or {}).get("n") or 0) for a in animals]
    fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * len(animals)), 3.6), dpi=150)
    _bar_with_n(ax, x, pcts, ns, [MOVING_COLOR] * len(animals))
    ax.set_xticks(x)
    ax.set_xticklabels(animals, fontsize=8)
    ax.set_ylabel("% of large saccades during head movement")
    ax.set_title("Back-and-forth sequence during head movement")
    fig.tight_layout()
    return fig


def captions_markdown(summary: dict[str, Any]) -> str:
    overall = summary.get("overall") or {}
    thr = summary.get("amp_threshold_deg")
    thr_s = f"{float(thr):.1f}" if thr is not None and np.isfinite(float(thr)) else "n/a"
    params = summary.get("params") or {}
    w = params.get("post_window_ms", POST_WINDOW_MS)
    n_move = summary.get("n_unique_head_moving", 0)
    n_drop = summary.get("n_unique_dropped_still", 0)
    return "\n".join(
        [
            "# Captions — back-and-forth sequences during head movement",
            "",
            "Universe is unique gaze events whose primary on–off span overlaps a `lizMov` "
            f"head-movement sample ({n_move} events; {n_drop} head-stationary unique events dropped). "
            "A back-and-forth sequence is a **large** unique gaze event "
            f"(≥ {thr_s}°) followed within **{w:.0f} ms** by **≥2 smaller** detected saccades, "
            "with at least one counter-directed relative to the primary (circular difference > 90°) "
            "and at least one consecutive pair of extras that reverse relative to each other.",
            "",
            f"**{PDF_OVERALL}.** Frequency among large saccades during head movement: "
            f"**{overall.get('pct', float('nan')):.1f}%** "
            f"({overall.get('n_hit', overall.get('n_reverse', 0))}/{overall.get('n', 0)}).",
            "",
            f"**{PDF_BY_ANIMAL}.** Same rate by animal.",
            "",
        ]
    )


def reviewer_response_text(summary: dict[str, Any]) -> str:
    overall = summary.get("overall") or {}
    params = summary.get("params") or {}
    rev_min = float(params.get("reverse_min_deg", REVERSE_MIN_DEG))
    window = float(params.get("post_window_ms", POST_WINDOW_MS))
    thr = summary.get("amp_threshold_deg")
    thr_s = f"{float(thr):.0f}" if thr is not None and np.isfinite(float(thr)) else "large"
    n_all = int(overall.get("n") or 0)
    n_hit = int(overall.get("n_hit") or overall.get("n_reverse") or 0)
    pct_all = float(overall.get("pct") or 0.0)
    return (
        "Response: We thank the reviewer for this observation. We agree that the small "
        "back-and-forth movements visible after some saccades in Video S1 are an interesting "
        "feature of the P. vitticeps recordings. We interpret these events as nystagmus-like "
        "compensatory eye movements that stabilize gaze during head motion, rather than as "
        "post-saccadic instability of a stationary eye. Restricting the analysis to saccades "
        "that occurred during annotated head-moving periods, we identified large saccades "
        f"(≥ {thr_s}° net angular displacement) that were followed within {window:.0f} ms by a "
        "sequence of at least two additional smaller saccades that reversed direction "
        f"(back-and-forth: circular direction difference > {rev_min:.0f}° between consecutive extra "
        "saccades, with at least one extra counter-directed relative to the primary). Such "
        "post-saccadic back-and-forth sequences occurred following "
        f"{pct_all:.1f}% of large saccades during head movement (n = {n_hit}/{n_all} events). "
        "This frequency is consistent with a gaze-stabilization role during head motion, in "
        "contrast to the typically stable post-saccadic eye position observed in primates. "
        "We have added this analysis to Fig. Sx and discuss this interpretation in the revised "
        "manuscript.\n"
    )


def export_double_steps_move_only(
    tables: EventTables,
    out_dir: Path,
    *,
    post_window_ms: float = POST_WINDOW_MS,
    reverse_min_deg: float = REVERSE_MIN_DEG,
    amp_percentile: float = AMP_PERCENTILE,
    amp_floor_deg: float = AMP_FLOOR_DEG,
    amp_threshold_deg: float | None = DEFAULT_AMP_THRESHOLD_DEG,
    show: bool = False,
    plot_id: str = PLOT_ID,
) -> dict[str, Path]:
    params = {
        "post_window_ms": float(post_window_ms),
        "reverse_min_deg": float(reverse_min_deg),
        "amp_percentile": float(amp_percentile),
        "amp_floor_deg": float(amp_floor_deg),
        "amp_threshold_deg": amp_threshold_deg,
        "sync_diff_ms": _sync_diff_ms(tables),
        "head_filter": "primary_span_overlaps_lizMov",
    }
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind=KIND,
        tables=tables,
        logic_key=KIND,
        params={**dict(getattr(tables, "params", {}) or {}), KIND: params},
        extra=params,
    )
    payload = collect_double_steps_move_only(
        tables,
        post_window_ms=post_window_ms,
        reverse_min_deg=reverse_min_deg,
        amp_percentile=amp_percentile,
        amp_floor_deg=amp_floor_deg,
        amp_threshold_deg=amp_threshold_deg,
    )
    summary = summarize_double_steps_move_only(payload)
    large = payload["large"]
    _drop_constituents(large).to_csv(bundle.metadata_dir / EVENTS_CSV, index=False)
    diag = payload.get("diag")
    if diag is not None:
        diag.to_csv(bundle.metadata_dir / DIAG_CSV, index=False)
    with open(bundle.metadata_dir / SUMMARY_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    pickle_payload = {
        "summary": summary,
        "amp_stats": payload["amp_stats"],
        "threshold_deg": payload["threshold_deg"],
        "params": payload["params"],
        "stationary_amp_deg": pd.to_numeric(payload["all_unique"]["amp_deg"], errors="coerce").to_numpy(float)
        if payload["all_unique"] is not None and not payload["all_unique"].empty
        else np.asarray([], dtype=float),
        "pdf_amp": PDF_AMP,
        "pdf_overall": PDF_OVERALL,
        "pdf_by_animal": PDF_BY_ANIMAL,
    }
    pkl = bundle.metadata_dir / PICKLE_NAME
    write_pickle_with_meta(
        pickle_payload,
        pkl,
        meta={
            "n_large": (summary.get("overall") or {}).get("n"),
            "pct_overall": (summary.get("overall") or {}).get("pct"),
            "n_unique_head_moving": summary.get("n_unique_head_moving"),
        },
        entrypoint="eye_tracking_system_tools.analysis.double_steps_move_only.export_double_steps_move_only",
    )
    written: dict[str, Path] = {
        EVENTS_CSV: bundle.metadata_dir / EVENTS_CSV,
        DIAG_CSV: bundle.metadata_dir / DIAG_CSV,
        SUMMARY_YAML: bundle.metadata_dir / SUMMARY_YAML,
        PICKLE_NAME: pkl,
    }

    def _save(fig: plt.Figure, name: str) -> None:
        p = bundle.plots_dir / name
        fig.savefig(p, format="pdf", bbox_inches="tight")
        fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
        show_and_close(fig, show)
        written[name] = p

    fig_amp = figure_amplitude_distribution(
        payload["all_unique"], threshold_deg=payload["threshold_deg"], amp_stats=payload["amp_stats"]
    )
    fig_amp.axes[0].set_title(
        f"Amplitude (head-moving unique events)  "
        f"n={int(summary.get('n_unique_head_moving') or 0)}  "
        f"large n={int((summary.get('overall') or {}).get('n') or 0)}"
    )
    _save(fig_amp, PDF_AMP)
    _save(figure_overall(summary), PDF_OVERALL)
    _save(figure_by_animal(summary), PDF_BY_ANIMAL)

    captions = captions_markdown(summary)
    (bundle.metadata_dir / CAPTIONS_MD).write_text(captions, encoding="utf-8")
    (bundle.bundle_dir / CAPTIONS_MD).write_text(captions, encoding="utf-8")
    written[CAPTIONS_MD] = bundle.bundle_dir / CAPTIONS_MD
    response = reviewer_response_text(summary)
    (bundle.metadata_dir / "reviewer_response.md").write_text(response, encoding="utf-8")
    (bundle.bundle_dir / "reviewer_response.md").write_text(response, encoding="utf-8")
    written["reviewer_response.md"] = bundle.bundle_dir / "reviewer_response.md"

    finish_plot_bundle(bundle)
    written["params.yaml"] = bundle.metadata_dir / "params.yaml"
    written["LOGIC.md"] = bundle.metadata_dir / "LOGIC.md"
    written["replot.py"] = bundle.bundle_dir / "replot.py"
    return written


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[3]
    default_out = (
        repo / "development" / "movement_associated_nystagmus" / "double_steps_rev_verbatim" / PLOT_ID
    )
    parser = argparse.ArgumentParser(
        description="Back-and-forth sequences after large saccades, head-moving periods only."
    )
    parser.add_argument("--registry", type=Path, default=repo / "configs" / "paper_blocks.yaml")
    parser.add_argument("--params", type=Path, default=repo / "configs" / "analysis_params.yaml")
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--cache-meta", type=Path, default=None)
    parser.add_argument("--post-window-ms", type=float, default=POST_WINDOW_MS)
    parser.add_argument("--reverse-min-deg", type=float, default=REVERSE_MIN_DEG)
    parser.add_argument("--amp-percentile", type=float, default=AMP_PERCENTILE)
    parser.add_argument("--amp-floor-deg", type=float, default=AMP_FLOOR_DEG)
    parser.add_argument("--amp-threshold-deg", type=float, default=DEFAULT_AMP_THRESHOLD_DEG)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)

    from eye_tracking_system_tools.analysis.block_registry import load_registry
    from eye_tracking_system_tools.analysis.event_cache import build_or_load_event_tables
    from eye_tracking_system_tools.analysis.export_meta import load_params_yaml

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_meta = Path(args.cache_meta) if args.cache_meta is not None else out_dir / "metadata"
    cache_meta.mkdir(parents=True, exist_ok=True)

    params = load_params_yaml(args.params)
    specs = load_registry(args.registry)
    tables, cache_path, from_cache = build_or_load_event_tables(
        specs,
        params,
        cache_meta,
        keep_traces=False,
        prefer_finalized=True,
        force=bool(args.force_cache),
    )
    print(f"registry: {args.registry}")
    print(f"blocks={len(tables.blocks)}  events={len(tables.all_saccades)}  cache={'hit' if from_cache else 'miss'}")
    print(f"cache:    {cache_path}")
    print(f"out:      {out_dir}")

    written = export_double_steps_move_only(
        tables,
        out_dir,
        post_window_ms=float(args.post_window_ms),
        reverse_min_deg=float(args.reverse_min_deg),
        amp_percentile=float(args.amp_percentile),
        amp_floor_deg=float(args.amp_floor_deg),
        amp_threshold_deg=args.amp_threshold_deg,
        show=False,
        plot_id=PLOT_ID,
    )
    summary_path = written.get(SUMMARY_YAML)
    if summary_path is not None and summary_path.is_file():
        summary = yaml.safe_load(summary_path.read_text()) or {}
        overall = summary.get("overall") or {}
        print(f"threshold: {summary.get('amp_threshold_deg')} deg")
        print(f"unique moving: {summary.get('n_unique_head_moving')}  dropped still: {summary.get('n_unique_dropped_still')}")
        print(f"overall:   {overall.get('pct')}%  ({overall.get('n_hit')}/{overall.get('n')})")
    print("\nWrote:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
