"""Back-and-forth (double-step) sequences after large saccades.

Verbatim reviewer description (video S1): after a saccade the eye moves
back-and-forth a couple of times with small amplitude. Operationally that is a
*sequence*, not a single reverse saccade:

  large unique saccade
  + ≥2 additional detected saccades within the post window
  + those extras are smaller than the primary
  + at least one is counter-directed relative to the primary
  + at least one consecutive pair of extras reverse direction relative to each
    other (the ‘back’ then the ‘forth’)

Head still vs moving uses the existing ``lizMov.mat`` annotation on the primary
saccade span. Window is 150 ms from primary onset so two extra events can fit
immediately after the large saccade.
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

from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.head_labels import find_lizmov_mat, load_lizmov_times_ms
from eye_tracking_system_tools.analysis.movement_associated_nystagmus import (
    AMP_FLOOR_DEG,
    AMP_PERCENTILE,
    MOVING_COLOR,
    OVERALL_COLOR,
    REVERSE_MIN_DEG,
    STILL_COLOR,
    _bar_with_n,
    _drop_constituents,
    _rate,
    _sync_diff_ms,
    figure_amplitude_distribution,
    list_followups,
    span_overlaps_movement,
)
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle
from eye_tracking_system_tools.analysis.secondary_after_saccade import (
    direction_class,
    propose_amplitude_threshold,
    unique_gaze_events,
)

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

PLOT_ID = "double_steps_rev_verbatim"
KIND = "double_steps_rev_verbatim"
POST_WINDOW_MS = 150.0

PDF_AMP = "amplitude_distribution.pdf"
PDF_OVERALL = "back_and_forth_overall.pdf"
PDF_BY_HEAD = "back_and_forth_by_head.pdf"
PDF_BY_ANIMAL = "back_and_forth_by_head_animal.pdf"
EVENTS_CSV = "large_saccades.csv"
DIAG_CSV = "block_diagnostics.csv"
SUMMARY_YAML = "summary.yaml"
CAPTIONS_MD = "captions.md"
PICKLE_NAME = "double_steps_rev_verbatim.pkl"


def is_back_and_forth_sequence(
    followups: list[dict[str, Any]],
    *,
    primary_amp: float,
    primary_angle: float,
    reverse_min_deg: float = REVERSE_MIN_DEG,
) -> tuple[bool, int, int]:
    """True when ≥2 smaller extras include a reverse vs primary and a consecutive reversal.

    Returns ``(hit, n_small_followups, n_consecutive_reversals)``.
    """
    small: list[dict[str, Any]] = []
    for rec in followups:
        amp = rec.get("amp_deg")
        if np.isfinite(primary_amp) and np.isfinite(amp) and float(amp) >= float(primary_amp):
            continue
        small.append(rec)
    n_small = len(small)
    if n_small < 2:
        return False, n_small, 0
    vs_primary = [
        direction_class(primary_angle, rec["overall_angle_deg"], reverse_min_deg=reverse_min_deg)
        for rec in small
    ]
    if "reverse" not in vs_primary:
        return False, n_small, 0
    n_flip = 0
    for a, b in zip(small, small[1:]):
        if direction_class(a["overall_angle_deg"], b["overall_angle_deg"], reverse_min_deg=reverse_min_deg) == "reverse":
            n_flip += 1
    return bool(n_flip > 0), n_small, n_flip


def collect_double_steps(
    tables: EventTables,
    *,
    post_window_ms: float = POST_WINDOW_MS,
    reverse_min_deg: float = REVERSE_MIN_DEG,
    amp_percentile: float = AMP_PERCENTILE,
    amp_floor_deg: float = AMP_FLOOR_DEG,
    amp_threshold_deg: float | None = None,
    sync_diff_ms: float | None = None,
) -> dict[str, Any]:
    if sync_diff_ms is None:
        sync_diff_ms = _sync_diff_ms(tables)
    diag_rows: list[dict[str, Any]] = []
    unique_parts: list[pd.DataFrame] = []

    for bundle in tables.blocks:
        spec = bundle.spec
        ev = bundle.all_saccades
        n_saccades = 0 if ev is None or ev.empty else int(len(ev))
        mat = find_lizmov_mat(spec)
        if mat is None:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "lizmov_path": "",
                    "n_eye_saccades": n_saccades,
                    "n_unique_events": 0,
                    "n_head_moving": 0,
                    "n_head_still": 0,
                    "note": "no lizMov.mat",
                }
            )
            continue
        try:
            mov_times = np.sort(load_lizmov_times_ms(mat))
            mov_times = mov_times[np.isfinite(mov_times)]
        except Exception as exc:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "lizmov_path": str(mat),
                    "n_eye_saccades": n_saccades,
                    "n_unique_events": 0,
                    "n_head_moving": 0,
                    "n_head_still": 0,
                    "note": f"lizMov load failed: {exc}",
                }
            )
            continue
        if ev is None or ev.empty:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "lizmov_path": str(mat),
                    "n_eye_saccades": 0,
                    "n_unique_events": 0,
                    "n_head_moving": 0,
                    "n_head_still": 0,
                    "note": "no saccades",
                }
            )
            continue
        unique = unique_gaze_events(
            ev,
            sync_diff_ms=float(sync_diff_ms),
            animal=spec.animal,
            block=spec.block_num,
            block_key=spec.block_key,
        )
        unique["animal"] = spec.animal
        unique["block"] = spec.block_num
        unique["block_key"] = spec.block_key
        unique["lizmov_path"] = str(mat)
        moving: list[bool] = []
        hits: list[bool] = []
        n_smalls: list[int] = []
        n_flips: list[int] = []
        first_lat: list[float] = []
        for rec in unique.itertuples(index=False):
            t0 = float(rec.saccade_on_ms)
            t_off = float(rec.saccade_off_ms)
            follows = list_followups(
                ev,
                t_on=t0,
                constituents=list(rec.constituent_onsets or []),
                post_window_ms=post_window_ms,
            )
            hit, n_small, n_flip = is_back_and_forth_sequence(
                follows,
                primary_amp=float(rec.amp_deg) if rec.amp_deg is not None else float("nan"),
                primary_angle=float(rec.overall_angle_deg) if rec.overall_angle_deg is not None else float("nan"),
                reverse_min_deg=reverse_min_deg,
            )
            moving.append(span_overlaps_movement(t0, t_off, mov_times))
            hits.append(hit)
            n_smalls.append(n_small)
            n_flips.append(n_flip)
            first_lat.append(float(follows[0]["latency_ms"]) if follows else float("nan"))
        unique["head_movement"] = moving
        unique["has_back_and_forth"] = hits
        unique["n_small_followups"] = n_smalls
        unique["n_consecutive_reversals"] = n_flips
        unique["first_followup_latency_ms"] = first_lat
        unique_parts.append(unique)
        diag_rows.append(
            {
                "block_key": spec.block_key,
                "animal": spec.animal,
                "block": spec.block_num,
                "lizmov_path": str(mat),
                "n_eye_saccades": n_saccades,
                "n_unique_events": int(len(unique)),
                "n_head_moving": int(np.sum(moving)),
                "n_head_still": int(len(unique) - np.sum(moving)),
                "note": "",
            }
        )

    all_unique = (
        pd.concat(unique_parts, ignore_index=True)
        if unique_parts
        else pd.DataFrame()
    )
    amps = (
        pd.to_numeric(all_unique["amp_deg"], errors="coerce").to_numpy(float)
        if not all_unique.empty
        else np.asarray([], dtype=float)
    )
    proposed, amp_stats = propose_amplitude_threshold(amps, percentile=amp_percentile, floor_deg=amp_floor_deg)
    threshold = float(amp_threshold_deg) if amp_threshold_deg is not None else proposed
    amp_stats["threshold_deg"] = float(threshold)
    amp_stats["threshold_source"] = "user" if amp_threshold_deg is not None else "proposed"
    if all_unique.empty:
        large = all_unique.copy()
    else:
        large = all_unique.loc[pd.to_numeric(all_unique["amp_deg"], errors="coerce") >= threshold].copy()
        large = large.reset_index(drop=True)
    print(
        f"[double_steps_rev_verbatim] unique={len(all_unique)}  "
        f"threshold={threshold:.1f}°  large={len(large)}"
    )
    return {
        "all_unique": all_unique,
        "large": large,
        "diag": pd.DataFrame(diag_rows),
        "amp_stats": amp_stats,
        "threshold_deg": float(threshold),
        "params": {
            "post_window_ms": float(post_window_ms),
            "reverse_min_deg": float(reverse_min_deg),
            "amp_percentile": float(amp_percentile),
            "amp_floor_deg": float(amp_floor_deg),
            "sync_diff_ms": float(sync_diff_ms),
        },
    }


def summarize_double_steps(payload: dict[str, Any]) -> dict[str, Any]:
    large = payload.get("large")
    if large is None or large.empty:
        large = pd.DataFrame()
        n = 0
    else:
        n = int(len(large))
    params = dict(payload.get("params") or {})
    amp_stats = dict(payload.get("amp_stats") or {})
    hits = large["has_back_and_forth"].astype(bool) if n and "has_back_and_forth" in large.columns else pd.Series(dtype=bool)
    overall = _rate(n, int(hits.sum()) if n else 0)
    overall["n_hit"] = overall.pop("n_reverse") if "n_reverse" in overall else overall.get("n_hit", 0)

    def _split(moving: bool) -> dict[str, float | int]:
        if n == 0 or "head_movement" not in large.columns:
            d = _rate(0, 0)
            d["n_hit"] = d.pop("n_reverse")
            return d
        m = large["head_movement"].astype(bool) == moving
        d = _rate(int(m.sum()), int(hits.loc[m].sum()))
        d["n_hit"] = d.pop("n_reverse")
        return d

    still = _split(False)
    moving = _split(True)
    # Alias so existing bar helpers can read n_reverse.
    for d in (overall, still, moving):
        d["n_reverse"] = d.get("n_hit", 0)

    table = np.array(
        [
            [int(moving["n_hit"]), int(moving["n"]) - int(moving["n_hit"])],
            [int(still["n_hit"]), int(still["n"]) - int(still["n_hit"])],
        ],
        dtype=int,
    )
    fisher_p = float("nan")
    odds = float("nan")
    if table.min() >= 0 and table.sum() > 0 and still["n"] and moving["n"]:
        try:
            from scipy.stats import fisher_exact

            odds, fisher_p = fisher_exact(table, alternative="two-sided")
            odds, fisher_p = float(odds), float(fisher_p)
        except Exception:
            pass

    per_animal: dict[str, Any] = {}
    if n:
        for animal, g in large.groupby("animal", dropna=False):
            gh = g["has_back_and_forth"].astype(bool)
            gm = g["head_movement"].astype(bool)
            os_ = _rate(int(len(g)), int(gh.sum()))
            st = _rate(int((~gm).sum()), int(gh.loc[~gm].sum()))
            mv = _rate(int(gm.sum()), int(gh.loc[gm].sum()))
            for d in (os_, st, mv):
                d["n_hit"] = d.get("n_reverse", 0)
            per_animal[str(animal)] = {"overall": os_, "head_still": st, "head_moving": mv}

    return {
        "question": (
            "Among large saccades, how often is there a small-amplitude back-and-forth "
            "sequence (≥2 extra saccades with a direction reversal) immediately afterward, "
            "and is that almost restricted to head-moving periods?"
        ),
        "rule": (
            f"large = unique-event net_angular_disp ≥ {payload.get('threshold_deg')}°; "
            f"back-and-forth = ≥2 smaller extras in (t0, t0+{params.get('post_window_ms', POST_WINDOW_MS):.0f} ms], "
            f"≥1 counter-directed vs primary (> {params.get('reverse_min_deg', REVERSE_MIN_DEG):.0f}°), "
            "and ≥1 consecutive pair of extras that reverse relative to each other"
        ),
        "n_unique_lizmov": int(len(payload["all_unique"])) if payload.get("all_unique") is not None else 0,
        "amp_threshold_deg": float(payload.get("threshold_deg", amp_stats.get("threshold_deg", float("nan")))),
        "amp_stats": amp_stats,
        "params": params,
        "overall": overall,
        "head_still": still,
        "head_moving": moving,
        "fisher_odds_ratio_moving_vs_still": odds,
        "fisher_p_two_sided": fisher_p,
        "per_animal": per_animal,
    }


def figure_overall(summary: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=150)
    overall = summary.get("overall") or {}
    _bar_with_n(ax, np.array([0]), [float(overall.get("pct") or 0.0)], [int(overall.get("n") or 0)], [OVERALL_COLOR])
    ax.set_xticks([0])
    ax.set_xticklabels(["back-and-forth\nsequence"])
    n_hit = int(overall.get("n_hit") or overall.get("n_reverse") or 0)
    n = int(overall.get("n") or 0)
    ax.set_title(f"{n_hit} / {n} large saccades")
    fig.tight_layout()
    return fig


def figure_by_head(summary: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=150)
    still = summary.get("head_still") or {}
    moving = summary.get("head_moving") or {}
    _bar_with_n(
        ax,
        np.array([0, 1]),
        [float(still.get("pct") or 0.0), float(moving.get("pct") or 0.0)],
        [int(still.get("n") or 0), int(moving.get("n") or 0)],
        [STILL_COLOR, MOVING_COLOR],
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["head stationary", "head moving"])
    p = summary.get("fisher_p_two_sided")
    p_s = f"{p:.3g}" if p is not None and np.isfinite(p) else "n/a"
    ax.set_title(f"Back-and-forth sequence   Fisher p={p_s}")
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
    still = [float((per[a].get("head_still") or {}).get("pct") or 0.0) for a in animals]
    moving = [float((per[a].get("head_moving") or {}).get("pct") or 0.0) for a in animals]
    fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * len(animals)), 3.6), dpi=150)
    ax.bar(x - 0.18, still, width=0.36, color=STILL_COLOR, edgecolor="0.2", label="head stationary")
    ax.bar(x + 0.18, moving, width=0.36, color=MOVING_COLOR, edgecolor="0.2", label="head moving")
    ax.set_xticks(x)
    ax.set_xticklabels(animals, fontsize=8)
    ax.set_ylabel("% of large saccades")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Back-and-forth sequence by animal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def captions_markdown(summary: dict[str, Any]) -> str:
    overall = summary.get("overall") or {}
    still = summary.get("head_still") or {}
    moving = summary.get("head_moving") or {}
    thr = summary.get("amp_threshold_deg")
    thr_s = f"{float(thr):.1f}" if thr is not None and np.isfinite(float(thr)) else "n/a"
    p = summary.get("fisher_p_two_sided")
    p_s = f"{p:.3g}" if p is not None and np.isfinite(p) else "n/a"
    params = summary.get("params") or {}
    w = params.get("post_window_ms", POST_WINDOW_MS)
    return "\n".join(
        [
            "# Captions — back-and-forth sequences after large saccades",
            "",
            "A back-and-forth sequence is a **large** unique gaze event "
            f"(≥ {thr_s}°) followed within **{w:.0f} ms** by **≥2 smaller** detected saccades, "
            "with at least one counter-directed relative to the primary (circular difference > 90°) "
            "and at least one consecutive pair of extras that reverse relative to each other.",
            "",
            f"**{PDF_OVERALL}.** Frequency among large saccades: "
            f"**{overall.get('pct', float('nan')):.1f}%** "
            f"({overall.get('n_hit', overall.get('n_reverse', 0))}/{overall.get('n', 0)}).",
            "",
            f"**{PDF_BY_HEAD}.** Same sequences, split by `lizMov` head annotation on the primary "
            f"saccade. Stationary: **{still.get('pct', float('nan')):.1f}%** (n={still.get('n', 0)}). "
            f"Moving: **{moving.get('pct', float('nan')):.1f}%** (n={moving.get('n', 0)}). Fisher p={p_s}.",
            "",
        ]
    )


def reviewer_response_text(summary: dict[str, Any]) -> str:
    overall = summary.get("overall") or {}
    still = summary.get("head_still") or {}
    moving = summary.get("head_moving") or {}
    params = summary.get("params") or {}
    rev_min = float(params.get("reverse_min_deg", REVERSE_MIN_DEG))
    window = float(params.get("post_window_ms", POST_WINDOW_MS))
    thr = summary.get("amp_threshold_deg")
    thr_s = f"{float(thr):.0f}" if thr is not None and np.isfinite(float(thr)) else "large"
    p = summary.get("fisher_p_two_sided")
    odds = summary.get("fisher_odds_ratio_moving_vs_still")
    p_s = "P < 0.001" if p is not None and np.isfinite(p) and p < 0.001 else (
        f"P = {p:.3g}" if p is not None and np.isfinite(p) else "P = n/a"
    )
    odds_s = f"{odds:.2f}" if odds is not None and np.isfinite(odds) else "n/a"
    n_all = int(overall.get("n") or 0)
    n_hit = int(overall.get("n_hit") or overall.get("n_reverse") or 0)
    pct_all = float(overall.get("pct") or 0.0)
    pct_move = float(moving.get("pct") or 0.0)
    pct_still = float(still.get("pct") or 0.0)
    n_move = int(moving.get("n") or 0)
    n_move_hit = int(moving.get("n_hit") or moving.get("n_reverse") or 0)
    n_still = int(still.get("n") or 0)
    n_still_hit = int(still.get("n_hit") or still.get("n_reverse") or 0)
    if pct_still < 5.0 and pct_move > pct_still:
        assoc_sentence = (
            "Thus, these back-and-forth movements occurred almost exclusively during "
            "head-moving periods."
        )
    else:
        assoc_sentence = (
            "Thus, these back-and-forth movements were strongly associated with concurrent "
            "head movement."
        )
    return (
        "Response: We thank the reviewer for this observation. We agree that the small "
        "back-and-forth movements visible after some saccades in Video S1 are an interesting "
        "feature of the P. vitticeps recordings. To quantify their occurrence, we identified "
        f"large saccades (≥ {thr_s}° net angular displacement) that were followed within {window:.0f} ms "
        "by a sequence of at least two additional smaller saccades that reversed direction "
        f"(back-and-forth: circular direction difference > {rev_min:.0f}° between consecutive extra "
        "saccades, with at least one extra counter-directed relative to the primary). Such "
        f"post-saccadic back-and-forth sequences occurred following {pct_all:.1f}% of large saccades "
        f"(n = {n_hit}/{n_all} events).\n"
        "\n"
        "We next asked whether these sequences were associated with head movement. "
        f"Back-and-forth sequences occurred in {pct_move:.1f}% of large saccades during head-moving "
        f"periods (n = {n_move_hit}/{n_move}) compared with {pct_still:.1f}% during head-stationary "
        f"periods (n = {n_still_hit}/{n_still}; two-sided Fisher’s exact test, odds ratio = {odds_s}, "
        f"{p_s}). {assoc_sentence} We interpret this "
        "association as consistent with nystagmus-like compensatory eye movements contributing to "
        "gaze stabilization during head motion, rather than the eye necessarily remaining fixed at "
        "its initial post-saccadic position as is commonly observed following primate saccades. "
        "We have added this analysis to Fig. Sx and discuss this interpretation in the revised manuscript.\n"
    )


def export_double_steps(
    tables: EventTables,
    out_dir: Path,
    *,
    post_window_ms: float = POST_WINDOW_MS,
    reverse_min_deg: float = REVERSE_MIN_DEG,
    amp_percentile: float = AMP_PERCENTILE,
    amp_floor_deg: float = AMP_FLOOR_DEG,
    amp_threshold_deg: float | None = None,
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
    }
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind=KIND,
        tables=tables,
        logic_key=KIND,
        params={**dict(getattr(tables, "params", {}) or {}), "double_steps_rev_verbatim": params},
        extra=params,
    )
    payload = collect_double_steps(
        tables,
        post_window_ms=post_window_ms,
        reverse_min_deg=reverse_min_deg,
        amp_percentile=amp_percentile,
        amp_floor_deg=amp_floor_deg,
        amp_threshold_deg=amp_threshold_deg,
    )
    summary = summarize_double_steps(payload)
    large = payload["large"]
    _drop_constituents(large).to_csv(bundle.metadata_dir / EVENTS_CSV, index=False)
    payload["diag"].to_csv(bundle.metadata_dir / DIAG_CSV, index=False)
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
        "pdf_by_head": PDF_BY_HEAD,
        "pdf_by_animal": PDF_BY_ANIMAL,
    }
    pkl = bundle.metadata_dir / PICKLE_NAME
    write_pickle_with_meta(
        pickle_payload,
        pkl,
        meta={
            "n_large": (summary.get("overall") or {}).get("n"),
            "pct_overall": (summary.get("overall") or {}).get("pct"),
            "pct_still": (summary.get("head_still") or {}).get("pct"),
            "pct_moving": (summary.get("head_moving") or {}).get("pct"),
        },
        entrypoint="eye_tracking_system_tools.analysis.double_steps_rev_verbatim.export_double_steps",
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

    _save(
        figure_amplitude_distribution(
            payload["all_unique"], threshold_deg=payload["threshold_deg"], amp_stats=payload["amp_stats"]
        ),
        PDF_AMP,
    )
    _save(figure_overall(summary), PDF_OVERALL)
    _save(figure_by_head(summary), PDF_BY_HEAD)
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
    default_out = repo / "development" / "movement_associated_nystagmus" / PLOT_ID
    parser = argparse.ArgumentParser(
        description="Back-and-forth sequences after large saccades vs lizMov head labels."
    )
    parser.add_argument("--registry", type=Path, default=repo / "configs" / "paper_blocks.yaml")
    parser.add_argument("--params", type=Path, default=repo / "configs" / "analysis_params.yaml")
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--cache-meta", type=Path, default=None)
    parser.add_argument("--post-window-ms", type=float, default=POST_WINDOW_MS)
    parser.add_argument("--reverse-min-deg", type=float, default=REVERSE_MIN_DEG)
    parser.add_argument("--amp-percentile", type=float, default=AMP_PERCENTILE)
    parser.add_argument("--amp-floor-deg", type=float, default=AMP_FLOOR_DEG)
    parser.add_argument("--amp-threshold-deg", type=float, default=None)
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

    written = export_double_steps(
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
        still = summary.get("head_still") or {}
        moving = summary.get("head_moving") or {}
        print(f"threshold: {summary.get('amp_threshold_deg')} deg")
        print(f"overall:   {overall.get('pct')}%  ({overall.get('n_hit')}/{overall.get('n')})")
        print(f"still:     {still.get('pct')}%  ({still.get('n_hit')}/{still.get('n')})")
        print(f"moving:    {moving.get('pct')}%  ({moving.get('n_hit')}/{moving.get('n')})")
    print("\nWrote:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


def figure_back_and_forth_example(
    t_s: np.ndarray,
    phi: dict[str, np.ndarray],
    theta: dict[str, np.ndarray],
    *,
    t0_s: float,
    t_off_s: float,
    extra_onsets_s: list[float],
    title: str = "",
    pre_s: float = 0.08,
    post_s: float = 0.22,
):
    """Short φ/θ snippet around one back-and-forth primary saccade."""
    t = np.asarray(t_s, dtype=float)
    lo, hi = float(t0_s) - float(pre_s), float(t0_s) + float(post_s)
    m = (t >= lo) & (t <= hi)
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 3.2), dpi=150, sharex=True)
    colors = {"L": "#0072B2", "R": "#D55E00"}
    for ax, key, ylab in (
        (axes[0], "phi", "φ [deg]"),
        (axes[1], "theta", "θ [deg]"),
    ):
        body = phi if key == "phi" else theta
        for side in ("L", "R"):
            y = np.asarray(body.get(side, []), dtype=float)
            n = min(t.size, y.size)
            if n < 2:
                continue
            mm = m[:n]
            ax.plot(t[:n][mm], y[:n][mm], color=colors[side], lw=1.1, label=side)
        ax.axvline(float(t0_s), color="0.2", ls="-", lw=0.9)
        ax.axvline(float(t_off_s), color="0.2", ls="--", lw=0.8)
        for te in extra_onsets_s:
            ax.axvline(float(te), color="#009E73", ls=":", lw=0.9)
        ax.set_ylabel(ylab, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    axes[1].set_xlabel("Time [s]", fontsize=8)
    if title:
        axes[0].set_title(title, fontsize=8)
    fig.tight_layout()
    return fig


def export_back_and_forth_example_from_trace(
    *,
    trace_pkl: Path | str,
    events_csv: Path | str,
    event_id: str,
    out_dir: Path | str,
    plot_id: str = "back_and_forth_example",
    show: bool = False,
    pre_s: float = 0.25,
    post_s: float = 0.55,
) -> dict[str, Path]:
    """Example trace for one ``has_back_and_forth`` event using a species-trace pickle."""
    import pickle

    ev = pd.read_csv(events_csv)
    row = ev.loc[ev["event_id"].astype(str) == str(event_id)]
    if row.empty:
        raise ValueError(f"event_id {event_id!r} not in {events_csv}")
    rec = row.iloc[0]
    t0_ms = float(rec["saccade_on_ms"])
    t_off_ms = float(rec["saccade_off_ms"])
    extras = ev[
        (ev["animal"].astype(str) == str(rec["animal"]))
        & (ev["block"].astype(str) == str(rec["block"]))
        & (ev["saccade_on_ms"] > t0_ms)
        & (ev["saccade_on_ms"] <= t0_ms + float(POST_WINDOW_MS))
        & (ev["event_id"].astype(str) != str(event_id))
    ]
    extra_onsets_s = [float(x) / 1000.0 for x in extras["saccade_on_ms"].tolist()]

    with open(trace_pkl, "rb") as f:
        tr = pickle.load(f)
    t_rel = np.asarray(tr["t_s"], dtype=float)
    win0 = float(tr.get("window_start_s") or 0.0)
    t_abs = t_rel + win0
    t0_s = t0_ms / 1000.0
    title = (
        f"{event_id}  amp={float(rec['amp_deg']):.1f}°  "
        f"head_moving={bool(rec['head_movement'])}  "
        f"n_extra={int(rec['n_small_followups'])}"
    )
    fig = figure_back_and_forth_example(
        t_abs,
        tr.get("phi") or {},
        tr.get("theta") or {},
        t0_s=t0_s,
        t_off_s=t_off_ms / 1000.0,
        extra_onsets_s=extra_onsets_s,
        title=title,
        pre_s=pre_s,
        post_s=post_s,
    )
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="back_and_forth_example",
        logic_key="double_steps_rev_verbatim",
        cohort={"cohort": "lizard_paper", "animals": [str(rec["animal"])], "block_keys": [str(rec["block_key"])]},
        params={
            "event_id": str(event_id),
            "post_window_ms": float(POST_WINDOW_MS),
            "trace_pickle": str(trace_pkl),
            "events_csv": str(events_csv),
        },
    )
    pdf = bundle.plots_dir / "back_and_forth_example.pdf"
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    payload = {
        "t_s": t_abs,
        "phi": tr.get("phi"),
        "theta": tr.get("theta"),
        "t0_s": t0_s,
        "t_off_s": t_off_ms / 1000.0,
        "extra_onsets_s": extra_onsets_s,
        "event_id": str(event_id),
        "title": title,
        "pdf_name": "back_and_forth_example.pdf",
        "pre_s": float(pre_s),
        "post_s": float(post_s),
        "animal": str(rec["animal"]),
        "block": str(rec["block"]),
        "amp_deg": float(rec["amp_deg"]),
        "head_movement": bool(rec["head_movement"]),
    }
    pkl = bundle.metadata_dir / "back_and_forth_example.pkl"
    write_pickle_with_meta(
        payload,
        pkl,
        meta={"event_id": str(event_id), "amp_deg": float(rec["amp_deg"])},
        entrypoint=(
            "eye_tracking_system_tools.analysis.double_steps_rev_verbatim."
            "export_back_and_forth_example_from_trace"
        ),
    )
    finish_plot_bundle(bundle)
    # Custom replot: the shared template has no back_and_forth_example kind.
    (bundle.bundle_dir / "replot.py").write_text(
        _EXAMPLE_REPLOT,
        encoding="utf-8",
    )
    return {
        "back_and_forth_example.pdf": pdf,
        "back_and_forth_example.pkl": pkl,
        "replot.py": bundle.bundle_dir / "replot.py",
    }


def rank_binocular_back_and_forth_examples(events_csv: Path | str) -> pd.DataFrame:
    """Score back-and-forth primaries; prefer binocular events with L and R extras."""
    ev = pd.read_csv(events_csv)
    hits = ev[ev["has_back_and_forth"].astype(str).str.lower().isin({"true", "1"})].copy()
    rows: list[dict[str, Any]] = []
    for _, rec in hits.iterrows():
        t0 = float(rec["saccade_on_ms"])
        extras = ev[
            (ev["animal"].astype(str) == str(rec["animal"]))
            & (ev["block"].astype(str) == str(rec["block"]))
            & (ev["saccade_on_ms"] > t0)
            & (ev["saccade_on_ms"] <= t0 + float(POST_WINDOW_MS))
            & (ev["event_id"].astype(str) != str(rec["event_id"]))
        ]
        extra_eyes = {str(x).upper()[:1] for x in extras["primary_eye"].tolist()}
        prim_eye = str(rec["primary_eye"]).upper()[:1]
        rows.append(
            {
                "event_id": str(rec["event_id"]),
                "animal": str(rec["animal"]),
                "block": str(rec["block"]),
                "block_key": str(rec["block_key"]),
                "amp_deg": float(rec["amp_deg"]),
                "concurrency": str(rec.get("concurrency", "")),
                "primary_eye": prim_eye,
                "n_small_followups": int(rec["n_small_followups"]),
                "n_consecutive_reversals": int(rec["n_consecutive_reversals"]),
                "both_eyes_extras": bool(extra_eyes >= {"L", "R"}),
                "binocular": str(rec.get("concurrency", "")).lower().startswith("binoc"),
                "n_extra_eyes": int(len(extra_eyes)),
                "extra_amp_sum": float(extras["amp_deg"].sum()) if len(extras) else 0.0,
                "head_movement": bool(rec["head_movement"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        by=[
            "both_eyes_extras",
            "binocular",
            "n_consecutive_reversals",
            "n_small_followups",
            "extra_amp_sum",
            "amp_deg",
        ],
        ascending=False,
    ).reset_index(drop=True)


def export_back_and_forth_example_from_block(
    *,
    spec,
    events_csv: Path | str,
    event_id: str,
    out_dir: Path | str,
    plot_id: str = "back_and_forth_example",
    show: bool = False,
    pre_s: float = 0.25,
    post_s: float = 0.55,
) -> dict[str, Path]:
    """Example from full-block Kerr traces (not the 80 s S8 pickle)."""
    from eye_tracking_system_tools.analysis.eye_trace_io import load_block_eyes
    from eye_tracking_system_tools.analysis.species_traces import time_seconds_from_ms_axis

    ev = pd.read_csv(events_csv)
    row = ev.loc[ev["event_id"].astype(str) == str(event_id)]
    if row.empty:
        raise ValueError(f"event_id {event_id!r} not in {events_csv}")
    rec = row.iloc[0]
    t0_ms = float(rec["saccade_on_ms"])
    t_off_ms = float(rec["saccade_off_ms"])
    extras = ev[
        (ev["animal"].astype(str) == str(rec["animal"]))
        & (ev["block"].astype(str) == str(rec["block"]))
        & (ev["saccade_on_ms"] > t0_ms)
        & (ev["saccade_on_ms"] <= t0_ms + float(POST_WINDOW_MS))
        & (ev["event_id"].astype(str) != str(event_id))
    ]
    extra_onsets_s = [float(x) / 1000.0 for x in extras["saccade_on_ms"].tolist()]
    loaded = load_block_eyes(spec, log=True)
    phi: dict[str, np.ndarray] = {}
    theta: dict[str, np.ndarray] = {}
    t_abs = None
    for side, df in (("L", loaded.left), ("R", loaded.right)):
        if df is None or df.empty or "k_phi" not in df.columns:
            continue
        if "ms_axis" in df.columns:
            t = time_seconds_from_ms_axis(df["ms_axis"].to_numpy(float))
        else:
            t = np.arange(len(df), dtype=float) / 60.0
        if t_abs is None:
            t_abs = t
        phi[side] = df["k_phi"].to_numpy(float)
        theta[side] = (
            df["k_theta"].to_numpy(float) if "k_theta" in df.columns else np.zeros(len(df))
        )
    if t_abs is None:
        raise RuntimeError(f"No Kerr traces for {spec.block_path}")
    t0_s = t0_ms / 1000.0
    title = (
        f"{event_id}  amp={float(rec['amp_deg']):.1f}°  "
        f"head_moving={bool(rec['head_movement'])}  "
        f"n_extra={int(rec['n_small_followups'])}"
    )
    fig = figure_back_and_forth_example(
        t_abs,
        phi,
        theta,
        t0_s=t0_s,
        t_off_s=t_off_ms / 1000.0,
        extra_onsets_s=extra_onsets_s,
        title=title,
        pre_s=pre_s,
        post_s=post_s,
    )
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="back_and_forth_example",
        logic_key="double_steps_rev_verbatim",
        cohort={"cohort": "lizard_paper", "animals": [str(rec["animal"])], "block_keys": [str(rec["block_key"])]},
        params={
            "event_id": str(event_id),
            "post_window_ms": float(POST_WINDOW_MS),
            "block_path": str(spec.block_path),
            "events_csv": str(events_csv),
            "pre_s": float(pre_s),
            "post_s": float(post_s),
        },
    )
    pdf = bundle.plots_dir / "back_and_forth_example.pdf"
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    payload = {
        "t_s": t_abs,
        "phi": phi,
        "theta": theta,
        "t0_s": t0_s,
        "t_off_s": t_off_ms / 1000.0,
        "extra_onsets_s": extra_onsets_s,
        "event_id": str(event_id),
        "title": title,
        "pdf_name": "back_and_forth_example.pdf",
        "pre_s": float(pre_s),
        "post_s": float(post_s),
        "animal": str(rec["animal"]),
        "block": str(rec["block"]),
        "amp_deg": float(rec["amp_deg"]),
        "head_movement": bool(rec["head_movement"]),
    }
    pkl = bundle.metadata_dir / "back_and_forth_example.pkl"
    write_pickle_with_meta(
        payload,
        pkl,
        meta={"event_id": str(event_id), "amp_deg": float(rec["amp_deg"]), "pre_s": float(pre_s)},
        entrypoint=(
            "eye_tracking_system_tools.analysis.double_steps_rev_verbatim."
            "export_back_and_forth_example_from_block"
        ),
    )
    finish_plot_bundle(bundle)
    (bundle.bundle_dir / "replot.py").write_text(_EXAMPLE_REPLOT, encoding="utf-8")
    return {
        "back_and_forth_example.pdf": pdf,
        "back_and_forth_example.pkl": pkl,
        "replot.py": bundle.bundle_dir / "replot.py",
    }


_EXAMPLE_REPLOT = r'''#!/usr/bin/env python3
"""Redraw the S12 example trace from metadata/ into plots/replot/."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
META = ROOT / "metadata"
PLOTS = ROOT / "plots"
OUT = PLOTS if "--overwrite" in sys.argv else (PLOTS / "replot")


def main() -> int:
    import matplotlib.pyplot as plt

    pkl = next(iter(sorted(META.glob("*.pkl"))))
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    t = np.asarray(data["t_s"], dtype=float)
    t0 = float(data["t0_s"])
    t_off = float(data["t_off_s"])
    extras = [float(x) for x in (data.get("extra_onsets_s") or [])]
    pre, post = float(data.get("pre_s", 0.08)), float(data.get("post_s", 0.22))
    lo, hi = t0 - pre, t0 + post
    m = (t >= lo) & (t <= hi)
    colors = {"L": "#0072B2", "R": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 3.2), dpi=150, sharex=True)
    for ax, key, ylab in ((axes[0], "phi", "φ [deg]"), (axes[1], "theta", "θ [deg]")):
        body = data.get(key) or {}
        for side in ("L", "R"):
            y = np.asarray(body.get(side, []), dtype=float)
            n = min(t.size, y.size)
            if n < 2:
                continue
            mm = m[:n]
            ax.plot(t[:n][mm], y[:n][mm], color=colors[side], lw=1.1, label=side)
        ax.axvline(t0, color="0.2", ls="-", lw=0.9)
        ax.axvline(t_off, color="0.2", ls="--", lw=0.8)
        for te in extras:
            ax.axvline(te, color="#009E73", ls=":", lw=0.9)
        ax.set_ylabel(ylab, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    axes[1].set_xlabel("Time [s]", fontsize=8)
    axes[0].set_title(str(data.get("title") or ""), fontsize=8)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / str(data.get("pdf_name", "back_and_forth_example.pdf")), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


if __name__ == "__main__":
    raise SystemExit(main())
