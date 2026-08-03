"""
Oculomotor span characteristics across many analyzed blocks (tool 1).

For each eye CSV, report **full range** (max − min) and a **95th-percentile
clip** (p5–p95 by default) of pupil position in **pixels** (``center_x`` /
``center_y``) and **degrees** (recentered Kerr ``k_phi`` / ``k_theta``).

Block collection reuses :class:`~eye_tracking_system_tools.analysis.jitter_gui.JitterBlockBrowser`
with ``require="eye"`` (see ``development/eye_span_pipeline.ipynb``).

Tool 2 (µm scaling + jitter ratio) lives with the jitter mount pipeline.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterable, Literal

import numpy as np
import pandas as pd
import yaml

from eye_tracking_system_tools.analysis.eye_trace_io import (
    load_eye_dataframe,
    resolve_eye_csv,
)
from eye_tracking_system_tools.analysis.pixel_calibration import (
    has_pixel_calibration,
    read_pixel_size,
)
from eye_tracking_system_tools.analysis.run_layout import resolve_run_dir

Species = Literal["lizard", "mouse"]
_DEFAULT_LO = 5.0
_DEFAULT_HI = 95.0

SPAN_REGISTRY_HEADER = """\
# Eye-movement span registry (tool 1).
# Each block_path must contain analysis/ eye CSVs with center_x/center_y
# and preferably Kerr k_phi/k_theta (degrees).
#
# Edit by hand, or populate with development/eye_span_pipeline.ipynb
# (JitterBlockBrowser with require=\"eye\", registry_format=\"paper\").
#
"""


@dataclass(frozen=True)
class AxisSpan:
    """1-D range after percentile clipping (or full min/max when lo=0, hi=100)."""

    lo_pct: float
    hi_pct: float
    span: float
    p_lo: float
    p_hi: float
    n: int


def percentile_axis_span(
    values: np.ndarray,
    *,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> AxisSpan | None:
    """``hi−lo`` percentile width of finite samples; ``None`` if empty."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if lo <= 0 and hi >= 100:
        p_lo = float(np.nanmin(arr))
        p_hi = float(np.nanmax(arr))
    else:
        p_lo = float(np.nanpercentile(arr, lo))
        p_hi = float(np.nanpercentile(arr, hi))
    return AxisSpan(
        lo_pct=float(lo),
        hi_pct=float(hi),
        span=float(p_hi - p_lo),
        p_lo=p_lo,
        p_hi=p_hi,
        n=int(arr.size),
    )


def _xy_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lo: float,
    hi: float,
) -> dict[str, float] | None:
    """Single percentile-window metrics (used by the jitter µm helper)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if x.size == 0:
        return None
    sx = percentile_axis_span(x, lo=lo, hi=hi)
    sy = percentile_axis_span(y, lo=lo, hi=hi)
    assert sx is not None and sy is not None
    mx = float(np.nanmedian(x))
    my = float(np.nanmedian(y))
    r = np.hypot(x - mx, y - my)
    r_hi = float(np.nanpercentile(r, hi)) if hi < 100 else float(np.nanmax(r))
    return {
        "n": float(x.size),
        "span_x": sx.span,
        "span_y": sy.span,
        "span_hypot": float(np.hypot(sx.span, sy.span)),
        "r95": r_hi,
        "diam_r95": 2.0 * r_hi,
        "p_lo_x": sx.p_lo,
        "p_hi_x": sx.p_hi,
        "p_lo_y": sy.p_lo,
        "p_hi_y": sy.p_hi,
    }


def _xy_full_and_p95(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> dict[str, float] | None:
    """Full (min–max) and percentile-clipped spans for a 2-D cloud."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if x.size == 0:
        return None
    full = _xy_metrics(x, y, lo=0.0, hi=100.0)
    clipped = _xy_metrics(x, y, lo=lo, hi=hi)
    assert full is not None and clipped is not None
    return {
        "n": full["n"],
        "full_x": full["span_x"],
        "full_y": full["span_y"],
        "full_hypot": full["span_hypot"],
        "full_diam": full["diam_r95"],
        "p95_x": clipped["span_x"],
        "p95_y": clipped["span_y"],
        "p95_hypot": clipped["span_hypot"],
        "p95_diam": clipped["diam_r95"],
        "lo_pct": float(lo),
        "hi_pct": float(hi),
    }


def species_of(mount_type: str | None, animal: str | None = None) -> Species:
    mt = str(mount_type or "").strip().lower()
    if mt == "mouse":
        return "mouse"
    if animal and str(animal).upper().startswith("M_"):
        return "mouse"
    return "lizard"


def has_eye_span_data(block_path: Path | str) -> bool:
    """True when ``analysis/`` has at least one eye CSV usable for span (px or deg)."""
    analysis = Path(block_path) / "analysis"
    if not analysis.is_dir():
        return False
    for side in ("left", "right"):
        try:
            _resolve_eye_csv_for_span(analysis, side)
            return True
        except FileNotFoundError:
            continue
    return False


def find_blocks_with_eye_data(
    root: Path | str,
    *,
    max_depth: int = 4,
    limit: int = 500,
) -> list[Path]:
    """Bounded walk for block folders with eye CSVs (center or Kerr angles)."""
    root = Path(root)
    if not root.is_dir():
        return []
    base_depth = len(root.resolve().parts)
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if len(here.resolve().parts) - base_depth >= max_depth:
            dirnames[:] = []
        if here.name != "analysis":
            continue
        block = here.parent
        if has_eye_span_data(block):
            found.append(block)
            dirnames[:] = []
        if len(found) >= limit:
            break
    return found


def _block_key(animal: str, block_path: Path) -> str:
    return f"{animal}_{Path(block_path).name}"


def _spec_fields(spec: Any) -> tuple[str, Path, str | None]:
    """Normalize jitter / paper specs (or path-like) to animal, path, mount."""
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        from eye_tracking_system_tools.analysis.jitter_epochs import infer_animal

        return infer_animal(path), path, None
    animal = str(getattr(spec, "animal", "") or "")
    path = Path(getattr(spec, "block_path"))
    if not animal:
        from eye_tracking_system_tools.analysis.jitter_epochs import infer_animal

        animal = infer_animal(path)
    mount = getattr(spec, "mount_type", None)
    return animal, path, None if mount is None else str(mount)


def span_for_eye_df(
    df: pd.DataFrame,
    *,
    um_per_px: float | None,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> dict[str, Any]:
    """
    Per-eye span dict (jitter-pipeline helper: single percentile window + µm).

    Prefer :func:`span_characteristics_for_eye` for tool 1 (full + p95, deg/px).
    """
    out: dict[str, Any] = {
        "lo_pct": float(lo),
        "hi_pct": float(hi),
        "um_per_px": None if um_per_px is None else float(um_per_px),
        "n_px": 0,
        "span_x_px": np.nan,
        "span_y_px": np.nan,
        "span_hypot_px": np.nan,
        "diam_r95_px": np.nan,
        "span_x_um": np.nan,
        "span_y_um": np.nan,
        "span_hypot_um": np.nan,
        "diam_r95_um": np.nan,
        "n_deg": 0,
        "span_phi_deg": np.nan,
        "span_theta_deg": np.nan,
        "span_hypot_deg": np.nan,
        "diam_r95_deg": np.nan,
    }
    if {"center_x", "center_y"} <= set(df.columns):
        cx = pd.to_numeric(df["center_x"], errors="coerce").to_numpy(dtype=float)
        cy = pd.to_numeric(df["center_y"], errors="coerce").to_numpy(dtype=float)
        px = _xy_metrics(cx, cy, lo=lo, hi=hi)
        if px is not None:
            out["n_px"] = int(px["n"])
            out["span_x_px"] = px["span_x"]
            out["span_y_px"] = px["span_y"]
            out["span_hypot_px"] = px["span_hypot"]
            out["diam_r95_px"] = px["diam_r95"]
            if um_per_px is not None and np.isfinite(um_per_px) and um_per_px > 0:
                s = float(um_per_px)
                out["span_x_um"] = px["span_x"] * s
                out["span_y_um"] = px["span_y"] * s
                out["span_hypot_um"] = px["span_hypot"] * s
                out["diam_r95_um"] = px["diam_r95"] * s

    if {"k_phi", "k_theta"} <= set(df.columns):
        phi = pd.to_numeric(df["k_phi"], errors="coerce").to_numpy(dtype=float)
        th = pd.to_numeric(df["k_theta"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(phi) & np.isfinite(th)
        phi, th = phi[mask], th[mask]
        if phi.size:
            phi = phi - float(np.nanmedian(phi))
            th = th - float(np.nanmedian(th))
            deg = _xy_metrics(phi, th, lo=lo, hi=hi)
            if deg is not None:
                out["n_deg"] = int(deg["n"])
                out["span_phi_deg"] = deg["span_x"]
                out["span_theta_deg"] = deg["span_y"]
                out["span_hypot_deg"] = deg["span_hypot"]
                out["diam_r95_deg"] = deg["diam_r95"]
    return out


def span_characteristics_for_eye(
    df: pd.DataFrame,
    *,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> dict[str, Any]:
    """
    Tool-1 metrics: full range and p5–p95 (default) in **pixels** and **degrees**.

    Columns use prefixes ``full_`` / ``p95_`` and suffixes ``_px`` / ``_deg``.
    Axis names: pixels → ``x``/``y``; degrees → ``phi``/``theta``.
    """
    out: dict[str, Any] = {
        "lo_pct": float(lo),
        "hi_pct": float(hi),
        "n_px": 0,
        "n_deg": 0,
        "full_x_px": np.nan,
        "full_y_px": np.nan,
        "full_hypot_px": np.nan,
        "full_diam_px": np.nan,
        "p95_x_px": np.nan,
        "p95_y_px": np.nan,
        "p95_hypot_px": np.nan,
        "p95_diam_px": np.nan,
        "full_phi_deg": np.nan,
        "full_theta_deg": np.nan,
        "full_hypot_deg": np.nan,
        "full_diam_deg": np.nan,
        "p95_phi_deg": np.nan,
        "p95_theta_deg": np.nan,
        "p95_hypot_deg": np.nan,
        "p95_diam_deg": np.nan,
    }
    if {"center_x", "center_y"} <= set(df.columns):
        cx = pd.to_numeric(df["center_x"], errors="coerce").to_numpy(dtype=float)
        cy = pd.to_numeric(df["center_y"], errors="coerce").to_numpy(dtype=float)
        px = _xy_full_and_p95(cx, cy, lo=lo, hi=hi)
        if px is not None:
            out["n_px"] = int(px["n"])
            out["full_x_px"] = px["full_x"]
            out["full_y_px"] = px["full_y"]
            out["full_hypot_px"] = px["full_hypot"]
            out["full_diam_px"] = px["full_diam"]
            out["p95_x_px"] = px["p95_x"]
            out["p95_y_px"] = px["p95_y"]
            out["p95_hypot_px"] = px["p95_hypot"]
            out["p95_diam_px"] = px["p95_diam"]

    if {"k_phi", "k_theta"} <= set(df.columns):
        phi = pd.to_numeric(df["k_phi"], errors="coerce").to_numpy(dtype=float)
        th = pd.to_numeric(df["k_theta"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(phi) & np.isfinite(th)
        phi, th = phi[mask], th[mask]
        if phi.size:
            phi = phi - float(np.nanmedian(phi))
            th = th - float(np.nanmedian(th))
            deg = _xy_full_and_p95(phi, th, lo=lo, hi=hi)
            if deg is not None:
                out["n_deg"] = int(deg["n"])
                out["full_phi_deg"] = deg["full_x"]
                out["full_theta_deg"] = deg["full_y"]
                out["full_hypot_deg"] = deg["full_hypot"]
                out["full_diam_deg"] = deg["full_diam"]
                out["p95_phi_deg"] = deg["p95_x"]
                out["p95_theta_deg"] = deg["p95_y"]
                out["p95_hypot_deg"] = deg["p95_hypot"]
                out["p95_diam_deg"] = deg["p95_diam"]
    return out


def _resolve_eye_csv_for_span(analysis: Path, side: str) -> tuple[Path, str]:
    """Prefer Kerr-angle CSVs; fall back to any eye CSV with ``center_x``/``center_y``."""
    try:
        choice = resolve_eye_csv(analysis, side)
        return choice.path, choice.rule
    except FileNotFoundError:
        pass

    pattern = "left_eye_data*.csv" if side == "left" else "right_eye_data*.csv"
    cands = sorted(analysis.glob(pattern))
    usable: list[Path] = []
    for path in cands:
        try:
            cols = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if {"center_x", "center_y"} <= cols:
            usable.append(path)
    if not usable:
        raise FileNotFoundError(
            f"{analysis}: no {side} eye CSV with center_x/center_y "
            f"(candidates={[p.name for p in cands]})"
        )
    chosen = max(usable, key=lambda p: p.stat().st_mtime)
    return chosen.resolve(), "center_xy_fallback"


def block_span_characteristic_rows(
    spec: Any,
    *,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> list[dict[str, Any]]:
    """One row per eye: full + p95 spans in degrees and pixels (tool 1)."""
    animal, block_path, mount = _spec_fields(spec)
    rows: list[dict[str, Any]] = []
    analysis = block_path / "analysis"
    for side in ("left", "right"):
        row: dict[str, Any] = {
            "block_key": _block_key(animal, block_path),
            "animal": animal,
            "block": block_path.name,
            "block_path": str(block_path),
            "mount_type": mount,
            "species": species_of(mount, animal),
            "eye": side,
            "csv": None,
            "csv_rule": None,
            "ok": False,
            "error": None,
        }
        try:
            path, rule = _resolve_eye_csv_for_span(analysis, side)
            df = load_eye_dataframe(path)
            metrics = span_characteristics_for_eye(df, lo=lo, hi=hi)
            row.update(metrics)
            row["csv"] = path.name
            row["csv_rule"] = rule
            row["ok"] = bool(metrics["n_px"] or metrics["n_deg"])
            notes = []
            if not metrics["n_deg"]:
                notes.append("no Kerr angles — deg span unavailable")
            if not metrics["n_px"]:
                notes.append("no center_x/center_y — px span unavailable")
            if notes and not row["ok"]:
                row["error"] = "; ".join(notes)
            elif notes:
                row["error"] = "; ".join(notes)
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
        rows.append(row)
    return rows


def collect_span_characteristics(
    specs: Iterable[Any],
    *,
    include: Collection[str] | None = None,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> pd.DataFrame:
    """Per-eye full + p95 span table (degrees and pixels) for many blocks."""
    wanted = None if include is None else {str(k) for k in include}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        animal, path, _ = _spec_fields(spec)
        key = _block_key(animal, path)
        if wanted is not None and key not in wanted:
            continue
        rows.extend(block_span_characteristic_rows(spec, lo=lo, hi=hi))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Jitter-pipeline helpers (µm + ratio) — kept for notebook section 7 / tool 2
# ---------------------------------------------------------------------------


def block_eye_span_rows(
    spec: Any,
    *,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> list[dict[str, Any]]:
    """One row per eye with percentile-window spans including µm (jitter context)."""
    animal, block_path, mount = _spec_fields(spec)
    rows: list[dict[str, Any]] = []
    ps = read_pixel_size(block_path) if has_pixel_calibration(block_path) else None
    analysis = block_path / "analysis"
    for side in ("left", "right"):
        row: dict[str, Any] = {
            "block_key": _block_key(animal, block_path),
            "animal": animal,
            "block": block_path.name,
            "mount_type": mount,
            "species": species_of(mount, animal),
            "eye": side,
            "csv": None,
            "ok": False,
            "error": None,
        }
        try:
            path, rule = _resolve_eye_csv_for_span(analysis, side)
            df = load_eye_dataframe(path)
            um = None if ps is None else ps.um_per_px(side)
            metrics = span_for_eye_df(df, um_per_px=um, lo=lo, hi=hi)
            row.update(metrics)
            row["csv"] = path.name
            row["csv_rule"] = rule
            row["ok"] = bool(metrics["n_px"] or metrics["n_deg"])
            notes = []
            if ps is None:
                notes.append("no LR_pix_size.csv (µm columns NaN)")
            if rule == "center_xy_fallback" and not metrics["n_deg"]:
                notes.append("no Kerr angles — deg span unavailable")
            if notes:
                row["error"] = "; ".join(notes)
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
        rows.append(row)
    return rows


def collect_eye_spans(
    specs: Iterable[Any],
    *,
    include: Collection[str] | None = None,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> pd.DataFrame:
    """Per-eye µm/deg/px percentile spans (jitter story)."""
    wanted = None if include is None else {str(k) for k in include}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        animal, path, _ = _spec_fields(spec)
        key = _block_key(animal, path)
        if wanted is not None and key not in wanted:
            continue
        rows.extend(block_eye_span_rows(spec, lo=lo, hi=hi))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize_span_vs_jitter(
    spans: pd.DataFrame,
    *,
    jitter_p95_um: dict[str, float],
    span_col: str = "span_hypot_um",
) -> pd.DataFrame:
    """Per-species mean/median eye span and ``jitter_p95 / eye_span`` ratios."""
    if spans.empty or span_col not in spans.columns:
        return pd.DataFrame()
    out_rows: list[dict[str, Any]] = []
    for species in ("lizard", "mouse"):
        sub = spans.loc[
            (spans["species"] == species) & spans[span_col].notna() & np.isfinite(spans[span_col])
        ]
        if sub.empty:
            continue
        vals = sub[span_col].to_numpy(dtype=float)
        span_mean = float(np.mean(vals))
        span_med = float(np.median(vals))
        j_p95 = jitter_p95_um.get(species)
        out_rows.append(
            {
                "species": species,
                "n_eyes": int(len(vals)),
                "n_blocks": int(sub["block_key"].nunique()),
                "span_mean_um": span_mean,
                "span_median_um": span_med,
                "span_min_um": float(np.min(vals)),
                "span_max_um": float(np.max(vals)),
                "jitter_p95_um": None if j_p95 is None else float(j_p95),
                "ratio_jitter_over_span_mean": (
                    None if j_p95 is None or span_mean <= 0 else float(j_p95) / span_mean
                ),
                "ratio_jitter_over_span_median": (
                    None if j_p95 is None or span_med <= 0 else float(j_p95) / span_med
                ),
                "span_over_jitter_mean": (
                    None if j_p95 is None or float(j_p95) <= 0 else span_mean / float(j_p95)
                ),
                "span_over_jitter_median": (
                    None if j_p95 is None or float(j_p95) <= 0 else span_med / float(j_p95)
                ),
            }
        )
    return pd.DataFrame(out_rows)


def jitter_p95_by_species_from_pool_summary(summary: dict[str, Any]) -> dict[str, float]:
    """Read pooled group ``p95`` from ``jitter_pool_summary.yaml`` / finalize manifest."""
    groups = summary.get("group_stats") or summary.get("groups") or {}
    out: dict[str, float] = {}
    lizard_vals = []
    for key in ("modular", "rigid"):
        g = groups.get(key) or {}
        p95 = g.get("p95")
        if p95 is not None and np.isfinite(float(p95)):
            lizard_vals.append(float(p95))
    if lizard_vals:
        out["lizard"] = float(max(lizard_vals))
    mouse = (groups.get("mouse") or {}).get("p95")
    if mouse is not None and np.isfinite(float(mouse)):
        out["mouse"] = float(mouse)
    return out


def format_span_report(spans: pd.DataFrame, ratios: pd.DataFrame) -> str:
    """Human-readable summary for the jitter notebook (µm ratios)."""
    lines: list[str] = []
    if not ratios.empty:
        lines.append("=== Conservative eye-movement span vs jitter (µm) ===")
        for _, r in ratios.iterrows():
            lines.append(
                f"{r['species']}: span mean={r['span_mean_um']:.0f} µm "
                f"(median={r['span_median_um']:.0f}, n_eyes={int(r['n_eyes'])}, "
                f"n_blocks={int(r['n_blocks'])})"
            )
            if r.get("jitter_p95_um") is not None:
                lines.append(
                    f"  jitter p95={r['jitter_p95_um']:.1f} µm  →  "
                    f"jitter/span={r['ratio_jitter_over_span_median']:.3f} "
                    f"(median span), span/jitter={r['span_over_jitter_median']:.1f}×"
                )
        lines.append("")
    if spans.empty:
        lines.append("(no eye-span rows)")
        return "\n".join(lines)

    show = [
        "block_key",
        "mount_type",
        "eye",
        "span_hypot_deg",
        "span_hypot_um",
        "span_hypot_px",
        "diam_r95_um",
        "n_px",
        "error",
    ]
    cols = [c for c in show if c in spans.columns]
    lines.append("=== Per-block / per-eye spans (p5–p95 hypot) ===")
    lines.append(spans[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return "\n".join(lines)


def format_characteristics_report(df: pd.DataFrame) -> str:
    """Stdout summary for tool 1 (deg + px, full vs p95)."""
    if df.empty:
        return "(no span characteristic rows)"
    lines = ["=== Eye-movement span characteristics (full vs p5–p95) ==="]
    show = [
        "block_key",
        "eye",
        "full_hypot_deg",
        "p95_hypot_deg",
        "full_hypot_px",
        "p95_hypot_px",
        "n_px",
        "n_deg",
        "error",
    ]
    cols = [c for c in show if c in df.columns]
    lines.append(df[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    ok = int(df["ok"].sum()) if "ok" in df.columns else len(df)
    lines.append(f"\n{ok}/{len(df)} eye rows OK across {df['block_key'].nunique()} blocks")
    return "\n".join(lines)


DISPLAY_COLS = [
    "block_key",
    "animal",
    "eye",
    "full_hypot_deg",
    "p95_hypot_deg",
    "full_phi_deg",
    "full_theta_deg",
    "p95_phi_deg",
    "p95_theta_deg",
    "full_hypot_px",
    "p95_hypot_px",
    "full_x_px",
    "full_y_px",
    "p95_x_px",
    "p95_y_px",
    "n_px",
    "n_deg",
    "csv",
    "error",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tool 1: eye-movement span characteristics (full + p95, deg & px).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/eye_span_blocks.yaml"),
        help="Paper-format animals: registry (or jitter blocks: YAML)",
    )
    parser.add_argument("--out-root", type=Path, default=Path("outputs"))
    parser.add_argument("--tag", default="", help="Empty → span_latest overwrite")
    parser.add_argument("--lo", type=float, default=_DEFAULT_LO)
    parser.add_argument("--hi", type=float, default=_DEFAULT_HI)
    args = parser.parse_args(argv)

    run = resolve_run_dir(args.out_root, args.tag or None, prefix="span", default_name="span_latest")
    specs = _load_specs(args.registry)
    if not specs:
        raise SystemExit(f"No blocks in {args.registry}")
    df = collect_span_characteristics(specs, lo=args.lo, hi=args.hi)
    out = run.metadata_dir / "eye_span_characteristics.csv"
    df.to_csv(out, index=False)
    print(format_characteristics_report(df))
    print(f"wrote {out}")
    return 0


def _load_specs(path: Path) -> list[Any]:
    path = Path(path)
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    if "animals" in data:
        from eye_tracking_system_tools.analysis.block_registry import read_paper_registry

        return list(read_paper_registry(path))
    from eye_tracking_system_tools.analysis.jitter_epochs import read_registry_blocks

    return list(read_registry_blocks(path))


if __name__ == "__main__":
    raise SystemExit(main())
