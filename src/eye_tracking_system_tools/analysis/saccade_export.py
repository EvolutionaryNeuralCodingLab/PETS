"""
Per-block finalized saccade detection I/O.

Writes under ``block/analysis/saccades/``:

* ``saccade_events.csv`` — all events (scalar columns + concurrency)
* ``saccade_synced.csv`` / ``saccade_monocular.csv`` — binocular partitions
* ``saccade_events.pkl`` — full event tables (incl. speed profiles) for figures
* ``detection_params.yaml`` / ``detection_summary.yaml`` (+ ``.meta.yaml``)

The preprocessing Saccades tab and paper compile path share this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eye_tracking_system_tools.analysis.binocular import find_synced_saccades_ms
from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.export_meta import git_hash
from eye_tracking_system_tools.analysis.eye_trace_io import csv_choices_meta, load_block_eyes
from eye_tracking_system_tools.analysis.head_labels import label_saccades_head_movement
from eye_tracking_system_tools.analysis.param_tune import (
    detect_eye,
    prepare_traces,
    saccade_params_from_dict,
)

SACCADES_DIRNAME = "saccades"
EVENTS_CSV = "saccade_events.csv"
SYNCED_CSV = "saccade_synced.csv"
MONOCULAR_CSV = "saccade_monocular.csv"
EVENTS_PKL = "saccade_events.pkl"
PARAMS_YAML = "detection_params.yaml"
SUMMARY_YAML = "detection_summary.yaml"

# Array-valued detector columns — kept in the pickle, dropped from CSV.
_PROFILE_COLS = (
    "speed_profile_pixel",
    "speed_profile_pixel_calib",
    "speed_profile_angular",
    "diameter_profile",
)

_DEFAULT_FRAME_MS = 1000.0 / 60.0


@dataclass
class DetectResult:
    """One-block detection output (traces + events + binocular partitions)."""

    spec: BlockSpec
    left: pd.DataFrame
    right: pd.DataFrame
    l_saccades: pd.DataFrame
    r_saccades: pd.DataFrame
    all_saccades: pd.DataFrame
    synced: pd.DataFrame
    non_synced: pd.DataFrame
    csv_meta: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    frame_ms: float = _DEFAULT_FRAME_MS
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalizedSaccades:
    """On-disk finalized events for one block."""

    block_path: Path
    all_saccades: pd.DataFrame
    synced: pd.DataFrame
    non_synced: pd.DataFrame
    params: dict[str, Any]
    summary: dict[str, Any]
    source: str  # "pickle" | "csv"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def saccades_dir(block_path: Path | str) -> Path:
    return Path(block_path) / "analysis" / SACCADES_DIRNAME


def has_finalized_saccades(block_path: Path | str) -> bool:
    d = saccades_dir(block_path)
    return (d / EVENTS_CSV).is_file() and (d / PARAMS_YAML).is_file()


def finalized_paths(block_path: Path | str) -> dict[str, Path]:
    d = saccades_dir(block_path)
    return {
        "dir": d,
        "events_csv": d / EVENTS_CSV,
        "synced_csv": d / SYNCED_CSV,
        "monocular_csv": d / MONOCULAR_CSV,
        "events_pkl": d / EVENTS_PKL,
        "params": d / PARAMS_YAML,
        "summary": d / SUMMARY_YAML,
    }


# ---------------------------------------------------------------------------
# Unit conversion (GUI deg/ms ↔ detector deg/frame)
# ---------------------------------------------------------------------------


def infer_frame_ms(*dfs: pd.DataFrame, fallback: float = _DEFAULT_FRAME_MS) -> float:
    """Median positive ``diff(ms_axis)`` across provided eye DataFrames."""
    samples: list[float] = []
    for df in dfs:
        if df is None or df.empty or "ms_axis" not in df.columns:
            continue
        ms = df["ms_axis"].to_numpy(dtype=float)
        d = np.diff(ms)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            samples.append(float(np.median(d)))
    if not samples:
        return float(fallback)
    return float(np.median(samples))


def deg_per_frame_from_deg_per_ms(thr_deg_per_ms: float, frame_ms: float) -> float:
    """``deg/frame = (deg/ms) * (ms/frame)``."""
    fm = float(frame_ms) if float(frame_ms) > 0 else _DEFAULT_FRAME_MS
    return float(thr_deg_per_ms) * fm


def deg_per_ms_from_deg_per_frame(thr_deg_per_frame: float, frame_ms: float) -> float:
    """``deg/ms = (deg/frame) / (ms/frame)``."""
    fm = float(frame_ms) if float(frame_ms) > 0 else _DEFAULT_FRAME_MS
    return float(thr_deg_per_frame) / fm


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _spec_from_parts(
    block_path: Path | str,
    animal: str,
    block_num: str | None = None,
) -> BlockSpec:
    from eye_tracking_system_tools.analysis.block_registry import _infer_block_num

    block_path = Path(block_path)
    num = block_num or _infer_block_num(block_path)
    return BlockSpec(animal=str(animal), block_path=block_path, block_num=str(num).zfill(3))


def detect_block_saccades(
    block_path: Path | str,
    animal: str,
    block_num: str | None = None,
    params: dict[str, Any] | None = None,
    *,
    keep_traces: bool = True,
    spec: BlockSpec | None = None,
) -> DetectResult:
    """
    Run angular detection + head labels + binocular pairing for one block.

    ``params`` should look like ``analysis_params.yaml`` (``saccade`` / ``binocular``).
    """
    params = dict(params or {})
    if spec is None:
        spec = _spec_from_parts(block_path, animal, block_num)

    loaded = load_block_eyes(spec)
    left = prepare_traces(loaded.left)
    right = prepare_traces(loaded.right)
    frame_ms = infer_frame_ms(left, right)

    sp = saccade_params_from_dict(params)
    _, l_ev = detect_eye(left, saccade_params=sp)
    _, r_ev = detect_eye(right, saccade_params=sp)

    for ev, eye in ((l_ev, "L"), (r_ev, "R")):
        if ev.empty:
            continue
        ev["eye"] = eye
        ev["block"] = spec.block_num
        ev["animal"] = spec.animal

    l_ev = label_saccades_head_movement(l_ev, spec)
    r_ev = label_saccades_head_movement(r_ev, spec)
    all_ev = (
        pd.concat([l_ev, r_ev], ignore_index=True)
        if not (l_ev.empty and r_ev.empty)
        else pd.DataFrame()
    )

    sync_diff_ms = float(params.get("binocular", {}).get("sync_diff_ms", 34.0))
    synced, non_synced = find_synced_saccades_ms(all_ev, sync_diff_ms=sync_diff_ms)
    all_ev = annotate_concurrency(all_ev, synced, non_synced)

    empty = pd.DataFrame()
    result = DetectResult(
        spec=spec,
        left=left if keep_traces else empty,
        right=right if keep_traces else empty,
        l_saccades=l_ev,
        r_saccades=r_ev,
        all_saccades=all_ev,
        synced=synced,
        non_synced=non_synced,
        csv_meta=csv_choices_meta(loaded),
        params=params,
        frame_ms=frame_ms,
    )
    result.summary = summarize_saccades(
        all_ev, synced, non_synced, frame_ms=frame_ms, params=params
    )
    return result


def annotate_concurrency(
    all_ev: pd.DataFrame,
    synced: pd.DataFrame,
    non_synced: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ``concurrency`` (+ optional ``Main``/``Sub``) to the all-events table."""
    if all_ev is None or all_ev.empty:
        return all_ev.copy() if all_ev is not None else pd.DataFrame()

    out = all_ev.copy()
    out["concurrency"] = "monocular"
    if "Main" not in out.columns:
        out["Main"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    else:
        out["Main"] = out["Main"].astype("Int64", errors="ignore")
    if "Sub" not in out.columns:
        out["Sub"] = pd.Series([pd.NA] * len(out), dtype="object")
    else:
        out["Sub"] = out["Sub"].astype(object)

    if synced is not None and not synced.empty and {"eye", "saccade_on_ms"}.issubset(
        synced.columns
    ):
        for _, row in synced.iterrows():
            eye = row["eye"]
            on = float(row["saccade_on_ms"])
            mask = (out["eye"] == eye) & np.isclose(
                out["saccade_on_ms"].to_numpy(dtype=float), on, rtol=0.0, atol=1e-3
            )
            out.loc[mask, "concurrency"] = "concurrent"
            if "Main" in synced.columns:
                out.loc[mask, "Main"] = row["Main"]
            if "Sub" in synced.columns:
                out.loc[mask, "Sub"] = row["Sub"]
            else:
                out.loc[mask, "Sub"] = eye

    if non_synced is not None and not non_synced.empty:
        # Already defaulted to monocular; leave as-is.
        pass
    return out


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def _isi_ms(events: pd.DataFrame, *, on_col: str = "saccade_on_ms") -> np.ndarray:
    if events is None or events.empty or on_col not in events.columns:
        return np.array([], dtype=float)
    # Dedupe binocular pairs: use earliest onset per Main when present.
    work = events.copy()
    has_conc = "concurrency" in work.columns
    has_main = "Main" in work.columns and work["Main"].notna().any()
    if has_conc and has_main:
        concurrent = work[work["concurrency"] == "concurrent"]
        mono = work[work["concurrency"] != "concurrent"]
        parts: list[pd.DataFrame] = []
        if not concurrent.empty:
            parts.append(
                concurrent.groupby("Main", sort=False)[on_col]
                .min()
                .reset_index(name=on_col)
            )
        if not mono.empty:
            parts.append(mono[[on_col]].copy())
        if parts:
            times = np.sort(
                pd.concat(parts, ignore_index=True)[on_col].to_numpy(dtype=float)
            )
        else:
            times = np.sort(work[on_col].to_numpy(dtype=float))
    else:
        times = np.sort(work[on_col].to_numpy(dtype=float))
    times = times[np.isfinite(times)]
    if times.size < 2:
        return np.array([], dtype=float)
    return np.diff(times)


def summarize_saccades(
    all_ev: pd.DataFrame,
    synced: pd.DataFrame,
    non_synced: pd.DataFrame,
    *,
    frame_ms: float | None = None,
    params: dict[str, Any] | None = None,
    amp_bins: int = 40,
    amp_max_deg: float = 25.0,
) -> dict[str, Any]:
    """Brief statistical description for the GUI / summary YAML."""
    n_events = int(len(all_ev)) if all_ev is not None else 0
    n_concurrent_rows = int(len(synced)) if synced is not None else 0
    n_concurrent_pairs = (
        int(synced["Main"].nunique())
        if synced is not None and not synced.empty and "Main" in synced.columns
        else n_concurrent_rows // 2
    )
    n_monocular = int(len(non_synced)) if non_synced is not None else 0

    isi = _isi_ms(all_ev if all_ev is not None else pd.DataFrame())
    isi_mean = float(np.mean(isi)) if isi.size else float("nan")
    isi_std = float(np.std(isi, ddof=1)) if isi.size > 1 else float("nan")

    amps = np.array([], dtype=float)
    if all_ev is not None and not all_ev.empty and "net_angular_disp" in all_ev.columns:
        amps = all_ev["net_angular_disp"].to_numpy(dtype=float)
        amps = amps[np.isfinite(amps)]
    edges = np.linspace(0.0, float(amp_max_deg), int(amp_bins) + 1)
    counts, _ = np.histogram(amps, bins=edges) if amps.size else (np.zeros(amp_bins), edges)

    thr = None
    if params:
        thr = params.get("saccade", {}).get("speed_threshold_deg_per_frame")

    return {
        "n_events": n_events,
        "n_concurrent_pairs": n_concurrent_pairs,
        "n_concurrent_rows": n_concurrent_rows,
        "n_monocular": n_monocular,
        "isi_mean_ms": isi_mean,
        "isi_std_ms": isi_std,
        "isi_n": int(isi.size),
        "amplitude_hist_counts": [int(c) for c in counts.tolist()],
        "amplitude_hist_edges_deg": [float(e) for e in edges.tolist()],
        "amp_mean_deg": float(np.mean(amps)) if amps.size else float("nan"),
        "amp_median_deg": float(np.median(amps)) if amps.size else float("nan"),
        "frame_ms": float(frame_ms) if frame_ms is not None else None,
        "speed_threshold_deg_per_frame": thr,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _scalar_events_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    drop = [c for c in _PROFILE_COLS if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    # Convert remaining object arrays to strings if any slipped through.
    for col in out.columns:
        if out[col].dtype == object:
            sample = out[col].dropna().head(1)
            if len(sample) and isinstance(sample.iloc[0], (np.ndarray, list, tuple)):
                out = out.drop(columns=[col])
    return out


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def write_finalized_saccades(
    block_path: Path | str,
    result: DetectResult,
    *,
    overwrite: bool = True,
) -> dict[str, Path]:
    """Persist detection outputs under ``analysis/saccades/``."""
    block_path = Path(block_path)
    paths = finalized_paths(block_path)
    out_dir = paths["dir"]
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"Finalized saccades already exist: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ev = annotate_concurrency(
        result.all_saccades, result.synced, result.non_synced
    )
    synced = result.synced.copy() if result.synced is not None else pd.DataFrame()
    non_synced = (
        result.non_synced.copy() if result.non_synced is not None else pd.DataFrame()
    )

    _scalar_events_df(all_ev).to_csv(paths["events_csv"], index=False)
    _scalar_events_df(synced).to_csv(paths["synced_csv"], index=False)
    _scalar_events_df(non_synced).to_csv(paths["monocular_csv"], index=False)

    import pickle

    pkl_payload = {
        "all_saccades": all_ev,
        "synced": synced,
        "non_synced": non_synced,
        "l_saccades": result.l_saccades,
        "r_saccades": result.r_saccades,
        "spec": {
            "animal": result.spec.animal,
            "block_path": str(result.spec.block_path),
            "block_num": result.spec.block_num,
        },
        "csv_meta": result.csv_meta,
        "frame_ms": result.frame_ms,
    }
    with open(paths["events_pkl"], "wb") as f:
        pickle.dump(pkl_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    params_out = {
        "saccade": dict(result.params.get("saccade", {})),
        "binocular": dict(result.params.get("binocular", {})),
        "frame_ms": float(result.frame_ms),
        "speed_threshold_deg_per_ms": deg_per_ms_from_deg_per_frame(
            float(
                result.params.get("saccade", {}).get(
                    "speed_threshold_deg_per_frame", 0.8
                )
            ),
            result.frame_ms,
        ),
        "block_key": result.spec.block_key,
        "animal": result.spec.animal,
        "block_num": result.spec.block_num,
        "csv_meta": result.csv_meta,
    }
    # Ensure detector keys are present even if caller passed a partial params dict.
    sp = saccade_params_from_dict(result.params)
    params_out["saccade"].setdefault(
        "speed_threshold_deg_per_frame", float(sp["speed_threshold"])
    )
    params_out["saccade"].setdefault(
        "directional_delta_threshold_deg", float(sp["directional_delta_threshold_deg"])
    )
    params_out["saccade"].setdefault(
        "min_subsaccade_samples", int(sp["min_subsaccade_samples"])
    )
    params_out["saccade"].setdefault("min_net_disp_deg", float(sp["min_net_disp"]))
    params_out["saccade"].setdefault("speed_profile", bool(sp["speed_profile"]))
    params_out["binocular"].setdefault(
        "sync_diff_ms",
        float(result.params.get("binocular", {}).get("sync_diff_ms", 34.0)),
    )
    _write_yaml(paths["params"], params_out)

    summary = result.summary or summarize_saccades(
        all_ev, synced, non_synced, frame_ms=result.frame_ms, params=result.params
    )
    _write_yaml(paths["summary"], summary)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "block_path": str(block_path.resolve()),
        "entrypoint": "eye_tracking_system_tools.analysis.saccade_export",
        "git_hash": git_hash(),
        "n_events": int(summary.get("n_events", 0)),
        "n_concurrent_pairs": int(summary.get("n_concurrent_pairs", 0)),
        "n_monocular": int(summary.get("n_monocular", 0)),
        "files": {k: str(v.name) for k, v in paths.items() if k != "dir"},
    }
    with open(paths["events_csv"].with_suffix(".csv.meta.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    return paths


def read_finalized_saccades(block_path: Path | str) -> FinalizedSaccades:
    """Load finalized events; prefer pickle (profiles) when present."""
    block_path = Path(block_path)
    if not has_finalized_saccades(block_path):
        raise FileNotFoundError(
            f"{block_path}: missing finalized saccades under analysis/{SACCADES_DIRNAME}/"
        )
    paths = finalized_paths(block_path)
    params = _read_yaml(paths["params"])
    summary = _read_yaml(paths["summary"]) if paths["summary"].is_file() else {}

    if paths["events_pkl"].is_file():
        import pickle

        with open(paths["events_pkl"], "rb") as f:
            payload = pickle.load(f)
        return FinalizedSaccades(
            block_path=block_path,
            all_saccades=payload.get("all_saccades", pd.DataFrame()),
            synced=payload.get("synced", pd.DataFrame()),
            non_synced=payload.get("non_synced", pd.DataFrame()),
            params=params,
            summary=summary,
            source="pickle",
        )

    all_ev = pd.read_csv(paths["events_csv"])
    synced = (
        pd.read_csv(paths["synced_csv"])
        if paths["synced_csv"].is_file()
        else pd.DataFrame()
    )
    non_synced = (
        pd.read_csv(paths["monocular_csv"])
        if paths["monocular_csv"].is_file()
        else pd.DataFrame()
    )
    if synced.empty and non_synced.empty and "concurrency" in all_ev.columns:
        synced = all_ev[all_ev["concurrency"] == "concurrent"].copy()
        non_synced = all_ev[all_ev["concurrency"] != "concurrent"].copy()
    return FinalizedSaccades(
        block_path=block_path,
        all_saccades=all_ev,
        synced=synced,
        non_synced=non_synced,
        params=params,
        summary=summary,
        source="csv",
    )


def detection_params_fingerprint(params: dict[str, Any]) -> str:
    """Stable hash of saccade + binocular detection knobs."""
    payload = {
        "saccade": params.get("saccade", {}),
        "binocular": params.get("binocular", {}),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def finalized_params_match(
    block_path: Path | str,
    run_params: dict[str, Any],
    *,
    rtol: float = 1e-6,
) -> bool:
    """True when on-disk detection params match the run YAML (saccade + binocular)."""
    if not has_finalized_saccades(block_path):
        return False
    saved = _read_yaml(finalized_paths(block_path)["params"])
    return detection_params_fingerprint(saved) == detection_params_fingerprint(run_params)


def finalized_fingerprint(specs: list[BlockSpec]) -> dict[str, Any]:
    """Per-block mtime + params hash for event-cache invalidation."""
    out: dict[str, Any] = {}
    for spec in specs:
        if not has_finalized_saccades(spec.block_path):
            continue
        paths = finalized_paths(spec.block_path)
        try:
            params = _read_yaml(paths["params"])
            mtime = max(
                p.stat().st_mtime
                for p in (
                    paths["events_csv"],
                    paths["params"],
                    paths["events_pkl"],
                )
                if p.is_file()
            )
        except OSError:
            continue
        out[spec.block_key] = {
            "mtime": float(mtime),
            "params_fp": detection_params_fingerprint(params),
            "source": "pickle" if paths["events_pkl"].is_file() else "csv",
        }
    return out


def block_bundle_from_finalized(
    spec: BlockSpec,
    finalized: FinalizedSaccades,
    *,
    keep_traces: bool = True,
):
    """Build a :class:`~pipeline.BlockBundle` from finalized on-disk events."""
    from eye_tracking_system_tools.analysis.pipeline import BlockBundle

    empty = pd.DataFrame()
    left = empty
    right = empty
    left_meta: dict[str, Any] = {}
    right_meta: dict[str, Any] = {}
    if keep_traces:
        try:
            loaded = load_block_eyes(spec, log=False)
            left = prepare_traces(loaded.left)
            right = prepare_traces(loaded.right)
            meta = csv_choices_meta(loaded)
            left_meta = {"path": meta["left_csv"], "rule": meta["left_rule"]}
            right_meta = {"path": meta["right_csv"], "rule": meta["right_rule"]}
        except Exception:
            pass

    all_ev = finalized.all_saccades.copy()
    if not all_ev.empty:
        if "animal" not in all_ev.columns:
            all_ev["animal"] = spec.animal
        if "block" not in all_ev.columns:
            all_ev["block"] = spec.block_num

    l_ev = (
        all_ev[all_ev["eye"] == "L"].reset_index(drop=True)
        if not all_ev.empty and "eye" in all_ev.columns
        else empty
    )
    r_ev = (
        all_ev[all_ev["eye"] == "R"].reset_index(drop=True)
        if not all_ev.empty and "eye" in all_ev.columns
        else empty
    )

    return BlockBundle(
        spec=spec,
        left=left,
        right=right,
        left_csv_meta=left_meta,
        right_csv_meta=right_meta,
        l_saccades=l_ev,
        r_saccades=r_ev,
        all_saccades=all_ev,
    )
