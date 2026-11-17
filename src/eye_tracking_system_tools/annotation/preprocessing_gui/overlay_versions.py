"""Discover / load / promote ellipse overlay versions for the Verify tab."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from eye_tracking_system_tools.preprocessing.ellipse_geometry_meta import (
    ELLIPSE_GEOMETRY_META_FILENAME,
    read_ellipse_geometry_version,
    write_ellipse_geometry_meta,
)

EyeSide = Literal["left", "right"]

# Files moved/copied when promoting a version into analysis/.
PROMOTE_FILENAMES = (
    "left_eye_data.csv",
    "right_eye_data.csv",
    "le_df.csv",
    "re_df.csv",
    ELLIPSE_GEOMETRY_META_FILENAME,
)


@dataclass(frozen=True)
class OverlaySource:
    """One selectable overlay dataset under a block's ``analysis/`` tree."""

    key: str
    label: str
    directory: Path
    kind: str  # main | subdir | upstream | hybrid
    geometry_version: int
    notes: str = ""


def _infer_phi_geometry_version(df: pd.DataFrame, *, default: int = 1) -> int:
    """Heuristic: typical Verify-edited degree tables have |phi| >> π."""
    if "phi" not in df.columns:
        return int(default)
    phi = pd.to_numeric(df["phi"], errors="coerce").to_numpy(dtype=float)
    finite = phi[np.isfinite(phi)]
    if finite.size == 0:
        return int(default)
    med = float(np.nanmedian(np.abs(finite)))
    # Radians from the fitter are usually <|1|; degree-edited tables are tens–hundreds.
    if med > 2.0:
        return 1  # OpenCV passthrough (already degrees / bogus degree arithmetic)
    return 2


def _normalize_fit_df(df: pd.DataFrame, eye: EyeSide) -> pd.DataFrame:
    """Map le_df/re_df frame columns into Verify overlay schema."""
    out = df.copy()
    frame_src = "L_eye_frame" if eye == "left" else "R_eye_frame"
    if "eye_frame" not in out.columns and frame_src in out.columns:
        out = out.rename(columns={frame_src: "eye_frame"})
    # Prefer corrected centers when present.
    if "center_x" not in out.columns and "center_x_corrected" in out.columns:
        out["center_x"] = out["center_x_corrected"]
    if "center_y" not in out.columns and "center_y_corrected" in out.columns:
        out["center_y"] = out["center_y_corrected"]
    if "center_x_corrected" in out.columns:
        out["center_x"] = out["center_x_corrected"]
    if "center_y_corrected" in out.columns:
        out["center_y"] = out["center_y_corrected"]
    return out


def _eye_table_paths(directory: Path, eye: EyeSide) -> tuple[Path | None, Path | None]:
    """Return (preferred_eye_data, fit_df) paths that exist under ``directory``."""
    if eye == "left":
        eye_data = directory / "left_eye_data.csv"
        fit = directory / "le_df.csv"
    else:
        eye_data = directory / "right_eye_data.csv"
        fit = directory / "re_df.csv"
    return (eye_data if eye_data.is_file() else None, fit if fit.is_file() else None)


def discover_overlay_sources(analysis_path: Path | str) -> list[OverlaySource]:
    """List overlay sources available for a block analysis folder."""
    analysis = Path(analysis_path)
    if not analysis.is_dir():
        return []

    sources: list[OverlaySource] = []
    main_left = analysis / "left_eye_data.csv"
    main_right = analysis / "right_eye_data.csv"
    main_le = analysis / "le_df.csv"
    main_re = analysis / "re_df.csv"
    main_ver = read_ellipse_geometry_version(analysis, default=1)
    if main_left.is_file() or main_right.is_file():
        # Refine from table if meta absent.
        sample = main_left if main_left.is_file() else main_right
        try:
            df = pd.read_csv(sample, nrows=5000)
            main_ver = read_ellipse_geometry_version(analysis, default=_infer_phi_geometry_version(df))
        except Exception:
            pass
        sources.append(
            OverlaySource(
                key="main",
                label="Main (left/right_eye_data)",
                directory=analysis,
                kind="main",
                geometry_version=main_ver,
                notes="Current analysis eye tables used by Kerr / downstream steps.",
            )
        )
    if main_le.is_file() or main_re.is_file():
        sources.append(
            OverlaySource(
                key="upstream",
                label="Upstream fits (le_df / re_df)",
                directory=analysis,
                kind="upstream",
                geometry_version=read_ellipse_geometry_version(analysis, default=1),
                notes="Raw synced ellipse tables before Verify export.",
            )
        )

    for sub in sorted(p for p in analysis.iterdir() if p.is_dir()):
        if sub.name.startswith("."):
            continue
        left_p, le_p = _eye_table_paths(sub, "left")
        right_p, re_p = _eye_table_paths(sub, "right")
        if not any((left_p, right_p, le_p, re_p)):
            continue
        ver = read_ellipse_geometry_version(sub, default=2 if sub.name == "ellipse_v2" else 1)
        label = {
            "ellipse_v2": "Repaired ellipse_v2",
            "legacy": "Legacy (previous main)",
        }.get(sub.name, f"Version: {sub.name}")
        sources.append(
            OverlaySource(
                key=f"dir:{sub.name}",
                label=label,
                directory=sub,
                kind="subdir",
                geometry_version=ver,
                notes=f"Tables under analysis/{sub.name}/",
            )
        )
        # Hybrid: repaired geometry on main centers (best visual check for orientation-only).
        if (
            sub.name == "ellipse_v2"
            and (le_p is not None or re_p is not None)
            and (main_left.is_file() or main_right.is_file())
        ):
            sources.append(
                OverlaySource(
                    key="ellipse_v2_on_main_centers",
                    label="Repaired v2 geometry on main centers",
                    directory=sub,
                    kind="hybrid",
                    geometry_version=2,
                    notes=(
                        "Uses main left/right_eye_data centers/frames; overlays "
                        "width/height/phi from ellipse_v2 le_df/re_df."
                    ),
                )
            )

    # Stable order: main, hybrid, ellipse_v2, upstream, legacy, others
    order = {
        "main": 0,
        "ellipse_v2_on_main_centers": 1,
        "dir:ellipse_v2": 2,
        "upstream": 3,
        "dir:legacy": 4,
    }
    sources.sort(key=lambda s: (order.get(s.key, 50), s.label))
    return sources


def _merge_geometry(
    base: pd.DataFrame,
    geom: pd.DataFrame,
    *,
    eye: EyeSide,
    frame_width: float | None = None,
) -> pd.DataFrame:
    """Copy width/height/phi from ``geom`` onto ``base`` rows matched by eye_frame.

    When ``base`` centers look like a horizontal mirror of ``geom`` centers
    (common after Verify X-flip), also reflect ``phi`` with ``π - phi`` so the
    overlay matches the displayed image coordinates.
    """
    g = _normalize_fit_df(geom, eye)
    if "eye_frame" not in base.columns or "eye_frame" not in g.columns:
        raise ValueError("both tables need an eye_frame column for geometry merge")
    cols = [c for c in ("width", "height", "phi", "major_ax", "minor_ax", "ratio") if c in g.columns]
    if not cols:
        raise ValueError("geometry table has no width/height/phi columns")
    slim = g[["eye_frame", *cols]].drop_duplicates(subset=["eye_frame"], keep="first")

    # Detect historical x-flip between upstream fits and verified eye tables.
    reflect_phi = False
    if (
        frame_width is not None
        and frame_width > 0
        and "center_x" in base.columns
        and "center_x" in g.columns
    ):
        joined = base[["eye_frame", "center_x"]].merge(
            g[["eye_frame", "center_x"]],
            on="eye_frame",
            how="inner",
            suffixes=("_base", "_geom"),
        )
        if len(joined) > 20:
            xb = pd.to_numeric(joined["center_x_base"], errors="coerce")
            xg = pd.to_numeric(joined["center_x_geom"], errors="coerce")
            both = xb.notna() & xg.notna()
            if int(both.sum()) > 20:
                mirror_err = (float(frame_width) - xg[both] - xb[both]).abs().median()
                direct_err = (xg[both] - xb[both]).abs().median()
                reflect_phi = bool(mirror_err + 1.0 < direct_err)

    if reflect_phi and "phi" in slim.columns:
        slim = slim.copy()
        slim["phi"] = (np.pi - pd.to_numeric(slim["phi"], errors="coerce")) % np.pi

    out = base.copy()
    drop_cols = [c for c in cols if c in out.columns]
    out = out.drop(columns=drop_cols)
    out = out.merge(slim, on="eye_frame", how="left")
    return out


def load_overlay_eye_df(
    analysis_path: Path | str,
    source: OverlaySource,
    eye: EyeSide,
    *,
    frame_width: float | None = None,
) -> tuple[pd.DataFrame, int]:
    """Load one eye's overlay dataframe + geometry version for ``source``."""
    analysis = Path(analysis_path)
    eye_name = "left_eye_data.csv" if eye == "left" else "right_eye_data.csv"
    fit_name = "le_df.csv" if eye == "left" else "re_df.csv"

    if source.kind == "hybrid":
        main_path = analysis / eye_name
        geom_path = source.directory / fit_name
        if not main_path.is_file():
            raise FileNotFoundError(f"Missing {main_path} for hybrid overlay")
        if not geom_path.is_file():
            raise FileNotFoundError(f"Missing {geom_path} for hybrid overlay")
        base = pd.read_csv(main_path)
        geom = pd.read_csv(geom_path)
        return (
            _merge_geometry(base, geom, eye=eye, frame_width=frame_width),
            int(source.geometry_version),
        )

    if source.kind == "upstream":
        path = analysis / fit_name
        if not path.is_file():
            raise FileNotFoundError(path)
        return _normalize_fit_df(pd.read_csv(path), eye), int(source.geometry_version)

    # main or subdir: prefer eye_data, else fit df
    eye_path = source.directory / eye_name
    fit_path = source.directory / fit_name
    if eye_path.is_file():
        df = pd.read_csv(eye_path)
        ver = source.geometry_version
        if source.kind == "main" and not (source.directory / ELLIPSE_GEOMETRY_META_FILENAME).is_file():
            ver = _infer_phi_geometry_version(df, default=ver)
        return df, int(ver)
    if fit_path.is_file():
        return _normalize_fit_df(pd.read_csv(fit_path), eye), int(source.geometry_version)
    raise FileNotFoundError(f"No overlay table for {eye} under {source.directory}")


def promote_overlay_source_to_main(
    analysis_path: Path | str,
    source: OverlaySource,
    *,
    frame_width: float | None = None,
) -> dict[str, Any]:
    """Move current main ellipse tables to ``legacy/`` and install ``source`` as main.

    Returns a small report dict. Does not delete originals outside ``legacy/``.
    """
    analysis = Path(analysis_path)
    if source.key == "main":
        raise ValueError("Selected source is already main.")

    legacy = analysis / "legacy"
    legacy.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stamped = legacy / f"promoted_away_{stamp}"
    stamped.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    for name in PROMOTE_FILENAMES:
        src = analysis / name
        if src.is_file():
            dest = stamped / name
            shutil.move(str(src), str(dest))
            moved.append(name)

    copied: list[str] = []
    install_dir = source.directory
    for name in ("le_df.csv", "re_df.csv", "left_eye_data.csv", "right_eye_data.csv"):
        src = install_dir / name
        if src.is_file():
            shutil.copy2(src, analysis / name)
            copied.append(name)

    # Rebuild eye tables by merging geometry onto legacy eye_data when needed.
    rebuilt: list[str] = []
    for eye, eye_name, fit_name in (
        ("left", "left_eye_data.csv", "le_df.csv"),
        ("right", "right_eye_data.csv", "re_df.csv"),
    ):
        eye_out = analysis / eye_name
        fit_path = analysis / fit_name
        legacy_eye = stamped / eye_name
        if not fit_path.is_file() or not legacy_eye.is_file():
            continue
        # Always rebuild from legacy centers + new geometry for hybrid/v2 promotes
        # so verified centers / Kerr columns are preserved.
        if source.kind in ("hybrid", "subdir") or eye_name not in copied:
            base = pd.read_csv(legacy_eye)
            geom = pd.read_csv(fit_path)
            merged = _merge_geometry(
                base, geom, eye=eye, frame_width=frame_width  # type: ignore[arg-type]
            )
            merged.to_csv(eye_out, index=True)
            rebuilt.append(eye_name)
            if eye_name not in copied:
                copied.append(eye_name)

    write_ellipse_geometry_meta(
        analysis,
        extra={
            "source": "verify_tab.promote_overlay_source_to_main",
            "promoted_from": source.key,
            "promoted_label": source.label,
            "promoted_utc": datetime.now(timezone.utc).isoformat(),
            "legacy_backup": str(stamped),
        },
    )
    (stamped / "PROMOTION_NOTE.txt").write_text(
        f"Replaced by {source.key} ({source.label}) at {stamp}\n"
        f"Moved files: {', '.join(moved) or '(none)'}\n"
        f"Installed: {', '.join(copied) or '(none)'}\n"
        f"Rebuilt: {', '.join(rebuilt) or '(none)'}\n",
        encoding="utf-8",
    )
    return {
        "legacy_dir": str(stamped),
        "moved": moved,
        "copied": copied,
        "rebuilt": rebuilt,
        "source_key": source.key,
    }
