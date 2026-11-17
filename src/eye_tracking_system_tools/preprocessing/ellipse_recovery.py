"""Ellipse geometry recovery: inventory, canonicalize, versioned outputs.

Conservative, resumable recovery for multi-block eye tables. Originals are
never overwritten; repaired artifacts land under ``analysis/ellipse_v2/``.

Usage::

    python -m eye_tracking_system_tools.preprocessing.ellipse_recovery \\
        inventory --root /Volumes/Data-1/Nimrod/experiments --out ledger.csv

    python -m eye_tracking_system_tools.preprocessing.ellipse_recovery \\
        repair --ledger ledger.csv --dry-run

    python -m eye_tracking_system_tools.preprocessing.ellipse_recovery \\
        repair --ledger ledger.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from eye_tracking_system_tools.preprocessing.ellipse_fit import (
    ELLIPSE_GEOMETRY_VERSION,
    canonicalize_ellipse_axes_phi,
)
from eye_tracking_system_tools.preprocessing.ellipse_geometry_meta import (
    write_ellipse_geometry_meta,
)

LedgerClass = Literal[
    "auto_repairable",
    "requires_refit",
    "requires_info",
    "manual_review",
    "already_compliant",
]

EYE_TABLE_NAMES = (
    "left_eye_data.csv",
    "right_eye_data.csv",
    "le_df.csv",
    "re_df.csv",
)


@dataclass
class BlockLedgerEntry:
    block_id: str
    block_path: str
    classification: LedgerClass
    evidence: str
    action: str
    ambiguity: str = ""
    affected_outputs: list[str] = field(default_factory=list)
    validation: str = "pending"
    kerr_recompute: str = "no"
    phi_p05: float | None = None
    phi_p95: float | None = None
    height_gt_width_frac: float | None = None
    n_phi: int = 0
    has_dlc: bool = False
    has_eye_tables: bool = False
    has_syncfree: bool = False
    geometry_version: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["affected_outputs"] = ";".join(self.affected_outputs)
        return row


def _iter_block_dirs(root: Path) -> Iterable[Path]:
    root = Path(root)
    if (root / "analysis").is_dir() and root.name.startswith("block_"):
        yield root
        return
    for p in sorted(root.rglob("block_*")):
        if p.is_dir() and (p / "analysis").is_dir():
            yield p


def _block_id(block_path: Path) -> str:
    parts = block_path.parts
    # Prefer animal/date/block
    try:
        i = next(i for i, p in enumerate(parts) if p.startswith("block_"))
        animal = parts[i - 2] if i >= 2 else ""
        date = parts[i - 1] if i >= 1 else ""
        return f"{animal}/{date}/{parts[i]}"
    except StopIteration:
        return str(block_path)


def _phi_stats(df: pd.DataFrame) -> tuple[float | None, float | None, float | None, int]:
    if "phi" not in df.columns:
        return None, None, None, 0
    phi = pd.to_numeric(df["phi"], errors="coerce").to_numpy(dtype=float)
    finite = phi[np.isfinite(phi)]
    n = int(finite.size)
    if n == 0:
        return None, None, None, 0
    p05 = float(np.percentile(finite, 5))
    p95 = float(np.percentile(finite, 95))
    h_gt_w = None
    if "width" in df.columns and "height" in df.columns:
        w = pd.to_numeric(df["width"], errors="coerce")
        h = pd.to_numeric(df["height"], errors="coerce")
        mask = w.notna() & h.notna()
        if mask.any():
            h_gt_w = float((h[mask] > w[mask]).mean())
    return p05, p95, h_gt_w, n


def _looks_like_raw_fitter_radians(p05: float | None, p95: float | None, n: int) -> bool:
    if n == 0 or p05 is None or p95 is None:
        return False
    # Legacy LsqEllipse phi ∈ (~-π/4, π/4)
    return abs(p05) <= 1.0 and abs(p95) <= 1.0 and max(abs(p05), abs(p95)) <= (np.pi / 4 + 0.15)


def _looks_like_degree_scale(p05: float | None, p95: float | None, n: int) -> bool:
    if n == 0 or p05 is None or p95 is None:
        return False
    return max(abs(p05), abs(p95)) > 2.0  # larger than ~115° in radians → degrees-ish


def _has_dlc(block_path: Path) -> bool:
    eye = block_path / "eye_videos"
    if not eye.is_dir():
        return False
    return any(eye.rglob("*DLC*.csv"))


def _has_syncfree(block_path: Path) -> bool:
    eye = block_path / "eye_videos"
    if not eye.is_dir():
        return False
    return any(eye.rglob("*syncfree*"))


def classify_block(block_path: Path) -> BlockLedgerEntry:
    block_path = Path(block_path)
    analysis = block_path / "analysis"
    bid = _block_id(block_path)
    has_dlc = _has_dlc(block_path)
    has_syncfree = _has_syncfree(block_path)

    version = None
    meta_path = analysis / "ellipse_geometry_meta.json"
    if meta_path.is_file():
        try:
            version = int(json.loads(meta_path.read_text())["ellipse_geometry_version"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            version = None

    verified = []
    for name in ("left_eye_data.csv", "right_eye_data.csv"):
        p = analysis / name
        if p.is_file():
            verified.append(p)
    upstream = []
    for name in ("le_df.csv", "re_df.csv"):
        p = analysis / name
        if p.is_file():
            upstream.append(p)

    if not verified and not upstream:
        return BlockLedgerEntry(
            block_id=bid,
            block_path=str(block_path),
            classification="requires_info" if not has_dlc else "requires_refit",
            evidence="no left/right_eye_data.csv or le_df/re_df.csv",
            action="refit" if has_dlc else "locate_inputs",
            ambiguity="missing eye tables" if has_dlc else "missing DLC and eye tables",
            has_dlc=has_dlc,
            has_eye_tables=False,
            has_syncfree=has_syncfree,
            geometry_version=version,
            kerr_recompute="yes (refit)" if has_dlc else "no",
        )

    # Prefer upstream fit tables for auto-canonicalize when available.
    prefer = upstream if upstream else verified
    frames = [pd.read_csv(p) for p in prefer]
    df = pd.concat(frames, ignore_index=True)
    p05, p95, h_gt_w, n = _phi_stats(df)

    verified_degree = False
    if verified:
        vdf = pd.concat([pd.read_csv(p) for p in verified], ignore_index=True)
        vp05, vp95, _, vn = _phi_stats(vdf)
        verified_degree = _looks_like_degree_scale(vp05, vp95, vn)

    if version == ELLIPSE_GEOMETRY_VERSION and (h_gt_w is None or h_gt_w < 1e-6):
        return BlockLedgerEntry(
            block_id=bid,
            block_path=str(block_path),
            classification="already_compliant",
            evidence=f"meta v{version}; height>width={h_gt_w}",
            action="skip",
            has_dlc=has_dlc,
            has_eye_tables=True,
            has_syncfree=has_syncfree,
            geometry_version=version,
            phi_p05=p05,
            phi_p95=p95,
            height_gt_width_frac=h_gt_w,
            n_phi=n,
            validation="pass",
        )

    if upstream and _looks_like_raw_fitter_radians(p05, p95, n):
        ambiguity = ""
        action = "canonicalize_upstream"
        if verified_degree:
            ambiguity = (
                "verified left/right_eye_data look degree-edited; "
                "canonicalize le_df/re_df only; do not reverse Verify history"
            )
            action = "canonicalize_upstream; manual_review_verified"
        return BlockLedgerEntry(
            block_id=bid,
            block_path=str(block_path),
            classification="auto_repairable",
            evidence=(
                f"upstream raw-fitter radian phi (p05={p05}, p95={p95}); "
                f"height>width frac={h_gt_w}; verified_degree={verified_degree}"
            ),
            action=action,
            ambiguity=ambiguity,
            has_dlc=has_dlc,
            has_eye_tables=True,
            has_syncfree=has_syncfree,
            geometry_version=version,
            phi_p05=p05,
            phi_p95=p95,
            height_gt_width_frac=h_gt_w,
            n_phi=n,
            kerr_recompute="no",
        )

    if _looks_like_degree_scale(p05, p95, n) or verified_degree:
        return BlockLedgerEntry(
            block_id=bid,
            block_path=str(block_path),
            classification="manual_review",
            evidence=f"phi range suggests degrees or Verify edits (p05={p05}, p95={p95})",
            action="manual_review_or_refit",
            ambiguity="do not infer units from angle range alone",
            has_dlc=has_dlc,
            has_eye_tables=True,
            has_syncfree=has_syncfree,
            geometry_version=version,
            phi_p05=p05,
            phi_p95=p95,
            height_gt_width_frac=h_gt_w,
            n_phi=n,
            kerr_recompute="maybe",
        )

    if _looks_like_raw_fitter_radians(p05, p95, n):
        return BlockLedgerEntry(
            block_id=bid,
            block_path=str(block_path),
            classification="auto_repairable",
            evidence=(
                f"raw-fitter radian phi (p05={p05}, p95={p95}); "
                f"height>width frac={h_gt_w}"
            ),
            action="canonicalize",
            has_dlc=has_dlc,
            has_eye_tables=True,
            has_syncfree=has_syncfree,
            geometry_version=version,
            phi_p05=p05,
            phi_p95=p95,
            height_gt_width_frac=h_gt_w,
            n_phi=n,
            kerr_recompute="no",
        )

    return BlockLedgerEntry(
        block_id=bid,
        block_path=str(block_path),
        classification="manual_review",
        evidence=f"ambiguous phi stats p05={p05} p95={p95} n={n}",
        action="manual_review",
        ambiguity="phi distribution not clearly raw radians or degrees",
        has_dlc=has_dlc,
        has_eye_tables=True,
        has_syncfree=has_syncfree,
        geometry_version=version,
        phi_p05=p05,
        phi_p95=p95,
        height_gt_width_frac=h_gt_w,
        n_phi=n,
    )


def inventory_root(root: Path) -> list[BlockLedgerEntry]:
    return [classify_block(p) for p in _iter_block_dirs(Path(root))]


def canonicalize_eye_table(df: pd.DataFrame) -> pd.DataFrame:
    """Swap axes + add π/2 when height > width; normalize phi mod π."""
    out = df.copy()
    if not {"width", "height", "phi"}.issubset(out.columns):
        return out
    w = pd.to_numeric(out["width"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(out["height"], errors="coerce").to_numpy(dtype=float)
    phi = pd.to_numeric(out["phi"], errors="coerce").to_numpy(dtype=float)
    w2, h2, phi2 = canonicalize_ellipse_axes_phi(w, h, phi)
    out["width"] = w2
    out["height"] = h2
    out["phi"] = phi2
    if "major_ax" in out.columns or "minor_ax" in out.columns:
        out["major_ax"] = out["width"]
        out["minor_ax"] = out["height"]
        if "ratio" in out.columns:
            out["ratio"] = out["major_ax"] / out["minor_ax"]
    return out


def repair_block(
    entry: BlockLedgerEntry,
    *,
    dry_run: bool = True,
) -> BlockLedgerEntry:
    """Write versioned repaired tables for an auto-repairable block."""
    block_path = Path(entry.block_path)
    analysis = block_path / "analysis"
    out_dir = analysis / "ellipse_v2"
    entry = BlockLedgerEntry(**asdict(entry))

    if entry.classification == "already_compliant":
        entry.validation = "pass"
        entry.action = "skip"
        return entry

    if entry.classification != "auto_repairable":
        entry.validation = "skipped"
        return entry

    sources = [
        analysis / "le_df.csv",
        analysis / "re_df.csv",
    ]
    # Only rewrite verified eye tables when they look like raw radians too.
    for name in ("left_eye_data.csv", "right_eye_data.csv"):
        p = analysis / name
        if not p.is_file():
            continue
        try:
            ed = pd.read_csv(p)
            p05, p95, _, n = _phi_stats(ed)
            if _looks_like_raw_fitter_radians(p05, p95, n):
                sources.append(p)
        except Exception:
            continue
    written: list[str] = []
    if dry_run:
        for src in sources:
            if src.is_file():
                written.append(str(out_dir / src.name))
        entry.affected_outputs = written
        entry.validation = "dry_run"
        return entry

    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        if not src.is_file():
            continue
        df = pd.read_csv(src)
        repaired = canonicalize_eye_table(df)
        dest = out_dir / src.name
        repaired.to_csv(dest, index=False)
        written.append(str(dest))

    # Preserve Kerr sidecars by copy (orientation-only: no recompute)
    for name in (
        "left_kerr_angle_raw_verified.csv",
        "right_kerr_angle_raw_verified.csv",
        "self_kerr_refs.csv",
        "left_eye_data_degrees_raw_verified.csv",
        "right_eye_data_degrees_raw_verified.csv",
    ):
        src = analysis / name
        if src.is_file():
            dest = out_dir / name
            shutil.copy2(src, dest)
            written.append(str(dest))

    meta_extra = {
        "source": "ellipse_recovery.canonicalize",
        "block_id": entry.block_id,
        "recovered_utc": datetime.now(timezone.utc).isoformat(),
        "orientation_only": True,
        "kerr_recompute": False,
        "original_analysis": str(analysis),
    }
    write_ellipse_geometry_meta(out_dir, extra=meta_extra)
    provenance = {
        "block_id": entry.block_id,
        "classification": entry.classification,
        "evidence": entry.evidence,
        "action": "canonicalize",
        "written": written,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "recovery_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    entry.affected_outputs = written
    entry.validation = "pass"
    return entry


def write_ledger(entries: list[BlockLedgerEntry], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([e.to_row() for e in entries]).to_csv(path, index=False)


def read_ledger(path: Path) -> list[BlockLedgerEntry]:
    df = pd.read_csv(path)
    entries: list[BlockLedgerEntry] = []
    for _, row in df.iterrows():
        affected = row.get("affected_outputs", "")
        if isinstance(affected, float) and np.isnan(affected):
            affected_list: list[str] = []
        else:
            affected_list = [x for x in str(affected).split(";") if x]
        entries.append(
            BlockLedgerEntry(
                block_id=str(row["block_id"]),
                block_path=str(row["block_path"]),
                classification=row["classification"],
                evidence=str(row.get("evidence", "")),
                action=str(row.get("action", "")),
                ambiguity=str(row.get("ambiguity", "") or ""),
                affected_outputs=affected_list,
                validation=str(row.get("validation", "pending") or "pending"),
                kerr_recompute=str(row.get("kerr_recompute", "no") or "no"),
                phi_p05=_maybe_float(row.get("phi_p05")),
                phi_p95=_maybe_float(row.get("phi_p95")),
                height_gt_width_frac=_maybe_float(row.get("height_gt_width_frac")),
                n_phi=int(row.get("n_phi", 0) or 0),
                has_dlc=bool(row.get("has_dlc", False)),
                has_eye_tables=bool(row.get("has_eye_tables", False)),
                has_syncfree=bool(row.get("has_syncfree", False)),
                geometry_version=(
                    None
                    if pd.isna(row.get("geometry_version"))
                    else int(row.get("geometry_version"))
                ),
            )
        )
    return entries


def _maybe_float(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("inventory", help="Scan blocks and write a ledger CSV")
    inv.add_argument("--root", type=Path, required=True)
    inv.add_argument("--out", type=Path, required=True)

    rep = sub.add_parser("repair", help="Apply canonicalize repairs from a ledger")
    rep.add_argument("--ledger", type=Path, required=True)
    rep.add_argument("--dry-run", action="store_true")
    rep.add_argument(
        "--out-ledger",
        type=Path,
        default=None,
        help="Optional updated ledger path (default: overwrite --ledger)",
    )
    rep.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional substring filter on block_id",
    )

    args = parser.parse_args(argv)
    if args.cmd == "inventory":
        entries = inventory_root(args.root)
        write_ledger(entries, args.out)
        counts = pd.Series([e.classification for e in entries]).value_counts().to_dict()
        print(f"Wrote {len(entries)} rows to {args.out}")
        print("class counts:", counts)
        return 0

    if args.cmd == "repair":
        entries = read_ledger(args.ledger)
        updated: list[BlockLedgerEntry] = []
        for e in entries:
            if args.only and args.only not in e.block_id:
                updated.append(e)
                continue
            updated.append(repair_block(e, dry_run=args.dry_run))
        out = args.out_ledger or args.ledger
        write_ledger(updated, out)
        print(f"Updated ledger -> {out} (dry_run={args.dry_run})")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
