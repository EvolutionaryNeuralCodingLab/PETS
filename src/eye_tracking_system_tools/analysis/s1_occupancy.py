"""Empirical eccentricity occupancy for Fig S1 (no simulation rebuild)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.event_cache import reload_traces
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

RADIAL_CUT_DEG = 35.0


def collect_ratio_and_radius(tables: EventTables) -> pd.DataFrame:
    """Per-frame ellipse ratio (if present) and Euclidean eccentricity in degrees."""
    tables = reload_traces(tables)
    rows = []
    for bundle in tables.blocks:
        for eye, df in (("L", bundle.left), ("R", bundle.right)):
            if df is None or df.empty:
                continue
            if "k_phi" not in df.columns or "k_theta" not in df.columns:
                continue
            phi = pd.to_numeric(df["k_phi"], errors="coerce").to_numpy()
            th = pd.to_numeric(df["k_theta"], errors="coerce").to_numpy()
            radial = np.hypot(phi, th)
            ratio = np.full(radial.shape, np.nan)
            if "ratio" in df.columns:
                ratio = pd.to_numeric(df["ratio"], errors="coerce").to_numpy()
            elif "major_ax" in df.columns and "minor_ax" in df.columns:
                maj = pd.to_numeric(df["major_ax"], errors="coerce").to_numpy()
                minor = pd.to_numeric(df["minor_ax"], errors="coerce").to_numpy()
                ratio = np.divide(minor, maj, out=np.full_like(minor, np.nan), where=maj > 0)
            rows.append(
                pd.DataFrame(
                    {
                        "animal": bundle.spec.animal,
                        "block": bundle.spec.block_num,
                        "eye": eye,
                        "ratio": ratio,
                        "radial_deg": radial,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["animal", "block", "eye", "ratio", "radial_deg"]
    )


def ratio_at_radial_cut(df: pd.DataFrame, *, cut_deg: float = RADIAL_CUT_DEG, band: float = 5.0) -> float:
    if df.empty or "ratio" not in df.columns:
        return float("nan")
    r = pd.to_numeric(df["radial_deg"], errors="coerce")
    near = df.loc[(r >= cut_deg - band) & (r <= cut_deg + band), "ratio"]
    near = pd.to_numeric(near, errors="coerce").dropna()
    if near.empty:
        return float("nan")
    return float(near.median())


def export_s1_occupancy(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
) -> dict[str, Path]:
    bundle = begin_plot_bundle(
        out_dir,
        "s1_occupancy",
        kind="s1_occupancy",
        tables=tables,
        logic_key="s1_occupancy",
        params=dict(getattr(tables, "params", {}) or {}),
    )
    figures_dir, metadata_dir = bundle.plots_dir, bundle.metadata_dir
    df = collect_ratio_and_radius(tables)
    df.to_csv(metadata_dir / "s1_empirical_ratio_samples.csv", index=False)
    vals = (
        pd.to_numeric(df["radial_deg"], errors="coerce").dropna().to_numpy()
        if not df.empty
        else np.array([], dtype=float)
    )

    fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
    if vals.size:
        hi = float(np.nanpercentile(vals, 99.5))
        if not np.isfinite(hi) or hi <= 0:
            hi = float(np.nanmax(vals)) if vals.size else RADIAL_CUT_DEG * 1.5
        ax.hist(
            vals,
            bins=50,
            range=(0.0, max(hi, RADIAL_CUT_DEG * 1.2)),
            color="#0072B2",
            edgecolor="black",
            density=True,
        )
    ax.axvline(RADIAL_CUT_DEG, color="#D55E00", lw=1.2, ls="--", label=f"±{RADIAL_CUT_DEG:.0f}°")
    ax.legend(fontsize=6, frameon=False)
    ax.set_xlabel("Eccentricity [deg]", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    p = figures_dir / "s1_empirical_ratio_hist.pdf"
    fig.tight_layout()
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    finish_plot_bundle(bundle)
    return {p.name: p}
