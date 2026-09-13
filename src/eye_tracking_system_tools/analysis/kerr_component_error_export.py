"""S1 Kerr leftover-error bars: per-animal (S1b) and overall φ/θ (S1c)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.paper_mpl_style import TRACE_L, TRACE_R, apply_paper_style
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle

COHORT_COLORS = {"L": TRACE_L, "R": TRACE_R}
COHORT_LABELS = {"L": "Left", "R": "Right"}
AXIS_ORDER = ("phi", "theta", "hypot")
EYE_ORDER = ("L", "R")
PICKLE_NAME = "kerr_component_error.pkl"

KERR_OUTPUTS = Path(__file__).resolve().parents[3] / "development" / "kerr_relative_error" / "outputs"


def animal_level_from_blocks(per_block: pd.DataFrame) -> pd.DataFrame:
    """Animal-level mean |Δ| (unweighted mean of that animal's blocks)."""
    rows: list[dict[str, Any]] = []
    for (animal, eye, axis_name), g in per_block.groupby(["animal", "eye", "axis"], sort=False):
        mean_abs = pd.to_numeric(g["mean_abs_deg"], errors="coerce")
        rows.append(
            {
                "animal": animal,
                "eye": eye,
                "axis": axis_name,
                "n_blocks": int(g["block_key"].nunique()) if "block_key" in g.columns else int(len(g)),
                "mean_abs_deg": float(mean_abs.mean()),
                "std_of_block_mean_abs_deg": float(mean_abs.std(ddof=1)) if len(g) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def load_kerr_cohort_tables(
    per_block_csv: Path | str | None = None,
    across_csv: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load per-block, animal-level, and across-animal Kerr |Δ| tables."""
    per_block_path = Path(per_block_csv) if per_block_csv else KERR_OUTPUTS / "cohort_component_error_per_block.csv"
    across_path = Path(across_csv) if across_csv else KERR_OUTPUTS / "cohort_component_error_across_animals.csv"
    per_block = pd.read_csv(per_block_path)
    animal_level = animal_level_from_blocks(per_block)
    across = pd.read_csv(across_path)
    return per_block, animal_level, across


def figure_s1b_per_animal(
    per_block: pd.DataFrame,
    animal_level: pd.DataFrame,
    *,
    seed: int = 0,
) -> plt.Figure:
    """Per-animal L/R bars with per-block dots (published S1b)."""
    animals = [a for a in animal_level["animal"].drop_duplicates().tolist()]
    x = np.arange(len(animals))
    width = 0.35
    panel_title = {
        "phi": "φ",
        "theta": "θ",
        "hypot": "| (Δφ, Δθ) |",
    }
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), dpi=150, sharey=True)
    rng = np.random.default_rng(seed)
    for ax, axis_name in zip(axes, AXIS_ORDER):
        for i, side in enumerate(EYE_ORDER):
            g_an = (
                animal_level[(animal_level["eye"] == side) & (animal_level["axis"] == axis_name)]
                .set_index("animal")
                .reindex(animals)
            )
            g_blk = per_block[(per_block["eye"] == side) & (per_block["axis"] == axis_name)]
            offset = (i - 0.5) * width
            yerr = g_an["std_of_block_mean_abs_deg"].to_numpy(dtype=float)
            ax.bar(
                x + offset,
                g_an["mean_abs_deg"].to_numpy(dtype=float),
                width,
                yerr=np.where(np.isfinite(yerr), yerr, 0.0),
                color=COHORT_COLORS[side],
                alpha=0.75,
                label=COHORT_LABELS[side],
                capsize=2.5,
                error_kw={"lw": 0.8},
            )
            for j, animal in enumerate(animals):
                ys = g_blk.loc[g_blk["animal"] == animal, "mean_abs_deg"].to_numpy(dtype=float)
                jitter = rng.uniform(-0.08, 0.08, size=ys.size)
                ax.scatter(
                    np.full(ys.shape, x[j] + offset) + jitter,
                    ys,
                    s=12,
                    color="0.15",
                    zorder=3,
                    linewidths=0,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(animals, rotation=30, ha="right", fontsize=7)
        ax.set_title(panel_title[axis_name], fontsize=10)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.0, color="0.6", lw=0.5)
    axes[0].set_ylabel(
        "mean |Δ| (deg)\nbars = animal mean of blocks"
        + "\ndots = blocks; whiskers = SD across blocks",
        fontsize=7,
    )
    axes[2].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Leftover Kerr shift after rest-zero  |  all ready blocks / animal",
        fontsize=10,
        y=1.03,
    )
    fig.tight_layout()
    return fig


def figure_s1c_overall(across: pd.DataFrame) -> plt.Figure:
    """Two-bar overall φ/θ mean |Δ| (animal as unit), manuscript 0.84 / 1.19 values."""
    both = across[(across["eye"] == "both") & (across["axis"].isin(["phi", "theta"]))].copy()
    both["axis"] = pd.Categorical(both["axis"], ["phi", "theta"], ordered=True)
    both = both.sort_values("axis")
    labels = ["φ", "θ"]
    y = both["mean_of_mean_abs_deg"].to_numpy(dtype=float)
    yerr = both["std_of_mean_abs_deg"].to_numpy(dtype=float)
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(2.6, 2.8), dpi=150)
    x = np.arange(len(labels))
    ax.bar(
        x,
        y,
        yerr=np.where(np.isfinite(yerr), yerr, 0.0),
        color=["#0072B2", "#D55E00"],
        alpha=0.85,
        capsize=3,
        error_kw={"lw": 0.9},
        width=0.65,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("mean |Δ| (deg)\nerror = SD across animals", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    n = int(both["n_animals"].iloc[0]) if "n_animals" in both.columns and len(both) else 0
    ax.set_title(f"Eye-pooled leftover error  (n = {n} animals)", fontsize=9)
    fig.tight_layout()
    return fig


def kerr_payload(
    per_block: pd.DataFrame,
    animal_level: pd.DataFrame,
    across: pd.DataFrame,
) -> dict[str, Any]:
    both_phi = across[(across["eye"] == "both") & (across["axis"] == "phi")]
    both_th = across[(across["eye"] == "both") & (across["axis"] == "theta")]
    return {
        "per_block": per_block.to_dict(orient="list"),
        "animal_level": animal_level.to_dict(orient="list"),
        "across": across.to_dict(orient="list"),
        "colors": dict(COHORT_COLORS),
        "phi_mean": float(both_phi["mean_of_mean_abs_deg"].iloc[0]) if not both_phi.empty else float("nan"),
        "phi_sd": float(both_phi["std_of_mean_abs_deg"].iloc[0]) if not both_phi.empty else float("nan"),
        "theta_mean": float(both_th["mean_of_mean_abs_deg"].iloc[0]) if not both_th.empty else float("nan"),
        "theta_sd": float(both_th["std_of_mean_abs_deg"].iloc[0]) if not both_th.empty else float("nan"),
        "n_animals": int(both_phi["n_animals"].iloc[0]) if not both_phi.empty else 0,
        "s1b_pdf": "cohort_component_error_across_animals.pdf",
        "s1c_pdf": "cohort_component_error_overall_phi_theta.pdf",
    }


def export_s1_kerr_bars(
    out_dir: Path | str,
    *,
    per_block_csv: Path | str | None = None,
    across_csv: Path | str | None = None,
    show: bool = False,
    plot_id: str = "kerr_component_error",
) -> dict[str, Path]:
    """Write S1b + S1c PDFs, pickle, and replot sidecars."""
    per_block, animal_level, across = load_kerr_cohort_tables(per_block_csv, across_csv)
    payload = kerr_payload(per_block, animal_level, across)
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="kerr_component_error",
        logic_key="kerr_component_error",
        params={"source_csvs": "cohort_component_error_per_block + across_animals"},
        extra={"n_animals": payload["n_animals"]},
    )
    fig_b = figure_s1b_per_animal(per_block, animal_level)
    pdf_b = bundle.plots_dir / payload["s1b_pdf"]
    fig_b.savefig(pdf_b, format="pdf", bbox_inches="tight")
    show_and_close(fig_b, show)
    fig_c = figure_s1c_overall(across)
    pdf_c = bundle.plots_dir / payload["s1c_pdf"]
    fig_c.savefig(pdf_c, format="pdf", bbox_inches="tight")
    show_and_close(fig_c, show)

    pkl = bundle.metadata_dir / PICKLE_NAME
    write_pickle_with_meta(
        payload,
        pkl,
        meta={
            "n_animals": payload["n_animals"],
            "phi_mean": payload["phi_mean"],
            "theta_mean": payload["theta_mean"],
        },
        entrypoint="eye_tracking_system_tools.analysis.kerr_component_error_export.export_s1_kerr_bars",
    )
    per_block.to_csv(bundle.metadata_dir / "cohort_component_error_per_block.csv", index=False)
    across.to_csv(bundle.metadata_dir / "cohort_component_error_across_animals.csv", index=False)
    finish_plot_bundle(bundle)
    return {
        payload["s1b_pdf"]: pdf_b,
        payload["s1c_pdf"]: pdf_c,
        PICKLE_NAME: pkl,
        "_bundle_dir": bundle.bundle_dir,
    }
