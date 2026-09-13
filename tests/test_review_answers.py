"""Review-answer exporters: catalog cosmetics, robustness, ISI, QC, occupancy."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.export_meta import load_params_yaml
from eye_tracking_system_tools.analysis.figure_catalog import CATALOG
from eye_tracking_system_tools.analysis.diagnostics_2e import export_diagnostics_2e
from eye_tracking_system_tools.analysis.figures_2c_2e import export_amplitude_velocity_fit
from eye_tracking_system_tools.analysis.figures_2f_2h_2i import (
    Figure2fPoints,
    collect_figure_2f_points,
    display_data_from_collected,
    export_figure_2h,
    histogram2d_from_points,
    rebin_figure_2f_data,
    resolve_figure_2f_view_limits,
)
from eye_tracking_system_tools.analysis.behavior_state import smooth_behavior_state
from eye_tracking_system_tools.analysis.figures_3d_isi import (
    _compute_blockwise_isis_same_epoch,
    _dedupe_lr_pairs,
    collect_epoch_durations,
    epoch_duration_bin_edges,
    epoch_duration_linear_edges,
    epoch_duration_zoom_xmax_s,
    export_epoch_duration_bin_trials,
    export_epoch_durations,
)
from eye_tracking_system_tools.analysis.figures_3a_3c_vignettes import export_figure_3c
from eye_tracking_system_tools.analysis.pixel_calibration import write_pixel_size
from eye_tracking_system_tools.analysis.species_traces import (
    pick_trace_window,
    time_seconds_from_ms_axis,
)
from eye_tracking_system_tools.analysis.paper_gui import _iter_leaf_widgets
from eye_tracking_system_tools.analysis.pipeline import BlockBundle, EventTables
from eye_tracking_system_tools.analysis.qc_reversals import event_returns_toward_start
from eye_tracking_system_tools.analysis.saccade_robustness import (
    export_noise_floor,
    export_robustness,
    noise_floor,
    shuffle_monocular_fractions,
    threshold_sweep,
    window_sweep,
)
from eye_tracking_system_tools.analysis.s1_occupancy import (
    collect_ratio_and_radius,
    export_s1_occupancy,
    ratio_at_radial_cut,
)
from eye_tracking_system_tools.analysis.corrective_head import (
    _angle_diff_deg,
    corrective_partners,
    export_corrective_head,
)
from eye_tracking_system_tools.figures.plotting_functions import _add_anatomical_nt_dv

REPO = Path(__file__).resolve().parents[1]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "animal",
            "block",
            "eye",
            "saccade_on_ms",
            "saccade_off_ms",
            "net_angular_disp",
            "peak_velocity",
            "head_movement",
            "overall_angle_deg",
        ]
    )


def _tables(
    tmp_path: Path,
    *,
    events: pd.DataFrame,
    left: pd.DataFrame | None = None,
    right: pd.DataFrame | None = None,
    params: dict | None = None,
    animal: str = "PV_126",
    block_num: str = "007",
) -> EventTables:
    block_path = tmp_path / f"block_{block_num}"
    (block_path / "analysis").mkdir(parents=True, exist_ok=True)
    spec = BlockSpec(animal=animal, block_path=block_path, block_num=block_num)
    empty = _empty_events()
    left_df = left if left is not None else pd.DataFrame()
    right_df = right if right is not None else pd.DataFrame()
    l_ev = events.loc[events["eye"] == "L"].reset_index(drop=True) if not events.empty else empty.copy()
    r_ev = events.loc[events["eye"] == "R"].reset_index(drop=True) if not events.empty else empty.copy()
    bundle = BlockBundle(
        spec=spec,
        left=left_df,
        right=right_df,
        left_csv_meta={},
        right_csv_meta={},
        l_saccades=l_ev,
        r_saccades=r_ev,
        all_saccades=events.copy(),
    )
    from eye_tracking_system_tools.analysis.binocular import find_synced_saccades_ms

    synced, nons = find_synced_saccades_ms(events, sync_diff_ms=34.0) if not events.empty else (empty.copy(), empty.copy())
    return EventTables(
        blocks=[bundle],
        all_saccades=events.copy(),
        synced=synced,
        non_synced=nons,
        params=params or {},
    )


def test_figure_2f_yaml_matches_published_caption():
    cfg = load_params_yaml(REPO / "configs" / "analysis_params.yaml")["figure_2f"]
    assert str(cfg["event_mode"]).lower() == "all"
    assert cfg["require_head_stationary"] is True
    assert list(cfg["exclude_animals"]) == []
    assert str(cfg["sample_mode"]) == "contra_window"
    assert float(cfg["contra_sample_ms"]) == 51.0
    assert cfg.get("also_export_s3") is False
    assert cfg.get("auto_view_limits") in (None, False)


def test_figure_2f_auto_view_limits_from_speeds():
    """Mouse 2f: macro xmax from the plotted percentile; micro zooms the origin."""
    right = np.linspace(0.05, 1.24, 2000)
    left = np.linspace(0.05, 1.20, 2000)
    fixed = resolve_figure_2f_view_limits(
        right, left, {"macro_range": [0.0, 0.5], "micro_range": [0.0, 0.1]}
    )
    assert tuple(fixed["macro_range"]) == (0.0, 0.5)
    assert tuple(fixed["micro_range"]) == (0.0, 0.1)

    auto = resolve_figure_2f_view_limits(
        right,
        left,
        {"auto_view_limits": True, "macro_pct": 99.5, "micro_frac_of_macro": 0.2},
    )
    assert auto["macro_range"][0] == 0.0
    assert auto["macro_range"][1] == pytest.approx(1.25)
    assert auto["micro_range"][1] == pytest.approx(0.25)
    assert auto["micro_range"][1] < auto["macro_range"][1]
    assert len(auto["macro_tick_list"]) == 3
    assert len(auto["micro_tick_list"]) == 3
    assert auto["macro_tick_list"][0] == pytest.approx(0.0)
    assert auto["macro_tick_list"][-1] == pytest.approx(1.25)

    rebinned = rebin_figure_2f_data(
        {
            "right_eye_speeds": right,
            "left_eye_speeds": left,
            "weights": np.ones(right.size),
            "bins": 20,
            "macro_range": (0.0, 0.5),
            "micro_range": (0.0, 0.1),
            "auto_view_limits": True,
            "macro_pct": 99.5,
            "micro_frac_of_macro": 0.2,
        }
    )
    assert tuple(rebinned["macro_range"]) == tuple(auto["macro_range"])
    assert rebinned["macro"]["xedges"][-1] == pytest.approx(auto["macro_range"][1])


def test_mouse_figure_2f_yaml_uses_auto_limits():
    cfg = load_params_yaml(REPO / "configs" / "analysis_params_mouse.yaml")["figure_2f"]
    assert cfg.get("auto_view_limits") is True
    assert float(cfg.get("macro_pct", 99.5)) == 99.5


def _eye_trace(times: np.ndarray, spikes: dict[float, float]) -> pd.DataFrame:
    speed = np.full(times.size, 0.01, dtype=float)
    for t_ms, val in spikes.items():
        speed[int(np.argmin(np.abs(times - t_ms)))] = val
    return pd.DataFrame({"ms_axis": times, "angular_speed_r": speed})


def test_figure_2f_one_point_per_binocular_pair(tmp_path: Path):
    """Concurrent L/R → one event-peak point; unpaired keep window contra."""
    frame_ms = 17.0
    times = np.arange(0.0, 2100.0, frame_ms)
    # Decoy spike on R at the L-pair onset: window-sampling would pick this.
    left = _eye_trace(times, {2000.0: 0.34})
    right = _eye_trace(times, {100.0: 99.0, 1000.0: 0.85})
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 4,
            "block": ["007"] * 4,
            "eye": ["L", "R", "L", "R"],
            "saccade_on_ms": [100.0, 110.0, 1000.0, 2000.0],
            "saccade_off_ms": [134.0, 144.0, 1034.0, 2034.0],
            "head_movement": [False] * 4,
            "speed_profile_angular": [[3.4], [5.1], [1.7], [2.55]],
            # Finalized tables already carry pair labels; collector must re-pair.
            "Main": [0, 0, pd.NA, pd.NA],
            "Sub": ["L", "R", pd.NA, pd.NA],
        }
    )
    tables = _tables(tmp_path, events=events, left=left, right=right, params={
        "binocular": {"sync_diff_ms": 34.0},
        "figure_2f": {
            "event_mode": "all",
            "sample_mode": "contra_window",
            "contra_sample_ms": 51.0,
            "require_head_stationary": False,
            "exclude_animals": [],
        },
    })
    collected = collect_figure_2f_points(tables, apply_iqr_clip=False)
    pts = collected.points
    assert len(pts) == 3

    pairs = pts.loc[pts["2f_source"] == "binocular_peaks"]
    monos = pts.loc[pts["2f_source"] == "monocular_window"]
    assert len(pairs) == 1
    assert len(monos) == 2
    assert pairs.iloc[0]["eye"] == "LR"
    assert pairs.iloc[0]["left_peak_v"] == pytest.approx(3.4 / frame_ms)
    assert pairs.iloc[0]["right_peak_v"] == pytest.approx(5.1 / frame_ms)
    # Window decoy on R at t=100 must not replace the R event peak.
    assert pairs.iloc[0]["right_peak_v"] != pytest.approx(99.0 / frame_ms)

    mono_l = monos.loc[monos["eye"] == "L"].iloc[0]
    mono_r = monos.loc[monos["eye"] == "R"].iloc[0]
    assert mono_l["left_peak_v"] == pytest.approx(1.7 / frame_ms)
    assert mono_l["right_peak_v"] == pytest.approx(0.85 / frame_ms)
    assert mono_r["right_peak_v"] == pytest.approx(2.55 / frame_ms)
    assert mono_r["left_peak_v"] == pytest.approx(0.34 / frame_ms)

    only_pairs = collect_figure_2f_points(
        tables, cfg={"event_mode": "binocular"}, apply_iqr_clip=False
    )
    assert len(only_pairs.points) == 1
    assert only_pairs.n_skip_mode == 2

    only_mono = collect_figure_2f_points(
        tables, cfg={"event_mode": "monocular"}, apply_iqr_clip=False
    )
    assert len(only_mono.points) == 2
    assert only_mono.n_skip_mode == 1


def test_catalog_2e_s3_2b_outputs():
    assert "s3" in CATALOG
    assert CATALOG["2e"].outputs == (
        "figure_2e.pdf",
        "figure_2e_per_animal_means.pdf",
        "figure_2e_per_animal_means_concurrent.pdf",
        "figure_2e_per_animal_means_monocular.pdf",
        "figure_2e_all_animals_scatter.pdf",
        "figure_2e_all_animals_density.pdf",
        "figure_2e_all_events_scatter.pdf",
        "figure_2e_concurrent_scatter.pdf",
        "figure_2e_monocular_scatter.pdf",
    )
    assert CATALOG["s3"].outputs == (
        "figure_S3_head_still.pdf",
        "figure_S3_head_moving.pdf",
        "figure_S3_colorbar.pdf",
    )
    assert "2b" in CATALOG


def test_2f_macro_micro_share_vmax():
    rng = np.random.default_rng(0)
    n = 400
    points = pd.DataFrame(
        {
            "right_peak_v": np.concatenate([rng.normal(0.05, 0.01, n), rng.normal(0.3, 0.05, 40)]),
            "left_peak_v": np.concatenate([rng.normal(0.05, 0.01, n), rng.normal(0.3, 0.05, 40)]),
            "weight": np.ones(n + 40),
        }
    )
    collected = Figure2fPoints(
        points=points,
        macro_range=(0.0, 0.5),
        micro_range=(0.0, 0.1),
        bins=40,
        cfg={"macro_tick_list": [0, 0.25, 0.5], "micro_tick_list": [0, 0.05, 0.1]},
    )
    disp = display_data_from_collected(collected)
    macro_peak = float(np.nanmax(disp.macro["norm_counts"]))
    micro_peak = float(np.nanmax(disp.micro["norm_counts"]))
    assert disp.vmax_all == pytest.approx(max(macro_peak, micro_peak))
    assert macro_peak != pytest.approx(micro_peak)


def test_s3_still_moving_share_vmax_not_empty(tmp_path: Path):
    n = 80
    points = pd.DataFrame(
        {
            "right_peak_v": np.linspace(0.05, 0.4, n),
            "left_peak_v": np.linspace(0.05, 0.4, n),
            "weight": np.ones(n),
            "head_movement": [False] * (n // 2) + [True] * (n - n // 2),
        }
    )
    still = Figure2fPoints(
        points=points.loc[~points["head_movement"]].reset_index(drop=True),
        macro_range=(0.0, 0.5),
        micro_range=(0.0, 0.1),
        bins=20,
        cfg={},
    )
    moving = Figure2fPoints(
        points=points.loc[points["head_movement"]].reset_index(drop=True),
        macro_range=(0.0, 0.5),
        micro_range=(0.0, 0.1),
        bins=20,
        cfg={},
    )
    h_still = histogram2d_from_points(still, view="macro")
    h_move = histogram2d_from_points(moving, view="macro")
    vmax = max(float(np.nanmax(h_still["norm_counts"])), float(np.nanmax(h_move["norm_counts"])))
    assert vmax > 0
    empty = histogram2d_from_points(
        Figure2fPoints(
            points=points.iloc[0:0].copy(),
            macro_range=(0.0, 0.5),
            micro_range=(0.0, 0.1),
            bins=20,
            cfg={},
        ),
        view="macro",
    )
    assert empty["norm_counts"].shape == h_still["norm_counts"].shape
    assert float(np.nanmax(empty["norm_counts"])) == 0.0


def test_2e_writes_three_pdfs(tmp_path: Path):
    rng = np.random.default_rng(1)
    n = 80
    on = np.arange(n) * 40.0
    # Pair the first half of L/R couples (5 ms < 34 ms window); leave the rest monocular.
    on[1 : n // 2 : 2] = on[0 : n // 2 : 2] + 5.0
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * (n // 2) + ["PV_62"] * (n // 2),
            "block": ["007"] * n,
            "eye": ["L", "R"] * (n // 2),
            "saccade_on_ms": on,
            "saccade_off_ms": on + 20.0,
            "net_angular_disp": rng.uniform(1.0, 12.0, n),
            "peak_velocity": rng.uniform(0.8, 3.0, n),
            "head_movement": False,
        }
    )
    tables = _tables(
        tmp_path,
        events=events,
        params={"main_sequence": {"min_events_per_bin": 2, "frame_rate_fps": 60.0}},
    )
    export_amplitude_velocity_fit(tables, tmp_path, show=False)
    figures = tmp_path / "plots"
    for name in (
        "figure_2e.pdf",
        "figure_2e_per_animal_means.pdf",
        "figure_2e_per_animal_means_concurrent.pdf",
        "figure_2e_per_animal_means_monocular.pdf",
        "figure_2e_all_animals_scatter.pdf",
        "figure_2e_all_animals_density.pdf",
        "figure_2e_all_events_scatter.pdf",
        "figure_2e_concurrent_scatter.pdf",
        "figure_2e_monocular_scatter.pdf",
    ):
        assert (figures / name).is_file(), name
    meta = tmp_path / "metadata"
    pkl = meta / "amplitude_velocity_linear_fit_bundle.pkl"
    assert pkl.is_file()
    import pickle

    with open(pkl, "rb") as f:
        bundle = pickle.load(f)
    assert set(bundle["classes"]) == {"all", "concurrent", "monocular"}
    assert bundle["classes"]["concurrent"]["n"] > 0
    assert bundle["classes"]["monocular"]["n"] > 0
    assert bundle["pooled"]["cohort"] == "pogona"
    assert bundle["pooled"]["n"] == bundle["classes"]["all"]["n"]
    assert bundle["pooled"]["n"] >= n - 2
    assert set(bundle["pooled"]["animals"]) == {"PV_126", "PV_62"}
    assert (meta / "figure_2e_class_counts.csv").is_file()
    assert np.isfinite(bundle["global_fit"]["r"])
    assert bundle["density"]["method"] == "gaussian_kde"
    assert bundle["density"]["zi"] is not None


def test_diagnostics_2e_writes_length_scatter(tmp_path: Path):
    rng = np.random.default_rng(3)
    n = 80
    n1 = n // 2
    on = np.arange(n) * 40.0
    # Pair first half of L/R couples for concurrent subset.
    on[1 : n // 2 : 2] = on[0 : n // 2 : 2] + 5.0
    lengths = np.array([1, 1, 2, 3, 4, 5, 7, 12] * ((n // 8) + 1), dtype=int)[:n]
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * n1 + ["PV_62"] * (n - n1),
            "block": ["007"] * n,
            "eye": ["L", "R"] * (n // 2),
            "saccade_on_ms": on,
            "saccade_off_ms": on + 20.0,
            "net_angular_disp": rng.uniform(1.0, 12.0, n),
            "peak_velocity": rng.uniform(0.8, 3.0, n),
            "length": lengths,
            "head_movement": False,
        }
    )
    tables = _tables(
        tmp_path,
        events=events,
        params={"main_sequence": {"min_amp_deg": 0.5, "frame_rate_fps": 60.0, "min_events_per_bin": 2}},
    )
    written = export_diagnostics_2e(tables, tmp_path, show=False)
    bundle_dir = tmp_path / "diagnostics_2e"
    plots = bundle_dir / "plots"
    assert (plots / "figure_2e_scatter_by_length.pdf").is_file()
    assert (plots / "2e_all_animals_gray.pdf").is_file()
    assert (plots / "2e_per_animal_gray.pdf").is_file()
    assert (plots / "fig_2e_mono_conc_all.pdf").is_file()
    omit_plots = bundle_dir / "omission_trial" / "plots"
    assert (omit_plots / "2e_all_animals_gray.pdf").is_file()
    assert (omit_plots / "2e_per_animal_gray.pdf").is_file()
    assert (omit_plots / "fig_2e_mono_conc_all.pdf").is_file()
    assert (omit_plots / "figure_2e_scatter_by_length.pdf").is_file()
    pkl = bundle_dir / "metadata" / "diagnostics_2e.pkl"
    assert pkl.is_file()
    assert written["fig_2e_mono_conc_all.pdf"] == plots / "fig_2e_mono_conc_all.pdf"
    assert "omission_trial/2e_all_animals_gray.pdf" in written
    import pickle

    with open(pkl, "rb") as f:
        payload = pickle.load(f)
    assert payload["params"]["cohort"] == "pogona"
    assert int(payload["amp"].size) >= n - 2
    assert set(payload["counts"]) >= {"1", "2", "9+"}
    assert payload["counts"]["1"] > 0
    assert payload["animal_order"] == ["PV_126", "PV_62"]
    assert float(payload["params"]["gray_alpha"]) == 0.5
    assert set(payload["classes"]) == {"all", "concurrent", "monocular"}
    assert payload["classes"]["concurrent"]["n"] > 0
    omit_pkl = bundle_dir / "omission_trial" / "metadata" / "diagnostics_2e.pkl"
    with open(omit_pkl, "rb") as f:
        omit = pickle.load(f)
    assert omit["params"]["omit_length_le"] == 2
    assert int(omit["counts"].get("1", 0)) == 0
    assert int(omit["counts"].get("2", 0)) == 0
    assert int(omit["amp"].size) < int(payload["amp"].size)
    assert (bundle_dir / "metadata" / "length_bin_counts.csv").is_file()
    assert (bundle_dir / "replot.py").is_file()
    logic = (bundle_dir / "metadata" / "LOGIC.md").read_text(encoding="utf-8")
    assert "colored by detector length" in logic
    assert "fig_2e_mono_conc_all" in logic
    omit_logic = (bundle_dir / "omission_trial" / "metadata" / "LOGIC.md").read_text(encoding="utf-8")
    assert "length ≤ 2" in omit_logic or "length <= 2" in omit_logic or "≤ 2" in omit_logic


def test_length_identity_summary_counts_on_line():
    from eye_tracking_system_tools.analysis.diagnostics_2e import (
        length_identity_summary,
        on_length_identity_mask,
    )

    frame_ms = 1000.0 / 60.0
    # length-1: half exactly on V=A/frame_ms, half far above
    amp = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    length = np.array([1, 1, 2, 2], dtype=float)
    vel = np.array(
        [
            amp[0] / (1 * frame_ms),
            amp[1] / (1 * frame_ms) * 2.0,
            amp[2] / (2 * frame_ms),
            amp[3] / (2 * frame_ms) * 3.0,
        ]
    )
    df = pd.DataFrame(
        {
            "net_angular_disp": amp,
            "peak_velocity_deg_per_ms": vel,
            "length": length,
        }
    )
    summary = length_identity_summary(
        df, amp_col="net_angular_disp", frame_ms=frame_ms, rel_tol=0.05, abs_tol=1e-6
    )
    by_bin = summary.set_index("length_bin")
    assert int(by_bin.loc["1", "n"]) == 2
    assert int(by_bin.loc["1", "n_on_identity"]) == 1
    assert int(by_bin.loc["2", "n"]) == 2
    assert int(by_bin.loc["2", "n_on_identity"]) == 1
    assert int(by_bin.loc["ALL", "n_on_identity"]) == 2

    mask = on_length_identity_mask(
        df, amp_col="net_angular_disp", frame_ms=frame_ms, rel_tol=0.01, abs_tol=0.0
    )
    assert mask.dtype == bool
    assert int(mask.sum()) == 2
    kept = df.loc[~mask]
    assert len(kept) == 2


def test_attach_pre_event_amplitude(tmp_path: Path):
    from eye_tracking_system_tools.analysis.diagnostics_2e import attach_pre_event_amplitude

    times = np.arange(0.0, 200.0, 1000.0 / 60.0)
    left = pd.DataFrame(
        {
            "ms_axis": times,
            "k_phi": np.linspace(0.0, 10.0, times.size),
            "k_theta": np.zeros(times.size),
            "center_x": np.zeros(times.size),
            "center_y": np.zeros(times.size),
            "OE_timestamp": times,
        }
    )
    events = pd.DataFrame(
        {
            "animal": ["PV_126"],
            "block": ["007"],
            "eye": ["L"],
            "saccade_on_ms": [times[3]],
            "saccade_off_ms": [times[4]],
            "saccade_start_ind": [3],
            "saccade_end_ind": [4],
            "phi_init_pos": [left.loc[3, "k_phi"]],
            "theta_init_pos": [0.0],
            "phi_end_pos": [left.loc[4, "k_phi"]],
            "theta_end_pos": [0.0],
            "net_angular_disp": [float(left.loc[4, "k_phi"] - left.loc[3, "k_phi"])],
            "peak_velocity": [1.0],
            "length": [1],
            "head_movement": [False],
        }
    )
    tables = _tables(tmp_path, events=events, left=left)
    out = attach_pre_event_amplitude(events, tables)
    assert np.isfinite(out["net_angular_disp_pre"].iloc[0])
    assert out["net_angular_disp_pre"].iloc[0] > out["net_angular_disp"].iloc[0]


def test_2h_marks_origin_pre(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(2)
    n = 30
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * n,
            "block": ["007"] * n,
            "eye": ["L", "R"] * (n // 2),
            "saccade_on_ms": np.arange(n) * 50.0,
            "saccade_off_ms": np.arange(n) * 50.0 + 20.0,
            "delta_phi": rng.normal(0, 2, n),
            "delta_theta": rng.normal(0, 2, n),
            "head_movement": False,
        }
    )
    tables = _tables(tmp_path, events=events)
    seen_pre = []
    orig = plt.Figure.savefig

    def _savefig(self, *args, **kwargs):
        texts = [t.get_text() for ax in self.axes for t in ax.texts]
        xlabels = [ax.get_xlabel() for ax in self.axes]
        seen_pre.append(("pre" in texts, any("post" in s for s in xlabels)))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", _savefig)
    export_figure_2h(tables, tmp_path, show=False)
    assert any(pre and post for pre, post in seen_pre)


def test_anatomical_nt_opposite_for_left_vs_right():
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    _add_anatomical_nt_dv(ax, eye="L")
    pos_l = {t.get_text(): t.get_position()[0] for t in ax.texts}
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    _add_anatomical_nt_dv(ax, eye="R")
    pos_r = {t.get_text(): t.get_position()[0] for t in ax.texts}
    plt.close(fig)
    assert pos_l["N"] > pos_l["T"]
    assert pos_r["N"] < pos_r["T"]
    assert pos_l["N"] > pos_r["N"]


def test_3a_xlabel_seconds():
    from eye_tracking_system_tools.figures.plotting_functions import plot_zoomed_in_with_head_rate

    t = np.arange(0, 2000, 16.67)
    df = pd.DataFrame(
        {
            "ms_axis": t,
            "k_phi": np.zeros(t.size),
            "k_theta": np.zeros(t.size),
            "pupil_diameter": np.ones(t.size),
        }
    )
    fig, axes = plot_zoomed_in_with_head_rate(
        start_time=0.0,
        end_time=1.0,
        left_df=df,
        right_df=df,
        traces=["center_x", "center_y"],
        figure_size=(2.3, 1.8),
        export_path=None,
    )
    assert axes[-1].get_xlabel() == "[s]"
    plt.close(fig)


def test_vignette_widget_walk_finds_nested_start_s():
    class W:
        def __init__(self, description="", children=(), value=0.0):
            self.description = description
            self.children = children
            self.value = value

    start = W("start_s", value=210.0)
    end = W("end_s", value=240.0)
    box = W("", children=(start, end))
    found = {w.description: w.value for w in _iter_leaf_widgets([box])}
    assert found["start_s"] == 210.0
    assert found["end_s"] == 240.0


def test_robustness_threshold_and_window_and_shuffle(tmp_path: Path):
    times_l = np.array([0.0, 200.0, 400.0, 800.0, 1200.0])
    times_r = times_l + 10.0
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 10,
            "block": ["007"] * 10,
            "eye": ["L"] * 5 + ["R"] * 5,
            "saccade_on_ms": np.concatenate([times_l, times_r]),
            "saccade_off_ms": np.concatenate([times_l, times_r]) + 20.0,
            "peak_velocity": [1.2, 1.2, 0.5, 1.2, 1.2, 1.2, 1.2, 0.5, 1.2, 1.2],
            "head_movement": False,
        }
    )
    tables = _tables(tmp_path, events=events)
    thr = threshold_sweep(tables, thresholds=(0.4, 0.8, 1.0), sync_diff_ms=34.0)
    assert list(thr["speed_threshold_deg_per_frame"]) == [0.4, 0.8, 1.0]
    assert thr["monocular_fraction"].notna().all()
    win = window_sweep(tables, windows_ms=(17.0, 34.0, 100.0), threshold=0.8)
    assert win.loc[win["pairing_window_ms"] == 34.0, "monocular_fraction"].iloc[0] < 0.5
    observed, null = shuffle_monocular_fractions(
        tables, n_shuffle=50, rng=np.random.default_rng(0), threshold=0.8, sync_diff_ms=34.0
    )
    assert np.isfinite(observed)
    assert null.size == 50
    assert float(np.nanmean(null)) > observed


def test_isi_drops_cross_epoch_intervals(tmp_path: Path):
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 4,
            "block": ["007"] * 4,
            "eye": ["L"] * 4,
            "saccade_on_ms": [100.0, 200.0, 600.0, 700.0],
            "head_movement": False,
        }
    )
    tables = _tables(tmp_path, events=events)
    spec = tables.blocks[0].spec
    pd.DataFrame(
        {
            "start_time": [0.0, 500.0],
            "end_time": [500.0, 1000.0],
            "annotation": ["quiet", "active"],
        }
    ).to_csv(spec.analysis_path / "block_007_behavior_state.csv", index=False)
    dedup = _dedupe_lr_pairs(
        events.assign(head_movement=False), pair_threshold_ms=60.0
    )
    quiet = _compute_blockwise_isis_same_epoch(
        dedup, tables, min_isi_ms=1.0, require_label="quiet"
    )
    active = _compute_blockwise_isis_same_epoch(
        dedup, tables, min_isi_ms=1.0, require_label="active"
    )
    assert quiet["PV_126"].tolist() == pytest.approx([100.0])
    assert active["PV_126"].tolist() == pytest.approx([100.0])


def test_epoch_durations_use_raw_csv(tmp_path: Path):
    tables = _tables(tmp_path, events=_empty_events())
    spec = tables.blocks[0].spec
    pd.DataFrame(
        {
            "start_time": [0.0, 10_000.0],
            "end_time": [4000.0, 16_000.0],
            "annotation": ["quiet", "active"],
        }
    ).to_csv(spec.analysis_path / "block_007_behavior_state.csv", index=False)
    written = export_epoch_durations(tables, tmp_path, show=False)
    assert "epoch_durations_quiet_log.pdf" in written
    assert "epoch_durations_quiet_linear.pdf" in written
    assert "epoch_durations_active_log.pdf" in written
    assert "epoch_durations_active_linear.pdf" in written
    quiet = written["epoch_durations_quiet_log.pdf"]
    assert quiet.parent.name == "plots"
    assert quiet.parent.parent.name == "epoch_durations"
    assert (quiet.parent.parent / "replot.py").is_file()
    assert (quiet.parent.parent / "metadata" / "epoch_durations.pkl").is_file()
    log_edges = epoch_duration_bin_edges(np.array([4.0, 6.0]), scale="log")
    lin_edges = epoch_duration_bin_edges(np.array([4.0, 6.0]), scale="linear")
    assert log_edges.size == lin_edges.size == 31
    assert lin_edges[0] == pytest.approx(0.0)
    assert lin_edges[-1] == pytest.approx(6.0)
    lin_from_1 = epoch_duration_bin_edges(np.array([4.0, 6.0]), scale="linear", xmin=1.0)
    assert lin_from_1[0] == pytest.approx(1.0)


def test_epoch_duration_bin_trials_triptych(tmp_path: Path):
    tables = _tables(tmp_path, events=_empty_events())
    spec = tables.blocks[0].spec
    pd.DataFrame(
        {
            "start_time": [0.0, 20_000.0, 40_000.0, 100_000.0],
            "end_time": [20_000.0, 40_000.0, 100_000.0, 167_000.0],
            "annotation": ["quiet", "active", "quiet", "active"],
        }
    ).to_csv(spec.analysis_path / "block_007_behavior_state.csv", index=False)
    assert epoch_duration_zoom_xmax_s(np.array([67.0]), round_to=5.0) == pytest.approx(70.0)
    edges = epoch_duration_linear_edges(xmax_s=70.0, bin_width_s=5.0)
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(70.0)
    assert np.allclose(np.diff(edges), 5.0)
    written = export_epoch_duration_bin_trials(
        tables,
        tmp_path,
        show=False,
        smooth_state=False,
        min_duration_s=0.0,
        trials=(
            {
                "name": "width5s_quiet50s",
                "zoom_bin_width_s": 5.0,
                "quiet_full_bin_width_s": 50.0,
                "label": "zoom 5 s",
            },
        ),
    )
    key = "epoch_duration_triptych_width5s_quiet50s.pdf"
    assert key in written
    assert written[key].parent.parent.name == "epoch_duration_bin_trials"
    assert (written[key].parent.parent / "metadata" / "epoch_duration_bin_trials.pkl").is_file()


def test_smooth_behavior_state_bridges_one_second_island():
    raw = pd.DataFrame(
        {
            "start_time": [0.0, 100_000.0, 101_000.0],
            "end_time": [100_000.0, 101_000.0, 200_000.0],
            "annotation": ["quiet", "active", "quiet"],
        }
    )
    sm = smooth_behavior_state(raw, min_active_ms=5000.0, min_quiet_ms=5000.0, gap_bridge_ms=3000.0)
    assert len(sm) == 1
    assert sm.iloc[0]["annotation"] == "quiet"
    assert sm.iloc[0]["end_time"] - sm.iloc[0]["start_time"] == pytest.approx(200_000.0)


def test_epoch_durations_smoothed_bundle_and_drop(tmp_path: Path):
    tables = _tables(tmp_path, events=_empty_events())
    spec = tables.blocks[0].spec
    pd.DataFrame(
        {
            "start_time": [0.0, 100_000.0, 101_000.0],
            "end_time": [100_000.0, 101_000.0, 200_000.0],
            "annotation": ["quiet", "active", "quiet"],
        }
    ).to_csv(spec.analysis_path / "block_007_behavior_state.csv", index=False)
    raw = collect_epoch_durations(tables, smooth_state=False)
    assert sorted(raw["active"]) == pytest.approx([1.0])
    sm = collect_epoch_durations(tables, smooth_state=True, min_duration_s=1.0)
    assert sm["active"] == []
    assert sm["quiet"] == pytest.approx([200.0])
    written = export_epoch_durations(
        tables,
        tmp_path,
        show=False,
        plot_id="epoch_durations_smoothed",
        smooth_state=True,
        min_duration_s=1.0,
        linear_xmin_s=1.0,
    )
    quiet = written["epoch_durations_quiet_log.pdf"]
    assert quiet.parent.parent.name == "epoch_durations_smoothed"


def test_aggregate_pupil_smooth_moves_blip_samples(tmp_path: Path):
    t = np.arange(0.0, 20_000.0, 10.0)
    eye = pd.DataFrame(
        {
            "ms_axis": t,
            "major_ax": np.full(t.size, 10.0),
            "k_phi": np.zeros(t.size),
            "k_theta": np.zeros(t.size),
        }
    )
    tables = _tables(tmp_path, events=_empty_events(), left=eye, right=eye)
    spec = tables.blocks[0].spec
    write_pixel_size(spec.block_path, 0.1, 0.1)
    pd.DataFrame(
        {
            "start_time": [0.0, 10_000.0, 11_000.0],
            "end_time": [10_000.0, 11_000.0, 20_000.0],
            "annotation": ["quiet", "active", "quiet"],
        }
    ).to_csv(spec.analysis_path / "block_007_behavior_state.csv", index=False)
    raw = aggregate_pupil_mm_by_state(tables, smooth_state=False)
    sm = aggregate_pupil_mm_by_state(tables, smooth_state=True)
    assert len(raw["PV_126"]["active"]) > 0
    assert len(sm["PV_126"]["active"]) == 0
    assert len(sm["PV_126"]["quiet"]) > len(raw["PV_126"]["quiet"])


def test_figure_3c_writes_raw_and_smoothed_state_pdfs(tmp_path: Path):
    t = np.arange(200_000.0, 230_000.0, 16.67)
    eye = pd.DataFrame(
        {
            "ms_axis": t,
            "k_phi": np.zeros(t.size),
            "k_theta": np.zeros(t.size),
            "major_ax": np.full(t.size, 10.0),
            "pupil_diameter": np.full(t.size, 1.8),
        }
    )
    tables = _tables(tmp_path, events=_empty_events(), left=eye, right=eye)
    spec = tables.blocks[0].spec
    pd.DataFrame(
        {
            "start_time": [200_000.0, 210_000.0, 211_000.0],
            "end_time": [210_000.0, 211_000.0, 230_000.0],
            "annotation": ["quiet", "active", "quiet"],
        }
    ).to_csv(spec.analysis_path / "block_007_behavior_state.csv", index=False)
    written = export_figure_3c(tables, tmp_path, show=False, start_s=200.0, end_s=230.0)
    assert written["figure_3c_raw.pdf"].is_file()
    assert written["figure_3c_smoothed.pdf"].is_file()
    assert written["figure_3c.pdf"].is_file()
    assert written["figure_3c.pdf"].parent.name == "plots"



def test_species_trace_time_axis_and_nan_window():
    ms = np.arange(0.0, 2000.0, 16.67)
    t_s = time_seconds_from_ms_axis(ms)
    assert t_s[-1] == pytest.approx(ms[-1] / 1000.0, rel=1e-6)
    already_s = np.arange(0.0, 10.0, 0.01667)
    assert time_seconds_from_ms_axis(already_s)[-1] == pytest.approx(already_s[-1], rel=1e-6)

    t = np.arange(0.0, 200.0, 0.05)
    finite = np.zeros(t.size, dtype=bool)
    finite[(t >= 50.0) & (t < 150.0)] = True
    lo, hi = pick_trace_window(t, finite, start_s=10.0, duration_s=80.0)
    assert lo == pytest.approx(50.0, abs=0.06)
    assert hi - lo == pytest.approx(80.0)


def test_reversal_trace_return():
    t = np.arange(0, 400, 10.0)
    phi = np.zeros_like(t)
    # saccade 100→160 ms: 0 → 10 deg, then return toward 0 after offset
    phi[(t >= 100) & (t <= 160)] = np.linspace(0, 10, np.count_nonzero((t >= 100) & (t <= 160)))
    phi[t > 160] = 10.0
    phi[(t > 160) & (t <= 220)] = np.linspace(10, 3, np.count_nonzero((t > 160) & (t <= 220)))
    eye = pd.DataFrame({"ms_axis": t, "k_phi": phi, "k_theta": np.zeros_like(t)})
    row = pd.Series({"saccade_on_ms": 100.0, "saccade_off_ms": 160.0})
    assert event_returns_toward_start(row, eye) is True
    phi2 = phi.copy()
    phi2[t > 160] = 10.0
    eye2 = pd.DataFrame({"ms_axis": t, "k_phi": phi2, "k_theta": np.zeros_like(t)})
    assert event_returns_toward_start(row, eye2) is False


def test_corrective_window_and_angle(tmp_path: Path):
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 4,
            "block": ["007"] * 4,
            "eye": ["L", "R", "L", "R"],
            "saccade_on_ms": [0.0, 50.0, 1000.0, 1100.0],
            "saccade_off_ms": [20.0, 70.0, 1020.0, 1120.0],
            "overall_angle_deg": [0.0, 10.0, 0.0, 180.0],
            "peak_velocity": 1.2,
            "head_movement": False,
        }
    )
    tables = _tables(tmp_path, events=events)
    partners = corrective_partners(tables, window_ms=80.0, angle_deg=45.0, sync_diff_ms=10.0)
    # With 10 ms pairing, none of these are binocular (min dt is 50 ms).
    first = partners.loc[partners["saccade_on_ms"] == 0.0].iloc[0]
    late = partners.loc[partners["saccade_on_ms"] == 1000.0].iloc[0]
    assert bool(first["has_corrective"]) is True
    assert first["latency_ms"] == pytest.approx(50.0)
    assert bool(late["has_corrective"]) is False
    assert _angle_diff_deg(0.0, 180.0) == pytest.approx(180.0)


def test_s1_eccentricity_hist(tmp_path: Path):
    n = 200
    left = pd.DataFrame(
        {
            "k_phi": np.linspace(0, 40, n),
            "k_theta": np.zeros(n),
            "ms_axis": np.arange(n) * 16.67,
        }
    )
    tables = _tables(tmp_path, events=_empty_events(), left=left, right=left.copy())
    written = export_s1_occupancy(tables, tmp_path, show=False)
    assert "s1_empirical_ratio_hist.pdf" in written
    pdf = written["s1_empirical_ratio_hist.pdf"]
    assert pdf.parent.parent.name == "s1_occupancy"
    df = collect_ratio_and_radius(tables)
    ecc = pd.to_numeric(df["radial_deg"], errors="coerce").dropna()
    assert not ecc.empty
    assert float(ecc.max()) >= 35.0
    vline = ratio_at_radial_cut(df, cut_deg=35.0, band=5.0)
    assert not np.isfinite(vline)  # no ellipse-ratio column


def test_corrective_percent_two_bars(tmp_path: Path, monkeypatch):
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 4,
            "block": ["007"] * 4,
            "eye": ["L", "R", "L", "R"],
            "saccade_on_ms": [0.0, 50.0, 1000.0, 1100.0],
            "saccade_off_ms": [20.0, 70.0, 1020.0, 1120.0],
            "overall_angle_deg": [0.0, 10.0, 0.0, 180.0],
            "peak_velocity": 1.2,
            "head_movement": False,
        }
    )
    tables = _tables(tmp_path, events=events)
    ylims = []
    n_bars = []
    orig = plt.Figure.savefig

    def _savefig(self, *args, **kwargs):
        if self.axes and len(self.axes[0].patches) == 2:
            ylims.append(self.axes[0].get_ylim())
            n_bars.append(len(self.axes[0].patches))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", _savefig)
    written = export_corrective_head(tables, tmp_path, show=False)
    assert "corrective_percent.pdf" in written
    assert n_bars and n_bars[0] == 2
    assert ylims and ylims[0][1] == pytest.approx(100.0)


def test_noise_floor_and_fast_export(tmp_path: Path):
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 2000.0, 16.67)
    phi = rng.normal(0.0, 0.04, t.size)
    phi[40:48] += np.linspace(0, 8, 8)
    left = pd.DataFrame({"ms_axis": t, "k_phi": phi, "k_theta": np.zeros_like(t)})
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 4,
            "block": ["007"] * 4,
            "eye": ["L", "R", "L", "R"],
            "saccade_on_ms": [t[40], t[40] + 10.0, 800.0, 810.0],
            "saccade_off_ms": [t[47], t[47] + 10.0, 830.0, 840.0],
            "peak_velocity": 1.2,
            "head_movement": False,
        }
    )
    tables = _tables(tmp_path, events=events, left=left, right=left.copy())
    nf = noise_floor(tables)
    assert not nf.empty
    assert nf["engbert_sigma_deg_per_frame"].notna().any()
    written = export_robustness(tables, tmp_path, show=False, n_shuffle=8)
    assert "robustness_noise_floor.pdf" in written
    assert "robustness_shuffle.pdf" in written
    pdf = written["robustness_noise_floor.pdf"]
    assert pdf.parent.parent.name == "robustness"
    assert (pdf.parent.parent / "replot.py").is_file()

    mouse_written = export_noise_floor(
        tables,
        tmp_path,
        show=False,
        plot_id="mouse_robustness",
        detector_thresholds=(2.5, 3.91, 3.06, 3.43),
    )
    mouse_pdf = mouse_written["robustness_noise_floor.pdf"]
    assert mouse_pdf.parent.parent.name == "mouse_robustness"
    assert pdf != mouse_pdf
    assert (tmp_path / "robustness" / "plots" / "robustness_noise_floor.pdf").is_file()
    thr_csv = tmp_path / "mouse_robustness" / "metadata" / "gui_detector_thresholds.csv"
    assert thr_csv.is_file()
    thr = pd.read_csv(thr_csv)
    assert float(thr["speed_threshold_deg_per_frame"].mean()) == pytest.approx(
        np.mean([2.5, 3.91, 3.06, 3.43])
    )
    assert not (tmp_path / "mouse_robustness" / "plots" / "robustness_threshold_sweep.pdf").is_file()


def test_unified_noise_shared_xmax(tmp_path: Path):
    from eye_tracking_system_tools.analysis.saccade_robustness import (
        UNIFIED_NOISE_PDF,
        UNIFIED_NOISE_PLOT_ID,
        export_unified_noise,
        figure_unified_noise,
        interframe_dangle,
        non_saccade_interframe_dangle,
    )

    t = np.arange(0.0, 500.0, 16.67)
    quiet = pd.DataFrame(
        {"ms_axis": t, "k_phi": np.cumsum(np.random.default_rng(0).normal(0, 0.02, t.size)), "k_theta": 0.0}
    )
    events = pd.DataFrame({"saccade_on_ms": [80.0], "saccade_off_ms": [120.0]})
    kept = non_saccade_interframe_dangle(quiet, events)
    raw = interframe_dangle(quiet)
    assert raw.size == t.size - 1
    assert kept.size < raw.size

    rng = np.random.default_rng(1)
    pools = {
        "rigid": np.abs(rng.normal(0.05, 0.02, 2000)),
        "modular": np.abs(rng.normal(0.06, 0.02, 2000)),
        "turtle": np.abs(rng.normal(0.08, 0.03, 2000)),
        "mouse": np.abs(rng.normal(0.4, 0.15, 2000)),
    }
    written = export_unified_noise(tmp_path, pools=pools, show=False, n_bins=40)
    pdf = written[UNIFIED_NOISE_PDF]
    assert pdf.parent.parent.name == UNIFIED_NOISE_PLOT_ID
    params = yaml.safe_load((pdf.parent.parent / "metadata" / "params.yaml").read_text())
    assert params["xmax_source"] == "mouse"
    fig = figure_unified_noise(pools, xmax=params["xmax"], n_bins=40)
    xlims = [ax.get_xlim() for ax in fig.axes]
    assert all(hi == pytest.approx(params["xmax"]) for _lo, hi in xlims)
    plt.close(fig)


def test_export_raw_interframe_delta(tmp_path: Path):
    from eye_tracking_system_tools.analysis.saccade_robustness import (
        RAW_INTERFRAME_LOGX_PDF,
        RAW_INTERFRAME_PDF,
        RAW_INTERFRAME_PLOT_ID,
        RAYLEIGH_EXCESS_KURTOSIS,
        RAYLEIGH_SKEW,
        export_raw_interframe_delta,
        fit_rayleigh_distance,
        figure_raw_interframe_delta,
    )

    rng = np.random.default_rng(0)
    synth = np.hypot(rng.normal(0, 1.0, 20000), rng.normal(0, 1.0, 20000))
    fit = fit_rayleigh_distance(synth)
    assert fit["B"] == pytest.approx(1.0, rel=0.05)
    assert fit["skew"] == pytest.approx(RAYLEIGH_SKEW, rel=0.15)
    assert fit["excess_kurtosis"] == pytest.approx(RAYLEIGH_EXCESS_KURTOSIS, abs=0.15)

    pools = {
        "rigid": np.abs(rng.normal(0.05, 0.02, 2000)),
        "modular": np.abs(rng.normal(0.06, 0.02, 2000)),
        "turtle": np.abs(rng.normal(0.08, 0.03, 2000)),
        "mouse": np.concatenate(
            [np.abs(rng.normal(0.4, 0.15, 1800)), np.abs(rng.normal(3.0, 0.4, 200))]
        ),
    }
    written = export_raw_interframe_delta(tmp_path, pools=pools, show=False, n_bins=40)
    pdf = written[RAW_INTERFRAME_PDF]
    assert pdf.parent.parent.name == RAW_INTERFRAME_PLOT_ID
    assert written[RAW_INTERFRAME_LOGX_PDF].is_file()
    pkl = pdf.parent.parent / "metadata" / "raw_interframe_delta.pkl"
    assert pkl.is_file()
    params = yaml.safe_load((pdf.parent.parent / "metadata" / "params.yaml").read_text())
    assert params["exclude_saccades"] is False
    assert params["trimmed"] is False
    fig = figure_raw_interframe_delta(pools, n_bins=40)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_s1_ratio_hist_and_vline(tmp_path: Path):
    n = 200
    left = pd.DataFrame(
        {
            "ratio": np.clip(np.random.default_rng(0).normal(0.7, 0.1, n), 0.1, 1.0),
            "k_phi": np.linspace(0, 40, n),
            "k_theta": np.zeros(n),
        }
    )
    tables = _tables(tmp_path, events=_empty_events(), left=left, right=left.copy())
    written = export_s1_occupancy(tables, tmp_path, show=False)
    assert "s1_empirical_ratio_hist.pdf" in written
    df = collect_ratio_and_radius(tables)
    vline = ratio_at_radial_cut(df, cut_deg=35.0, band=5.0)
    assert np.isfinite(vline)
    assert 0.0 < vline <= 1.0
