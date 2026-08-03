"""Angular saccade detection (paper notebook robust direction-segmentation)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _minimal_angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def create_saccade_events_with_direction_segmentation_robust(
    eye_data_df: pd.DataFrame,
    speed_threshold: float,
    *,
    directional_delta_threshold_deg: float = 90.0,
    magnitude_calib: float = 1.0,
    speed_profile: bool = True,
    min_subsaccade_samples: int = 2,
    min_net_disp: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect/segment saccades using angular speed on ``k_phi`` / ``k_theta``.

    ``speed_threshold`` is degrees per video frame (paper default 0.8 ≈ 50°/s at 60 Hz).
    """
    required = {"center_x", "center_y", "k_phi", "k_theta", "OE_timestamp", "ms_axis"}
    missing = required - set(eye_data_df.columns)
    if missing:
        raise KeyError(f"eye_data_df missing columns: {sorted(missing)}")

    df = eye_data_df.copy()
    df["speed_x"] = df["center_x"].diff()
    df["speed_y"] = df["center_y"].diff()
    df["speed_r"] = np.sqrt(df["speed_x"] ** 2 + df["speed_y"] ** 2)
    df["angular_speed_phi"] = df["k_phi"].diff()
    df["angular_speed_theta"] = df["k_theta"].diff()
    df["angular_speed_r"] = np.sqrt(
        df["angular_speed_phi"] ** 2 + df["angular_speed_theta"] ** 2
    )
    df["is_saccade_angle"] = df["angular_speed_r"] > speed_threshold

    on_off = df["is_saccade_angle"].astype(int) - df["is_saccade_angle"].shift(
        1, fill_value=0
    ).astype(int)
    on_inds = np.where(on_off == 1)[0]
    off_inds = np.where(on_off == -1)[0]
    if len(on_inds) > len(off_inds):
        on_inds = on_inds[:-1]

    events: list[dict] = []
    for start_ind, end_ind in zip(on_inds, off_inds):
        saccade_df = df.iloc[start_ind : end_ind + 1].copy()
        if saccade_df.empty or len(saccade_df) < min_subsaccade_samples:
            continue

        saccade_df["inst_angle_deg"] = np.degrees(
            np.arctan2(
                saccade_df["angular_speed_theta"],
                saccade_df["angular_speed_phi"],
            )
        )
        angles = saccade_df["inst_angle_deg"].to_numpy(dtype=float)
        angle_diffs = np.array(
            [
                _minimal_angle_diff_deg(angles[i + 1], angles[i])
                for i in range(len(angles) - 1)
            ]
        )
        candidate_boundaries = np.where(
            np.abs(angle_diffs) > directional_delta_threshold_deg
        )[0].tolist()
        boundaries = [0] + candidate_boundaries + [len(saccade_df) - 1]

        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            subsaccade = saccade_df.iloc[seg_start : seg_end + 1]
            if len(subsaccade) < min_subsaccade_samples:
                continue

            initial = subsaccade.iloc[0][["k_phi", "k_theta"]]
            final = subsaccade.iloc[-1][["k_phi", "k_theta"]]
            net_disp = float(
                np.sqrt(
                    (final["k_phi"] - initial["k_phi"]) ** 2
                    + (final["k_theta"] - initial["k_theta"]) ** 2
                )
            )
            if net_disp < min_net_disp:
                continue

            if "pupil_diameter" in subsaccade.columns:
                diameter_profile = subsaccade["pupil_diameter"].to_numpy()
            else:
                diameter_profile = np.full(len(subsaccade), np.nan)

            speed_r = subsaccade["speed_r"].to_numpy(dtype=float)
            ang_r = subsaccade["angular_speed_r"].to_numpy(dtype=float)
            overall_angle_deg = float(
                np.degrees(
                    np.arctan2(
                        final["k_theta"] - initial["k_theta"],
                        final["k_phi"] - initial["k_phi"],
                    )
                )
                % 360.0
            )

            events.append(
                {
                    "saccade_start_ind": int(subsaccade.index[0]),
                    "saccade_end_ind": int(subsaccade.index[-1]),
                    "saccade_start_timestamp": float(subsaccade["OE_timestamp"].iloc[0]),
                    "saccade_end_timestamp": float(subsaccade["OE_timestamp"].iloc[-1]),
                    "saccade_on_ms": float(subsaccade["ms_axis"].iloc[0]),
                    "saccade_off_ms": float(subsaccade["ms_axis"].iloc[-1]),
                    "length": int(subsaccade.index[-1] - subsaccade.index[0]),
                    "magnitude_raw_pixel": float(np.nansum(speed_r)),
                    "magnitude_pixel": float(np.nansum(speed_r) * magnitude_calib),
                    "magnitude_raw_angular": float(np.nansum(ang_r)),
                    "overall_angle_deg": overall_angle_deg,
                    "net_angular_disp": net_disp,
                    "speed_profile_pixel": speed_r if speed_profile else None,
                    "speed_profile_pixel_calib": (
                        speed_r * magnitude_calib if speed_profile else None
                    ),
                    "speed_profile_angular": ang_r if speed_profile else None,
                    "diameter_profile": diameter_profile,
                    "theta_init_pos": float(initial["k_theta"]),
                    "theta_end_pos": float(final["k_theta"]),
                    "phi_init_pos": float(initial["k_phi"]),
                    "phi_end_pos": float(final["k_phi"]),
                }
            )

    saccade_events_df = pd.DataFrame(events)
    df = df.drop(columns=["is_saccade_angle"], errors="ignore")
    if not saccade_events_df.empty:
        saccade_events_df["delta_theta"] = (
            saccade_events_df["theta_end_pos"] - saccade_events_df["theta_init_pos"]
        )
        saccade_events_df["delta_phi"] = (
            saccade_events_df["phi_end_pos"] - saccade_events_df["phi_init_pos"]
        )
        # peak_velocity: deg/frame (= max of speed_profile_angular), matching the
        # paper notebook / Fig_2_j reproduction pickle convention. Downstream
        # exporters convert to deg/ms or deg/sec as needed (see figures_2c_2e).
        peak_v = []
        ttp = []
        for _, row in saccade_events_df.iterrows():
            sp = row.get("speed_profile_angular")
            if sp is None or len(sp) == 0 or not np.any(np.isfinite(sp)):
                peak_v.append(np.nan)
                ttp.append(np.nan)
                continue
            sp = np.asarray(sp, dtype=float)
            dur_ms = float(row["saccade_off_ms"] - row["saccade_on_ms"])
            n = max(len(sp) - 1, 1)
            frame_ms = dur_ms / n if dur_ms > 0 else np.nan
            if not np.isfinite(frame_ms) or frame_ms <= 0:
                frame_ms = 1000.0 / 60.0
            peak_idx = int(np.nanargmax(sp))
            peak_v.append(float(sp[peak_idx]))  # deg/frame
            ttp.append(float(peak_idx * frame_ms))  # ms
        saccade_events_df["peak_velocity"] = peak_v
        saccade_events_df["time_to_peak_v"] = ttp

    return df, saccade_events_df
