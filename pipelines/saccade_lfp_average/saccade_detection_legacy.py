"""Legacy saccade detection (speed threshold + binocular sync pairing)."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, total=None, desc=None):
        return iterable


def create_saccade_events_df(
    eye_data_df: pd.DataFrame,
    speed_threshold: float,
    magnitude_calib: float = 1.0,
    speed_profile: bool = False,
    use_pupil_diameter: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect saccade events from eye tracking data (speed_r > threshold).

    Returns (df_with_speed, saccade_events_df). ``df`` is a copy with speed cols;
    the input dataframe is unchanged.
    """
    required = {"center_x", "center_y", "ms_axis", "OE_timestamp"}
    missing = required - set(eye_data_df.columns)
    if missing:
        raise KeyError(f"Eye dataframe missing columns for legacy detection: {sorted(missing)}")

    df = eye_data_df.copy()
    df["speed_x"] = df["center_x"].diff()
    df["speed_y"] = df["center_y"].diff()
    df["speed_r"] = (df["speed_x"] ** 2 + df["speed_y"] ** 2) ** 0.5
    df["is_saccade"] = df["speed_r"] > speed_threshold

    on_off = df["is_saccade"].astype(int) - df["is_saccade"].shift(periods=1, fill_value=False).astype(
        int
    )
    on_inds = np.where(on_off == 1)[0] - 1
    off_inds = np.where(on_off == -1)[0]

    empty_cols = [
        "saccade_start_ind",
        "saccade_end_ind",
        "saccade_start_timestamp",
        "saccade_end_timestamp",
        "saccade_on_ms",
        "saccade_off_ms",
        "length",
        "magnitude_raw",
        "magnitude",
        "angle",
        "initial_x",
        "initial_y",
        "end_x",
        "end_y",
        "calib_dx",
        "calib_dy",
    ]
    if len(on_inds) == 0 or len(off_inds) == 0:
        ev = pd.DataFrame(columns=empty_cols)
        if speed_profile:
            ev["speed_profile"] = []
        if use_pupil_diameter and "pupil_diameter" in df.columns:
            ev["diameter_profile"] = []
        df = df.drop(columns=["speed_x", "speed_y", "speed_r", "is_saccade"], errors="ignore")
        return df, ev

    on_ms = df["ms_axis"].iloc[on_inds].values
    on_ts = df["OE_timestamp"].iloc[on_inds].values
    off_ts = df["OE_timestamp"].iloc[off_inds].values
    off_ms = df["ms_axis"].iloc[off_inds].values

    ev = pd.DataFrame(
        {
            "saccade_start_ind": on_inds,
            "saccade_end_ind": off_inds,
            "saccade_start_timestamp": on_ts,
            "saccade_end_timestamp": off_ts,
            "saccade_on_ms": on_ms,
            "saccade_off_ms": off_ms,
        }
    )
    ev["length"] = ev["saccade_end_ind"] - ev["saccade_start_ind"]

    distances, angles, speed_list, diameter_list = [], [], [], []
    for _, row in tqdm(ev.iterrows(), total=len(ev), desc="saccade metrics"):
        seg = df.loc[
            (df["OE_timestamp"] >= row["saccade_start_timestamp"])
            & (df["OE_timestamp"] <= row["saccade_end_timestamp"])
        ]
        dist = seg["speed_r"].sum()
        distances.append(dist)
        if speed_profile:
            speed_list.append(seg["speed_r"].values)
        if use_pupil_diameter and "pupil_diameter" in df.columns:
            diameter_list.append(seg["pupil_diameter"].values)
        elif use_pupil_diameter:
            diameter_list.append(np.full(len(seg), np.nan))
        xi, yi = seg.iloc[0][["center_x", "center_y"]]
        xe, ye = seg.iloc[-1][["center_x", "center_y"]]
        ang = np.arctan2(ye - yi, xe - xi)
        angles.append(ang)

    ev["magnitude_raw"] = np.array(distances)
    ev["magnitude"] = np.array(distances) * magnitude_calib
    ev["angle"] = np.where(np.isnan(angles), np.nan, np.rad2deg(angles) % 360)
    start_ts = ev["saccade_start_timestamp"].values
    end_ts = ev["saccade_end_timestamp"].values
    start_df = df[df["OE_timestamp"].isin(start_ts)]
    end_df = df[df["OE_timestamp"].isin(end_ts)]
    ev["initial_x"] = start_df["center_x"].values
    ev["initial_y"] = start_df["center_y"].values
    ev["end_x"] = end_df["center_x"].values
    ev["end_y"] = end_df["center_y"].values
    ev["calib_dx"] = (ev["end_x"].values - ev["initial_x"].values) * magnitude_calib
    ev["calib_dy"] = (ev["end_y"].values - ev["initial_y"].values) * magnitude_calib
    if speed_profile:
        ev["speed_profile"] = speed_list
    if use_pupil_diameter and (("pupil_diameter" in df.columns) or diameter_list):
        ev["diameter_profile"] = diameter_list

    df = df.drop(columns=["speed_x", "speed_y", "speed_r", "is_saccade"], errors="ignore")
    return df, ev


def find_synced_saccades(
    df: pd.DataFrame,
    diff_threshold_ms: float = 680,
    on_col: str = "saccade_on_ms",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split saccades into synced (L–R pairs) and non_synced (unpaired)."""
    l_df = df.query('eye == "L"').copy()
    r_df = df.query('eye == "R"').copy()
    synced_rows: list[tuple[pd.Series, pd.Series]] = []
    non_synced_rows: list[pd.Series] = []

    for _, row in l_df.iterrows():
        t_l = row[on_col]
        dt = np.abs(r_df[on_col].values - t_l)
        ind = int(np.argmin(dt))
        if dt[ind] < diff_threshold_ms:
            synced_rows.append((row, r_df.iloc[ind]))
        else:
            non_synced_rows.append(row)

    r_matched = r_df.index.isin([r.index for _, r in synced_rows])
    r_leftovers = r_df.loc[~r_matched]

    n = len(synced_rows)
    idx = pd.MultiIndex.from_tuples(
        [(i, "L") for i in range(n)] + [(i, "R") for i in range(n)],
        names=["Main", "Sub"],
    )
    synced_df = pd.DataFrame(index=idx, columns=df.columns)
    for i, (l_row, r_row) in enumerate(synced_rows):
        synced_df.loc[(i, "L")] = l_row
        synced_df.loc[(i, "R")] = r_row

    non_synced_df = pd.concat(
        [pd.DataFrame(non_synced_rows, columns=df.columns), r_leftovers],
        ignore_index=True,
    )
    return synced_df, non_synced_df


def annotate_events_sync_status(
    events: pd.DataFrame,
    *,
    sync_diff_ms: float,
    on_col: str = "saccade_start_ms",
) -> pd.DataFrame:
    """Label each event ``synced`` or ``non_synced``; assign ``sync_pair_id`` for pairs."""
    if events.empty:
        return events

    out = events.copy()
    out["sync_status"] = "non_synced"
    out["sync_pair_id"] = pd.NA

    l_df = out[out["eye"] == "L"]
    r_df = out[out["eye"] == "R"]
    if l_df.empty or r_df.empty:
        return out

    r_matched: set = set()
    pair_id = 0
    r_times = r_df[on_col].to_numpy(dtype=float)

    for l_idx, l_row in l_df.iterrows():
        t_l = float(l_row[on_col])
        dt = np.abs(r_times - t_l)
        r_rel = int(np.argmin(dt))
        r_idx = r_df.index[r_rel]
        if dt[r_rel] < sync_diff_ms and r_idx not in r_matched:
            out.loc[l_idx, "sync_status"] = "synced"
            out.loc[r_idx, "sync_status"] = "synced"
            out.loc[l_idx, "sync_pair_id"] = pair_id
            out.loc[r_idx, "sync_pair_id"] = pair_id
            r_matched.add(r_idx)
            pair_id += 1

    return out


def combine_synced_dataframes(synced_df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-block synced DFs, reindex Main across blocks, reset_index."""
    out: list[pd.DataFrame] = []
    start = 0
    for sdf in synced_df_list:
        n = len(sdf) // 2
        if n == 0:
            continue
        idx = pd.MultiIndex.from_tuples(
            [(start + i, "L") for i in range(n)] + [(start + i, "R") for i in range(n)],
            names=["Main", "Sub"],
        )
        sdf = sdf.set_axis(idx)
        out.append(sdf)
        start += n
    if not out:
        return pd.DataFrame()
    combined = pd.concat(out)
    return combined.reset_index()


def detect_legacy_block_events(
    block,
    *,
    speed_threshold: float,
    magnitude_calib: float,
    sync_diff_ms: float,
    use_pupil_diameter: bool,
    acc_df: pd.DataFrame | None,
    behavior_df: pd.DataFrame | None,
    accel_fn,
    behavior_fn,
) -> pd.DataFrame:
    """Run legacy per-eye detection + binocular sync categorization for one block."""
    combined_parts: list[pd.DataFrame] = []
    for eye, eye_df in (("L", block.le_df), ("R", block.re_df)):
        _, ev = create_saccade_events_df(
            eye_df,
            speed_threshold,
            magnitude_calib=magnitude_calib,
            speed_profile=False,
            use_pupil_diameter=use_pupil_diameter,
        )
        if ev.empty:
            continue
        ev = ev.copy()
        ev["eye"] = eye
        ev["block"] = block.block_num
        combined_parts.append(ev)

    if not combined_parts:
        return pd.DataFrame()

    combined = pd.concat(combined_parts, ignore_index=True)
    rows: list[dict] = []
    for _, sacc in combined.iterrows():
        t_ms = float(sacc["saccade_on_ms"])
        rows.append(
            {
                "block": block.block_num,
                "eye": sacc["eye"],
                "saccade_start_ms": t_ms,
                "saccade_on_ms": t_ms,
                "saccade_length_frames": int(sacc.get("length", 0) or 0),
                "peak_velocity": float("nan"),
                "magnitude_raw": float(sacc.get("magnitude_raw", np.nan)),
                "magnitude": float(sacc.get("magnitude", np.nan)),
                "angle": float(sacc.get("angle", np.nan)),
                "accel": accel_fn(acc_df, t_ms - 50.0, t_ms + 100.0),
                "behavior": behavior_fn(behavior_df, t_ms),
            }
        )
    events = pd.DataFrame(rows)
    return annotate_events_sync_status(events, sync_diff_ms=sync_diff_ms)
