"""Pupil diameter during Active vs. Quiet behavioral states.

run separately for each animal: median pupil diameter
in Quiet epochs minus the median in Active epochs. Active/Quiet labels are
shuffled within each block and eye, and the pupil samples stay fixed.
Two-sided Monte-Carlo p-value.

Each block must already have ``behavior_state`` (``start_time``, ``end_time``,
``annotation``) and ``left_eye_data`` / ``right_eye_data`` with ``ms_axis`` and
``pupil_diameter``. ``animal_call`` and ``block_num`` are read from the block.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp

PARAMS_PATH = Path(__file__).with_name("params.yaml")


def load_settings() -> dict:
    with open(PARAMS_PATH, encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    return params["pupil_diameter_by_state"]


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    y_sorted = np.sort(y)
    from bisect import bisect_left, bisect_right

    less = greater = 0
    for value in x:
        less += bisect_left(y_sorted, value)
        greater += y_sorted.size - bisect_right(y_sorted, value)
    n = x.size * y_sorted.size
    return (greater - less) / n if n else np.nan


def collect_pupil_by_state(blocks, min_epoch_ms: float = 1000.0) -> pd.DataFrame:
    """One median pupil diameter per annotated epoch per eye."""
    rows = []
    for block in blocks:
        behavior = getattr(block, "behavior_state", None)
        left = getattr(block, "left_eye_data", None)
        right = getattr(block, "right_eye_data", None)
        if behavior is None or left is None or right is None:
            continue
        if any(frame is None or frame.empty for frame in (behavior, left, right)):
            continue
        epochs = behavior.rename(
            columns={"start_time": "start", "end_time": "end", "annotation": "state"}
        ).copy()
        if {"start", "end", "state"}.difference(epochs.columns):
            continue
        if min_epoch_ms > 0:
            epochs = epochs[(epochs["end"] - epochs["start"]) >= min_epoch_ms]
        if epochs.empty:
            continue
        animal = block.animal_call
        block_num = str(block.block_num).zfill(3)
        for eye_label, eye in (("L", left), ("R", right)):
            if "ms_axis" not in eye.columns or "pupil_diameter" not in eye.columns:
                continue
            samples = eye[["ms_axis", "pupil_diameter"]].dropna().sort_values("ms_axis")
            epoch_index = 0
            for _, row in epochs.iterrows():
                if row["state"] not in ("quiet", "active"):
                    continue
                segment = samples[(samples["ms_axis"] > row["start"]) & (samples["ms_axis"] <= row["end"])]
                if segment.empty:
                    continue
                rows.append(
                    {
                        "value": float(np.nanmedian(segment["pupil_diameter"].values)),
                        "state": row["state"],
                        "animal": animal,
                        "block": block_num,
                        "eye": eye_label,
                        "epoch_id": f"{animal}_B{block_num}_{eye_label}_e{epoch_index}",
                    }
                )
                epoch_index += 1
    return pd.DataFrame(rows)


def per_animal_permutation_tests(
    tidy: pd.DataFrame,
    n_permutations: int = 20000,
    random_state: int = 123,
) -> pd.DataFrame:
    """Shuffle Quiet/Active labels within block x eye. Statistic is
    median(quiet) - median(active).
    """
    required = {"animal", "state", "value", "block", "eye"}
    missing = required - set(tidy.columns)
    if missing:
        raise ValueError(f"tidy is missing columns: {missing}")

    data = tidy[tidy["state"].isin(["quiet", "active"])].copy()
    data = data[np.isfinite(data["value"].values)]
    if data.empty:
        raise ValueError("No data for quiet and active.")

    rng = np.random.default_rng(random_state)
    results = []
    for animal in sorted(data["animal"].unique()):
        group = data[data["animal"] == animal].copy()
        quiet = group.loc[group["state"] == "quiet", "value"].to_numpy()
        active = group.loc[group["state"] == "active", "value"].to_numpy()
        n_quiet, n_active = int(quiet.size), int(active.size)
        if n_quiet == 0 or n_active == 0:
            results.append(
                {
                    "animal": animal,
                    "observed_stat": np.nan,
                    "p_value": np.nan,
                    "n_quiet": n_quiet,
                    "n_active": n_active,
                    "cliffs_delta": np.nan,
                    "ks_stat": np.nan,
                    "n_perm": 0,
                }
            )
            continue

        observed = float(np.nanmedian(quiet) - np.nanmedian(active))
        cliffs = _cliffs_delta(quiet, active)
        ks_stat = ks_2samp(quiet, active, mode="auto").statistic

        strata = list(group.groupby(["block", "eye"], sort=False))
        values = np.concatenate([sub["value"].to_numpy() for _, sub in strata])
        order_states = np.concatenate([sub["state"].to_numpy() for _, sub in strata])
        null = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            perm_states = order_states.copy()
            position = 0
            for _, sub in strata:
                count = len(sub)
                piece = perm_states[position : position + count]
                rng.shuffle(piece)
                perm_states[position : position + count] = piece
                position += count
            x = values[perm_states == "quiet"]
            y = values[perm_states == "active"]
            if x.size == 0 or y.size == 0:
                null[i] = np.nan
            else:
                null[i] = float(np.nanmedian(x) - np.nanmedian(y))

        null = null[np.isfinite(null)]
        if null.size == 0:
            p_value = np.nan
        else:
            p_value = (np.sum(np.abs(null) >= np.abs(observed)) + 1) / (null.size + 1)
        results.append(
            {
                "animal": animal,
                "observed_stat": observed,
                "p_value": float(p_value),
                "n_quiet": n_quiet,
                "n_active": n_active,
                "cliffs_delta": float(cliffs),
                "ks_stat": float(ks_stat),
                "n_perm": int(null.size),
            }
        )

    columns = [
        "animal",
        "observed_stat",
        "p_value",
        "n_quiet",
        "n_active",
        "cliffs_delta",
        "ks_stat",
        "n_perm",
    ]
    return pd.DataFrame(results, columns=columns)


def run(blocks, settings: dict | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    tidy = collect_pupil_by_state(blocks, min_epoch_ms=float(settings["min_epoch_ms"]))
    if tidy.empty:
        raise RuntimeError("No pupil samples fell inside annotated quiet/active epochs.")
    table = per_animal_permutation_tests(
        tidy,
        n_permutations=int(settings["n_permutations"]),
        random_state=int(settings["random_seed"]),
    )
    print(table.sort_values("p_value").to_string(index=False))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Per-animal Monte-Carlo test of median pupil diameter in Quiet vs "
            "Active epochs. The pickle must contain a list of already-processed "
            "BlockSync objects. The paper's blocks are not included."
        )
    )
    parser.add_argument("blocks_pickle", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    with open(args.blocks_pickle, "rb") as handle:
        blocks = pickle.load(handle)
    table = run(blocks)
    if args.out is not None:
        table.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
