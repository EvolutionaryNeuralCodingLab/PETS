"""Saccade amplitude with vs. without head movement.

mean saccade amplitude during head movement minus mean
amplitude while the head is still, averaged across animals. Head-movement
labels are shuffled within each animal. Two-sided Monte-Carlo p-value.

Each block must already have detected saccades on ``all_saccade_df``, or on
``l_saccade_df`` and ``r_saccade_df``, including ``net_angular_disp`` and a
boolean ``head_movement`` label. ``animal_call`` and ``block_num`` are read
from the block. 
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PARAMS_PATH = Path(__file__).with_name("params.yaml")


def load_settings() -> dict:
    with open(PARAMS_PATH, encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    return params["saccade_amplitude_head_movement"]


def saccade_table(blocks) -> pd.DataFrame:
    """Stack already-detected saccades, as the notebook does before the test."""
    frames = []
    for block in blocks:
        if getattr(block, "all_saccade_df", None) is not None:
            frame = block.all_saccade_df.copy()
        else:
            parts = []
            if getattr(block, "l_saccade_df", None) is not None:
                left = block.l_saccade_df.copy()
                left["eye"] = "L"
                parts.append(left)
            if getattr(block, "r_saccade_df", None) is not None:
                right = block.r_saccade_df.copy()
                right["eye"] = "R"
                parts.append(right)
            if not parts:
                raise AttributeError(
                    "Each block needs all_saccade_df, or l_saccade_df and r_saccade_df."
                )
            frame = pd.concat(parts, ignore_index=True)
        if hasattr(block, "animal_call"):
            animal = block.animal_call
        elif hasattr(block, "animal_id"):
            animal = block.animal_id
        else:
            raise AttributeError("Block has no animal_call or animal_id.")
        frame["animal"] = animal
        frame["block"] = block.block_num
        frames.append(frame)
    if not frames:
        raise ValueError("No blocks were provided.")
    return pd.concat(frames, ignore_index=True)


def mc_headmove_vs_amplitude(
    all_saccade_collection,
    amp_col="net_angular_disp",
    head_col="head_movement",
    animal_col="animal",
    min_per_group=5,
    n_perm=10000,
    n_boot=5000,
    random_state=42,
):
    """Within-animal shuffle of head labels. Effect is the mean across animals
    of (mean amplitude | moving) - (mean amplitude | still).
    """
    rng = np.random.default_rng(random_state)
    required = {amp_col, head_col, animal_col}
    missing = required - set(all_saccade_collection.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    data = all_saccade_collection.loc[:, [animal_col, amp_col, head_col]].copy()
    data = data[data[amp_col].notna()].copy()
    data = data[np.isfinite(data[amp_col].astype(float))]
    data = data[data[head_col].notna()].copy()
    data = data[data[animal_col].notna()].copy()
    data[amp_col] = data[amp_col].astype(float)
    data[head_col] = data[head_col].astype(bool)

    animals = []
    amp_by_animal = {}
    nmove_by_animal = {}
    diffs_obs = {}
    nstill_by_animal = {}
    for animal, group in data.groupby(animal_col, sort=True):
        amps = group[amp_col].to_numpy()
        labels = group[head_col].to_numpy().astype(bool)
        n_move = int(labels.sum())
        n_still = int((~labels).sum())
        if n_move < min_per_group or n_still < min_per_group:
            continue
        animals.append(animal)
        amp_by_animal[animal] = amps
        nmove_by_animal[animal] = n_move
        nstill_by_animal[animal] = n_still
        diffs_obs[animal] = float(np.mean(amps[labels]) - np.mean(amps[~labels]))

    if len(animals) < 2:
        raise ValueError(
            f"{len(animals)} animal(s) have at least {min_per_group} saccades in both head-movement conditions. "
            "This comparison needs at least 2 animals."
        )

    diffs_obs_vec = np.array([diffs_obs[animal] for animal in animals], dtype=float)
    effect_obs = float(np.mean(diffs_obs_vec))

    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(len(animals), size=len(animals), replace=True)
        boot[i] = float(np.mean(diffs_obs_vec[sample]))
    ci95 = (float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)))

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        diffs_i = []
        for animal in animals:
            amps = amp_by_animal[animal]
            n_move = nmove_by_animal[animal]
            perm = rng.permutation(len(amps))
            move_idx = perm[:n_move]
            still_idx = perm[n_move:]
            diffs_i.append(float(np.mean(amps[move_idx]) - np.mean(amps[still_idx])))
        null[i] = float(np.mean(diffs_i))

    p_two = float((np.sum(np.abs(null) >= abs(effect_obs)) + 1.0) / (len(null) + 1.0))
    per_animal = pd.DataFrame(
        {
            animal_col: animals,
            "diff_mean_moving_minus_still": [diffs_obs[animal] for animal in animals],
            "n_moving": [nmove_by_animal[animal] for animal in animals],
            "n_still": [nstill_by_animal[animal] for animal in animals],
        }
    )
    print(
        "[MC] animals used: {} | effect(obs)={:.6g} | p(two-sided)={:.6g} | CI95=[{:.6g},{:.6g}]".format(
            len(animals), effect_obs, p_two, ci95[0], ci95[1]
        )
    )
    return {
        "effect_mean_delta": effect_obs,
        "ci95_bootstrap_animals": ci95,
        "p_mc_two_sided": p_two,
        "per_animal": per_animal,
    }


def run(blocks, settings: dict | None = None) -> dict:
    settings = settings or load_settings()
    table = saccade_table(blocks)
    return mc_headmove_vs_amplitude(
        table,
        amp_col=settings["amp_col"],
        head_col=settings["head_col"],
        min_per_group=int(settings["min_per_group"]),
        n_perm=int(settings["n_perm"]),
        n_boot=int(settings["n_boot"]),
        random_state=int(settings["random_seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Within-animal Monte-Carlo test of saccade amplitude with vs. without "
            "head movement. The pickle must contain a list of already-processed "
            "BlockSync objects. The paper's blocks are not included."
        )
    )
    parser.add_argument(
        "blocks_pickle",
        type=Path,
        help="Pickle of a list of already-processed BlockSync objects.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV for the per-animal table.")
    args = parser.parse_args()
    with open(args.blocks_pickle, "rb") as handle:
        blocks = pickle.load(handle)
    result = run(blocks)
    print(result["per_animal"].to_string(index=False))
    if args.out is not None:
        result["per_animal"].to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
