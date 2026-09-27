"""Saccade rate during Active vs. Quiet behavioral states.

for each animal, saccade rate in Active 1-second bins
minus the rate in Quiet bins, then the mean of that difference across animals.
State labels are shuffled across bins within each animal, keeping the number
of Active bins fixed. Two-sided Monte-Carlo p-value.

This is ``mc_saccade_rate_by_behavior_state`` from
``multiple_figures_pipeline_migration.ipynb`` (the call uses onset column
``saccade_on_ms``, 1 s bins, 10,000 permutations, 5,000 animal bootstraps,
seed 42, no head-movement filter). Plotting from that cell is omitted.

Each block must already have detected saccades (``all_saccade_df``, or
``l_saccade_df`` and ``r_saccade_df``) with ``saccade_on_ms``, and
``behavior_state`` with ``start_time``, ``end_time``, and ``annotation``
(``active`` / ``quiet``). ``animal_call`` and ``block_num`` are read from the
block. The experimental blocks used for the paper are not distributed.
Passing other compatible blocks does not reproduce the published numbers.
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
    return params["saccade_rate_by_state"]


def _animal_id(block):
    if hasattr(block, "animal_call"):
        return block.animal_call
    if hasattr(block, "animal_id"):
        return block.animal_id
    raise AttributeError("Block has no animal_call or animal_id.")


def saccade_table(blocks) -> pd.DataFrame:
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
        frame["animal"] = _animal_id(block)
        frame["block"] = block.block_num
        frames.append(frame)
    if not frames:
        raise ValueError("No blocks were provided.")
    return pd.concat(frames, ignore_index=True)


def collect_behavior_state_from_blocks(block_collection) -> pd.DataFrame:
    rows = []
    for block in block_collection:
        if not hasattr(block, "behavior_state") or block.behavior_state is None:
            continue
        frame = block.behavior_state.copy()
        frame["animal"] = _animal_id(block)
        if getattr(block, "block_num", None) is None:
            raise AttributeError("Block has no block_num attribute.")
        frame["block"] = block.block_num
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["animal", "block", "start_time", "end_time", "annotation"])
    return pd.concat(rows, ignore_index=True)


def mc_saccade_rate_by_behavior_state(
    all_saccade_collection,
    behavior_state_all,
    onset_col="saccade_on_ms",
    animal_col="animal",
    block_col="block",
    state_col="annotation",
    start_col="start_time",
    end_col="end_time",
    bin_size_ms=1000,
    min_total_time_s=10.0,
    n_perm=10000,
    n_boot=5000,
    random_state=42,
):
    """1 s bins, overlap vote for Active vs Quiet, within-animal label shuffle."""
    rng = np.random.default_rng(random_state)
    missing = {animal_col, block_col, onset_col} - set(all_saccade_collection.columns)
    if missing:
        raise KeyError(f"Missing columns in saccades: {missing}")
    missing = {animal_col, block_col, start_col, end_col, state_col} - set(behavior_state_all.columns)
    if missing:
        raise KeyError(f"Missing columns in behavior_state: {missing}")

    sacc = all_saccade_collection.copy()
    beh = behavior_state_all.copy()
    beh = beh[beh[state_col].isin(["active", "quiet"])].copy()
    sacc = sacc[sacc[onset_col].notna()].copy()
    sacc[onset_col] = sacc[onset_col].astype(float)

    sacc_groups = {}
    for (animal, block), group in sacc.groupby([animal_col, block_col], sort=False):
        sacc_groups[(animal, block)] = group[onset_col].to_numpy()

    def total_overlap(windows, start, end):
        if windows.size == 0:
            return 0.0
        lo = np.maximum(windows[:, 0], start)
        hi = np.minimum(windows[:, 1], end)
        return float(np.sum(np.clip(hi - lo, 0.0, None)))

    bin_rows = []
    for (animal, block), group in beh.groupby([animal_col, block_col], sort=False):
        t0 = float(np.nanmin(group[start_col].to_numpy()))
        t1 = float(np.nanmax(group[end_col].to_numpy()))
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue
        bin_starts = np.arange(
            np.floor(t0 / bin_size_ms) * bin_size_ms,
            np.ceil(t1 / bin_size_ms) * bin_size_ms,
            bin_size_ms,
        )
        onsets = np.sort(sacc_groups.get((animal, block), np.array([], dtype=float)))
        active_windows = group.loc[group[state_col] == "active", [start_col, end_col]].to_numpy(dtype=float)
        quiet_windows = group.loc[group[state_col] == "quiet", [start_col, end_col]].to_numpy(dtype=float)
        cursor = 0
        for start in bin_starts:
            end = start + bin_size_ms
            overlap_active = total_overlap(active_windows, start, end)
            overlap_quiet = total_overlap(quiet_windows, start, end)
            if overlap_active <= 0 and overlap_quiet <= 0:
                continue
            state = "active" if overlap_active > overlap_quiet else "quiet"
            while cursor < len(onsets) and onsets[cursor] < start:
                cursor += 1
            stop = cursor
            while stop < len(onsets) and onsets[stop] < end:
                stop += 1
            bin_rows.append((animal, block, float(start), state, int(stop - cursor)))

    if not bin_rows:
        raise ValueError("No binned behavior data created. Check behavior_state coverage and block identifiers.")

    bins = pd.DataFrame(bin_rows, columns=[animal_col, block_col, "bin_start_ms", "state", "n_sacc"])
    bins["bin_dur_s"] = bin_size_ms / 1000.0

    animals = []
    diff_obs = []
    per_animal_rows = []
    for animal, group in bins.groupby(animal_col, sort=True):
        time_active = float(group.loc[group["state"] == "active", "bin_dur_s"].sum())
        time_quiet = float(group.loc[group["state"] == "quiet", "bin_dur_s"].sum())
        if time_active < min_total_time_s or time_quiet < min_total_time_s:
            continue
        n_active = int(group.loc[group["state"] == "active", "n_sacc"].sum())
        n_quiet = int(group.loc[group["state"] == "quiet", "n_sacc"].sum())
        rate_active = n_active / time_active
        rate_quiet = n_quiet / time_quiet
        delta = float(rate_active - rate_quiet)
        animals.append(animal)
        diff_obs.append(delta)
        per_animal_rows.append(
            {
                "animal": animal,
                "rate_active_hz": rate_active,
                "rate_quiet_hz": rate_quiet,
                "diff_hz": delta,
                "time_active_s": time_active,
                "time_quiet_s": time_quiet,
                "n_sacc_active": n_active,
                "n_sacc_quiet": n_quiet,
            }
        )

    if len(animals) < 2:
        raise ValueError("Not enough animals pass min_total_time_s in both states.")

    diff_obs = np.array(diff_obs, dtype=float)
    effect_obs = float(np.mean(diff_obs))
    per_animal = pd.DataFrame(per_animal_rows).sort_values("animal")

    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        index = rng.integers(0, len(diff_obs), size=len(diff_obs))
        boot[i] = float(np.mean(diff_obs[index]))
    ci95 = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    pack = {}
    for animal in animals:
        group = bins[bins[animal_col] == animal]
        pack[animal] = (
            group["n_sacc"].to_numpy(dtype=float),
            group["state"].to_numpy() == "active",
        )

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        diffs_i = []
        for animal in animals:
            counts, labels = pack[animal]
            n_active_bins = int(labels.sum())
            n_bins = len(labels)
            perm = rng.permutation(n_bins)
            active_mask = np.zeros(n_bins, dtype=bool)
            active_mask[perm[:n_active_bins]] = True
            n_active = float(counts[active_mask].sum())
            n_quiet = float(counts[~active_mask].sum())
            time_active = (n_active_bins * bin_size_ms) / 1000.0
            time_quiet = ((n_bins - n_active_bins) * bin_size_ms) / 1000.0
            diffs_i.append(n_active / time_active - n_quiet / time_quiet)
        null[i] = float(np.mean(diffs_i))

    p_two = float((np.sum(np.abs(null) >= abs(effect_obs)) + 1.0) / (len(null) + 1.0))
    print(
        "[MC-rate] animals used: {} | effect(obs)={:.6g} Hz | p(two-sided)={:.6g} | CI95=[{:.6g},{:.6g}] Hz".format(
            len(animals), effect_obs, p_two, ci95[0], ci95[1]
        )
    )
    return {
        "effect_mean_delta_rate_hz": effect_obs,
        "ci95_bootstrap_animals_hz": ci95,
        "p_mc_two_sided": p_two,
        "per_animal": per_animal,
    }


def run(blocks, settings: dict | None = None) -> dict:
    settings = settings or load_settings()
    return mc_saccade_rate_by_behavior_state(
        saccade_table(blocks),
        collect_behavior_state_from_blocks(blocks),
        onset_col=settings["onset_col"],
        bin_size_ms=float(settings["bin_size_ms"]),
        min_total_time_s=float(settings["min_total_time_s"]),
        n_perm=int(settings["n_perm"]),
        n_boot=int(settings["n_boot"]),
        random_state=int(settings["random_seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Within-animal Monte-Carlo test of saccade rate in Active vs Quiet "
            "states. The pickle must contain a list of already-processed "
            "BlockSync objects. The paper's blocks are not included."
        )
    )
    parser.add_argument("blocks_pickle", type=Path)
    parser.add_argument("--out", type=Path, default=None)
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
