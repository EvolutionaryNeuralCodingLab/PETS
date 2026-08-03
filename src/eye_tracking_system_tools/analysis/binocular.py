"""Binocular pairing of per-eye saccade events."""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_synced_saccades_ms(
    df: pd.DataFrame,
    *,
    sync_diff_ms: float = 34.0,
    on_col: str = "saccade_on_ms",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pair L/R events whose onset times differ by < ``sync_diff_ms``.

    Returns (synced_df with MultiIndex Main/Sub, non_synced_df).
    """
    if df.empty:
        empty = df.copy()
        return empty, empty

    work = df.dropna(subset=[on_col, "eye"]).copy()
    l_df = work.query('eye == "L"')
    r_df = work.query('eye == "R"')
    if l_df.empty or r_df.empty:
        return work.iloc[0:0].copy(), work.copy()

    synced_pairs: list[tuple[pd.Series, pd.Series]] = []
    non_synced_l: list[pd.Series] = []
    r_times = r_df[on_col].to_numpy(dtype=float)
    r_matched: set[int] = set()

    for _, l_row in l_df.iterrows():
        t_l = float(l_row[on_col])
        dt = np.abs(r_times - t_l)
        r_rel = int(np.argmin(dt))
        r_idx = r_df.index[r_rel]
        if dt[r_rel] < sync_diff_ms and r_idx not in r_matched:
            synced_pairs.append((l_row, r_df.loc[r_idx]))
            r_matched.add(r_idx)
        else:
            non_synced_l.append(l_row)

    r_leftovers = r_df.loc[~r_df.index.isin(r_matched)]

    n = len(synced_pairs)
    if n == 0:
        synced_df = work.iloc[0:0].copy()
    else:
        idx = pd.MultiIndex.from_tuples(
            [(i, "L") for i in range(n)] + [(i, "R") for i in range(n)],
            names=["Main", "Sub"],
        )
        synced_df = pd.DataFrame(index=idx, columns=work.columns)
        for i, (l_row, r_row) in enumerate(synced_pairs):
            synced_df.loc[(i, "L")] = l_row
            synced_df.loc[(i, "R")] = r_row
        synced_df = synced_df.reset_index()

    non_parts = []
    if non_synced_l:
        non_parts.append(pd.DataFrame(non_synced_l))
    if len(r_leftovers):
        non_parts.append(r_leftovers)
    non_synced_df = (
        pd.concat(non_parts, ignore_index=True) if non_parts else work.iloc[0:0].copy()
    )
    return synced_df, non_synced_df


def combine_synced_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-block synced tables with a continuous Main index."""
    if not dataframes:
        return pd.DataFrame()
    combined = []
    start = 0
    for df in dataframes:
        if df is None or df.empty:
            continue
        d = df.copy()
        if "Main" in d.columns and "Sub" in d.columns:
            n_pairs = int(d["Main"].nunique())
            mapping = {
                old: new
                for new, old in enumerate(sorted(d["Main"].unique()), start=start)
            }
            d["Main"] = d["Main"].map(mapping)
            start += n_pairs
        combined.append(d)
    if not combined:
        return pd.DataFrame()
    return pd.concat(combined, ignore_index=True)
